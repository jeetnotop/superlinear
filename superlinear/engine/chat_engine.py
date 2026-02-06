"""Chunked async chat inference engine (single-GPU, single-flight).

This module provides the core, reusable engine:
- request normalization -> tokenizer prompt
- serialized adapter execution
- chunk-level async streaming
- tool call detection + parsing

It deliberately contains no HTTP/FastAPI code.
"""

from __future__ import annotations

import asyncio
from collections import deque
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Sequence

from .chat_types import (
    ChatMessage,
    ChatRequest,
    DeltaEvent,
    ErrorEvent,
    FinalEvent,
    ThinkingDeltaEvent,
    StreamEvent,
    Timing,
    ToolCall,
    ToolCallEvent,
    Usage,
)
from .repetition import RepetitionDetectionConfig, detect_repetition_kmp_tail
from .tool_parser import ToolCallParseError, parse_tool_call_block

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EngineConfig:
    """Engine-wide defaults and limits."""

    default_backend: str = "custom"
    enable_thinking: bool = True
    discard_thinking: bool = True
    max_prompt_tokens: int = 262_144
    max_tool_calls_per_turn: int = 8
    repetition_detection: RepetitionDetectionConfig = field(
        default_factory=RepetitionDetectionConfig
    )


class _ModelOutputParser:
    """Incremental parser for model output.

    Responsibilities:
    - Remove <think>...</think> blocks (if the model emits them).
    - Detect and buffer a complete <tool_call>...</tool_call>.
    - Apply stop sequences (string-based) to normal text.
    """

    _THINK_OPEN = "<think>"
    _THINK_CLOSE = "</think>"
    _TOOL_OPEN = "<tool_call>"
    _TOOL_CLOSE = "</tool_call>"

    def __init__(
        self,
        *,
        stop_sequences: Sequence[str],
        valid_tool_names: set[str],
        max_tool_calls: int,
        allow_tool_calls: bool,
        start_in_think: bool = False,
        emit_thinking: bool = False,
    ) -> None:
        self._stop_sequences = [s for s in stop_sequences if s]
        self._max_stop_len = max((len(s) for s in self._stop_sequences), default=0)

        self._valid_tool_names = valid_tool_names
        self._max_tool_calls = max_tool_calls
        self._allow_tool_calls = allow_tool_calls

        self._buffer = ""
        self._tool_buffer = ""
        # When enable_thinking=True, the generation prompt already includes <think>,
        # so we start inside the think block and wait for </think> before emitting.
        self._in_think = start_in_think
        self._in_tool = False

        self._emit_thinking = bool(emit_thinking)
        self._emitted_think_open = False

        self.stopped: bool = False
        self.stop_reason: str | None = None  # "stop" | "tool_calls"
        self.tool_calls: list[ToolCall] = []

        self._tail_keep = max(
            len(self._THINK_OPEN) - 1,
            len(self._THINK_CLOSE) - 1,
            len(self._TOOL_OPEN) - 1,
            len(self._TOOL_CLOSE) - 1,
            max(self._max_stop_len - 1, 0),
        )

    def feed(self, text: str) -> list[StreamEvent]:
        if not text or self.stopped:
            return []

        self._buffer += text
        events: list[StreamEvent] = []

        if self._emit_thinking and self._in_think and not self._emitted_think_open:
            events.append(ThinkingDeltaEvent(self._THINK_OPEN))
            self._emitted_think_open = True

        while self._buffer and not self.stopped:
            if self._in_think:
                end = self._buffer.find(self._THINK_CLOSE)
                if end == -1:
                    if not self._emit_thinking:
                        # Drop accumulated thinking content; keep small tail in case the close tag is split.
                        if self._tail_keep:
                            self._buffer = self._buffer[-self._tail_keep :]
                        else:
                            self._buffer = ""
                        break

                    # Emit thinking content, keeping a small tail in case the close tag is split.
                    if self._tail_keep and len(self._buffer) > self._tail_keep:
                        emit_text = self._buffer[: -self._tail_keep]
                        self._buffer = self._buffer[-self._tail_keep :]
                    else:
                        emit_text = self._buffer
                        self._buffer = ""

                    if emit_text:
                        events.append(ThinkingDeltaEvent(emit_text))
                    break

                if self._emit_thinking:
                    think_text = self._buffer[:end]
                    if think_text:
                        events.append(ThinkingDeltaEvent(think_text))
                    events.append(ThinkingDeltaEvent(self._THINK_CLOSE))
                    self._buffer = self._buffer[end + len(self._THINK_CLOSE) :]
                    self._in_think = False
                    self._emitted_think_open = False
                    continue

                # Drop everything through the closing tag.
                self._buffer = self._buffer[end + len(self._THINK_CLOSE) :]
                self._in_think = False
                continue

            if self._in_tool:
                self._tool_buffer += self._buffer
                self._buffer = ""

                end = self._tool_buffer.find(self._TOOL_CLOSE)
                if end == -1:
                    break

                block = self._tool_buffer[: end + len(self._TOOL_CLOSE)]
                trailing = self._tool_buffer[end + len(self._TOOL_CLOSE) :]
                self._tool_buffer = ""
                self._in_tool = False

                try:
                    parsed = parse_tool_call_block(block)
                except ToolCallParseError as exc:
                    self.stopped = True
                    self.stop_reason = "error"
                    events.append(ErrorEvent(f"Failed to parse tool call: {exc}"))
                    break

                if not self._allow_tool_calls:
                    self.stopped = True
                    self.stop_reason = "error"
                    events.append(ErrorEvent("Tool calls are disabled for this request (tool_choice='none')."))
                    break

                if self._valid_tool_names and parsed.name not in self._valid_tool_names:
                    self.stopped = True
                    self.stop_reason = "error"
                    events.append(
                        ErrorEvent(
                            f"Model called unknown tool {parsed.name!r}. "
                            "Ensure the tool is present in the request 'tools' list."
                        )
                    )
                    break

                if len(self.tool_calls) >= self._max_tool_calls:
                    self.stopped = True
                    self.stop_reason = "error"
                    events.append(
                        ErrorEvent(
                            f"Too many tool calls in one turn (max={self._max_tool_calls})."
                        )
                    )
                    break

                self.tool_calls.append(
                    ToolCall(id=f"call_{uuid.uuid4().hex}", name=parsed.name, arguments=parsed.arguments)
                )
                events.append(ToolCallEvent(tool_calls=list(self.tool_calls)))

                # Tool call completes the turn.
                self.stopped = True
                self.stop_reason = "tool_calls"

                # Any trailing content after </tool_call> is ignored for v0.
                _ = trailing
                break

            # Normal (non-think, non-tool) mode.
            next_think = self._buffer.find(self._THINK_OPEN)
            next_tool = self._buffer.find(self._TOOL_OPEN)

            next_special = -1
            if next_think != -1 and next_tool != -1:
                next_special = min(next_think, next_tool)
            elif next_think != -1:
                next_special = next_think
            elif next_tool != -1:
                next_special = next_tool

            if next_special != -1:
                # Emit content before the special tag.
                before = self._buffer[:next_special]
                self._buffer = self._buffer[next_special:]
                events.extend(self._emit_text(before))

                if self.stopped:
                    break

                if self._buffer.startswith(self._THINK_OPEN):
                    self._buffer = self._buffer[len(self._THINK_OPEN) :]
                    self._in_think = True
                    if self._emit_thinking:
                        events.append(ThinkingDeltaEvent(self._THINK_OPEN))
                        self._emitted_think_open = True
                    continue
                if self._buffer.startswith(self._TOOL_OPEN):
                    self._tool_buffer = self._TOOL_OPEN
                    self._buffer = self._buffer[len(self._TOOL_OPEN) :]
                    self._in_tool = True
                    continue

            # No special tags found in the current buffer.
            events.extend(self._emit_available_text())
            break

        return [e for e in events if not isinstance(e, DeltaEvent) or e.text]

    def finish(self) -> list[StreamEvent]:
        """Flush any remaining buffered content at end-of-generation."""
        if self.stopped:
            return []

        if self._in_tool:
            self.stopped = True
            self.stop_reason = "error"
            return [ErrorEvent("Incomplete <tool_call> block in model output.")]

        # If we're still inside a think block, drop it.
        if self._in_think:
            self._buffer = ""
            return []

        # Emit remaining buffer (apply stop sequences if needed).
        remaining = self._buffer
        self._buffer = ""
        return self._emit_text(remaining)

    def _emit_text(self, text: str) -> list[StreamEvent]:
        if not text:
            return []

        if self._stop_sequences:
            idx = self._find_earliest_stop(text)
            if idx is not None:
                before = text[:idx]
                self.stopped = True
                self.stop_reason = "stop"
                return [DeltaEvent(before)] if before else []

        return [DeltaEvent(text)]

    def _emit_available_text(self) -> list[StreamEvent]:
        if not self._buffer:
            return []

        if self._tail_keep <= 0:
            text = self._buffer
            self._buffer = ""
            return self._emit_text(text)

        if len(self._buffer) <= self._tail_keep:
            return []

        safe_end = len(self._buffer) - self._tail_keep
        safe = self._buffer[:safe_end]
        self._buffer = self._buffer[safe_end:]
        return self._emit_text(safe)

    def _find_earliest_stop(self, text: str) -> int | None:
        earliest: int | None = None
        for s in self._stop_sequences:
            idx = text.find(s)
            if idx == -1:
                continue
            if earliest is None or idx < earliest:
                earliest = idx
        return earliest


class ChatEngine:
    """Core chat inference engine.

    Thread-safety:
        The underlying adapter is not thread-safe. This engine serializes access
        with a global lock (single-flight).
    """

    def __init__(self, adapter: Any, *, config: EngineConfig | None = None) -> None:
        self._adapter = adapter
        self._config = config or EngineConfig()
        self._lock = threading.Lock()

    @property
    def adapter(self) -> Any:
        """Access to the underlying adapter for session management."""
        return self._adapter

    @property
    def model_info(self) -> dict[str, Any]:
        return getattr(self._adapter, "model_info", {})

    def shutdown(self) -> None:
        unload = getattr(self._adapter, "unload", None)
        if callable(unload):
            unload()

    def _tool_names(self, tools: Sequence[dict[str, Any]]) -> set[str]:
        names: set[str] = set()
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") != "function":
                continue
            fn = tool.get("function")
            if isinstance(fn, dict):
                name = fn.get("name")
                if isinstance(name, str) and name:
                    names.add(name)
        return names

    def _inject_tool_choice(self, messages: list[ChatMessage], tool_choice: Any) -> list[ChatMessage]:
        if tool_choice is None:
            return messages

        instruction: str | None = None
        if tool_choice == "none":
            instruction = "Tool choice: do not call any tools."
        elif tool_choice == "required":
            instruction = "Tool choice: you must call a tool for this response."
        elif isinstance(tool_choice, dict):
            # OpenAI format: {"type":"function","function":{"name":"..."}}
            fn = tool_choice.get("function") if tool_choice.get("type") == "function" else None
            name = fn.get("name") if isinstance(fn, dict) else None
            if isinstance(name, str) and name:
                instruction = f"Tool choice: call only the tool named {name!r}."

        if instruction is None:
            return messages

        if messages and messages[0].role == "system":
            updated = ChatMessage(
                role="system",
                content=((messages[0].content or "").rstrip() + "\n\n" + instruction).strip(),
                tool_calls=messages[0].tool_calls,
                tool_call_id=messages[0].tool_call_id,
            )
            return [updated, *messages[1:]]

        return [ChatMessage(role="system", content=instruction), *messages]

    def _messages_for_template(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            msg: dict[str, Any] = {"role": m.role}
            if m.content is not None:
                msg["content"] = m.content
            else:
                msg["content"] = ""

            if m.role == "assistant" and m.tool_calls:
                # The model chat template expects tool_call.function.arguments as a mapping (not a JSON string).
                tool_calls: list[dict[str, Any]] = []
                for tc in m.tool_calls:
                    tool_calls.append(
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": tc.arguments},
                        }
                    )
                msg["tool_calls"] = tool_calls

            if m.role == "tool" and m.tool_call_id:
                msg["tool_call_id"] = m.tool_call_id

            out.append(msg)
        return out

    def _validate_tool_choice(self, request: ChatRequest) -> None:
        tc = request.tool_choice
        if tc is None or tc == "auto":
            return

        tool_names = self._tool_names(request.tools)
        if tc == "none":
            return

        if not request.tools:
            raise ValueError("tool_choice was provided but no tools were supplied in the request.")

        if tc == "required":
            return

        if isinstance(tc, dict) and tc.get("type") == "function":
            fn = tc.get("function")
            name = fn.get("name") if isinstance(fn, dict) else None
            if isinstance(name, str) and name and tool_names and name not in tool_names:
                raise ValueError(f"tool_choice requested unknown tool {name!r}.")

    def _single_token_stop_ids(self, stop: Sequence[str]) -> tuple[list[int], list[str]]:
        """Split stops into (single-token stop ids, string stops).

        We only early-stop on single-token sequences. Anything else is enforced
        by the output parser (string scan).
        """
        stop_token_ids: list[int] = []
        stop_strings: list[str] = []
        tokenizer = getattr(self._adapter, "tokenizer", None)

        # Dynamically get special tokens from the tokenizer to filter from output.
        # This keeps the engine model-agnostic.
        if tokenizer is not None:
            all_special = getattr(tokenizer, "all_special_tokens", None)
            if all_special:
                stop_strings.extend(s for s in all_special if s)

        encode = getattr(tokenizer, "encode", None)
        if callable(encode):
            for s in stop:
                if not s:
                    continue
                try:
                    ids = encode(s, add_special_tokens=False)
                except Exception:
                    stop_strings.append(s)
                    continue
                if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], int):
                    stop_token_ids.append(ids[0])
                else:
                    stop_strings.append(s)
            return stop_token_ids, stop_strings

        return [], [s for s in stop if s]

    def _build_input_ids(self, request: ChatRequest) -> Any:
        apply_chat_template, template_messages, kwargs = self._prepare_chat_template(request)
        return apply_chat_template(template_messages, add_generation_prompt=True, **kwargs)

    def _effective_enable_thinking(self, request: ChatRequest) -> bool:
        enable_thinking = bool(self._config.enable_thinking)
        if request.reasoning_budget is not None:
            enable_thinking = True
        if request.chat_template_kwargs and "enable_thinking" in request.chat_template_kwargs:
            enable_thinking = bool(request.chat_template_kwargs["enable_thinking"])
        return enable_thinking

    def _effective_discard_thinking(self, request: ChatRequest) -> bool:
        discard_thinking = bool(self._config.discard_thinking)
        if request.discard_thinking is not None:
            discard_thinking = bool(request.discard_thinking)
        return discard_thinking

    def _chat_template_kwargs(self, request: ChatRequest, *, enable_thinking: bool) -> dict[str, Any]:
        """Build kwargs for tokenizer.apply_chat_template with strict parity.

        Important: This preserves the distinction between "tools omitted" vs `tools=[]`.
        """
        kwargs: dict[str, Any] = {
            "return_tensors": "pt",
            "enable_thinking": bool(enable_thinking),
        }

        # Merge request-level chat_template_kwargs (can override enable_thinking)
        if request.chat_template_kwargs:
            kwargs.update(request.chat_template_kwargs)
            # The engine controls add_generation_prompt explicitly.
            kwargs.pop("add_generation_prompt", None)

        # For tool_choice="none", omit tool definitions from the prompt to reduce the chance of
        # accidental tool calls (in addition to the injected instruction).
        tools = request.tools
        if request.tool_choice == "none":
            tools = []

        # Preserve "tools omitted" when tools==[].
        if tools:
            kwargs["tools"] = tools

        return kwargs

    def _prepare_chat_template(self, request: ChatRequest) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
        tokenizer = getattr(self._adapter, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Adapter has no tokenizer loaded.")

        self._validate_tool_choice(request)

        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
        if not callable(apply_chat_template):
            raise RuntimeError("Tokenizer does not support apply_chat_template().")

        messages = self._inject_tool_choice(list(request.messages), request.tool_choice)
        template_messages = self._messages_for_template(messages)
        kwargs = self._chat_template_kwargs(request, enable_thinking=self._effective_enable_thinking(request))
        return apply_chat_template, template_messages, kwargs

    def _compute_generation_boundary(
        self,
        *,
        apply_chat_template: Any,
        template_messages: list[dict[str, Any]],
        kwargs: dict[str, Any],
    ) -> tuple[int, Any, Any, Any]:
        """Compute the (end-of-user) boundary and generation prompt suffix.

        Returns:
            (boundary_pos, ids_no_gen, ids_with_gen, gen_prompt_ids)

        Raises:
            ValueError if the strict-prefix invariant is violated.
        """
        ids_no_gen = apply_chat_template(template_messages, add_generation_prompt=False, **kwargs)
        ids_with_gen = apply_chat_template(template_messages, add_generation_prompt=True, **kwargs)

        try:
            boundary_pos = int(getattr(ids_no_gen, "shape")[1])
            with_len = int(getattr(ids_with_gen, "shape")[1])
        except Exception as exc:  # pragma: no cover
            raise ValueError("apply_chat_template() must return a tensor with shape (1, seq_len).") from exc

        if boundary_pos < 0 or with_len < 0 or with_len < boundary_pos:
            raise ValueError("Invalid chat template boundary lengths.")
        if with_len == boundary_pos:
            raise ValueError("Chat template boundary is not a strict prefix (no generation prompt suffix).")

        prefix_ok = False
        try:
            import torch

            if (
                isinstance(ids_no_gen, torch.Tensor)
                and isinstance(ids_with_gen, torch.Tensor)
                and ids_no_gen.ndim == 2
                and ids_with_gen.ndim == 2
                and ids_no_gen.shape[0] == 1
                and ids_with_gen.shape[0] == 1
            ):
                prefix_ok = torch.equal(ids_with_gen[:, :boundary_pos], ids_no_gen)
        except Exception:
            prefix_ok = False

        if not prefix_ok:
            # Best-effort diagnostic suffix to help debug template drift.
            suffix = None
            try:
                tokenizer = getattr(self._adapter, "tokenizer", None)
                decode = getattr(tokenizer, "decode", None)
                if callable(decode):
                    suffix_ids = ids_with_gen[0, max(boundary_pos - 16, 0) : boundary_pos + 16].tolist()
                    suffix = decode(suffix_ids, skip_special_tokens=False)
            except Exception:
                suffix = None

            if suffix:
                logger.warning("apply_chat_template prefix invariant failed near boundary: %r", suffix)
            raise ValueError("Chat template strict-prefix boundary invariant failed.")

        gen_prompt_ids = ids_with_gen[:, boundary_pos:]
        return boundary_pos, ids_no_gen, ids_with_gen, gen_prompt_ids

    async def astream_chat(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        """Async iterator streaming internal events."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        cancel = threading.Event()

        repetition_cfg = self._config.repetition_detection.merged(
            request.extra.get("repetition_detection") if request.extra else None
        )

        enable_thinking = self._effective_enable_thinking(request)
        discard_thinking = self._effective_discard_thinking(request)
        emit_thinking = bool(request.stream_thinking) and enable_thinking

        apply_chat_template, template_messages, kwargs = self._prepare_chat_template(request)

        boundary_pos: int | None = None
        ids_no_gen = None
        gen_prompt_ids = None

        if request.session_id and enable_thinking and discard_thinking:
            try:
                boundary_pos, ids_no_gen, input_ids, gen_prompt_ids = self._compute_generation_boundary(
                    apply_chat_template=apply_chat_template,
                    template_messages=template_messages,
                    kwargs=kwargs,
                )
            except ValueError as exc:
                # Safety-over-performance fallback: keep discard-thinking enabled but use
                # checkpoint-before-user (Option A) to avoid relying on a potentially-wrong boundary.
                logger.warning("Falling back to checkpoint-before-user: %s", exc)
                input_ids = apply_chat_template(template_messages, add_generation_prompt=True, **kwargs)
        else:
            input_ids = apply_chat_template(template_messages, add_generation_prompt=True, **kwargs)

        try:
            prompt_tokens = int(getattr(input_ids, "shape")[1])
        except Exception:
            prompt_tokens = 0

        if prompt_tokens and prompt_tokens > self._config.max_prompt_tokens:
            yield ErrorEvent(
                f"Prompt too long: {prompt_tokens} tokens (max={self._config.max_prompt_tokens})."
            )
            return

        stop_token_ids, stop_strings = self._single_token_stop_ids(request.stop)
        tool_names = self._tool_names(request.tools)
        parser = _ModelOutputParser(
            stop_sequences=stop_strings,
            valid_tool_names=tool_names,
            max_tool_calls=self._config.max_tool_calls_per_turn,
            allow_tool_calls=request.tool_choice != "none",
            start_in_think=enable_thinking,
            emit_thinking=emit_thinking,
        )

        def worker() -> None:
            started = time.monotonic()
            first_token_at: float | None = None
            completion_tokens = 0
            finish_reason: str = "stop"

            normalized_raw_content_for_history: str | None = None

            # Per-request stream policy.
            flush_n = max(int(request.stream_options.flush_every_n_tokens), 1)
            flush_ms = max(int(request.stream_options.flush_every_ms), 1)
            flush_s = flush_ms / 1000.0

            token_buffer: list[int] = []
            last_flush = time.monotonic()

            assistant_text_parts: list[str] = []
            assistant_raw_text_parts: list[str] = []  # Raw text including <think> blocks
            assistant_tool_calls: list[ToolCall] = []

            repetition_tail: deque[int] | None = None
            if repetition_cfg.enabled:
                repetition_tail = deque(maxlen=repetition_cfg.tail_len)

            def _emit_event(event: StreamEvent) -> None:
                nonlocal assistant_tool_calls
                if isinstance(event, DeltaEvent):
                    if event.text:
                        assistant_text_parts.append(event.text)
                elif isinstance(event, ThinkingDeltaEvent):
                    # Thinking deltas are streamed but never counted as assistant output content.
                    pass
                elif isinstance(event, ToolCallEvent):
                    assistant_tool_calls = list(event.tool_calls)
                loop.call_soon_threadsafe(queue.put_nowait, event)

            try:
                with self._lock:

                    # Session-based generation path
                    if request.session_id:
                        append_from = request.session_append_from_pos
                        if append_from is None:
                            append_from = 0
                        try:
                            append_from = int(append_from)
                        except Exception:
                            append_from = 0
                        if append_from < 0:
                            append_from = 0

                        discard_session_thinking = enable_thinking and discard_thinking

                        checkpoint = None
                        commit_from_pos = None
                        fallback_checkpoint = None
                        fallback_commit_from_pos = None

                        if discard_session_thinking:
                            if not hasattr(self._adapter, "checkpoint_session") or not hasattr(
                                self._adapter, "restore_session_checkpoint"
                            ):
                                raise RuntimeError(
                                    "Adapter does not support checkpoint/restore required for discard_thinking."
                                )

                            # Option B (fast path): checkpoint after user boundary.
                            if boundary_pos is not None and ids_no_gen is not None and gen_prompt_ids is not None:
                                # Keep a fallback checkpoint in case boundary invariants drift mid-flight.
                                fallback_checkpoint = self._adapter.checkpoint_session(request.session_id)
                                fallback_commit_from_pos = int(append_from)

                                if append_from > int(boundary_pos):
                                    raise ValueError(
                                        f"session_append_from_pos={append_from} exceeds boundary_pos={boundary_pos}."
                                    )

                                try:
                                    delta_user_ids = ids_no_gen[:, append_from:boundary_pos]
                                except Exception:
                                    delta_user_ids = ids_no_gen

                                if getattr(delta_user_ids, "numel", lambda: 0)() > 0:
                                    self._adapter.append_to_session(
                                        cache_id=request.session_id,
                                        input_ids=delta_user_ids,
                                    )

                                sess_info = self._adapter.get_session_info(request.session_id)
                                cur = int(sess_info.get("current_pos", -1))
                                if cur != int(boundary_pos):
                                    raise ValueError(
                                        f"Session cursor mismatch after user prefill: current_pos={cur} "
                                        f"!= boundary_pos={boundary_pos}."
                                    )

                                checkpoint = self._adapter.checkpoint_session(request.session_id)
                                commit_from_pos = int(boundary_pos)

                                if getattr(gen_prompt_ids, "numel", lambda: 0)() > 0:
                                    self._adapter.append_to_session(
                                        cache_id=request.session_id,
                                        input_ids=gen_prompt_ids,
                                    )
                            else:
                                # Option A (fallback): checkpoint before appending user.
                                checkpoint = self._adapter.checkpoint_session(request.session_id)
                                commit_from_pos = int(append_from)

                                # Append full delta prompt tokens (includes user + gen prompt).
                                try:
                                    delta_input_ids = input_ids[:, append_from:]
                                except Exception:
                                    delta_input_ids = input_ids

                                if getattr(delta_input_ids, "numel", lambda: 0)() > 0:
                                    self._adapter.append_to_session(
                                        cache_id=request.session_id,
                                        input_ids=delta_input_ids,
                                    )
                        else:
                            # Default: append full delta prompt tokens (server may provide full-history messages).
                            try:
                                delta_input_ids = input_ids[:, append_from:]
                            except Exception:
                                delta_input_ids = input_ids

                            if getattr(delta_input_ids, "numel", lambda: 0)() > 0:
                                self._adapter.append_to_session(
                                    cache_id=request.session_id,
                                    input_ids=delta_input_ids,
                                )

                        # Generate from the session
                        token_iter = self._adapter.stream_generate_session(
                            cache_id=request.session_id,
                            max_new_tokens=int(request.max_tokens),
                            temperature=float(request.temperature or 0.0),
                            stop_token_ids=stop_token_ids or None,
                        )
                    else:
                        # Stateless generation path
                        token_iter = self._adapter.stream_generate(
                            input_ids,
                            max_new_tokens=int(request.max_tokens),
                            temperature=float(request.temperature or 0.0),
                            stop_token_ids=stop_token_ids or None,
                            backend=self._config.default_backend,
                            reasoning_budget=request.reasoning_budget,
                            enable_thinking=enable_thinking,
                        )

                    # Track if we've bailed out of thinking due to repetition
                    thinking_bailout_done = False

                    try:
                        for token in token_iter:
                            if cancel.is_set():
                                finish_reason = "cancelled"
                                break

                            completion_tokens += 1
                            if first_token_at is None:
                                first_token_at = time.monotonic()

                            try:
                                token_id = int(token.item())
                            except Exception:
                                # Fall back to best-effort stringification.
                                token_id = int(token)  # type: ignore[arg-type]

                            token_buffer.append(token_id)

                            if repetition_tail is not None:
                                repetition_tail.append(token_id)
                                if (
                                    completion_tokens >= repetition_cfg.min_generated_tokens
                                    and (completion_tokens % repetition_cfg.check_every) == 0
                                ):
                                    hit = detect_repetition_kmp_tail(
                                        list(repetition_tail),
                                        tail_len=repetition_cfg.tail_len,
                                        min_generated_tokens=0,
                                        min_repeats=repetition_cfg.min_repeats,
                                        max_period=repetition_cfg.max_period,
                                        min_unique_tokens=repetition_cfg.min_unique_tokens,
                                    )
                                    if hit is not None:
                                        # Flush buffer before checking parser state
                                        if token_buffer:
                                            _flush_token_buffer(token_buffer, parser, _emit_event, assistant_raw_text_parts)
                                            token_buffer.clear()

                                        # If we're in thinking mode and haven't bailed out yet,
                                        # inject </think> and continue instead of stopping
                                        if parser._in_think and not thinking_bailout_done:
                                            logger.debug(
                                                "Repetition in thinking - injecting </think>: period=%d repeats=%d completion_tokens=%d",
                                                hit.period,
                                                hit.repeats,
                                                completion_tokens,
                                            )
                                            # Feed </think> to parser to exit thinking mode
                                            for event in parser.feed("</think>"):
                                                _emit_event(event)
                                            # Clear repetition tail to give fresh start
                                            repetition_tail.clear()
                                            thinking_bailout_done = True
                                            # Continue generating (don't break)
                                            continue

                                        logger.debug(
                                            "Repetition early-stop: period=%d repeats=%d checked_tail_len=%d completion_tokens=%d",
                                            hit.period,
                                            hit.repeats,
                                            hit.checked_tail_len,
                                            completion_tokens,
                                        )
                                        finish_reason = "repetition"
                                        break

                            now = time.monotonic()
                            if len(token_buffer) < flush_n and (now - last_flush) < flush_s:
                                continue

                            last_flush = now
                            _flush_token_buffer(token_buffer, parser, _emit_event, assistant_raw_text_parts)
                            token_buffer.clear()

                            if parser.stopped and parser.stop_reason == "tool_calls":
                                finish_reason = "tool_calls"
                                break
                            if parser.stopped and parser.stop_reason == "stop":
                                finish_reason = "stop"
                                break
                            if parser.stopped and parser.stop_reason == "error":
                                finish_reason = "error"
                                break
                    finally:
                        # Explicitly close the generator to ensure its finally block runs.
                        # This is critical for session-based generation where the generator's
                        # finally block persists the KV cache state.
                        if hasattr(token_iter, 'close'):
                            token_iter.close()

                    # Final flush.
                    if token_buffer and not parser.stopped:
                        _flush_token_buffer(token_buffer, parser, _emit_event, assistant_raw_text_parts)
                        token_buffer.clear()

                    # Flush parser tail.
                    if not parser.stopped:
                        for event in parser.finish():
                            _emit_event(event)
                            if isinstance(event, ErrorEvent):
                                finish_reason = "error"

                    # Discard-thinking commit: restore to a checkpoint and append tokens for persisted history.
                    if request.session_id and enable_thinking and discard_thinking:
                        if checkpoint is None or commit_from_pos is None:
                            raise RuntimeError("Discard-thinking flow missing checkpoint state.")

                        self._adapter.restore_session_checkpoint(
                            cache_id=request.session_id,
                            checkpoint=checkpoint,
                        )

                        # Persist tool calls as a structured assistant message.
                        assistant_msg = None
                        if assistant_tool_calls:
                            assistant_msg = ChatMessage(
                                role="assistant",
                                content=None,
                                tool_calls=list(assistant_tool_calls),
                            )
                        else:
                            assistant_msg = ChatMessage(
                                role="assistant",
                                content="".join(assistant_text_parts),
                            )

                        persisted_messages = self._inject_tool_choice(
                            [*list(request.messages), assistant_msg],
                            request.tool_choice,
                        )
                        template_persisted = self._messages_for_template(persisted_messages)

                        ids_persisted = apply_chat_template(
                            template_persisted,
                            add_generation_prompt=False,
                            **kwargs,
                        )

                        # Sanity check: persisted prompt should start with the end-of-user prefix.
                        if boundary_pos is not None and ids_no_gen is not None:
                            import torch

                            if (
                                isinstance(ids_persisted, torch.Tensor)
                                and isinstance(ids_no_gen, torch.Tensor)
                                and ids_persisted.ndim == 2
                                and ids_no_gen.ndim == 2
                                and ids_persisted.shape[0] == 1
                                and ids_no_gen.shape[0] == 1
                                and ids_persisted.shape[1] >= int(boundary_pos)
                                and not torch.equal(ids_persisted[:, : int(boundary_pos)], ids_no_gen)
                            ):
                                if fallback_checkpoint is not None and fallback_commit_from_pos is not None:
                                    logger.warning(
                                        "Persisted prompt prefix mismatch; falling back to checkpoint-before-user."
                                    )
                                    self._adapter.restore_session_checkpoint(
                                        cache_id=request.session_id,
                                        checkpoint=fallback_checkpoint,
                                    )
                                    commit_from_pos = int(fallback_commit_from_pos)
                                else:
                                    raise ValueError(
                                        "Persisted prompt no longer matches the end-of-user prefix."
                                    )

                        try:
                            delta_commit_ids = ids_persisted[:, int(commit_from_pos) :]
                        except Exception:
                            delta_commit_ids = ids_persisted

                        if getattr(delta_commit_ids, "numel", lambda: 0)() > 0:
                            self._adapter.append_to_session(
                                cache_id=request.session_id,
                                input_ids=delta_commit_ids,
                            )

                    # discard_thinking=False session path: the model often stops *before* emitting <|im_end|>,
                    # but apply_chat_template(add_generation_prompt=False) will include it for assistant messages.
                    # To keep KV/history in sync, append any missing tail tokens after generation.
                    if request.session_id and enable_thinking and not discard_thinking and not assistant_tool_calls:
                        # Build normalized raw content that matches the template's generation prompt prefix.
                        normalized_raw_content_for_history = (
                            "".join(assistant_raw_text_parts) if assistant_raw_text_parts else None
                        )
                        if normalized_raw_content_for_history:
                            if (
                                _ModelOutputParser._THINK_CLOSE in normalized_raw_content_for_history
                                and _ModelOutputParser._THINK_OPEN not in normalized_raw_content_for_history[:64]
                            ):
                                normalized_raw_content_for_history = (
                                    _ModelOutputParser._THINK_OPEN + "\n" + normalized_raw_content_for_history
                                )

                        assistant_msg = ChatMessage(
                            role="assistant",
                            content=normalized_raw_content_for_history,
                        )
                        persisted_messages = self._inject_tool_choice(
                            [*list(request.messages), assistant_msg],
                            request.tool_choice,
                        )
                        template_persisted = self._messages_for_template(persisted_messages)
                        ids_persisted = apply_chat_template(
                            template_persisted,
                            add_generation_prompt=False,
                            **kwargs,
                        )

                        sess_info = self._adapter.get_session_info(request.session_id)
                        cur = int(sess_info.get("current_pos", 0))
                        try:
                            expected_total = int(getattr(ids_persisted, "shape")[1])
                        except Exception:
                            expected_total = cur

                        if expected_total > cur:
                            try:
                                delta_tail_ids = ids_persisted[:, cur:expected_total]
                            except Exception:
                                delta_tail_ids = ids_persisted
                            if getattr(delta_tail_ids, "numel", lambda: 0)() > 0:
                                self._adapter.append_to_session(
                                    cache_id=request.session_id,
                                    input_ids=delta_tail_ids,
                                )

                    # If we exhausted the token budget without an explicit stop/tool call/cancel,
                    # report a length stop (best-effort; adapter doesn't expose stop reason).
                    if (
                        finish_reason == "stop"
                        and not parser.stopped
                        and completion_tokens >= int(request.max_tokens)
                    ):
                        finish_reason = "length"

            except Exception as exc:
                finish_reason = "error"
                loop.call_soon_threadsafe(queue.put_nowait, ErrorEvent(f"Generation failed: {exc}"))
            finally:
                ended = time.monotonic()
                prefill_s = None if first_token_at is None else max(first_token_at - started, 0.0)
                decode_s = None
                if first_token_at is not None:
                    decode_s = max(ended - first_token_at, 0.0)

                tok_per_s = None
                if decode_s and decode_s > 0 and completion_tokens > 0:
                    tok_per_s = completion_tokens / decode_s

                # Include raw content (with thinking) when discard_thinking=False for sessions
                raw_content = None
                if request.session_id and enable_thinking and not discard_thinking:
                    raw_content = normalized_raw_content_for_history
                    if raw_content is None:
                        raw_content = "".join(assistant_raw_text_parts) if assistant_raw_text_parts else None
                        # When enable_thinking=True, the generation prompt already includes <think>.
                        # The model output stream may therefore omit the opening tag and begin directly
                        # with the thinking text, later emitting only </think>. For history/KV sync,
                        # normalize by re-introducing the opening tag when needed.
                        if raw_content:
                            if (
                                _ModelOutputParser._THINK_CLOSE in raw_content
                                and _ModelOutputParser._THINK_OPEN not in raw_content[:64]
                            ):
                                raw_content = _ModelOutputParser._THINK_OPEN + "\n" + raw_content

                final = FinalEvent(
                    finish_reason=finish_reason
                    if finish_reason in {"stop", "length", "tool_calls", "cancelled", "error", "repetition"}
                    else "stop",
                    usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
                    timing=Timing(
                        prefill_s=prefill_s,
                        decode_s=decode_s,
                        total_s=max(ended - started, 0.0),
                        tok_per_s=tok_per_s,
                    ),
                    raw_content=raw_content,
                )
                loop.call_soon_threadsafe(queue.put_nowait, final)
                loop.call_soon_threadsafe(queue.put_nowait, None)

        def _flush_token_buffer(
            token_ids: list[int],
            parser_: _ModelOutputParser,
            emit: Any,
            raw_text_parts: list[str] | None = None,
        ) -> None:
            tokenizer = getattr(self._adapter, "tokenizer", None)
            decode = getattr(tokenizer, "decode", None)
            if not callable(decode):
                raise RuntimeError("Tokenizer does not support decode().")

            text = decode(token_ids, skip_special_tokens=False)
            if raw_text_parts is not None:
                raw_text_parts.append(text)
            for event in parser_.feed(text):
                emit(event)

        thread = threading.Thread(target=worker, name=f"superlinear-gen-{uuid.uuid4().hex}", daemon=True)
        thread.start()

        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
        except asyncio.CancelledError:
            cancel.set()
            raise
        finally:
            # If the consumer stops early (disconnect / generator close), cancel generation promptly.
            cancel.set()

    async def generate_chat(self, request: ChatRequest) -> dict[str, Any]:
        """Non-streaming chat completion.

        Returns:
            Dict containing:
              - content: str | None
              - tool_calls: list[ToolCall]
              - finish_reason: str
              - usage: Usage
              - timing: Timing
              - raw_content: str | None (if discard_thinking=False)
        """
        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        usage: Usage | None = None
        timing: Timing | None = None
        finish_reason = "stop"
        raw_content: str | None = None

        async for event in self.astream_chat(request):
            if isinstance(event, DeltaEvent):
                content_parts.append(event.text)
            elif isinstance(event, ToolCallEvent):
                tool_calls = event.tool_calls
                finish_reason = "tool_calls"
            elif isinstance(event, FinalEvent):
                usage = event.usage
                timing = event.timing
                raw_content = event.raw_content
                # Don't override tool_calls finish reason
                if finish_reason != "tool_calls":
                    finish_reason = event.finish_reason
            elif isinstance(event, ErrorEvent):
                raise RuntimeError(event.message)

        return {
            "content": "".join(content_parts) if content_parts else None,
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
            "usage": usage,
            "timing": timing,
            "raw_content": raw_content,
        }
