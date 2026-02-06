"""Core chat request/streaming event types.

These types are internal to the library and are intentionally decoupled from:
- HTTP transport (FastAPI / SSE)
- OpenAI request/response JSON envelopes

The goal is to keep the core engine reusable for future API surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ToolChoice = Literal["auto", "none", "required"] | dict[str, Any]


@dataclass(frozen=True)
class StreamOptions:
    """Chunked streaming policy."""

    flush_every_n_tokens: int = 8
    flush_every_ms: int = 50


@dataclass(frozen=True)
class ToolCall:
    """A parsed tool call (function name + JSON-serializable arguments)."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ChatMessage:
    """A normalized chat message.

    Notes:
    - `tool_calls` is only meaningful for assistant messages.
    - `tool_call_id` is only meaningful for tool messages (tool outputs).
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None


@dataclass(frozen=True)
class ChatRequest:
    """Normalized internal chat request."""

    messages: list[ChatMessage]
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_choice: ToolChoice | None = None
    max_tokens: int = 4096
    temperature: float = 0.0
    top_p: float = 1.0
    stop: list[str] = field(default_factory=list)
    stream: bool = False
    stream_options: StreamOptions = field(default_factory=StreamOptions)
    chat_template_kwargs: dict[str, Any] | None = None  # Additional kwargs for chat template
    reasoning_budget: int | None = None  # Max tokens for thinking phase (enables thinking when set)
    discard_thinking: bool | None = None  # If set, discard <think>...</think> from persisted session state
    stream_thinking: bool | None = None  # If True, stream <think>...</think> content as separate deltas
    session_id: str | None = None  # Session ID for stateful chat (KV cache reuse)
    session_append_from_pos: int | None = None  # If set, only append prompt tokens from this position
    extra: dict[str, Any] = field(default_factory=dict)  # Engine-specific per-request overrides


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class Timing:
    prefill_s: float | None = None
    decode_s: float | None = None
    total_s: float | None = None
    tok_per_s: float | None = None


@dataclass(frozen=True)
class DeltaEvent:
    """Chunked text delta."""

    text: str


@dataclass(frozen=True)
class ThinkingDeltaEvent:
    """Chunked thinking delta (text inside <think>...</think>)."""

    text: str


@dataclass(frozen=True)
class ToolCallEvent:
    """A completed tool call (or batch of tool calls)."""

    tool_calls: list[ToolCall]


@dataclass(frozen=True)
class FinalEvent:
    """Terminal event for a generation."""

    finish_reason: Literal["stop", "length", "tool_calls", "cancelled", "error", "repetition"]
    usage: Usage
    timing: Timing
    raw_content: str | None = None  # Unstripped content (includes <think> if any) when discard_thinking=False


@dataclass(frozen=True)
class ErrorEvent:
    """Non-terminal or terminal error event."""

    message: str
    retryable: bool = False


StreamEvent = DeltaEvent | ThinkingDeltaEvent | ToolCallEvent | FinalEvent | ErrorEvent
