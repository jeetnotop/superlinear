from __future__ import annotations

import atexit
import os
import select
import shutil
import shlex
import sys
import time
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Enable readline for arrow keys, history navigation, and line editing.
# This gives us: ↑/↓ (history), ←/→ (cursor), Ctrl+A/E (start/end), Ctrl+R (search), Ctrl+L (clear).
try:
    import readline
except ImportError:
    readline = None  # type: ignore[assignment]  # Windows fallback

from apps.cli.client import HttpError, SuperlinearClient
from apps.cli.local_snapshots import delete_local_snapshot, list_local_snapshots
from apps.cli.locks import AlreadyLockedError, SessionLock
from apps.cli.state import CliState, load_state, save_state


def _chat_history_file_path() -> Path:
    return Path.home() / ".config" / "spl" / "chat_history"


def _setup_readline_history() -> None:
    """Set up persistent command history for the REPL."""
    if readline is None:
        return
    history_file = _chat_history_file_path()
    history_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        readline.read_history_file(history_file)
    except FileNotFoundError:
        pass
    readline.set_history_length(1000)
    atexit.register(readline.write_history_file, history_file)


# Commands for tab completion
_CHAT_COMMANDS = [
    "/help", "/exit", "/clear", "/history", "/new", "/rm", "/head", "/tail",
    "/show", "/ls", "/switch", "/save", "/load", "/stats",
]


def _setup_completer() -> None:
    """Set up tab completion for REPL commands."""
    if readline is None:
        return

    def completer(text: str, state: int) -> str | None:
        if text.startswith("/"):
            matches = [cmd for cmd in _CHAT_COMMANDS if cmd.startswith(text)]
        else:
            matches = []
        return matches[state] if state < len(matches) else None

    readline.set_completer(completer)
    readline.set_completer_delims(" \t\n")
    readline.parse_and_bind("tab: complete")


def _cmd_history(n: int = 20) -> None:
    """Show the last n entries from readline input history."""
    if readline is None:
        print("history not available (readline not loaded)", file=sys.stderr)
        return
    length = readline.get_current_history_length()
    if length == 0:
        print("(no history)")
        return
    start = max(1, length - n + 1)
    for i in range(start, length + 1):
        item = readline.get_history_item(i)
        if item:
            print(f"{i:4d}  {item}")


def _cmd_history_clear() -> None:
    """Clear readline input history (both in-memory and on disk)."""
    if readline is None:
        print("history not available (readline not loaded)", file=sys.stderr)
        return
    try:
        readline.clear_history()
    except Exception as exc:
        print(f"failed to clear history: {exc}", file=sys.stderr)
        return

    # Persist the empty history immediately.
    history_file = _chat_history_file_path()
    try:
        history_file.parent.mkdir(parents=True, exist_ok=True)
        readline.write_history_file(history_file)
    except Exception:
        # Best effort; the in-memory history is already cleared.
        pass
    print("cleared input history")


# Default system prompt for Superlinear Chat.
# Emphasizes long-context review and instruction-following.
DEFAULT_SYSTEM_PROMPT = """\
You are Superlinear Chat, a helpful, harmless, and honest AI assistant developed by concavity.ai.

You are operating in a stateful chat session: you can see the prior messages in this conversation (they are provided as chat history). Do not claim you are “stateless” or that you “can’t access previous messages”. If asked to recall, summarize, or list prior turns, do so to the best of your ability. If a user requests exact verbatim quotes of very long messages, provide short excerpts and offer to continue with the full text.

## Core Principles

1. **Review conversation history thoroughly**: Before responding, carefully review the entire conversation history. This is a long-context assistant—important details, constraints, or prior decisions may appear much earlier in the conversation. Never assume you remember everything; re-read to ensure continuity and avoid contradicting or repeating yourself.

2. **Follow instructions precisely**: Prioritize the user's explicit instructions over assumptions. When instructions conflict with conventions or best practices, follow the instructions while noting any concerns. Ask clarifying questions only when truly necessary.

3. **Be helpful and thorough**: Provide comprehensive, accurate answers that fully address the user's needs. Anticipate follow-up questions and include relevant context proactively.

4. **Be honest about limitations**: If you don't know something, say so. Don't fabricate information. Distinguish clearly between facts and opinions or speculation.

## Response Style

- Be concise for simple questions; be thorough for complex ones
- Use markdown formatting effectively: headers, lists, code blocks, emphasis
- Match the user's tone and technical level
- Structure long responses with clear sections

## For Technical and Code Requests

- Write clean, well-documented, idiomatic code
- Follow language-specific best practices and conventions
- Include error handling and edge case considerations
- Explain your approach when helpful, but prioritize working code
- When modifying existing code, preserve the original style and conventions

## For Complex Problems

- Use <think>...</think> blocks for extended reasoning when helpful
- Break down complex tasks into clear steps
- Show your reasoning process for non-trivial decisions
- Consider multiple approaches before recommending one

## For Creative Tasks

- Be imaginative and explore possibilities
- Offer alternatives and variations when appropriate
- Respect the user's creative vision while offering constructive input

## Boundaries

- Politely decline requests that would cause harm
- Respect privacy and don't ask for unnecessary personal information
- For medical, legal, or financial matters, recommend consulting qualified professionals

## Multi-turn Conversations

- Maintain awareness of the full conversation context
- Reference previous messages and decisions appropriately
- Adapt smoothly when the user changes topics or refines their request
- When asked to revise or build on prior work, ensure changes are consistent with the established context
"""


class ChatReplError(RuntimeError):
    pass


@dataclass
class TurnStats:
    finish_reason: str | None = None
    ttft_s: float | None = None
    total_s: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tok_per_s: float | None = None
    server_prefill_s: float | None = None
    server_decode_s: float | None = None
    server_total_s: float | None = None


def _now_utc_compact() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def _new_session_id(*, prefix: str = "chat") -> str:
    # Keep IDs short, URL-safe, and easy to read.
    import secrets

    return f"{prefix}_{_now_utc_compact()}_{secrets.token_hex(3)}"


def _ensure_reachable(client: SuperlinearClient) -> None:
    try:
        client.health()
    except HttpError as exc:
        raise ChatReplError(
            f"Server unreachable at {client.base_url}. Start it with `spl server start --model <model>` "
            f"or pass `--url`.\n{exc}"
        ) from exc


def _get_session_counters(*, client: SuperlinearClient, session_id: str) -> tuple[int, int]:
    """Return (message_count, cache_position) for a session."""
    try:
        info = client.request_json("GET", f"/v1/sessions/{session_id}", timeout_s=5.0)
    except HttpError as exc:
        raise ChatReplError(str(exc)) from exc

    if not isinstance(info, dict):
        raise ChatReplError("Invalid response from server for /v1/sessions/<id>")

    msg_count = info.get("message_count")
    cache_pos = info.get("cache_position")
    if cache_pos is None:
        cache_pos = info.get("current_pos")

    try:
        msg_count_i = int(msg_count or 0)
    except Exception:
        msg_count_i = 0
    try:
        cache_pos_i = int(cache_pos or 0)
    except Exception:
        cache_pos_i = 0

    return msg_count_i, cache_pos_i


def _session_exists(client: SuperlinearClient, session_id: str) -> bool:
    try:
        client.request_json("GET", f"/v1/sessions/{session_id}", timeout_s=5.0)
        return True
    except HttpError as exc:
        if exc.status_code == 404:
            return False
        raise


def _create_session(client: SuperlinearClient, session_id: str, *, max_seq_len: int | None = None) -> None:
    payload: dict[str, Any] = {"session_id": session_id}
    if max_seq_len is not None:
        payload["max_seq_len"] = int(max_seq_len)
    try:
        client.request_json("POST", "/v1/sessions", payload=payload, timeout_s=30.0)
    except HttpError as exc:
        # Idempotent behavior for "start"/resume flows.
        if exc.status_code == 409:
            return
        raise


def _maybe_resize_session(
    client: SuperlinearClient,
    session_id: str,
    *,
    min_max_seq_len: int | None,
    strategy: str = "auto",
) -> None:
    if min_max_seq_len is None:
        return
    try:
        info = client.request_json("GET", f"/v1/sessions/{session_id}", timeout_s=10.0)
    except HttpError:
        return
    if not isinstance(info, dict):
        return
    try:
        cur = int(info.get("max_seq_len") or 0)
    except Exception:
        cur = 0
    target = int(min_max_seq_len)
    if target <= 0 or (cur > 0 and target <= cur):
        return

    try:
        client.request_json(
            "POST",
            f"/v1/sessions/{session_id}/resize",
            payload={"max_seq_len": target, "strategy": strategy},
            timeout_s=300.0,
        )
    except HttpError as exc:
        raise ChatReplError(
            "Failed to resize session context length. "
            "This can happen if the target is too large for GPU memory. "
            f"(session_id={session_id} target_max_seq_len={target}): {exc}"
        ) from exc


def _banner(*, url: str, session_id: str, resumed: bool) -> None:
    mode = "resumed" if resumed else "new"
    print(f"server={url}")
    print(f"session_id={session_id} ({mode})")
    print("type /help for commands")


def _format_metrics(stats: TurnStats) -> str:
    parts: list[str] = []
    if stats.ttft_s is not None:
        parts.append(f"ttft={stats.ttft_s:.3f}s")
    if stats.tok_per_s is not None:
        parts.append(f"tok/s={stats.tok_per_s:.2f}")
    if stats.prompt_tokens is not None and stats.completion_tokens is not None:
        parts.append(f"tokens={stats.prompt_tokens}+{stats.completion_tokens}")
    if stats.finish_reason is not None:
        parts.append(f"finish={stats.finish_reason}")
    if stats.total_s is not None:
        parts.append(f"wall={stats.total_s:.3f}s")
    return " ".join(parts) if parts else ""


def _stats_detail(stats: TurnStats) -> str:
    lines = []
    if stats.finish_reason is not None:
        lines.append(f"finish_reason={stats.finish_reason}")
    if stats.ttft_s is not None:
        lines.append(f"ttft_s={stats.ttft_s:.6f}")
    if stats.total_s is not None:
        lines.append(f"wall_s={stats.total_s:.6f}")
    if stats.prompt_tokens is not None:
        lines.append(f"prompt_tokens={stats.prompt_tokens}")
    if stats.completion_tokens is not None:
        lines.append(f"completion_tokens={stats.completion_tokens}")
    if stats.tok_per_s is not None:
        lines.append(f"tok_per_s={stats.tok_per_s:.6f}")
    if stats.server_prefill_s is not None:
        lines.append(f"server_prefill_s={stats.server_prefill_s:.6f}")
    if stats.server_decode_s is not None:
        lines.append(f"server_decode_s={stats.server_decode_s:.6f}")
    if stats.server_total_s is not None:
        lines.append(f"server_total_s={stats.server_total_s:.6f}")
    return "\n".join(lines)


def _set_active_session(state: CliState, session_id: str) -> None:
    state.active_chat_session_id = session_id
    # Keep legacy field aligned (also handled in save_state()).
    state.chat_checkpoint_snapshot_id = state.chat_checkpoints.get(session_id)


def _cmd_help() -> None:
    print(
        "\n".join(
            [
                "commands:",
                "  /help",
                "  /exit [-c]      exit (--clean/-c: delete session)",
                "  /clear          clear screen",
                "  /history [n]    show last n input commands (default 20)",
                "  /history clear  clear input command history",
                "  /new            start new session (keeps old)",
                "  /rm             delete current session, start new",
                "  /rm <id...>     delete session(s) or snapshot(s)",
                "  /rm --all       delete all chat sessions",
                "  /head [n]       show first n messages",
                "  /tail [n]       show last n messages",
                "  /show <i>       show message i in full (use /tail to find ids)",
                "  /ls             list sessions and snapshots",
                "  /switch <id>    switch to another session",
                "  /save [title]   save snapshot",
                "  /load <snap>    load snapshot into new session",
                "  /stats          show last turn metrics",
            ]
        )
    )


def _wrap_for_terminal(text: str, *, indent: str = "", width: int | None = None) -> str:
    cols = shutil.get_terminal_size(fallback=(120, 24)).columns
    target_width = cols if width is None else int(width)
    target_width = max(20, target_width)

    normalized = text.replace("\r", "")
    out_lines: list[str] = []
    for logical in normalized.split("\n"):
        if not logical:
            out_lines.append(indent)
            continue
        wrapped = textwrap.wrap(
            logical,
            width=max(10, target_width - len(indent)),
            replace_whitespace=False,
            drop_whitespace=False,
            break_long_words=True,
            break_on_hyphens=False,
        )
        if not wrapped:
            out_lines.append(indent)
        else:
            out_lines.extend([indent + w for w in wrapped])
    return "\n".join(out_lines)


def _cmd_show(*, client: SuperlinearClient, session_id: str, index: int) -> None:
    """Show a single message in full by 1-based index."""
    try:
        resp = client.request_json("GET", f"/v1/sessions/{session_id}/history", timeout_s=10.0)
    except HttpError as exc:
        raise ChatReplError(str(exc)) from exc

    msgs = resp.get("messages") if isinstance(resp, dict) else None
    if not isinstance(msgs, list):
        raise ChatReplError("Invalid response from server for /show")

    n = len(msgs)
    if n == 0:
        print("(empty)")
        return
    if index < 1 or index > n:
        raise ChatReplError(f"Message index out of range: {index} (1..{n})")

    m = msgs[index - 1]
    if not isinstance(m, dict):
        raise ChatReplError("Invalid message format")

    role = str(m.get("role") or "")
    content = m.get("content")
    tool_calls = m.get("tool_calls")

    if content is None and tool_calls is not None:
        content_str = f"<tool_calls {len(tool_calls) if isinstance(tool_calls, list) else 1}>"
    else:
        content_str = "" if content is None else str(content)

    header = f"{index:>4} {role}:"
    print(header)
    if content_str:
        print(_wrap_for_terminal(content_str, indent="     "))
    else:
        print("     (empty)")


def _cmd_head(*, client: SuperlinearClient, session_id: str, limit: int = 10) -> None:
    """Show first n messages."""
    try:
        resp = client.request_json("GET", f"/v1/sessions/{session_id}/history", timeout_s=10.0)
    except HttpError as exc:
        raise ChatReplError(str(exc)) from exc

    msgs = resp.get("messages") if isinstance(resp, dict) else None
    if not isinstance(msgs, list):
        raise ChatReplError("Invalid response from server for /head")

    limit = max(1, min(int(limit), 200))
    head = msgs[:limit]
    if not head:
        print("(empty)")
        return

    for i, m in enumerate(head, start=1):
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        text = content if isinstance(content, str) else ""
        one_line = text.replace("\r", "").replace("\n", " ").strip()
        if len(one_line) > 200:
            one_line = one_line[:197] + "…"
        print(f"{i:>4} {role}: {one_line}")


def _cmd_tail(*, client: SuperlinearClient, session_id: str, limit: int = 10) -> None:
    """Show last n messages."""
    try:
        resp = client.request_json("GET", f"/v1/sessions/{session_id}/history", timeout_s=10.0)
    except HttpError as exc:
        raise ChatReplError(str(exc)) from exc

    msgs = resp.get("messages") if isinstance(resp, dict) else None
    if not isinstance(msgs, list):
        raise ChatReplError("Invalid response from server for /tail")

    limit = max(1, min(int(limit), 200))
    tail = msgs[-limit:]
    if not tail:
        print("(empty)")
        return

    for i, m in enumerate(tail, start=max(1, len(msgs) - len(tail) + 1)):
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        text = content if isinstance(content, str) else ""
        one_line = text.replace("\r", "").replace("\n", " ").strip()
        if len(one_line) > 200:
            one_line = one_line[:197] + "…"
        print(f"{i:>4} {role}: {one_line}")


def _cmd_ls(*, client: SuperlinearClient, current_session_id: str) -> None:
    """List all sessions and snapshots."""
    # Sessions
    try:
        payload = client.request_json("GET", "/v1/sessions", timeout_s=10.0)
    except HttpError as exc:
        raise ChatReplError(str(exc)) from exc

    raw_sessions = payload.get("sessions") if isinstance(payload, dict) else None
    if not isinstance(raw_sessions, list):
        raw_sessions = []
    session_ids = [s for s in raw_sessions if isinstance(s, str)]

    print("sessions:")
    if not session_ids:
        print("  (none)")
    else:
        for sid in session_ids:
            marker = " *" if sid == current_session_id else ""
            print(f"  {sid}{marker}")

    # Snapshots (local)
    snapshots = list_local_snapshots()
    print("\nsnapshots:")
    if not snapshots:
        print("  (none)")
    else:
        for snap in snapshots:
            sid = snap.get("snapshot_id") or ""
            title = snap.get("title") or ""
            if title:
                print(f"  {sid}  {title}")
            else:
                print(f"  {sid}")


def _cmd_rm(
    *,
    client: SuperlinearClient,
    target_ids: list[str],
    current_session_id: str,
) -> bool:
    """Remove session(s) and/or snapshot(s). Returns True if current session was removed."""
    removed_current = False
    for target_id in target_ids:
        # Check if it's a snapshot ID (32-char hex)
        raw_id = target_id[5:] if target_id.startswith("snap-") else target_id
        is_snapshot = len(raw_id) == 32 and all(c in "0123456789abcdef" for c in raw_id.lower())

        if is_snapshot:
            # Delete snapshot
            deleted = delete_local_snapshot(raw_id)
            if deleted:
                print(f"removed snapshot_id={raw_id}")
            else:
                print(f"error: snapshot not found: {raw_id}", file=sys.stderr)
        else:
            # Delete session
            try:
                client.request_json("DELETE", f"/v1/sessions/{target_id}", timeout_s=10.0)
                print(f"removed session_id={target_id}")
                if target_id == current_session_id:
                    removed_current = True
            except HttpError as exc:
                if exc.status_code == 404:
                    print(f"error: session not found: {target_id}", file=sys.stderr)
                else:
                    print(f"error: failed to remove {target_id}: {exc}", file=sys.stderr)
    return removed_current


def _cmd_rm_all(
    *,
    client: SuperlinearClient,
    current_session_id: str,
) -> bool:
    """Remove all chat-* sessions. Returns True if current session was removed."""
    try:
        payload = client.request_json("GET", "/v1/sessions", timeout_s=10.0)
    except HttpError as exc:
        raise ChatReplError(str(exc)) from exc

    raw_sessions = payload.get("sessions") if isinstance(payload, dict) else []
    if not isinstance(raw_sessions, list):
        raw_sessions = []
    chat_sessions = [s for s in raw_sessions if isinstance(s, str) and s.startswith("chat")]

    if not chat_sessions:
        print("(no chat sessions to remove)")
        return False

    return _cmd_rm(client=client, target_ids=chat_sessions, current_session_id=current_session_id)


def _cmd_save(
    *,
    client: SuperlinearClient,
    state: CliState,
    session_id: str,
    title: str | None,
    archive: bool,
) -> None:
    payload: dict[str, Any] = {}
    if title:
        payload["title"] = title

    resp = client.request_json("POST", f"/v1/sessions/{session_id}/save", payload=payload, timeout_s=300.0)
    snapshot_id = resp.get("snapshot_id") if isinstance(resp, dict) else None
    if not isinstance(snapshot_id, str) or not snapshot_id:
        raise ChatReplError("Invalid response from server for /save")

    if archive:
        print(f"saved (archive) snapshot_id={snapshot_id}")
        return

    prev = state.chat_checkpoints.get(session_id)
    state.chat_checkpoints[session_id] = snapshot_id
    _set_active_session(state, session_id)
    save_state(state)

    if prev and prev != snapshot_id:
        try:
            client.request_json("DELETE", f"/v1/snapshots/{prev}", timeout_s=30.0)
            print(f"saved checkpoint snapshot_id={snapshot_id} (deleted previous {prev})")
        except HttpError:
            print(f"saved checkpoint snapshot_id={snapshot_id} (failed to delete previous {prev})")
    else:
        print(f"saved checkpoint snapshot_id={snapshot_id}")


def _cmd_load(
    *,
    client: SuperlinearClient,
    state: CliState,
    snapshot_id: str,
    as_session_id: str | None,
) -> str:
    target = as_session_id or _new_session_id(prefix="chat")
    try:
        resp = client.request_json(
            "POST",
            f"/v1/snapshots/{snapshot_id}/load",
            payload={"session_id": target},
            timeout_s=300.0,
        )
    except HttpError as exc:
        if exc.status_code == 404:
            raise ChatReplError(f"Snapshot not found: {snapshot_id} (use `spl snapshot ls`).") from exc
        if exc.status_code == 409:
            raise ChatReplError(
                f"Target session already exists: {target} (choose a different --as, or omit --as)."
            ) from exc
        if exc.status_code == 429:
            raise ChatReplError("Server is busy (429). Try again.") from exc
        raise
    if not isinstance(resp, dict) or not isinstance(resp.get("session_id"), str):
        raise ChatReplError("Invalid response from server for /load")

    new_session_id = resp["session_id"]
    _set_active_session(state, new_session_id)
    # Do not automatically treat the loaded snapshot as the checkpoint.
    state.chat_checkpoints.pop(new_session_id, None)
    state.chat_checkpoint_snapshot_id = None
    save_state(state)
    print(f"loaded snapshot_id={snapshot_id} session_id={new_session_id}")
    return new_session_id


def _cmd_switch(*, client: SuperlinearClient, state: CliState, session_id: str) -> str:
    try:
        client.request_json("GET", f"/v1/sessions/{session_id}", timeout_s=10.0)
    except HttpError as exc:
        if exc.status_code == 404:
            raise ChatReplError(f"Unknown session: {session_id} (use /new to create one)") from exc
        raise ChatReplError(str(exc)) from exc

    _set_active_session(state, session_id)
    save_state(state)
    print(f"switched session_id={session_id}")
    return session_id


def _stream_chat_turn(
    *,
    client: SuperlinearClient,
    session_id: str,
    user_text: str,
    think_budget: int | None,
    temperature: float = 0.1,
    top_p: float = 0.95,
    system_prompt: str | None = None,
) -> TurnStats:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})

    payload: dict[str, Any] = {
        "stream": True,
        "session_id": session_id,
        "messages": messages,
        "max_completion_tokens": 32768,
        "temperature": temperature,
        "top_p": top_p,
    }

    enable_thinking_ui = think_budget is not None and think_budget > 0
    if enable_thinking_ui:
        # Enable thinking mode (Superlinear-specific). The model will emit <think>...</think>.
        payload["reasoning_budget"] = int(think_budget)
        # Do not persist thinking into the server-side session transcript.
        payload["discard_thinking"] = True
        # Ask the server to stream thinking deltas separately (delta.thinking).
        payload["stream_thinking"] = True

    started = time.monotonic()
    ttft_s: float | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    timing: dict[str, Any] | None = None

    started_answer = False
    in_think = False
    thinking_accum: str = ""
    thinking_panel_active = False
    thinking_panel_lines = 0
    content_buf = ""
    saw_thinking_delta = False
    thinking_start_time: float | None = None
    thinking_end_time: float | None = None

    def _thinking_panel_format(text: str) -> list[str]:
        cols = shutil.get_terminal_size(fallback=(120, 24)).columns
        prefix = "thinking: "
        width = max(20, cols - len(prefix) - 1)

        normalized = text.replace("\r", "")
        wrapped: list[str] = []
        for logical in normalized.split("\n"):
            parts = textwrap.wrap(
                logical,
                width=width,
                replace_whitespace=False,
                drop_whitespace=False,
                break_long_words=True,
                break_on_hyphens=False,
            )
            if not parts:
                wrapped.append("")
            else:
                wrapped.extend(parts)

        tail = wrapped[-10:]
        # Always show at least one line once thinking starts.
        if not tail:
            tail = [""]
        return [prefix + ln for ln in tail]

    def _thinking_panel_move_to_top() -> None:
        nonlocal thinking_panel_lines
        if thinking_panel_lines > 1:
            sys.stdout.write(f"\x1b[{thinking_panel_lines - 1}A")

    def _thinking_panel_render(text: str) -> None:
        nonlocal thinking_panel_active, thinking_panel_lines
        lines = _thinking_panel_format(text)

        if not thinking_panel_active:
            # Allocate space for the panel.
            sys.stdout.write("\n")
            thinking_panel_active = True
            thinking_panel_lines = 1

        _thinking_panel_move_to_top()

        # Clear the old panel area.
        for i in range(thinking_panel_lines):
            sys.stdout.write("\r\x1b[2K")
            if i < thinking_panel_lines - 1:
                sys.stdout.write("\n")
        _thinking_panel_move_to_top()

        # Draw the new panel.
        for i, ln in enumerate(lines):
            sys.stdout.write("\r\x1b[2K" + ln)
            if i < len(lines) - 1:
                sys.stdout.write("\n")

        thinking_panel_lines = len(lines)
        sys.stdout.flush()

    def _thinking_panel_clear() -> None:
        nonlocal thinking_panel_active, thinking_panel_lines, thinking_start_time, thinking_end_time
        if not thinking_panel_active:
            return

        _thinking_panel_move_to_top()
        for i in range(thinking_panel_lines):
            sys.stdout.write("\r\x1b[2K")
            if i < thinking_panel_lines - 1:
                sys.stdout.write("\n")
        _thinking_panel_move_to_top()

        thinking_panel_active = False
        thinking_panel_lines = 0

        # Show thinking duration summary if we have timing info
        if thinking_start_time is not None and thinking_end_time is not None:
            duration = thinking_end_time - thinking_start_time
            if duration >= 60:
                minutes = int(duration // 60)
                seconds = duration % 60
                sys.stdout.write(f"[thinking complete] duration: {minutes} minute{'s' if minutes != 1 else ''} {seconds:.1f} seconds\n")
            else:
                sys.stdout.write(f"[thinking complete] duration: {duration:.1f} seconds\n")

        sys.stdout.flush()

    def _answer_start_if_needed() -> None:
        nonlocal started_answer
        if not started_answer:
            _thinking_panel_clear()
            print("assistant: ", end="", flush=True)
            started_answer = True

    gen = client.request_sse("POST", "/v1/chat/completions", payload=payload, timeout_s=3600.0)
    try:
        for event in gen:
            if isinstance(event, dict) and "error" in event:
                err = event.get("error")
                msg = err.get("message") if isinstance(err, dict) else str(err)
                raise ChatReplError(str(msg))

            if not isinstance(event, dict):
                continue

            choices = event.get("choices")
            if isinstance(choices, list) and choices:
                ch0 = choices[0]
                if isinstance(ch0, dict):
                    delta = ch0.get("delta") if isinstance(ch0.get("delta"), dict) else {}
                    if isinstance(delta, dict):
                        thinking = delta.get("thinking")
                        if isinstance(thinking, str) and thinking:
                            saw_thinking_delta = True
                            if ttft_s is None:
                                ttft_s = time.monotonic() - started

                            buf = thinking
                            while buf:
                                if not in_think:
                                    start_idx = buf.find("<think>")
                                    if start_idx == -1:
                                        # If server doesn't send tags, ignore stray thinking outside think.
                                        break
                                    buf = buf[start_idx + len("<think>") :]
                                    in_think = True
                                    thinking_accum = ""
                                    if thinking_start_time is None:
                                        thinking_start_time = time.monotonic()
                                    _thinking_panel_render(thinking_accum)
                                    continue

                                end_idx = buf.find("</think>")
                                if end_idx == -1:
                                    thinking_accum += buf
                                    buf = ""
                                    _thinking_panel_render(thinking_accum)
                                    break

                                thinking_accum += buf[:end_idx]
                                buf = buf[end_idx + len("</think>") :]
                                if thinking_start_time is not None:
                                    thinking_end_time = time.monotonic()
                                _thinking_panel_clear()
                                in_think = False
                                break

                        content = delta.get("content")
                        if isinstance(content, str) and content:
                            if ttft_s is None:
                                ttft_s = time.monotonic() - started

                            # If the server is streaming thinking deltas separately, don't run the
                            # fallback <think>-tag parser on content.
                            if not enable_thinking_ui or saw_thinking_delta:
                                _answer_start_if_needed()
                                sys.stdout.write(content)
                                sys.stdout.flush()
                            else:
                                # Stream thinking live, then clear it once </think> arrives.
                                content_buf += content
                                while content_buf:
                                    if in_think:
                                        end_idx = content_buf.find("</think>")
                                        if end_idx == -1:
                                            thinking_accum += content_buf
                                            content_buf = ""
                                            _thinking_panel_render(thinking_accum)
                                            break

                                        thinking_accum += content_buf[:end_idx]
                                        content_buf = content_buf[end_idx + len("</think>") :]
                                        in_think = False
                                        if thinking_start_time is not None:
                                            thinking_end_time = time.monotonic()
                                        _thinking_panel_clear()
                                        continue

                                    start_idx = content_buf.find("<think>")
                                    if start_idx == -1:
                                        _answer_start_if_needed()
                                        sys.stdout.write(content_buf)
                                        sys.stdout.flush()
                                        content_buf = ""
                                        break

                                    # Emit any prelude before <think> as answer.
                                    if start_idx > 0:
                                        _answer_start_if_needed()
                                        sys.stdout.write(content_buf[:start_idx])
                                        sys.stdout.flush()

                                    content_buf = content_buf[start_idx + len("<think>") :]
                                    in_think = True
                                    thinking_accum = ""
                                    if thinking_start_time is None:
                                        thinking_start_time = time.monotonic()
                                    _thinking_panel_render(thinking_accum)
                                    continue

                        tool_calls = delta.get("tool_calls")
                        if tool_calls is not None:
                            if ttft_s is None:
                                ttft_s = time.monotonic() - started
                            _thinking_panel_clear()
                            sys.stdout.write(
                                f"\n<tool_calls {len(tool_calls) if isinstance(tool_calls, list) else 1}>\n"
                            )
                            sys.stdout.flush()

                    fr = ch0.get("finish_reason")
                    if fr is not None:
                        finish_reason = str(fr)

            # Terminal chunk may include usage/timing.
            if event.get("usage") is not None and isinstance(event.get("usage"), dict):
                usage = event["usage"]
            if event.get("x_superlinear_timing") is not None and isinstance(event.get("x_superlinear_timing"), dict):
                timing = event["x_superlinear_timing"]
    except KeyboardInterrupt:
        try:
            gen.close()
        except Exception:
            pass
        raise
    finally:
        try:
            gen.close()
        except Exception:
            pass

        # Always clear any on-screen thinking UI, even on errors.
        _thinking_panel_clear()
        if in_think and thinking_start_time is not None and thinking_end_time is None:
            sys.stdout.write("[thinking incomplete] (no </think> received before stream ended)\n")
        sys.stdout.flush()

    ended = time.monotonic()
    sys.stdout.write("\n")
    sys.stdout.flush()

    stats = TurnStats()
    stats.finish_reason = finish_reason
    stats.ttft_s = ttft_s
    stats.total_s = max(ended - started, 0.0)

    if usage is not None:
        pt = usage.get("prompt_tokens")
        ct = usage.get("completion_tokens")
        if isinstance(pt, int):
            stats.prompt_tokens = pt
        if isinstance(ct, int):
            stats.completion_tokens = ct

    if timing is not None:
        prefill_s = timing.get("prefill_s")
        decode_s = timing.get("decode_s")
        total_s = timing.get("total_s")
        tok_per_s = timing.get("tok_per_s")
        if isinstance(prefill_s, (float, int)):
            stats.server_prefill_s = float(prefill_s)
        if isinstance(decode_s, (float, int)):
            stats.server_decode_s = float(decode_s)
        if isinstance(total_s, (float, int)):
            stats.server_total_s = float(total_s)
        if isinstance(tok_per_s, (float, int)):
            stats.tok_per_s = float(tok_per_s)

    return stats


def chat_repl(
    *,
    url: str,
    new: bool = False,
    session: str | None = None,
    max_seq_len: int | None = 1_048_576,
    think_budget: int | None = 8192,
    temperature: float = 0.1,
    top_p: float = 0.95,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
) -> int:
    _setup_readline_history()
    _setup_completer()
    client = SuperlinearClient(base_url=url, timeout_s=3600.0)
    try:
        _ensure_reachable(client)
    except ChatReplError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    state = load_state()

    resumed = False
    if new:
        session_id = _new_session_id(prefix="chat")
        lock = SessionLock(session_id=session_id, kind="chat", label="spl chat")
    elif session:
        session_id = session
        lock = SessionLock(session_id=session_id, kind="chat", label="spl chat")
    else:
        session_id = state.active_chat_session_id
        if not session_id:
            session_id = _new_session_id(prefix="chat")
        lock = SessionLock(session_id=session_id, kind="chat", label="spl chat")

    try:
        lock.acquire()
    except AlreadyLockedError as exc:
        pid = exc.info.pid
        label = exc.info.label or exc.info.kind or "spl"
        print(
            f"error: session is already open in another REPL (session_id={session_id} pid={pid} label={label}).",
            file=sys.stderr,
        )
        if not new:
            print("next steps: `spl chat --new` or choose a different `--session`.", file=sys.stderr)
        return 2

    try:
        if new:
            _create_session(client, session_id, max_seq_len=max_seq_len)
            _maybe_resize_session(client, session_id, min_max_seq_len=max_seq_len)
            _set_active_session(state, session_id)
            state.chat_checkpoints.pop(session_id, None)
            state.chat_checkpoint_snapshot_id = None
            save_state(state)
        elif session:
            if _session_exists(client, session_id):
                resumed = True
                _maybe_resize_session(client, session_id, min_max_seq_len=max_seq_len)
            else:
                _create_session(client, session_id, max_seq_len=max_seq_len)
                _maybe_resize_session(client, session_id, min_max_seq_len=max_seq_len)
            _set_active_session(state, session_id)
            save_state(state)
        else:
            if session_id and _session_exists(client, session_id):
                resumed = True
                _maybe_resize_session(client, session_id, min_max_seq_len=max_seq_len)
            else:
                if session_id:
                    print(f"note: session not found on server: {session_id} (starting a new one)")
                # Release lock for the missing session id, and start a new chat session.
                lock.release()
                session_id = _new_session_id(prefix="chat")
                lock = SessionLock(session_id=session_id, kind="chat", label="spl chat")
                lock.acquire()
                _create_session(client, session_id, max_seq_len=max_seq_len)
                _maybe_resize_session(client, session_id, min_max_seq_len=max_seq_len)
                _set_active_session(state, session_id)
                state.chat_checkpoints.pop(session_id, None)
                state.chat_checkpoint_snapshot_id = None
                save_state(state)

        _banner(url=client.base_url, session_id=session_id, resumed=resumed)

        # Only send the system prompt when the session is truly empty.
        # Sending a system prompt on an already-prefilled KV cache would be a prefix edit and can
        # corrupt session append-from behavior.
        should_send_system_prompt = False
        if system_prompt:
            try:
                msg_count, cache_pos = _get_session_counters(client=client, session_id=session_id)
                should_send_system_prompt = (msg_count <= 0 and cache_pos <= 0)
            except ChatReplError:
                # If we can't verify, default to not sending to avoid KV corruption.
                should_send_system_prompt = False

        last_stats: TurnStats | None = None

        while True:
            prompt = f"spl(chat:{session_id})> "
            try:
                raw = input(prompt)
            except EOFError:
                print()
                return 0
            except KeyboardInterrupt:
                print("^C")
                continue

            raw = _coalesce_pasted_lines(raw)

            line = raw.strip()
            if not line:
                continue

            # Guard against accidental paste leftovers from transcripts.
            # These frequently show up as a single role label on its own line
            # (e.g. the user copied/pasted a block that included "assistant:").
            non_empty_lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            if len(non_empty_lines) == 1:
                lone = non_empty_lines[0]
                # Normalize aggressively: drop punctuation, spaces, and control chars.
                alpha = "".join(ch for ch in lone.lower() if "a" <= ch <= "z")
                if alpha in {"assistant", "user", "system"}:
                    continue

            # Commands are single-line only. If the user pasted a block that starts
            # with '/', treat it as a normal message.
            if line.startswith("/") and "\n" not in raw:
                cmdline = line[1:].strip()
                try:
                    parts = shlex.split(cmdline)
                except ValueError as exc:
                    print(f"parse error: {exc}", file=sys.stderr)
                    continue
                if not parts:
                    continue
                cmd, args = parts[0], parts[1:]

                if cmd in {"exit", "quit"}:
                    if args in [["--clean"], ["-c"]]:
                        try:
                            client.request_json("DELETE", f"/v1/sessions/{session_id}", timeout_s=10.0)
                            print(f"removed session_id={session_id}")
                        except HttpError:
                            pass  # Best effort
                    return 0
                if cmd == "help":
                    _cmd_help()
                    continue
                if cmd == "clear":
                    print("\033[2J\033[H", end="", flush=True)
                    continue
                if cmd == "head":
                    n = 10
                    if len(args) == 1:
                        try:
                            n = int(args[0])
                        except Exception:
                            print("usage: /head [n]", file=sys.stderr)
                            continue
                    elif len(args) > 1:
                        print("usage: /head [n]", file=sys.stderr)
                        continue
                    try:
                        _cmd_head(client=client, session_id=session_id, limit=n)
                    except ChatReplError as exc:
                        print(str(exc), file=sys.stderr)
                    continue
                if cmd == "tail":
                    n = 10
                    if len(args) == 1:
                        try:
                            n = int(args[0])
                        except Exception:
                            print("usage: /tail [n]", file=sys.stderr)
                            continue
                    elif len(args) > 1:
                        print("usage: /tail [n]", file=sys.stderr)
                        continue
                    try:
                        _cmd_tail(client=client, session_id=session_id, limit=n)
                    except ChatReplError as exc:
                        print(str(exc), file=sys.stderr)
                    continue
                if cmd == "show":
                    if len(args) != 1:
                        print("usage: /show <i>", file=sys.stderr)
                        continue
                    try:
                        i = int(args[0])
                    except Exception:
                        print("usage: /show <i>", file=sys.stderr)
                        continue
                    try:
                        _cmd_show(client=client, session_id=session_id, index=i)
                    except ChatReplError as exc:
                        print(str(exc), file=sys.stderr)
                    continue
                if cmd == "history":
                    if args in [["clear"], ["--clear"], ["-c"]]:
                        _cmd_history_clear()
                        continue
                    n = 20
                    if len(args) == 1:
                        try:
                            n = int(args[0])
                        except Exception:
                            print("usage: /history [n] | /history clear", file=sys.stderr)
                            continue
                    elif len(args) > 1:
                        print("usage: /history [n] | /history clear", file=sys.stderr)
                        continue
                    _cmd_history(n)
                    continue
                if cmd == "ls":
                    try:
                        _cmd_ls(client=client, current_session_id=session_id)
                    except ChatReplError as exc:
                        print(str(exc), file=sys.stderr)
                    continue
                if cmd == "rm":
                    try:
                        if not args:
                            # /rm with no args = delete current session
                            removed_current = _cmd_rm(
                                client=client,
                                target_ids=[session_id],
                                current_session_id=session_id,
                            )
                        elif args == ["--all"]:
                            removed_current = _cmd_rm_all(
                                client=client,
                                current_session_id=session_id,
                            )
                        else:
                            removed_current = _cmd_rm(
                                client=client,
                                target_ids=args,
                                current_session_id=session_id,
                            )
                    except ChatReplError as exc:
                        print(str(exc), file=sys.stderr)
                        continue
                    if removed_current:
                        # Current session removed; create a new one
                        next_id = _new_session_id(prefix="chat")
                        next_lock = SessionLock(session_id=next_id, kind="chat", label="spl chat")
                        try:
                            next_lock.acquire()
                        except AlreadyLockedError:
                            print(f"error: session is already open: {next_id}", file=sys.stderr)
                            continue
                        try:
                            _create_session(client, next_id, max_seq_len=max_seq_len)
                            _maybe_resize_session(client, next_id, min_max_seq_len=max_seq_len)
                        except Exception as exc:
                            next_lock.release()
                            print(str(exc), file=sys.stderr)
                            continue
                        lock.release()
                        lock = next_lock
                        session_id = next_id
                        _set_active_session(state, session_id)
                        state.chat_checkpoints.pop(session_id, None)
                        state.chat_checkpoint_snapshot_id = None
                        save_state(state)
                        last_stats = None
                        _banner(url=client.base_url, session_id=session_id, resumed=False)
                        should_send_system_prompt = bool(system_prompt)
                    continue
                if cmd == "stats":
                    if last_stats is None:
                        print("no stats yet")
                    else:
                        print(_stats_detail(last_stats))
                    continue
                if cmd == "new":
                    clean = args == ["--clean"]
                    if args and not clean:
                        print("usage: /new [--clean]", file=sys.stderr)
                        continue
                    # If --clean, delete current session first
                    if clean:
                        try:
                            client.request_json("DELETE", f"/v1/sessions/{session_id}", timeout_s=10.0)
                            print(f"removed session_id={session_id}")
                        except HttpError as exc:
                            if exc.status_code != 404:
                                print(f"warning: failed to delete current session: {exc}", file=sys.stderr)
                    next_id = _new_session_id(prefix="chat")
                    next_lock = SessionLock(session_id=next_id, kind="chat", label="spl chat")
                    try:
                        next_lock.acquire()
                    except AlreadyLockedError:
                        print(f"error: session is already open: {next_id}", file=sys.stderr)
                        continue
                    try:
                        _create_session(client, next_id, max_seq_len=max_seq_len)
                        _maybe_resize_session(client, next_id, min_max_seq_len=max_seq_len)
                    except Exception as exc:
                        next_lock.release()
                        print(str(exc), file=sys.stderr)
                        continue
                    lock.release()
                    lock = next_lock
                    session_id = next_id
                    _set_active_session(state, session_id)
                    state.chat_checkpoints.pop(session_id, None)
                    state.chat_checkpoint_snapshot_id = None
                    save_state(state)
                    last_stats = None
                    _banner(url=client.base_url, session_id=session_id, resumed=False)
                    should_send_system_prompt = bool(system_prompt)
                    continue
                if cmd == "switch":
                    if len(args) != 1:
                        print("usage: /switch <session_id>", file=sys.stderr)
                        continue
                    target_id = args[0]
                    next_lock = SessionLock(session_id=target_id, kind="chat", label="spl chat")
                    try:
                        next_lock.acquire()
                    except AlreadyLockedError as exc:
                        print(
                            f"error: session is already open in another REPL (session_id={target_id} pid={exc.info.pid}).",
                            file=sys.stderr,
                        )
                        continue
                    try:
                        target_id = _cmd_switch(client=client, state=state, session_id=target_id)
                    except ChatReplError as exc:
                        next_lock.release()
                        print(str(exc), file=sys.stderr)
                        continue
                    lock.release()
                    lock = next_lock
                    session_id = target_id
                    last_stats = None
                    if system_prompt:
                        try:
                            msg_count, cache_pos = _get_session_counters(client=client, session_id=session_id)
                            should_send_system_prompt = (msg_count <= 0 and cache_pos <= 0)
                        except ChatReplError:
                            should_send_system_prompt = False
                    else:
                        should_send_system_prompt = False
                    continue
                if cmd == "load":
                    if len(args) != 1:
                        print("usage: /load <snapshot_id>", file=sys.stderr)
                        continue
                    snap = args[0]

                    target_id = _new_session_id(prefix="chat")
                    next_lock = SessionLock(session_id=target_id, kind="chat", label="spl chat")
                    try:
                        next_lock.acquire()
                    except AlreadyLockedError as exc:
                        print(
                            f"error: session is already open in another REPL (session_id={target_id} pid={exc.info.pid}).",
                            file=sys.stderr,
                        )
                        continue

                    try:
                        loaded_id = _cmd_load(
                            client=client,
                            state=state,
                            snapshot_id=snap,
                            as_session_id=target_id,
                        )
                    except (ChatReplError, HttpError) as exc:
                        next_lock.release()
                        print(str(exc), file=sys.stderr)
                        continue

                    lock.release()
                    lock = next_lock
                    session_id = loaded_id
                    last_stats = None
                    _banner(url=client.base_url, session_id=session_id, resumed=False)
                    continue
                if cmd == "save":
                    title: str | None = None
                    if args:
                        title = " ".join(args).strip() or None
                    try:
                        _cmd_save(
                            client=client,
                            state=state,
                            session_id=session_id,
                            title=title,
                            archive=True,  # Always keep snapshots (no auto-delete)
                        )
                    except (ChatReplError, HttpError) as exc:
                        print(str(exc), file=sys.stderr)
                    continue

                print(f"unknown command: /{cmd}", file=sys.stderr)
                continue

            # Regular user message.
            try:
                stats = _stream_chat_turn(
                    client=client,
                    session_id=session_id,
                    user_text=line,
                    think_budget=think_budget,
                    temperature=temperature,
                    top_p=top_p,
                    system_prompt=system_prompt if should_send_system_prompt else None,
                )
                should_send_system_prompt = False
                last_stats = stats
                footer = _format_metrics(stats)
                if footer:
                    print(footer)
            except KeyboardInterrupt:
                print("\n(cancelled)")
                continue
            except ChatReplError as exc:
                print(str(exc), file=sys.stderr)
                continue
            except HttpError as exc:
                if exc.status_code == 429:
                    print("Server is busy (429). Try again.", file=sys.stderr)
                else:
                    print(str(exc), file=sys.stderr)
                continue
    except (ChatReplError, HttpError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            lock.release()
        except Exception:
            pass


def _coalesce_pasted_lines(first_line: str) -> str:
    """Best-effort: if the user pasted multiple lines, treat them as one message.

    Python's built-in `input()` reads only one line. When a user pastes a block
    containing newlines, the remaining lines are immediately available on stdin
    and would otherwise be consumed as separate turns.
    """

    if not sys.stdin.isatty():
        return first_line

    def _strip_paste_markers(text: str) -> str:
        # Some terminals use bracketed paste mode. If those markers leak into
        # stdin, remove them.
        return text.replace("\x1b[200~", "").replace("\x1b[201~", "")

    lines = [_strip_paste_markers(first_line.rstrip("\r\n"))]

    # Hard limits avoid accidental runaway reads.
    max_lines = 128
    max_chars = 64_000
    total_chars = len(lines[0])

    # Fast path: if nothing else is immediately queued, don't add latency.
    try:
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
    except Exception:
        return lines[0]
    if not ready:
        return lines[0]

    # Pasted blocks can arrive over a few scheduling ticks. Once we detect that
    # stdin is readable, keep draining lines until a short quiet period elapses
    # (or we hit limits).
    deadline = time.monotonic() + 0.10
    quiet_s = 0.02

    while len(lines) < max_lines and total_chars < max_chars:
        timeout = max(0.0, min(quiet_s, deadline - time.monotonic()))
        if timeout <= 0.0:
            break
        try:
            ready, _, _ = select.select([sys.stdin], [], [], timeout)
        except Exception:
            break
        if not ready:
            break

        extra = sys.stdin.readline()
        if not extra:
            break
        extra = _strip_paste_markers(extra.rstrip("\r\n"))
        lines.append(extra)
        total_chars += len(extra) + 1

    return "\n".join(lines)
