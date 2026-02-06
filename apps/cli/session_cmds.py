from __future__ import annotations

import os
import sys
import time
from typing import Any
from urllib.parse import urlparse

from apps.cli.client import HttpError, SuperlinearClient
from apps.cli.local_snapshots import list_local_snapshots
from apps.cli.locks import read_active_lock
from apps.cli.output import format_table, print_json


class SessionCommandError(RuntimeError):
    pass


def _is_local_url(url: str) -> bool:
    """Check if URL points to localhost."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        return host in {"localhost", "127.0.0.1", "::1", ""} or host.startswith("127.")
    except Exception:
        return True  # Assume local if can't parse


def _format_unreachable(exc: HttpError, *, base_url: str) -> str | None:
    if exc.status_code is None and exc.message in {"Failed to reach server", "Request timed out"}:
        return (
            f"Server unreachable at {base_url}. Start it with `spl server start --model <model>` or pass `--url`.\n{exc}"
        )
    return None


def _fmt_unix_utc(ts: int | None) -> str:
    if not ts:
        return ""
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(ts)))
    except Exception:
        return str(ts)


def _coerce_session_info(obj: Any, *, session_id: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        return {"session_id": session_id, "error": "invalid_response"}
    out = dict(obj)
    if "session_id" not in out:
        out["session_id"] = session_id
    return out


def session_ls(*, url: str, json_output: bool = False) -> int:
    client = SuperlinearClient(base_url=url, timeout_s=10.0)
    try:
        payload = client.request_json("GET", "/v1/sessions", timeout_s=10.0)
    except HttpError as exc:
        msg = _format_unreachable(exc, base_url=url)
        if msg:
            raise SessionCommandError(msg) from exc
        if exc.status_code == 429:
            raise SessionCommandError("Server is busy (429). Try again.") from exc
        raise SessionCommandError(str(exc)) from exc

    raw_sessions = payload.get("sessions") if isinstance(payload, dict) else None
    if not isinstance(raw_sessions, list):
        raw_sessions = []
    session_ids = [s for s in raw_sessions if isinstance(s, str)]

    infos: list[dict[str, Any]] = []
    for sid in session_ids:
        try:
            info = client.request_json("GET", f"/v1/sessions/{sid}", timeout_s=10.0)
            infos.append(_coerce_session_info(info, session_id=sid))
        except HttpError as exc:
            infos.append({"session_id": sid, "error": str(exc)})

    if json_output:
        print_json({"sessions": infos})
        return 0

    rows: list[list[str]] = []
    for info in infos:
        msg_count = info.get("message_count")
        pos = info.get("current_pos")
        max_seq_len = info.get("max_seq_len")
        rows.append(
            [
                str(info.get("cache_id") or info.get("session_id") or ""),
                "" if msg_count is None else str(msg_count),
                "" if pos is None else str(pos),
                "" if max_seq_len is None else str(max_seq_len),
                "yes" if info.get("has_prefill") else "no",
            ]
        )
    print(format_table(["session_id", "messages", "pos", "max_seq_len", "prefill"], rows))
    return 0


def session_info(*, url: str, session_id: str, json_output: bool = False) -> int:
    client = SuperlinearClient(base_url=url, timeout_s=10.0)
    try:
        info = client.request_json("GET", f"/v1/sessions/{session_id}", timeout_s=10.0)
    except HttpError as exc:
        msg = _format_unreachable(exc, base_url=url)
        if msg:
            raise SessionCommandError(msg) from exc
        if exc.status_code == 404:
            raise SessionCommandError(f"Session not found: {session_id} (use `spl session ls`).") from exc
        if exc.status_code == 429:
            raise SessionCommandError("Server is busy (429). Try again.") from exc
        raise SessionCommandError(str(exc)) from exc

    if json_output:
        print_json(info)
        return 0

    if not isinstance(info, dict):
        raise SessionCommandError("Invalid response from server")

    # Keep this output stable and greppable.
    print(f"session_id={info.get('cache_id') or session_id}")
    print(f"message_count={info.get('message_count')}")
    print(f"current_pos={info.get('current_pos')}")
    print(f"max_seq_len={info.get('max_seq_len')}")
    print(f"has_prefill={bool(info.get('has_prefill'))}")
    created_at = info.get("created_at")
    if created_at is not None:
        print(f"created_at={_fmt_unix_utc(int(created_at))}")
    return 0


def session_close(*, url: str, session_ids: list[str]) -> int:
    """Close one or more sessions."""
    client = SuperlinearClient(base_url=url, timeout_s=10.0)
    errors = 0

    for session_id in session_ids:
        # If session looks open in another REPL, warn but proceed (MVP smoothness)
        info = read_active_lock(session_id)
        if info is not None and info.pid != os.getpid():
            label = info.label or info.kind or "spl"
            print(
                f"warning: session appears open in another REPL (session_id={session_id} pid={info.pid} label={label}); closing anyway.",
                file=sys.stderr,
            )

        try:
            resp = client.request_json("DELETE", f"/v1/sessions/{session_id}", timeout_s=10.0)
            if isinstance(resp, dict) and isinstance(resp.get("session_id"), str):
                print(f"closed session_id={resp['session_id']}")
            else:
                print(f"closed session_id={session_id}")
        except HttpError as exc:
            if exc.status_code == 404:
                print(f"error: session not found: {session_id}", file=sys.stderr)
            elif exc.status_code is None:
                msg = _format_unreachable(exc, base_url=url)
                raise SessionCommandError(msg or str(exc)) from exc
            else:
                print(f"error: failed to close {session_id}: {exc}", file=sys.stderr)
            errors += 1

    return 1 if errors > 0 else 0


def session_close_all(*, url: str) -> int:
    """Close all sessions."""
    client = SuperlinearClient(base_url=url, timeout_s=10.0)
    try:
        payload = client.request_json("GET", "/v1/sessions", timeout_s=10.0)
    except HttpError as exc:
        msg = _format_unreachable(exc, base_url=url)
        if msg:
            raise SessionCommandError(msg) from exc
        raise SessionCommandError(str(exc)) from exc

    raw_sessions = payload.get("sessions") if isinstance(payload, dict) else []
    if not isinstance(raw_sessions, list):
        raw_sessions = []
    session_ids = [s for s in raw_sessions if isinstance(s, str)]

    if not session_ids:
        print("(no sessions to remove)")
        return 0

    return session_close(url=url, session_ids=session_ids)


def session_history(
    *,
    url: str,
    session_id: str,
    tail: int | None = None,
    json_output: bool = False,
) -> int:
    client = SuperlinearClient(base_url=url, timeout_s=10.0)
    try:
        resp = client.request_json("GET", f"/v1/sessions/{session_id}/history", timeout_s=10.0)
    except HttpError as exc:
        msg = _format_unreachable(exc, base_url=url)
        if msg:
            raise SessionCommandError(msg) from exc
        if exc.status_code == 404:
            raise SessionCommandError(f"Session not found: {session_id} (use `spl session ls`).") from exc
        if exc.status_code == 429:
            raise SessionCommandError("Server is busy (429). Try again.") from exc
        raise SessionCommandError(str(exc)) from exc

    if json_output:
        print_json(resp)
        return 0

    if not isinstance(resp, dict) or not isinstance(resp.get("messages"), list):
        raise SessionCommandError("Invalid response from server")

    msgs = [m for m in resp["messages"] if isinstance(m, dict)]
    if tail is not None:
        try:
            tail_n = int(tail)
        except Exception:
            raise SessionCommandError("--tail must be an integer")
        if tail_n > 0:
            msgs = msgs[-tail_n:]

    for m in msgs:
        role = str(m.get("role") or "")
        content = m.get("content")
        tool_calls = m.get("tool_calls")
        if content is None and tool_calls is not None:
            content_str = f"<tool_calls {len(tool_calls) if isinstance(tool_calls, list) else 1}>"
        else:
            content_str = "" if content is None else str(content)
        sys.stdout.write(f"{role}: {content_str}\n")
    return 0


def unified_ls(
    *,
    url: str,
    json_output: bool = False,
    active_chat_session_id: str | None = None,
    docs_workspaces: dict[str, Any] | None = None,
) -> int:
    """List sessions and snapshots in a unified view."""
    client = SuperlinearClient(base_url=url, timeout_s=10.0)

    # Gather active session IDs from state
    active_sessions: dict[str, str] = {}  # session_id -> label
    if active_chat_session_id:
        active_sessions[active_chat_session_id] = "chat"
    if docs_workspaces:
        for name, ws in docs_workspaces.items():
            if isinstance(ws, dict):
                sid = ws.get("session_id")
            else:
                sid = getattr(ws, "session_id", None)
            if sid:
                active_sessions[sid] = f"docs:{name}"

    # Fetch sessions
    sessions_data: list[dict[str, Any]] = []
    server_reachable = True
    try:
        payload = client.request_json("GET", "/v1/sessions", timeout_s=10.0)
        raw_sessions = payload.get("sessions") if isinstance(payload, dict) else []
        if isinstance(raw_sessions, list):
            for sid in raw_sessions:
                if isinstance(sid, str):
                    try:
                        info = client.request_json("GET", f"/v1/sessions/{sid}", timeout_s=10.0)
                        if isinstance(info, dict):
                            info["session_id"] = sid
                            sessions_data.append(info)
                    except HttpError:
                        sessions_data.append({"session_id": sid, "error": True})
    except HttpError as exc:
        if exc.status_code is None and exc.message in {"Failed to reach server", "Request timed out"}:
            server_reachable = False
        else:
            raise SessionCommandError(str(exc)) from exc

    # Always get local snapshots
    local_snaps = list_local_snapshots()
    local_by_id: dict[str, dict[str, Any]] = {}
    for snap in local_snaps:
        sid = snap.get("snapshot_id")
        if sid:
            local_by_id[sid] = {
                "snapshot_id": sid,
                "created_at": snap.get("created_at"),
                "metadata": {"title": snap.get("title")},
                "session": {"current_pos": snap.get("pos")},
                "model_id": snap.get("model_id"),
                "source": "local",
            }

    # If connected to remote server, also fetch remote snapshots
    is_remote = not _is_local_url(url)
    merged = dict(local_by_id)

    if is_remote and server_reachable:
        try:
            resp = client.request_json("GET", "/v1/snapshots", timeout_s=10.0)
            raw = resp.get("snapshots") if isinstance(resp, dict) else []
            if isinstance(raw, list):
                for s in raw:
                    if isinstance(s, dict):
                        sid = s.get("snapshot_id")
                        if sid:
                            metadata = s.get("metadata") or {}
                            session = s.get("session") or {}
                            if sid in merged:
                                merged[sid]["source"] = "both"
                            else:
                                merged[sid] = {
                                    "snapshot_id": sid,
                                    "created_at": s.get("created_at"),
                                    "metadata": {"title": metadata.get("title") if isinstance(metadata, dict) else None},
                                    "session": {"current_pos": session.get("current_pos") if isinstance(session, dict) else None},
                                    "model_id": s.get("model_id") or "",
                                    "source": "remote",
                                }
        except HttpError:
            pass  # Remote snapshots are optional

    snapshots_data = sorted(merged.values(), key=lambda x: int(x.get("created_at") or 0), reverse=True)

    if json_output:
        print_json({
            "sessions": sessions_data,
            "snapshots": snapshots_data,
            "active": active_sessions,
        })
        return 0

    # Print sessions
    if server_reachable:
        print("=== Sessions ===")
        if not sessions_data:
            print("  (none)")
        else:
            rows: list[list[str]] = []
            for info in sessions_data:
                sid = str(info.get("cache_id") or info.get("session_id") or "")
                msg_count = info.get("message_count")
                pos = info.get("current_pos")
                max_seq_len = info.get("max_seq_len")
                label = active_sessions.get(sid, "")
                rows.append([
                    sid,
                    "" if msg_count is None else str(msg_count),
                    "" if pos is None else str(pos),
                    "" if max_seq_len is None else str(max_seq_len),
                    label,
                ])
            print(format_table(["session_id", "messages", "pos", "max_seq_len", "active"], rows))
        print()
    else:
        print("Server not running\n")

    print("=== Snapshots ===")
    if not snapshots_data:
        print("  (none)")
    else:
        rows = []
        for m in snapshots_data:
            sid = str(m.get("snapshot_id") or "")
            created_at = m.get("created_at")
            title = ""
            metadata = m.get("metadata")
            if isinstance(metadata, dict):
                title = str(metadata.get("title") or "")
            session = m.get("session")
            pos = ""
            if isinstance(session, dict) and session.get("current_pos") is not None:
                pos = str(session.get("current_pos"))
            rows.append([
                sid,
                _fmt_unix_utc(int(created_at) if created_at is not None else None),
                title,
                pos,
            ])
        print(format_table(["snapshot_id", "created_at", "title", "pos"], rows))

    return 0


def unified_rm(
    *,
    url: str,
    ids: list[str],
    allow_remote_snapshot_delete: bool = False,
) -> int:
    """Remove sessions and snapshots by ID.

    Detects type from ID prefix:
    - chat-* or docs-* -> session
    - snap-* -> snapshot

    Remote snapshot deletion is only attempted when `allow_remote_snapshot_delete=True`
    to avoid surprising deletes when a remote URL is loaded from saved state.
    """
    from apps.cli.local_snapshots import delete_local_snapshot

    if not ids:
        print("error: no IDs provided", file=sys.stderr)
        return 2

    # Categorize IDs by type
    session_ids: list[str] = []
    snapshot_ids: list[str] = []
    unknown_ids: list[str] = []

    # Helper to detect raw snapshot IDs (32-char hex without snap- prefix)
    def _is_raw_snapshot_id(s: str) -> bool:
        return len(s) == 32 and all(c in "0123456789abcdef" for c in s.lower())

    for id_ in ids:
        if id_.startswith(("chat-", "chat_", "docs-", "docs_")):
            session_ids.append(id_)
        elif id_.startswith("snap-"):
            snapshot_ids.append(id_[5:])  # strip snap- prefix for API
        elif _is_raw_snapshot_id(id_):
            snapshot_ids.append(id_)  # raw 32-char hex snapshot ID
        else:
            unknown_ids.append(id_)

    if unknown_ids:
        print(f"error: cannot determine type for IDs (expected chat_*, docs_*, snap-*, or 32-char snapshot ID): {', '.join(unknown_ids)}", file=sys.stderr)
        return 2

    errors = 0

    # Delete sessions
    if session_ids:
        client = SuperlinearClient(base_url=url, timeout_s=10.0)
        for session_id in session_ids:
            # If session looks open in another REPL, warn but proceed (MVP smoothness)
            info = read_active_lock(session_id)
            if info is not None and info.pid != os.getpid():
                label = info.label or info.kind or "spl"
                print(
                    f"warning: session appears open in another REPL (session_id={session_id} pid={info.pid} label={label}); closing anyway.",
                    file=sys.stderr,
                )

            try:
                client.request_json("DELETE", f"/v1/sessions/{session_id}", timeout_s=10.0)
                print(f"closed session_id={session_id}")
            except HttpError as exc:
                if exc.status_code == 404:
                    print(f"error: session not found: {session_id}", file=sys.stderr)
                elif exc.status_code is None:
                    print(f"error: server unreachable", file=sys.stderr)
                    errors += 1
                    break  # Stop if server is down
                else:
                    print(f"error: failed to close {session_id}: {exc}", file=sys.stderr)
                errors += 1

    # Delete snapshots
    is_remote = bool(allow_remote_snapshot_delete and (not _is_local_url(url)))
    remote_client = SuperlinearClient(base_url=url, timeout_s=30.0) if is_remote else None

    for snapshot_id in snapshot_ids:
        deleted_local = delete_local_snapshot(snapshot_id)
        deleted_remote = False

        if remote_client:
            try:
                remote_client.request_json("DELETE", f"/v1/snapshots/{snapshot_id}", timeout_s=30.0)
                deleted_remote = True
            except HttpError:
                pass  # Ignore remote errors

        if not deleted_local and not deleted_remote:
            print(f"error: snapshot not found: {snapshot_id}", file=sys.stderr)
            errors += 1
            continue

        if deleted_local and deleted_remote:
            print(f"deleted snapshot_id={snapshot_id} (local + remote)")
        elif deleted_remote:
            print(f"deleted snapshot_id={snapshot_id} (remote)")
        else:
            print(f"deleted snapshot_id={snapshot_id}")

    return 1 if errors > 0 else 0
