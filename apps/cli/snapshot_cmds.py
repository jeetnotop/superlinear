from __future__ import annotations

import sys
import time
from typing import Any
from urllib.parse import urlparse

from apps.cli.client import HttpError, SuperlinearClient
from apps.cli.local_snapshots import delete_local_snapshot, list_local_snapshots
from apps.cli.output import format_table, print_json


class SnapshotCommandError(RuntimeError):
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


def _normalize_snapshot(m: dict[str, Any], *, source: str = "") -> dict[str, Any]:
    """Normalize snapshot dict to common format."""
    metadata = m.get("metadata") or {}
    session = m.get("session") or {}
    return {
        "snapshot_id": m.get("snapshot_id") or "",
        "created_at": m.get("created_at"),
        "title": metadata.get("title") if isinstance(metadata, dict) else m.get("title"),
        "pos": session.get("current_pos") if isinstance(session, dict) else m.get("pos"),
        "max_seq_len": session.get("max_seq_len") if isinstance(session, dict) else m.get("max_seq_len"),
        "model_id": m.get("model_id") or "",
        "source": source,
    }


def _print_snapshots(snaps: list[dict[str, Any]], *, json_output: bool, show_source: bool = False) -> int:
    if json_output:
        print_json({"snapshots": snaps})
        return 0

    if not snaps:
        print("No snapshots found")
        return 0

    rows: list[list[str]] = []
    for m in snaps:
        row = [
            str(m.get("snapshot_id") or ""),
            _fmt_unix_utc(int(m["created_at"]) if m.get("created_at") is not None else None),
            str(m.get("title") or ""),
            str(m.get("pos") or ""),
            str(m.get("model_id") or ""),
        ]
        if show_source:
            row.append(str(m.get("source") or ""))
        rows.append(row)

    headers = ["snapshot_id", "created_at", "title", "pos", "model_id"]
    if show_source:
        headers.append("source")
    print(format_table(headers, rows))
    return 0


def snapshot_ls(*, url: str | None = None, json_output: bool = False) -> int:
    """List snapshots. Shows local + remote (deduplicated) if connected to remote server."""
    # Always get local snapshots
    local_snaps = list_local_snapshots()
    local_by_id = {s["snapshot_id"]: _normalize_snapshot(s, source="local") for s in local_snaps}

    # If we have a remote server URL, also fetch remote snapshots
    remote_snaps: list[dict[str, Any]] = []
    is_remote = url and not _is_local_url(url)

    if is_remote:
        client = SuperlinearClient(base_url=url, timeout_s=10.0)
        try:
            resp = client.request_json("GET", "/v1/snapshots", timeout_s=10.0)
            raw = resp.get("snapshots") if isinstance(resp, dict) else []
            if isinstance(raw, list):
                remote_snaps = [s for s in raw if isinstance(s, dict)]
        except HttpError:
            pass  # Remote not reachable, just show local

    # Merge: remote snapshots override local (they have the same ID means same snapshot)
    merged: dict[str, dict[str, Any]] = dict(local_by_id)
    for s in remote_snaps:
        sid = s.get("snapshot_id")
        if sid:
            if sid in merged:
                # Present in both - mark as "both"
                merged[sid] = _normalize_snapshot(s, source="both")
            else:
                # Only on remote
                merged[sid] = _normalize_snapshot(s, source="remote")

    # Sort by created_at descending
    result = sorted(merged.values(), key=lambda x: int(x.get("created_at") or 0), reverse=True)

    return _print_snapshots(result, json_output=json_output, show_source=is_remote)


def snapshot_save(*, url: str, session_id: str, title: str | None = None, json_output: bool = False) -> int:
    client = SuperlinearClient(base_url=url, timeout_s=300.0)
    payload: dict[str, Any] = {}
    if title:
        payload["title"] = title
    try:
        resp = client.request_json(
            "POST",
            f"/v1/sessions/{session_id}/save",
            payload=payload,
            timeout_s=300.0,
        )
    except HttpError as exc:
        msg = _format_unreachable(exc, base_url=url)
        if msg:
            raise SnapshotCommandError(msg) from exc
        if exc.status_code == 404:
            raise SnapshotCommandError(f"Session not found: {session_id} (use `spl session ls`).") from exc
        if exc.status_code == 429:
            raise SnapshotCommandError("Server is busy (429). Try again.") from exc
        raise SnapshotCommandError(str(exc)) from exc

    if json_output:
        print_json(resp)
        return 0

    snapshot_id = resp.get("snapshot_id") if isinstance(resp, dict) else None
    if isinstance(snapshot_id, str) and snapshot_id:
        print(f"saved snapshot_id={snapshot_id} session_id={session_id}")
        return 0
    print(f"saved session_id={session_id}")
    return 0


def snapshot_load(
    *,
    url: str,
    snapshot_id: str,
    session_id: str | None,
    force: bool = False,
    json_output: bool = False,
) -> int:
    client = SuperlinearClient(base_url=url, timeout_s=300.0)
    payload: dict[str, Any] = {}
    if session_id:
        payload["session_id"] = session_id
    if force:
        payload["force"] = True
    try:
        resp = client.request_json("POST", f"/v1/snapshots/{snapshot_id}/load", payload=payload, timeout_s=300.0)
    except HttpError as exc:
        msg = _format_unreachable(exc, base_url=url)
        if msg:
            raise SnapshotCommandError(msg) from exc
        if exc.status_code == 404:
            raise SnapshotCommandError(f"Snapshot not found: {snapshot_id} (use `spl snapshot ls`).") from exc
        if exc.status_code == 409:
            raise SnapshotCommandError(
                "Target session already exists. Choose a different `--session`, omit it to create a new one, or pass `--force`."
            ) from exc
        if exc.status_code == 429:
            raise SnapshotCommandError("Server is busy (429). Try again.") from exc
        raise SnapshotCommandError(str(exc)) from exc

    if json_output:
        print_json(resp)
        return 0

    if isinstance(resp, dict):
        sid = resp.get("session_id")
        if isinstance(sid, str) and sid:
            print(f"loaded snapshot_id={snapshot_id} session_id={sid}")
            return 0
    print(f"loaded snapshot_id={snapshot_id}")
    return 0


def snapshot_rm(
    *,
    url: str | None = None,
    snapshot_ids: list[str],
    json_output: bool = False,
    allow_remote_delete: bool = False,
) -> int:
    """Delete one or more snapshots.

    Deletes immediately (no confirmation prompt).

    To keep MVP UX safe, remote deletion only happens when `allow_remote_delete=True`
    (set by the CLI when the user explicitly passes `--url` on that invocation).
    """

    is_remote = bool(allow_remote_delete and url and not _is_local_url(url))
    client = SuperlinearClient(base_url=url, timeout_s=30.0) if is_remote else None

    results: list[dict[str, Any]] = []
    errors = 0

    for snapshot_id in snapshot_ids:
        deleted_local = delete_local_snapshot(snapshot_id)
        deleted_remote = False

        # If explicitly requested, also try to delete on remote server
        if client and is_remote:
            try:
                client.request_json("DELETE", f"/v1/snapshots/{snapshot_id}", timeout_s=30.0)
                deleted_remote = True
            except HttpError as exc:
                if exc.status_code != 404:  # Ignore not found on remote
                    pass  # Don't fail if remote delete fails

        if not deleted_local and not deleted_remote:
            print(f"error: snapshot not found: {snapshot_id}", file=sys.stderr)
            errors += 1
            continue

        results.append(
            {
                "snapshot_id": snapshot_id,
                "local": deleted_local,
                "remote": deleted_remote,
            }
        )

        if not json_output:
            if deleted_remote and deleted_local:
                print(f"deleted snapshot_id={snapshot_id} (local + remote)")
            elif deleted_remote:
                print(f"deleted snapshot_id={snapshot_id} (remote)")
            else:
                print(f"deleted snapshot_id={snapshot_id}")

    if json_output:
        print_json({"deleted": results})

    return 1 if errors > 0 else 0


def snapshot_rm_all(
    *,
    url: str | None = None,
    json_output: bool = False,
    allow_remote_delete: bool = False,
) -> int:
    """Delete all snapshots."""
    # Gather all snapshot IDs (local + remote)
    local_snaps = list_local_snapshots()
    local_ids = [s["snapshot_id"] for s in local_snaps if s.get("snapshot_id")]

    remote_ids: list[str] = []
    is_remote = bool(allow_remote_delete and url and not _is_local_url(url))

    if is_remote:
        client = SuperlinearClient(base_url=url, timeout_s=10.0)
        try:
            resp = client.request_json("GET", "/v1/snapshots", timeout_s=10.0)
            raw = resp.get("snapshots") if isinstance(resp, dict) else []
            if isinstance(raw, list):
                remote_ids = [s.get("snapshot_id") for s in raw if isinstance(s, dict) and s.get("snapshot_id")]
        except HttpError:
            pass  # Remote not reachable

    # Deduplicate
    all_ids = list(set(local_ids) | set(remote_ids))

    if not all_ids:
        if not json_output:
            print("(no snapshots)")
        else:
            print_json({"deleted": []})
        return 0

    return snapshot_rm(
        url=url,
        snapshot_ids=all_ids,
        json_output=json_output,
        allow_remote_delete=allow_remote_delete,
    )
