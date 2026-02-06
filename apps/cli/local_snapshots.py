"""Local snapshot operations that work without the server running.

Snapshots are stored in ~/.cache/spl/snapshots/{model_id}/{fingerprint}/{snapshot_id}/
This module provides direct filesystem access for listing and deleting snapshots.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any


def _default_snapshot_dir() -> Path:
    xdg_cache = os.environ.get("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache"))
    return Path(xdg_cache) / "spl" / "snapshots"


def _fmt_unix_utc(ts: int | None) -> str:
    if not ts:
        return ""
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(ts)))
    except Exception:
        return str(ts)


def list_local_snapshots(*, root_dir: Path | None = None) -> list[dict[str, Any]]:
    """List all snapshots across all models/fingerprints."""
    root = root_dir or _default_snapshot_dir()
    if not root.exists():
        return []

    snapshots: list[dict[str, Any]] = []

    # Walk: root / model_id / fingerprint / snapshot_id / manifest.json
    for model_dir in root.iterdir():
        if not model_dir.is_dir():
            continue
        model_id = model_dir.name

        for fp_dir in model_dir.iterdir():
            if not fp_dir.is_dir():
                continue
            fingerprint = fp_dir.name

            for snap_dir in fp_dir.iterdir():
                if not snap_dir.is_dir():
                    continue
                if snap_dir.name.startswith(".tmp-"):
                    continue

                manifest_path = snap_dir / "manifest.json"
                if not manifest_path.is_file():
                    continue

                try:
                    data = json.loads(manifest_path.read_text(encoding="utf-8"))
                    snapshot_id = data.get("snapshot_id") or snap_dir.name
                    created_at = data.get("created_at")
                    metadata = data.get("metadata", {})
                    session = data.get("session", {})

                    snapshots.append({
                        "snapshot_id": snapshot_id,
                        "model_id": model_id,
                        "fingerprint": fingerprint[:12] + "..." if len(fingerprint) > 12 else fingerprint,
                        "created_at": created_at,
                        "title": metadata.get("title"),
                        "pos": session.get("current_pos"),
                        "max_seq_len": session.get("max_seq_len"),
                        "path": str(snap_dir),
                    })
                except Exception:
                    continue

    # Sort by created_at descending
    snapshots.sort(key=lambda s: int(s.get("created_at") or 0), reverse=True)
    return snapshots


def delete_local_snapshot(snapshot_id: str, *, root_dir: Path | None = None) -> bool:
    """Delete a snapshot by ID. Returns True if found and deleted."""
    root = root_dir or _default_snapshot_dir()
    if not root.exists():
        return False

    # Search for the snapshot across all models/fingerprints
    for model_dir in root.iterdir():
        if not model_dir.is_dir():
            continue
        for fp_dir in model_dir.iterdir():
            if not fp_dir.is_dir():
                continue
            snap_dir = fp_dir / snapshot_id
            if snap_dir.is_dir() and (snap_dir / "manifest.json").is_file():
                shutil.rmtree(snap_dir)
                return True

    return False


def get_local_snapshot(snapshot_id: str, *, root_dir: Path | None = None) -> dict[str, Any] | None:
    """Get snapshot info by ID."""
    root = root_dir or _default_snapshot_dir()
    if not root.exists():
        return None

    for model_dir in root.iterdir():
        if not model_dir.is_dir():
            continue
        for fp_dir in model_dir.iterdir():
            if not fp_dir.is_dir():
                continue
            snap_dir = fp_dir / snapshot_id
            manifest_path = snap_dir / "manifest.json"
            if manifest_path.is_file():
                try:
                    data = json.loads(manifest_path.read_text(encoding="utf-8"))
                    metadata = data.get("metadata", {})
                    session = data.get("session", {})
                    return {
                        "snapshot_id": data.get("snapshot_id") or snapshot_id,
                        "model_id": model_dir.name,
                        "fingerprint": fp_dir.name,
                        "created_at": data.get("created_at"),
                        "title": metadata.get("title"),
                        "description": metadata.get("description"),
                        "pos": session.get("current_pos"),
                        "max_seq_len": session.get("max_seq_len"),
                        "message_count": session.get("message_count"),
                        "path": str(snap_dir),
                    }
                except Exception:
                    continue
    return None
