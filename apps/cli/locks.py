from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from apps.cli.state import config_dir


@dataclass(frozen=True)
class LockInfo:
    session_id: str
    pid: int
    pid_start_time_ticks: int | None = None
    created_at_unix_s: int | None = None
    kind: str | None = None
    label: str | None = None


class AlreadyLockedError(RuntimeError):
    def __init__(self, *, lock_path: Path, info: LockInfo) -> None:
        super().__init__(f"Session is already locked: {info.session_id}")
        self.lock_path = lock_path
        self.info = info


def lock_dir() -> Path:
    return config_dir() / "locks"


def _sanitize_filename(s: str) -> str:
    s = s.strip()
    if not s:
        return "empty"
    out = []
    for ch in s:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)[:200]


def lock_path_for_session(session_id: str) -> Path:
    name = _sanitize_filename(session_id)
    return lock_dir() / f"{name}.lock"


def _proc_start_time_ticks(pid: int) -> int | None:
    if pid <= 0:
        return None
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        raw = stat_path.read_text(encoding="utf-8")
    except Exception:
        return None

    rparen = raw.rfind(")")
    if rparen < 0:
        return None
    after = raw[rparen + 2 :].strip()
    parts = after.split()
    # In /proc/<pid>/stat, starttime is field 22; after removing pid+comm, it's at index 19 (0-based).
    if len(parts) <= 19:
        return None
    try:
        return int(parts[19])
    except Exception:
        return None


def _pid_is_running(pid: int, *, expected_start_time_ticks: int | None) -> bool:
    if pid <= 0:
        return False

    if expected_start_time_ticks is not None:
        actual = _proc_start_time_ticks(pid)
        if actual is not None and actual != int(expected_start_time_ticks):
            return False

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


_PID_RE = re.compile(r'"pid"\s*:\s*(-?\d+)')
_START_RE = re.compile(r'"pid_start_time_ticks"\s*:\s*(\d+)')
_SESSION_RE = re.compile(r'"session_id"\s*:\s*"([^"]+)"')


def _parse_lock_text(text: str, *, fallback_session_id: str) -> LockInfo:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return LockInfo(
                session_id=str(obj.get("session_id") or fallback_session_id),
                pid=int(obj.get("pid") or 0),
                pid_start_time_ticks=(
                    int(obj["pid_start_time_ticks"]) if obj.get("pid_start_time_ticks") is not None else None
                ),
                created_at_unix_s=(
                    int(obj["created_at_unix_s"]) if obj.get("created_at_unix_s") is not None else None
                ),
                kind=str(obj.get("kind")) if obj.get("kind") is not None else None,
                label=str(obj.get("label")) if obj.get("label") is not None else None,
            )
    except Exception:
        pass

    # Best-effort fallback parsing (for truncated/corrupt files).
    pid_m = _PID_RE.search(text)
    pid = int(pid_m.group(1)) if pid_m else 0
    start_m = _START_RE.search(text)
    start_ticks = int(start_m.group(1)) if start_m else None
    sess_m = _SESSION_RE.search(text)
    session_id = sess_m.group(1) if sess_m else fallback_session_id
    return LockInfo(session_id=session_id, pid=pid, pid_start_time_ticks=start_ticks)


def read_active_lock(session_id: str) -> LockInfo | None:
    """Return active lock info, auto-removing stale locks."""
    p = lock_path_for_session(session_id)
    try:
        text = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except Exception:
        return None

    info = _parse_lock_text(text, fallback_session_id=session_id)
    if _pid_is_running(info.pid, expected_start_time_ticks=info.pid_start_time_ticks):
        return info

    # Stale lock: remove best-effort.
    try:
        p.unlink()
    except Exception:
        pass
    return None


class SessionLock:
    def __init__(self, *, session_id: str, kind: str, label: str | None = None) -> None:
        self.session_id = session_id
        self.kind = kind
        self.label = label
        self.path = lock_path_for_session(session_id)
        self._acquired = False

        self._pid = os.getpid()
        self._pid_start_time_ticks = _proc_start_time_ticks(self._pid)

    def acquire(self) -> None:
        if self._acquired:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "session_id": self.session_id,
            "pid": self._pid,
            "pid_start_time_ticks": self._pid_start_time_ticks,
            "created_at_unix_s": int(time.time()),
            "kind": self.kind,
            "label": self.label,
        }
        data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

        while True:
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                info = read_active_lock(self.session_id)
                if info is None:
                    # Stale lock removed; retry.
                    continue
                if info.pid == self._pid and info.pid_start_time_ticks == self._pid_start_time_ticks:
                    self._acquired = True
                    return
                raise AlreadyLockedError(lock_path=self.path, info=info)

            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
            self._acquired = True
            return

    def release(self) -> None:
        if not self._acquired:
            return
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass
        self._acquired = False

    def __enter__(self) -> "SessionLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

