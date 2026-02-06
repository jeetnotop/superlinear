from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


STATE_SCHEMA_VERSION = 1

DocsPhase = Literal["INGEST", "QA"]


class StateError(RuntimeError):
    pass


@dataclass
class DocsWorkspaceState:
    session_id: str
    phase: DocsPhase = "INGEST"
    base_snapshot_id: str | None = None
    sources: list[dict[str, Any]] = field(default_factory=list)
    # Retrieval backend for `spl docs`: "bm25" (default), "light", or "off".
    rag_backend: str = "bm25"
    # Light RAG (client-side retrieval) settings for `spl docs`.
    light_rag_enabled: bool = True
    light_rag_k: int = 5
    light_rag_total_chars: int = 12000
    light_rag_per_source_chars: int = 2600
    light_rag_debug: bool = False
    # BM25 backend knobs (client-side; paragraph-level indexing).
    bm25_k_paragraphs: int = 40
    bm25_k_sources: int = 0  # 0 means "use light_rag_k"


@dataclass
class CliState:
    schema_version: int = STATE_SCHEMA_VERSION

    # Saved server URL for CLI commands (set via `spl server connect`)
    server_url: str | None = None

    active_chat_session_id: str | None = None
    chat_checkpoint_snapshot_id: str | None = None
    chat_checkpoints: dict[str, str] = field(default_factory=dict)

    docs_workspaces: dict[str, DocsWorkspaceState] = field(default_factory=dict)


def config_dir() -> Path:
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg_config_home) if xdg_config_home else (Path.home() / ".config")
    return base / "spl"


def state_path(*, base_dir: Path | None = None) -> Path:
    return (base_dir or config_dir()) / "state.json"


def load_state(*, path: Path | None = None) -> CliState:
    p = path or state_path()
    if not p.exists():
        return CliState()

    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        raise StateError(f"Failed to read state file: {p}") from exc

    if not isinstance(raw, dict):
        raise StateError(f"State file must contain a JSON object: {p}")

    version = raw.get("schema_version", 0)
    if version not in {0, STATE_SCHEMA_VERSION}:
        raise StateError(f"Unsupported state schema_version={version!r} in {p}")

    docs_raw = raw.get("docs_workspaces", {})
    if docs_raw is None:
        docs_raw = {}
    if not isinstance(docs_raw, dict):
        raise StateError(f"'docs_workspaces' must be an object in {p}")

    docs_workspaces: dict[str, DocsWorkspaceState] = {}
    for name, ws in docs_raw.items():
        if not isinstance(name, str) or not name:
            continue
        if not isinstance(ws, dict):
            continue
        session_id = ws.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            continue
        phase = ws.get("phase", "INGEST")
        if phase not in {"INGEST", "QA"}:
            phase = "INGEST"
        base_snapshot_id = ws.get("base_snapshot_id")
        if base_snapshot_id is not None and not isinstance(base_snapshot_id, str):
            base_snapshot_id = None

        light_rag_enabled = ws.get("light_rag_enabled", True)
        if not isinstance(light_rag_enabled, bool):
            light_rag_enabled = True
        light_rag_k = ws.get("light_rag_k", 5)
        if not isinstance(light_rag_k, int):
            light_rag_k = 5
        light_rag_total_chars = ws.get("light_rag_total_chars", 12000)
        if not isinstance(light_rag_total_chars, int):
            light_rag_total_chars = 12000
        light_rag_per_source_chars = ws.get("light_rag_per_source_chars", 2600)
        if not isinstance(light_rag_per_source_chars, int):
            light_rag_per_source_chars = 2600
        light_rag_debug = ws.get("light_rag_debug", False)
        if not isinstance(light_rag_debug, bool):
            light_rag_debug = False

        rag_backend = ws.get("rag_backend")
        if rag_backend is not None and not isinstance(rag_backend, str):
            rag_backend = None
        if isinstance(rag_backend, str):
            rag_backend = rag_backend.strip().lower()
            if rag_backend not in {"light", "bm25", "off"}:
                rag_backend = None
        if rag_backend is None:
            rag_backend = "bm25" if light_rag_enabled else "off"
        # Keep the legacy enabled toggle aligned with backend selection.
        light_rag_enabled = rag_backend != "off"

        bm25_k_paragraphs = ws.get("bm25_k_paragraphs", 40)
        if not isinstance(bm25_k_paragraphs, int):
            bm25_k_paragraphs = 40
        bm25_k_sources = ws.get("bm25_k_sources", 0)
        if not isinstance(bm25_k_sources, int):
            bm25_k_sources = 0

        raw_sources = ws.get("sources") or []
        sources: list[dict[str, Any]] = []
        if isinstance(raw_sources, list):
            for s in raw_sources:
                if isinstance(s, str) and s:
                    sources.append({"path": s})
                    continue
                if not isinstance(s, dict):
                    continue
                path_v = s.get("path")
                if not isinstance(path_v, str) or not path_v:
                    continue
                item: dict[str, Any] = {"path": path_v}
                title = s.get("title")
                if isinstance(title, str) and title.strip():
                    item["title"] = title.strip()
                source = s.get("source")
                if isinstance(source, str) and source.strip():
                    item["source"] = source.strip()
                url = s.get("url")
                if isinstance(url, str) and url.strip():
                    item["url"] = url.strip()
                b = s.get("bytes")
                if isinstance(b, int) and b >= 0:
                    item["bytes"] = b
                sha = s.get("sha256")
                if isinstance(sha, str) and sha:
                    item["sha256"] = sha
                added_at = s.get("added_at_unix_s") if s.get("added_at_unix_s") is not None else s.get("added_at")
                if isinstance(added_at, int) and added_at > 0:
                    item["added_at_unix_s"] = added_at
                sources.append(item)
        docs_workspaces[name] = DocsWorkspaceState(
            session_id=session_id,
            phase=phase,  # type: ignore[arg-type]
            base_snapshot_id=base_snapshot_id,
            sources=sources,
            rag_backend=rag_backend,
            light_rag_enabled=light_rag_enabled,
            light_rag_k=light_rag_k,
            light_rag_total_chars=light_rag_total_chars,
            light_rag_per_source_chars=light_rag_per_source_chars,
            light_rag_debug=light_rag_debug,
            bm25_k_paragraphs=bm25_k_paragraphs,
            bm25_k_sources=bm25_k_sources,
        )

    server_url = raw.get("server_url")
    if server_url is not None and not isinstance(server_url, str):
        server_url = None

    active_chat_session_id = raw.get("active_chat_session_id")
    if active_chat_session_id is not None and not isinstance(active_chat_session_id, str):
        active_chat_session_id = None

    chat_checkpoint_snapshot_id = raw.get("chat_checkpoint_snapshot_id")
    if chat_checkpoint_snapshot_id is not None and not isinstance(chat_checkpoint_snapshot_id, str):
        chat_checkpoint_snapshot_id = None

    checkpoints_raw = raw.get("chat_checkpoints", {})
    if checkpoints_raw is None:
        checkpoints_raw = {}
    if not isinstance(checkpoints_raw, dict):
        raise StateError(f"'chat_checkpoints' must be an object in {p}")
    chat_checkpoints: dict[str, str] = {}
    for sid, snap in checkpoints_raw.items():
        if not isinstance(sid, str) or not sid:
            continue
        if not isinstance(snap, str) or not snap:
            continue
        chat_checkpoints[sid] = snap

    # Back-compat: older state stored only the active checkpoint.
    if (
        active_chat_session_id
        and chat_checkpoint_snapshot_id
        and active_chat_session_id not in chat_checkpoints
    ):
        chat_checkpoints[active_chat_session_id] = chat_checkpoint_snapshot_id

    # Keep the legacy field aligned with the active session for callers that still read it.
    if active_chat_session_id:
        chat_checkpoint_snapshot_id = chat_checkpoints.get(active_chat_session_id)
    else:
        chat_checkpoint_snapshot_id = None

    return CliState(
        schema_version=STATE_SCHEMA_VERSION,
        server_url=server_url,
        active_chat_session_id=active_chat_session_id,
        chat_checkpoint_snapshot_id=chat_checkpoint_snapshot_id,
        chat_checkpoints=chat_checkpoints,
        docs_workspaces=docs_workspaces,
    )


def save_state(state: CliState, *, path: Path | None = None) -> None:
    p = path or state_path()
    p.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = asdict(state)
    payload["schema_version"] = STATE_SCHEMA_VERSION
    if state.active_chat_session_id:
        payload["chat_checkpoint_snapshot_id"] = state.chat_checkpoints.get(state.active_chat_session_id)
    else:
        payload["chat_checkpoint_snapshot_id"] = None

    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    tmp_dir = str(p.parent)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=tmp_dir,
            delete=False,
            prefix=".state.",
            suffix=".tmp",
        ) as f:
            f.write(encoded)
            tmp_path = Path(f.name)
        tmp_path.replace(p)
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
