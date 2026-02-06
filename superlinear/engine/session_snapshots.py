"""Session snapshot persistence (v1).

This module implements the v1 design from `local/docs/session_cache_roadmap_v1_v2_v3.md`:
- A **session** is GPU-resident and mutable.
- A **snapshot** is disk-resident and immutable.

V1 deliberately keeps the format simple:
- one snapshot directory per save
- one `manifest.json`
- one `transcript.json`
- one `cache.pt` containing torch tensors (KV + Mamba states)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

_SNAPSHOT_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}$")


class SnapshotCompatibilityError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stable_json_dumps(obj: Any) -> str:
    # We only need stability for hashing; allow NaN/Infinity from model configs.
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str, allow_nan=True)


def compute_model_compatibility(*, adapter: Any, model_id: str) -> dict[str, Any]:
    """Compute a best-effort compatibility fingerprint for snapshot load checks."""
    model_path = None
    dtype = None
    try:
        info = getattr(adapter, "model_info", None)
        if isinstance(info, Mapping):
            model_path = info.get("model_path")
            dtype = info.get("dtype")
    except Exception:
        pass

    cfg_hash = None
    try:
        model = getattr(adapter, "model", None)
        cfg = getattr(model, "config", None)
        if cfg is not None and hasattr(cfg, "to_dict"):
            cfg_hash = hashlib.sha256(_stable_json_dumps(cfg.to_dict()).encode("utf-8")).hexdigest()
    except Exception:
        cfg_hash = None

    file_hashes: dict[str, str] = {}
    try:
        if model_path and isinstance(model_path, str) and os.path.isdir(model_path):
            base = Path(model_path)
            for rel in ("config.json", "model.safetensors.index.json"):
                p = base / rel
                if p.is_file():
                    file_hashes[rel] = _sha256_file(p)
    except Exception:
        file_hashes = {}

    payload = {
        "model_id": model_id,
        "model_path": model_path,
        "dtype": dtype,
        "config_sha256": cfg_hash,
        "files_sha256": file_hashes,
        "schema": "superlinear.session_snapshot.v1",
    }
    fingerprint = hashlib.sha256(_stable_json_dumps(payload).encode("utf-8")).hexdigest()
    return {"fingerprint": fingerprint, "payload": payload}


def export_hybrid_mamba_attention_static_cache(*, cache: Any, current_pos: int) -> dict[str, Any]:
    """Export a HybridMambaAttentionStaticCache-like object into CPU tensors.

    This is a structural serializer: it only depends on attribute names used by
    `local/superlinear-exp-v0.1`'s `HybridMambaAttentionStaticCache`.
    """
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch is required to save/load session snapshots.") from exc

    transformer_layers = list(getattr(cache, "transformer_layers", []) or [])
    key_cache = getattr(cache, "key_cache", None)
    value_cache = getattr(cache, "value_cache", None)
    conv_states = getattr(cache, "conv_states", None)
    ssm_states = getattr(cache, "ssm_states", None)
    max_seq_len = int(getattr(cache, "max_seq_len", 0) or 0)

    if not isinstance(current_pos, int) or current_pos < 0:
        raise ValueError(f"current_pos must be a non-negative int, got {current_pos!r}")
    if max_seq_len <= 0:
        raise ValueError(f"cache.max_seq_len must be > 0, got {max_seq_len!r}")

    if not isinstance(key_cache, list) or not isinstance(value_cache, list):
        raise TypeError("Unsupported cache: expected .key_cache and .value_cache lists.")

    kv_layers: list[int] = []
    keys: dict[str, torch.Tensor] = {}
    values: dict[str, torch.Tensor] = {}

    # Prefer transformer_layers if available; otherwise scan all layers.
    candidate_layers = transformer_layers or list(range(len(key_cache)))
    for layer_idx in candidate_layers:
        try:
            k = key_cache[layer_idx]
            v = value_cache[layer_idx]
        except Exception:
            continue
        if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
            continue
        if k.dim() != 4 or v.dim() != 4:
            continue

        kv_layers.append(int(layer_idx))
        end = min(current_pos, int(k.shape[2]))
        keys[str(layer_idx)] = k[:, :, :end, :].detach().contiguous().to("cpu")
        values[str(layer_idx)] = v[:, :, :end, :].detach().contiguous().to("cpu")

    conv_cpu = None
    if isinstance(conv_states, torch.Tensor):
        conv_cpu = conv_states.detach().contiguous().to("cpu")

    ssm_cpu = None
    if isinstance(ssm_states, torch.Tensor):
        ssm_cpu = ssm_states.detach().contiguous().to("cpu")

    return {
        "cache_format": "hybrid_mamba_attention_static_v1",
        "cache_class": f"{cache.__class__.__module__}.{cache.__class__.__qualname__}",
        "max_seq_len": max_seq_len,
        "current_pos": int(current_pos),
        "kv_layers": kv_layers,
        "key_cache": keys,
        "value_cache": values,
        "conv_states": conv_cpu,
        "ssm_states": ssm_cpu,
    }


def import_hybrid_mamba_attention_static_cache(*, cache: Any, payload: Mapping[str, Any]) -> int:
    """Populate `cache` in-place from a payload produced by export_*.

    Returns:
        current_pos restored into the cache.
    """
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch is required to save/load session snapshots.") from exc

    if payload.get("cache_format") != "hybrid_mamba_attention_static_v1":
        raise ValueError(f"Unsupported cache_format: {payload.get('cache_format')!r}")

    max_seq_len = int(payload.get("max_seq_len") or 0)
    current_pos = int(payload.get("current_pos") or 0)
    if max_seq_len <= 0:
        raise ValueError(f"Invalid max_seq_len in snapshot: {max_seq_len}")
    if current_pos < 0:
        raise ValueError(f"Invalid current_pos in snapshot: {current_pos}")

    # Validate target cache shape.
    # We allow loading into a *larger* cache allocation (e.g., for session resize).
    cache_max = int(getattr(cache, "max_seq_len", 0) or 0)
    if cache_max < max_seq_len:
        raise ValueError(
            f"Cache max_seq_len too small: snapshot={max_seq_len} > target={cache_max}"
        )

    key_cache = getattr(cache, "key_cache", None)
    value_cache = getattr(cache, "value_cache", None)
    if not isinstance(key_cache, list) or not isinstance(value_cache, list):
        raise TypeError("Unsupported target cache: expected .key_cache and .value_cache lists.")

    kv_layers = payload.get("kv_layers") or []
    key_dict = payload.get("key_cache") or {}
    val_dict = payload.get("value_cache") or {}

    if not isinstance(kv_layers, list) or not isinstance(key_dict, Mapping) or not isinstance(val_dict, Mapping):
        raise TypeError("Invalid KV payload types.")

    for layer_idx in kv_layers:
        idx = int(layer_idx)
        if idx < 0 or idx >= len(key_cache):
            raise ValueError(f"Snapshot KV layer out of range: {idx}")

        src_k = key_dict.get(str(idx))
        src_v = val_dict.get(str(idx))
        if not isinstance(src_k, torch.Tensor) or not isinstance(src_v, torch.Tensor):
            raise TypeError(f"Missing KV tensors for layer {idx}")

        dst_k = key_cache[idx]
        dst_v = value_cache[idx]
        if not isinstance(dst_k, torch.Tensor) or not isinstance(dst_v, torch.Tensor):
            raise TypeError(f"Target cache missing tensors for layer {idx}")
        if dst_k.dim() != 4 or dst_v.dim() != 4:
            raise TypeError(f"Target cache tensors must be 4D for layer {idx}")

        end = int(src_k.shape[2])
        if end > dst_k.shape[2]:
            raise ValueError(f"Snapshot KV length {end} exceeds target allocation {dst_k.shape[2]} for {idx}")

        dst_k[:, :, :end, :].copy_(src_k.to(device=dst_k.device, dtype=dst_k.dtype))
        dst_v[:, :, :end, :].copy_(src_v.to(device=dst_v.device, dtype=dst_v.dtype))

    conv_src = payload.get("conv_states")
    if conv_src is not None:
        conv_dst = getattr(cache, "conv_states", None)
        if not isinstance(conv_src, torch.Tensor) or not isinstance(conv_dst, torch.Tensor):
            raise TypeError("conv_states must be tensors in both snapshot and target cache.")
        conv_dst.copy_(conv_src.to(device=conv_dst.device, dtype=conv_dst.dtype))

    ssm_src = payload.get("ssm_states")
    if ssm_src is not None:
        ssm_dst = getattr(cache, "ssm_states", None)
        if not isinstance(ssm_src, torch.Tensor) or not isinstance(ssm_dst, torch.Tensor):
            raise TypeError("ssm_states must be tensors in both snapshot and target cache.")
        ssm_dst.copy_(ssm_src.to(device=ssm_dst.device, dtype=ssm_dst.dtype))

    # Sync the static-cache prefill cursor if present (required for session append).
    seen = getattr(cache, "_layer_seen_tokens", None)
    transformer_layers = getattr(cache, "transformer_layers", None)
    if isinstance(seen, list):
        if isinstance(transformer_layers, list) and transformer_layers:
            for layer in transformer_layers:
                li = int(layer)
                if 0 <= li < len(seen):
                    seen[li] = int(current_pos)
        else:
            for i in range(len(seen)):
                seen[i] = int(current_pos)

    return current_pos


@dataclass(frozen=True)
class SnapshotManifest:
    snapshot_id: str
    created_at: int
    compat: dict[str, Any]
    session: dict[str, Any]
    metadata: dict[str, Any]
    files: dict[str, str]

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SnapshotManifest":
        return cls(
            snapshot_id=str(d.get("snapshot_id") or ""),
            created_at=int(d.get("created_at") or 0),
            compat=dict(d.get("compat") or {}),
            session=dict(d.get("session") or {}),
            metadata=dict(d.get("metadata") or {}),
            files=dict(d.get("files") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "snapshot_id": self.snapshot_id,
            "created_at": self.created_at,
            "compat": self.compat,
            "session": self.session,
            "metadata": self.metadata,
            "files": self.files,
        }


class SnapshotStoreV1:
    """Filesystem-backed snapshot store (v1)."""

    def __init__(self, *, root_dir: str | Path, model_id: str, compat: Mapping[str, Any]) -> None:
        self._root_dir = Path(root_dir)
        self._model_id = str(model_id)
        self._compat = dict(compat)
        fp = self._compat.get("fingerprint") or "unknown"
        self._model_dir = self._root_dir / self._model_id / str(fp)
        self._model_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model_dir(self) -> Path:
        return self._model_dir

    @property
    def compat(self) -> dict[str, Any]:
        return dict(self._compat)

    def _validate_snapshot_id(self, snapshot_id: str) -> None:
        if not snapshot_id or not _SNAPSHOT_ID_RE.match(snapshot_id):
            raise ValueError("Invalid snapshot_id.")

    def list_snapshots(self) -> list[SnapshotManifest]:
        out: list[SnapshotManifest] = []
        if not self._model_dir.exists():
            return out
        for p in self._model_dir.iterdir():
            if not p.is_dir():
                continue
            if p.name.startswith(".tmp-"):
                continue
            mf = p / "manifest.json"
            if not mf.is_file():
                continue
            try:
                d = json.loads(mf.read_text(encoding="utf-8"))
                out.append(SnapshotManifest.from_dict(d))
            except Exception:
                continue
        out.sort(key=lambda m: int(m.created_at), reverse=True)
        return out

    def get_manifest(self, snapshot_id: str) -> SnapshotManifest:
        self._validate_snapshot_id(snapshot_id)
        mf = self._model_dir / snapshot_id / "manifest.json"
        if not mf.is_file():
            raise FileNotFoundError(snapshot_id)
        d = json.loads(mf.read_text(encoding="utf-8"))
        return SnapshotManifest.from_dict(d)

    def patch_metadata(
        self,
        snapshot_id: str,
        *,
        title: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
    ) -> SnapshotManifest:
        manifest = self.get_manifest(snapshot_id)
        meta = dict(manifest.metadata)
        if title is not None:
            meta["title"] = str(title)
        if description is not None:
            meta["description"] = str(description)
        if tags is not None:
            meta["tags"] = [str(t) for t in tags]

        updated = SnapshotManifest(
            snapshot_id=manifest.snapshot_id,
            created_at=manifest.created_at,
            compat=manifest.compat,
            session=manifest.session,
            metadata=meta,
            files=manifest.files,
        )

        mf = self._model_dir / snapshot_id / "manifest.json"
        tmp = mf.with_suffix(".json.tmp")
        tmp.write_text(_stable_json_dumps(updated.to_dict()), encoding="utf-8")
        tmp.replace(mf)
        return updated

    def delete_snapshot(self, snapshot_id: str) -> None:
        self._validate_snapshot_id(snapshot_id)
        d = self._model_dir / snapshot_id
        if not d.exists():
            raise FileNotFoundError(snapshot_id)
        shutil.rmtree(d)

    def create_snapshot(
        self,
        *,
        transcript: list[dict[str, Any]],
        cache_payload: Mapping[str, Any],
        session: Mapping[str, Any],
        title: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
    ) -> SnapshotManifest:
        try:
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("torch is required to save/load session snapshots.") from exc

        snapshot_id = uuid.uuid4().hex
        created_at = int(time.time())

        final_dir = self._model_dir / snapshot_id
        tmp_dir = self._model_dir / f".tmp-{snapshot_id}-{uuid.uuid4().hex}"
        tmp_dir.mkdir(parents=True, exist_ok=False)

        try:
            (tmp_dir / "transcript.json").write_text(
                _stable_json_dumps({"messages": transcript}), encoding="utf-8"
            )
            torch.save(dict(cache_payload), tmp_dir / "cache.pt")

            manifest = SnapshotManifest(
                snapshot_id=snapshot_id,
                created_at=created_at,
                compat=dict(self._compat),
                session=dict(session),
                metadata={
                    "title": title,
                    "description": description,
                    "tags": list(tags) if tags is not None else [],
                },
                files={"transcript": "transcript.json", "cache": "cache.pt"},
            )
            (tmp_dir / "manifest.json").write_text(_stable_json_dumps(manifest.to_dict()), encoding="utf-8")

            tmp_dir.rename(final_dir)
            return manifest
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    def load_snapshot_payload(self, snapshot_id: str) -> tuple[SnapshotManifest, list[dict[str, Any]], dict[str, Any]]:
        try:
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("torch is required to save/load session snapshots.") from exc

        manifest = self.get_manifest(snapshot_id)
        if (manifest.compat.get("fingerprint") or None) != (self._compat.get("fingerprint") or None):
            raise SnapshotCompatibilityError("Snapshot fingerprint does not match the loaded model.")

        d = self._model_dir / snapshot_id
        transcript_path = d / manifest.files.get("transcript", "transcript.json")
        cache_path = d / manifest.files.get("cache", "cache.pt")
        if not transcript_path.is_file() or not cache_path.is_file():
            raise FileNotFoundError(snapshot_id)

        transcript_obj = json.loads(transcript_path.read_text(encoding="utf-8"))
        messages = transcript_obj.get("messages")
        if not isinstance(messages, list):
            messages = []

        cache_payload = torch.load(cache_path, map_location="cpu")
        if not isinstance(cache_payload, dict):
            raise TypeError("Invalid cache payload (expected dict).")

        return manifest, [m for m in messages if isinstance(m, dict)], cache_payload

