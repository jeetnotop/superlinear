"""FastAPI app for OpenAI-style Chat Completions.

The HTTP layer lives under `apps/` and can depend on heavier deps (FastAPI, uvicorn).
All model execution is delegated to the core engine (`superlinear/engine`).
"""

from __future__ import annotations

import asyncio
import os
import threading
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from starlette.responses import JSONResponse, StreamingResponse

from superlinear.engine.chat_engine import ChatEngine
from superlinear.engine.chat_types import ChatMessage, ChatRequest, StreamOptions, Timing, ToolCall, Usage
from superlinear.engine.session_snapshots import (
    SnapshotCompatibilityError,
    SnapshotStoreV1,
    compute_model_compatibility,
    export_hybrid_mamba_attention_static_cache,
    import_hybrid_mamba_attention_static_cache,
)


def create_app(
    *,
    engine: ChatEngine,
    model_id: str,
    http_max_concurrency: int | None = None,
    http_max_completion_tokens: int | None = None,
) -> FastAPI:
    app = FastAPI(title="Superlinear Inference Server", version="0.1.0")

    default_max_seq_len = 131_072

    @dataclass
    class _HttpSession:
        max_seq_len: int
        messages: list[dict[str, Any]] = field(default_factory=list)

    _sessions_lock = threading.Lock()
    _sessions: dict[str, _HttpSession] = {}

    _engine_lock = getattr(engine, "_lock", threading.Lock())

    _snapshot_store_lock = threading.Lock()
    _snapshot_store: SnapshotStoreV1 | None = None

    http_semaphore: asyncio.Semaphore | None = None
    if http_max_concurrency is not None:
        try:
            http_max_concurrency = int(http_max_concurrency)
        except Exception as exc:
            raise ValueError("http_max_concurrency must be an integer") from exc
        if http_max_concurrency > 0:
            http_semaphore = asyncio.Semaphore(http_max_concurrency)
        elif http_max_concurrency < 0:
            raise ValueError("http_max_concurrency must be >= 0")

    if http_max_completion_tokens is not None:
        try:
            http_max_completion_tokens = int(http_max_completion_tokens)
        except Exception as exc:
            raise ValueError("http_max_completion_tokens must be an integer") from exc
        if http_max_completion_tokens <= 0:
            raise ValueError("http_max_completion_tokens must be > 0")

    async def _wait_for_disconnect(request: Request, poll_s: float = 0.1) -> None:
        while True:
            if await request.is_disconnected():
                return
            await asyncio.sleep(poll_s)

    async def _run_with_disconnect_cancellation(request: Request, coro: Any) -> Any:
        task = asyncio.create_task(coro)
        disconnect_task = asyncio.create_task(_wait_for_disconnect(request))
        # Yield to let both tasks start (handles coroutines that return synchronously).
        await asyncio.sleep(0)
        done, pending = await asyncio.wait(
            {task, disconnect_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if disconnect_task in done:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            raise HTTPException(status_code=499, detail="Client disconnected")

        disconnect_task.cancel()
        try:
            await disconnect_task
        except asyncio.CancelledError:
            pass
        return task.result()

    async def _try_acquire_semaphore() -> None:
        if http_semaphore is None:
            return
        try:
            await asyncio.wait_for(http_semaphore.acquire(), timeout=0.001)
        except TimeoutError as exc:
            raise HTTPException(status_code=429, detail="Server is busy") from exc

    def _get_snapshot_store() -> SnapshotStoreV1:
        nonlocal _snapshot_store
        with _snapshot_store_lock:
            if _snapshot_store is not None:
                return _snapshot_store

            adapter = getattr(engine, "adapter", None)
            if adapter is None:
                raise HTTPException(status_code=500, detail="Engine does not expose an adapter.")

            xdg_cache = os.environ.get("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache"))
            default_snapshot_dir = os.path.join(xdg_cache, "spl", "snapshots")
            root_dir = os.environ.get("SUPERLINEAR_SNAPSHOT_DIR", default_snapshot_dir)
            compat = compute_model_compatibility(adapter=adapter, model_id=model_id)
            _snapshot_store = SnapshotStoreV1(root_dir=root_dir, model_id=model_id, compat=compat)
            return _snapshot_store

    async def _json_dict_or_empty(request: Request) -> dict[str, Any]:
        try:
            payload = await request.json()
        except Exception:
            return {}
        if isinstance(payload, dict):
            return payload
        raise HTTPException(status_code=400, detail="Request body must be a JSON object.")

    # -------------------------------------------------------------------------
    # Health & Models
    # -------------------------------------------------------------------------

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        now = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": now,
                    "owned_by": "superlinear",
                }
            ],
        }

    # -------------------------------------------------------------------------
    # Session Management
    # -------------------------------------------------------------------------

    @app.post("/v1/sessions")
    async def create_session(request: Request) -> Any:
        """Create a new stateful session for multi-turn conversations."""
        payload = await request.json()
        session_id = payload.get("session_id")
        if not session_id or not isinstance(session_id, str):
            raise HTTPException(status_code=400, detail="'session_id' is required and must be a string.")

        max_seq_len = payload.get("max_seq_len", default_max_seq_len)
        try:
            max_seq_len = int(max_seq_len)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail="'max_seq_len' must be an integer.") from exc

        try:
            with _engine_lock:
                engine.adapter.create_session(
                    cache_id=session_id,
                    max_seq_len=max_seq_len,
                )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        with _sessions_lock:
            _sessions[session_id] = _HttpSession(max_seq_len=max_seq_len)

        return JSONResponse({"status": "created", "session_id": session_id})

    @app.get("/v1/sessions")
    async def list_sessions() -> Any:
        """List all active sessions."""
        with _engine_lock:
            sessions = engine.adapter.list_sessions()
        return JSONResponse({"sessions": sessions})

    @app.get("/v1/sessions/{session_id}")
    async def get_session_info(session_id: str) -> Any:
        """Get information about a specific session."""
        try:
            with _engine_lock:
                info = engine.adapter.get_session_info(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        # Convenience aliases for client code/tests.
        info = dict(info)
        info["cache_position"] = info.get("current_pos")
        with _sessions_lock:
            meta = _sessions.get(session_id)
            if meta is not None:
                info["message_count"] = len(meta.messages)
        return JSONResponse(info)

    @app.delete("/v1/sessions/{session_id}")
    async def close_session(session_id: str) -> Any:
        """Close a session and free its resources."""
        # Check if session exists
        try:
            with _engine_lock:
                engine.adapter.get_session_info(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        with _engine_lock:
            engine.adapter.close_session(session_id)
        with _sessions_lock:
            _sessions.pop(session_id, None)
        return JSONResponse({"status": "closed", "session_id": session_id})

    @app.get("/v1/sessions/{session_id}/history")
    async def get_session_history(session_id: str) -> Any:
        """Get the stored chat history for a session."""
        try:
            with _engine_lock:
                engine.adapter.get_session_info(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        with _sessions_lock:
            meta = _sessions.get(session_id)
            if meta is None:
                return JSONResponse({"session_id": session_id, "messages": []})
            return JSONResponse({"session_id": session_id, "messages": meta.messages})

    @app.post("/v1/sessions/{session_id}/rollback")
    async def rollback_session(session_id: str, request: Request) -> Any:
        """Rollback a session to an earlier message index by replaying history.

        Body:
          - keep_messages: int  (number of messages to keep from the start)
        """
        payload = await request.json()
        keep_messages = payload.get("keep_messages")
        try:
            keep_messages = int(keep_messages)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="'keep_messages' must be an integer.") from exc
        if keep_messages < 0:
            raise HTTPException(status_code=400, detail="'keep_messages' must be >= 0.")

        # Ensure session exists and retrieve max_seq_len.
        try:
            with _engine_lock:
                adapter_info = engine.adapter.get_session_info(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        with _sessions_lock:
            meta = _sessions.get(session_id)
            if meta is None:
                raise HTTPException(status_code=404, detail=f"No stored history for session: {session_id}")
            meta.messages = meta.messages[:keep_messages]
            max_seq_len = meta.max_seq_len
            history_msgs = list(meta.messages)

        # Recreate adapter session and replay history prompt.
        with _engine_lock:
            engine.adapter.close_session(session_id)
            engine.adapter.create_session(cache_id=session_id, max_seq_len=max_seq_len)

        if history_msgs:
            # Build prompt from stored history WITHOUT adding a generation prompt.
            # The next /chat/completions call will add user msg + generation prompt.
            chat_req = _parse_chat_request({"messages": history_msgs, "max_tokens": 1})
            tokenizer = getattr(engine.adapter, "tokenizer", None)
            apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
            if not callable(apply_chat_template):
                raise HTTPException(status_code=500, detail="Tokenizer does not support apply_chat_template().")

            injected = engine._inject_tool_choice(list(chat_req.messages), chat_req.tool_choice)  # type: ignore[attr-defined]
            template_messages = engine._messages_for_template(injected)  # type: ignore[attr-defined]
            kwargs = engine._chat_template_kwargs(  # type: ignore[attr-defined]
                chat_req, enable_thinking=engine._effective_enable_thinking(chat_req)  # type: ignore[attr-defined]
            )
            input_ids = apply_chat_template(
                template_messages,
                add_generation_prompt=False,
                **kwargs,
            )
            with _engine_lock:
                engine.adapter.append_to_session(cache_id=session_id, input_ids=input_ids)

        with _engine_lock:
            new_info = engine.adapter.get_session_info(session_id)
        new_info = dict(new_info)
        new_info["cache_position"] = new_info.get("current_pos")
        new_info["message_count"] = keep_messages
        return JSONResponse({"status": "ok", "session_id": session_id, "session": new_info})

    def _parse_resize_strategy(payload: dict[str, Any]) -> str:
        strategy = payload.get("strategy", "auto")
        if strategy is None:
            strategy = "auto"
        if not isinstance(strategy, str):
            raise HTTPException(status_code=400, detail="'strategy' must be a string.")
        strategy = strategy.lower().strip()
        if strategy not in {"auto", "gpu", "disk"}:
            raise HTTPException(status_code=400, detail="'strategy' must be one of: auto, gpu, disk.")
        return strategy

    def _next_pow2_strictly_greater(n: int) -> int:
        if n <= 0:
            return 1
        p = 1 << ((n - 1).bit_length())
        if p == n:
            p *= 2
        return p

    def _resize_session_to(*, session_id: str, target_max_seq_len: int, strategy: str) -> dict[str, Any]:
        if target_max_seq_len <= 0:
            raise HTTPException(status_code=400, detail="'max_seq_len' must be > 0.")

        def _allocate_and_restore(*, close_first: bool) -> dict[str, Any]:
            # Everything here runs under _engine_lock.
            exported = engine.adapter.export_session(session_id)
            current_pos = int(exported.get("current_pos") or 0)
            old_max = int(exported.get("max_seq_len") or 0)

            if current_pos < 0:
                raise HTTPException(status_code=500, detail="Invalid session current_pos.")
            if old_max <= 0:
                raise HTTPException(status_code=500, detail="Invalid session max_seq_len.")
            if target_max_seq_len < current_pos:
                raise HTTPException(
                    status_code=400,
                    detail=f"'max_seq_len' ({target_max_seq_len}) must be >= current_pos ({current_pos}).",
                )
            if target_max_seq_len == old_max:
                info = engine.adapter.get_session_info(session_id)
                info = dict(info)
                info["cache_position"] = info.get("current_pos")
                return {"status": "noop", "session_id": session_id, "session": info}

            cache_payload = export_hybrid_mamba_attention_static_cache(
                cache=exported["past_key_values"],
                current_pos=current_pos,
            )

            model = getattr(engine.adapter, "model", None)
            if model is None or not hasattr(model, "create_static_cache"):
                raise HTTPException(status_code=500, detail="Adapter does not expose create_static_cache().")

            if close_first:
                # Free the old cache to reduce peak VRAM.
                engine.adapter.close_session(session_id)

            past_key_values = model.create_static_cache(batch_size=1, max_seq_len=target_max_seq_len)
            restored_pos = import_hybrid_mamba_attention_static_cache(cache=past_key_values, payload=cache_payload)
            engine.adapter.restore_session(
                cache_id=session_id,
                past_key_values=past_key_values,
                current_pos=restored_pos,
                max_seq_len=target_max_seq_len,
                next_token_logits=None,
                overwrite=True,
            )

            info = engine.adapter.get_session_info(session_id)
            info = dict(info)
            info["cache_position"] = info.get("current_pos")
            return {
                "status": "resized",
                "session_id": session_id,
                "old_max_seq_len": old_max,
                "max_seq_len": target_max_seq_len,
                "current_pos": restored_pos,
                "session": info,
            }

        try:
            with _engine_lock:
                if strategy == "gpu":
                    result = _allocate_and_restore(close_first=False)
                elif strategy == "disk":
                    result = _allocate_and_restore(close_first=True)
                else:  # auto
                    try:
                        result = _allocate_and_restore(close_first=False)
                    except Exception as exc:
                        # Best-effort fallback on CUDA OOM by freeing old cache first.
                        try:
                            import torch  # type: ignore

                            if isinstance(exc, torch.cuda.OutOfMemoryError):
                                result = _allocate_and_restore(close_first=True)
                            else:
                                raise
                        except HTTPException:
                            raise
                        except Exception:
                            raise
        except HTTPException:
            raise
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        with _sessions_lock:
            meta = _sessions.get(session_id)
            if meta is not None:
                meta.max_seq_len = int(target_max_seq_len)
                if isinstance(result, dict) and isinstance(result.get("session"), dict):
                    result["session"]["message_count"] = len(meta.messages)

        return result

    @app.post("/v1/sessions/{session_id}/resize")
    async def resize_session(session_id: str, request: Request) -> Any:
        """Resize a session KV cache to a new max sequence length.

        Body:
          - max_seq_len: int (required)
          - strategy: "auto" | "gpu" | "disk" (optional; default: "auto")

        Strategies:
          - gpu: allocate new cache while old cache is still resident (higher peak VRAM)
          - disk: free old cache before allocating new cache (lower peak VRAM)
          - auto: try gpu, fall back to disk on OOM
        """

        payload = await _json_dict_or_empty(request)
        raw_max = payload.get("max_seq_len")
        if raw_max is None:
            raise HTTPException(status_code=400, detail="'max_seq_len' is required.")
        try:
            new_max_seq_len = int(raw_max)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="'max_seq_len' must be an integer.") from exc

        strategy = _parse_resize_strategy(payload)
        return JSONResponse(
            _resize_session_to(session_id=session_id, target_max_seq_len=new_max_seq_len, strategy=strategy)
        )

    @app.post("/v1/sessions/{session_id}/resize/next_pow2")
    async def resize_session_next_pow2(session_id: str, request: Request) -> Any:
        """Resize a session KV cache to the next power-of-two max sequence length.

        Body (optional):
          - strategy: "auto" | "gpu" | "disk" (default: "auto")

        Example:
          - 131072 -> 262144
          - 262144 -> 524288
        """

        payload = await _json_dict_or_empty(request)
        strategy = _parse_resize_strategy(payload)

        try:
            with _engine_lock:
                info = engine.adapter.get_session_info(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        old_max = int(info.get("max_seq_len") or 0)
        new_max = _next_pow2_strictly_greater(old_max)

        result = _resize_session_to(session_id=session_id, target_max_seq_len=new_max, strategy=strategy)
        if isinstance(result, dict):
            result = dict(result)
            result["mode"] = "next_pow2"
        return JSONResponse(result)

    # -------------------------------------------------------------------------
    # Snapshot Management (v1)
    # -------------------------------------------------------------------------

    @app.post("/v1/sessions/{session_id}/save")
    async def save_session_snapshot(session_id: str, request: Request) -> Any:
        """Save a session to an immutable on-disk snapshot."""
        payload = await _json_dict_or_empty(request)

        title = payload.get("title")
        description = payload.get("description")
        tags = payload.get("tags")
        if tags is not None and not isinstance(tags, list):
            raise HTTPException(status_code=400, detail="'tags' must be a list of strings.")

        transcript: list[dict[str, Any]] = []
        try:
            with _engine_lock:
                exported = engine.adapter.export_session(session_id)
                with _sessions_lock:
                    meta = _sessions.get(session_id)
                    transcript = list(meta.messages) if meta is not None else []
                cache_payload = export_hybrid_mamba_attention_static_cache(
                    cache=exported["past_key_values"],
                    current_pos=int(exported["current_pos"]),
                )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        store = _get_snapshot_store()
        try:
            manifest = store.create_snapshot(
                transcript=transcript,
                cache_payload=cache_payload,
                session={"max_seq_len": int(exported["max_seq_len"]), "current_pos": int(exported["current_pos"])},
                title=title if isinstance(title, str) else None,
                description=description if isinstance(description, str) else None,
                tags=[str(t) for t in tags] if isinstance(tags, list) else None,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return JSONResponse({"status": "saved", "snapshot_id": manifest.snapshot_id})

    @app.get("/v1/snapshots")
    async def list_snapshots() -> Any:
        store = _get_snapshot_store()
        snaps = [m.to_dict() for m in store.list_snapshots()]
        return JSONResponse({"snapshots": snaps})

    @app.get("/v1/snapshots/{snapshot_id}")
    async def get_snapshot(snapshot_id: str) -> Any:
        store = _get_snapshot_store()
        try:
            manifest = store.get_manifest(snapshot_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(manifest.to_dict())

    @app.patch("/v1/snapshots/{snapshot_id}")
    async def patch_snapshot(snapshot_id: str, request: Request) -> Any:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")

        title = payload.get("title")
        description = payload.get("description")
        tags = payload.get("tags")
        if tags is not None and not isinstance(tags, list):
            raise HTTPException(status_code=400, detail="'tags' must be a list of strings.")

        store = _get_snapshot_store()
        try:
            updated = store.patch_metadata(
                snapshot_id,
                title=title if isinstance(title, str) else None,
                description=description if isinstance(description, str) else None,
                tags=[str(t) for t in tags] if isinstance(tags, list) else None,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(updated.to_dict())

    @app.delete("/v1/snapshots/{snapshot_id}")
    async def delete_snapshot(snapshot_id: str) -> Any:
        store = _get_snapshot_store()
        try:
            store.delete_snapshot(snapshot_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse({"status": "deleted", "snapshot_id": snapshot_id})

    @app.post("/v1/snapshots/{snapshot_id}/load")
    async def load_snapshot(snapshot_id: str, request: Request) -> Any:
        payload = await _json_dict_or_empty(request)

        target_session_id = payload.get("session_id")
        if target_session_id is not None and not isinstance(target_session_id, str):
            raise HTTPException(status_code=400, detail="'session_id' must be a string.")
        force = bool(payload.get("force", False))
        if not target_session_id:
            target_session_id = f"sess_{uuid.uuid4().hex}"

        store = _get_snapshot_store()
        try:
            manifest, transcript, cache_payload = store.load_snapshot_payload(snapshot_id)
        except SnapshotCompatibilityError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        max_seq_len = int(manifest.session.get("max_seq_len") or default_max_seq_len)
        expected_pos = int(manifest.session.get("current_pos") or 0)
        restored_pos = expected_pos

        try:
            with _engine_lock:
                # Avoid accidental overwrite unless explicitly forced.
                try:
                    engine.adapter.get_session_info(target_session_id)
                    if not force:
                        raise HTTPException(
                            status_code=409,
                            detail=f"Session already exists: {target_session_id} (use force=true to overwrite).",
                        )
                    engine.adapter.close_session(target_session_id)
                except KeyError:
                    pass

                model = getattr(engine.adapter, "model", None)
                if model is None or not hasattr(model, "create_static_cache"):
                    raise HTTPException(status_code=500, detail="Adapter does not expose create_static_cache().")

                past_key_values = model.create_static_cache(batch_size=1, max_seq_len=max_seq_len)
                restored_pos = import_hybrid_mamba_attention_static_cache(
                    cache=past_key_values, payload=cache_payload
                )
                engine.adapter.restore_session(
                    cache_id=target_session_id,
                    past_key_values=past_key_values,
                    current_pos=restored_pos,
                    max_seq_len=max_seq_len,
                    next_token_logits=None,
                    overwrite=False,
                )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        with _sessions_lock:
            _sessions[target_session_id] = _HttpSession(max_seq_len=max_seq_len, messages=transcript)

        return JSONResponse(
            {
                "status": "loaded",
                "snapshot_id": snapshot_id,
                "session_id": target_session_id,
                "session": {
                    "current_pos": restored_pos,
                    "max_seq_len": max_seq_len,
                    "message_count": len(transcript),
                },
            }
        )

    # -------------------------------------------------------------------------
    # Chat Completions
    # -------------------------------------------------------------------------

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        payload = await request.json()

        req_model = payload.get("model")
        if req_model is not None and req_model != model_id:
            raise HTTPException(status_code=404, detail=f"Unknown model: {req_model}")

        chat_req = _parse_chat_request(payload, http_max_completion_tokens=http_max_completion_tokens)

        # Session chat: maintain server-side message history and only append delta tokens.
        if chat_req.session_id:
            session_id = chat_req.session_id
            # Ensure session exists.
            try:
                with _engine_lock:
                    sess_info = engine.adapter.get_session_info(session_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc

            try:
                current_pos = int(sess_info.get("current_pos", 0) or 0)
            except Exception:
                current_pos = 0

            with _sessions_lock:
                meta = _sessions.get(session_id)
                if meta is None:
                    # Session exists in adapter but not in HTTP store (e.g., server restarted).
                    meta = _HttpSession(max_seq_len=int(sess_info.get("max_seq_len", default_max_seq_len)))
                    _sessions[session_id] = meta

                # Safety: a non-empty KV cache with an empty HTTP transcript means the server cannot
                # correctly compute delta tokens to append. Proceeding would cause the model to ignore
                # new user input or append mismatched tokens.
                if current_pos > 0 and not meta.messages:
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "Session KV cache is non-empty but HTTP transcript is empty. "
                            "This indicates a corrupted/incomplete session state. "
                            "Start a new session or restore from a snapshot."
                        ),
                    )

                # Treat incoming messages as delta and append to stored history.
                incoming_raw = payload.get("messages")
                if isinstance(incoming_raw, list):
                    incoming_msgs = [m for m in incoming_raw if isinstance(m, dict)]

                    # Keep at most one leading system message in the stored transcript.
                    # The CLI may send the same system prompt every turn; accumulating
                    # duplicates wastes context and can degrade multi-turn coherence.
                    if incoming_msgs and incoming_msgs[0].get("role") == "system":
                        if meta.messages and isinstance(meta.messages[0], dict) and meta.messages[0].get("role") == "system":
                            # Never mutate the prompt prefix once the KV cache has advanced.
                            # Even a small change to the system message would invalidate the cached
                            # token stream and corrupt append-from slicing.
                            if current_pos <= 0:
                                meta.messages[0] = incoming_msgs[0]
                            # Drop the incoming system message to avoid duplicates.
                            incoming_msgs = incoming_msgs[1:]

                    meta.messages.extend(incoming_msgs)

                full_messages = list(meta.messages)

            # Build a new ChatRequest with full history messages.
            # Append-from position is the current KV cache position.
            chat_req = ChatRequest(
                messages=_parse_chat_request({"messages": full_messages}).messages,
                tools=chat_req.tools,
                tool_choice=chat_req.tool_choice,
                max_tokens=chat_req.max_tokens,
                temperature=chat_req.temperature,
                top_p=chat_req.top_p,
                stop=chat_req.stop,
                stream=chat_req.stream,
                stream_options=chat_req.stream_options,
                chat_template_kwargs=chat_req.chat_template_kwargs,
                reasoning_budget=chat_req.reasoning_budget,
                discard_thinking=chat_req.discard_thinking,
                stream_thinking=chat_req.stream_thinking,
                session_id=session_id,
                session_append_from_pos=current_pos,
                extra=chat_req.extra,
            )

        created = int(time.time())
        chatcmpl_id = f"chatcmpl-{uuid.uuid4().hex}"

        if chat_req.stream:
            await _try_acquire_semaphore()
            event_iter = _stream_chat_completions(
                engine=engine,
                chat_request=chat_req,
                model_id=model_id,
                created=created,
                chatcmpl_id=chatcmpl_id,
                sessions=_sessions,
                sessions_lock=_sessions_lock,
                request=request,
                http_semaphore=http_semaphore,
                semaphore_already_acquired=True,
            )
            return StreamingResponse(event_iter, media_type="text/event-stream")

        await _try_acquire_semaphore()
        try:
            result = await _run_with_disconnect_cancellation(request, engine.generate_chat(chat_req))
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            if http_semaphore is not None:
                http_semaphore.release()

        usage = result.get("usage")
        finish_reason = result.get("finish_reason") or "stop"
        content = result.get("content")
        tool_calls = result.get("tool_calls") or []
        raw_content = result.get("raw_content")

        message: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            message["content"] = None
            message["tool_calls"] = [_openai_tool_call(tc) for tc in tool_calls]

        # Persist assistant message to session history (non-streaming)
        if chat_req.session_id:
            # When discard_thinking=True, persist only the stripped content (no <think> blocks).
            # Otherwise, persist raw_content (with thinking) if available.
            if chat_req.discard_thinking:
                history_content = content if content is not None else ""
            else:
                if raw_content is not None:
                    history_content = raw_content
                else:
                    history_content = content if content is not None else ""
            history_msg: dict[str, Any] = {"role": "assistant", "content": history_content}
            if tool_calls:
                history_msg["content"] = None
                history_msg["tool_calls"] = [_openai_tool_call(tc) for tc in tool_calls]

            # Persist empty-string assistant messages too (never null).
            if tool_calls or isinstance(history_content, str):
                with _sessions_lock:
                    meta = _sessions.get(chat_req.session_id)
                    if meta is not None:
                        meta.messages.append(history_msg)

        resp: dict[str, Any] = {
            "id": chatcmpl_id,
            "object": "chat.completion",
            "created": created,
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
        }

        if isinstance(usage, Usage):
            resp["usage"] = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }

        return JSONResponse(resp)

    return app


def _sse(data: str) -> str:
    return f"data: {data}\n\n"


def _openai_tool_call(tc: ToolCall) -> dict[str, Any]:
    return {
        "id": tc.id,
        "type": "function",
        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
    }


def _openai_stream_tool_call(tc: ToolCall, *, index: int) -> dict[str, Any]:
    d = _openai_tool_call(tc)
    d["index"] = index
    return d


async def _stream_chat_completions(
    *,
    engine: ChatEngine,
    chat_request: ChatRequest,
    model_id: str,
    created: int,
    chatcmpl_id: str,
    sessions: dict[str, Any],
    sessions_lock: threading.Lock,
    request: Request,
    http_semaphore: asyncio.Semaphore | None,
    semaphore_already_acquired: bool = False,
) -> AsyncIterator[str]:
    if http_semaphore is not None and not semaphore_already_acquired:
        try:
            await asyncio.wait_for(http_semaphore.acquire(), timeout=0.001)
        except TimeoutError as exc:
            raise HTTPException(status_code=429, detail="Server is busy") from exc

    # Initial chunk announces the role.
    yield _sse(
        json.dumps(
            {
                "id": chatcmpl_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            },
            ensure_ascii=False,
        )
    )

    final_finish_reason: str | None = None
    final_usage: Usage | None = None
    final_timing: Timing | None = None
    assistant_text_parts: list[str] = []
    assistant_tool_calls: list[ToolCall] = []
    raw_content_for_history: str | None = None  # Raw content with thinking for discard_thinking=False
    cancelled = False

    try:
        async for event in engine.astream_chat(chat_request):
            from superlinear.engine.chat_types import DeltaEvent, ThinkingDeltaEvent, ErrorEvent, FinalEvent, ToolCallEvent

            # If the client disconnects mid-stream, stop consuming promptly.
            # The engine stream is cancelled/closed when this generator unwinds.
            if await request.is_disconnected():
                cancelled = True
                final_finish_reason = "cancelled"
                break

            if isinstance(event, DeltaEvent):
                if not event.text:
                    continue
                assistant_text_parts.append(event.text)
                yield _sse(
                    json.dumps(
                        {
                            "id": chatcmpl_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": event.text},
                                    "finish_reason": None,
                                }
                            ],
                        },
                        ensure_ascii=False,
                    )
                )
                continue

            if isinstance(event, ThinkingDeltaEvent):
                if not event.text:
                    continue
                yield _sse(
                    json.dumps(
                        {
                            "id": chatcmpl_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"thinking": event.text},
                                    "finish_reason": None,
                                }
                            ],
                        },
                        ensure_ascii=False,
                    )
                )
                continue

            if isinstance(event, ToolCallEvent):
                # Tool call detected - set finish reason to tool_calls
                final_finish_reason = "tool_calls"
                assistant_tool_calls = list(event.tool_calls)
                yield _sse(
                    json.dumps(
                        {
                            "id": chatcmpl_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            _openai_stream_tool_call(tc, index=i)
                                            for i, tc in enumerate(event.tool_calls)
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        },
                        ensure_ascii=False,
                    )
                )
                continue

            if isinstance(event, FinalEvent):
                # Don't override tool_calls finish reason
                if final_finish_reason != "tool_calls":
                    final_finish_reason = event.finish_reason
                # Capture raw content if provided (for discard_thinking=False sessions)
                if event.raw_content is not None:
                    raw_content_for_history = event.raw_content
                final_usage = event.usage
                final_timing = event.timing
                continue

            if isinstance(event, ErrorEvent):
                yield _sse(
                    json.dumps(
                        {
                            "error": {
                                "message": event.message,
                                "type": "server_error",
                                "param": None,
                                "code": None,
                            }
                        },
                        ensure_ascii=False,
                    )
                )
                final_finish_reason = "error"
                break
    except asyncio.CancelledError:
        cancelled = True
        final_finish_reason = "cancelled"
        raise
    except ValueError as exc:
        yield _sse(
            json.dumps(
                {
                    "error": {
                        "message": str(exc),
                        "type": "invalid_request_error",
                        "param": None,
                        "code": None,
                    }
                },
                ensure_ascii=False,
            )
        )
        final_finish_reason = "error"
    except Exception as exc:
        yield _sse(
            json.dumps(
                {
                    "error": {
                        "message": str(exc),
                        "type": "server_error",
                        "param": None,
                        "code": None,
                    }
                },
                ensure_ascii=False,
            )
        )
        final_finish_reason = "error"
    finally:
        if http_semaphore is not None:
            http_semaphore.release()

        # Persist assistant message to session history (best-effort).
        # Important: on cancelled/incomplete streams, persist an *empty string* (not null)
        # so HTTP transcript stays aligned with the adapter session KV state.
        if chat_request.session_id:
            if chat_request.discard_thinking:
                text = "".join(assistant_text_parts) if assistant_text_parts else ""
                history_content: str | None = text
            else:
                if raw_content_for_history is not None:
                    history_content = raw_content_for_history
                else:
                    history_content = "".join(assistant_text_parts) if assistant_text_parts else ""

            msg: dict[str, Any] = {
                "role": "assistant",
                "content": history_content,
            }
            if assistant_tool_calls:
                msg["content"] = None
                msg["tool_calls"] = [_openai_tool_call(tc) for tc in assistant_tool_calls]

            # Avoid persisting pure-null assistant messages.
            should_persist = True
            if not assistant_tool_calls and history_content is None:
                should_persist = False

            if should_persist:
                with sessions_lock:
                    meta = sessions.get(chat_request.session_id)
                    if meta is not None and hasattr(meta, "messages"):
                        meta.messages.append(msg)  # type: ignore[attr-defined]

        # Terminal chunk + DONE (skip if the request was cancelled/disconnected).
        if not cancelled:
            if final_finish_reason is None:
                final_finish_reason = "stop"

            terminal: dict[str, Any] = {
                "id": chatcmpl_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [{"index": 0, "delta": {}, "finish_reason": final_finish_reason}],
            }
            if isinstance(final_usage, Usage):
                terminal["usage"] = {
                    "prompt_tokens": final_usage.prompt_tokens,
                    "completion_tokens": final_usage.completion_tokens,
                    "total_tokens": final_usage.total_tokens,
                }
            if isinstance(final_timing, Timing):
                terminal["x_superlinear_timing"] = {
                    "prefill_s": final_timing.prefill_s,
                    "decode_s": final_timing.decode_s,
                    "total_s": final_timing.total_s,
                    "tok_per_s": final_timing.tok_per_s,
                }
            yield _sse(json.dumps(terminal, ensure_ascii=False))
            yield "data: [DONE]\n\n"


def _parse_chat_request(payload: Any, *, http_max_completion_tokens: int | None = None) -> ChatRequest:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object.")

    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise HTTPException(status_code=400, detail="'messages' must be a non-empty list.")

    messages: list[ChatMessage] = []
    for msg in raw_messages:
        if not isinstance(msg, dict):
            raise HTTPException(status_code=400, detail="Each message must be an object.")

        role = msg.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise HTTPException(status_code=400, detail=f"Invalid message role: {role!r}.")

        content = _coerce_content(msg.get("content"))

        tool_call_id = msg.get("tool_call_id") if role == "tool" else None
        tool_calls: list[ToolCall] = []

        if role == "assistant" and msg.get("tool_calls") is not None:
            raw_tool_calls = msg.get("tool_calls")
            if not isinstance(raw_tool_calls, list):
                raise HTTPException(status_code=400, detail="'tool_calls' must be a list.")

            for tc in raw_tool_calls:
                tool_calls.append(_parse_assistant_tool_call(tc))

        messages.append(
            ChatMessage(
                role=role,
                content=content,
                tool_calls=tool_calls,
                tool_call_id=tool_call_id,
            )
        )

    tools = payload.get("tools") or []
    if tools is None:
        tools = []
    if not isinstance(tools, list):
        raise HTTPException(status_code=400, detail="'tools' must be a list.")

    tool_choice = payload.get("tool_choice")

    max_tokens = payload.get("max_tokens")
    max_completion_tokens = payload.get("max_completion_tokens")

    if max_tokens is None and max_completion_tokens is None:
        max_tokens = 4096
    elif max_tokens is not None and max_completion_tokens is not None:
        try:
            if int(max_tokens) != int(max_completion_tokens):
                raise HTTPException(
                    status_code=400,
                    detail="'max_tokens' and 'max_completion_tokens' must match when both are provided.",
                )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail="'max_tokens' and 'max_completion_tokens' must be integers.",
            ) from exc
    elif max_completion_tokens is not None:
        max_tokens = max_completion_tokens

    try:
        max_tokens = int(max_tokens)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="'max_tokens' must be an integer.") from exc
    if max_tokens <= 0:
        raise HTTPException(status_code=400, detail="'max_tokens' must be > 0.")

    if http_max_completion_tokens is not None and max_tokens > http_max_completion_tokens:
        raise HTTPException(
            status_code=400,
            detail=f"'max_tokens' too large: {max_tokens} (cap={http_max_completion_tokens}).",
        )

    try:
        temperature = float(payload.get("temperature", 0.1) or 0.1)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="'temperature' must be a number.") from exc

    try:
        top_p = float(payload.get("top_p", 0.95) or 0.95)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="'top_p' must be a number.") from exc

    stop = payload.get("stop") or []
    if isinstance(stop, str):
        stop = [stop]
    if not isinstance(stop, list):
        raise HTTPException(status_code=400, detail="'stop' must be a string or list of strings.")
    stop = [s for s in stop if isinstance(s, str)]

    stream = bool(payload.get("stream", False))

    stream_options = payload.get("stream_options") or {}
    if stream_options is None:
        stream_options = {}
    if not isinstance(stream_options, dict):
        raise HTTPException(status_code=400, detail="'stream_options' must be an object.")

    try:
        flush_every_n_tokens = int(stream_options.get("flush_every_n_tokens", 8))
        flush_every_ms = int(stream_options.get("flush_every_ms", 50))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="'stream_options.flush_every_n_tokens' and 'stream_options.flush_every_ms' must be integers.",
        ) from exc

    # Parse chat_template_kwargs (optional, vLLM-compatible)
    chat_template_kwargs = payload.get("chat_template_kwargs")
    if chat_template_kwargs is not None and not isinstance(chat_template_kwargs, dict):
        raise HTTPException(status_code=400, detail="'chat_template_kwargs' must be an object.")

    # Parse reasoning_budget (optional, Superlinear-specific)
    reasoning_budget = payload.get("reasoning_budget")
    if reasoning_budget is not None:
        try:
            reasoning_budget = int(reasoning_budget)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail="'reasoning_budget' must be an integer.") from exc
        if reasoning_budget <= 0:
            raise HTTPException(status_code=400, detail="'reasoning_budget' must be > 0.")

    # Parse discard_thinking (optional, Superlinear-specific)
    discard_thinking = payload.get("discard_thinking")
    if discard_thinking is not None and not isinstance(discard_thinking, bool):
        raise HTTPException(status_code=400, detail="'discard_thinking' must be a boolean.")

    # Parse stream_thinking (optional, Superlinear-specific)
    stream_thinking = payload.get("stream_thinking")
    if stream_thinking is not None and not isinstance(stream_thinking, bool):
        raise HTTPException(status_code=400, detail="'stream_thinking' must be a boolean.")

    # Parse session_id (optional, for stateful chat)
    session_id = payload.get("session_id")
    if session_id is not None and not isinstance(session_id, str):
        raise HTTPException(status_code=400, detail="'session_id' must be a string.")

    # Parse extra (optional, engine-specific)
    extra = payload.get("extra")
    if extra is None:
        extra = {}
    if not isinstance(extra, dict):
        raise HTTPException(status_code=400, detail="'extra' must be an object.")

    # Convenience alias: allow top-level repetition_detection to be passed through.
    repetition_detection = payload.get("repetition_detection")
    if repetition_detection is not None and "repetition_detection" not in extra:
        extra = dict(extra)
        extra["repetition_detection"] = repetition_detection

    return ChatRequest(
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
        stream=stream,
        stream_options=StreamOptions(
            flush_every_n_tokens=flush_every_n_tokens,
            flush_every_ms=flush_every_ms,
        ),
        chat_template_kwargs=chat_template_kwargs,
        reasoning_budget=reasoning_budget,
        discard_thinking=discard_thinking,
        stream_thinking=stream_thinking,
        session_id=session_id,
        extra=extra,
    )


def _coerce_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content

    # Minimal support for OpenAI "content parts" format (text-only).
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") != "text":
                continue
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)

    raise HTTPException(status_code=400, detail="Unsupported message content type.")


def _parse_assistant_tool_call(tc: Any) -> ToolCall:
    if not isinstance(tc, dict):
        raise HTTPException(status_code=400, detail="Each tool_call must be an object.")

    fn = tc.get("function")
    if not isinstance(fn, dict):
        raise HTTPException(status_code=400, detail="tool_call.function must be an object.")

    name = fn.get("name")
    if not isinstance(name, str) or not name:
        raise HTTPException(status_code=400, detail="tool_call.function.name must be a string.")

    arguments = fn.get("arguments")
    args_dict: dict[str, Any] = {}
    if isinstance(arguments, str) and arguments.strip():
        try:
            parsed = json.loads(arguments)
            if isinstance(parsed, dict):
                args_dict = parsed
        except Exception:
            # Best-effort fallback: preserve raw payload under a reserved key.
            args_dict = {"__raw__": arguments}
    elif isinstance(arguments, dict):
        args_dict = arguments

    tool_call_id = tc.get("id")
    if not isinstance(tool_call_id, str) or not tool_call_id:
        tool_call_id = f"call_{uuid.uuid4().hex}"

    return ToolCall(id=tool_call_id, name=name, arguments=args_dict)
