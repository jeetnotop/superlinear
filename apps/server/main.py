"""Superlinear inference server entrypoint (FastAPI + OpenAI-style Chat Completions).

Example:
    python -m apps.server.main --model superlinear-exp-v0.1 --host 0.0.0.0 --port 8787
"""

from __future__ import annotations

import argparse
import os
import time
import uuid
from typing import Any

from apps.server.app import create_app
from superlinear.engine.chat_engine import ChatEngine, EngineConfig
from superlinear.engine.adapters.superlinear import SuperlinearAdapter


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Superlinear inference server")
    p.add_argument("--model", required=True, help="Model path or HF repo id")
    p.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=8787, help="Bind port (default: 8787)")

    p.add_argument("--device", default="cuda", help="Device / device_map (default: cuda)")
    p.add_argument("--dtype", default="bfloat16", help="Torch dtype: float16|bfloat16|float32 (default: bfloat16)")
    p.add_argument(
        "--backend",
        default="custom",
        choices=["custom", "hf"],
        help="Generation backend (default: custom)",
    )
    thinking_group = p.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        help="Enable <think> reasoning tokens (default: on)",
    )
    thinking_group.add_argument(
        "--disable-thinking",
        dest="enable_thinking",
        action="store_false",
        help="Disable <think> reasoning tokens",
    )
    p.set_defaults(enable_thinking=None)

    discard_group = p.add_mutually_exclusive_group()
    discard_group.add_argument(
        "--discard-thinking",
        dest="discard_thinking",
        action="store_true",
        help="Discard <think>...</think> from committed session state (default: on)",
    )
    discard_group.add_argument(
        "--keep-thinking",
        dest="discard_thinking",
        action="store_false",
        help="Keep <think>...</think> in committed session state",
    )
    p.set_defaults(discard_thinking=None)
    p.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=1_048_576,
        help="Reject prompts longer than this many tokens",
    )
    p.add_argument(
        "--max-tool-calls",
        type=int,
        default=8,
        help="Max tool calls per turn",
    )

    p.add_argument(
        "--http-max-concurrency",
        type=int,
        default=0,
        help="Max in-flight /v1/chat/completions requests (0 = unlimited)",
    )
    p.add_argument(
        "--http-max-completion-tokens",
        type=int,
        default=0,
        help="Reject requests with max_tokens above this cap (0 = unlimited)",
    )

    warmup_group = p.add_mutually_exclusive_group()
    warmup_group.add_argument(
        "--warmup",
        dest="warmup",
        action="store_true",
        help="Run blocking CUDA-graph warmup after model load (default: on)",
    )
    warmup_group.add_argument(
        "--no-warmup",
        dest="warmup",
        action="store_false",
        help="Disable startup warmup",
    )
    p.set_defaults(warmup=True)
    p.add_argument(
        "--warmup-max-seq-len",
        type=int,
        default=131_072,
        help="Warmup session max_seq_len (default: 131072)",
    )
    p.add_argument(
        "--warmup-prefill-tokens",
        type=int,
        default=2048,
        help="Warmup prefill length in tokens (approx; default: 2048)",
    )
    p.add_argument(
        "--warmup-decode-tokens",
        type=int,
        default=16,
        help="Warmup decode steps (default: 16)",
    )

    p.add_argument("--chunk-size", type=int, default=8192,
                   help="Chunk size for chunked prefill (default: 8192)")
    p.add_argument("--attn-implementation", default="block-span-gqa")
    p.add_argument("--decode-kernel", default="staged-gqa")
    p.add_argument("--disable-cuda-graph", action="store_true")
    p.add_argument("--disable-shared-fused-moe", action="store_true")
    p.add_argument("--reload", action="store_true", help="Enable uvicorn reload (dev only)")
    return p.parse_args()


def _dtype_from_string(dtype: str) -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch is required to run the server.") from exc

    dt = dtype.strip().lower()
    if dt in {"fp16", "float16", "half"}:
        return torch.float16
    if dt in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dt in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype!r}")


def _run_blocking_warmup(
    *,
    adapter: SuperlinearAdapter,
    chunk_size: int,
    warmup_max_seq_len: int,
    warmup_prefill_tokens: int,
    warmup_decode_tokens: int,
) -> None:
    if warmup_prefill_tokens <= 0 and warmup_decode_tokens <= 0:
        print("[warmup] skipped (warmup token counts are 0)", flush=True)
        return

    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch is required for warmup") from exc

    tok = adapter.tokenizer
    if tok is None:
        raise RuntimeError("Adapter tokenizer not available for warmup")

    # Choose a non-EOS token id for a stable warmup prompt.
    warm_ids = tok.encode("Hello", add_special_tokens=False)
    warm_token_id = None
    if warm_ids:
        warm_token_id = int(warm_ids[0])
    elif tok.bos_token_id is not None:
        warm_token_id = int(tok.bos_token_id)
    elif tok.pad_token_id is not None:
        warm_token_id = int(tok.pad_token_id)
    else:
        warm_token_id = 1

    session_id = f"__warmup__{uuid.uuid4().hex[:8]}"
    t0 = time.time()
    print(
        "[warmup] starting: "
        f"session_max_seq_len={warmup_max_seq_len} "
        f"prefill_tokens={warmup_prefill_tokens} "
        f"decode_tokens={warmup_decode_tokens} "
        f"chunk_size={chunk_size}",
        flush=True,
    )

    try:
        input_ids = None
        if warmup_prefill_tokens > 0:
            input_ids = torch.full(
                (1, int(warmup_prefill_tokens)),
                fill_value=warm_token_id,
                dtype=torch.long,
                device=adapter.model.device,
            )

        adapter.create_session(
            cache_id=session_id,
            input_ids=input_ids,
            max_seq_len=int(warmup_max_seq_len),
            chunk_size=int(chunk_size),
        )

        if warmup_decode_tokens > 0:
            n = 0
            for _ in adapter.stream_generate_session(
                cache_id=session_id,
                max_new_tokens=int(warmup_decode_tokens),
                temperature=0.0,
            ):
                n += 1
            print(f"[warmup] decode_steps_executed={n}", flush=True)

        if torch.cuda.is_available() and str(adapter.device).startswith("cuda"):
            torch.cuda.synchronize()
    finally:
        try:
            adapter.close_session(session_id)
        except Exception:
            pass

    dt = time.time() - t0
    print(f"[warmup] done in {dt:.2f}s", flush=True)


def main() -> None:
    args = _parse_args()

    adapter = SuperlinearAdapter()
    print(
        "[server] loading model... "
        f"model={args.model!r} device={args.device!r} dtype={args.dtype!r} "
        f"chunk_size={int(args.chunk_size)} cuda_graph={'off' if args.disable_cuda_graph else 'on'}",
        flush=True,
    )
    print(
        "[server] note: when started detached, Transformers progress bars may be suppressed; "
        "use `spl server start --foreground ...` to see interactive shard-loading progress.",
        flush=True,
    )
    adapter.load(
        args.model,
        device=args.device,
        dtype=_dtype_from_string(args.dtype),
        chunk_size=args.chunk_size,
        attn_implementation=args.attn_implementation,
        decode_kernel=args.decode_kernel,
        enable_cuda_graph=not args.disable_cuda_graph,
        enable_shared_fused_moe=not args.disable_shared_fused_moe,
        trust_remote_code=True,
    )
    print("[server] model loaded", flush=True)

    if bool(args.warmup):
        print(
            "[server] warmup enabled: "
            f"warmup_max_seq_len={int(args.warmup_max_seq_len)} "
            f"warmup_prefill_tokens={int(args.warmup_prefill_tokens)} "
            f"warmup_decode_tokens={int(args.warmup_decode_tokens)}",
            flush=True,
        )
        _run_blocking_warmup(
            adapter=adapter,
            chunk_size=int(args.chunk_size),
            warmup_max_seq_len=int(args.warmup_max_seq_len),
            warmup_prefill_tokens=int(args.warmup_prefill_tokens),
            warmup_decode_tokens=int(args.warmup_decode_tokens),
        )
    else:
        print("[server] warmup disabled", flush=True)

    engine = ChatEngine(
        adapter,
        config=EngineConfig(
            default_backend=args.backend,
            enable_thinking=True if args.enable_thinking is None else bool(args.enable_thinking),
            discard_thinking=True if args.discard_thinking is None else bool(args.discard_thinking),
            max_prompt_tokens=args.max_prompt_tokens,
            max_tool_calls_per_turn=args.max_tool_calls,
        ),
    )

    model_id = os.path.basename(args.model.rstrip("/")) or "superlinear"
    app = create_app(
        engine=engine,
        model_id=model_id,
        http_max_concurrency=None if args.http_max_concurrency <= 0 else int(args.http_max_concurrency),
        http_max_completion_tokens=None
        if args.http_max_completion_tokens <= 0
        else int(args.http_max_completion_tokens),
    )

    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("uvicorn is required to run the server.") from exc

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
