"""Adapter for the Superlinear model family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator

from .base import BaseAdapter

if TYPE_CHECKING:
    import torch


# =============================================================================
# Session State
# =============================================================================


@dataclass
class _SessionState:
    """Internal state for a stateful KV-cache session."""

    past_key_values: Any
    current_pos: int
    max_seq_len: int
    next_token_logits: torch.Tensor | None


@dataclass(frozen=True)
class _SessionCheckpoint:
    """Lightweight, in-memory checkpoint for a session.

    Notes:
    - We intentionally do NOT copy the full attention KV tensors. Restoring
      to an earlier `current_pos` makes any KV written beyond that position
      unreachable and safe to ignore/overwrite later.
    - We MUST capture rolling Mamba state (conv/ssm) which is mutated during
      decode and is not position-sliced.
    """

    cache_obj_id: int
    current_pos: int
    next_token_logits: torch.Tensor | None
    conv_states: torch.Tensor | None
    ssm_states: torch.Tensor | None


# =============================================================================
# Adapter
# =============================================================================


class SuperlinearAdapter(BaseAdapter):
    """
    Adapter for Superlinear models.

    All generation methods operate on token IDs (torch.Tensor), not strings.
    Tokenization/detokenization should be handled at a higher layer (e.g., server).

    Thread Safety:
        This adapter is NOT thread-safe. Do not call generation methods concurrently
        from multiple threads on the same adapter instance. For concurrent requests,
        use separate adapter instances or external synchronization.

    Provides three generation modes:

    1. **Stateless with custom backend** (default):
       - `generate()` / `stream_generate()` with `backend="custom"`
       - Uses chunked prefill + manual decode loop with static KV cache.
       - Best for long contexts (256k+); cache is discarded after each call.

    2. **Stateless with HF backend**:
       - `generate()` / `stream_generate()` with `backend="hf"`
       - Uses Transformers' built-in `model.generate()`.
       - Simpler, good for shorter prompts; cache is discarded after each call.

    3. **Stateful sessions**:
       - `create_session()` → `append_to_session()` → `stream_generate_session()`
       - Persists KV cache across calls for multi-turn chat without re-prefill.
       - Best for interactive, multi-turn conversations at long context.

    Example (stateless):
        >>> adapter = SuperlinearAdapter()
        >>> adapter.load("path/to/superlinear-exp-v0.1")
        >>> input_ids = tokenizer.encode("Hello!", return_tensors="pt").to("cuda")
        >>> for token_id in adapter.stream_generate(input_ids, max_new_tokens=50):
        ...     print(tokenizer.decode(token_id), end="", flush=True)

    Example (stateful session):
        >>> adapter.create_session(cache_id="chat1", max_seq_len=8192)
        >>> input_ids = tokenizer.encode("<user>Hi!</user>", return_tensors="pt").to("cuda")
        >>> adapter.append_to_session(cache_id="chat1", input_ids=input_ids)
        >>> for token_id in adapter.stream_generate_session(cache_id="chat1"):
        ...     print(tokenizer.decode(token_id), end="", flush=True)
    """

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._model_path: str | None = None
        self._device: str = "cuda"
        self._dtype = None
        self._chunk_size: int = 8192
        self._sessions: dict[str, _SessionState] = {}

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def model(self):
        """Access the underlying model (for advanced use cases)."""
        return self._model

    @property
    def tokenizer(self):
        """Access the tokenizer (for encoding/decoding at higher layers)."""
        return self._tokenizer

    @property
    def device(self) -> str:
        """Device the model is loaded on."""
        return self._device

    @property
    def eos_token_id(self) -> int | None:
        """EOS token ID, or None if tokenizer not loaded."""
        if self._tokenizer is None:
            return None
        return self._tokenizer.eos_token_id

    @property
    def think_open_token_ids(self) -> list[int] | None:
        """Token IDs for '<think>' tag, or None if tokenizer not loaded."""
        if self._tokenizer is None:
            return None
        return self._tokenizer.encode("<think>", add_special_tokens=False)

    @property
    def think_close_token_ids(self) -> list[int] | None:
        """Token IDs for '</think>' tag, or None if tokenizer not loaded."""
        if self._tokenizer is None:
            return None
        return self._tokenizer.encode("</think>", add_special_tokens=False)

    @property
    def model_info(self) -> dict[str, Any]:
        """Return model metadata."""
        return {
            "model_path": self._model_path,
            "device": self._device,
            "dtype": str(self._dtype),
            "loaded": self._model is not None,
            "active_sessions": len(self._sessions),
        }

    # -------------------------------------------------------------------------
    # Loading / Unloading
    # -------------------------------------------------------------------------

    def load(self, model_path: str, **kwargs) -> None:
        """Load a Superlinear model and tokenizer.

        Args:
            model_path: Path to the model (local or HF hub).
            device: Device to load the model on (default: "cuda").
            dtype: Torch dtype (default: torch.float16).
            attn_implementation: Attention impl (default: "block-span-gqa").
            decode_kernel: Decode kernel (default: "staged-gqa").
            enable_cuda_graph: Enable CUDA graphs (default: True).
            enable_shared_fused_moe: Enable shared fused MoE (default: True).
            chunk_size: Default chunk size for chunked prefill (default: 8192).
            **kwargs: Additional kwargs passed to from_pretrained().
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._model_path = model_path
        self._device = kwargs.pop("device", "cuda")
        self._dtype = kwargs.pop("dtype", torch.float16)
        self._chunk_size = int(kwargs.pop("chunk_size", 8192))

        # Model-specific config options
        attn_implementation = kwargs.pop("attn_implementation", "block-span-gqa")
        decode_kernel = kwargs.pop("decode_kernel", "staged-gqa")
        enable_cuda_graph = kwargs.pop("enable_cuda_graph", True)
        enable_shared_fused_moe = kwargs.pop("enable_shared_fused_moe", True)

        # HF remote code flag (callers may pass it; don't pass twice).
        trust_remote_code = kwargs.pop("trust_remote_code", True)

        # Separate span_attention_* kwargs
        span_kwargs = {k: v for k, v in kwargs.items() if k.startswith("span_attention_")}
        other_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("span_attention_")}

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            _attn_implementation=attn_implementation,
            decode_kernel=decode_kernel,
            enable_cuda_graph=enable_cuda_graph,
            enable_shared_fused_moe=enable_shared_fused_moe,
            torch_dtype=self._dtype,
            device_map=self._device,
            trust_remote_code=trust_remote_code,
            **span_kwargs,
            **other_kwargs,
        )

        # Validate model type
        model_type = getattr(self._model.config, "model_type", None)
        if model_type != "superlinear-exp":
            raise ValueError(
                f"SuperlinearAdapter requires model_type='superlinear-exp', got '{model_type}'"
            )

    def unload(self) -> None:
        """Unload the model and free GPU memory."""
        import gc
        import torch

        # Close all sessions first
        for cache_id in list(self._sessions.keys()):
            self.close_session(cache_id)

        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        self._sessions.clear()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Stateless Generation (Public API)
    # -------------------------------------------------------------------------

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        stop_token_ids: list[int] | None = None,
        backend: str = "custom",
        **kwargs,
    ) -> torch.Tensor:
        """Generate tokens (non-streaming, stateless).

        Args:
            input_ids: Input token IDs, shape (batch_size, seq_len). Batch size must be 1.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            stop_token_ids: Stop generation when any of these token IDs are produced.
            backend: "custom" (chunked prefill + manual decode) or "hf" (HF generate).
            **kwargs: Additional generation kwargs.

        Returns:
            Generated token IDs (excluding input), shape (num_generated,).
        """
        self._ensure_loaded()
        if input_ids.shape[0] != 1:
            raise ValueError(f"Batch size must be 1, got {input_ids.shape[0]}")

        if backend == "hf":
            return self._generate_hf(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop_token_ids=stop_token_ids,
                **kwargs,
            )
        if backend == "custom":
            import torch
            tokens = list(
                self._stream_generate_custom(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    stop_token_ids=stop_token_ids,
                    **kwargs,
                )
            )
            if not tokens:
                return torch.tensor([], dtype=torch.long, device=input_ids.device)
            return torch.cat(tokens, dim=0)

        raise ValueError(f"Unknown backend: {backend!r}. Expected 'custom' or 'hf'.")

    def stream_generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        stop_token_ids: list[int] | None = None,
        backend: str = "custom",
        reasoning_budget: int | None = None,
        enable_thinking: bool = True,
        **kwargs,
    ) -> Iterator[torch.Tensor]:
        """Stream generated tokens one at a time (stateless).

        Args:
            input_ids: Input token IDs, shape (batch_size, seq_len). Batch size must be 1.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            stop_token_ids: Stop generation when any of these token IDs are produced.
            backend: "custom" (chunked prefill + manual decode) or "hf" (HF generate).
            reasoning_budget: Max tokens for thinking phase. When exceeded, '</think>' is
                injected and generation continues for the answer. None = no limit.
                Only active when enable_thinking=True.
            enable_thinking: Whether thinking mode is enabled. If False, reasoning_budget
                is ignored.
            **kwargs: Additional generation kwargs.

        Yields:
            Generated token IDs, each shape (1,).
        """
        self._ensure_loaded()
        if input_ids.shape[0] != 1:
            raise ValueError(f"Batch size must be 1, got {input_ids.shape[0]}")

        # Only enforce reasoning_budget when thinking is enabled
        effective_budget = reasoning_budget if enable_thinking else None

        if backend == "hf":
            yield from self._stream_generate_hf(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop_token_ids=stop_token_ids,
                reasoning_budget=effective_budget,
                **kwargs,
            )
            return
        if backend == "custom":
            yield from self._stream_generate_custom(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop_token_ids=stop_token_ids,
                reasoning_budget=effective_budget,
                **kwargs,
            )
            return

        raise ValueError(f"Unknown backend: {backend!r}. Expected 'custom' or 'hf'.")

    # -------------------------------------------------------------------------
    # Stateful Sessions (Public API)
    # -------------------------------------------------------------------------

    def create_session(
        self,
        *,
        cache_id: str,
        input_ids: torch.Tensor | None = None,
        max_seq_len: int | None = None,
        reserve_tokens: int = 1024,
        chunk_size: int | None = None,
    ) -> None:
        """Create a stateful KV-cache session.

        The session persists `past_key_values` and position state, allowing you
        to append follow-up tokens and continue decoding without re-prefilling.

        Args:
            cache_id: Unique identifier for this session.
            input_ids: Optional initial tokens to prefill, shape (1, seq_len).
            max_seq_len: Maximum sequence length for the cache. If None, computed
                from input length + reserve_tokens.
            reserve_tokens: Extra tokens to reserve if max_seq_len is auto-computed.
            chunk_size: Chunk size for prefill (default: adapter-level setting).
        """
        self._ensure_loaded()
        if chunk_size is None:
            chunk_size = self._chunk_size

        # Ensure no stale transient state (e.g. from prior stateless calls)
        # interferes with cache allocation/prefill.
        self._reset_transient_model_state()

        if cache_id in self._sessions:
            raise ValueError(f"Session already exists: {cache_id}")

        # Determine input length
        input_len = 0
        if input_ids is not None:
            if input_ids.shape[0] != 1:
                raise ValueError(f"Batch size must be 1, got {input_ids.shape[0]}")
            input_ids = input_ids.to(self._model.device)
            input_len = input_ids.shape[1]

        # Determine cache size
        if max_seq_len is None:
            max_seq_len = input_len + reserve_tokens + 1
        if max_seq_len < input_len + 1:
            raise ValueError(
                f"max_seq_len ({max_seq_len}) must be >= input_len+1 ({input_len + 1})"
            )

        # Allocate static cache
        past_key_values = self._model.create_static_cache(batch_size=1, max_seq_len=max_seq_len)

        # Empty session (no input)
        if input_ids is None or input_len == 0:
            self._sessions[cache_id] = _SessionState(
                past_key_values=past_key_values,
                current_pos=0,
                max_seq_len=max_seq_len,
                next_token_logits=None,
            )
            return

        # Prefill with input
        try:
            outputs, past_key_values, current_pos = self._prefill_tokens(
                input_ids,
                past_key_values=past_key_values,
                start_pos=0,
                chunk_size=chunk_size,
            )
            if outputs is None:
                raise RuntimeError("Prefill produced no outputs.")

            self._sessions[cache_id] = _SessionState(
                past_key_values=past_key_values,
                current_pos=current_pos,
                max_seq_len=max_seq_len,
                next_token_logits=outputs.logits[:, -1, :].detach(),
            )
        finally:
            self._reset_transient_model_state()

    def append_to_session(
        self,
        *,
        cache_id: str,
        input_ids: torch.Tensor,
        chunk_size: int | None = None,
    ) -> None:
        """Append tokens to an existing session (chunked prefill).

        Note: The model's static cache does not advance its internal prefill
        cursor during decode. For session workflows (decode → append prefill),
        we must synchronize the cache cursor to the session position before
        running chunked prefill.

        Args:
            cache_id: Session identifier.
            input_ids: Tokens to append, shape (1, seq_len).
            chunk_size: Chunk size for prefill (default: adapter-level setting).
        """
        self._ensure_loaded()
        if chunk_size is None:
            chunk_size = self._chunk_size

        # Prevent transient state from prior calls from leaking into this prefill.
        self._reset_transient_model_state()

        session = self._sessions.get(cache_id)
        if session is None:
            raise KeyError(f"Unknown session: {cache_id}")
        if input_ids.shape[0] != 1:
            raise ValueError(f"Batch size must be 1, got {input_ids.shape[0]}")

        input_ids = input_ids.to(self._model.device)
        new_len = input_ids.shape[1]

        if session.current_pos + new_len >= session.max_seq_len:
            raise ValueError(
                f"Session {cache_id} would exceed max_seq_len={session.max_seq_len}. "
                f"Need {session.current_pos + new_len + 1}. Create a new session with larger max_seq_len."
            )

        try:
            # Critical: make the static cache's prefill cursor match the session
            # position (decode does not increment _layer_seen_tokens).
            self._sync_static_cache_seen_tokens(session.past_key_values, session.current_pos)

            outputs, past_key_values, current_pos = self._prefill_tokens(
                input_ids,
                past_key_values=session.past_key_values,
                start_pos=session.current_pos,
                chunk_size=chunk_size,
            )
            if outputs is None:
                return

            session.past_key_values = past_key_values
            session.current_pos = current_pos
            session.next_token_logits = outputs.logits[:, -1, :].detach()
        finally:
            self._reset_transient_model_state()

    def stream_generate_session(
        self,
        *,
        cache_id: str,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        stop_token_ids: list[int] | None = None,
    ) -> Iterator[torch.Tensor]:
        """Decode from a session, streaming tokens.

        The session state (cache, position) is updated as tokens are generated,
        so you can call this multiple times for multi-turn conversation.

        The session state (cache, position) is persisted as tokens are generated.
        If you stop early (break out of the loop), the generator finalizer will
        still persist whatever progress has been made.

        Note: When generation stops early (EOS or stop token), the stop token is
        yielded but not written to the KV cache. The session state reflects all
        tokens up to (but not including) the stop token's KV. This is intentional
        since you typically won't continue generating after a stop token.

        Args:
            cache_id: Session identifier.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            stop_token_ids: Stop when any of these token IDs are produced.

        Yields:
            Generated token IDs, each shape (1,).
        """
        self._ensure_loaded()

        session = self._sessions.get(cache_id)
        if session is None:
            raise KeyError(f"Unknown session: {cache_id}")
        if session.next_token_logits is None:
            raise RuntimeError(
                f"Session {cache_id} has no prefill state. "
                "Call append_to_session() with some tokens before decoding."
            )

        import torch

        stop_token_ids = set(stop_token_ids or [])
        eos_token_id = self._tokenizer.eos_token_id

        past_key_values = session.past_key_values
        current_pos = session.current_pos
        next_token_logits = session.next_token_logits

        # Persist state even if the generator is closed early (e.g., user breaks
        # out of the loop). This prevents `current_pos` and `past_key_values`
        # from getting out of sync across notebook re-runs.
        try:
            for _ in range(max_new_tokens):
                # Sample next token
                next_token = self._sample_token(next_token_logits, temperature)
                token_id = next_token.item()

                yield next_token.squeeze(0)  # Shape (1,)

                # Check stop conditions (intentionally do not write stop token KV)
                if token_id == eos_token_id:
                    break
                if token_id in stop_token_ids:
                    break

                # Check capacity (we will write at `current_pos`)
                if current_pos >= session.max_seq_len:
                    raise ValueError(
                        f"Session {cache_id} exceeded max_seq_len={session.max_seq_len} during decode."
                    )

                # Forward pass for next token; only advance position after
                # a successful KV write.
                cache_position = torch.tensor([current_pos], device=self._model.device)

                with torch.no_grad():
                    outputs = self._model(
                        next_token,
                        past_key_values=past_key_values,
                        cache_position=cache_position,
                        use_cache=True,
                    )
                    past_key_values = outputs.past_key_values
                    next_token_logits = outputs.logits[:, -1, :]

                current_pos += 1
        finally:
            # Persist updated state
            session.past_key_values = past_key_values
            session.current_pos = current_pos
            session.next_token_logits = next_token_logits.detach()

            # Keep the static cache's prefill cursor aligned with the session.
            # The model's static cache does not advance its internal counter
            # during decode, but chunked prefill relies on it.
            self._sync_static_cache_seen_tokens(session.past_key_values, session.current_pos)

            # Ensure no stale CUDA-graph/static-cache state leaks across calls.
            self._reset_transient_model_state()

    def close_session(self, cache_id: str) -> None:
        """Close a session and free its KV cache."""
        session = self._sessions.pop(cache_id, None)
        if session is not None:
            del session.past_key_values

    def list_sessions(self) -> list[str]:
        """List all active session IDs."""
        return sorted(self._sessions.keys())

    def get_session_info(self, cache_id: str) -> dict[str, Any]:
        """Get info about a session."""
        session = self._sessions.get(cache_id)
        if session is None:
            raise KeyError(f"Unknown session: {cache_id}")
        return {
            "cache_id": cache_id,
            "current_pos": session.current_pos,
            "max_seq_len": session.max_seq_len,
            "has_prefill": session.next_token_logits is not None,
        }

    def checkpoint_session(self, cache_id: str) -> _SessionCheckpoint:
        """Create a lightweight in-memory checkpoint for a session."""
        self._ensure_loaded()

        session = self._sessions.get(cache_id)
        if session is None:
            raise KeyError(f"Unknown session: {cache_id}")

        cache = session.past_key_values
        conv_states = getattr(cache, "conv_states", None)
        ssm_states = getattr(cache, "ssm_states", None)

        # torch is a required dependency for this project, but keep import local.
        import torch

        conv_snap = None
        if isinstance(conv_states, torch.Tensor):
            conv_snap = conv_states.detach().clone()

        ssm_snap = None
        if isinstance(ssm_states, torch.Tensor):
            ssm_snap = ssm_states.detach().clone()

        logits_snap = None
        if isinstance(session.next_token_logits, torch.Tensor):
            logits_snap = session.next_token_logits.detach().clone()

        return _SessionCheckpoint(
            cache_obj_id=id(cache),
            current_pos=int(session.current_pos),
            next_token_logits=logits_snap,
            conv_states=conv_snap,
            ssm_states=ssm_snap,
        )

    def restore_session_checkpoint(self, *, cache_id: str, checkpoint: _SessionCheckpoint) -> None:
        """Restore a session from a checkpoint created by checkpoint_session()."""
        self._ensure_loaded()

        # Avoid stale CUDA-graph/static-cache state leaking into subsequent calls.
        self._reset_transient_model_state()

        session = self._sessions.get(cache_id)
        if session is None:
            raise KeyError(f"Unknown session: {cache_id}")

        cache = session.past_key_values
        if id(cache) != int(checkpoint.cache_obj_id):
            raise ValueError("Session cache object changed since checkpoint was created.")

        session.current_pos = int(checkpoint.current_pos)
        session.next_token_logits = checkpoint.next_token_logits

        import torch

        if checkpoint.conv_states is not None:
            conv_dst = getattr(cache, "conv_states", None)
            if not isinstance(conv_dst, torch.Tensor):
                raise TypeError("Session cache is missing conv_states tensor required for restore.")
            conv_dst.copy_(checkpoint.conv_states.to(device=conv_dst.device, dtype=conv_dst.dtype))

        if checkpoint.ssm_states is not None:
            ssm_dst = getattr(cache, "ssm_states", None)
            if not isinstance(ssm_dst, torch.Tensor):
                raise TypeError("Session cache is missing ssm_states tensor required for restore.")
            ssm_dst.copy_(checkpoint.ssm_states.to(device=ssm_dst.device, dtype=ssm_dst.dtype))

        # Ensure the static cache's prefill cursor matches the restored position.
        self._sync_static_cache_seen_tokens(cache, session.current_pos)

        # Clear any model transient state that may have been captured with previous caches.
        self._reset_transient_model_state()

    def export_session(self, cache_id: str) -> dict[str, Any]:
        """Export the raw in-memory session state.

        This returns references to the underlying cache object and tensors. Callers
        that want a durable representation should serialize the returned objects
        (e.g., via `superlinear.engine.session_snapshots`).
        """
        session = self._sessions.get(cache_id)
        if session is None:
            raise KeyError(f"Unknown session: {cache_id}")
        return {
            "cache_id": cache_id,
            "past_key_values": session.past_key_values,
            "current_pos": int(session.current_pos),
            "max_seq_len": int(session.max_seq_len),
            "next_token_logits": session.next_token_logits,
        }

    def restore_session(
        self,
        *,
        cache_id: str,
        past_key_values: Any,
        current_pos: int,
        max_seq_len: int,
        next_token_logits: torch.Tensor | None = None,
        overwrite: bool = False,
    ) -> None:
        """Restore a session from raw state (e.g., loaded snapshot).

        Note: This does not recompute any logits. For typical chat usage, the next
        `/chat/completions` call will append new tokens, which will prefill and set
        `next_token_logits` automatically.
        """
        self._ensure_loaded()

        # Avoid stale CUDA-graph/static-cache state leaking into subsequent calls.
        self._reset_transient_model_state()

        try:
            current_pos = int(current_pos)
            max_seq_len = int(max_seq_len)
        except Exception as exc:
            raise ValueError("current_pos and max_seq_len must be integers.") from exc

        if current_pos < 0:
            raise ValueError("current_pos must be >= 0.")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be > 0.")
        if current_pos > max_seq_len:
            raise ValueError("current_pos must be <= max_seq_len.")

        if cache_id in self._sessions:
            if not overwrite:
                raise ValueError(f"Session already exists: {cache_id}")
            self.close_session(cache_id)

        # Best-effort consistency checks with the provided cache object.
        cache_max = getattr(past_key_values, "max_seq_len", None)
        if cache_max is not None:
            try:
                cache_max = int(cache_max)
            except Exception:
                cache_max = None
        if cache_max is not None and cache_max != max_seq_len:
            raise ValueError(f"past_key_values.max_seq_len ({cache_max}) != max_seq_len ({max_seq_len})")

        self._sessions[cache_id] = _SessionState(
            past_key_values=past_key_values,
            current_pos=current_pos,
            max_seq_len=max_seq_len,
            next_token_logits=next_token_logits,
        )

        # Ensure the static cache's prefill cursor matches the restored position.
        self._sync_static_cache_seen_tokens(past_key_values, current_pos)

        # Clear any model transient state that may have been captured with previous caches.
        self._reset_transient_model_state()

    # -------------------------------------------------------------------------
    # Internal: Validation
    # -------------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Raise if model/tokenizer not loaded."""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

    # -------------------------------------------------------------------------
    # Internal: Prefill
    # -------------------------------------------------------------------------

    def _prefill_tokens(
        self,
        input_ids: torch.Tensor,
        *,
        past_key_values,
        start_pos: int,
        chunk_size: int,
    ):
        """Chunked prefill into a cache starting at `start_pos`."""
        import torch

        seq_len = input_ids.shape[1]
        outputs = None
        num_chunks = (seq_len + chunk_size - 1) // chunk_size

        with torch.no_grad():
            for chunk_idx in range(num_chunks):
                chunk_start = chunk_idx * chunk_size
                chunk_end = min((chunk_idx + 1) * chunk_size, seq_len)
                chunk_input_ids = input_ids[:, chunk_start:chunk_end]

                abs_start = start_pos + chunk_start
                abs_end = start_pos + chunk_end
                cache_position = torch.arange(abs_start, abs_end, device=self._model.device)

                outputs = self._model(
                    chunk_input_ids,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values

        return outputs, past_key_values, start_pos + seq_len

    def _sync_static_cache_seen_tokens(self, past_key_values: Any, current_pos: int) -> None:
        """Synchronize StaticCache internal counters with the session position.

        Superlinear's HybridMambaAttentionStaticCache intentionally does not
        increment `_layer_seen_tokens` during decode (seq_len==1). However,
        chunked prefill uses `_layer_seen_tokens` to decide where to write.
        For session workflows (decode → append prefill), we must set it.
        """
        cache = past_key_values
        if cache is None:
            return
        if not hasattr(cache, "_layer_seen_tokens"):
            return

        try:
            seen = getattr(cache, "_layer_seen_tokens")
            if not isinstance(seen, list):
                return

            # Keep all layer cursors aligned.
            #
            # In discard-thinking mode we checkpoint/restore Mamba conv/SSM state to an
            # earlier position, then re-prefill persisted history. If we only advance
            # transformer layers, Mamba layers can retain a stale cursor, creating an
            # inconsistent cache state that can trigger CUDA device-side asserts during
            # subsequent append/prefill.
            for i in range(len(seen)):
                seen[i] = int(current_pos)
        except Exception:
            # Best-effort sync; if cache implementation changes, don't crash.
            return

    # -------------------------------------------------------------------------
    # Internal: Sampling
    # -------------------------------------------------------------------------

    def _sample_token(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """Sample a single token from logits. Returns shape (1, 1)."""
        import torch

        if temperature is None or temperature < 0:
            raise ValueError(f"Temperature must be >= 0, got {temperature}")
        if temperature == 0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        else:
            # Numerical stability: compute softmax in fp32 to avoid overflow
            # that can occur with fp16 logits and low temperature.
            logits_f = logits.float() / float(temperature)
            probs = torch.softmax(logits_f, dim=-1)
            return torch.multinomial(probs, 1)

    # -------------------------------------------------------------------------
    # Internal: Cleanup
    # -------------------------------------------------------------------------

    def _cleanup_model_state(self) -> None:
        """Clear transient CUDA graph / static cache state from the model."""
        if self._model is None:
            return

        import gc
        import torch

        self._reset_transient_model_state()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _reset_transient_model_state(self) -> None:
        """Reset transient model state without aggressive memory reclamation.

        This clears CUDA-graph/static-input artifacts that can leak across calls
        and corrupt subsequent prefill/decode when using static KV caches.
        """
        if self._model is None:
            return

        for attr in ("_graph_cache", "_graph_cache_params", "_graph_batch_size"):
            if hasattr(self._model, attr):
                try:
                    setattr(self._model, attr, None)
                except Exception:
                    pass

        for attr in (
            "_static_input_ids",
            "_static_cache_position",
            "_static_attention_mask",
            "_static_output",
        ):
            if hasattr(self._model, attr):
                try:
                    delattr(self._model, attr)
                except Exception:
                    pass

    # -------------------------------------------------------------------------
    # Internal: Custom Backend (Stateless)
    # -------------------------------------------------------------------------

    def _stream_generate_custom(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float,
        stop_token_ids: list[int] | None,
        reasoning_budget: int | None = None,
        chunk_size: int | None = None,
    ) -> Iterator[torch.Tensor]:
        """Stateless generation with chunked prefill + manual decode.
        
        If reasoning_budget is set, tracks tokens inside <think>...</think> blocks.
        When the budget is exceeded, injects '</think>' tokens and continues generation.
        """
        import torch

        if chunk_size is None:
            chunk_size = self._chunk_size

        stop_token_ids = set(stop_token_ids or [])
        eos_token_id = self._tokenizer.eos_token_id

        input_ids = input_ids.to(self._model.device)
        prompt_length = input_ids.shape[1]

        # Reasoning budget tracking
        think_open_ids = self.think_open_token_ids or []
        think_close_ids = self.think_close_token_ids or []
        in_think = False
        think_tokens = 0
        recent_tokens: list[int] = []  # Ring buffer for tag detection
        max_tag_len = max(len(think_open_ids), len(think_close_ids), 1)
        budget_exceeded = False

        # Allocate static cache (extra space for potential </think> injection)
        extra_tokens = len(think_close_ids) + 2 if reasoning_budget else 0
        max_cache_len = prompt_length + max_new_tokens + extra_tokens + 1
        past_key_values = self._model.create_static_cache(batch_size=1, max_seq_len=max_cache_len)

        try:
            # Chunked prefill
            outputs, past_key_values, current_pos = self._prefill_tokens(
                input_ids,
                past_key_values=past_key_values,
                start_pos=0,
                chunk_size=chunk_size,
            )
            if outputs is None:
                return

            # Decode loop
            next_token_logits = outputs.logits[:, -1, :]
            tokens_generated = 0

            while tokens_generated < max_new_tokens:
                next_token = self._sample_token(next_token_logits, temperature)
                token_id = next_token.item()

                # Track recent tokens for tag detection
                recent_tokens.append(token_id)
                if len(recent_tokens) > max_tag_len:
                    recent_tokens.pop(0)

                # Detect <think> and </think> tags
                if think_open_ids and recent_tokens[-len(think_open_ids):] == think_open_ids:
                    in_think = True
                    think_tokens = 0  # Reset counter at start of think block
                elif think_close_ids and recent_tokens[-len(think_close_ids):] == think_close_ids:
                    in_think = False

                # Count thinking tokens
                if in_think:
                    think_tokens += 1

                # Check if reasoning budget exceeded
                if (
                    reasoning_budget is not None
                    and in_think
                    and think_tokens >= reasoning_budget
                    and not budget_exceeded
                ):
                    budget_exceeded = True
                    # Inject </think> tokens
                    for close_token_id in think_close_ids:
                        close_token = torch.tensor([[close_token_id]], device=self._model.device)
                        yield close_token.squeeze(0)
                        tokens_generated += 1

                        # Update model state for injected token
                        cache_position = torch.tensor([current_pos], device=self._model.device)
                        current_pos += 1
                        with torch.no_grad():
                            outputs = self._model(
                                close_token,
                                past_key_values=past_key_values,
                                cache_position=cache_position,
                                use_cache=True,
                            )
                            past_key_values = outputs.past_key_values
                            next_token_logits = outputs.logits[:, -1, :]

                    in_think = False
                    # Skip the token that triggered budget exceeded, continue with new logits
                    continue

                yield next_token.squeeze(0)  # Shape (1,)
                tokens_generated += 1

                # Check stop conditions
                if token_id == eos_token_id:
                    break
                if token_id in stop_token_ids:
                    break

                # Forward pass
                cache_position = torch.tensor([current_pos], device=self._model.device)
                current_pos += 1

                with torch.no_grad():
                    outputs = self._model(
                        next_token,
                        past_key_values=past_key_values,
                        cache_position=cache_position,
                        use_cache=True,
                    )
                    past_key_values = outputs.past_key_values
                    next_token_logits = outputs.logits[:, -1, :]
        finally:
            self._cleanup_model_state()

    # -------------------------------------------------------------------------
    # Internal: HF Backend (Stateless)
    # -------------------------------------------------------------------------

    def _generate_hf(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float,
        stop_token_ids: list[int] | None,
        **kwargs,
    ) -> torch.Tensor:
        """Stateless generation using Transformers' built-in generate()."""
        import torch
        from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

        input_ids = input_ids.to(self._model.device)
        stop_token_ids = set(stop_token_ids or [])

        do_sample = temperature is not None and temperature > 0
        pad_token_id = self._tokenizer.pad_token_id or self._tokenizer.eos_token_id

        # Build stopping criteria
        stopping_criteria = None
        if stop_token_ids:
            class _StopOnTokens(StoppingCriteria):
                def __init__(self, stop_ids: set[int]):
                    super().__init__()
                    self._stop_ids = stop_ids

                def __call__(self, gen_ids, scores, **kwargs) -> bool:
                    return gen_ids[0, -1].item() in self._stop_ids

            stopping_criteria = StoppingCriteriaList([_StopOnTokens(stop_token_ids)])

        try:
            with torch.no_grad():
                out_ids = self._model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    use_cache=True,
                    pad_token_id=pad_token_id,
                    stopping_criteria=stopping_criteria,
                    **kwargs,
                )

            # Return only new tokens
            return out_ids[0, input_ids.shape[1]:]
        finally:
            self._cleanup_model_state()

    def _stream_generate_hf(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float,
        stop_token_ids: list[int] | None,
        reasoning_budget: int | None = None,
        **kwargs,
    ) -> Iterator[torch.Tensor]:
        """Stateless streaming via Transformers' TextIteratorStreamer.

        Note: Threading is required because HF's generate() is blocking.
        We run it in a background thread and yield tokens via a queue.
        
        Note: reasoning_budget is not enforced in the HF backend (would require
        stopping generation mid-stream and resuming, which HF doesn't support well).
        Use the custom backend for reasoning budget enforcement.
        """
        import queue
        import threading
        import torch
        from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
        from transformers.generation.streamers import BaseStreamer

        input_ids = input_ids.to(self._model.device)
        stop_token_ids = set(stop_token_ids or [])
        eos_token_id = self._tokenizer.eos_token_id

        do_sample = temperature is not None and temperature > 0
        pad_token_id = self._tokenizer.pad_token_id or eos_token_id

        # Build stopping criteria
        stopping_criteria = None
        if stop_token_ids:
            class _StopOnTokens(StoppingCriteria):
                def __init__(self, stop_ids: set[int]):
                    super().__init__()
                    self._stop_ids = stop_ids

                def __call__(self, gen_ids, scores, **kwargs) -> bool:
                    return gen_ids[0, -1].item() in self._stop_ids

            stopping_criteria = StoppingCriteriaList([_StopOnTokens(stop_token_ids)])

        # Token-level streamer with early-stop support
        token_queue: queue.Queue[torch.Tensor | BaseException | None] = queue.Queue()
        prompt_len = input_ids.shape[1]
        end_signaled = threading.Event()

        class _TokenStreamer(BaseStreamer):
            def __init__(self):
                self._generated_count = 0

            def put(self, value):
                # HF's generate() calls put() with individual tokens (1D) or batches (2D)
                if isinstance(value, torch.Tensor):
                    # Handle both 1D (single token) and 2D (batch) cases
                    if value.dim() == 1:
                        # Single token batch: value is shape (batch_size,), extract first element
                        # and create a shape (1,) tensor to match custom backend output
                        token_queue.put(value[:1])  # Take first element, shape (1,)
                        self._generated_count += 1
                    else:
                        # Full sequence: extract new tokens
                        new_tokens = value[0, prompt_len + self._generated_count:]
                        for i in range(new_tokens.shape[0]):
                            token_queue.put(new_tokens[i:i+1])
                            self._generated_count += 1

            def end(self):
                if not end_signaled.is_set():
                    end_signaled.set()
                    token_queue.put(None)

        streamer = _TokenStreamer()

        def _run_generate() -> None:
            try:
                with torch.no_grad():
                    self._model.generate(
                        input_ids=input_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature if do_sample else None,
                        use_cache=True,
                        pad_token_id=pad_token_id,
                        stopping_criteria=stopping_criteria,
                        streamer=streamer,
                        **kwargs,
                    )
            except Exception as e:
                # Signal error to main thread
                token_queue.put(e)
            finally:
                # Ensure end signal is sent even on error (idempotent due to Event)
                streamer.end()
                self._cleanup_model_state()

        thread = threading.Thread(target=_run_generate, daemon=True)
        thread.start()

        try:
            while True:
                token = token_queue.get()
                if token is None:
                    break
                if isinstance(token, BaseException):
                    raise token
                token_id = token.item()
                yield token
                # Check stop conditions (generate might not have stopped yet)
                if token_id == eos_token_id or token_id in stop_token_ids:
                    break
        finally:
            # Always wait for thread to finish to ensure cleanup runs
            thread.join()
