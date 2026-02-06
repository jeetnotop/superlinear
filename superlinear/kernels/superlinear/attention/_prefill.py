"""
Span attention helpers for the span_attention_exp model.

This module houses the fused span attention paths that reuse span-search logits
for gating (`full_span_attention_fused_with_search_values` and
`full_span_attention_fused_with_search_values_gqa`) as well as the flex-attention
prefill paths (`fused_prefill_with_swflex` and `fused_prefill_with_swflex_gqa`).

It also re-exports decoding-only kernel overrides used by `span_attention_exp`
(`decode_span_attention_staged` and `decode_span_attention_staged_gqa_kernel_v2`)
so the model code can stay self-contained inside this directory.

It mirrors the reference implementations in the repo root so models that rely
on files inside `span_attention_exp` can import everything they need without
touching the top-level scripts.
"""

import functools
import math
import os

import torch
 

from ..search._triton import _cache_position_type, span_search_with_values
from ..search._triton_gqa import span_search_with_values_gqa
from superlinear.kernels.common.power import (
    derive_stripe_power_params,
    max_stripe_index_for_token_pos,
    window_len_from_sw_index,
)
from superlinear.kernels.common.adjustment import compute_qend_from_qanchor
from ..span._triton_impl import (
    _assert_no_span_sw_overlap,
    _next_power_of_two,
    decode_span_attention_staged,
    fused_span_attention,
)
from ..span._triton_precomputed_sw import fused_span_attention_with_precomputed_sw
from ..span._triton_precomputed_sw_gqa import fused_span_attention_with_precomputed_sw_gqa
from ..span._triton_gqa import decode_span_attention_staged_gqa_kernel_v2, fused_span_attention_gqa
from torch.nn.attention.flex_attention import AuxRequest, create_block_mask, flex_attention

from ._sliding_window import (
    DEFAULT_SW_BLOCK_K,
    DEFAULT_SW_NUM_STAGES,
    DEFAULT_SW_NUM_WARPS,
    sliding_window_attention_triton,
    sliding_window_attention_triton_gqa,
)

from ..span._triton_bucketed_gqa import (
    DEFAULT_BLOCK_K as DEFAULT_SPAN_BLOCK_K,
    DEFAULT_BLOCK_P as DEFAULT_SPAN_BLOCK_P,
    DEFAULT_NUM_WARPS_SPAN,
    DEFAULT_NUM_STAGES_SPAN,
    span_attention_bucketed_packed_tiles_gqa_independent,
    merge_gate_qmajor,
)


# =============================================================================
# Hardware-Aware Flex Attention Configuration
# =============================================================================
#
# Problem: PyTorch's flex_attention uses Triton kernels that require shared memory.
# On GPUs with limited shared memory (~100KB on RTX A6000), aggressive kernel
# configurations can exceed hardware limits, causing runtime failures.
#
# Solution: Proactively select kernel configurations that fit within the GPU's
# shared memory budget. If the "fast" default config fits, return None to allow
# Triton's autotuner to optimize. Otherwise, fall back to smaller block sizes.
# =============================================================================

# Environment variable to force fallback configs (for reproducibility/debugging)
_FORCE_FLEX_FALLBACK = os.environ.get("SPAN_ATTN_FORCE_FLEX_FALLBACK", "0") == "1"

# Safety margin for SMEM estimate (accounts for Triton metadata/pipelining overhead)
_SMEM_SAFETY_BYTES = 12 * 1024  # 12KB


@functools.lru_cache(maxsize=16)
def _get_smem_limit(device_index: int = 0) -> int:
    """
    Query the available shared memory limit for a given GPU.

    Uses the opt-in per-block limit if available (some architectures allow
    configuring more shared memory per block), clamped by the per-SM limit
    for safety.
    """
    if not torch.cuda.is_available() or device_index >= torch.cuda.device_count():
        return 48 * 1024  # Conservative 48KB fallback

    props = torch.cuda.get_device_properties(device_index)

    # Some architectures support opt-in extended shared memory per block
    per_block = getattr(props, "shared_memory_per_block_optin", props.shared_memory_per_block)
    per_sm = props.shared_memory_per_multiprocessor

    # Use the smaller of per-block and per-SM limits for safety
    return min(per_block, per_sm)


def _estimate_flex_smem(
    block_m: int,
    block_n: int,
    head_dim: int,
    num_stages: int,
    dtype_bytes: int = 2,
    safety_bytes: int = _SMEM_SAFETY_BYTES,
) -> int:
    """
    Estimate shared memory usage for flex_attention kernel.
    """
    q_smem = block_m * head_dim * dtype_bytes
    k_smem = block_n * head_dim * dtype_bytes * num_stages
    v_smem = block_n * head_dim * dtype_bytes * num_stages
    attn_smem = block_m * block_n * 4  # FP32 scores
    out_smem = block_m * head_dim * 4   # FP32 accumulator
    lse_smem = block_m * 4              # FP32 LSE

    return q_smem + k_smem + v_smem + attn_smem + out_smem + lse_smem + safety_bytes


# Default "fast" configuration - used when hardware supports it
_FAST_CONFIG = {"BLOCK_M": 128, "BLOCK_N": 128, "num_stages": 2}

# Ordered fallback configurations (most performant to most conservative)
_FALLBACK_CONFIGS = [
    {"BLOCK_M": 128, "BLOCK_N": 128, "num_stages": 1},
    {"BLOCK_M": 128, "BLOCK_N": 64,  "num_stages": 2},
    {"BLOCK_M": 128, "BLOCK_N": 64,  "num_stages": 1},
    {"BLOCK_M": 64,  "BLOCK_N": 128, "num_stages": 1},
    {"BLOCK_M": 64,  "BLOCK_N": 64,  "num_stages": 2},
    {"BLOCK_M": 64,  "BLOCK_N": 64,  "num_stages": 1},
    {"BLOCK_M": 64,  "BLOCK_N": 32,  "num_stages": 1},
    {"BLOCK_M": 32,  "BLOCK_N": 64,  "num_stages": 1},
    {"BLOCK_M": 32,  "BLOCK_N": 32,  "num_stages": 1},
]

# Ultimate fallback - guaranteed to fit on any reasonable GPU
_MINIMUM_CONFIG = {"BLOCK_M": 32, "BLOCK_N": 32, "num_stages": 1}


@functools.lru_cache(maxsize=32)
def _choose_flex_kernel_options(
    head_dim: int,
    dtype_bytes: int,
    device_index: int,
) -> dict | None:
    """
    Choose flex_attention kernel options based on GPU shared memory constraints.
    """
    smem_limit = _get_smem_limit(device_index)

    # Check if fast config fits (unless force fallback is enabled)
    if not _FORCE_FLEX_FALLBACK:
        fast_smem = _estimate_flex_smem(
            _FAST_CONFIG["BLOCK_M"],
            _FAST_CONFIG["BLOCK_N"],
            head_dim,
            _FAST_CONFIG["num_stages"],
            dtype_bytes,
        )
        if fast_smem <= smem_limit:
            return None

    # Find first fallback config that fits
    for cfg in _FALLBACK_CONFIGS:
        est_smem = _estimate_flex_smem(
            cfg["BLOCK_M"],
            cfg["BLOCK_N"],
            head_dim,
            cfg["num_stages"],
            dtype_bytes,
        )
        if est_smem <= smem_limit:
            return {"FORCE_USE_FLEX_ATTENTION": True, **cfg}

    # Ultimate fallback
    return {"FORCE_USE_FLEX_ATTENTION": True, **_MINIMUM_CONFIG}


def _get_flex_options_for_tensor(Q: torch.Tensor) -> dict | None:
    """
    Convenience function to get flex_attention kernel options for a query tensor.
    """
    if not torch.cuda.is_available() or Q.device.type != "cuda":
        return None

    _, _, _, head_dim = Q.shape
    dtype_bytes = 2 if Q.dtype in (torch.float16, torch.bfloat16) else 4
    device_index = Q.device.index if Q.device.index is not None else 0

    return _choose_flex_kernel_options(head_dim, dtype_bytes, device_index)


flex_attention = torch.compile(flex_attention, mode="default")
create_block_mask = torch.compile(create_block_mask, mode="default")


def build_sw_blockmask(
    batch_size,
    q_len,
    kv_len,
    sw_index=0,
    attention_mask=None,
    cache_position=None,
    block_size=128,
    device=None,
    search_power: float | None = None,
    inv_search_power_int: int | None = 2,
):
    """
    Build a cache-position-aware sliding window BlockMask for FlexAttention using lengths only.

    The mask is broadcast across heads (H=None) for efficiency; create_block_mask handles the
    per-head expansion internally.
    """
    if cache_position is None:
        raise ValueError("build_sw_blockmask requires cache_position to align the sliding window with true token positions.")

    try:
        sw_index_int = int(sw_index)
    except (TypeError, ValueError):
        raise TypeError("sw_index must be an integer value") from None
    if sw_index_int < 0:
        raise ValueError("sw_index must be non-negative")

    if device is None:
        device = cache_position.device

    cache_position = cache_position.to(device=device, dtype=torch.int64).flatten()
    if cache_position.numel() != q_len:
        raise ValueError(f"cache_position length {cache_position.numel()} must match q_len {q_len}")

    cache_position = cache_position.clamp(max=kv_len - 1)
    window = window_len_from_sw_index(
        sw_index_int, search_power=search_power, inv_search_power_int=inv_search_power_int
    )

    attn_mask_bool = None
    if attention_mask is not None:
        attn_mask_bool = attention_mask.to(device=device).to(torch.bool)

    def sw_mask(b, h, q_idx, kv_idx):
        c_idx = cache_position[q_idx]
        diff = c_idx - kv_idx + 1
        mask = (diff <= window) & (diff > 0)
        if attn_mask_bool is not None:
            mask = mask & attn_mask_bool[b][kv_idx]
        return mask

    return create_block_mask(
        sw_mask, B=batch_size, H=None, Q_LEN=q_len, KV_LEN=kv_len, device=device, BLOCK_SIZE=block_size
    )


def full_span_attention_fused_with_search_values(
    Q1,
    Q2,
    K,
    V,
    cache_position,
    attention_mask=None,
    sw_index: int = 0,
    num_spans: int = 3,
    block_k: int = 64,
    search_block_q: int = 128,
    backward_factor: float = 2.0,
    forward_factor: float = 0.0,
    span_power: float = 0.5,
    search_power: float | None = None,
    inv_search_power_int: int | None = 2,
):
    """
    Fused span attention that reuses search logits for gating.

    Args:
        Q1: Search/query tensor used for span selection. Shape [B, H, L_Q, D].
        Q2: Query tensor used for attention computation. Shape [B, H, L_Q, D].
        K: Key tensor. Shape [B, H, L_KV, D].
        V: Value tensor. Shape [B, H, L_KV, D].
        cache_position: 1D tensor of absolute cache positions for each query token.
        attention_mask: Optional boolean/int mask over keys with shape [B, L_KV].
        sw_index: Sliding-window index controlling local context span.
        num_spans: Number of top spans to aggregate.
        block_k: Block size for the fused span attention kernel.
        search_block_q: Block size for the span search kernel.

    Returns:
        Tensor of shape [B, H, L_Q, D] containing the attention output.
    """
    qstart, qanchor, values = span_search_with_values(
        Q1,
        K,
        cache_position,
        attention_mask=attention_mask,
        sw_index=sw_index,
        k=num_spans,
        block_size=search_block_q,
        backward_factor=backward_factor,
        span_power=span_power,
        search_power=search_power,
        inv_search_power_int=inv_search_power_int,
    )

    qend = compute_qend_from_qanchor(
        qanchor,
        cache_position=cache_position,
        key_length=K.shape[2],
        sw_index=sw_index,
        attention_mask=attention_mask,
        forward_factor=forward_factor,
        span_power=span_power,
        search_power=search_power,
        inv_search_power_int=inv_search_power_int,
    )
    span_len_factor = backward_factor + forward_factor

    O_span = fused_span_attention(
        Q2,
        K,
        V,
        qstart,
        qend,
        cache_position,
        attention_mask=attention_mask,
        sw_index=sw_index,
        block_k=block_k,
        span_len_factor=span_len_factor,
        span_power=span_power,
        search_power=search_power,
        inv_search_power_int=inv_search_power_int,
    )

    values = torch.where(qanchor < 0, values.new_full((), -1e9), values)
    span_scores = torch.softmax(values, dim=-1)
    span_scores = torch.nan_to_num(span_scores, nan=1.0 / num_spans, posinf=0.0, neginf=0.0)
    span_scores = span_scores.to(O_span.dtype)
    O = span_scores[:, :, :, None, :].matmul(O_span).squeeze(-2)
    return O.to(Q2.dtype)


# =============================================================================
# Span Search With Values (GQA)
# =============================================================================

# (GQA span-search Triton kernels live in superlinear.search._triton_gqa)


def full_span_attention_fused_with_search_values_gqa(
    Q1,
    Q2,
    K,
    V,
    cache_position,
    attention_mask=None,
    sw_index: int = 0,
    num_spans: int = 3,
    block_k: int = 64,
    search_block_q: int = 128,
    backward_factor: float = 2.0,
    forward_factor: float = 0.0,
    span_power: float = 0.5,
    search_power: float | None = None,
    inv_search_power_int: int | None = 2,
):
    """Full-span fused attention using grouped-query KV heads (no K/V repetition)."""
    B, H_q, _, _ = Q1.shape
    _, H_kv, L_KV, _ = K.shape
    assert H_q % H_kv == 0, "Query heads must be divisible by KV heads when using GQA"

    # Ensure all tensors are on the same device (multi-GPU compatibility)
    target_device = Q1.device
    K = K.to(target_device)
    V = V.to(target_device)
    Q2 = Q2.to(target_device)
    cache_position = cache_position.to(target_device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(target_device)

    # Use torch.cuda.device context for all kernel operations
    with torch.cuda.device(target_device):
        qstart, qanchor, values = span_search_with_values_gqa(
            Q1,
            K,
            cache_position,
            attention_mask=attention_mask,
            sw_index=sw_index,
            k=num_spans,
            block_size=search_block_q,
            backward_factor=backward_factor,
            span_power=span_power,
            search_power=search_power,
            inv_search_power_int=inv_search_power_int,
        )

        qend = compute_qend_from_qanchor(
            qanchor,
            cache_position=cache_position,
            key_length=K.shape[2],
            sw_index=sw_index,
            attention_mask=attention_mask,
            forward_factor=forward_factor,
            span_power=span_power,
            search_power=search_power,
            inv_search_power_int=inv_search_power_int,
        )
        span_len_factor = backward_factor + forward_factor

        O_span = fused_span_attention_gqa(
            Q2,
            K,
            V,
            qstart,
            qend,
            cache_position,
            attention_mask=attention_mask,
            sw_index=sw_index,
            block_k=block_k,
            span_len_factor=span_len_factor,
            span_power=span_power,
            search_power=search_power,
            inv_search_power_int=inv_search_power_int,
        )

    values = torch.where(qanchor < 0, values.new_full((), -1e9), values)
    span_scores = torch.softmax(values, dim=-1)
    span_scores = torch.nan_to_num(span_scores, nan=1.0 / num_spans, posinf=0.0, neginf=0.0).to(O_span.dtype)

    O = span_scores[:, :, :, None, :].matmul(O_span).squeeze(-2)
    return O.to(Q2.dtype)


def fused_prefill_with_swflex(
    Q1,
    Q2,
    K,
    V,
    cache_pos,
    attention_mask,
    sw_block_mask,
    sw_index,
    topk,
    backward_factor: float = 2.0,
    forward_factor: float = 0.0,
    span_power: float = 0.5,
    search_power: float | None = None,
    inv_search_power_int: int | None = 2,
):
    """
    Fused prefill combining span search, flex attention sliding window, and gating.
    """
    # Step 1: Search for spans (returns indices and logits for gating)
    qstart, qanchor, values = span_search_with_values(
        Q1,
        K,
        cache_pos,
        attention_mask=attention_mask,
        sw_index=sw_index,
        k=topk,
        backward_factor=backward_factor,
        span_power=span_power,
        search_power=search_power,
        inv_search_power_int=inv_search_power_int,
    )
    qend = compute_qend_from_qanchor(
        qanchor,
        cache_position=cache_pos,
        key_length=K.shape[2],
        sw_index=sw_index,
        attention_mask=attention_mask,
        forward_factor=forward_factor,
        span_power=span_power,
        search_power=search_power,
        inv_search_power_int=inv_search_power_int,
    )
    span_len_factor = backward_factor + forward_factor

    # Step 2: Compute sliding window attention with hardware-aware config
    flex_opts = _get_flex_options_for_tensor(Q2)
    sw_out, sw_aux = flex_attention(
        Q2, K, V,
        block_mask=sw_block_mask,
        return_aux=AuxRequest(lse=True, max_scores=False),
        kernel_options=flex_opts,
    )

    # Step 3: Compute span attention using precomputed SW
    O_span = fused_span_attention_with_precomputed_sw(
        Q2,
        K,
        V,
        qstart,
        qend,
        cache_pos,
        sw_out, sw_aux.lse,
        attention_mask=attention_mask, sw_index=sw_index,
        span_len_factor=span_len_factor,
    )

    # Step 4: Gate and aggregate
    values = torch.where(qanchor < 0, values.new_full((), -1e9), values)
    span_scores = torch.softmax(values, dim=-1)
    span_scores = torch.nan_to_num(span_scores, nan=1.0 / topk, posinf=0.0, neginf=0.0).to(O_span.dtype)
    O = span_scores[:, :, :, None, :].matmul(O_span).squeeze(-2)
    return O



# (GQA precomputed-SW span kernels live in superlinear.span._triton_precomputed_sw_gqa)


def fused_prefill_with_swflex_gqa(
    Q1,
    Q2,
    K,
    V,
    cache_pos,
    attention_mask,
    sw_block_mask,
    sw_index,
    topk,
    backward_factor: float = 2.0,
    forward_factor: float = 0.0,
    span_power: float = 0.5,
    search_power: float | None = None,
    inv_search_power_int: int | None = 2,
):
    """
    GQA variant of fused prefill combining span search, flex sliding window, and gating.

    Unlike `fused_prefill_with_swflex`, this path keeps K/V in their grouped-query
    (num_kv_heads) layout and relies on flex_attention's GQA support.
    """
    qstart, qanchor, values = span_search_with_values_gqa(
        Q1,
        K,
        cache_pos,
        attention_mask=attention_mask,
        sw_index=sw_index,
        k=topk,
        backward_factor=backward_factor,
        span_power=span_power,
        search_power=search_power,
        inv_search_power_int=inv_search_power_int,
    )
    qend = compute_qend_from_qanchor(
        qanchor,
        cache_position=cache_pos,
        key_length=K.shape[2],
        sw_index=sw_index,
        attention_mask=attention_mask,
        forward_factor=forward_factor,
        span_power=span_power,
        search_power=search_power,
        inv_search_power_int=inv_search_power_int,
    )
    span_len_factor = backward_factor + forward_factor

    flex_opts = _get_flex_options_for_tensor(Q2)
    sw_out, sw_aux = flex_attention(
        Q2,
        K,
        V,
        block_mask=sw_block_mask,
        return_aux=AuxRequest(lse=True, max_scores=False),
        kernel_options=flex_opts,
        enable_gqa=True,
    )

    O_span = fused_span_attention_with_precomputed_sw_gqa(
        Q2,
        K,
        V,
        qstart,
        qend,
        cache_pos,
        sw_out,
        sw_aux.lse,
        attention_mask=attention_mask,
        sw_index=sw_index,
        span_len_factor=span_len_factor,
        span_power=span_power,
        search_power=search_power,
        inv_search_power_int=inv_search_power_int,
    )

    values = torch.where(qanchor < 0, values.new_full((), -1e9), values)
    span_scores = torch.softmax(values, dim=-1)
    span_scores = torch.nan_to_num(span_scores, nan=1.0 / topk, posinf=0.0, neginf=0.0).to(O_span.dtype)
    O = span_scores[:, :, :, None, :].matmul(O_span).squeeze(-2)
    return O


def fused_prefill_with_swtriton(
    Q1,
    Q2,
    K,
    V,
    cache_pos,
    attention_mask,
    sw_index,
    topk,
    *,
    key_cache_position: torch.Tensor | None = None,
    sw_block_k: int = DEFAULT_SW_BLOCK_K,
    sw_block_m_dense: int = 16,
    sw_num_warps: int = DEFAULT_SW_NUM_WARPS,
    sw_num_stages: int = DEFAULT_SW_NUM_STAGES,
    use_bf16_pv: bool = True,
    backward_factor: float = 2.0,
    forward_factor: float = 0.0,
    span_power: float = 0.5,
    search_power: float | None = None,
    inv_search_power_int: int | None = 2,
):
    """
    Fused prefill combining span search, Triton sliding-window attention, and gating.

    This mirrors `fused_prefill_with_swflex` but replaces FlexAttention with a
    cache-position-aware Triton sliding-window kernel (block-of-queries forward,
    token-parallel backward).
    """
    qstart, qanchor, values = span_search_with_values(
        Q1,
        K,
        cache_pos,
        attention_mask=attention_mask,
        sw_index=sw_index,
        k=topk,
        backward_factor=backward_factor,
        span_power=span_power,
        search_power=search_power,
        inv_search_power_int=inv_search_power_int,
    )
    qend = compute_qend_from_qanchor(
        qanchor,
        cache_position=cache_pos,
        key_length=K.shape[2],
        sw_index=sw_index,
        attention_mask=attention_mask,
        forward_factor=forward_factor,
        span_power=span_power,
        search_power=search_power,
        inv_search_power_int=inv_search_power_int,
    )
    span_len_factor = backward_factor + forward_factor

    sw_out, sw_lse = sliding_window_attention_triton(
        Q2,
        K,
        V,
        cache_position=cache_pos,
        key_cache_position=key_cache_position,
        attention_mask=attention_mask,
        sw_index=sw_index,
        search_power=search_power,
        inv_search_power_int=inv_search_power_int,
        block_k=int(sw_block_k),
        block_m_dense=int(sw_block_m_dense),
        num_warps=int(sw_num_warps),
        num_stages=int(sw_num_stages),
        use_bf16_pv=bool(use_bf16_pv),
    )

    O_span = fused_span_attention_with_precomputed_sw(
        Q2,
        K,
        V,
        qstart,
        qend,
        cache_pos,
        sw_out,
        sw_lse,
        attention_mask=attention_mask,
        sw_index=sw_index,
        span_len_factor=span_len_factor,
        span_power=span_power,
        search_power=search_power,
        inv_search_power_int=inv_search_power_int,
    )

    values = torch.where(qanchor < 0, values.new_full((), -1e9), values)
    span_scores = torch.softmax(values, dim=-1)
    span_scores = torch.nan_to_num(span_scores, nan=1.0 / topk, posinf=0.0, neginf=0.0).to(O_span.dtype)
    O = span_scores[:, :, :, None, :].matmul(O_span).squeeze(-2)
    return O


def fused_prefill_with_swtriton_gqa(
    Q1,
    Q2,
    K,
    V,
    cache_pos,
    attention_mask,
    sw_index,
    topk,
    *,
    key_cache_position: torch.Tensor | None = None,
    sw_block_k: int = DEFAULT_SW_BLOCK_K,
    sw_block_m_dense: int = 16,
    sw_num_warps: int = DEFAULT_SW_NUM_WARPS,
    sw_num_stages: int = DEFAULT_SW_NUM_STAGES,
    use_bf16_pv: bool = True,
    backward_factor: float = 2.0,
    forward_factor: float = 0.0,
    span_power: float = 0.5,
    search_power: float | None = None,
    inv_search_power_int: int | None = 2,
):
    """
    GQA variant of fused prefill combining span search, Triton sliding window, and gating.

    Keeps K/V in grouped-query (num_kv_heads) layout and relies on the Triton
    SW kernel's `kv_for_q` mapping + atomic accumulation in backward.
    """
    qstart, qanchor, values = span_search_with_values_gqa(
        Q1,
        K,
        cache_pos,
        attention_mask=attention_mask,
        sw_index=sw_index,
        k=topk,
        backward_factor=backward_factor,
        span_power=span_power,
        search_power=search_power,
        inv_search_power_int=inv_search_power_int,
    )
    qend = compute_qend_from_qanchor(
        qanchor,
        cache_position=cache_pos,
        key_length=K.shape[2],
        sw_index=sw_index,
        attention_mask=attention_mask,
        forward_factor=forward_factor,
        span_power=span_power,
        search_power=search_power,
        inv_search_power_int=inv_search_power_int,
    )
    span_len_factor = backward_factor + forward_factor

    sw_out, sw_lse = sliding_window_attention_triton_gqa(
        Q2,
        K,
        V,
        cache_position=cache_pos,
        key_cache_position=key_cache_position,
        attention_mask=attention_mask,
        sw_index=sw_index,
        search_power=search_power,
        inv_search_power_int=inv_search_power_int,
        block_k=int(sw_block_k),
        block_m_dense=int(sw_block_m_dense),
        num_warps=int(sw_num_warps),
        num_stages=int(sw_num_stages),
        use_bf16_pv=bool(use_bf16_pv),
    )

    O_span = fused_span_attention_with_precomputed_sw_gqa(
        Q2,
        K,
        V,
        qstart,
        qend,
        cache_pos,
        sw_out,
        sw_lse,
        attention_mask=attention_mask,
        sw_index=sw_index,
        span_len_factor=span_len_factor,
        span_power=span_power,
        search_power=search_power,
        inv_search_power_int=inv_search_power_int,
    )

    values = torch.where(qanchor < 0, values.new_full((), -1e9), values)
    span_scores = torch.softmax(values, dim=-1)
    span_scores = torch.nan_to_num(span_scores, nan=1.0 / topk, posinf=0.0, neginf=0.0).to(O_span.dtype)
    O = span_scores[:, :, :, None, :].matmul(O_span).squeeze(-2)
    return O


def fused_prefill_with_swtriton_bucketed_gqa(
    Q1,
    Q2,
    K,
    V,
    cache_pos,
    attention_mask,
    sw_index,
    topk,
    *,
    key_cache_position: torch.Tensor | None = None,
    sw_block_k: int = DEFAULT_SW_BLOCK_K,
    sw_block_m_dense: int = 16,
    sw_num_warps: int = DEFAULT_SW_NUM_WARPS,
    sw_num_stages: int = DEFAULT_SW_NUM_STAGES,
    span_block_k: int = DEFAULT_SPAN_BLOCK_K,
    span_block_p: int = DEFAULT_SPAN_BLOCK_P,
    span_num_warps: int = DEFAULT_NUM_WARPS_SPAN,
    span_num_stages: int = DEFAULT_NUM_STAGES_SPAN,
    use_bf16_pv: bool = True,
    backward_factor: float = 2.0,
    forward_factor: float = 0.0,
    span_power: float = 0.5,
    search_power: float | None = None,
    inv_search_power_int: int | None = 2,
):
    """
    GQA variant of fused prefill using BUCKETED span attention kernel.
    
    This is the optimized implementation from notebook 56.12 that achieves
    1.3-1.6x speedup over FlexAttention by:
    1. Using histogram-based bucketing to group queries with similar span ranges
    2. Processing queries in the same bucket together for better memory locality
    3. Using separate SW + span attention kernels with Triton merge
    
    Args:
        Q1: Search/query tensor for span selection [B, H_q, L_Q, D]
        Q2: Query tensor for attention computation [B, H_q, L_Q, D]
        K: Key tensor [B, H_kv, L_KV, D]
        V: Value tensor [B, H_kv, L_KV, D]
        cache_pos: Cache positions for query tokens [L_Q]
        attention_mask: Boolean mask over keys [B, L_KV]
        sw_index: Sliding window index
        topk: Number of top spans to aggregate
        key_cache_position: Optional key cache positions (for non-contiguous KV cache)
        sw_block_k: Block size for K in SW kernel
        sw_block_m_dense: Block size for dense path in SW kernel
        sw_num_warps: Number of warps for SW kernel
        sw_num_stages: Number of stages for SW kernel
        span_block_k: Block size for K in span kernel
        span_block_p: Block size for processing query pairs in span kernel
        span_num_warps: Number of warps for span kernel
        span_num_stages: Number of stages for span kernel
        use_bf16_pv: Use bf16 for P@V multiplication
        backward_factor: Factor for backward span direction
        forward_factor: Factor for forward span direction
        span_power: Power for span length computation
        search_power: Power for search (if different from span_power)
        inv_search_power_int: Integer inverse of search power
        
    Returns:
        Output tensor [B, H_q, L_Q, D]
    """
    # Step 1: Span search to find top-k span starts and gate values
    qstart, qanchor, values = span_search_with_values_gqa(
        Q1,
        K,
        cache_pos,
        attention_mask=attention_mask,
        sw_index=sw_index,
        k=topk,
        backward_factor=backward_factor,
        span_power=span_power,
        search_power=search_power,
        inv_search_power_int=inv_search_power_int,
    )
    
    # Step 2: Compute span end positions
    qend = compute_qend_from_qanchor(
        qanchor,
        cache_position=cache_pos,
        key_length=K.shape[2],
        sw_index=sw_index,
        attention_mask=attention_mask,
        forward_factor=forward_factor,
        span_power=span_power,
        search_power=search_power,
        inv_search_power_int=inv_search_power_int,
    )
    span_len_factor = backward_factor + forward_factor

    # Step 3: Compute sliding window attention using Triton kernel
    sw_out, sw_lse = sliding_window_attention_triton_gqa(
        Q2,
        K,
        V,
        cache_position=cache_pos,
        key_cache_position=key_cache_position,
        attention_mask=attention_mask,
        sw_index=sw_index,
        search_power=search_power,
        inv_search_power_int=inv_search_power_int,
        block_k=int(sw_block_k),
        block_m_dense=int(sw_block_m_dense),
        num_warps=int(sw_num_warps),
        num_stages=int(sw_num_stages),
        use_bf16_pv=bool(use_bf16_pv),
    )

    # Step 4: Compute span attention using BUCKETED kernel
    span_out, span_lse = span_attention_bucketed_packed_tiles_gqa_independent(
        Q2,
        K,
        V,
        qstart,
        qend,
        attention_mask=attention_mask,
        block_k=int(span_block_k),
        block_p=int(span_block_p),
        span_len_factor=float(span_len_factor),
        span_power=float(span_power),
        max_span_blocks_cap=None,
        num_warps=int(span_num_warps),
        num_stages=int(span_num_stages),
        use_bf16_pv=bool(use_bf16_pv),
    )

    # Step 5: Merge SW and span outputs with gated aggregation
    # Mark invalid spans (where qanchor < 0) with very negative values
    values_masked = torch.where(qanchor < 0, values.new_full((), -1e9), values)
    
    return merge_gate_qmajor(
        sw_out,
        sw_lse,
        span_out,
        span_lse,
        values_masked,
        qend,
    ).to(Q2.dtype)


__all__ = [
    "build_sw_blockmask",
    "decode_span_attention_staged",
    "decode_span_attention_staged_gqa_kernel_v2",
    "full_span_attention_fused_with_search_values",
    "full_span_attention_fused_with_search_values_gqa",
    "sliding_window_attention_triton",
    "sliding_window_attention_triton_gqa",
    "fused_prefill_with_swflex",
    "fused_prefill_with_swflex_gqa",
    "fused_prefill_with_swtriton",
    "fused_prefill_with_swtriton_gqa",
    "fused_prefill_with_swtriton_bucketed_gqa",
]
