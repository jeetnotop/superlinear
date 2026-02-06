"""Public API for span attention operations.

This module provides stable entry points for span attention. Users should import
from here rather than from the private implementation modules.

Key functions:
- fused_span_attention: Basic span attention with autograd support
- fused_span_attention_gqa: GQA variant for models with grouped KV heads
- decode_span_attention_staged: Efficient decode-only path (L_Q=1)
- decode_span_attention_staged_gqa: GQA variant of staged decode
"""

from __future__ import annotations

from typing import Optional

import torch

# Re-export mask utilities
from .masks import (
    create_span_mask,
    create_sort_indices,
    create_sorted_span_mask,
    create_sliding_window_mask,
    invert_sorted_matrix,
)

# Re-export internal utilities (used by attention module)
from ._triton_forward import (
    _next_power_of_two,
    _assert_no_span_sw_overlap,
)

from superlinear.kernels.common.adjustment import compute_qend_from_qanchor
from superlinear.kernels.common.power import window_len_from_sw_index

__all__ = [
    # High-level functions
    "fused_span_attention",
    "fused_span_attention_gqa",
    "span_attention",
    "span_attention_gqa",
    "decode_span_attention_staged",
    "decode_span_attention_staged_gqa",
    "decode_span_attention",
    "decode_span_attention_gqa",
    # Mask utilities
    "create_span_mask",
    "create_sort_indices",
    "create_sorted_span_mask",
    "create_sliding_window_mask",
    "invert_sorted_matrix",
    # Adjustment utilities
    "compute_qend_from_qanchor",
    # Internal utilities (for advanced users)
    "_next_power_of_two",
    "_assert_no_span_sw_overlap",
]


def fused_span_attention(
    Q2: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    qstart: torch.Tensor,
    qend: torch.Tensor,
    cache_position: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    sw_index: int = 0,
    block_k: int = 64,
    span_len_factor: float = 2.0,
    *,
    span_power: float = 0.5,
    search_power: Optional[float] = None,
    inv_search_power_int: Optional[int] = 2,
) -> torch.Tensor:
    """
    Fused span attention with Triton kernels.

    Args:
        Q2: Query tensor [B, H, L_Q, D]
        K: Key tensor [B, H, L_KV, D]
        V: Value tensor [B, H, L_KV, D]
        qstart: Span start indices [B, H, L_Q, K] (inclusive)
        qend: Span end indices [B, H, L_Q, K] (inclusive)
        cache_position: Absolute positions for each query token [L_Q]
        attention_mask: Optional mask [B, L_KV]
        sw_index: Sliding window index
        block_k: Triton block size for KV dimension
        span_len_factor: Factor controlling max span length estimate
        span_power: Exponent for span length computation
        search_power: Optional power for search
        inv_search_power_int: Integer inverse power (2-6)

    Returns:
        Output tensor [B, H, L_Q, K, D] with attention results per span
    """
    # Import the full implementation
    from ._triton_impl import FusedSpanTriton
    
    return FusedSpanTriton.apply(
        Q2, K, V, qstart, qend, cache_position,
        attention_mask, sw_index, block_k, span_len_factor,
        float(span_power), search_power, inv_search_power_int,
    )


def span_attention(*args, **kwargs) -> torch.Tensor:
    """Alias for fused_span_attention (preferred public name)."""
    return fused_span_attention(*args, **kwargs)


def fused_span_attention_gqa(
    Q2: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    qstart: torch.Tensor,
    qend: torch.Tensor,
    cache_position: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    sw_index: int = 0,
    block_k: int = 64,
    span_len_factor: float = 2.0,
    span_power: float = 0.5,
    search_power: Optional[float] = None,
    inv_search_power_int: Optional[int] = 2,
) -> torch.Tensor:
    """
    Fused span attention for Grouped Query Attention (GQA) models.

    This variant efficiently handles models where multiple query heads share
    fewer key/value heads (e.g., H_Q=32, H_KV=8 for kv_repeat=4).

    Args:
        Q2: Query tensor [B, H_Q, L_Q, D]
        K: Key tensor [B, H_KV, L_KV, D]
        V: Value tensor [B, H_KV, L_KV, D]
        qstart: Span start indices [B, H_Q, L_Q, K]
        qend: Span end indices [B, H_Q, L_Q, K]
        cache_position: Absolute positions for each query token [L_Q]
        attention_mask: Optional mask [B, L_KV]
        sw_index: Sliding window index
        block_k: Triton block size
        span_len_factor: Factor controlling max span length estimate
        span_power: Exponent for span length computation
        search_power: Optional power for search
        inv_search_power_int: Integer inverse power (2-6)

    Returns:
        Output tensor [B, H_Q, L_Q, K, D]
    """
    from ._triton_gqa import FusedSpanGQATriton
    
    B, H_q, L_Q, _ = Q2.shape
    _, H_kv, _, _ = K.shape
    assert H_q % H_kv == 0, "Query heads must be divisible by KV heads when using GQA"

    return FusedSpanGQATriton.apply(
        Q2, K, V, qstart, qend, cache_position,
        attention_mask, sw_index, block_k, span_len_factor,
        float(span_power), search_power, inv_search_power_int,
    )


def span_attention_gqa(*args, **kwargs) -> torch.Tensor:
    """Alias for fused_span_attention_gqa (preferred public name)."""
    return fused_span_attention_gqa(*args, **kwargs)


def decode_span_attention_staged(
    Q1: torch.Tensor,
    Q2: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    cache_position: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    sw_index: int = 0,
    topk: int = 3,
    backward_factor: float = 2.0,
    forward_factor: float = 0.0,
    *,
    span_power: float = 0.5,
    search_power: Optional[float] = None,
    inv_search_power_int: Optional[int] = 2,
) -> torch.Tensor:
    """
    Staged decode attention for efficient single-token generation.

    This specialized path (L_Q=1) stages stripe keys into a contiguous buffer,
    runs a single matvec search to pick top-k spans, then applies fused span
    attention. Backward is not supported (inference-only).

    Args:
        Q1: Search query tensor [B, H, 1, D]
        Q2: Attention query tensor [B, H, 1, D]
        K: Key tensor [B, H, L_KV, D]
        V: Value tensor [B, H, L_KV, D]
        cache_position: Position of the decode token [1]
        attention_mask: Optional mask [B, L_KV]
        sw_index: Sliding window index
        topk: Number of top spans to aggregate
        backward_factor: Factor for backward span extension
        forward_factor: Factor for forward span extension
        span_power: Exponent for span length
        search_power: Optional power for search
        inv_search_power_int: Integer inverse power (2-6)

    Returns:
        Output tensor [B, H, 1, D]
    """
    from ._triton_impl import decode_span_attention_staged as _decode_staged
    
    return _decode_staged(
        Q1, Q2, K, V, cache_position,
        attention_mask=attention_mask,
        sw_index=sw_index,
        topk=topk,
        backward_factor=backward_factor,
        forward_factor=forward_factor,
        span_power=span_power,
        search_power=search_power,
        inv_search_power_int=inv_search_power_int,
    )


def decode_span_attention(*args, **kwargs) -> torch.Tensor:
    """Alias for decode_span_attention_staged (preferred public name)."""
    return decode_span_attention_staged(*args, **kwargs)


def decode_span_attention_staged_gqa(
    Q1: torch.Tensor,
    Q2: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    cache_position: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    sw_index: int = 0,
    topk: int = 3,
    enable_gqa: bool = True,
    block_k: int = 64,
    backward_factor: float = 2.0,
    forward_factor: float = 0.0,
    span_power: float = 0.5,
    search_power: Optional[float] = None,
    inv_search_power_int: Optional[int] = 2,
    force_mode: Optional[str] = None,
) -> torch.Tensor:
    """
    Staged decode attention for GQA models.

    Combines staged decode with GQA support for efficient inference
    with models that have grouped key/value heads.

    Args:
        Q1: Search query tensor [B, H_Q, 1, D]
        Q2: Attention query tensor [B, H_Q, 1, D]
        K: Key tensor [B, H_KV, L_KV, D]
        V: Value tensor [B, H_KV, L_KV, D]
        cache_position: Position of the decode token [1]
        attention_mask: Optional mask [B, L_KV]
        sw_index: Sliding window index
        topk: Number of top spans
        enable_gqa: Whether to use GQA-optimized path
        block_k: Triton block size
        backward_factor: Factor for backward span extension
        forward_factor: Factor for forward span extension
        span_power: Exponent for span length
        search_power: Optional power for search
        inv_search_power_int: Integer inverse power (2-6)
        force_mode: Force "sdpa" or "span" mode (for testing)

    Returns:
        Output tensor [B, H_Q, 1, D]
    """
    from ._triton_gqa import decode_span_attention_staged_gqa_kernel_v2
    
    return decode_span_attention_staged_gqa_kernel_v2(
        Q1, Q2, K, V, cache_position,
        attention_mask=attention_mask,
        sw_index=sw_index,
        topk=topk,
        enable_gqa=enable_gqa,
        block_k=block_k,
        backward_factor=backward_factor,
        forward_factor=forward_factor,
        span_power=span_power,
        search_power=search_power,
        inv_search_power_int=inv_search_power_int,
        force_mode=force_mode,
    )


def decode_span_attention_gqa(*args, **kwargs) -> torch.Tensor:
    """Alias for decode_span_attention_staged_gqa (preferred public name)."""
    return decode_span_attention_staged_gqa(*args, **kwargs)
