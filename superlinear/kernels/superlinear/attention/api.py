"""Public API for attention operations.

This module provides the main entry points for superlinear attention.
These are the primary functions users should interact with:

- fused_prefill_with_swtriton_gqa: Main prefill function for training/inference
- fused_prefill_with_swtriton_bucketed_gqa: Optimized bucketed variant
- decode_span_attention_staged_gqa: Efficient decode for autoregressive generation
"""

from __future__ import annotations

from typing import Optional

import torch

# Re-export decode functions from span module
from superlinear.kernels.superlinear.span import (
    decode_span_attention_staged,
    decode_span_attention_staged_gqa,
)

__all__ = [
    # Main prefill entry points
    "build_sw_blockmask",
    "fused_prefill_with_swflex",
    "fused_prefill_with_swflex_gqa",
    "fused_prefill_with_swtriton",
    "fused_prefill_with_swtriton_gqa",
    "fused_prefill_with_swtriton_bucketed_gqa",
    "prefill",
    "prefill_gqa",
    "prefill_bucketed_gqa",
    "prefill_mha",
    "prefill_bucketed",
    # Decode entry points
    "decode_span_attention_staged",
    "decode_span_attention_staged_gqa",
    "decode_attention",
    "decode_attention_gqa",
    # Lower-level prefill functions (for advanced use)
    "full_span_attention_fused_with_search_values",
    "full_span_attention_fused_with_search_values_gqa",
]


def build_sw_blockmask(*args, **kwargs):
    """Public helper for constructing the flex-attention sliding-window block mask."""
    from ._prefill import build_sw_blockmask as _impl

    return _impl(*args, **kwargs)


def fused_prefill_with_swflex(*args, **kwargs) -> torch.Tensor:
    """FlexAttention-based prefill (non-GQA)."""
    from ._prefill import fused_prefill_with_swflex as _impl

    return _impl(*args, **kwargs)


def fused_prefill_with_swflex_gqa(*args, **kwargs) -> torch.Tensor:
    """FlexAttention-based prefill (GQA)."""
    from ._prefill import fused_prefill_with_swflex_gqa as _impl

    return _impl(*args, **kwargs)


def fused_prefill_with_swtriton(*args, **kwargs) -> torch.Tensor:
    """Triton sliding-window prefill (non-GQA)."""
    from ._prefill import fused_prefill_with_swtriton as _impl

    return _impl(*args, **kwargs)


def fused_prefill_with_swtriton_gqa(
    Q1: torch.Tensor,
    Q2: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    cache_pos: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    sw_index: int = 0,
    num_spans: int = 3,
    block_k: int = 64,
    search_block_q: int = 128,
    backward_factor: float = 2.0,
    forward_factor: float = 0.0,
    span_power: float = 0.5,
    search_power: Optional[float] = None,
    inv_search_power_int: Optional[int] = 2,
    *,
    cache_position: Optional[torch.Tensor] = None,
    topk: Optional[int] = None,
) -> torch.Tensor:
    """
    Fused prefill attention with Triton sliding window + span attention.

    This is the main entry point for prefill operations. It combines:
    1. Span search to find top-k anchor positions
    2. Span attention over those anchor regions
    3. Sliding window attention for local context
    4. Gating to combine span outputs

    Args:
        Q1: Search/gate query tensor [B, H_Q, L_Q, D]
        Q2: Attention query tensor [B, H_Q, L_Q, D]
        K: Key tensor [B, H_KV, L_KV, D]
        V: Value tensor [B, H_KV, L_KV, D]
        cache_pos: Absolute positions for each query token [L_Q]
        attention_mask: Optional mask [B, L_KV]
        sw_index: Sliding window index
        num_spans: Number of top spans to aggregate
        block_k: Triton block size for KV
        search_block_q: Block size for search kernel
        backward_factor: Factor for backward span extension
        forward_factor: Factor for forward span extension
        span_power: Exponent for span length (typically 0.5)
        search_power: Optional power for search
        inv_search_power_int: Integer inverse power (2-6), default 2

    Returns:
        Output tensor [B, H_Q, L_Q, D]
    """
    from ._prefill import fused_prefill_with_swtriton_gqa as _impl

    if cache_position is not None:
        cache_pos = cache_position

    if cache_pos is None:
        raise TypeError(
            "fused_prefill_with_swtriton_gqa() missing required argument: "
            "provide either 'cache_pos' or 'cache_position'"
        )
    if topk is not None:
        num_spans = int(topk)
    
    return _impl(
        Q1, Q2, K, V, cache_pos,
        attention_mask=attention_mask,
        sw_index=sw_index,
        topk=num_spans,
        backward_factor=backward_factor,
        forward_factor=forward_factor,
        span_power=span_power,
        search_power=search_power,
        inv_search_power_int=inv_search_power_int,
    )


def prefill(
    Q1: torch.Tensor,
    Q2: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    cache_position: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    sw_index: int = 0,
    num_spans: int = 3,
    block_k: int = 64,
    search_block_q: int = 128,
    backward_factor: float = 2.0,
    forward_factor: float = 0.0,
    span_power: float = 0.5,
    search_power: Optional[float] = None,
    inv_search_power_int: Optional[int] = 2,
    *,
    topk: Optional[int] = None,
) -> torch.Tensor:
    """Preferred public name for prefill.

    Dispatches between MHA (H_Q == H_KV) and GQA (H_Q != H_KV) based on tensor shapes.
    """
    if topk is not None:
        num_spans = int(topk)
    if Q1.shape[1] == K.shape[1]:
        from ._prefill import fused_prefill_with_swtriton as _impl
        return _impl(
            Q1, Q2, K, V, cache_position,
            attention_mask=attention_mask,
            sw_index=sw_index,
            topk=num_spans,
            backward_factor=backward_factor,
            forward_factor=forward_factor,
            span_power=span_power,
            search_power=search_power,
            inv_search_power_int=inv_search_power_int,
        )

    return fused_prefill_with_swtriton_gqa(
        Q1, Q2, K, V, cache_position,
        attention_mask=attention_mask,
        sw_index=sw_index,
        num_spans=num_spans,
        block_k=block_k,
        search_block_q=search_block_q,
        backward_factor=backward_factor,
        forward_factor=forward_factor,
        span_power=span_power,
        search_power=search_power,
        inv_search_power_int=inv_search_power_int,
    )


def prefill_mha(
    Q1: torch.Tensor,
    Q2: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    cache_position: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    sw_index: int = 0,
    num_spans: int = 3,
    block_k: int = 64,
    search_block_q: int = 128,
    backward_factor: float = 2.0,
    forward_factor: float = 0.0,
    span_power: float = 0.5,
    search_power: Optional[float] = None,
    inv_search_power_int: Optional[int] = 2,
    *,
    topk: Optional[int] = None,
) -> torch.Tensor:
    """Explicit MHA prefill (H_Q == H_KV)."""
    from ._prefill import fused_prefill_with_swtriton as _impl
    if topk is not None:
        num_spans = int(topk)
    return _impl(
        Q1, Q2, K, V, cache_position,
        attention_mask=attention_mask,
        sw_index=sw_index,
        topk=num_spans,
        backward_factor=backward_factor,
        forward_factor=forward_factor,
        span_power=span_power,
        search_power=search_power,
        inv_search_power_int=inv_search_power_int,
    )


def prefill_gqa(*args, **kwargs) -> torch.Tensor:
    """Alias for fused_prefill_with_swtriton_gqa (preferred public name)."""
    return fused_prefill_with_swtriton_gqa(*args, **kwargs)


def fused_prefill_with_swtriton_bucketed_gqa(
    Q1: torch.Tensor,
    Q2: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    cache_pos: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    sw_index: int = 0,
    num_spans: int = 3,
    block_k: int = 64,
    search_block_q: int = 128,
    backward_factor: float = 2.0,
    forward_factor: float = 0.0,
    span_power: float = 0.5,
    search_power: Optional[float] = None,
    inv_search_power_int: Optional[int] = 2,
    *,
    cache_position: Optional[torch.Tensor] = None,
    topk: Optional[int] = None,
) -> torch.Tensor:
    """
    Bucketed variant of fused prefill for better memory efficiency.

    This variant uses tile-based processing with independent span handling,
    which can be more efficient for certain sequence lengths and hardware.

    Args:
        Same as fused_prefill_with_swtriton_gqa

    Returns:
        Output tensor [B, H_Q, L_Q, D]
    """
    from ._prefill import fused_prefill_with_swtriton_bucketed_gqa as _impl

    if cache_position is not None:
        cache_pos = cache_position

    if cache_pos is None:
        raise TypeError(
            "fused_prefill_with_swtriton_bucketed_gqa() missing required argument: "
            "provide either 'cache_pos' or 'cache_position'"
        )
    if topk is not None:
        num_spans = int(topk)
    
    return _impl(
        Q1, Q2, K, V, cache_pos,
        attention_mask=attention_mask,
        sw_index=sw_index,
        topk=num_spans,
        backward_factor=backward_factor,
        forward_factor=forward_factor,
        span_power=span_power,
        search_power=search_power,
        inv_search_power_int=inv_search_power_int,
    )


def prefill_bucketed_gqa(*args, **kwargs) -> torch.Tensor:
    """Alias for fused_prefill_with_swtriton_bucketed_gqa (preferred public name)."""
    return fused_prefill_with_swtriton_bucketed_gqa(*args, **kwargs)


def prefill_bucketed(*args, **kwargs) -> torch.Tensor:
    """Alias for prefill_bucketed_gqa (bucketed path is currently GQA-first)."""
    return prefill_bucketed_gqa(*args, **kwargs)


def decode_attention(*args, **kwargs) -> torch.Tensor:
    """Alias for decode_span_attention_staged (preferred shorter name)."""
    return decode_span_attention_staged(*args, **kwargs)


def decode_attention_gqa(*args, **kwargs) -> torch.Tensor:
    """Alias for decode_span_attention_staged_gqa (preferred shorter name)."""
    return decode_span_attention_staged_gqa(*args, **kwargs)


def full_span_attention_fused_with_search_values(
    Q1: torch.Tensor,
    Q2: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    cache_pos: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    sw_index: int = 0,
    num_spans: int = 3,
    block_k: int = 64,
    search_block_q: int = 128,
    backward_factor: float = 2.0,
    forward_factor: float = 0.0,
    span_power: float = 0.5,
    search_power: Optional[float] = None,
    inv_search_power_int: Optional[int] = 2,
    *,
    cache_position: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Non-GQA variant of fused span attention (Q/K/V have same head count).

    For models with H_Q == H_KV (standard multi-head attention without GQA).

    Args:
        Same as fused_prefill_with_swtriton_gqa, but K/V have shape [B, H, L_KV, D]

    Returns:
        Output tensor [B, H, L_Q, D]
    """
    from ._prefill import full_span_attention_fused_with_search_values as _impl

    if cache_position is not None:
        cache_pos = cache_position

    if cache_pos is None:
        raise TypeError(
            "full_span_attention_fused_with_search_values() missing required argument: "
            "provide either 'cache_pos' or 'cache_position'"
        )
    
    return _impl(
        Q1, Q2, K, V, cache_pos,
        attention_mask=attention_mask,
        sw_index=sw_index,
        num_spans=num_spans,
        block_k=block_k,
        search_block_q=search_block_q,
        backward_factor=backward_factor,
        forward_factor=forward_factor,
        span_power=span_power,
        search_power=search_power,
        inv_search_power_int=inv_search_power_int,
    )


def full_span_attention_fused_with_search_values_gqa(
    Q1: torch.Tensor,
    Q2: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    cache_pos: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    sw_index: int = 0,
    num_spans: int = 3,
    block_k: int = 64,
    search_block_q: int = 128,
    backward_factor: float = 2.0,
    forward_factor: float = 0.0,
    span_power: float = 0.5,
    search_power: Optional[float] = None,
    inv_search_power_int: Optional[int] = 2,
    *,
    cache_position: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    GQA variant of fused span attention with search value reuse.

    This is the lower-level function that fused_prefill_with_swtriton_gqa builds upon.
    Use fused_prefill_with_swtriton_gqa for most use cases.

    Args:
        Same as fused_prefill_with_swtriton_gqa

    Returns:
        Output tensor [B, H_Q, L_Q, D]
    """
    from ._prefill import full_span_attention_fused_with_search_values_gqa as _impl

    if cache_position is not None:
        cache_pos = cache_position

    if cache_pos is None:
        raise TypeError(
            "full_span_attention_fused_with_search_values_gqa() missing required argument: "
            "provide either 'cache_pos' or 'cache_position'"
        )
    
    return _impl(
        Q1, Q2, K, V, cache_pos,
        attention_mask=attention_mask,
        sw_index=sw_index,
        num_spans=num_spans,
        block_k=block_k,
        search_block_q=search_block_q,
        backward_factor=backward_factor,
        forward_factor=forward_factor,
        span_power=span_power,
        search_power=search_power,
        inv_search_power_int=inv_search_power_int,
    )
