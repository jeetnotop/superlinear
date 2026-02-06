"""Public API for span search operations.

This module provides stable entry points for span search. Users should import
from here rather than from the private implementation modules.
"""

from __future__ import annotations

from typing import Optional

import torch

from superlinear.runtime import is_triton_available

# Import implementations
from ._reference import get_search_mask, get_search_scores, get_spans
from ._triton import (
    span_search_with_values,
    span_search_triton_with_values,
)
from ._triton_gqa import (
    span_search_with_values_gqa,
    span_search_triton_with_values_gqa,
)


__all__ = [
    # Primary API
    "span_search_with_values",
    "span_search_with_values_gqa",
    "span_search_triton",
    "span_search_triton_gqa",
    "search_spans",
    "search_spans_triton",
    "search_spans_triton_gqa",
    # Lower-level functions
    "span_search_triton_with_values",
    "span_search_triton_with_values_gqa",
    "get_search_mask",
    "get_search_scores",
    "get_spans",
]


def span_search(
    Q: torch.Tensor,
    K: torch.Tensor,
    cache_position: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    sw_index: int = 0,
    k: int = 3,
    block_size: int = 128,
    backward_factor: float = 2.0,
    span_power: float = 0.5,
    search_power: Optional[float] = None,
    inv_search_power_int: Optional[int] = 2,
    use_triton: Optional[bool] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Search for top-k anchor positions along power-law stripes.

    This is the main entry point for span search. It automatically selects
    the best implementation based on hardware availability.

    Args:
        Q: Query tensor [B, H, L_Q, D]
        K: Key tensor [B, H, L_KV, D]
        cache_position: Absolute positions for each query token [L_Q]
        attention_mask: Optional attention mask [B, L_KV]
        sw_index: Sliding window index (stripes with i <= sw_index are skipped)
        k: Number of top-k anchors to select (2 or 3 for Triton, any for reference)
        block_size: Triton block size (ignored for reference impl)
        backward_factor: Factor for span length computation (b_b)
        span_power: Exponent for span length (p_s, typically 0.5)
        search_power: Optional power for search (mutually exclusive with inv_search_power_int)
        inv_search_power_int: Integer inverse power (2-6), default 2
        use_triton: Force Triton (True), PyTorch (False), or auto-detect (None)

    Returns:
        qstart: [B, H, L_Q, k] start indices of each span (inclusive)
        qend: [B, H, L_Q, k] end indices / anchor positions (inclusive)
        values: [B, H, L_Q, k] attention scores at anchor positions
    """
    if Q.ndim != 4 or K.ndim != 4:
        raise ValueError(f"Expected Q and K to be rank-4 tensors [B, H, L, D] (got Q.ndim={Q.ndim}, K.ndim={K.ndim})")

    q_heads = int(Q.shape[1])
    kv_heads = int(K.shape[1])
    is_gqa = q_heads != kv_heads

    if is_gqa and (kv_heads <= 0 or q_heads % kv_heads != 0):
        raise ValueError(f"For GQA search, Q heads must be divisible by K heads (got H_Q={q_heads}, H_KV={kv_heads})")

    if use_triton is None:
        use_triton = is_triton_available() and Q.is_cuda and k in (2, 3)

    if use_triton:
        if is_gqa:
            return span_search_with_values_gqa(
                Q, K, cache_position, attention_mask,
                sw_index=sw_index,
                k=k,
                block_size=block_size,
                backward_factor=backward_factor,
                span_power=span_power,
                search_power=search_power,
                inv_search_power_int=inv_search_power_int,
            )
        return span_search_with_values(
            Q, K, cache_position, attention_mask,
            sw_index=sw_index,
            k=k,
            block_size=block_size,
            backward_factor=backward_factor,
            span_power=span_power,
            search_power=search_power,
            inv_search_power_int=inv_search_power_int,
        )

    # Reference fallback: expects matching head counts.
    if is_gqa:
        kv_repeat = q_heads // kv_heads
        K = K.repeat_interleave(kv_repeat, dim=1)

    return get_spans(
        Q, K, cache_position, attention_mask,
        sw_index=sw_index,
        topk=k,
        backward_factor=backward_factor,
        span_power=span_power,
        search_power=search_power,
        inv_search_power_int=inv_search_power_int,
    )


def search_spans(*args, **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Alias for span_search (preferred public name)."""
    return span_search(*args, **kwargs)


def span_search_triton(
    Q: torch.Tensor,
    K: torch.Tensor,
    cache_position: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    sw_index: int = 0,
    k: int = 3,
    block_size: int = 128,
    backward_factor: float = 2.0,
    span_power: float = 0.5,
    search_power: Optional[float] = None,
    inv_search_power_int: Optional[int] = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Explicit Triton span search (non-GQA)."""
    return span_search_triton_with_values(
        Q, K, cache_position, attention_mask,
        sw_index=sw_index,
        k=k,
        BLOCK_SIZE=block_size,
        backward_factor=backward_factor,
        span_power=span_power,
        search_power=search_power,
        inv_search_power_int=inv_search_power_int,
    )


def search_spans_triton(*args, **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Alias for span_search_triton."""
    return span_search_triton(*args, **kwargs)


def span_search_triton_gqa(
    Q: torch.Tensor,
    K: torch.Tensor,
    cache_position: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    sw_index: int = 0,
    k: int = 3,
    block_size: int = 128,
    backward_factor: float = 2.0,
    span_power: float = 0.5,
    search_power: Optional[float] = None,
    inv_search_power_int: Optional[int] = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Explicit Triton span search for grouped-query attention (GQA)."""
    return span_search_triton_with_values_gqa(
        Q, K, cache_position, attention_mask,
        sw_index=sw_index,
        k=k,
        BLOCK_SIZE=block_size,
        backward_factor=backward_factor,
        span_power=span_power,
        search_power=search_power,
        inv_search_power_int=inv_search_power_int,
    )


def search_spans_triton_gqa(*args, **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Alias for span_search_triton_gqa."""
    return span_search_triton_gqa(*args, **kwargs)
