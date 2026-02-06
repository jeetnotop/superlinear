"""Span adjustment utilities for computing span end positions."""

from __future__ import annotations

import math
from typing import Optional

import torch

from .power import window_len_from_sw_index


def compute_qend_from_qanchor(
    qanchor: torch.Tensor,
    *,
    cache_position: torch.Tensor,
    key_length: int,
    sw_index: int = 0,
    attention_mask: Optional[torch.Tensor] = None,
    forward_factor: float = 0.0,
    span_power: float = 0.5,
    search_power: Optional[float] = None,
    inv_search_power_int: Optional[int] = 2,
) -> torch.Tensor:
    """
    Compute span end indices (qend) from anchor indices (qanchor) with a forward factor.

    This helper is the forward-span counterpart to the search kernel's backward span length.

    Args:
        qanchor: Anchor indices (selected top-k key positions), shape [B, H, L_Q, K].
                 Uses -1 for invalid entries.
        cache_position: Absolute cache positions for each query token, shape [L_Q].
        key_length: L_KV (size of the key/value sequence dimension).
        sw_index: Sliding-window index. If >0, spans must satisfy qend < sw_start to avoid overlap.
        attention_mask: Optional mask over keys, shape [B, >=L_KV]. When provided, the first valid
                        position per batch is computed via argmax to match the search kernels.
        forward_factor: Forward extension factor (f_f). Must be >= 0.0.

    Returns:
        qend tensor with the same shape/dtype/device as qanchor.
    """
    forward_factor = float(forward_factor)
    if not math.isfinite(forward_factor) or forward_factor < 0.0:
        raise ValueError(f"forward_factor must be finite and >= 0 (got {forward_factor})")
    if key_length <= 0:
        raise ValueError(f"key_length must be > 0 (got {key_length})")
    if sw_index < 0:
        raise ValueError(f"sw_index must be >= 0 (got {sw_index})")

    if cache_position.dim() != 1:
        raise ValueError(f"cache_position must be 1D (got shape {tuple(cache_position.shape)})")

    B, _, L_Q, _ = qanchor.shape
    if cache_position.numel() != L_Q:
        raise ValueError(f"cache_position length {cache_position.numel()} must match L_Q {L_Q}")

    if forward_factor == 0.0:
        return qanchor

    device = qanchor.device
    qanchor_i64 = qanchor.to(torch.int64)

    cache_pos = cache_position.to(device=device, dtype=torch.int64).view(1, L_Q)
    cache_pos = cache_pos.clamp(min=0, max=key_length - 1)

    if attention_mask is not None:
        attn = attention_mask.to(device=device)
        if attn.dim() != 2:
            raise ValueError(f"attention_mask must be rank-2 [B, L] (got shape {tuple(attn.shape)})")
        if attn.shape[0] != B:
            raise ValueError(f"attention_mask batch {attn.shape[0]} must match qanchor batch {B}")
        attn = attn[:, :key_length].to(torch.int32)
        start_pos = attn.argmax(dim=1).to(torch.int64)
    else:
        start_pos = torch.zeros((B,), device=device, dtype=torch.int64)

    token_positions = cache_pos + 1 - start_pos[:, None]
    token_positions = token_positions.clamp(min=0)

    span_power_f = float(span_power)
    if not math.isfinite(span_power_f) or not (0.0 < span_power_f < 1.0):
        raise ValueError(f"span_power must be finite and in (0, 1) (got {span_power})")

    forward_len = torch.ceil(
        forward_factor * (token_positions.to(torch.float32) ** span_power_f)
    ).to(torch.int64)
    forward_len = forward_len[:, None, :, None]  # [B, 1, L_Q, 1]

    qend = qanchor_i64 + forward_len

    qpos = cache_pos.view(1, 1, L_Q, 1)
    qend = torch.minimum(qend, qpos)
    qend = qend.clamp(max=key_length - 1)

    window_len = window_len_from_sw_index(
        sw_index, search_power=search_power, inv_search_power_int=inv_search_power_int
    )
    if window_len > 0:
        sw_start = cache_pos - (window_len - 1)
        sw_start = sw_start.clamp(min=0)
        sw_upper = (sw_start - 1).view(1, 1, L_Q, 1)
        qend = torch.minimum(qend, sw_upper)

    qend = torch.where(qanchor_i64 < 0, -1, qend)
    return qend.to(dtype=qanchor.dtype)
