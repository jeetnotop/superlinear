"""PyTorch reference implementation of span search.

This implementation is useful for:
- Testing and validation against the Triton kernels
- Environments without GPU/Triton support
- Understanding the algorithm in pure PyTorch
"""

from __future__ import annotations

from typing import Optional

import torch

from superlinear.kernels.common.power import derive_stripe_power_params


def get_search_mask(
    cache_position: torch.Tensor,
    device: torch.device,
    batch_size: int = 1,
    attention_mask: Optional[torch.Tensor] = None,
    sw_index: int = 0,
    *,
    kv_length: Optional[int] = None,
    search_power: Optional[float] = None,
    inv_search_power_int: Optional[int] = 2,
) -> torch.Tensor:
    """
    Build a boolean mask indicating which (query, key) pairs lie on valid stripes.

    Args:
        cache_position: Absolute positions for each query token [L_Q]
        device: Target device
        batch_size: Batch dimension
        attention_mask: Optional mask [B, L_KV]
        sw_index: Sliding window index (stripes <= sw_index are excluded)
        kv_length: Key/value sequence length (defaults to cache_position[-1] + 1)
        search_power: Optional power for search
        inv_search_power_int: Integer inverse power (2-6)

    Returns:
        mask: [B, 1, L_Q, L_KV] boolean tensor
    """
    target_length = kv_length if kv_length is not None else int(cache_position[-1].item()) + 1
    if attention_mask is not None:
        attention_mask = attention_mask.to(torch.bool)

    cache_position = cache_position.to(device=device, dtype=torch.int64).flatten()
    q_pos = cache_position.view(1, 1, -1)  # [1, 1, L]
    L = q_pos.shape[-1]

    power_params = derive_stripe_power_params(
        search_power=search_power, inv_search_power_int=inv_search_power_int
    )

    # Global stripe range (conservative upper bound)
    max_q_pos = int(cache_position.max().item())
    max_i = int(torch.floor(torch.tensor(float(max_q_pos + 1)) ** power_params.p).item()) + 2
    max_i = max(max_i, int(sw_index) + 1)

    stripe_idx = torch.arange(int(sw_index) + 1, max_i + 1, device=device, dtype=torch.int64)
    if stripe_idx.numel() == 0:
        return torch.zeros((batch_size, 1, L, int(target_length)), device=device, dtype=torch.bool)

    if power_params.triton_inv_n != 0:
        stripe_floor_power = stripe_idx ** int(power_params.triton_inv_n)
    else:
        stripe_floor_power = torch.floor(stripe_idx.to(torch.float32) ** power_params.inv_p).to(torch.int64)

    # For query at position q, stripe anchor is at:
    #   k = q - floor_power + 1
    k_pos = q_pos.transpose(-1, -2) - stripe_floor_power.view(1, -1) + 1  # [L, S]
    k_pos = k_pos.to(torch.int64)

    # Valid stripes must satisfy floor_power <= q+1, otherwise k_pos < 0
    stripe_valid = (k_pos >= 0) & (k_pos < target_length)

    # Scatter into [B, 1, L, KV] mask
    mask = torch.zeros((batch_size, 1, L, int(target_length)), device=device, dtype=torch.bool)
    if stripe_valid.any():
        b_idx = torch.arange(batch_size, device=device)[:, None, None].expand(batch_size, L, stripe_idx.numel())
        q_idx = torch.arange(L, device=device)[None, :, None].expand(batch_size, L, stripe_idx.numel())
        kv_idx = k_pos.view(1, L, -1).expand(batch_size, L, -1)
        valid = stripe_valid.view(1, L, -1).expand(batch_size, L, -1)
        mask[b_idx[valid], 0, q_idx[valid], kv_idx[valid]] = True

    if attention_mask is not None:
        attn = attention_mask[:, :target_length].to(device=device, dtype=torch.bool)
        mask = mask & attn[:, None, None, :]

    return mask


def get_spans(
    Q: torch.Tensor,
    K: torch.Tensor,
    cache_position: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    sw_index: int = 0,
    topk: int = 2,
    *,
    backward_factor: float = 2.0,
    span_power: float = 0.5,
    search_power: Optional[float] = None,
    inv_search_power_int: Optional[int] = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    PyTorch reference implementation for span search.

    Args:
        Q: Query tensor [B, H, L_Q, D]
        K: Key tensor [B, H, L_KV, D]
        cache_position: Absolute positions for each query token [L_Q]
        attention_mask: Optional mask [B, L_KV]
        sw_index: Sliding window index
        topk: Number of top-k anchors to select
        backward_factor: Factor for span length computation
        span_power: Exponent for span length
        search_power: Optional power for search
        inv_search_power_int: Integer inverse power (2-6)

    Returns:
        qstart: [B, H, L_Q, topk] start indices (inclusive)
        qend: [B, H, L_Q, topk] end indices / anchors (inclusive)
        values: [B, H, L_Q, topk] attention scores at anchors
    """
    batch_size = Q.shape[0]
    device = Q.device
    dtype = Q.dtype
    int_dtype = torch.int32
    L_KV = K.shape[2]

    start_positions = (
        attention_mask.to(torch.int32).argmax(dim=1)
        if attention_mask is not None
        else torch.zeros(batch_size, device=device, dtype=int_dtype)
    )

    search_mask = get_search_mask(
        cache_position,
        device=device,
        batch_size=batch_size,
        attention_mask=attention_mask,
        sw_index=sw_index,
        kv_length=L_KV,
        search_power=search_power,
        inv_search_power_int=inv_search_power_int,
    )
    search_mask = torch.where(search_mask == 0, float('-inf'), 0.0)

    scores = Q.matmul(K.transpose(-2, -1)) + search_mask.to(dtype)
    _values, indices = torch.topk(scores, k=topk, dim=-1)
    qend = torch.where(_values != float('-inf'), indices, -1)  # inclusive

    token_positions = cache_position[None, None, :] + 1 - start_positions[:, None, None]
    token_positions = torch.where(token_positions < 0, 0, token_positions)

    span_lengths = torch.ceil(
        float(backward_factor) * (token_positions.to(torch.float32) ** float(span_power))
    ).to(int_dtype)
    qstart = qend - span_lengths[:, :, :, None]  # inclusive
    qstart = torch.where((qstart < 0) & (qend >= 0), 0, qstart)

    # Clamp qstart to start_positions
    qstart = torch.where(
        (qstart < start_positions[:, None, None, None]) & (qend >= 0),
        start_positions[:, None, None, None],
        qstart,
    )

    return qstart, qend, _values


def get_search_scores(
    Q: torch.Tensor,
    K: torch.Tensor,
    qend: torch.Tensor,
) -> torch.Tensor:
    """
    Compute attention scores at the anchor positions (qend).

    Args:
        Q: Query tensor [B, H, L_Q, D]
        K: Key tensor [B, H, L_KV, D]
        qend: Anchor indices [B, H, L_Q, k]

    Returns:
        scores: [B, H, L_Q, k] attention scores
    """
    device = Q.device
    batch_size = Q.shape[0]
    num_heads = Q.shape[1]

    batch_indices = torch.arange(batch_size, device=device)[:, None, None, None]
    head_indices = torch.arange(num_heads, device=device)[None, :, None, None]

    Q_expanded = Q.unsqueeze(3)  # [B, H, L, 1, D]
    gathered_K = K[batch_indices, head_indices, qend]  # [B, H, L, k, D]

    scores = Q_expanded.matmul(gathered_K.transpose(-2, -1)).squeeze(-2)  # [B, H, L, k]
    scores = torch.where(qend < 0, float('-inf'), scores)

    return scores
