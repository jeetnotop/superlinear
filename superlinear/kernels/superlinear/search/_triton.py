"""Triton implementation of span search with values.

This is the optimized GPU implementation for searching and selecting top-k anchor positions
along stripes defined by a power-law schedule.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from superlinear.kernels.common.power import derive_stripe_power_params, max_stripe_index_for_token_pos


@triton.jit
def span_search_with_values_kernel(
    Q_ptr, K_ptr,
    attention_mask_ptr, start_positions_ptr,
    qstart_out_ptr, qend_out_ptr, values_out_ptr,
    batch_size, num_heads, query_length, key_length,
    cache_position_start, sw_index,
    max_iters,
    backward_factor,
    inv_p,
    span_power,
    BLOCK_SIZE: tl.constexpr, D_SIZE: tl.constexpr,
    K_VAL: tl.constexpr,
    HAS_ATTN_MASK: tl.constexpr,
    INV_N: tl.constexpr,
):
    pid = tl.program_id(0)

    queries_per_batch_head = (query_length + BLOCK_SIZE - 1) // BLOCK_SIZE
    batch_head_idx = pid // queries_per_batch_head
    query_block_idx = pid % queries_per_batch_head

    batch_idx = batch_head_idx // num_heads
    head_idx = batch_head_idx % num_heads

    Q_block_start = query_block_idx * BLOCK_SIZE
    q_offsets = Q_block_start + tl.arange(0, BLOCK_SIZE)
    q_mask = q_offsets < query_length

    absolute_positions = cache_position_start + q_offsets

    # Use 64-bit arithmetic throughout to avoid overflow when num_heads * key_length * D is large
    # Cast to int64 BEFORE multiplication to prevent int32 overflow
    batch_idx_i64 = batch_idx.to(tl.int64)
    head_idx_i64 = head_idx.to(tl.int64)
    num_heads_i64 = tl.cast(num_heads, tl.int64)
    query_length_i64 = tl.cast(query_length, tl.int64)
    key_length_i64 = tl.cast(key_length, tl.int64)
    D_SIZE_i64 = tl.cast(D_SIZE, tl.int64)

    Q_base = Q_ptr + batch_idx_i64 * (num_heads_i64 * query_length_i64 * D_SIZE_i64) + head_idx_i64 * (query_length_i64 * D_SIZE_i64)
    Q_offsets = q_offsets[:, None].to(tl.int64) * D_SIZE_i64 + tl.arange(0, D_SIZE)[None, :].to(tl.int64)
    d_mask = tl.arange(0, D_SIZE)[None, :] < D_SIZE
    Q_load_mask = q_mask[:, None] & d_mask
    Q_block = tl.load(Q_base + Q_offsets, mask=Q_load_mask, other=0.0)

    top_scores_0 = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)
    top_indices_0 = tl.full((BLOCK_SIZE,), -1, dtype=tl.int32)

    top_scores_1 = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)
    top_indices_1 = tl.full((BLOCK_SIZE,), -1, dtype=tl.int32)

    if K_VAL == 3:
        top_scores_2 = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)
        top_indices_2 = tl.full((BLOCK_SIZE,), -1, dtype=tl.int32)

    K_base = K_ptr + batch_idx_i64 * (num_heads_i64 * key_length_i64 * D_SIZE_i64) + head_idx_i64 * (key_length_i64 * D_SIZE_i64)
    start_pos = tl.load(start_positions_ptr + batch_idx).to(tl.int32)
    start_pos_vec = start_pos + tl.zeros((BLOCK_SIZE,), dtype=tl.int32)

    for i in tl.range(max_iters, sw_index, -1):
        i_i64 = tl.cast(i, tl.int64)

        if INV_N == 2:
            floor_power = i_i64 * i_i64
        elif INV_N == 3:
            floor_power = i_i64 * i_i64 * i_i64
        elif INV_N == 4:
            i2 = i_i64 * i_i64
            floor_power = i2 * i2
        elif INV_N == 5:
            i2 = i_i64 * i_i64
            i4 = i2 * i2
            floor_power = i4 * i_i64
        elif INV_N == 6:
            i2 = i_i64 * i_i64
            i3 = i2 * i_i64
            floor_power = i3 * i3
        else:
            i_float = tl.cast(i, tl.float32)
            floor_power = tl.math.floor(tl.exp(inv_p * tl.log(i_float))).to(tl.int64)

        shift = floor_power - 1

        k_positions_i64 = absolute_positions.to(tl.int64) - shift
        k_valid = (k_positions_i64 >= 0) & (k_positions_i64 < key_length_i64)
        k_positions = k_positions_i64.to(tl.int32)

        K_offsets = k_positions_i64[:, None] * D_SIZE_i64 + tl.arange(0, D_SIZE)[None, :].to(tl.int64)
        K_load_mask = k_valid[:, None] & d_mask
        K_block = tl.load(K_base + K_offsets, mask=K_load_mask, other=0.0)

        acc = tl.sum(Q_block.to(tl.float32) * K_block.to(tl.float32), axis=1)

        stripe_valid = k_valid
        if HAS_ATTN_MASK:
            attn_offsets = batch_idx * key_length + k_positions
            attn_mask = tl.load(attention_mask_ptr + attn_offsets, mask=k_valid, other=0).to(tl.int1)
            stripe_valid = stripe_valid & attn_mask

        acc = tl.where(stripe_valid, acc, float("-inf"))

        if K_VAL == 2:
            insert_at_0 = acc > top_scores_0
            insert_at_1 = (acc > top_scores_1) & ~insert_at_0

            top_scores_1 = tl.where(insert_at_0, top_scores_0, tl.where(insert_at_1, acc, top_scores_1))
            top_indices_1 = tl.where(insert_at_0, top_indices_0, tl.where(insert_at_1, k_positions, top_indices_1))

            top_scores_0 = tl.where(insert_at_0, acc, top_scores_0)
            top_indices_0 = tl.where(insert_at_0, k_positions, top_indices_0)
        else:
            insert_at_0 = acc > top_scores_0
            insert_at_1 = (acc > top_scores_1) & (acc <= top_scores_0)
            insert_at_2 = (acc > top_scores_2) & (acc <= top_scores_1)

            top_scores_2 = tl.where(acc > top_scores_1, top_scores_1, tl.where(insert_at_2, acc, top_scores_2))
            top_indices_2 = tl.where(acc > top_scores_1, top_indices_1, tl.where(insert_at_2, k_positions, top_indices_2))

            top_scores_1 = tl.where(acc > top_scores_0, top_scores_0, tl.where(insert_at_1, acc, top_scores_1))
            top_indices_1 = tl.where(acc > top_scores_0, top_indices_0, tl.where(insert_at_1, k_positions, top_indices_1))

            top_scores_0 = tl.where(insert_at_0, acc, top_scores_0)
            top_indices_0 = tl.where(insert_at_0, k_positions, top_indices_0)

    token_positions = absolute_positions + 1 - start_pos_vec
    token_positions = tl.where(token_positions < 0, 0, token_positions)
    backward_factor_f = tl.full((1,), backward_factor, tl.float32)
    span_power_f = tl.full((1,), span_power, tl.float32)
    tp_f = token_positions.to(tl.float32)
    tp_pow = tl.where(tp_f > 0, tl.exp(span_power_f * tl.log(tp_f)), 0.0)
    span_lengths = tl.math.ceil(backward_factor_f * tp_pow).to(tl.int32)
    span_lengths = tl.where(q_mask, span_lengths, 0)

    qstart_0 = top_indices_0 - span_lengths
    qstart_1 = top_indices_1 - span_lengths
    if K_VAL == 3:
        qstart_2 = top_indices_2 - span_lengths

    qstart_0 = tl.where((qstart_0 < 0) & (top_indices_0 >= 0), 0, qstart_0)
    qstart_0 = tl.where((qstart_0 < start_pos_vec) & (top_indices_0 >= 0), start_pos_vec, qstart_0)

    qstart_1 = tl.where((qstart_1 < 0) & (top_indices_1 >= 0), 0, qstart_1)
    qstart_1 = tl.where((qstart_1 < start_pos_vec) & (top_indices_1 >= 0), start_pos_vec, qstart_1)

    if K_VAL == 3:
        qstart_2 = tl.where((qstart_2 < 0) & (top_indices_2 >= 0), 0, qstart_2)
        qstart_2 = tl.where((qstart_2 < start_pos_vec) & (top_indices_2 >= 0), start_pos_vec, qstart_2)

    qend_base = qend_out_ptr + batch_idx * (num_heads * query_length * K_VAL) + head_idx * (query_length * K_VAL)
    qstart_base = qstart_out_ptr + batch_idx * (num_heads * query_length * K_VAL) + head_idx * (query_length * K_VAL)
    values_base = values_out_ptr + batch_idx * (num_heads * query_length * K_VAL) + head_idx * (query_length * K_VAL)

    q_offsets_flat = q_offsets * K_VAL
    tl.store(qend_base + q_offsets_flat + 0, top_indices_0, mask=q_mask)
    tl.store(qend_base + q_offsets_flat + 1, top_indices_1, mask=q_mask)
    if K_VAL == 3:
        tl.store(qend_base + q_offsets_flat + 2, top_indices_2, mask=q_mask)

    tl.store(qstart_base + q_offsets_flat + 0, qstart_0, mask=q_mask)
    tl.store(qstart_base + q_offsets_flat + 1, qstart_1, mask=q_mask)
    if K_VAL == 3:
        tl.store(qstart_base + q_offsets_flat + 2, qstart_2, mask=q_mask)

    tl.store(values_base + q_offsets_flat + 0, top_scores_0, mask=q_mask)
    tl.store(values_base + q_offsets_flat + 1, top_scores_1, mask=q_mask)
    if K_VAL == 3:
        tl.store(values_base + q_offsets_flat + 2, top_scores_2, mask=q_mask)


def _cache_position_type(cache_position: torch.Tensor) -> str:
    """Determine the type of cache position sequence."""
    device = cache_position.device
    if len(cache_position) == 1:
        return "single"
    if torch.all(cache_position == torch.arange(len(cache_position), device=device, dtype=cache_position.dtype)):
        return "full"
    if torch.all(cache_position == cache_position[0] + torch.arange(len(cache_position), device=device, dtype=cache_position.dtype)):
        return "partial_contiguous"
    return "partial_discontiguous"


def span_search_triton_with_values_contiguous(
    Q: torch.Tensor,
    K: torch.Tensor,
    cache_position: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    sw_index: int = 0,
    k: int = 3,
    BLOCK_SIZE: int = 128,
    backward_factor: float = 2.0,
    span_power: float = 0.5,
    search_power: Optional[float] = None,
    inv_search_power_int: Optional[int] = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Triton-accelerated span search for contiguous cache positions.

    This is the core implementation. For non-contiguous positions, use
    span_search_triton_with_values which handles the expansion/contraction.
    """
    batch_size, num_heads, query_length, D = Q.shape
    _, _, key_length, _ = K.shape

    backward_factor = float(backward_factor)
    if backward_factor <= 0.0:
        raise ValueError(f"backward_factor must be > 0 (got {backward_factor})")

    assert len(cache_position) == query_length, "cache_position length must match query_length"
    assert torch.all(cache_position[1:] == cache_position[:-1] + 1), "cache_position must be contiguous"
    assert D <= 256, f"This kernel is optimized for small D (got D={D})"
    assert k in (2, 3), "This kernel supports k=2 or k=3"
    assert key_length >= cache_position[-1] + 1, "K must cover the largest cache position"

    Qc = Q.contiguous()
    Kc = K.contiguous()

    cache_position_start = int(cache_position[0].item())

    has_attention_mask = attention_mask is not None
    if has_attention_mask:
        attention_mask_tensor = attention_mask[:, :key_length].to(torch.int32).contiguous()
        start_positions = attention_mask_tensor.argmax(dim=1).to(torch.int32)
    else:
        attention_mask_tensor = torch.empty((1,), device=Q.device, dtype=torch.int32)
        start_positions = torch.zeros(batch_size, device=Q.device, dtype=torch.int32)

    qstart_out = torch.full((batch_size, num_heads, query_length, k), -1, device=Q.device, dtype=torch.int32)
    qend_out = torch.full((batch_size, num_heads, query_length, k), -1, device=Q.device, dtype=torch.int32)
    values_out = torch.full((batch_size, num_heads, query_length, k), float('-inf'), device=Q.device, dtype=torch.float32)

    max_absolute_position = cache_position_start + query_length - 1
    power_params = derive_stripe_power_params(
        search_power=search_power, inv_search_power_int=inv_search_power_int
    )
    max_stripe_idx = max_stripe_index_for_token_pos(
        max_absolute_position + 1,
        search_power=search_power,
        inv_search_power_int=inv_search_power_int,
    )
    max_iters = int(max_stripe_idx) + 1

    num_query_blocks = (query_length + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (batch_size * num_heads * num_query_blocks,)

    span_search_with_values_kernel[grid](
        Qc, Kc,
        attention_mask_tensor, start_positions,
        qstart_out, qend_out, values_out,
        batch_size, num_heads, query_length, key_length,
        cache_position_start, sw_index,
        max_iters,
        backward_factor,
        power_params.inv_p,
        float(span_power),
        BLOCK_SIZE, D,
        K_VAL=k,
        HAS_ATTN_MASK=has_attention_mask,
        INV_N=power_params.triton_inv_n,
    )

    return qstart_out, qend_out, values_out


def span_search_triton_with_values(
    Q: torch.Tensor,
    K: torch.Tensor,
    cache_position: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    sw_index: int = 0,
    k: int = 3,
    BLOCK_SIZE: int = 128,
    backward_factor: float = 2.0,
    span_power: float = 0.5,
    search_power: Optional[float] = None,
    inv_search_power_int: Optional[int] = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Triton-accelerated span search supporting both contiguous and discontiguous positions.

    Returns:
        qstart: [B, H, L_Q, k] start indices of each span (inclusive)
        qend: [B, H, L_Q, k] end indices of each span (inclusive, i.e., the anchor)
        values: [B, H, L_Q, k] attention scores at the anchor positions
    """
    assert len(cache_position) > 0, "cache_position cannot be empty"
    assert k in (2, 3), "Triton span search supports k=2 or k=3"
    assert Q.shape[2] == len(cache_position), f"Q length {Q.shape[2]} must match cache_position length {len(cache_position)}"
    assert K.shape[2] >= cache_position[-1] + 1, f"K length {K.shape[2]} must be at least {cache_position[-1] + 1}"

    cache_position_type = _cache_position_type(cache_position)
    starting_cache_position = int(cache_position.min().item())
    ending_cache_position = int(cache_position.max().item())

    if cache_position_type == 'partial_discontiguous':
        batch_size, num_heads, _, D = Q.shape
        device = Q.device
        dtype = Q.dtype

        contiguous_cache_position = torch.arange(starting_cache_position, ending_cache_position + 1,
                                                 device=cache_position.device, dtype=cache_position.dtype)

        Q_expanded = torch.zeros(batch_size, num_heads, len(contiguous_cache_position), D, device=device, dtype=dtype)
        Q_expanded[:, :, cache_position - starting_cache_position, :] = Q

        target_length = contiguous_cache_position[-1] + 1
        K_sliced = K[:, :, :target_length, :]
        if attention_mask is not None:
            attention_mask_to_use = attention_mask[:, :target_length]
        else:
            attention_mask_to_use = None

        qstart_expanded, qend_expanded, values_expanded = span_search_triton_with_values_contiguous(
            Q_expanded, K_sliced, contiguous_cache_position,
            attention_mask=attention_mask_to_use,
            sw_index=sw_index,
            k=k,
            BLOCK_SIZE=BLOCK_SIZE,
            backward_factor=backward_factor,
            span_power=span_power,
            search_power=search_power,
            inv_search_power_int=inv_search_power_int,
        )

        indices = cache_position - starting_cache_position
        qstart = qstart_expanded[:, :, indices, :]
        qend = qend_expanded[:, :, indices, :]
        values = values_expanded[:, :, indices, :]
    else:
        qstart, qend, values = span_search_triton_with_values_contiguous(
            Q,
            K,
            cache_position,
            attention_mask=attention_mask,
            sw_index=sw_index,
            k=k,
            BLOCK_SIZE=BLOCK_SIZE,
            backward_factor=backward_factor,
            span_power=span_power,
            search_power=search_power,
            inv_search_power_int=inv_search_power_int,
        )

    return qstart, qend, values


class SpanSearchWithValuesFn(torch.autograd.Function):
    """Autograd wrapper for span search with backward pass for gradients."""

    @staticmethod
    def forward(
        ctx,
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
    ):
        qstart, qend, values = span_search_triton_with_values(
            Q,
            K,
            cache_position,
            attention_mask,
            sw_index,
            k,
            block_size,
            backward_factor=backward_factor,
            span_power=span_power,
            search_power=search_power,
            inv_search_power_int=inv_search_power_int,
        )
        ctx.save_for_backward(Q, K, qend)
        return qstart, qend, values

    @staticmethod
    def backward(ctx, grad_qstart, grad_qend, grad_values):
        Q, K, qend = ctx.saved_tensors

        needs_dQ = ctx.needs_input_grad[0]
        needs_dK = ctx.needs_input_grad[1]

        if grad_values is None:
            grad_values = torch.zeros_like(qend, dtype=Q.dtype)

        grad_values = torch.nan_to_num(grad_values, nan=0.0, posinf=0.0, neginf=0.0)
        grad_values = grad_values.to(torch.float32)

        mask = (qend >= 0)
        grad_values = grad_values * mask

        B, H, L_Q, k = qend.shape
        D = Q.shape[-1]
        qend_long = qend.clamp(min=0).to(torch.int64)

        batch_idx = torch.arange(B, device=Q.device)[:, None, None, None]
        head_idx = torch.arange(H, device=Q.device)[None, :, None, None]

        dQ = dK = None

        if needs_dQ:
            gathered_K = K[batch_idx, head_idx, qend_long]  # [B, H, L_Q, k, D]
            gathered_K = gathered_K * mask.unsqueeze(-1)
            dQ = torch.sum(grad_values.unsqueeze(-1) * gathered_K, dim=3)
            dQ = dQ.to(Q.dtype)

        if needs_dK:
            grad_term = grad_values.unsqueeze(-1) * Q.unsqueeze(3)  # [B, H, L_Q, k, D]
            grad_term = grad_term * mask.unsqueeze(-1)
            grad_term_flat = grad_term.reshape(B, H, -1, D)
            index_flat = qend_long.reshape(B, H, -1, 1).expand(-1, -1, -1, D)

            dK = torch.zeros_like(K, dtype=torch.float32)
            dK.scatter_add_(2, index_flat, grad_term_flat)
            dK = dK.to(K.dtype)

        return dQ, dK, None, None, None, None, None, None, None, None, None


def span_search_with_values(
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
    """
    Span search with autograd support.

    Args:
        Q: Query tensor [B, H, L_Q, D]
        K: Key tensor [B, H, L_KV, D]
        cache_position: Absolute positions for each query token [L_Q]
        attention_mask: Optional mask [B, L_KV]
        sw_index: Sliding window index (stripes with i <= sw_index are skipped)
        k: Number of top-k anchors to select (2 or 3)
        block_size: Triton block size
        backward_factor: Factor for span length computation
        span_power: Exponent for span length (typically 0.5)
        search_power: Optional power for search (mutually exclusive with inv_search_power_int)
        inv_search_power_int: Integer inverse power (2-6), default 2

    Returns:
        qstart: [B, H, L_Q, k] start indices of each span (inclusive)
        qend: [B, H, L_Q, k] end indices / anchors (inclusive)
        values: [B, H, L_Q, k] attention scores at anchors
    """
    return SpanSearchWithValuesFn.apply(
        Q,
        K,
        cache_position,
        attention_mask,
        sw_index,
        k,
        block_size,
        backward_factor,
        float(span_power),
        search_power,
        inv_search_power_int,
    )
