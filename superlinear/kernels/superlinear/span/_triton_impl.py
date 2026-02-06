import math
import os

import torch
import triton
import triton.language as tl

from superlinear.kernels.common.adjustment import compute_qend_from_qanchor
from superlinear.kernels.common.power import derive_stripe_power_params, max_stripe_index_for_token_pos, window_len_from_sw_index

# Disable assertions by default (required for CUDA graph capture)
# Set SPAN_ATTENTION_ENABLE_ASSERTS=1 to enable validation checks
_DISABLE_ASSERTS = os.getenv("SPAN_ATTENTION_ENABLE_ASSERTS", "0") != "1"

@triton.jit
def fused_span_forward_kernel(
    Q_ptr, K_ptr, V_ptr,
    qstart_ptr, qend_ptr, cache_position_ptr,
    attn_mask_ptr,
    Out_ptr,
    B, H, L_Q, L_KV,
    window_size,
    sm_scale,
    K_VAL: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D_HEAD: tl.constexpr,
    SPAN_MAX_BLOCKS: tl.constexpr,
    SW_MAX_BLOCKS: tl.constexpr,
    HAS_ATTN_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    span_index = pid % K_VAL
    q_idx = (pid // K_VAL) % L_Q
    head_idx = (pid // (K_VAL * L_Q)) % H
    batch_idx = pid // (K_VAL * L_Q * H)

    d_range = tl.arange(0, BLOCK_D)
    d_mask = d_range < D_HEAD

    q_base = Q_ptr + ((batch_idx * H + head_idx) * L_Q + q_idx) * D_HEAD
    q = tl.load(q_base + d_range, mask=d_mask, other=0.0).to(tl.float32)

    cache_pos = tl.load(cache_position_ptr + q_idx, mask=True, other=0).to(tl.int32)
    cache_pos = tl.minimum(cache_pos, L_KV - 1)

    span_offset = ((batch_idx * H + head_idx) * L_Q + q_idx) * K_VAL + span_index
    span_start = tl.load(qstart_ptr + span_offset, mask=True, other=-1).to(tl.int32)
    span_end = tl.load(qend_ptr + span_offset, mask=True, other=-1).to(tl.int32)
    span_start = tl.maximum(span_start, 0)
    span_end = tl.minimum(span_end, L_KV - 1)
    span_valid = (span_end >= span_start) & (span_end >= 0)

    window = window_size
    sw_end = tl.minimum(cache_pos, L_KV - 1)
    sw_start = sw_end - (window - 1)
    sw_start = tl.maximum(sw_start, 0)
    sw_valid = (window > 0) & (sw_start <= sw_end) & (sw_end >= 0)

    seg1_start, seg1_end, seg1_valid = span_start, span_end, span_valid
    seg2_start, seg2_end, seg2_valid = sw_start, sw_end, sw_valid

    seg1_start = tl.maximum(seg1_start, 0)
    seg1_end = tl.minimum(seg1_end, L_KV - 1)
    seg1_valid = seg1_valid & (seg1_start <= seg1_end)

    seg2_start = tl.maximum(seg2_start, 0)
    seg2_end = tl.minimum(seg2_end, L_KV - 1)
    seg2_valid = seg2_valid & (seg2_start <= seg2_end)

    # Use 64-bit offsets to avoid overflow when H * L_KV * D exceeds int32
    k_head_offset = ((batch_idx * H + head_idx) * L_KV).to(tl.int64) * D_HEAD
    attn_base = batch_idx * L_KV

    m_i = -float('inf')
    l_i = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    scale = tl.full((1,), sm_scale, tl.float32)

    # Span blocks
    for block_idx in range(SPAN_MAX_BLOCKS):
        block_start = seg1_start + block_idx * BLOCK_K
        k_pos = block_start + tl.arange(0, BLOCK_K)
        in_range = seg1_valid & (k_pos >= seg1_start) & (k_pos <= seg1_end)
        k_pos_safe = tl.where(in_range, k_pos, 0)

        k_offsets = k_head_offset + k_pos_safe[:, None].to(tl.int64) * D_HEAD + d_range[None, :]
        v_offsets = k_head_offset + k_pos_safe[:, None].to(tl.int64) * D_HEAD + d_range[None, :]
        k_block = tl.load(K_ptr + k_offsets, mask=in_range[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        v_block = tl.load(V_ptr + v_offsets, mask=in_range[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        logits = tl.sum(k_block * q[None, :], axis=1) * scale

        if HAS_ATTN_MASK:
            attn_mask_vals = tl.load(attn_mask_ptr + attn_base + k_pos_safe, mask=in_range, other=0).to(tl.int1)
            in_range = in_range & attn_mask_vals
        logits = tl.where(in_range, logits, float('-inf'))

        if tl.sum(in_range, axis=0) > 0:
            block_max = tl.max(logits, axis=0)
            m_new = tl.maximum(m_i, block_max)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(logits - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            acc = acc * alpha + tl.sum(p[:, None] * v_block, axis=0)
            m_i = m_new

    # Sliding-window blocks
    if SW_MAX_BLOCKS > 0:
        for block_idx in range(SW_MAX_BLOCKS):
            block_start = seg2_start + block_idx * BLOCK_K
            k_pos = block_start + tl.arange(0, BLOCK_K)
            in_range = seg2_valid & (k_pos >= seg2_start) & (k_pos <= seg2_end)
            k_pos_safe = tl.where(in_range, k_pos, 0)

            k_offsets = k_head_offset + k_pos_safe[:, None].to(tl.int64) * D_HEAD + d_range[None, :]
            v_offsets = k_head_offset + k_pos_safe[:, None].to(tl.int64) * D_HEAD + d_range[None, :]
            k_block = tl.load(K_ptr + k_offsets, mask=in_range[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            v_block = tl.load(V_ptr + v_offsets, mask=in_range[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

            logits = tl.sum(k_block * q[None, :], axis=1) * scale

            if HAS_ATTN_MASK:
                attn_mask_vals = tl.load(attn_mask_ptr + attn_base + k_pos_safe, mask=in_range, other=0).to(tl.int1)
                in_range = in_range & attn_mask_vals
            logits = tl.where(in_range, logits, float('-inf'))

            if tl.sum(in_range, axis=0) > 0:
                block_max = tl.max(logits, axis=0)
                m_new = tl.maximum(m_i, block_max)
                alpha = tl.exp(m_i - m_new)
                p = tl.exp(logits - m_new)
                l_i = l_i * alpha + tl.sum(p, axis=0)
                acc = acc * alpha + tl.sum(p[:, None] * v_block, axis=0)
                m_i = m_new

    acc = tl.where(l_i > 0, acc / l_i, 0.0)
    # Use 64-bit indexing for the output to avoid overflow when L_Q * H * K_VAL * D is large
    out_index = ((batch_idx * H + head_idx) * L_Q + q_idx) * K_VAL + span_index
    out_base = Out_ptr + out_index.to(tl.int64) * D_HEAD
    tl.store(out_base + d_range, acc, mask=d_mask)



@triton.jit
def fused_span_backward_kernel(
    Q_ptr, K_ptr, V_ptr, dOut_ptr,
    qstart_ptr, qend_ptr, cache_position_ptr,
    attn_mask_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    B, H, L_Q, L_KV,
    window_size,
    sm_scale,
    K_VAL: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D_HEAD: tl.constexpr,
    SPAN_MAX_BLOCKS: tl.constexpr,
    SW_MAX_BLOCKS: tl.constexpr,
    HAS_ATTN_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    span_index = pid % K_VAL
    q_idx = (pid // K_VAL) % L_Q
    head_idx = (pid // (K_VAL * L_Q)) % H
    batch_idx = pid // (K_VAL * L_Q * H)

    d_range = tl.arange(0, BLOCK_D)
    d_mask = d_range < D_HEAD

    q_base = Q_ptr + ((batch_idx * H + head_idx) * L_Q + q_idx) * D_HEAD
    q = tl.load(q_base + d_range, mask=d_mask, other=0.0).to(tl.float32)

    cache_pos = tl.load(cache_position_ptr + q_idx, mask=True, other=0).to(tl.int32)
    cache_pos = tl.minimum(cache_pos, L_KV - 1)

    span_offset = ((batch_idx * H + head_idx) * L_Q + q_idx) * K_VAL + span_index
    span_start = tl.load(qstart_ptr + span_offset, mask=True, other=-1).to(tl.int32)
    span_end = tl.load(qend_ptr + span_offset, mask=True, other=-1).to(tl.int32)
    span_start = tl.maximum(span_start, 0)
    span_end = tl.minimum(span_end, L_KV - 1)
    span_valid = (span_end >= span_start) & (span_end >= 0)

    window = window_size
    sw_end = tl.minimum(cache_pos, L_KV - 1)
    sw_start = sw_end - (window - 1)
    sw_start = tl.maximum(sw_start, 0)
    sw_valid = (window > 0) & (sw_start <= sw_end) & (sw_end >= 0)

    seg1_start, seg1_end, seg1_valid = span_start, span_end, span_valid
    seg2_start, seg2_end, seg2_valid = sw_start, sw_end, sw_valid

    seg1_start = tl.maximum(seg1_start, 0)
    seg1_end = tl.minimum(seg1_end, L_KV - 1)
    seg1_valid = seg1_valid & (seg1_start <= seg1_end)

    seg2_start = tl.maximum(seg2_start, 0)
    seg2_end = tl.minimum(seg2_end, L_KV - 1)
    seg2_valid = seg2_valid & (seg2_start <= seg2_end)

    k_head_offset = ((batch_idx * H + head_idx) * L_KV).to(tl.int64) * D_HEAD
    attn_base = batch_idx * L_KV

    # First pass: compute m_i, l_i
    m_i = -float('inf')
    l_i = 0.0
    scale = tl.full((1,), sm_scale, tl.float32)

    for block_idx in range(SPAN_MAX_BLOCKS):
        block_start = seg1_start + block_idx * BLOCK_K
        k_pos = block_start + tl.arange(0, BLOCK_K)
        in_range = seg1_valid & (k_pos >= seg1_start) & (k_pos <= seg1_end)
        k_pos_safe = tl.where(in_range, k_pos, 0)
        k_offsets = k_head_offset + k_pos_safe[:, None].to(tl.int64) * D_HEAD + d_range[None, :]
        k_block = tl.load(K_ptr + k_offsets, mask=in_range[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        logits = tl.sum(k_block * q[None, :], axis=1) * scale
        if HAS_ATTN_MASK:
            attn_mask_vals = tl.load(attn_mask_ptr + attn_base + k_pos_safe, mask=in_range, other=0).to(tl.int1)
            in_range = in_range & attn_mask_vals
        logits = tl.where(in_range, logits, float('-inf'))
        if tl.sum(in_range, axis=0) > 0:
            block_max = tl.max(logits, axis=0)
            m_new = tl.maximum(m_i, block_max)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(logits - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            m_i = m_new

    if SW_MAX_BLOCKS > 0:
        for block_idx in range(SW_MAX_BLOCKS):
            block_start = seg2_start + block_idx * BLOCK_K
            k_pos = block_start + tl.arange(0, BLOCK_K)
            in_range = seg2_valid & (k_pos >= seg2_start) & (k_pos <= seg2_end)
            k_pos_safe = tl.where(in_range, k_pos, 0)
            k_offsets = k_head_offset + k_pos_safe[:, None].to(tl.int64) * D_HEAD + d_range[None, :]
            k_block = tl.load(K_ptr + k_offsets, mask=in_range[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            logits = tl.sum(k_block * q[None, :], axis=1) * scale
            if HAS_ATTN_MASK:
                attn_mask_vals = tl.load(attn_mask_ptr + attn_base + k_pos_safe, mask=in_range, other=0).to(tl.int1)
                in_range = in_range & attn_mask_vals
            logits = tl.where(in_range, logits, float('-inf'))
            if tl.sum(in_range, axis=0) > 0:
                block_max = tl.max(logits, axis=0)
                m_new = tl.maximum(m_i, block_max)
                alpha = tl.exp(m_i - m_new)
                p = tl.exp(logits - m_new)
                l_i = l_i * alpha + tl.sum(p, axis=0)
                m_i = m_new

    # Load grad_out for this (b,h,q,k)
    # 64-bit indexing to access dOut safely at large L_Q
    dO_index = ((batch_idx * H + head_idx) * L_Q + q_idx) * K_VAL + span_index
    dO_base = dOut_ptr + dO_index.to(tl.int64) * D_HEAD
    dO = tl.load(dO_base + d_range, mask=d_mask, other=0.0).to(tl.float32)

    grad_q = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # Second pass: accumulate dot_total over all keys
    dot_total = 0.0
    for block_idx in range(SPAN_MAX_BLOCKS):
        block_start = seg1_start + block_idx * BLOCK_K
        k_pos = block_start + tl.arange(0, BLOCK_K)
        in_range = seg1_valid & (k_pos >= seg1_start) & (k_pos <= seg1_end)
        k_pos_safe = tl.where(in_range, k_pos, 0)

        k_offsets = k_head_offset + k_pos_safe[:, None].to(tl.int64) * D_HEAD + d_range[None, :]
        v_offsets = k_head_offset + k_pos_safe[:, None].to(tl.int64) * D_HEAD + d_range[None, :]
        k_block = tl.load(K_ptr + k_offsets, mask=in_range[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        v_block = tl.load(V_ptr + v_offsets, mask=in_range[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        logits = tl.sum(k_block * q[None, :], axis=1) * scale
        if HAS_ATTN_MASK:
            attn_mask_vals = tl.load(attn_mask_ptr + attn_base + k_pos_safe, mask=in_range, other=0).to(tl.int1)
            in_range = in_range & attn_mask_vals
        logits = tl.where(in_range, logits, float('-inf'))
        if tl.sum(in_range, axis=0) > 0:
            weights = tl.exp(logits - m_i) / l_i
            weights = tl.where(in_range, weights, 0.0)
            grad_w = tl.sum(v_block * dO[None, :], axis=1)
            dot_total += tl.sum(grad_w * weights, axis=0)

    if SW_MAX_BLOCKS > 0:
        for block_idx in range(SW_MAX_BLOCKS):
            block_start = seg2_start + block_idx * BLOCK_K
            k_pos = block_start + tl.arange(0, BLOCK_K)
            in_range = seg2_valid & (k_pos >= seg2_start) & (k_pos <= seg2_end)
            k_pos_safe = tl.where(in_range, k_pos, 0)

            k_offsets = k_head_offset + k_pos_safe[:, None].to(tl.int64) * D_HEAD + d_range[None, :]
            v_offsets = k_head_offset + k_pos_safe[:, None].to(tl.int64) * D_HEAD + d_range[None, :]
            k_block = tl.load(K_ptr + k_offsets, mask=in_range[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            v_block = tl.load(V_ptr + v_offsets, mask=in_range[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

            logits = tl.sum(k_block * q[None, :], axis=1) * scale
            if HAS_ATTN_MASK:
                attn_mask_vals = tl.load(attn_mask_ptr + attn_base + k_pos_safe, mask=in_range, other=0).to(tl.int1)
                in_range = in_range & attn_mask_vals
            logits = tl.where(in_range, logits, float('-inf'))
            if tl.sum(in_range, axis=0) > 0:
                weights = tl.exp(logits - m_i) / l_i
                weights = tl.where(in_range, weights, 0.0)
                grad_w = tl.sum(v_block * dO[None, :], axis=1)
                dot_total += tl.sum(grad_w * weights, axis=0)

    # Third pass: compute gradients using dot_total
    for block_idx in range(SPAN_MAX_BLOCKS):
        block_start = seg1_start + block_idx * BLOCK_K
        k_pos = block_start + tl.arange(0, BLOCK_K)
        in_range = seg1_valid & (k_pos >= seg1_start) & (k_pos <= seg1_end)
        k_pos_safe = tl.where(in_range, k_pos, 0)

        k_offsets = k_head_offset + k_pos_safe[:, None] * D_HEAD + d_range[None, :]
        v_offsets = k_head_offset + k_pos_safe[:, None] * D_HEAD + d_range[None, :]
        k_block = tl.load(K_ptr + k_offsets, mask=in_range[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        v_block = tl.load(V_ptr + v_offsets, mask=in_range[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        logits = tl.sum(k_block * q[None, :], axis=1) * scale
        if HAS_ATTN_MASK:
            attn_mask_vals = tl.load(attn_mask_ptr + attn_base + k_pos_safe, mask=in_range, other=0).to(tl.int1)
            in_range = in_range & attn_mask_vals
        logits = tl.where(in_range, logits, float('-inf'))
        if tl.sum(in_range, axis=0) > 0:
            weights = tl.exp(logits - m_i) / l_i
            weights = tl.where(in_range, weights, 0.0)
            grad_w = tl.sum(v_block * dO[None, :], axis=1)
            grad_s = (grad_w - dot_total) * weights * sm_scale
            grad_s = tl.where(in_range, grad_s, 0.0)

            grad_q = grad_q + tl.sum(grad_s[:, None] * k_block, axis=0)

            dk = grad_s[:, None] * q[None, :]
            tl.atomic_add(dK_ptr + k_offsets, dk, mask=in_range[:, None] & d_mask[None, :])

            dv = weights[:, None] * dO[None, :]
            tl.atomic_add(dV_ptr + v_offsets, dv, mask=in_range[:, None] & d_mask[None, :])

    if SW_MAX_BLOCKS > 0:
        for block_idx in range(SW_MAX_BLOCKS):
            block_start = seg2_start + block_idx * BLOCK_K
            k_pos = block_start + tl.arange(0, BLOCK_K)
            in_range = seg2_valid & (k_pos >= seg2_start) & (k_pos <= seg2_end)
            k_pos_safe = tl.where(in_range, k_pos, 0)

            k_offsets = k_head_offset + k_pos_safe[:, None] * D_HEAD + d_range[None, :]
            v_offsets = k_head_offset + k_pos_safe[:, None] * D_HEAD + d_range[None, :]
            k_block = tl.load(K_ptr + k_offsets, mask=in_range[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            v_block = tl.load(V_ptr + v_offsets, mask=in_range[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

            logits = tl.sum(k_block * q[None, :], axis=1) * scale
            if HAS_ATTN_MASK:
                attn_mask_vals = tl.load(attn_mask_ptr + attn_base + k_pos_safe, mask=in_range, other=0).to(tl.int1)
                in_range = in_range & attn_mask_vals
            logits = tl.where(in_range, logits, float('-inf'))
            if tl.sum(in_range, axis=0) > 0:
                weights = tl.exp(logits - m_i) / l_i
                weights = tl.where(in_range, weights, 0.0)
                grad_w = tl.sum(v_block * dO[None, :], axis=1)
                grad_s = (grad_w - dot_total) * weights * sm_scale
                grad_s = tl.where(in_range, grad_s, 0.0)

                grad_q = grad_q + tl.sum(grad_s[:, None] * k_block, axis=0)

                dk = grad_s[:, None] * q[None, :]
                tl.atomic_add(dK_ptr + k_offsets, dk, mask=in_range[:, None] & d_mask[None, :])

                dv = weights[:, None] * dO[None, :]
                tl.atomic_add(dV_ptr + v_offsets, dv, mask=in_range[:, None] & d_mask[None, :])

    dq_base = dQ_ptr + ((batch_idx * H + head_idx) * L_Q + q_idx) * D_HEAD
    tl.atomic_add(dq_base + d_range, grad_q, mask=d_mask)



def _next_power_of_two(x: int) -> int:
    return 1 << (int(x) - 1).bit_length()


def _assert_no_span_sw_overlap(
    qend,
    cache_position,
    sw_index,
    L_KV,
    *,
    window_len: int | None = None,
    search_power: float | None = None,
    inv_search_power_int: int | None = 2,
):
    """
    Ensure span endpoints lie strictly before the sliding window start.
    Spans may touch the window boundary (qend == sw_start - 1) but not overlap it.
    """
    if sw_index <= 0:
        return
    if window_len is None:
        window = window_len_from_sw_index(
            int(sw_index),
            search_power=search_power,
            inv_search_power_int=inv_search_power_int,
        )
    else:
        window = int(window_len)
    if window <= 0:
        return

    # Broadcast sliding window start per query position
    cache_pos_clamped = cache_position.clamp(max=L_KV - 1)
    sw_end = cache_pos_clamped.view(1, 1, -1, 1)
    sw_start = (sw_end - (window - 1)).clamp(min=0)

    # Only enforce for valid spans (qend >= 0)
    # Skip assertion if disabled (e.g., during CUDA graph capture)
    if not _DISABLE_ASSERTS:
        overlap = (qend >= 0) & (qend >= sw_start)
        if torch.any(overlap):
            raise ValueError(
                "Span endpoints must satisfy qend < sliding window start; search outputs overlap the window."
            )


def fused_span_triton(
    Q2,
    K,
    V,
    qstart,
    qend,
    cache_position,
    attention_mask=None,
    sw_index=0,
    block_k: int = 64,
    span_len_factor: float = 2.0,
    *,
    span_power: float = 0.5,
    window_len: int | None = None,
    search_power: float | None = None,
    inv_search_power_int: int | None = 2,
):
    assert Q2.is_cuda, 'CUDA required for fused Triton path'
    B, H, L_Q, D = Q2.shape
    _, _, L_KV, _ = K.shape
    num_spans = qstart.shape[-1]

    span_len_factor = float(span_len_factor)
    if not math.isfinite(span_len_factor) or span_len_factor <= 0.0:
        raise ValueError(f"span_len_factor must be finite and > 0 (got {span_len_factor})")

    span_power_f = float(span_power)
    if not math.isfinite(span_power_f) or not (0.0 < span_power_f < 1.0):
        raise ValueError(f"span_power must be finite and in (0, 1) (got {span_power})")

    Q2c = Q2.contiguous()
    Kc = K.contiguous()
    Vc = V.contiguous()
    qstartc = qstart.contiguous()
    qendc = qend.contiguous()
    cachec = cache_position.to(torch.int32).contiguous()

    if attention_mask is not None:
        attn_mask = attention_mask[:, :L_KV].contiguous().to(torch.int8)
        has_mask = True
    else:
        attn_mask = torch.empty((1,), device=Q2.device, dtype=torch.int8)
        has_mask = False

    _assert_no_span_sw_overlap(
        qendc,
        cachec.view(1, 1, -1, 1),
        sw_index,
        L_KV,
        window_len=window_len,
        search_power=search_power,
        inv_search_power_int=inv_search_power_int,
    )

    out = torch.empty((B, H, L_Q, num_spans, D), device=Q2.device, dtype=Q2.dtype)

    if window_len is None:
        window = window_len_from_sw_index(
            int(sw_index),
            search_power=search_power,
            inv_search_power_int=inv_search_power_int,
        )
    else:
        window = int(window_len)
    max_span_len = int(span_len_factor * math.ceil(float(L_KV) ** span_power_f) + 2)
    span_max_blocks = triton.cdiv(max_span_len, block_k)
    span_max_blocks = max(1, span_max_blocks)
    sw_max_len = window if window > 0 else 0
    sw_max_blocks = triton.cdiv(sw_max_len, block_k) if sw_max_len > 0 else 0
    block_d = min(256, _next_power_of_two(D))

    grid = (B * H * L_Q * num_spans,)
    fused_span_forward_kernel[grid](
        Q2c, Kc, Vc,
        qstartc, qendc, cachec,
        attn_mask,
        out,
        B, H, L_Q, L_KV,
        window,
        1.0 / math.sqrt(D),
        K_VAL=num_spans,
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        D_HEAD=D,
        SPAN_MAX_BLOCKS=span_max_blocks,
        SW_MAX_BLOCKS=sw_max_blocks,
        HAS_ATTN_MASK=has_mask,
    )
    return out


class FusedSpanTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q2,
        K,
        V,
        qstart,
        qend,
        cache_position,
        attention_mask=None,
        sw_index: int = 0,
        block_k: int = 64,
        span_len_factor: float = 2.0,
        span_power: float = 0.5,
        search_power: float | None = None,
        inv_search_power_int: int | None = 2,
    ):
        out = fused_span_triton(
            Q2,
            K,
            V,
            qstart,
            qend,
            cache_position,
            attention_mask,
            sw_index,
            block_k,
            span_len_factor=span_len_factor,
            span_power=span_power,
            search_power=search_power,
            inv_search_power_int=inv_search_power_int,
        )
        saved_mask = attention_mask if attention_mask is not None else torch.tensor([], device=Q2.device)
        ctx.save_for_backward(Q2, K, V, qstart, qend, cache_position.to(torch.int32), saved_mask)
        ctx.sw_index = sw_index
        ctx.block_k = block_k
        ctx.span_len_factor = float(span_len_factor)
        ctx.span_power = float(span_power)
        ctx.search_power = search_power
        ctx.inv_search_power_int = inv_search_power_int
        return out

    @staticmethod
    def backward(ctx, grad_out):
        Q2, K, V, qstart, qend, cache_position, attention_mask_saved = ctx.saved_tensors
        attention_mask = None if attention_mask_saved.numel() == 0 else attention_mask_saved
        sw_index = ctx.sw_index
        block_k = ctx.block_k
        span_len_factor = float(getattr(ctx, "span_len_factor", 2.0))
        span_power = float(getattr(ctx, "span_power", 0.5))
        search_power = getattr(ctx, "search_power", None)
        inv_search_power_int = getattr(ctx, "inv_search_power_int", 2)

        B, H, L_Q, D = Q2.shape
        L_KV = K.shape[2]
        num_spans = qstart.shape[-1]

        window = window_len_from_sw_index(
            int(sw_index), search_power=search_power, inv_search_power_int=inv_search_power_int
        )
        max_span_len = int(span_len_factor * math.ceil(float(L_KV) ** float(span_power)) + 2)
        span_max_blocks = triton.cdiv(max_span_len, block_k)
        span_max_blocks = max(1, span_max_blocks)
        sw_max_len = window if window > 0 else 0
        sw_max_blocks = triton.cdiv(sw_max_len, block_k) if sw_max_len > 0 else 0
        block_d = min(256, _next_power_of_two(D))

        qstartc = qstart.contiguous()
        qendc = qend.contiguous()
        cachec = cache_position.to(torch.int32).contiguous()
        Q2c = Q2.contiguous()
        Kc = K.contiguous()
        Vc = V.contiguous()
        grad_out_c = grad_out.contiguous()

        if attention_mask is not None:
            attn_mask = attention_mask[:, :L_KV].contiguous().to(torch.int8)
            has_mask = True
        else:
            attn_mask = torch.empty((1,), device=Q2.device, dtype=torch.int8)
            has_mask = False

        dQ = torch.zeros_like(Q2c, dtype=torch.float32)
        dK = torch.zeros_like(Kc, dtype=torch.float32)
        dV = torch.zeros_like(Vc, dtype=torch.float32)

        grid = (B * H * L_Q * num_spans,)
        fused_span_backward_kernel[grid](
            Q2c, Kc, Vc, grad_out_c,
            qstartc, qendc, cachec,
            attn_mask,
            dQ, dK, dV,
            B, H, L_Q, L_KV,
            window,
            1.0 / math.sqrt(D),
            K_VAL=num_spans,
            BLOCK_K=block_k,
            BLOCK_D=block_d,
            D_HEAD=D,
            SPAN_MAX_BLOCKS=span_max_blocks,
            SW_MAX_BLOCKS=sw_max_blocks,
            HAS_ATTN_MASK=has_mask,
        )

        # Cast back to input dtype
        return (
            dQ.to(Q2.dtype),
            dK.to(K.dtype),
            dV.to(V.dtype),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def fused_span_attention(
    Q2,
    K,
    V,
    qstart,
    qend,
    cache_position,
    attention_mask=None,
    sw_index=0,
    block_k=64,
    span_len_factor: float = 2.0,
    *,
    span_power: float = 0.5,
    search_power: float | None = None,
    inv_search_power_int: int | None = 2,
):
    return FusedSpanTriton.apply(
        Q2,
        K,
        V,
        qstart,
        qend,
        cache_position,
        attention_mask,
        sw_index,
        block_k,
        span_len_factor,
        float(span_power),
        search_power,
        inv_search_power_int,
    )


def eager_span_sliding_attention_multi(
    Q2,
    K,
    V,
    cache_position,
    qstart,
    qend,
    attention_mask=None,
    sw_index=0,
    *,
    window_len: int | None = None,
    search_power: float | None = None,
    inv_search_power_int: int | None = 2,
):
    B, H, L_Q, D = Q2.shape
    L_KV = K.shape[2]
    num_spans = qstart.shape[-1]

    _assert_no_span_sw_overlap(
        qend,
        cache_position.view(1, 1, -1, 1),
        sw_index,
        L_KV,
        window_len=window_len,
        search_power=search_power,
        inv_search_power_int=inv_search_power_int,
    )

    q_idx = torch.arange(L_Q, device=Q2.device).view(1,1,L_Q,1,1)
    k_idx = torch.arange(L_KV, device=Q2.device).view(1,1,1,1,L_KV)
    c_idx = cache_position[q_idx]

    span_start = qstart.unsqueeze(-1)
    span_end = qend.unsqueeze(-1)
    span_mask = (span_end >= 0) & (k_idx >= span_start) & (k_idx <= span_end)

    if window_len is None:
        window = window_len_from_sw_index(
            int(sw_index),
            search_power=search_power,
            inv_search_power_int=inv_search_power_int,
        )
    else:
        window = int(window_len)
    diff = c_idx - k_idx + 1
    sw_mask = (diff > 0) & (diff <= window)

    mask = span_mask | sw_mask
    if attention_mask is not None:
        mask = mask & attention_mask[:, None, None, None, :].to(torch.bool)

    scores = torch.matmul(Q2, K.transpose(-2, -1)) / math.sqrt(D)  # [B,H,L,KV]
    scores = scores.unsqueeze(3).masked_fill(~mask, float('-inf'))  # [B,H,L,K,KV]
    weights = torch.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)

    out = torch.einsum('bhlkv,bhvd->bhlkd', weights, V)
    return out


def decode_span_attention_staged(
    Q1,
    Q2,
    K,
    V,
    cache_position,
    attention_mask=None,
    sw_index=0,
    topk=3,
    backward_factor: float = 2.0,
    forward_factor: float = 0.0,
    *,
    span_power: float = 0.5,
    search_power: float | None = None,
    inv_search_power_int: int | None = 2,
):
    """
    Decoding-only fast path (L_Q = 1) that stages stripe keys into a contiguous
    buffer, runs a single matvec search to pick top-k spans, then reuses the
    existing fused span attention for the span outputs.

    This avoids the separate span_search_triton kernel and the extra pass over K
    for search. Backward is not supported (inference-only).
    """
    B, H, L_Q, D = Q1.shape
    assert L_Q == 1, "decode_span_attention_staged is only for decoding (L_Q=1)"
    _, _, L_KV, _ = K.shape
    device = Q1.device

    span_power_f = float(span_power)
    if not math.isfinite(span_power_f) or not (0.0 < span_power_f < 1.0):
        raise ValueError(f"span_power must be finite and in (0, 1) (got {span_power})")

    window_len = window_len_from_sw_index(
        int(sw_index), search_power=search_power, inv_search_power_int=inv_search_power_int
    )
    decode_pos = int(cache_position[-1].item())
    max_stripe_idx = max_stripe_index_for_token_pos(
        decode_pos + 1,
        search_power=search_power,
        inv_search_power_int=inv_search_power_int,
    )

    # If the sliding window already covers the whole sequence, the span selection is irrelevant;
    # fall back to full SDPA for correctness and skip staging/gating overhead.
    if window_len >= L_KV or sw_index >= max_stripe_idx:
        if attention_mask is not None:
            # Keep (B, H, L, D) shape so the mask matches sdpa expectations.
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = torch.where(
                attn_mask,
                torch.tensor(0.0, device=device, dtype=Q2.dtype),
                torch.tensor(-1e9, device=device, dtype=Q2.dtype),
            )
            out = torch.nn.functional.scaled_dot_product_attention(
                Q2, K, V, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
            )
            return out

        # No mask: flatten heads to maximize flash SDPA coverage.
        q = Q2.reshape(B * H, L_Q, D)
        k = K.reshape(B * H, L_KV, D)
        v = V.reshape(B * H, L_KV, D)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        return out.reshape(B, H, L_Q, D)

    power_params = derive_stripe_power_params(
        search_power=search_power, inv_search_power_int=inv_search_power_int
    )

    stripe_idx = torch.arange(int(sw_index) + 1, int(max_stripe_idx) + 1, device=device, dtype=torch.int64)
    if stripe_idx.numel() == 0:
        if attention_mask is not None:
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = torch.where(
                attn_mask,
                torch.tensor(0.0, device=device, dtype=Q2.dtype),
                torch.tensor(-1e9, device=device, dtype=Q2.dtype),
            )
            return torch.nn.functional.scaled_dot_product_attention(
                Q2, K, V, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
            )

        q = Q2.reshape(B * H, L_Q, D)
        k = K.reshape(B * H, L_KV, D)
        v = V.reshape(B * H, L_KV, D)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        return out.reshape(B, H, L_Q, D)

    if power_params.triton_inv_n != 0:
        stripe_floor_power = stripe_idx.to(torch.int64) ** int(power_params.triton_inv_n)
    else:
        stripe_floor_power = torch.floor(
            torch.exp(torch.log(stripe_idx.to(torch.float32)) * float(power_params.inv_p))
        ).to(torch.int64)

    stripe_loc = (decode_pos + 1) - stripe_floor_power
    stripe_loc = stripe_loc[(stripe_loc >= 0) & (stripe_loc < L_KV)].to(torch.int64)

    # Filter stripes that would overlap with the sliding window at the decode position
    # sw_start = max(0, cache_position[-1] - (window_len - 1))
    # Valid stripes must have stripe_loc < sw_start
    sw_start = max(0, decode_pos - (window_len - 1))
    stripe_loc = stripe_loc[stripe_loc < sw_start]

    # Initialize topk_vals and topk_idx with proper dimensions
    # - topk_vals initialized to -inf (these spans will be ignored in softmax)
    # - topk_idx initialized to -1 (invalid index, will produce qend=-1)
    topk_vals = torch.full((B, H, topk), float('-inf'), device=device, dtype=torch.float32)
    topk_idx = torch.full((B, H, topk), -1, device=device, dtype=torch.int64)

    num_valid_stripes = len(stripe_loc)
    if num_valid_stripes > 0:
        # Gather candidate keys [B, H, S, D] (search is treated as non-differentiable)
        K_stripe = K.index_select(2, stripe_loc)
        # Compute search logits in fp32 to match the Triton search kernel's accumulation
        logits = torch.einsum(
            'bhid,bhsd->bhis',
            Q1.detach().float(),
            K_stripe.detach().float()
        ).squeeze(2)  # [B,H,S]

        if attention_mask is not None:
            stripe_valid = attention_mask[:, stripe_loc].to(torch.bool)  # [B,S]
            logits = logits.masked_fill(~stripe_valid[:, None, :], float("-inf"))

        # Top-k over stripes (use min to handle case where fewer stripes than topk)
        actual_k = min(topk, num_valid_stripes)
        actual_topk_vals, actual_topk_idx = torch.topk(logits, k=actual_k, dim=-1)

        # Fill the first actual_k slots with the results
        topk_vals[:, :, :actual_k] = actual_topk_vals
        topk_idx[:, :, :actual_k] = actual_topk_idx

    qanchor = torch.full_like(topk_idx, -1, dtype=torch.int64)
    if len(stripe_loc) > 0:
        safe_idx = topk_idx.clamp(min=0)
        gathered = stripe_loc[safe_idx]
        qanchor = torch.where(torch.isfinite(topk_vals), gathered, qanchor)

    # Span length formula matches search kernel for decoding
    if attention_mask is not None:
        start_pos = attention_mask.to(torch.int32).argmax(dim=1).to(torch.int64)
    else:
        start_pos = torch.zeros((B,), device=device, dtype=torch.int64)

    token_pos = torch.clamp(torch.tensor(decode_pos + 1, device=device, dtype=torch.int64) - start_pos, min=0)
    span_len = torch.ceil(float(backward_factor) * (token_pos.to(torch.float32) ** span_power_f)).to(torch.int64)
    qstart = qanchor - span_len[:, None, None]
    qstart = torch.maximum(qstart, start_pos[:, None, None])
    qstart = torch.clamp(qstart, min=0)
    qstart = torch.where(qanchor < 0, torch.tensor(-1, device=device, dtype=torch.int64), qstart)

    # Reshape to [B, H, 1, topk] to feed fused_span_attention
    qstart = qstart.unsqueeze(2)
    qanchor = qanchor.unsqueeze(2)

    # Span outputs [B, H, 1, topk, D]
    # For decode, pass only the last cache position (shape [1]) to avoid broadcast mismatch
    # in _assert_no_span_sw_overlap which computes sw_start per query position
    decode_cache_pos = cache_position[-1:]
    qend = compute_qend_from_qanchor(
        qanchor,
        cache_position=decode_cache_pos,
        key_length=L_KV,
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
        decode_cache_pos,
        attention_mask=attention_mask,
        sw_index=sw_index,
        span_len_factor=span_len_factor,
        span_power=span_power,
        search_power=search_power,
        inv_search_power_int=inv_search_power_int,
    )

    # Reuse topk_vals directly for gating (no need to recompute K[qend] @ Q1)
    values = topk_vals.unsqueeze(2)  # [B, H, 1, topk]
    # Invalid spans already have -inf in topk_vals, but ensure consistency
    values = torch.where(qanchor < 0, float("-inf"), values)

    span_scores = torch.nan_to_num(torch.softmax(values.float(), dim=-1), 1 / topk)  # [B,H,1,topk]
    span_scores = span_scores.to(O_span.dtype)
    O = (span_scores.unsqueeze(-1) * O_span).sum(dim=3)  # [B, H, L_Q, D]

    return O
