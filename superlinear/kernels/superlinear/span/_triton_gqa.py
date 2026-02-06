import math
import os

import torch
import triton
import triton.language as tl

from superlinear.kernels.common.power import derive_stripe_power_params, max_stripe_index_for_token_pos, window_len_from_sw_index
from superlinear.kernels.common.adjustment import compute_qend_from_qanchor
from ._triton_impl import (
    _assert_no_span_sw_overlap,
    _next_power_of_two,
    decode_span_attention_staged,
    fused_span_attention,
)


@triton.jit
def fused_span_forward_kernel_gqa(
    Q_ptr, K_ptr, V_ptr,
    qstart_ptr, qend_ptr, cache_position_ptr,
    attn_mask_ptr,
    Out_ptr,
    B, H_Q, H_KV, L_Q, L_KV,
    kv_repeat,
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
    q_head_idx = (pid // (K_VAL * L_Q)) % H_Q
    batch_idx = pid // (K_VAL * L_Q * H_Q)

    kv_head_idx = q_head_idx // kv_repeat
    kv_head_idx = tl.minimum(kv_head_idx, H_KV - 1)

    d_range = tl.arange(0, BLOCK_D)
    d_mask = d_range < D_HEAD

    q_base = Q_ptr + ((batch_idx * H_Q + q_head_idx) * L_Q + q_idx) * D_HEAD
    q = tl.load(q_base + d_range, mask=d_mask, other=0.0).to(tl.float32)

    cache_pos = tl.load(cache_position_ptr + q_idx, mask=True, other=0).to(tl.int32)
    cache_pos = tl.minimum(cache_pos, L_KV - 1)

    span_offset = ((batch_idx * H_Q + q_head_idx) * L_Q + q_idx) * K_VAL + span_index
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

    k_head_offset = ((batch_idx * H_KV + kv_head_idx) * L_KV).to(tl.int64) * D_HEAD
    attn_base = batch_idx * L_KV

    m_i = -float('inf')
    l_i = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    scale = tl.full((1,), sm_scale, tl.float32)

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
    out_index = ((batch_idx * H_Q + q_head_idx) * L_Q + q_idx) * K_VAL + span_index
    out_base = Out_ptr + out_index.to(tl.int64) * D_HEAD
    tl.store(out_base + d_range, acc, mask=d_mask)


@triton.jit
def fused_span_backward_kernel_gqa(
    Q_ptr, K_ptr, V_ptr, dOut_ptr,
    qstart_ptr, qend_ptr, cache_position_ptr,
    attn_mask_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    B, H_Q, H_KV, L_Q, L_KV,
    kv_repeat,
    window_size,
    sm_scale,
    K_VAL: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D_HEAD: tl.constexpr,
    SPAN_MAX_BLOCKS: tl.constexpr,
    SW_MAX_BLOCKS: tl.constexpr,
    HAS_ATTN_MASK: tl.constexpr,
    WRITE_DKV_PER_QHEAD: tl.constexpr,
):
    pid = tl.program_id(0)
    span_index = pid % K_VAL
    q_idx = (pid // K_VAL) % L_Q
    q_head_idx = (pid // (K_VAL * L_Q)) % H_Q
    batch_idx = pid // (K_VAL * L_Q * H_Q)

    kv_head_idx = q_head_idx // kv_repeat
    kv_head_idx = tl.minimum(kv_head_idx, H_KV - 1)

    d_range = tl.arange(0, BLOCK_D)
    d_mask = d_range < D_HEAD

    q_base = Q_ptr + ((batch_idx * H_Q + q_head_idx) * L_Q + q_idx) * D_HEAD
    q = tl.load(q_base + d_range, mask=d_mask, other=0.0).to(tl.float32)

    cache_pos = tl.load(cache_position_ptr + q_idx, mask=True, other=0).to(tl.int32)
    cache_pos = tl.minimum(cache_pos, L_KV - 1)

    span_offset = ((batch_idx * H_Q + q_head_idx) * L_Q + q_idx) * K_VAL + span_index
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

    k_head_offset = ((batch_idx * H_KV + kv_head_idx) * L_KV).to(tl.int64) * D_HEAD
    if WRITE_DKV_PER_QHEAD:
        dkv_head_offset = ((batch_idx * H_Q + q_head_idx) * L_KV).to(tl.int64) * D_HEAD
    else:
        dkv_head_offset = k_head_offset
    attn_base = batch_idx * L_KV

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

    dO_index = ((batch_idx * H_Q + q_head_idx) * L_Q + q_idx) * K_VAL + span_index
    dO_base = dOut_ptr + dO_index.to(tl.int64) * D_HEAD
    dO = tl.load(dO_base + d_range, mask=d_mask, other=0.0).to(tl.float32)

    grad_q = tl.zeros((BLOCK_D,), dtype=tl.float32)

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
            grad_s = (grad_w - dot_total) * weights * sm_scale
            grad_s = tl.where(in_range, grad_s, 0.0)

            grad_q = grad_q + tl.sum(grad_s[:, None] * k_block, axis=0)

            dk = grad_s[:, None] * q[None, :]
            dkv_offsets = dkv_head_offset + k_pos_safe[:, None].to(tl.int64) * D_HEAD + d_range[None, :]
            tl.atomic_add(dK_ptr + dkv_offsets, dk, mask=in_range[:, None] & d_mask[None, :])

            dv = weights[:, None] * dO[None, :]
            tl.atomic_add(dV_ptr + dkv_offsets, dv, mask=in_range[:, None] & d_mask[None, :])

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
                grad_s = (grad_w - dot_total) * weights * sm_scale
                grad_s = tl.where(in_range, grad_s, 0.0)

                grad_q = grad_q + tl.sum(grad_s[:, None] * k_block, axis=0)

                dk = grad_s[:, None] * q[None, :]
                dkv_offsets = dkv_head_offset + k_pos_safe[:, None].to(tl.int64) * D_HEAD + d_range[None, :]
                tl.atomic_add(dK_ptr + dkv_offsets, dk, mask=in_range[:, None] & d_mask[None, :])

                dv = weights[:, None] * dO[None, :]
                tl.atomic_add(dV_ptr + dkv_offsets, dv, mask=in_range[:, None] & d_mask[None, :])

    dq_base = dQ_ptr + ((batch_idx * H_Q + q_head_idx) * L_Q + q_idx) * D_HEAD
    tl.atomic_add(dq_base + d_range, grad_q, mask=d_mask)


@triton.jit
def fused_span_backward_kernel_gqa_fused_spans(
    Q_ptr, K_ptr, V_ptr, dOut_ptr,
    qstart_ptr, qend_ptr, cache_position_ptr,
    attn_mask_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    B, H_Q, H_KV, L_Q, L_KV,
    kv_repeat,
    window_size,
    sm_scale,
    K_VAL: tl.constexpr,  # Power of 2 for tl.arange
    K_ACTUAL: tl.constexpr,  # Actual number of spans
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D_HEAD: tl.constexpr,
    SPAN_MAX_BLOCKS: tl.constexpr,
    SW_MAX_BLOCKS: tl.constexpr,
    HAS_ATTN_MASK: tl.constexpr,
    WRITE_DKV_PER_QHEAD: tl.constexpr,
):
    pid = tl.program_id(0)
    q_idx = pid % L_Q
    q_head_idx = (pid // L_Q) % H_Q
    batch_idx = pid // (L_Q * H_Q)

    kv_head_idx = q_head_idx // kv_repeat
    kv_head_idx = tl.minimum(kv_head_idx, H_KV - 1)

    span_range = tl.arange(0, K_VAL)
    span_mask = span_range < K_ACTUAL  # Mask for valid spans
    k_range = tl.arange(0, BLOCK_K)
    d_range = tl.arange(0, BLOCK_D)
    d_mask = d_range < D_HEAD

    q_base = Q_ptr + ((batch_idx * H_Q + q_head_idx) * L_Q + q_idx) * D_HEAD
    q = tl.load(q_base + d_range, mask=d_mask, other=0.0).to(tl.float32)

    cache_pos = tl.load(cache_position_ptr + q_idx, mask=True, other=0).to(tl.int32)
    cache_pos = tl.minimum(cache_pos, L_KV - 1)

    # Use K_ACTUAL for actual data layout
    span_offset_base = ((batch_idx * H_Q + q_head_idx) * L_Q + q_idx) * K_ACTUAL
    span_offsets = span_offset_base + span_range
    span_start = tl.load(qstart_ptr + span_offsets, mask=span_mask, other=-1).to(tl.int32)
    span_end = tl.load(qend_ptr + span_offsets, mask=span_mask, other=-1).to(tl.int32)
    span_start = tl.maximum(span_start, 0)
    span_end = tl.minimum(span_end, L_KV - 1)
    span_valid = (span_end >= span_start) & (span_end >= 0) & span_mask

    window = window_size
    sw_end = tl.minimum(cache_pos, L_KV - 1)
    sw_start = sw_end - (window - 1)
    sw_start = tl.maximum(sw_start, 0)
    sw_valid = (window > 0) & (sw_start <= sw_end) & (sw_end >= 0)

    k_head_offset = ((batch_idx * H_KV + kv_head_idx) * L_KV).to(tl.int64) * D_HEAD
    if WRITE_DKV_PER_QHEAD:
        dkv_head_offset = ((batch_idx * H_Q + q_head_idx) * L_KV).to(tl.int64) * D_HEAD
    else:
        dkv_head_offset = k_head_offset
    attn_base = batch_idx * L_KV

    # Use K_ACTUAL for actual data layout
    dO_index_base = ((batch_idx * H_Q + q_head_idx) * L_Q + q_idx) * K_ACTUAL
    dO_offsets = (dO_index_base + span_range)[:, None].to(tl.int64) * D_HEAD + d_range[None, :]
    dO = tl.load(dOut_ptr + dO_offsets, mask=span_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

    m_i = tl.full((K_VAL,), -float('inf'), tl.float32)
    l_i = tl.zeros((K_VAL,), tl.float32)
    dot_num = tl.zeros((K_VAL,), tl.float32)
    scale = tl.full((1,), sm_scale, tl.float32)

    # Pass 1: compute m_i, l_i, dot_num (online) for each span.
    for block_idx in range(SPAN_MAX_BLOCKS):
        block_start = span_start + block_idx * BLOCK_K
        k_pos = block_start[:, None] + k_range[None, :]
        in_range = span_valid[:, None] & (k_pos >= span_start[:, None]) & (k_pos <= span_end[:, None])
        k_pos_safe = tl.where(in_range, k_pos, 0)

        kv_offsets = k_head_offset + k_pos_safe[:, :, None].to(tl.int64) * D_HEAD + d_range[None, None, :]
        k_block = tl.load(K_ptr + kv_offsets, mask=in_range[:, :, None] & d_mask[None, None, :], other=0.0).to(tl.float32)
        v_block = tl.load(V_ptr + kv_offsets, mask=in_range[:, :, None] & d_mask[None, None, :], other=0.0).to(tl.float32)

        logits = tl.sum(k_block * q[None, None, :], axis=2) * scale

        if HAS_ATTN_MASK:
            attn_mask_vals = tl.load(attn_mask_ptr + attn_base + k_pos_safe, mask=in_range, other=0).to(tl.int1)
            in_range = in_range & attn_mask_vals
        logits = tl.where(in_range, logits, float('-inf'))

        grad_w = tl.sum(v_block * dO[:, None, :], axis=2)

        block_max = tl.max(logits, axis=1)
        m_new = tl.maximum(m_i, block_max)
        alpha = tl.exp(m_i - m_new)
        alpha = tl.where(m_new == -float('inf'), 0.0, alpha)

        logits_shift = logits - m_new[:, None]
        logits_shift = tl.where(m_new[:, None] == -float('inf'), float('-inf'), logits_shift)
        p = tl.exp(logits_shift)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        dot_num = dot_num * alpha + tl.sum(p * grad_w, axis=1)
        m_i = m_new

    if SW_MAX_BLOCKS > 0:
        for block_idx in range(SW_MAX_BLOCKS):
            block_start = sw_start + block_idx * BLOCK_K
            k_pos = block_start + k_range
            in_range = sw_valid & (k_pos >= sw_start) & (k_pos <= sw_end)
            k_pos_safe = tl.where(in_range, k_pos, 0)

            kv_offsets = k_head_offset + k_pos_safe[:, None].to(tl.int64) * D_HEAD + d_range[None, :]
            k_block = tl.load(K_ptr + kv_offsets, mask=in_range[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            v_block = tl.load(V_ptr + kv_offsets, mask=in_range[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

            logits = tl.sum(k_block * q[None, :], axis=1) * scale

            if HAS_ATTN_MASK:
                attn_mask_vals = tl.load(attn_mask_ptr + attn_base + k_pos_safe, mask=in_range, other=0).to(tl.int1)
                in_range = in_range & attn_mask_vals
            logits = tl.where(in_range, logits, float('-inf'))

            grad_w = tl.sum(v_block[None, :, :] * dO[:, None, :], axis=2)

            block_max = tl.max(logits, axis=0)
            m_new = tl.maximum(m_i, block_max)
            alpha = tl.exp(m_i - m_new)
            alpha = tl.where(m_new == -float('inf'), 0.0, alpha)

            logits_shift = logits[None, :] - m_new[:, None]
            logits_shift = tl.where(m_new[:, None] == -float('inf'), float('-inf'), logits_shift)
            p = tl.exp(logits_shift)

            l_i = l_i * alpha + tl.sum(p, axis=1)
            dot_num = dot_num * alpha + tl.sum(p * grad_w, axis=1)
            m_i = m_new

    inv_l = tl.where(l_i > 0, 1.0 / l_i, 0.0)
    dot_total = dot_num * inv_l

    grad_q = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # Pass 2: accumulate gradients for all spans.
    for block_idx in range(SPAN_MAX_BLOCKS):
        block_start = span_start + block_idx * BLOCK_K
        k_pos = block_start[:, None] + k_range[None, :]
        in_range = span_valid[:, None] & (k_pos >= span_start[:, None]) & (k_pos <= span_end[:, None])
        k_pos_safe = tl.where(in_range, k_pos, 0)

        kv_offsets = k_head_offset + k_pos_safe[:, :, None].to(tl.int64) * D_HEAD + d_range[None, None, :]
        k_block = tl.load(K_ptr + kv_offsets, mask=in_range[:, :, None] & d_mask[None, None, :], other=0.0).to(tl.float32)
        v_block = tl.load(V_ptr + kv_offsets, mask=in_range[:, :, None] & d_mask[None, None, :], other=0.0).to(tl.float32)

        logits = tl.sum(k_block * q[None, None, :], axis=2) * scale

        if HAS_ATTN_MASK:
            attn_mask_vals = tl.load(attn_mask_ptr + attn_base + k_pos_safe, mask=in_range, other=0).to(tl.int1)
            in_range = in_range & attn_mask_vals
        logits = tl.where(in_range, logits, float('-inf'))

        weights = tl.exp(logits - m_i[:, None]) * inv_l[:, None]
        weights = tl.where(inv_l[:, None] > 0, weights, 0.0)
        weights = tl.where(in_range, weights, 0.0)

        grad_w = tl.sum(v_block * dO[:, None, :], axis=2)
        grad_s = (grad_w - dot_total[:, None]) * weights * sm_scale
        grad_s = tl.where(in_range, grad_s, 0.0)

        grad_q_span = tl.sum(grad_s[:, :, None] * k_block, axis=1)
        grad_q += tl.sum(grad_q_span, axis=0)

        dk = grad_s[:, :, None] * q[None, None, :]
        dv = weights[:, :, None] * dO[:, None, :]

        dkv_offsets = dkv_head_offset + k_pos_safe[:, :, None].to(tl.int64) * D_HEAD + d_range[None, None, :]
        tl.atomic_add(dK_ptr + dkv_offsets, dk, mask=in_range[:, :, None] & d_mask[None, None, :])
        tl.atomic_add(dV_ptr + dkv_offsets, dv, mask=in_range[:, :, None] & d_mask[None, None, :])

    if SW_MAX_BLOCKS > 0:
        for block_idx in range(SW_MAX_BLOCKS):
            block_start = sw_start + block_idx * BLOCK_K
            k_pos = block_start + k_range
            in_range = sw_valid & (k_pos >= sw_start) & (k_pos <= sw_end)
            k_pos_safe = tl.where(in_range, k_pos, 0)

            kv_offsets = k_head_offset + k_pos_safe[:, None].to(tl.int64) * D_HEAD + d_range[None, :]
            k_block = tl.load(K_ptr + kv_offsets, mask=in_range[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            v_block = tl.load(V_ptr + kv_offsets, mask=in_range[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

            logits = tl.sum(k_block * q[None, :], axis=1) * scale

            if HAS_ATTN_MASK:
                attn_mask_vals = tl.load(attn_mask_ptr + attn_base + k_pos_safe, mask=in_range, other=0).to(tl.int1)
                in_range = in_range & attn_mask_vals
            logits = tl.where(in_range, logits, float('-inf'))

            weights = tl.exp(logits[None, :] - m_i[:, None]) * inv_l[:, None]
            weights = tl.where(inv_l[:, None] > 0, weights, 0.0)
            weights = tl.where(in_range[None, :], weights, 0.0)

            grad_w = tl.sum(v_block[None, :, :] * dO[:, None, :], axis=2)
            grad_s = (grad_w - dot_total[:, None]) * weights * sm_scale
            grad_s = tl.where(in_range[None, :], grad_s, 0.0)

            grad_q_span = tl.sum(grad_s[:, :, None] * k_block[None, :, :], axis=1)
            grad_q += tl.sum(grad_q_span, axis=0)

            grad_s_sum = tl.sum(grad_s, axis=0)
            dk_total = grad_s_sum[:, None] * q[None, :]
            dv_total = tl.sum(weights[:, :, None] * dO[:, None, :], axis=0)

            dkv_offsets = dkv_head_offset + k_pos_safe[:, None].to(tl.int64) * D_HEAD + d_range[None, :]
            tl.atomic_add(dK_ptr + dkv_offsets, dk_total, mask=in_range[:, None] & d_mask[None, :])
            tl.atomic_add(dV_ptr + dkv_offsets, dv_total, mask=in_range[:, None] & d_mask[None, :])

    dq_base = dQ_ptr + ((batch_idx * H_Q + q_head_idx) * L_Q + q_idx) * D_HEAD
    tl.store(dq_base + d_range, grad_q, mask=d_mask)



def fused_span_triton_gqa(
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
    span_power: float = 0.5,
    search_power: float | None = None,
    inv_search_power_int: int | None = 2,
):
    assert Q2.is_cuda, "CUDA required for fused Triton path"
    B, H_q, L_Q, D = Q2.shape
    _, H_kv, L_KV, _ = K.shape
    num_spans = qstart.shape[-1]

    span_len_factor = float(span_len_factor)
    if not math.isfinite(span_len_factor) or span_len_factor <= 0.0:
        raise ValueError(f"span_len_factor must be finite and > 0 (got {span_len_factor})")

    span_power_f = float(span_power)
    if not math.isfinite(span_power_f) or not (0.0 < span_power_f < 1.0):
        raise ValueError(f"span_power must be finite and in (0, 1) (got {span_power})")

    kv_repeat = H_q // H_kv

    # Ensure all tensors are on the same device as Q2 for multi-GPU compatibility
    device = Q2.device
    K = K.to(device)
    V = V.to(device)
    qstart = qstart.to(device)
    qend = qend.to(device)
    cache_position = cache_position.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

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
        attn_mask = torch.empty((1,), device=device, dtype=torch.int8)
        has_mask = False

    window = window_len_from_sw_index(
        int(sw_index), search_power=search_power, inv_search_power_int=inv_search_power_int
    )
    _assert_no_span_sw_overlap(qendc, cachec.view(1, 1, -1, 1), sw_index, L_KV, window_len=window)

    out = torch.empty((B, H_q, L_Q, num_spans, D), device=device, dtype=Q2.dtype)

    max_span_len = int(span_len_factor * math.ceil(float(L_KV) ** span_power_f) + 2)
    span_max_blocks = triton.cdiv(max_span_len, block_k)
    span_max_blocks = max(1, span_max_blocks)
    sw_max_len = min(window, L_KV) if window > 0 else 0
    sw_max_blocks = triton.cdiv(sw_max_len, block_k) if sw_max_len > 0 else 0
    block_d = min(256, _next_power_of_two(D))

    grid = (B * H_q * L_Q * num_spans,)
    # Use torch.cuda.device context to ensure kernel launches on correct GPU
    with torch.cuda.device(device):
        fused_span_forward_kernel_gqa[grid](
            Q2c, Kc, Vc,
            qstartc, qendc, cachec,
            attn_mask,
            out,
            B, H_q, H_kv, L_Q, L_KV,
            kv_repeat,
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
        # Synchronize to ensure kernel completes before output is used elsewhere.
        # Skip during CUDA graph capture (synchronize is not allowed during capture).
        # NOTE: This sync was added in commit 9dcb65f to fix multi-GPU race conditions
        # with device_map='auto'. However, notebook 39.4 shows that the
        # `with torch.cuda.device(device):` context is sufficient for correctness,
        # and HuggingFace patterns don't use per-kernel sync. Consider removing
        # this sync entirely if multi-GPU inference remains stable without it.
        if not torch.cuda.is_current_stream_capturing():
            torch.cuda.current_stream().synchronize()
    return out


class FusedSpanGQATriton(torch.autograd.Function):
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
        B, H_q, L_Q, D = Q2.shape
        _, H_kv, _, _ = K.shape
        assert H_q % H_kv == 0, "Query heads must be divisible by KV heads"

        kv_repeat = H_q // H_kv
        out = fused_span_triton_gqa(
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
        ctx.kv_repeat = kv_repeat
        ctx.span_len_factor = float(span_len_factor)
        ctx.span_power = float(span_power)
        ctx.window_len = window_len_from_sw_index(
            int(sw_index), search_power=search_power, inv_search_power_int=inv_search_power_int
        )
        return out

    @staticmethod
    def backward(ctx, grad_out):
        Q2, K, V, qstart, qend, cache_position, attention_mask_saved = ctx.saved_tensors
        attention_mask = None if attention_mask_saved.numel() == 0 else attention_mask_saved
        sw_index = ctx.sw_index
        block_k = ctx.block_k
        kv_repeat = ctx.kv_repeat
        span_len_factor = float(getattr(ctx, "span_len_factor", 2.0))
        span_power = float(getattr(ctx, "span_power", 0.5))

        B, H_q, L_Q, D = Q2.shape
        _, H_kv, L_KV, _ = K.shape
        num_spans = qstart.shape[-1]

        window = int(getattr(ctx, "window_len", (sw_index + 1) ** 2 - 1))
        max_span_len = int(span_len_factor * math.ceil(float(L_KV) ** span_power) + 2)
        span_max_blocks = triton.cdiv(max_span_len, block_k)
        span_max_blocks = max(1, span_max_blocks)
        sw_max_len = min(window, L_KV) if window > 0 else 0
        sw_max_blocks = triton.cdiv(sw_max_len, block_k) if sw_max_len > 0 else 0
        block_d = min(256, _next_power_of_two(D))

        Q2c = Q2.contiguous()
        Kc = K.contiguous()
        Vc = V.contiguous()
        qstartc = qstart.contiguous()
        qendc = qend.contiguous()
        cachec = cache_position.to(torch.int32).contiguous()
        grad_out_c = grad_out.contiguous()

        if attention_mask is not None:
            attn_mask = attention_mask[:, :L_KV].contiguous().to(torch.int8)
            has_mask = True
        else:
            attn_mask = torch.empty((1,), device=Q2.device, dtype=torch.int8)
            has_mask = False

        dQ = torch.zeros_like(Q2c, dtype=torch.float32)
        split_dkv = os.getenv("SPAN_ATTN_GQA_BACKWARD_SPLIT_DKV", "1") != "0"
        if split_dkv and kv_repeat > 1:
            dK_rep = torch.zeros((B, H_q, L_KV, D), device=Kc.device, dtype=torch.float32)
            dV_rep = torch.zeros((B, H_q, L_KV, D), device=Vc.device, dtype=torch.float32)
            dK_out = dK_rep
            dV_out = dV_rep
        else:
            dK = torch.zeros_like(Kc, dtype=torch.float32)
            dV = torch.zeros_like(Vc, dtype=torch.float32)
            dK_out = dK
            dV_out = dV

        write_dkv_per_qhead = split_dkv and kv_repeat > 1
        fuse_spans = os.getenv("SPAN_ATTN_GQA_BACKWARD_FUSE_SPANS", "1") != "0"
        # K_VAL must be power of 2 for tl.arange
        k_val_padded = 1 << (num_spans - 1).bit_length() if num_spans > 0 else 1
        if fuse_spans and num_spans > 1:
            grid = (B * H_q * L_Q,)
            fused_span_backward_kernel_gqa_fused_spans[grid](
                Q2c, Kc, Vc, grad_out_c,
                qstartc, qendc, cachec,
                attn_mask,
                dQ, dK_out, dV_out,
                B, H_q, H_kv, L_Q, L_KV,
                kv_repeat,
                window,
                1.0 / math.sqrt(D),
                K_VAL=k_val_padded,
                K_ACTUAL=num_spans,
                BLOCK_K=block_k,
                BLOCK_D=block_d,
                D_HEAD=D,
                SPAN_MAX_BLOCKS=span_max_blocks,
                SW_MAX_BLOCKS=sw_max_blocks,
                HAS_ATTN_MASK=has_mask,
                WRITE_DKV_PER_QHEAD=write_dkv_per_qhead,
            )
        else:
            grid = (B * H_q * L_Q * num_spans,)
            fused_span_backward_kernel_gqa[grid](
                Q2c, Kc, Vc, grad_out_c,
                qstartc, qendc, cachec,
                attn_mask,
                dQ, dK_out, dV_out,
                B, H_q, H_kv, L_Q, L_KV,
                kv_repeat,
                window,
                1.0 / math.sqrt(D),
                K_VAL=num_spans,
                BLOCK_K=block_k,
                BLOCK_D=block_d,
                D_HEAD=D,
                SPAN_MAX_BLOCKS=span_max_blocks,
                SW_MAX_BLOCKS=sw_max_blocks,
                HAS_ATTN_MASK=has_mask,
                WRITE_DKV_PER_QHEAD=write_dkv_per_qhead,
            )

        if split_dkv and kv_repeat > 1:
            dK = dK_rep.view(B, H_kv, kv_repeat, L_KV, D).sum(dim=2)
            dV = dV_rep.view(B, H_kv, kv_repeat, L_KV, D).sum(dim=2)

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


def fused_span_attention_gqa(
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
    span_power: float = 0.5,
    search_power: float | None = None,
    inv_search_power_int: int | None = 2,
):
    B, H_q, L_Q, _ = Q2.shape
    _, H_kv, _, _ = K.shape
    assert H_q % H_kv == 0, "Query heads must be divisible by KV heads when using GQA"

    if not Q2.is_cuda:
        params = derive_stripe_power_params(
            search_power=search_power, inv_search_power_int=inv_search_power_int
        )
        if params.triton_inv_n != 2:
            raise NotImplementedError(
                "Non-CUDA fused_span_attention_gqa only supports p=0.5 for now "
                "(use inv_search_power_int=2 or search_power=0.5)."
            )
        kv_repeat = H_q // H_kv
        K_rep = K.repeat_interleave(kv_repeat, dim=1)
        V_rep = V.repeat_interleave(kv_repeat, dim=1)
        return fused_span_attention(
            Q2,
            K_rep,
            V_rep,
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

    return FusedSpanGQATriton.apply(
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



def decode_span_attention_staged_gqa_kernel(
    Q1,
    Q2,
    K,
    V,
    cache_position,
    attention_mask=None,
    sw_index=0,
    topk=3,
    enable_gqa=False,
    block_k=64,
    backward_factor: float = 2.0,
    forward_factor: float = 0.0,
    span_power: float = 0.5,
    search_power: float | None = None,
    inv_search_power_int: int | None = 2,
    force_mode=None,
):
    return decode_span_attention_staged_gqa_kernel_v2(
        Q1,
        Q2,
        K,
        V,
        cache_position,
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


def decode_span_attention_staged_gqa_kernel_v2(
    Q1,
    Q2,
    K,
    V,
    cache_position,
    attention_mask=None,
    sw_index=0,
    topk=3,
    enable_gqa=False,
    block_k=64,
    backward_factor: float = 2.0,
    forward_factor: float = 0.0,
    span_power: float = 0.5,
    search_power: float | None = None,
    inv_search_power_int: int | None = 2,
    force_mode=None,
):
    """Staged decode variant that repeats K/V for small-window SDPA to avoid GQA overhead."""
    if (not enable_gqa) or (K.shape[1] == Q1.shape[1]):
        params = derive_stripe_power_params(
            search_power=search_power, inv_search_power_int=inv_search_power_int
        )
        if params.triton_inv_n != 2:
            raise NotImplementedError(
                "decode_span_attention_staged_gqa_kernel_v2 only supports p!=0.5 when enable_gqa=True. "
                "Use enable_gqa=True or keep p=0.5 (inv_search_power_int=2 or search_power=0.5)."
            )
        return decode_span_attention_staged(
            Q1,
            Q2,
            K,
            V,
            cache_position,
            attention_mask=attention_mask,
            sw_index=sw_index,
            topk=topk,
            backward_factor=backward_factor,
            forward_factor=forward_factor,
            span_power=span_power,
            search_power=search_power,
            inv_search_power_int=inv_search_power_int,
        )

    B, H_q, L_Q, D = Q1.shape
    _, H_kv, L_KV, D_kv = K.shape
    assert L_Q == 1, "GQA decode only supports decoding (L_Q=1)"
    assert D_kv == D, "Q/K/V head dimensions must match"
    assert H_q % H_kv == 0, "Query heads must be divisible by KV heads when GQA is enabled"

    device = Q1.device
    kv_repeat = H_q // H_kv

    window_len = window_len_from_sw_index(
        int(sw_index), search_power=search_power, inv_search_power_int=inv_search_power_int
    )
    # ------------------------------------------------------------------------
    # IMPORTANT (StaticCache correctness):
    # StaticCache returns full fixed-size K/V buffers during decode (L_KV = max_seq_len) to
    # keep shapes stable for CUDA graphs. For short prefixes, we must still take the same
    # "full-attention" SDPA fallback that DynamicCache would take (based on the *effective*
    # prefix length, i.e., cache_position[-1] + 1), otherwise small numeric diffs can
    # accumulate and change argmax tokens.
    #
    # To keep CUDA-graph safety (no data-dependent Python branching), we compute both:
    # - SDPA fallback over a small fixed prefix window (<= window_len)
    # - the staged span-attention output
    # and select with a tensor mask.
    # ------------------------------------------------------------------------
    token_pos = cache_position[-1] + 1  # effective prefix length (1-indexed)
    use_sdpa = token_pos <= window_len

    if force_mode not in (None, "sdpa", "span"):
        force_mode = None

    out_sdpa = None
    if force_mode != "span":
        # SDPA fallback: only needs the first `min(L_KV, window_len)` keys. For StaticCache,
        # attention_mask is expected to mask out positions beyond the current prefix.
        kv_slice_len = min(L_KV, max(window_len, 1))
        K_sdpa = K[:, :, :kv_slice_len, :]
        V_sdpa = V[:, :, :kv_slice_len, :]
        if attention_mask is not None:
            sdpa_mask = attention_mask[:, :kv_slice_len]
        else:
            # Synthesize a prefix mask so StaticCache doesn't attend to the unused tail.
            pos = torch.arange(kv_slice_len, device=device, dtype=cache_position.dtype)
            sdpa_mask = (pos <= cache_position[-1]).unsqueeze(0).expand(B, -1)

        attn_mask = sdpa_mask.unsqueeze(1).unsqueeze(2)
        # CUDA graph safe: use arithmetic instead of torch.tensor()
        # True (1.0) -> 0.0, False (0.0) -> -1e9
        attn_mask_float = attn_mask.to(Q2.dtype)
        attn_mask = (attn_mask_float - 1.0) * 1e9

        K_rep = K_sdpa.repeat_interleave(kv_repeat, dim=1)
        V_rep = V_sdpa.repeat_interleave(kv_repeat, dim=1)
        out_sdpa = torch.nn.functional.scaled_dot_product_attention(
            Q2,
            K_rep,
            V_rep,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=False,
        )
        if force_mode == "sdpa":
            return out_sdpa

    # ========================================================================
    # CUDA Graph Compatible Version: Use fixed-size tensors with masking
    # instead of dynamic filtering operations (no stripe_loc[mask] indexing)
    # ========================================================================
    
    num_stripes = max(
        int(
            max_stripe_index_for_token_pos(
                int(L_KV),
                search_power=search_power,
                inv_search_power_int=inv_search_power_int,
            )
        ),
        sw_index + 1,
    )
    stripe_idx = torch.arange(sw_index + 1, num_stripes + 1, device=device)
    # IMPORTANT: For StaticCache decode, K/V are full fixed-size buffers (L_KV=max_seq_len),
    # so we must anchor stripes relative to the *effective* prefix length (token_pos),
    # not the allocation length. For DynamicCache, token_pos == L_KV so this is unchanged.
    power_params = derive_stripe_power_params(
        search_power=search_power, inv_search_power_int=inv_search_power_int
    )
    if power_params.triton_inv_n != 0:
        stripe_floor_power = stripe_idx.to(torch.int64) ** int(power_params.triton_inv_n)
    else:
        stripe_floor_power = torch.floor(
            stripe_idx.to(torch.float32) ** float(power_params.inv_p)
        ).to(torch.int64)
    stripe_loc_all = token_pos - stripe_floor_power
    
    # Create validity mask instead of filtering
    # Mask 1: stripe_loc >= 0
    valid_mask = stripe_loc_all >= 0
    
    # Mask 2: attention_mask check (if provided)
    if attention_mask is not None:
        # Use safe indexing with clamping
        safe_indices = stripe_loc_all.clamp(min=0, max=L_KV-1)
        attn_valid = attention_mask[0, safe_indices].to(torch.bool)
        valid_mask = valid_mask & attn_valid
    
    # Mask 3: no overlap with sliding window
    # Use tensor operations instead of .item() for CUDA graph compatibility
    decode_pos_tensor = cache_position[-1]
    sw_start_tensor = torch.clamp(decode_pos_tensor - (window_len - 1), min=0)
    sw_valid = stripe_loc_all < sw_start_tensor
    valid_mask = valid_mask & sw_valid
    
    # Convert stripe_loc to int64 and mark invalid entries as -1
    stripe_loc = torch.where(valid_mask, stripe_loc_all, -1).to(torch.int64)
    
    # Initialize topk_vals and topk_idx with proper dimensions
    topk_vals = torch.full((B, H_q, topk), float('-inf'), device=device, dtype=torch.float32)
    topk_idx = torch.full((B, H_q, topk), -1, device=device, dtype=torch.int64)
    
    # CUDA Graph Compatible: Always compute logits, use masking to handle empty cases
    # No conditional branches based on .item() calls
    max_stripes = stripe_loc.shape[0]
    
    # Use safe indexing with clamping to avoid out-of-bounds for invalid stripes
    safe_stripe_locs = stripe_loc.clamp(min=0, max=L_KV-1)
    K_stripe = K.index_select(2, safe_stripe_locs)
    
    logits = torch.einsum(
        'bhrld,bhsd->bhrls',
        Q1.detach().float().reshape(B, H_kv, kv_repeat, L_Q, D),
        K_stripe.detach().float()
    ).squeeze(3)
    logits = logits.reshape(B, H_q, -1)
    
    # Set invalid stripe logits to -inf so they don't get selected in topk
    invalid_stripe_mask = ~valid_mask.unsqueeze(0).unsqueeze(0).expand(B, H_q, -1)
    logits = torch.where(invalid_stripe_mask, float('-inf'), logits)
    
    # Top-k over all stripes (invalid ones have -inf logits)
    actual_k = min(topk, max_stripes)
    actual_topk_vals, actual_topk_idx = torch.topk(logits, k=actual_k, dim=-1)
    
    topk_vals[:, :, :actual_k] = actual_topk_vals
    topk_idx[:, :, :actual_k] = actual_topk_idx
    
    # Map indices to stripe locations using the full stripe_loc tensor
    # Invalid topk_idx (-1) will produce qanchor=-1
    qanchor = torch.full_like(topk_idx, -1, dtype=torch.int64)
    safe_idx = topk_idx.clamp(min=0, max=stripe_loc.shape[0]-1)
    valid_topk_mask = topk_idx >= 0
    selected_locs = stripe_loc[safe_idx]
    qanchor = torch.where(valid_topk_mask, selected_locs, qanchor)

    # Use tensor operations instead of .item() for CUDA graph compatibility
    # Compute span_len using tensor operations
    span_len = torch.ceil(float(backward_factor) * (token_pos.float() ** float(span_power))).to(torch.int64)
    qstart = qanchor - span_len
    qstart = torch.clamp(qstart, min=0)
    # For invalid spans (qanchor=-1), set qstart to -1 as well
    qstart = torch.where(qanchor < 0, -1, qstart)

    qstart = qstart.unsqueeze(2)
    qanchor = qanchor.unsqueeze(2)

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
    O_span = fused_span_attention_gqa(
        Q2,
        K,
        V,
        qstart,
        qend,
        decode_cache_pos,
        attention_mask=attention_mask,
        sw_index=sw_index,
        block_k=block_k,
        span_len_factor=span_len_factor,
        span_power=span_power,
        search_power=search_power,
        inv_search_power_int=inv_search_power_int,
    )

    # Reuse topk_vals directly for gating (no need to recompute K[qend] @ Q1)
    span_values = topk_vals.unsqueeze(2)  # [B, H_q, 1, topk]
    # Invalid spans already have -inf in topk_vals, but ensure consistency
    span_values = torch.where(qanchor < 0, float("-inf"), span_values)

    span_scores = torch.nan_to_num(torch.softmax(span_values.float(), dim=-1), 1 / topk)
    span_scores = span_scores.to(O_span.dtype)
    O = (span_scores.unsqueeze(-1) * O_span).sum(dim=3)

    if force_mode == "span":
        return O

    # Select SDPA for short prefixes, span attention otherwise.
    use_sdpa_broadcast = use_sdpa.view(1, 1, 1, 1)
    return torch.where(use_sdpa_broadcast, out_sdpa, O)
