from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from superlinear.kernels.common.power import window_len_from_sw_index

# Defaults copied from the reference notebooks (56.12 / 56.13)
DEFAULT_SW_BLOCK_K = 64
DEFAULT_SW_NUM_WARPS = 4
DEFAULT_SW_NUM_STAGES = 2

# Fixed LOG_KV for binary search - supports up to 33M tokens
# Using fixed value eliminates recompilation (binary search exits early anyway)
FIXED_LOG_KV = 25
MAX_SUPPORTED_SEQ_LEN = (1 << FIXED_LOG_KV)  # 33,554,432 tokens

# Dynamic loops: We use range(sw_blocks) instead of tl.static_range(SW_MAX_BLOCKS)
# This eliminates recompilation when window sizes change - the kernel compiles once
# and runs efficiently for any window size. Testing shows:
# - Static loop compiles in 8-160s depending on SW_MAX_BLOCKS value  
# - Dynamic loop compiles in ~0.3s regardless of window size
# - No recompilation when sw_blocks changes (0.0003s vs 8+s)


def _bucket_log_kv(L_KV: int) -> int:
    """Return fixed value to avoid any recompilation."""
    return FIXED_LOG_KV


def _next_pow2_int(x: int) -> int:
    x = int(x)
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


def make_kv_for_q_map(*, B: int, H_q: int, H_kv: int, device: torch.device) -> torch.Tensor:
    """Return int32 tensor of shape [B*H_q] mapping bh_q -> bh_kv."""
    if H_kv <= 0 or H_q <= 0:
        raise ValueError("H_q and H_kv must be > 0")
    if H_q % H_kv != 0:
        raise ValueError(f"H_q must be divisible by H_kv (got H_q={H_q}, H_kv={H_kv})")
    kv_repeat = H_q // H_kv

    bh_q = torch.arange(B * H_q, device=device, dtype=torch.int64)
    batch = bh_q // H_q
    hq = bh_q - batch * H_q
    hkv = hq // kv_repeat
    bh_kv = batch * H_kv + hkv
    return bh_kv.to(torch.int32)


@triton.jit
def _sliding_window_attn_forward_gqa_kernel_token(
    Q_ptr,
    K_ptr,
    V_ptr,
    kv_for_q_ptr,
    q_cache_position_ptr,
    k_cache_position_ptr,
    attn_mask_ptr,
    out_ptr,
    lse_ptr,
    L_Q,
    L_KV,
    stride_qb,
    stride_qh,
    stride_ql,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kl,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vl,
    stride_vd,
    D: tl.constexpr,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    sw_max_blocks,  # runtime: max SW blocks - use dynamic loop
    LOG_KV: tl.constexpr,
    KEY_POS_DENSE: tl.constexpr,
    SM_SCALE: tl.constexpr,
    USE_BF16_PV: tl.constexpr,
    HAS_ATTN_MASK: tl.constexpr,
):
    """Token-parallel sliding-window attention (GQA).

    General fallback; supports KEY_POS_DENSE=False via binary search.
    Uses dynamic loops for sw_blocks to avoid recompilation.
    """
    pid = tl.program_id(0)
    q_idx = pid % L_Q
    bh_q = pid // L_Q

    d = tl.arange(0, BLOCK_D)
    d_mask = d < D
    d_i64 = d.to(tl.int64)

    bh_q_i64 = bh_q.to(tl.int64)
    q_i64 = q_idx.to(tl.int64)

    batch = bh_q // H_Q
    batch_i64 = batch.to(tl.int64)
    hq = bh_q - batch * H_Q
    hq_i64 = hq.to(tl.int64)

    L_Q_i64 = L_Q.to(tl.int64)
    L_KV_i64 = L_KV.to(tl.int64)

    q_ptrs = Q_ptr + batch_i64 * stride_qb + hq_i64 * stride_qh + q_i64 * stride_ql + d_i64 * stride_qd
    q = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    kv_bh = tl.load(kv_for_q_ptr + bh_q).to(tl.int32)
    b_kv = kv_bh // H_KV
    h_kv = kv_bh - b_kv * H_KV
    b_kv_i64 = b_kv.to(tl.int64)
    h_kv_i64 = h_kv.to(tl.int64)

    q_pos = tl.load(q_cache_position_ptr + batch_i64 * L_Q_i64 + q_i64, mask=True, other=-1).to(tl.int32)

    if KEY_POS_DENSE:
        end_idx = tl.minimum(q_pos, L_KV - 1)
        end_idx = tl.maximum(end_idx, -1)
        start_idx = end_idx - (WINDOW - 1)
        start_idx = tl.maximum(start_idx, 0)
    else:
        last_i64 = L_KV_i64 - 1
        max_k_pos = tl.load(
            k_cache_position_ptr + batch_i64 * L_KV_i64 + last_i64, mask=True, other=0
        ).to(tl.int32)

        sw_end_pos = tl.minimum(q_pos, max_k_pos)
        sw_start_pos = sw_end_pos - (WINDOW - 1)

        lo = tl.zeros((), dtype=tl.int32)
        hi = L_KV.to(tl.int32)
        for _ in tl.static_range(0, LOG_KV):
            mid = (lo + hi) // 2
            mid_val = tl.load(
                k_cache_position_ptr + batch_i64 * L_KV_i64 + mid.to(tl.int64),
                mask=True,
                other=2147483647,
            ).to(tl.int32)
            go_right = mid_val < sw_start_pos
            lo = tl.where(go_right, mid + 1, lo)
            hi = tl.where(go_right, hi, mid)
        start_idx = lo

        lo = tl.zeros((), dtype=tl.int32)
        hi = L_KV.to(tl.int32)
        for _ in tl.static_range(0, LOG_KV):
            mid = (lo + hi) // 2
            mid_val = tl.load(
                k_cache_position_ptr + batch_i64 * L_KV_i64 + mid.to(tl.int64),
                mask=True,
                other=2147483647,
            ).to(tl.int32)
            go_right = mid_val <= sw_end_pos
            lo = tl.where(go_right, mid + 1, lo)
            hi = tl.where(go_right, hi, mid)
        end_excl = lo
        end_idx = end_excl - 1

    sw_valid = (WINDOW > 0) & (q_pos >= 0) & (start_idx < L_KV) & (end_idx >= 0) & (start_idx <= end_idx)

    # Compute actual number of SW blocks needed (dynamic)
    sw_len = tl.where(sw_valid, end_idx - start_idx + 1, 0)
    sw_blocks = (sw_len + BLOCK_K - 1) // BLOCK_K
    sw_blocks = tl.minimum(sw_blocks, sw_max_blocks)

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros((1, BLOCK_D), dtype=tl.float32)

    # Dynamic range loop - compiles once, no recompilation when sw_blocks changes
    for block_idx in range(sw_blocks):
        block_start = start_idx + block_idx * BLOCK_K
        k_idx = block_start + tl.arange(0, BLOCK_K)

        in_range = sw_valid & (k_idx >= start_idx) & (k_idx <= end_idx) & (k_idx < L_KV)
        k_safe = tl.where(in_range, k_idx, 0)
        k_i64 = k_safe.to(tl.int64)

        k_ptrs = (
            K_ptr
            + b_kv_i64 * stride_kb
            + h_kv_i64 * stride_kh
            + k_i64[:, None] * stride_kl
            + d_i64[None, :] * stride_kd
        )
        v_ptrs = (
            V_ptr
            + b_kv_i64 * stride_vb
            + h_kv_i64 * stride_vh
            + k_i64[:, None] * stride_vl
            + d_i64[None, :] * stride_vd
        )

        k_tile = tl.load(k_ptrs, mask=in_range[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        v_tile = tl.load(v_ptrs, mask=in_range[:, None] & d_mask[None, :], other=0.0)

        scores = tl.sum(k_tile * q[None, :], axis=1) * SM_SCALE

        if HAS_ATTN_MASK:
            key_mask = tl.load(attn_mask_ptr + batch_i64 * L_KV_i64 + k_i64, mask=in_range, other=0).to(tl.int8)
            in_range = in_range & (key_mask != 0)

        scores = tl.where(in_range, scores, -float("inf"))

        block_max = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, block_max)

        m_safe = tl.where(m_i == -float("inf"), 0.0, m_i)
        m_new_safe = tl.where(m_new == -float("inf"), 0.0, m_new)

        alpha = tl.exp(m_safe - m_new_safe)
        p = tl.exp(scores - m_new_safe)

        l_i = l_i * alpha + tl.sum(p, axis=0)

        if USE_BF16_PV:
            acc = acc * alpha + tl.dot(p[None, :].to(tl.bfloat16), v_tile.to(tl.bfloat16))
        else:
            acc = acc * alpha + tl.sum(p[:, None] * v_tile.to(tl.float32), axis=0)[None, :]

        m_i = m_new

    out = tl.where(l_i > 0, acc / l_i, 0.0)
    out_ptrs = out_ptr + bh_q_i64 * (L_Q_i64 * D) + q_i64 * D + d_i64[None, :]
    tl.store(out_ptrs, out, mask=d_mask[None, :])

    lse = m_i + tl.log(l_i)
    tl.store(lse_ptr + bh_q_i64 * L_Q_i64 + q_i64, lse)


@triton.jit
def _sliding_window_attn_forward_gqa_kernel_blockq_dense(
    Q_ptr,
    K_ptr,
    V_ptr,
    kv_for_q_ptr,
    q_cache_position_ptr,
    attn_mask_ptr,
    out_ptr,
    lse_ptr,
    L_Q,
    L_KV,
    stride_qb,
    stride_qh,
    stride_ql,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kl,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vl,
    stride_vd,
    D: tl.constexpr,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    sw_max_blocks,  # runtime: max SW blocks - use dynamic loop
    SM_SCALE: tl.constexpr,
    USE_BF16_PV: tl.constexpr,
    HAS_ATTN_MASK: tl.constexpr,
):
    """Block-of-queries sliding-window attention (GQA), dense key positions.
    
    Uses dynamic loops for sw_blocks to avoid recompilation.
    """
    bh_q = tl.program_id(0)
    pid_q = tl.program_id(1)

    q_offs = pid_q * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_q = q_offs < L_Q

    d = tl.arange(0, BLOCK_D)
    d_mask = d < D
    d_i64 = d.to(tl.int64)

    bh_q_i64 = bh_q.to(tl.int64)
    q_offs_i64 = q_offs.to(tl.int64)

    batch = bh_q // H_Q
    batch_i64 = batch.to(tl.int64)
    hq = bh_q - batch * H_Q
    hq_i64 = hq.to(tl.int64)

    L_Q_i64 = L_Q.to(tl.int64)
    L_KV_i64 = L_KV.to(tl.int64)

    # Strides can exceed int32 at long context (e.g. stride_kb = H_KV * L_KV * D).
    # Force all pointer arithmetic to use int64 to avoid wraparound near ~8M tokens.
    stride_qb_i64 = tl.cast(stride_qb, tl.int64)
    stride_qh_i64 = tl.cast(stride_qh, tl.int64)
    stride_ql_i64 = tl.cast(stride_ql, tl.int64)
    stride_qd_i64 = tl.cast(stride_qd, tl.int64)
    stride_kb_i64 = tl.cast(stride_kb, tl.int64)
    stride_kh_i64 = tl.cast(stride_kh, tl.int64)
    stride_kl_i64 = tl.cast(stride_kl, tl.int64)
    stride_kd_i64 = tl.cast(stride_kd, tl.int64)
    stride_vb_i64 = tl.cast(stride_vb, tl.int64)
    stride_vh_i64 = tl.cast(stride_vh, tl.int64)
    stride_vl_i64 = tl.cast(stride_vl, tl.int64)
    stride_vd_i64 = tl.cast(stride_vd, tl.int64)

    q_pos = tl.load(
        q_cache_position_ptr + batch_i64 * L_Q_i64 + q_offs_i64, mask=mask_q, other=-1
    ).to(tl.int32)

    end_idx = tl.minimum(q_pos, L_KV - 1)
    end_idx = tl.maximum(end_idx, -1)
    start_idx = end_idx - (WINDOW - 1)
    start_idx = tl.maximum(start_idx, 0)

    sw_valid = mask_q & (q_pos >= 0) & (start_idx < L_KV) & (end_idx >= 0) & (start_idx <= end_idx)

    start_eff = tl.where(sw_valid, start_idx, 0)
    end_eff = tl.where(sw_valid, end_idx, -1)
    start_u = -tl.max(-start_eff, axis=0)
    end_u = tl.max(end_eff, axis=0)

    kv_bh = tl.load(kv_for_q_ptr + bh_q).to(tl.int32)
    b_kv = kv_bh // H_KV
    h_kv = kv_bh - b_kv * H_KV
    b_kv_i64 = b_kv.to(tl.int64)
    h_kv_i64 = h_kv.to(tl.int64)

    q_ptrs = (
        Q_ptr
        + batch_i64 * stride_qb_i64
        + hq_i64 * stride_qh_i64
        + q_offs_i64[:, None] * stride_ql_i64
        + d_i64[None, :] * stride_qd_i64
    )
    q_tile = tl.load(q_ptrs, mask=mask_q[:, None] & d_mask[None, :], other=0.0).to(tl.bfloat16)

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    k_rel = tl.arange(0, BLOCK_K)
    k_rel_i64 = k_rel.to(tl.int64)

    # Compute actual number of SW blocks needed for union range
    sw_len = tl.where(end_u >= start_u, end_u - start_u + 1, 0)
    sw_blocks = (sw_len + BLOCK_K - 1) // BLOCK_K
    sw_blocks = tl.minimum(sw_blocks, sw_max_blocks)

    # Dynamic range loop - compiles once, no recompilation when sw_blocks changes
    for block_idx in range(sw_blocks):
        k_start = start_u + block_idx * BLOCK_K
        k_idx = k_start + k_rel

        k_in_bounds = k_idx < L_KV
        k_i64 = k_idx.to(tl.int64)

        k_ptrs = (
            K_ptr
            + b_kv_i64 * stride_kb_i64
            + h_kv_i64 * stride_kh_i64
            + k_i64[:, None] * stride_kl_i64
            + d_i64[None, :] * stride_kd_i64
        )
        v_ptrs = (
            V_ptr
            + b_kv_i64 * stride_vb_i64
            + h_kv_i64 * stride_vh_i64
            + k_i64[:, None] * stride_vl_i64
            + d_i64[None, :] * stride_vd_i64
        )

        k_tile = tl.load(k_ptrs, mask=k_in_bounds[:, None] & d_mask[None, :], other=0.0).to(tl.bfloat16)
        v_tile = tl.load(v_ptrs, mask=k_in_bounds[:, None] & d_mask[None, :], other=0.0)

        scores = tl.dot(q_tile, tl.trans(k_tile)).to(tl.float32) * SM_SCALE

        in_range = (
            sw_valid[:, None]
            & k_in_bounds[None, :]
            & (k_idx[None, :] >= start_idx[:, None])
            & (k_idx[None, :] <= end_idx[:, None])
        )

        if HAS_ATTN_MASK:
            key_mask = tl.load(attn_mask_ptr + batch_i64 * L_KV_i64 + k_i64, mask=k_in_bounds, other=0).to(tl.int8)
            in_range = in_range & (key_mask[None, :] != 0)

        scores = tl.where(in_range, scores, -float("inf"))

        block_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, block_max)

        m_safe = tl.where(m_i == -float("inf"), 0.0, m_i)
        m_new_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp(m_safe - m_new_safe)

        p = tl.exp(scores - m_new_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)

        if USE_BF16_PV:
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v_tile.to(tl.bfloat16))
        else:
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float32), v_tile.to(tl.float32))

        m_i = m_new

    out = tl.where(l_i[:, None] > 0, acc / l_i[:, None], 0.0)

    out_ptrs = out_ptr + bh_q_i64 * (L_Q_i64 * D) + q_offs_i64[:, None] * D + d_i64[None, :]
    tl.store(out_ptrs, out, mask=mask_q[:, None] & d_mask[None, :])

    lse = tl.where(l_i > 0, m_i + tl.log(l_i), -float("inf"))
    tl.store(lse_ptr + bh_q_i64 * L_Q_i64 + q_offs_i64, lse, mask=mask_q)


@triton.jit
def _sliding_window_attn_backward_gqa_kernel_token(
    Q_ptr,
    K_ptr,
    V_ptr,
    kv_for_q_ptr,
    q_cache_position_ptr,
    k_cache_position_ptr,
    attn_mask_ptr,
    out_ptr,
    lse_ptr,
    dOut_ptr,
    dLse_ptr,
    dQ_ptr,
    dK_ptr,
    dV_ptr,
    L_Q,
    L_KV,
    stride_qb,
    stride_qh,
    stride_ql,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kl,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vl,
    stride_vd,
    D: tl.constexpr,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    sw_max_blocks,  # runtime: max SW blocks - use dynamic loop
    LOG_KV: tl.constexpr,
    KEY_POS_DENSE: tl.constexpr,
    SM_SCALE: tl.constexpr,
    HAS_ATTN_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    q_idx = pid % L_Q
    bh_q = pid // L_Q

    d = tl.arange(0, BLOCK_D)
    d_mask = d < D
    d_i64 = d.to(tl.int64)

    bh_q_i64 = bh_q.to(tl.int64)
    q_i64 = q_idx.to(tl.int64)

    L_Q_i64 = L_Q.to(tl.int64)
    L_KV_i64 = L_KV.to(tl.int64)

    batch = bh_q // H_Q
    batch_i64 = batch.to(tl.int64)
    hq = bh_q - batch * H_Q
    hq_i64 = hq.to(tl.int64)

    q_ptrs = Q_ptr + batch_i64 * stride_qb + hq_i64 * stride_qh + q_i64 * stride_ql + d_i64 * stride_qd
    q = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    kv_bh = tl.load(kv_for_q_ptr + bh_q, mask=True, other=0).to(tl.int32)
    b_kv = kv_bh // H_KV
    h_kv = kv_bh - b_kv * H_KV
    b_kv_i64 = b_kv.to(tl.int64)
    h_kv_i64 = h_kv.to(tl.int64)

    out_ptrs = out_ptr + bh_q_i64 * (L_Q_i64 * D) + q_i64 * D + d_i64
    out = tl.load(out_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    dOut_ptrs = dOut_ptr + bh_q_i64 * (L_Q_i64 * D) + q_i64 * D + d_i64
    dO = tl.load(dOut_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    lse = tl.load(lse_ptr + bh_q_i64 * L_Q_i64 + q_i64, mask=True, other=-float("inf")).to(tl.float32)
    dLse = tl.load(dLse_ptr + bh_q_i64 * L_Q_i64 + q_i64, mask=True, other=0.0).to(tl.float32)

    dot = tl.sum(out * dO, axis=0)

    q_pos = tl.load(q_cache_position_ptr + batch_i64 * L_Q_i64 + q_i64, mask=True, other=-1).to(tl.int32)

    if KEY_POS_DENSE:
        end_idx = tl.minimum(q_pos, L_KV - 1)
        end_idx = tl.maximum(end_idx, -1)
        start_idx = end_idx - (WINDOW - 1)
        start_idx = tl.maximum(start_idx, 0)
    else:
        last_i64 = L_KV_i64 - 1
        max_k_pos = tl.load(
            k_cache_position_ptr + batch_i64 * L_KV_i64 + last_i64, mask=True, other=0
        ).to(tl.int32)

        sw_end_pos = tl.minimum(q_pos, max_k_pos)
        sw_start_pos = sw_end_pos - (WINDOW - 1)

        lo = tl.zeros((), dtype=tl.int32)
        hi = L_KV.to(tl.int32)
        for _ in tl.static_range(0, LOG_KV):
            mid = (lo + hi) // 2
            mid_val = tl.load(
                k_cache_position_ptr + batch_i64 * L_KV_i64 + mid.to(tl.int64),
                mask=True,
                other=2147483647,
            ).to(tl.int32)
            go_right = mid_val < sw_start_pos
            lo = tl.where(go_right, mid + 1, lo)
            hi = tl.where(go_right, hi, mid)
        start_idx = lo

        lo = tl.zeros((), dtype=tl.int32)
        hi = L_KV.to(tl.int32)
        for _ in tl.static_range(0, LOG_KV):
            mid = (lo + hi) // 2
            mid_val = tl.load(
                k_cache_position_ptr + batch_i64 * L_KV_i64 + mid.to(tl.int64),
                mask=True,
                other=2147483647,
            ).to(tl.int32)
            go_right = mid_val <= sw_end_pos
            lo = tl.where(go_right, mid + 1, lo)
            hi = tl.where(go_right, hi, mid)
        end_excl = lo
        end_idx = end_excl - 1

    sw_valid = (WINDOW > 0) & (q_pos >= 0) & (start_idx < L_KV) & (end_idx >= 0) & (start_idx <= end_idx)

    has_any = sw_valid & (lse != -float("inf"))
    lse_safe = tl.where(has_any, lse, 0.0)
    dot_safe = tl.where(has_any, dot, 0.0)
    dLse_safe = tl.where(has_any, dLse, 0.0)

    # Compute actual number of SW blocks needed (dynamic)
    sw_len = tl.where(has_any, end_idx - start_idx + 1, 0)
    sw_blocks = (sw_len + BLOCK_K - 1) // BLOCK_K
    sw_blocks = tl.minimum(sw_blocks, sw_max_blocks)

    grad_q = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # Dynamic range loop - compiles once, no recompilation when sw_blocks changes
    for kb in range(sw_blocks):
        block_start = start_idx + kb * BLOCK_K
        k_idx = block_start + tl.arange(0, BLOCK_K)

        in_range = has_any & (k_idx >= start_idx) & (k_idx <= end_idx) & (k_idx < L_KV)

        k_safe = tl.where(in_range, k_idx, 0).to(tl.int32)
        k_i64 = k_safe.to(tl.int64)

        k_ptrs = (
            K_ptr
            + b_kv_i64 * stride_kb
            + h_kv_i64 * stride_kh
            + k_i64[:, None] * stride_kl
            + d_i64[None, :] * stride_kd
        )
        k_tile = tl.load(k_ptrs, mask=in_range[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        v_ptrs = (
            V_ptr
            + b_kv_i64 * stride_vb
            + h_kv_i64 * stride_vh
            + k_i64[:, None] * stride_vl
            + d_i64[None, :] * stride_vd
        )
        v_tile = tl.load(v_ptrs, mask=in_range[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        scores = tl.sum(k_tile * q[None, :], axis=1) * SM_SCALE

        if HAS_ATTN_MASK:
            key_mask = tl.load(attn_mask_ptr + batch_i64 * L_KV_i64 + k_i64, mask=in_range, other=0).to(tl.int8)
            in_range = in_range & (key_mask != 0)

        scores = tl.where(in_range, scores, -float("inf"))

        w = tl.exp(scores - lse_safe)
        w = tl.where(in_range, w, 0.0)

        v_grad_ptrs = (
            dV_ptr
            + b_kv_i64 * stride_vb
            + h_kv_i64 * stride_vh
            + k_i64[:, None] * stride_vl
            + d_i64[None, :] * stride_vd
        )
        dv = w[:, None] * dO[None, :]
        tl.atomic_add(v_grad_ptrs, dv, mask=in_range[:, None] & d_mask[None, :])

        grad_w = tl.sum(v_tile * dO[None, :], axis=1)
        grad_logits = w * (grad_w - dot_safe + dLse_safe)
        grad_s = grad_logits * SM_SCALE

        grad_q += tl.sum(grad_s[:, None] * k_tile, axis=0)

        k_grad_ptrs = (
            dK_ptr
            + b_kv_i64 * stride_kb
            + h_kv_i64 * stride_kh
            + k_i64[:, None] * stride_kl
            + d_i64[None, :] * stride_kd
        )
        dk = grad_s[:, None] * q[None, :]
        tl.atomic_add(k_grad_ptrs, dk, mask=in_range[:, None] & d_mask[None, :])

    dq_ptrs = dQ_ptr + batch_i64 * stride_qb + hq_i64 * stride_qh + q_i64 * stride_ql + d_i64 * stride_qd
    tl.store(dq_ptrs, grad_q, mask=d_mask)


@torch.no_grad()
def _sliding_window_attention_triton_gqa_forward(
    Q2: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    cache_position: torch.Tensor,
    key_cache_position: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    sw_index: int = 0,
    search_power: float | None = None,
    inv_search_power_int: int | None = 2,
    block_k: int = DEFAULT_SW_BLOCK_K,
    block_m_dense: int = 16,
    num_warps: int = DEFAULT_SW_NUM_WARPS,
    num_stages: int = DEFAULT_SW_NUM_STAGES,
    use_bf16_pv: bool = True,
    validate_key_cache_position: bool = False,
):
    """Triton sliding-window attention (GQA) using cache-position-aware bounds.

    - `cache_position`: query cache positions. Shape: [L_Q] or [B, L_Q]
    - `key_cache_position`: key cache positions. Shape: [L_KV] or [B, L_KV]
      If None, assumes dense keys (key positions = indices) and takes the fast path.
    """
    if Q2.device.type != "cuda":
        raise ValueError("sliding_window_attention_triton_gqa requires CUDA tensors")

    B, H_q, L_Q, D = Q2.shape
    _, H_kv, L_KV, _ = K.shape

    if H_q % H_kv != 0:
        raise ValueError(f"H_q must be divisible by H_kv (got H_q={H_q}, H_kv={H_kv})")

    if cache_position.ndim == 1:
        if cache_position.numel() != L_Q:
            raise ValueError(f"cache_position length {cache_position.numel()} must match L_Q {L_Q}")
        q_cache_pos = cache_position[None, :].expand(B, L_Q)
    elif cache_position.ndim == 2:
        if cache_position.shape != (B, L_Q):
            raise ValueError(f"cache_position shape {tuple(cache_position.shape)} must be (B, L_Q)=({B}, {L_Q})")
        q_cache_pos = cache_position
    else:
        raise ValueError(f"cache_position must be 1D or 2D (got shape {tuple(cache_position.shape)})")

    key_pos_dense = key_cache_position is None
    if validate_key_cache_position and key_pos_dense:
        raise ValueError("validate_key_cache_position=True requires key_cache_position (got None)")

    if key_cache_position is None:
        k_cache_pos = None
    else:
        if key_cache_position.ndim == 1:
            if key_cache_position.numel() != L_KV:
                raise ValueError(f"key_cache_position length {key_cache_position.numel()} must match L_KV {L_KV}")
            k_cache_pos = key_cache_position[None, :].expand(B, L_KV)
        elif key_cache_position.ndim == 2:
            if key_cache_position.shape != (B, L_KV):
                raise ValueError(
                    f"key_cache_position shape {tuple(key_cache_position.shape)} must be (B, L_KV)=({B}, {L_KV})"
                )
            k_cache_pos = key_cache_position
        else:
            raise ValueError(f"key_cache_position must be 1D or 2D (got shape {tuple(key_cache_position.shape)})")

        if validate_key_cache_position and torch.any(k_cache_pos[:, 1:] < k_cache_pos[:, :-1]):
            raise ValueError("key_cache_position must be non-decreasing along the KV dimension")

    window = window_len_from_sw_index(int(sw_index), search_power=search_power, inv_search_power_int=inv_search_power_int)
    if window <= 0:
        sw_out = torch.zeros((B, H_q, L_Q, D), device=Q2.device, dtype=Q2.dtype)
        sw_lse = torch.full((B, H_q, L_Q), -float("inf"), device=Q2.device, dtype=torch.float32)
        return sw_out, sw_lse

    block_k = int(block_k)
    if block_k <= 0:
        raise ValueError(f"block_k must be > 0 (got {block_k})")

    if L_KV > MAX_SUPPORTED_SEQ_LEN:
        raise ValueError(
            f"Sequence length {L_KV:,} exceeds maximum supported {MAX_SUPPORTED_SEQ_LEN:,} tokens. "
            f"Increase MAX_LOG_KV (currently {MAX_LOG_KV}) to support longer sequences."
        )

    if D > 256:
        raise ValueError(f"sliding-window kernel expects D<=256 (got {D})")
    block_d = _next_pow2_int(D)

    # Compute bucketed values to limit kernel variants
    bucketed_log_kv = _bucket_log_kv(L_KV)

    stride_qb, stride_qh, stride_ql, stride_qd = (int(s) for s in Q2.stride())
    stride_kb, stride_kh, stride_kl, stride_kd = (int(s) for s in K.stride())
    stride_vb, stride_vh, stride_vl, stride_vd = (int(s) for s in V.stride())

    q_cache_pos_f = q_cache_pos.to(torch.int32).contiguous().view(B * L_Q)

    if key_pos_dense:
        k_cache_pos_f = torch.empty((1,), device=Q2.device, dtype=torch.int32)
    else:
        k_cache_pos_f = k_cache_pos.to(torch.int32).contiguous().view(B * L_KV)

    if attention_mask is not None:
        attn_mask = attention_mask[:, :L_KV].contiguous().to(torch.int8)
        has_mask = True
    else:
        attn_mask = torch.empty((1,), device=Q2.device, dtype=torch.int8)
        has_mask = False

    kv_for_q = make_kv_for_q_map(B=B, H_q=H_q, H_kv=H_kv, device=Q2.device)

    BH_q = B * H_q
    out = torch.empty((BH_q, L_Q, D), device=Q2.device, dtype=Q2.dtype)
    lse = torch.empty((BH_q, L_Q), device=Q2.device, dtype=torch.float32)

    if key_pos_dense:
        block_m = int(block_m_dense)
        if block_m <= 0:
            raise ValueError(f"block_m_dense must be > 0 (got {block_m})")
        sw_max_len = int(window) + (block_m - 1)
        sw_max_blocks = triton.cdiv(sw_max_len, block_k)

        grid = (BH_q, triton.cdiv(L_Q, block_m))
        _sliding_window_attn_forward_gqa_kernel_blockq_dense[grid](
            Q2,
            K,
            V,
            kv_for_q,
            q_cache_pos_f,
            attn_mask,
            out,
            lse,
            L_Q=L_Q,
            L_KV=L_KV,
            stride_qb=stride_qb,
            stride_qh=stride_qh,
            stride_ql=stride_ql,
            stride_qd=stride_qd,
            stride_kb=stride_kb,
            stride_kh=stride_kh,
            stride_kl=stride_kl,
            stride_kd=stride_kd,
            stride_vb=stride_vb,
            stride_vh=stride_vh,
            stride_vl=stride_vl,
            stride_vd=stride_vd,
            D=D,
            H_Q=H_q,
            H_KV=H_kv,
            WINDOW=int(window),
            BLOCK_M=block_m,
            BLOCK_K=block_k,
            BLOCK_D=block_d,
            sw_max_blocks=sw_max_blocks,  # runtime param for dynamic loop
            SM_SCALE=1.0 / math.sqrt(D),
            USE_BF16_PV=use_bf16_pv,
            HAS_ATTN_MASK=has_mask,
            num_warps=int(num_warps),
            num_stages=int(num_stages),
        )
    else:
        sw_max_blocks = triton.cdiv(int(window), block_k)
        grid = (BH_q * L_Q,)
        _sliding_window_attn_forward_gqa_kernel_token[grid](
            Q2,
            K,
            V,
            kv_for_q,
            q_cache_pos_f,
            k_cache_pos_f,
            attn_mask,
            out,
            lse,
            L_Q=L_Q,
            L_KV=L_KV,
            stride_qb=stride_qb,
            stride_qh=stride_qh,
            stride_ql=stride_ql,
            stride_qd=stride_qd,
            stride_kb=stride_kb,
            stride_kh=stride_kh,
            stride_kl=stride_kl,
            stride_kd=stride_kd,
            stride_vb=stride_vb,
            stride_vh=stride_vh,
            stride_vl=stride_vl,
            stride_vd=stride_vd,
            D=D,
            H_Q=H_q,
            H_KV=H_kv,
            WINDOW=int(window),
            BLOCK_K=block_k,
            BLOCK_D=block_d,
            sw_max_blocks=sw_max_blocks,  # runtime param for dynamic loop
            LOG_KV=bucketed_log_kv,  # Fixed to avoid recompilation
            KEY_POS_DENSE=False,
            SM_SCALE=1.0 / math.sqrt(D),
            USE_BF16_PV=use_bf16_pv,
            HAS_ATTN_MASK=has_mask,
            num_warps=int(num_warps),
            num_stages=int(num_stages),
        )

    return out.view(B, H_q, L_Q, D), lse.view(B, H_q, L_Q)


class _SlidingWindowAttentionTritonGQAFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q2,
        K,
        V,
        cache_position,
        key_cache_position,
        attention_mask,
        sw_index,
        search_power,
        inv_search_power_int,
        block_k,
        block_m_dense,
        num_warps,
        num_stages,
        use_bf16_pv,
        validate_key_cache_position,
    ):
        out, lse = _sliding_window_attention_triton_gqa_forward(
            Q2,
            K,
            V,
            cache_position=cache_position,
            key_cache_position=key_cache_position,
            attention_mask=attention_mask,
            sw_index=int(sw_index),
            search_power=search_power,
            inv_search_power_int=inv_search_power_int,
            block_k=int(block_k),
            block_m_dense=int(block_m_dense),
            num_warps=int(num_warps),
            num_stages=int(num_stages),
            use_bf16_pv=bool(use_bf16_pv),
            validate_key_cache_position=bool(validate_key_cache_position),
        )

        saved_mask = attention_mask if attention_mask is not None else torch.tensor([], device=Q2.device)
        saved_key_cache_pos = key_cache_position if key_cache_position is not None else torch.tensor([], device=Q2.device)
        ctx.save_for_backward(Q2, K, V, cache_position, saved_key_cache_pos, saved_mask, out, lse)
        ctx.sw_index = int(sw_index)
        ctx.search_power = search_power
        ctx.inv_search_power_int = inv_search_power_int
        ctx.block_k = int(block_k)
        ctx.num_warps = int(num_warps)
        ctx.num_stages = int(num_stages)
        return out, lse

    @staticmethod
    def backward(ctx, grad_out, grad_lse):
        Q2, K, V, cache_position, key_cache_position_saved, attention_mask_saved, out, lse = ctx.saved_tensors
        attention_mask = None if attention_mask_saved.numel() == 0 else attention_mask_saved
        key_cache_position = None if key_cache_position_saved.numel() == 0 else key_cache_position_saved

        if grad_out is None:
            grad_out = torch.zeros_like(out)
        if grad_lse is None:
            grad_lse = torch.zeros_like(lse)

        B, H_q, L_Q, D = Q2.shape
        _, H_kv, L_KV, _ = K.shape

        if H_q % H_kv != 0:
            raise ValueError(f"H_q must be divisible by H_kv (got H_q={H_q}, H_kv={H_kv})")
        if D > 256:
            raise ValueError(f"sliding-window backward expects D<=256 (got {D})")

        if cache_position.ndim == 1:
            if cache_position.numel() != L_Q:
                raise ValueError(f"cache_position length {cache_position.numel()} must match L_Q {L_Q}")
            q_cache_pos = cache_position[None, :].expand(B, L_Q)
        elif cache_position.ndim == 2:
            if cache_position.shape != (B, L_Q):
                raise ValueError(f"cache_position shape {tuple(cache_position.shape)} must be (B, L_Q)=({B}, {L_Q})")
            q_cache_pos = cache_position
        else:
            raise ValueError(f"cache_position must be 1D or 2D (got shape {tuple(cache_position.shape)})")

        key_pos_dense = key_cache_position is None
        if key_pos_dense:
            k_cache_pos = None
        else:
            if key_cache_position.ndim == 1:
                if key_cache_position.numel() != L_KV:
                    raise ValueError(f"key_cache_position length {key_cache_position.numel()} must match L_KV {L_KV}")
                k_cache_pos = key_cache_position[None, :].expand(B, L_KV)
            elif key_cache_position.ndim == 2:
                if key_cache_position.shape != (B, L_KV):
                    raise ValueError(
                        f"key_cache_position shape {tuple(key_cache_position.shape)} must be (B, L_KV)=({B}, {L_KV})"
                    )
                k_cache_pos = key_cache_position
            else:
                raise ValueError(f"key_cache_position must be 1D or 2D (got shape {tuple(key_cache_position.shape)})")

        window = window_len_from_sw_index(
            int(ctx.sw_index),
            search_power=ctx.search_power,
            inv_search_power_int=ctx.inv_search_power_int,
        )

        if window <= 0:
            return (
                torch.zeros_like(Q2),
                torch.zeros_like(K),
                torch.zeros_like(V),
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
                None,
                None,
            )

        block_k = int(ctx.block_k)
        block_d = _next_pow2_int(D)

        stride_qb, stride_qh, stride_ql, stride_qd = (int(s) for s in Q2.stride())
        stride_kb, stride_kh, stride_kl, stride_kd = (int(s) for s in K.stride())
        stride_vb, stride_vh, stride_vl, stride_vd = (int(s) for s in V.stride())

        q_cache_pos_f = q_cache_pos.to(torch.int32).contiguous().view(B * L_Q)
        if key_pos_dense:
            k_cache_pos_f = torch.empty((1,), device=Q2.device, dtype=torch.int32)
        else:
            k_cache_pos_f = k_cache_pos.to(torch.int32).contiguous().view(B * L_KV)

        if attention_mask is not None:
            attn_mask = attention_mask[:, :L_KV].contiguous().to(torch.int8)
            has_mask = True
        else:
            attn_mask = torch.empty((1,), device=Q2.device, dtype=torch.int8)
            has_mask = False

        kv_for_q = make_kv_for_q_map(B=B, H_q=H_q, H_kv=H_kv, device=Q2.device)

        dQ = torch.zeros_like(Q2, dtype=torch.float32)
        dK = torch.zeros_like(K, dtype=torch.float32)
        dV = torch.zeros_like(V, dtype=torch.float32)

        BH_q = B * H_q
        out_f = out.contiguous().view(BH_q, L_Q, D)
        lse_f = lse.contiguous().view(BH_q, L_Q)

        grad_out_f = grad_out.contiguous().view(BH_q, L_Q, D)
        grad_lse_f = grad_lse.contiguous().to(torch.float32).view(BH_q, L_Q)

        sw_max_blocks = triton.cdiv(int(window), block_k)
        bucketed_log_kv = _bucket_log_kv(L_KV)
        grid = (BH_q * L_Q,)

        _sliding_window_attn_backward_gqa_kernel_token[grid](
            Q2,
            K,
            V,
            kv_for_q,
            q_cache_pos_f,
            k_cache_pos_f,
            attn_mask,
            out_f,
            lse_f,
            grad_out_f,
            grad_lse_f,
            dQ,
            dK,
            dV,
            L_Q=L_Q,
            L_KV=L_KV,
            stride_qb=stride_qb,
            stride_qh=stride_qh,
            stride_ql=stride_ql,
            stride_qd=stride_qd,
            stride_kb=stride_kb,
            stride_kh=stride_kh,
            stride_kl=stride_kl,
            stride_kd=stride_kd,
            stride_vb=stride_vb,
            stride_vh=stride_vh,
            stride_vl=stride_vl,
            stride_vd=stride_vd,
            D=D,
            H_Q=H_q,
            H_KV=H_kv,
            WINDOW=int(window),
            BLOCK_K=block_k,
            BLOCK_D=block_d,
            sw_max_blocks=sw_max_blocks,  # runtime param for dynamic loop
            LOG_KV=bucketed_log_kv,  # Fixed to avoid recompilation
            KEY_POS_DENSE=key_pos_dense,
            SM_SCALE=1.0 / math.sqrt(D),
            HAS_ATTN_MASK=has_mask,
            num_warps=int(ctx.num_warps),
            num_stages=int(ctx.num_stages),
        )

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
            None,
            None,
        )


def sliding_window_attention_triton_gqa(
    Q2: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    cache_position: torch.Tensor,
    key_cache_position: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    sw_index: int = 0,
    search_power: float | None = None,
    inv_search_power_int: int | None = 2,
    block_k: int = DEFAULT_SW_BLOCK_K,
    block_m_dense: int = 16,
    num_warps: int = DEFAULT_SW_NUM_WARPS,
    num_stages: int = DEFAULT_SW_NUM_STAGES,
    use_bf16_pv: bool = True,
    validate_key_cache_position: bool = False,
):
    if not (torch.is_grad_enabled() and (Q2.requires_grad or K.requires_grad or V.requires_grad)):
        return _sliding_window_attention_triton_gqa_forward(
            Q2,
            K,
            V,
            cache_position=cache_position,
            key_cache_position=key_cache_position,
            attention_mask=attention_mask,
            sw_index=int(sw_index),
            search_power=search_power,
            inv_search_power_int=inv_search_power_int,
            block_k=int(block_k),
            block_m_dense=int(block_m_dense),
            num_warps=int(num_warps),
            num_stages=int(num_stages),
            use_bf16_pv=bool(use_bf16_pv),
            validate_key_cache_position=bool(validate_key_cache_position),
        )

    return _SlidingWindowAttentionTritonGQAFn.apply(
        Q2,
        K,
        V,
        cache_position,
        key_cache_position,
        attention_mask,
        int(sw_index),
        search_power,
        inv_search_power_int,
        int(block_k),
        int(block_m_dense),
        int(num_warps),
        int(num_stages),
        bool(use_bf16_pv),
        bool(validate_key_cache_position),
    )


def sliding_window_attention_triton(
    Q2: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    cache_position: torch.Tensor,
    key_cache_position: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    sw_index: int = 0,
    search_power: float | None = None,
    inv_search_power_int: int | None = 2,
    block_k: int = DEFAULT_SW_BLOCK_K,
    block_m_dense: int = 16,
    num_warps: int = DEFAULT_SW_NUM_WARPS,
    num_stages: int = DEFAULT_SW_NUM_STAGES,
    use_bf16_pv: bool = True,
    validate_key_cache_position: bool = False,
):
    """Non-GQA-friendly alias (works when H_q == H_kv)."""
    return sliding_window_attention_triton_gqa(
        Q2,
        K,
        V,
        cache_position=cache_position,
        key_cache_position=key_cache_position,
        attention_mask=attention_mask,
        sw_index=sw_index,
        search_power=search_power,
        inv_search_power_int=inv_search_power_int,
        block_k=block_k,
        block_m_dense=block_m_dense,
        num_warps=num_warps,
        num_stages=num_stages,
        use_bf16_pv=use_bf16_pv,
        validate_key_cache_position=validate_key_cache_position,
    )


__all__ = [
    "DEFAULT_SW_BLOCK_K",
    "DEFAULT_SW_NUM_STAGES",
    "DEFAULT_SW_NUM_WARPS",
    "make_kv_for_q_map",
    "sliding_window_attention_triton",
    "sliding_window_attention_triton_gqa",
]

