import math
import torch
import triton
import triton.language as tl

# Import common functions and kernels from the base span Triton implementation
from ._triton_impl import (
    _next_power_of_two,
    _assert_no_span_sw_overlap,
)


@triton.jit
def fused_span_forward_precomputed_sw_kernel(
    Q_ptr, K_ptr, V_ptr,
    qstart_ptr, qend_ptr, cache_position_ptr,
    attn_mask_ptr,
    sw_out_ptr, sw_lse_ptr,
    Out_ptr,
    B, H, L_Q, L_KV,
    sm_scale,
    K_VAL: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D_HEAD: tl.constexpr,
    SPAN_MAX_BLOCKS: tl.constexpr,
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

    seg_start, seg_end, seg_valid = span_start, span_end, span_valid
    seg_start = tl.maximum(seg_start, 0)
    seg_end = tl.minimum(seg_end, L_KV - 1)
    seg_valid = seg_valid & (seg_start <= seg_end)

    # Use 64-bit offsets to avoid overflow when H * L_KV * D exceeds int32
    k_head_offset = ((batch_idx * H + head_idx) * L_KV).to(tl.int64) * D_HEAD
    attn_base = batch_idx * L_KV

    m_i = -float('inf')
    l_i = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    scale = tl.full((1,), sm_scale, tl.float32)

    for block_idx in range(SPAN_MAX_BLOCKS):
        block_start = seg_start + block_idx * BLOCK_K
        k_pos = block_start + tl.arange(0, BLOCK_K)
        in_range = seg_valid & (k_pos >= seg_start) & (k_pos <= seg_end)
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

    sw_offset = ((batch_idx * H + head_idx) * L_Q + q_idx)
    sw_vec = tl.load(sw_out_ptr + sw_offset.to(tl.int64) * D_HEAD + d_range, mask=d_mask, other=0.0).to(tl.float32)
    sw_lse = tl.load(sw_lse_ptr + sw_offset, mask=True, other=float('-inf')).to(tl.float32)

    m_total = tl.maximum(m_i, sw_lse)
    span_scale = tl.exp(m_i - m_total)
    sw_scale = tl.exp(sw_lse - m_total)
    norm = l_i * span_scale + sw_scale

    acc = acc * span_scale + sw_scale * sw_vec
    acc = tl.where(norm > 0, acc / norm, 0.0)

    out_index = ((batch_idx * H + head_idx) * L_Q + q_idx) * K_VAL + span_index
    out_base = Out_ptr + out_index.to(tl.int64) * D_HEAD
    tl.store(out_base + d_range, acc, mask=d_mask)


@triton.jit
def fused_span_backward_precomputed_sw_kernel(
    Q_ptr, K_ptr, V_ptr, dOut_ptr,
    qstart_ptr, qend_ptr, cache_position_ptr,
    attn_mask_ptr,
    sw_out_ptr, sw_lse_ptr,
    dQ_ptr, dK_ptr, dV_ptr, dSw_out_ptr, dSw_lse_ptr,
    B, H, L_Q, L_KV,
    sm_scale,
    K_VAL: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D_HEAD: tl.constexpr,
    SPAN_MAX_BLOCKS: tl.constexpr,
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

    seg_start, seg_end, seg_valid = span_start, span_end, span_valid
    seg_start = tl.maximum(seg_start, 0)
    seg_end = tl.minimum(seg_end, L_KV - 1)
    seg_valid = seg_valid & (seg_start <= seg_end)

    k_head_offset = ((batch_idx * H + head_idx) * L_KV).to(tl.int64) * D_HEAD
    attn_base = batch_idx * L_KV

    m_i = -float('inf')
    l_i = 0.0
    scale = tl.full((1,), sm_scale, tl.float32)

    for block_idx in range(SPAN_MAX_BLOCKS):
        block_start = seg_start + block_idx * BLOCK_K
        k_pos = block_start + tl.arange(0, BLOCK_K)
        in_range = seg_valid & (k_pos >= seg_start) & (k_pos <= seg_end)
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

    sw_offset = ((batch_idx * H + head_idx) * L_Q + q_idx)
    sw_vec = tl.load(sw_out_ptr + sw_offset.to(tl.int64) * D_HEAD + d_range, mask=d_mask, other=0.0).to(tl.float32)
    sw_lse = tl.load(sw_lse_ptr + sw_offset, mask=True, other=float('-inf')).to(tl.float32)

    m_total = tl.maximum(m_i, sw_lse)
    span_scale = tl.exp(m_i - m_total)
    sw_scale = tl.exp(sw_lse - m_total)
    norm = l_i * span_scale + sw_scale
    inv_norm = tl.where(norm > 0, 1.0 / norm, 0.0)
    weight_sw = sw_scale * inv_norm

    dO_index = ((batch_idx * H + head_idx) * L_Q + q_idx) * K_VAL + span_index
    dO_base = dOut_ptr + dO_index.to(tl.int64) * D_HEAD
    dO = tl.load(dO_base + d_range, mask=d_mask, other=0.0).to(tl.float32)

    tl.atomic_add(dSw_out_ptr + sw_offset.to(tl.int64) * D_HEAD + d_range, dO * weight_sw, mask=d_mask)
    grad_w_sw = tl.sum(sw_vec * dO, axis=0)

    dot_total = grad_w_sw * weight_sw
    for block_idx in range(SPAN_MAX_BLOCKS):
        block_start = seg_start + block_idx * BLOCK_K
        k_pos = block_start + tl.arange(0, BLOCK_K)
        in_range = seg_valid & (k_pos >= seg_start) & (k_pos <= seg_end)
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
            weights = tl.exp(logits - m_i) * span_scale * inv_norm
            weights = tl.where(in_range, weights, 0.0)
            grad_w = tl.sum(v_block * dO[None, :], axis=1)
            dot_total += tl.sum(grad_w * weights, axis=0)

    grad_q = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for block_idx in range(SPAN_MAX_BLOCKS):
        block_start = seg_start + block_idx * BLOCK_K
        k_pos = block_start + tl.arange(0, BLOCK_K)
        in_range = seg_valid & (k_pos >= seg_start) & (k_pos <= seg_end)
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
            weights = tl.exp(logits - m_i) * span_scale * inv_norm
            weights = tl.where(in_range, weights, 0.0)
            grad_w = tl.sum(v_block * dO[None, :], axis=1)
            grad_s = (grad_w - dot_total) * weights * sm_scale

            grad_q = grad_q + tl.sum(grad_s[:, None] * k_block, axis=0)

            dk = grad_s[:, None] * q[None, :]
            tl.atomic_add(dK_ptr + k_offsets, dk, mask=in_range[:, None] & d_mask[None, :])

            dv = weights[:, None] * dO[None, :]
            tl.atomic_add(dV_ptr + v_offsets, dv, mask=in_range[:, None] & d_mask[None, :])

    dq_base = dQ_ptr + ((batch_idx * H + head_idx) * L_Q + q_idx) * D_HEAD
    tl.atomic_add(dq_base + d_range, grad_q, mask=d_mask)

    grad_sw_lse = (grad_w_sw - dot_total) * weight_sw
    tl.atomic_add(dSw_lse_ptr + sw_offset, grad_sw_lse)


def fused_span_triton_precomputed_sw(
    Q2, K, V, qstart, qend, cache_position,
    sw_out, sw_lse,
    attention_mask=None, sw_index=0, block_k: int = 64,
    span_len_factor: float = 2.0,
    *,
    span_power: float = 0.5,
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
    sw_out_c = sw_out.contiguous()
    sw_lse_c = sw_lse.contiguous()

    if attention_mask is not None:
        attn_mask = attention_mask[:, :L_KV].contiguous().to(torch.int8)
        has_mask = True
    else:
        attn_mask = torch.empty((1,), device=Q2.device, dtype=torch.int8)
        has_mask = False

    _assert_no_span_sw_overlap(qendc, cachec.view(1, 1, -1, 1), sw_index, L_KV)

    out = torch.empty((B, H, L_Q, num_spans, D), device=Q2.device, dtype=Q2.dtype)

    max_span_len = int(span_len_factor * math.ceil(float(L_KV) ** span_power_f) + 2)
    span_max_blocks = triton.cdiv(max_span_len, block_k)
    span_max_blocks = max(1, span_max_blocks)
    block_d = min(256, _next_power_of_two(D))

    grid = (B * H * L_Q * num_spans,)
    fused_span_forward_precomputed_sw_kernel[grid](
        Q2c, Kc, Vc,
        qstartc, qendc, cachec,
        attn_mask,
        sw_out_c, sw_lse_c,
        out,
        B, H, L_Q, L_KV,
        1.0 / math.sqrt(D),
        K_VAL=num_spans,
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        D_HEAD=D,
        SPAN_MAX_BLOCKS=span_max_blocks,
        HAS_ATTN_MASK=has_mask,
    )
    return out


class FusedSpanWithPrecomputedSW(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q2,
        K,
        V,
        qstart,
        qend,
        cache_position,
        sw_out,
        sw_lse,
        attention_mask=None,
        sw_index: int = 0,
        block_k: int = 64,
        span_len_factor: float = 2.0,
        span_power: float = 0.5,
    ):
        out = fused_span_triton_precomputed_sw(
            Q2, K, V, qstart, qend, cache_position,
            sw_out, sw_lse,
            attention_mask=attention_mask,
            sw_index=sw_index,
            block_k=block_k,
            span_len_factor=span_len_factor,
            span_power=span_power,
        )
        saved_mask = attention_mask if attention_mask is not None else torch.tensor([], device=Q2.device)
        ctx.save_for_backward(Q2, K, V, qstart, qend, cache_position.to(torch.int32), sw_out, sw_lse, saved_mask)
        ctx.sw_index = sw_index
        ctx.block_k = block_k
        ctx.span_len_factor = float(span_len_factor)
        ctx.span_power = float(span_power)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        Q2, K, V, qstart, qend, cache_position, sw_out, sw_lse, attention_mask_saved = ctx.saved_tensors
        attention_mask = None if attention_mask_saved.numel() == 0 else attention_mask_saved
        sw_index = ctx.sw_index
        block_k = ctx.block_k
        span_len_factor = float(getattr(ctx, "span_len_factor", 2.0))
        span_power = float(getattr(ctx, "span_power", 0.5))

        B, H, L_Q, D = Q2.shape
        L_KV = K.shape[2]
        num_spans = qstart.shape[-1]

        max_span_len = int(span_len_factor * math.ceil(float(L_KV) ** float(span_power)) + 2)
        span_max_blocks = triton.cdiv(max_span_len, block_k)
        span_max_blocks = max(1, span_max_blocks)
        block_d = min(256, _next_power_of_two(D))

        qstartc = qstart.contiguous()
        qendc = qend.contiguous()
        cachec = cache_position.to(torch.int32).contiguous()
        Q2c = Q2.contiguous()
        Kc = K.contiguous()
        Vc = V.contiguous()
        sw_out_c = sw_out.contiguous()
        sw_lse_c = sw_lse.contiguous()
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
        dSw_out = torch.zeros_like(sw_out_c, dtype=torch.float32)
        dSw_lse = torch.zeros_like(sw_lse_c, dtype=torch.float32)

        grid = (B * H * L_Q * num_spans,)
        fused_span_backward_precomputed_sw_kernel[grid](
            Q2c, Kc, Vc, grad_out_c,
            qstartc, qendc, cachec,
            attn_mask,
            sw_out_c, sw_lse_c,
            dQ, dK, dV, dSw_out, dSw_lse,
            B, H, L_Q, L_KV,
            1.0 / math.sqrt(D),
            K_VAL=num_spans,
            BLOCK_K=block_k,
            BLOCK_D=block_d,
            D_HEAD=D,
            SPAN_MAX_BLOCKS=span_max_blocks,
            HAS_ATTN_MASK=has_mask,
        )

        return (
            dQ.to(Q2.dtype),
            dK.to(K.dtype),
            dV.to(V.dtype),
            None, None, None,
            dSw_out.to(sw_out.dtype),
            dSw_lse.to(sw_lse.dtype),
            None, None, None,
            None,
            None,
        )


def fused_span_attention_with_precomputed_sw(
    Q2,
    K,
    V,
    qstart,
    qend,
    cache_position,
    sw_out,
    sw_lse,
    attention_mask=None,
    sw_index=0,
    block_k=64,
    span_len_factor: float = 2.0,
    *,
    span_power: float = 0.5,
):
    return FusedSpanWithPrecomputedSW.apply(
        Q2,
        K,
        V,
        qstart,
        qend,
        cache_position,
        sw_out,
        sw_lse,
        attention_mask,
        sw_index,
        block_k,
        span_len_factor,
        float(span_power),
    )
