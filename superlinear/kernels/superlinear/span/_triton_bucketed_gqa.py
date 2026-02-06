"""
Bucketed span attention kernels for GQA (Grouped Query Attention).

This module implements the optimized bucketed span attention from notebook 56.12,
which achieves significant speedups by grouping queries with similar span ranges
into buckets and processing them together.

Key optimizations:
1. Histogram-based bucketing: Groups (qstart, qend) pairs by their block range
2. Packed tiles: Processes multiple queries in the same bucket together
3. Separate SW + span attention: Computes sliding window and span attention
   independently, then merges with gated aggregation
"""

import math
import os
from typing import Dict

import torch
import triton
import triton.language as tl


_META_DEBUG_STATE = {
    "calls": 0,
    "last_print_bucketed_max_kblocks": None,
    "last_print_nb_window": None,
    "last_print_L_KV": None,
}


# =============================================================================
# Default kernel parameters (tuned for B200)
# =============================================================================

# Fixed LOGA for binary search over buckets - supports up to 20M buckets
# This avoids kernel recompilation when the number of active buckets changes
# log2(2^25) = 25, so 25 iterations handles up to ~33M buckets
FIXED_LOGA = 25
MAX_SUPPORTED_BUCKETS = (1 << FIXED_LOGA)  # 33,554,432 buckets

# Dynamic loops: We use range(span_blocks) instead of tl.static_range(MAX_KBLOCKS)
# This eliminates recompilation when span sizes change - the kernel compiles once
# and runs efficiently for any span size. Testing shows:
# - Static loop compiles in 8-160s depending on MAX_KBLOCKS value
# - Dynamic loop compiles in ~0.3s regardless of span size
# - No recompilation when span_blocks changes (0.0003s vs 8+s)


DEFAULT_BLOCK_K = 64  # Block size for K dimension in span attention
DEFAULT_BLOCK_P = 64  # Block size for processing query pairs
DEFAULT_NUM_WARPS_SPAN = 4
DEFAULT_NUM_STAGES_SPAN = 3

DEFAULT_BLOCK_Q_MERGE = 16  # Block size for merge kernel
DEFAULT_NUM_WARPS_MERGE = 4


# =============================================================================
# Helper functions
# =============================================================================

def _next_pow2_int(x: int) -> int:
    """Return the smallest power of 2 >= x."""
    return 1 << max(0, (int(x) - 1)).bit_length()


def max_span_blocks_upper_bound(
    *, L_kv: int, block_k: int, span_len_factor: float, span_power: float
) -> int:
    """
    Compute an upper bound on the number of K blocks a span can cover.
    
    The span length formula is: span_len = span_len_factor * (position ^ span_power)
    The maximum span occurs at position L_kv-1 (the last token).
    """
    max_position = max(1, L_kv - 1)
    max_span_len = span_len_factor * (max_position ** span_power)
    max_span_blocks = int(math.ceil(max_span_len / block_k)) + 2  # +2 for boundary blocks
    return max(1, max_span_blocks)


def make_kv_for_q_map(*, B: int, H_q: int, H_kv: int, device: torch.device) -> torch.Tensor:
    """
    Create a mapping tensor from query head indices to KV head indices for GQA.
    
    Returns:
        int32 tensor of shape [B*H_q] where entry i gives the corresponding KV head index.
    """
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


# =============================================================================
# Histogram/bucketing kernels
# Version 2: Fixed int64 pointer arithmetic for 10M+ context
# =============================================================================

@triton.jit
def _histogram_banded_kernel(
    qs_ptr,
    qe_ptr,
    counts_ptr,
    TOTAL,  # runtime: total number of query-span pairs
    P,  # runtime: pairs per batch-head
    nb,  # runtime: number of key blocks
    L_KV,  # runtime: key/value sequence length
    EB_BASE,  # runtime: base end-block for windowed bucketing
    NB_WINDOW,  # runtime: number of end-blocks in the window
    num_buckets_local,  # runtime: buckets per batch-head
    num_buckets_global,  # runtime: total buckets (for bounds checking)
    MAX_SPAN_BLOCKS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Count queries in each (end_block, delta) bucket."""
    pid = tl.program_id(0)
    # Use int64 for offs to avoid overflow when TOTAL > 2^31
    pid_i64 = tl.cast(pid, tl.int64)
    offs = pid_i64 * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = offs < TOTAL

    qs = tl.load(qs_ptr + offs, mask=m, other=0).to(tl.int32)
    qe = tl.load(qe_ptr + offs, mask=m, other=-1).to(tl.int32)

    valid = m & (qe >= 0) & (qs >= 0) & (qs <= qe) & (qs < L_KV) & (qe < L_KV)
    sb = tl.minimum(tl.maximum(qs // BLOCK_K, 0), nb - 1)
    eb = tl.minimum(tl.maximum(qe // BLOCK_K, 0), nb - 1)
    delta = eb - sb
    eb_rel = eb - EB_BASE
    valid = valid & (delta >= 0) & (delta < MAX_SPAN_BLOCKS) & (eb_rel >= 0) & (eb_rel < NB_WINDOW)

    bucket_local = eb_rel * MAX_SPAN_BLOCKS + delta
    # Cast all runtime params to int64 explicitly to ensure int64 arithmetic
    P_i64 = tl.cast(P, tl.int64)
    bh_i64 = offs // P_i64
    num_buckets_local_i64 = tl.cast(num_buckets_local, tl.int64)
    bucket_i64 = bh_i64 * num_buckets_local_i64 + bucket_local.to(tl.int64)
    # Bounds check: ensure bucket_i64 is within valid range
    num_buckets_global_i64 = tl.cast(num_buckets_global, tl.int64)
    valid = valid & (bucket_i64 >= 0) & (bucket_i64 < num_buckets_global_i64)
    tl.atomic_add(counts_ptr + tl.where(valid, bucket_i64, 0), tl.where(valid, 1, 0))


@triton.jit
def _scatter_pairs_kernel(
    qs_ptr,
    qe_ptr,
    write_ptr,
    out_pairs_ptr,
    TOTAL,  # runtime: total number of query-span pairs
    P,  # runtime: pairs per batch-head
    nb,  # runtime: number of key blocks
    L_KV,  # runtime: key/value sequence length
    EB_BASE,  # runtime: base end-block for windowed bucketing
    NB_WINDOW,  # runtime: number of end-blocks in the window
    num_buckets_local,  # runtime: buckets per batch-head
    num_buckets_global,  # runtime: total buckets (for bounds checking)
    MAX_SPAN_BLOCKS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Scatter query indices into their respective buckets."""
    pid = tl.program_id(0)
    # Use int64 for offs to avoid overflow when TOTAL > 2^31
    pid_i64 = tl.cast(pid, tl.int64)
    offs = pid_i64 * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = offs < TOTAL

    qs = tl.load(qs_ptr + offs, mask=m, other=0).to(tl.int32)
    qe = tl.load(qe_ptr + offs, mask=m, other=-1).to(tl.int32)

    valid = m & (qe >= 0) & (qs >= 0) & (qs <= qe) & (qs < L_KV) & (qe < L_KV)
    sb = tl.minimum(tl.maximum(qs // BLOCK_K, 0), nb - 1)
    eb = tl.minimum(tl.maximum(qe // BLOCK_K, 0), nb - 1)
    delta = eb - sb
    eb_rel = eb - EB_BASE
    valid = valid & (delta >= 0) & (delta < MAX_SPAN_BLOCKS) & (eb_rel >= 0) & (eb_rel < NB_WINDOW)

    bucket_local = eb_rel * MAX_SPAN_BLOCKS + delta
    # Cast all runtime params to int64 explicitly to ensure int64 arithmetic
    P_i64 = tl.cast(P, tl.int64)
    bh_i64 = offs // P_i64
    num_buckets_local_i64 = tl.cast(num_buckets_local, tl.int64)
    bucket_i64 = bh_i64 * num_buckets_local_i64 + bucket_local.to(tl.int64)
    # Bounds check: ensure bucket_i64 is within valid range
    num_buckets_global_i64 = tl.cast(num_buckets_global, tl.int64)
    valid = valid & (bucket_i64 >= 0) & (bucket_i64 < num_buckets_global_i64)
    bucket_i64 = tl.where(valid, bucket_i64, 0)
    w = tl.atomic_add(write_ptr + bucket_i64, tl.where(valid, 1, 0))
    tl.store(out_pairs_ptr + w, offs.to(tl.int32), mask=valid)


@torch.no_grad()
def build_meta_histogram_banded_direct_active(
    qstart: torch.Tensor,
    qend: torch.Tensor,
    *,
    L_KV: int,
    BLOCK_K: int,
    BLOCK_P: int,
    span_len_factor: float,
    span_power: float,
    max_span_blocks_cap: int | None = None,
) -> Dict:
    """
    Build histogram metadata for bucketed span attention.
    
    Groups (qstart, qend) pairs into buckets based on their block range (eb, delta)
    where eb is the end block and delta = eb - start_block.
    
    Returns a dict with:
        - active: indices of non-empty buckets
        - counts_active: count of queries in each active bucket
        - bucket_starts: starting offset in pairs array for each active bucket
        - pairs: flattened array of query indices grouped by bucket
        - MAX_SPAN_BLOCKS: upper bound on span length in blocks
    """
    B, H, L_Q, TOPK = qstart.shape
    BH = B * H
    P = L_Q * TOPK
    nb = (int(L_KV) + int(BLOCK_K) - 1) // int(BLOCK_K)

    # Window end-blocks to the (usually small) range actually present in qend.
    # This avoids allocating O(nb * MAX_SPAN_BLOCKS) buckets when qend only spans
    # a small band of blocks (e.g. chunked prefill).
    with torch.no_grad():
        # Clamp to valid token range to keep eb_min/max well-defined.
        qe_clamped = torch.clamp(qend, 0, int(L_KV) - 1)
        eb_min = int((qe_clamped.min() // int(BLOCK_K)).clamp(0, nb - 1).item())
        eb_max = int((qe_clamped.max() // int(BLOCK_K)).clamp(0, nb - 1).item())
    nb_window = int(eb_max - eb_min + 1)
    nb_window = max(1, min(nb_window, nb))

    # Compute actual max_span_blocks for validation and metadata
    max_span_blocks = max_span_blocks_upper_bound(
        L_kv=L_KV,
        block_k=BLOCK_K,
        span_len_factor=span_len_factor,
        span_power=span_power,
    )
    if max_span_blocks_cap is not None:
        max_span_blocks = min(max_span_blocks, int(max_span_blocks_cap))
    max_span_blocks = max(1, min(max_span_blocks, nb))

    # Bucket to next power of 2 for kernel caching (used in metadata/bucket indexing only)
    bucketed_max_kblocks = _next_pow2_int(max_span_blocks)

    # Use bucketed value for bucket indexing (power-of-2 for kernel caching).
    # Apply the end-block window to keep bucket arrays compact.
    num_buckets_local = nb_window * bucketed_max_kblocks
    num_buckets_global = BH * num_buckets_local
    TOTAL = BH * P

    # For very long contexts, `num_buckets_global` can exceed 2^31.
    # Allocating a dense `counts[num_buckets_global]` and running `cumsum` can fail
    # (or trigger illegal memory access) even if TOTAL is only a few million.
    # Switch to a sparse bucketing strategy in that regime.
    force_sparse = os.environ.get("SPAN_ATTENTION_FORCE_SPARSE_META", "1") not in ("0", "false", "False")
    use_sparse = force_sparse or (int(num_buckets_global) >= (2**31))

    if os.environ.get("SPAN_ATTENTION_DEBUG_META_PRINT", "0") in ("1", "true", "True"):
        # Throttle printing: by default print only when bucket size changes, and
        # at most `LIMIT` times.
        _META_DEBUG_STATE["calls"] += 1
        limit = int(os.environ.get("SPAN_ATTENTION_DEBUG_META_PRINT_LIMIT", "10"))
        every = int(os.environ.get("SPAN_ATTENTION_DEBUG_META_PRINT_EVERY", "0"))

        changed = (
            _META_DEBUG_STATE["last_print_bucketed_max_kblocks"] != int(bucketed_max_kblocks)
            or _META_DEBUG_STATE["last_print_nb_window"] != int(nb_window)
        )
        periodic = (every > 0) and ((_META_DEBUG_STATE["calls"] % every) == 0)

        should_print = (_META_DEBUG_STATE["calls"] <= limit) and (changed or periodic)
        if should_print:
            _META_DEBUG_STATE["last_print_bucketed_max_kblocks"] = int(bucketed_max_kblocks)
            _META_DEBUG_STATE["last_print_nb_window"] = int(nb_window)
            _META_DEBUG_STATE["last_print_L_KV"] = int(L_KV)
            print(
                "[span_attention] meta sizes:",
                {
                    "BH": int(BH),
                    "P": int(P),
                    "TOTAL": int(TOTAL),
                    "L_KV": int(L_KV),
                    "BLOCK_K": int(BLOCK_K),
                    "nb": int(nb),
                    "eb_min": int(eb_min),
                    "nb_window": int(nb_window),
                    "max_span_blocks": int(max_span_blocks),
                    "bucketed_max_kblocks": int(bucketed_max_kblocks),
                    "num_buckets_local": int(num_buckets_local),
                    "num_buckets_global": int(num_buckets_global),
                    "use_sparse": bool(use_sparse),
                },
            )

    qs_all = qstart.reshape(-1).contiguous().to(torch.int32)
    qe_all = qend.reshape(-1).contiguous().to(torch.int32)

    if use_sparse:
        device = qstart.device
        offs = torch.arange(TOTAL, device=device, dtype=torch.int64)
        bh = offs // int(P)

        qs = qs_all.to(torch.int64)
        qe = qe_all.to(torch.int64)
        valid = (qe >= 0) & (qs >= 0) & (qs <= qe) & (qs < int(L_KV)) & (qe < int(L_KV))

        sb = torch.clamp(qs // int(BLOCK_K), 0, int(nb) - 1)
        eb = torch.clamp(qe // int(BLOCK_K), 0, int(nb) - 1)
        delta = eb - sb
        eb_rel = eb - int(eb_min)
        valid = valid & (delta >= 0) & (delta < int(bucketed_max_kblocks)) & (eb_rel >= 0) & (eb_rel < int(nb_window))

        bucket_local = eb_rel * int(bucketed_max_kblocks) + delta
        bucket_ids = bh * int(num_buckets_local) + bucket_local

        bucket_ids_valid = bucket_ids[valid]
        pair_ids_valid = offs[valid].to(torch.int32)

        if bucket_ids_valid.numel() == 0:
            active = torch.empty((0,), device=device, dtype=torch.int64)
            counts_active = torch.empty((0,), device=device, dtype=torch.int32)
            bucket_starts = torch.empty((0,), device=device, dtype=torch.int32)
            pairs = torch.empty((0,), device=device, dtype=torch.int32)
        else:
            bucket_ids_sorted, perm = torch.sort(bucket_ids_valid)
            pairs = pair_ids_valid[perm].contiguous()

            active, counts_active = torch.unique_consecutive(bucket_ids_sorted, return_counts=True)
            active = active.to(torch.int64).contiguous()
            counts_active = counts_active.to(torch.int32).contiguous()

            bucket_starts = torch.empty((active.numel(),), device=device, dtype=torch.int32)
            bucket_starts[0] = 0
            if active.numel() > 1:
                bucket_starts[1:] = torch.cumsum(counts_active[:-1], dim=0)
    else:
        counts = torch.zeros((num_buckets_global,), device=qstart.device, dtype=torch.int32)
        grid = (triton.cdiv(TOTAL, 1024),)
        _histogram_banded_kernel[grid](
            qs_all,
            qe_all,
            counts,
            TOTAL=TOTAL,
            P=P,
            nb=nb,
            L_KV=L_KV,
            EB_BASE=eb_min,
            NB_WINDOW=nb_window,
            num_buckets_local=num_buckets_local,
            num_buckets_global=num_buckets_global,
            MAX_SPAN_BLOCKS=bucketed_max_kblocks,  # Bucketed (power of 2) for kernel caching
            BLOCK_K=BLOCK_K,
            BLOCK=1024,
            num_warps=4,
        )

        if os.environ.get("SPAN_ATTENTION_DEBUG_SYNC_META", "0") in ("1", "true", "True") and torch.cuda.is_available():
            torch.cuda.synchronize()

        offsets = torch.empty((num_buckets_global + 1,), device=qstart.device, dtype=torch.int32)
        offsets[0] = 0
        offsets[1:] = torch.cumsum(counts, dim=0)

        pairs = torch.empty((int(offsets[-1].item()),), device=qstart.device, dtype=torch.int32)
        write_ptr = offsets[:-1].clone()
        _scatter_pairs_kernel[grid](
            qs_all,
            qe_all,
            write_ptr,
            pairs,
            TOTAL=TOTAL,
            P=P,
            nb=nb,
            L_KV=L_KV,
            EB_BASE=eb_min,
            NB_WINDOW=nb_window,
            num_buckets_local=num_buckets_local,
            num_buckets_global=num_buckets_global,
            MAX_SPAN_BLOCKS=bucketed_max_kblocks,  # Bucketed (power of 2) for kernel caching
            BLOCK_K=BLOCK_K,
            BLOCK=1024,
            num_warps=4,
        )

        # Use int64 for bucket ids to support huge `num_buckets_global`.
        active = torch.nonzero(counts > 0).flatten().to(torch.int64)
        counts_active = counts[active]
        bucket_starts = offsets[active]

    return {
        "BH": BH,
        "P": P,
        "L_Q": L_Q,
        "TOPK": TOPK,
        "nb": nb,
        "EB_BASE": eb_min,
        "NB_WINDOW": nb_window,
        "L_KV": int(L_KV),
        "MAX_SPAN_BLOCKS": max_span_blocks,
        "BUCKETED_MAX_KBLOCKS": bucketed_max_kblocks,  # Power-of-2 bucket for kernel caching
        "num_buckets_local": num_buckets_local,
        "active": active,
        "counts_active": counts_active,
        "bucket_starts": bucket_starts,
        "pairs": pairs,
    }


# =============================================================================
# Bucketed span attention kernel (forward)
# =============================================================================

@triton.jit
def _span_attn_bucketed_banded_packed_tiles_gqa_independent_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    kv_for_q_ptr,
    qs_ptr,
    qe_ptr,
    attn_mask_ptr,
    active_buckets_ptr,
    bucket_starts_ptr,
    counts_active_ptr,
    tile_offsets_ptr,
    pairs_ptr,
    out_ptr,
    lse_ptr,
    A,  # runtime: number of active buckets (changes per call)
    P,  # runtime: total query-span pairs (changes per call)
    L_Q,  # runtime: query sequence length (changes per call)
    L_KV,  # runtime: key/value sequence length (changes per call)
    D: tl.constexpr,
    TOPK: tl.constexpr,
    H_q,  # runtime: number of query heads (could be constexpr but small)
    nb,  # runtime: number of buckets per head (changes per call)
    EB_BASE,  # runtime: base end-block for windowed bucketing
    NB_WINDOW,  # runtime: number of end-blocks in the window
    MAX_SPAN_BLOCKS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SM_SCALE: tl.constexpr,
    USE_BF16_PV: tl.constexpr,
    LOGA: tl.constexpr,  # keep constexpr - small range (1-20), used in static_range
    HAS_ATTN_MASK: tl.constexpr,
):
    """
    Bucketed span attention kernel for GQA.
    
    Each work item processes a tile of BLOCK_P queries from the same bucket
    (same end_block and delta). This enables better memory access patterns
    since queries in the same bucket access similar K/V ranges.
    """
    wid = tl.program_id(0)

    # Binary search to find which bucket this tile belongs to
    lo = tl.zeros((), dtype=tl.int32)
    hi = tl.full((), A, dtype=tl.int32)
    for _ in tl.static_range(0, LOGA):
        mid = (lo + hi) // 2
        mid_val = tl.load(tile_offsets_ptr + mid).to(tl.int32)
        go_right = mid_val <= wid
        lo = tl.where(go_right, mid, lo)
        hi = tl.where(go_right, hi, mid)
    idx = lo
    tile_base = tl.load(tile_offsets_ptr + idx).to(tl.int32)
    tile = wid - tile_base

    # Decode bucket info
    # NOTE: `active_buckets_ptr` is int64 to support huge `num_buckets_global`.
    gb_i64 = tl.load(active_buckets_ptr + idx).to(tl.int64)
    num_buckets_local_i64 = tl.cast(NB_WINDOW, tl.int64) * tl.cast(MAX_SPAN_BLOCKS, tl.int64)
    bh_q_i64 = gb_i64 // num_buckets_local_i64
    bucket_local_i64 = gb_i64 - bh_q_i64 * num_buckets_local_i64
    eb_rel_i64 = bucket_local_i64 // tl.cast(MAX_SPAN_BLOCKS, tl.int64)
    delta_i64 = bucket_local_i64 - eb_rel_i64 * tl.cast(MAX_SPAN_BLOCKS, tl.int64)
    bh_q = bh_q_i64.to(tl.int32)
    eb_rel = eb_rel_i64.to(tl.int32)
    delta = delta_i64.to(tl.int32)
    eb = eb_rel + EB_BASE
    sb = eb - delta
    span_blocks = delta + 1

    bucket_start = tl.load(bucket_starts_ptr + idx).to(tl.int32)
    bucket_count = tl.load(counts_active_ptr + idx).to(tl.int32)

    # Load query indices for this tile
    local_idx = tile * BLOCK_P + tl.arange(0, BLOCK_P)
    mask_pair = local_idx < bucket_count
    pair_offsets = bucket_start + local_idx
    gp = tl.load(pairs_ptr + pair_offsets, mask=mask_pair, other=0).to(tl.int32)
    p = gp - bh_q * P

    p_i64 = p.to(tl.int64)
    # Cast L_Q, L_KV, P to int64 to prevent overflow in pointer arithmetic at long sequences
    # Note: D is a tl.constexpr (Python int), so we can't use .to() on it
    L_Q_i64 = L_Q.to(tl.int64)
    L_KV_i64 = L_KV.to(tl.int64)
    P_i64 = P.to(tl.int64)

    # Load qstart/qend for each query
    qs = tl.load(qs_ptr + bh_q_i64 * P_i64 + p_i64, mask=mask_pair, other=0).to(tl.int32)
    qe = tl.load(qe_ptr + bh_q_i64 * P_i64 + p_i64, mask=mask_pair, other=-1).to(tl.int32)
    qs = tl.maximum(qs, 0)
    qe = tl.minimum(qe, L_KV - 1)

    # Load query vectors
    q = p // TOPK
    d = tl.arange(0, BLOCK_D)
    q_i64 = q.to(tl.int64)
    d_i64 = d.to(tl.int64)
    # Multiply int64 values first to prevent overflow, then multiply by constexpr D
    q_ptrs = Q_ptr + bh_q_i64 * L_Q_i64 * D + q_i64[:, None] * D + d_i64[None, :]
    q_tile = tl.load(q_ptrs, mask=mask_pair[:, None] & (d[None, :] < D), other=0.0)

    # Get KV head mapping
    kv_bh = tl.load(kv_for_q_ptr + bh_q_i64).to(tl.int32)
    kv_bh_i64 = kv_bh.to(tl.int64)

    # Initialize accumulators
    m = tl.where(mask_pair, -float("inf"), 0.0).to(tl.float32)
    l = tl.zeros((BLOCK_P,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_P, BLOCK_D), dtype=tl.float32)

    if HAS_ATTN_MASK:
        batch = bh_q // H_q
        batch_i64 = batch.to(tl.int64)
    else:
        batch_i64 = tl.zeros((), dtype=tl.int64)

    # Loop over K/V blocks - using dynamic range for fast compilation
    # (compiles once in ~0.3s, no recompilation when span_blocks changes)
    for kb in range(span_blocks):
        kv_block = sb + kb
        kv_start = kv_block * BLOCK_K
        k_idx = kv_start + tl.arange(0, BLOCK_K)
        kv_in_bounds = k_idx < L_KV

        k_i64 = k_idx.to(tl.int64)
        # Multiply int64 values first to prevent overflow, then multiply by constexpr D
        k_ptrs = K_ptr + kv_bh_i64 * L_KV_i64 * D + k_i64[None, :] * D + d_i64[:, None]
        k_tile = tl.load(
            k_ptrs,
            mask=kv_in_bounds[None, :] & (d[:, None] < D),
            other=0.0,
        )
        scores = tl.dot(q_tile, k_tile) * SM_SCALE

        # Apply span boundary masks
        is_first = kb == 0
        is_last = kb == (span_blocks - 1)
        mask_span = mask_pair[:, None] & kv_in_bounds[None, :]
        mask_span = tl.where(is_first, mask_span & (k_idx[None, :] >= qs[:, None]), mask_span)
        mask_span = tl.where(is_last, mask_span & (k_idx[None, :] <= qe[:, None]), mask_span)

        if HAS_ATTN_MASK:
            key_mask = tl.load(
                attn_mask_ptr + batch_i64 * L_KV_i64 + k_i64,
                mask=kv_in_bounds,
                other=0,
            ).to(tl.int8)
            mask_span = mask_span & (key_mask[None, :] != 0)

        scores = tl.where(mask_span, scores, -float("inf"))

        # Online softmax update
        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m, m_block)
        m_new_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp(m - m_new_safe)
        p_ = tl.exp(scores - m_new_safe[:, None])
        l_new = l * alpha + tl.sum(p_, axis=1)

        # Multiply int64 values first to prevent overflow, then multiply by constexpr D
        v_ptrs = V_ptr + kv_bh_i64 * L_KV_i64 * D + k_i64[:, None] * D + d_i64[None, :]
        v_tile = tl.load(
            v_ptrs,
            mask=kv_in_bounds[:, None] & (d[None, :] < D),
            other=0.0,
        )
        if USE_BF16_PV:
            acc = acc * alpha[:, None] + tl.dot(p_.to(tl.bfloat16), v_tile.to(tl.bfloat16))
        else:
            acc = acc * alpha[:, None] + tl.dot(p_, v_tile.to(tl.float32))

        m = m_new
        l = l_new

    # Compute final output and LSE
    lse = m + tl.log(l)
    tl.store(lse_ptr + bh_q_i64 * P_i64 + p_i64, lse, mask=mask_pair)
    inv_l = tl.where(l > 0, 1.0 / l, 0.0)
    out = acc * inv_l[:, None]
    out_ptrs = out_ptr + bh_q_i64 * P_i64 * D + p_i64[:, None] * D + d_i64[None, :]
    tl.store(out_ptrs, out, mask=mask_pair[:, None] & (d[None, :] < D))


# =============================================================================
# Backward kernel for bucketed span attention (from notebook 56.12)
# =============================================================================

@triton.jit
def _span_attn_bucketed_banded_backward_gqa_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    kv_for_q_ptr,
    qs_ptr,
    qe_ptr,
    attn_mask_ptr,
    out_ptr,
    lse_ptr,
    dOut_ptr,
    dLse_ptr,
    dQ_ptr,
    dK_ptr,
    dV_ptr,
    P,  # runtime: total query-span pairs
    L_Q,  # runtime: query sequence length
    L_KV,  # runtime: key/value sequence length
    D: tl.constexpr,
    TOPK: tl.constexpr,
    H_q,  # runtime: number of query heads
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SM_SCALE: tl.constexpr,
    HAS_ATTN_MASK: tl.constexpr,
):
    """Backward kernel for bucketed span attention - processes one (query, topk) pair per thread block."""
    pid = tl.program_id(0)

    bh_q = pid // P
    p = pid - bh_q * P
    q = p // TOPK

    bh_q_i64 = bh_q.to(tl.int64)
    p_i64 = p.to(tl.int64)
    q_i64 = q.to(tl.int64)
    # Cast L_Q, L_KV, P to int64 to prevent overflow in pointer arithmetic at long sequences
    # Note: D is a tl.constexpr (Python int), so we can't use .to() on it
    L_Q_i64 = L_Q.to(tl.int64)
    L_KV_i64 = L_KV.to(tl.int64)
    P_i64 = P.to(tl.int64)

    d = tl.arange(0, BLOCK_D)
    d_mask = d < D
    d_i64 = d.to(tl.int64)

    # Load query vector - multiply int64 values first to prevent overflow
    q_ptrs = Q_ptr + bh_q_i64 * L_Q_i64 * D + q_i64 * D + d_i64
    q_vec = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    # Load forward output and gradients
    out_ptrs = out_ptr + bh_q_i64 * P_i64 * D + p_i64 * D + d_i64
    out_vec = tl.load(out_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    dOut_ptrs = dOut_ptr + bh_q_i64 * P_i64 * D + p_i64 * D + d_i64
    dO = tl.load(dOut_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    lse = tl.load(lse_ptr + bh_q_i64 * P_i64 + p_i64, mask=True, other=-float("inf")).to(tl.float32)
    dLse = tl.load(dLse_ptr + bh_q_i64 * P_i64 + p_i64, mask=True, other=0.0).to(tl.float32)

    # Compute dot product of output and grad_output
    dot = tl.sum(out_vec * dO, axis=0)

    # Load span boundaries
    qs = tl.load(qs_ptr + bh_q_i64 * P_i64 + p_i64, mask=True, other=0).to(tl.int32)
    qe = tl.load(qe_ptr + bh_q_i64 * P_i64 + p_i64, mask=True, other=-1).to(tl.int32)

    valid = (qe >= 0) & (qs >= 0) & (qs <= qe)
    qs = tl.maximum(qs, 0)
    qe = tl.minimum(qe, L_KV - 1)
    valid = valid & (qs <= qe)

    has_any = valid & (lse != -float("inf"))
    lse_safe = tl.where(has_any, lse, 0.0)
    dot_safe = tl.where(has_any, dot, 0.0)
    dLse_safe = tl.where(has_any, dLse, 0.0)

    sb = qs // BLOCK_K
    eb = qe // BLOCK_K
    span_blocks = eb - sb + 1

    kv_bh = tl.load(kv_for_q_ptr + bh_q_i64, mask=True, other=0).to(tl.int32)
    kv_bh_i64 = kv_bh.to(tl.int64)

    if HAS_ATTN_MASK:
        batch = bh_q // H_q
        batch_i64 = batch.to(tl.int64)
    else:
        batch_i64 = tl.zeros((), dtype=tl.int64)

    grad_q = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # Dynamic range loop - compiles once, no recompilation when span_blocks changes
    for kb in range(span_blocks):
        kv_block = sb + kb
        kv_start = kv_block * BLOCK_K
        k_idx = kv_start + tl.arange(0, BLOCK_K)
        kv_in_bounds = k_idx < L_KV

        in_range = has_any & kv_in_bounds & (k_idx >= qs) & (k_idx <= qe)

        k_safe = tl.where(in_range, k_idx, 0).to(tl.int32)
        k_i64 = k_safe.to(tl.int64)

        if HAS_ATTN_MASK:
            key_mask = tl.load(attn_mask_ptr + batch_i64 * L_KV_i64 + k_i64, mask=kv_in_bounds, other=0).to(tl.int8)
            in_range = in_range & (key_mask != 0)
            k_safe = tl.where(in_range, k_idx, 0).to(tl.int32)
            k_i64 = k_safe.to(tl.int64)

        # Multiply int64 values first to prevent overflow, then multiply by constexpr D
        k_offsets = kv_bh_i64 * L_KV_i64 * D + k_i64[:, None] * D + d_i64[None, :]
        k_tile = tl.load(
            K_ptr + k_offsets,
            mask=in_range[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        v_offsets = kv_bh_i64 * L_KV_i64 * D + k_i64[:, None] * D + d_i64[None, :]
        v_tile = tl.load(
            V_ptr + v_offsets,
            mask=in_range[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # Compute attention scores
        scores = tl.sum(k_tile * q_vec[None, :], axis=1) * SM_SCALE
        scores = tl.where(in_range, scores, -float("inf"))

        # Compute attention weights
        w = tl.exp(scores - lse_safe)
        w = tl.where(in_range, w, 0.0)

        # Gradient for V: dV += w * dO
        dv = w[:, None] * dO[None, :]
        tl.atomic_add(dV_ptr + v_offsets, dv, mask=in_range[:, None] & d_mask[None, :])

        # Gradient for attention weights
        grad_w = tl.sum(v_tile * dO[None, :], axis=1)
        grad_logits = w * (grad_w - dot_safe + dLse_safe)
        grad_s = grad_logits * SM_SCALE

        # Gradient for Q: dQ += sum_k(grad_s * K)
        grad_q += tl.sum(grad_s[:, None] * k_tile, axis=0)

        # Gradient for K: dK += grad_s * Q
        dk = grad_s[:, None] * q_vec[None, :]
        tl.atomic_add(dK_ptr + k_offsets, dk, mask=in_range[:, None] & d_mask[None, :])

    # Write gradient for Q - multiply int64 values first to prevent overflow
    dq_ptrs = dQ_ptr + bh_q_i64 * L_Q_i64 * D + q_i64 * D + d_i64
    tl.atomic_add(dq_ptrs, grad_q, mask=d_mask)


# =============================================================================
# Forward-only function (used when gradients are not needed)
# =============================================================================

@torch.no_grad()
def _span_attention_bucketed_packed_tiles_gqa_independent_forward(
    Q2: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    qstart: torch.Tensor,
    qend: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None,
    block_k: int = DEFAULT_BLOCK_K,
    block_p: int = DEFAULT_BLOCK_P,
    span_len_factor: float,
    span_power: float,
    max_span_blocks_cap: int | None = None,
    num_warps: int = DEFAULT_NUM_WARPS_SPAN,
    num_stages: int = DEFAULT_NUM_STAGES_SPAN,
    use_bf16_pv: bool = True,
):
    """
    Bucketed span attention for GQA (forward only).
    
    This is the optimized span attention kernel that groups queries by their
    span bucket for better memory access patterns.
    
    Args:
        Q2: Query tensor [B, H_q, L_Q, D]
        K: Key tensor [B, H_kv, L_KV, D]
        V: Value tensor [B, H_kv, L_KV, D]
        qstart: Start positions [B, H_q, L_Q, TOPK]
        qend: End positions [B, H_q, L_Q, TOPK]
        attention_mask: Optional boolean mask [B, L_KV]
        block_k: Block size for K dimension
        block_p: Block size for processing queries
        span_len_factor: Factor for span length computation
        span_power: Power for span length computation
        max_span_blocks_cap: Optional cap on max span blocks
        num_warps: Number of warps per block
        num_stages: Number of pipeline stages
        use_bf16_pv: Use bf16 for P@V multiplication
        
    Returns:
        out: Output tensor [B, H_q, L_Q, TOPK, D]
        lse: Log-sum-exp tensor [B, H_q, L_Q, TOPK]
    """
    B, H_q, L_Q, D = Q2.shape
    _, H_kv, L_KV, _ = K.shape
    if H_q % H_kv != 0:
        raise ValueError(f"H_q must be divisible by H_kv (got H_q={H_q}, H_kv={H_kv})")

    if D > 256:
        raise ValueError(f"bucketed span kernel expects D<=256 (got {D})")
    block_d = _next_pow2_int(int(D))

    # No span length limit check needed - dynamic loops support any span size

    TOPK = int(qstart.shape[-1])
    BH_q = B * H_q
    P = L_Q * TOPK
    nb = (int(L_KV) + int(block_k) - 1) // int(block_k)

    Q2c = Q2.contiguous()
    Kc = K.contiguous()
    Vc = V.contiguous()

    out = torch.zeros((BH_q, P, D), device=Q2.device, dtype=Q2.dtype)
    lse = torch.full((BH_q, P), -float("inf"), device=Q2.device, dtype=torch.float32)

    meta = build_meta_histogram_banded_direct_active(
        qstart,
        qend,
        L_KV=L_KV,
        BLOCK_K=block_k,
        BLOCK_P=block_p,
        span_len_factor=span_len_factor,
        span_power=span_power,
        max_span_blocks_cap=max_span_blocks_cap,
    )
    A = int(meta["active"].numel())
    if A == 0:
        return out.view(B, H_q, L_Q, TOPK, D), lse.view(B, H_q, L_Q, TOPK)

    if A > MAX_SUPPORTED_BUCKETS:
        raise ValueError(
            f"Number of active buckets {A:,} exceeds maximum supported {MAX_SUPPORTED_BUCKETS:,}. "
            f"Increase FIXED_LOGA (currently {FIXED_LOGA}) to support more buckets."
        )

    max_kblocks = int(meta["MAX_SPAN_BLOCKS"])
    max_kblocks = max(1, min(max_kblocks, nb))
    bucketed_max_kblocks = int(meta["BUCKETED_MAX_KBLOCKS"])

    tiles_per = torch.div(
        meta["counts_active"] + (block_p - 1), block_p, rounding_mode="floor"
    ).to(torch.int32)
    tile_offsets = torch.empty((A + 1,), device=tiles_per.device, dtype=torch.int32)
    tile_offsets[0] = 0
    tile_offsets[1:] = torch.cumsum(tiles_per, dim=0)
    total_tiles = int(tile_offsets[-1].item())
    if total_tiles == 0:
        return out.view(B, H_q, L_Q, TOPK, D), lse.view(B, H_q, L_Q, TOPK)

    # Use fixed LOGA to avoid kernel recompilation when A changes
    qs_all = qstart.reshape(-1).contiguous().to(torch.int32)
    qe_all = qend.reshape(-1).contiguous().to(torch.int32)

    if attention_mask is not None:
        attn_mask = attention_mask[:, :L_KV].contiguous().to(torch.int8)
        has_mask = True
    else:
        attn_mask = torch.empty((1,), device=Q2.device, dtype=torch.int8)
        has_mask = False

    kv_for_q = make_kv_for_q_map(B=B, H_q=H_q, H_kv=H_kv, device=Q2.device)

    Qf = Q2c.view(BH_q, L_Q, D)
    Kf = Kc.view(B * H_kv, L_KV, D)
    Vf = Vc.view(B * H_kv, L_KV, D)

    _span_attn_bucketed_banded_packed_tiles_gqa_independent_kernel[(total_tiles,)](
        Qf,
        Kf,
        Vf,
        kv_for_q,
        qs_all,
        qe_all,
        attn_mask,
        meta["active"],
        meta["bucket_starts"],
        meta["counts_active"],
        tile_offsets,
        meta["pairs"],
        out,
        lse,
        A=A,
        P=P,
        L_Q=L_Q,
        L_KV=L_KV,
        D=D,
        TOPK=TOPK,
        H_q=H_q,
        nb=nb,
        EB_BASE=int(meta["EB_BASE"]),
        NB_WINDOW=int(meta["NB_WINDOW"]),
        MAX_SPAN_BLOCKS=bucketed_max_kblocks,  # Power-of-2 bucket for kernel caching
        BLOCK_K=block_k,
        BLOCK_P=block_p,
        BLOCK_D=block_d,
        SM_SCALE=1.0 / math.sqrt(D),
        USE_BF16_PV=use_bf16_pv,
        LOGA=FIXED_LOGA,
        HAS_ATTN_MASK=has_mask,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return out.view(B, H_q, L_Q, TOPK, D), lse.view(B, H_q, L_Q, TOPK)


# =============================================================================
# Merge kernel (combines SW output with span output using gated aggregation)
# =============================================================================

@triton.jit
def _merge_gate_sw_span_topk_qmajor_kernel(
    sw_O_ptr,
    sw_lse_ptr,
    span_O_ptr,
    span_lse_ptr,
    span_values_ptr,
    qend_ptr,
    out_ptr,
    L_Q,  # runtime: query sequence length (changes per call)
    D: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
    KMAX: tl.constexpr,
):
    """
    Merge sliding window output with span outputs using gated aggregation.
    
    For each query position:
    1. Compute softmax over span_values to get gate weights
    2. For each top-k span:
       - Merge SW and span outputs using LSE-weighted combination
       - Weight the merged result by the gate
    3. Sum over top-k
    """
    bh = tl.program_id(0)
    pid_q = tl.program_id(1)
    q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    d = tl.arange(0, BLOCK_D)
    mask_q = q < L_Q
    mask_d = d < D

    bh_i64 = bh.to(tl.int64)
    q_i64 = q.to(tl.int64)
    d_i64 = d.to(tl.int64)

    # Load SW output and LSE
    sw_lse = tl.load(
        sw_lse_ptr + bh_i64 * L_Q + q_i64, mask=mask_q, other=-float("inf")
    ).to(tl.float32)
    sw_O = tl.load(
        sw_O_ptr + bh_i64 * (L_Q * D) + q_i64[:, None] * D + d_i64[None, :],
        mask=mask_q[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.float32)

    kk = tl.arange(0, KMAX)
    kk_i64 = kk.to(tl.int64)
    mask_k = kk < TOPK

    # Load qend and values for gating
    qend_k = tl.load(
        qend_ptr + bh_i64 * (L_Q * TOPK) + q_i64[:, None] * TOPK + kk_i64[None, :],
        mask=mask_q[:, None] & mask_k[None, :],
        other=-1,
    ).to(tl.int32)
    vals_k = tl.load(
        span_values_ptr + bh_i64 * (L_Q * TOPK) + q_i64[:, None] * TOPK + kk_i64[None, :],
        mask=mask_q[:, None] & mask_k[None, :],
        other=-1e9,
    ).to(tl.float32)

    # Mask invalid spans
    vals_k = tl.where(mask_k[None, :], vals_k, -float("inf"))
    vals_k = tl.where(mask_k[None, :] & (qend_k < 0), -1e9, vals_k)

    # Compute softmax over span values (gate weights)
    m = tl.max(vals_k, axis=1)
    exps_k = tl.exp(vals_k - m[:, None])
    den = tl.sum(exps_k, axis=1)

    # Accumulate gated outputs
    acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)
    for ki in tl.static_range(0, TOPK):
        qend_i = tl.load(
            qend_ptr + bh_i64 * (L_Q * TOPK) + q_i64 * TOPK + ki,
            mask=mask_q,
            other=-1,
        ).to(tl.int32)
        val_i = tl.load(
            span_values_ptr + bh_i64 * (L_Q * TOPK) + q_i64 * TOPK + ki,
            mask=mask_q,
            other=-1e9,
        ).to(tl.float32)
        val_i = tl.where(qend_i >= 0, val_i, -1e9)

        exp_i = tl.exp(val_i - m)
        score_i = tl.where(den > 0, exp_i / den, 1.0 / TOPK)

        # Load span output and LSE for this top-k
        p = q * TOPK + ki
        p_i64 = p.to(tl.int64)
        span_lse = tl.load(
            span_lse_ptr + bh_i64 * (L_Q * TOPK) + p_i64,
            mask=mask_q,
            other=-float("inf"),
        ).to(tl.float32)
        span_O = tl.load(
            span_O_ptr + bh_i64 * (L_Q * TOPK * D) + p_i64[:, None] * D + d_i64[None, :],
            mask=mask_q[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)

        # Merge SW and span outputs using LSE-weighted combination
        m2 = tl.maximum(sw_lse, span_lse)
        both_none = m2 == -float("inf")
        e_sw = tl.where(both_none, 1.0, tl.exp(sw_lse - m2))
        e_sp = tl.where(both_none, 0.0, tl.exp(span_lse - m2))
        denom2 = e_sw + e_sp
        w_sw = tl.where(denom2 > 0, e_sw / denom2, 1.0)
        w_sp = tl.where(denom2 > 0, e_sp / denom2, 0.0)

        O_k = sw_O * w_sw[:, None] + span_O * w_sp[:, None]
        acc += O_k * score_i[:, None]

    tl.store(
        out_ptr + bh_i64 * (L_Q * D) + q_i64[:, None] * D + d_i64[None, :],
        acc,
        mask=mask_q[:, None] & mask_d[None, :],
    )


# =============================================================================
# PyTorch-based merge (for backward support)
# =============================================================================

def _merge_gate_torch_qmajor(
    sw_O: torch.Tensor,
    sw_lse: torch.Tensor,
    span_out: torch.Tensor,
    span_lse: torch.Tensor,
    span_values: torch.Tensor,
    qend: torch.Tensor,
) -> torch.Tensor:
    """
    PyTorch implementation of merge_gate_qmajor.
    
    This is used when gradients are needed, as it supports automatic differentiation.
    """
    B, H, L_Q, D = sw_O.shape
    TOPK = int(span_values.shape[-1])

    # LSE-weighted combination of SW and span outputs
    lse_total = torch.logaddexp(sw_lse[..., None], span_lse)
    w_sw = torch.exp(sw_lse[..., None] - lse_total).to(sw_O.dtype)
    w_sp = torch.exp(span_lse - lse_total).to(sw_O.dtype)
    O_k = w_sw[..., None] * sw_O[..., None, :] + w_sp[..., None] * span_out

    # Gate by span search values
    span_values_use = torch.where(qend < 0, span_values.new_full((), -1e9), span_values)
    span_scores = torch.softmax(span_values_use, dim=-1)
    span_scores = torch.nan_to_num(span_scores, nan=1.0 / TOPK, posinf=0.0, neginf=0.0).to(O_k.dtype)
    return (O_k.float() * span_scores.float()[..., None]).sum(dim=3).to(O_k.dtype)


# =============================================================================
# Triton-based merge (forward only, faster)
# =============================================================================

@torch.no_grad()
def _merge_gate_triton_qmajor(
    sw_O: torch.Tensor,
    sw_lse: torch.Tensor,
    span_out: torch.Tensor,
    span_lse: torch.Tensor,
    span_values: torch.Tensor,
    qend: torch.Tensor,
    *,
    block_q: int = DEFAULT_BLOCK_Q_MERGE,
    num_warps: int = DEFAULT_NUM_WARPS_MERGE,
) -> torch.Tensor:
    """
    Merge sliding window output with span outputs using gated aggregation.
    
    Args:
        sw_O: SW attention output [B, H, L_Q, D]
        sw_lse: SW log-sum-exp [B, H, L_Q]
        span_out: Span attention outputs [B, H, L_Q, TOPK, D]
        span_lse: Span log-sum-exp [B, H, L_Q, TOPK]
        span_values: Gate values (from search) [B, H, L_Q, TOPK]
        qend: End positions (for validity check) [B, H, L_Q, TOPK]
        block_q: Block size for Q dimension
        num_warps: Number of warps
        
    Returns:
        Merged output [B, H, L_Q, D]
    """
    B, H, L_Q, D = sw_O.shape
    BH = B * H
    TOPK = int(span_values.shape[-1])

    block_q = int(block_q)
    if D > 256:
        raise ValueError(f"merge kernel expects D<=256 (got {D})")
    BLOCK_D = _next_pow2_int(D)
    KMAX = _next_pow2_int(TOPK)

    sw_O_bh = sw_O.contiguous().view(BH, L_Q, D)
    sw_lse_bh = sw_lse.contiguous().view(BH, L_Q).to(torch.float32)
    span_O_bh = span_out.contiguous().view(BH, L_Q * TOPK, D)
    span_lse_bh = span_lse.contiguous().view(BH, L_Q * TOPK).to(torch.float32)
    span_values_bh = span_values.contiguous().view(BH, L_Q, TOPK).to(torch.float32)
    qend_bh = qend.contiguous().view(BH, L_Q, TOPK).to(torch.int32)

    out = torch.empty((BH, L_Q, D), device=sw_O.device, dtype=sw_O.dtype)
    grid = (BH, triton.cdiv(L_Q, block_q))
    _merge_gate_sw_span_topk_qmajor_kernel[grid](
        sw_O_bh,
        sw_lse_bh,
        span_O_bh,
        span_lse_bh,
        span_values_bh,
        qend_bh,
        out,
        L_Q=L_Q,
        D=D,
        TOPK=TOPK,
        BLOCK_Q=block_q,
        BLOCK_D=BLOCK_D,
        KMAX=KMAX,
        num_warps=num_warps,
    )

    return out.view(B, H, L_Q, D)


# =============================================================================
# Public API: Merge with automatic backward support
# =============================================================================

def merge_gate_qmajor(
    sw_O: torch.Tensor,
    sw_lse: torch.Tensor,
    span_out: torch.Tensor,
    span_lse: torch.Tensor,
    span_values: torch.Tensor,
    qend: torch.Tensor,
    *,
    block_q: int = DEFAULT_BLOCK_Q_MERGE,
    num_warps: int = DEFAULT_NUM_WARPS_MERGE,
) -> torch.Tensor:
    """
    Merge sliding window output with span outputs using gated aggregation.
    
    Automatically dispatches to PyTorch implementation when gradients are needed
    (for backward pass support), or to optimized Triton kernel otherwise.
    
    Args:
        sw_O: SW attention output [B, H, L_Q, D]
        sw_lse: SW log-sum-exp [B, H, L_Q]
        span_out: Span attention outputs [B, H, L_Q, TOPK, D]
        span_lse: Span log-sum-exp [B, H, L_Q, TOPK]
        span_values: Gate values (from search) [B, H, L_Q, TOPK]
        qend: End positions (for validity check) [B, H, L_Q, TOPK]
        block_q: Block size for Q dimension (Triton kernel only)
        num_warps: Number of warps (Triton kernel only)
        
    Returns:
        Merged output [B, H, L_Q, D]
    """
    # Use PyTorch implementation when gradients are needed
    if torch.is_grad_enabled() and (sw_O.requires_grad or span_out.requires_grad):
        return _merge_gate_torch_qmajor(sw_O, sw_lse, span_out, span_lse, span_values, qend)
    
    # Use optimized Triton kernel for inference
    return _merge_gate_triton_qmajor(
        sw_O, sw_lse, span_out, span_lse, span_values, qend,
        block_q=block_q, num_warps=num_warps,
    )


# =============================================================================
# Autograd Function for bucketed span attention with backward pass
# =============================================================================

class _SpanAttentionBucketedPackedTilesGQAFn(torch.autograd.Function):
    """Autograd wrapper for bucketed span attention with backward support."""
    
    @staticmethod
    def forward(
        ctx,
        Q2,
        K,
        V,
        qstart,
        qend,
        attention_mask,
        block_k,
        block_p,
        span_len_factor,
        span_power,
        max_span_blocks_cap,
        num_warps,
        num_stages,
        use_bf16_pv,
    ):
        out, lse = _span_attention_bucketed_packed_tiles_gqa_independent_forward(
            Q2,
            K,
            V,
            qstart,
            qend,
            attention_mask=attention_mask,
            block_k=int(block_k),
            block_p=int(block_p),
            span_len_factor=float(span_len_factor),
            span_power=float(span_power),
            max_span_blocks_cap=None if max_span_blocks_cap is None else int(max_span_blocks_cap),
            num_warps=int(num_warps),
            num_stages=int(num_stages),
            use_bf16_pv=bool(use_bf16_pv),
        )
        saved_mask = attention_mask if attention_mask is not None else torch.tensor([], device=Q2.device)
        ctx.save_for_backward(Q2, K, V, qstart, qend, saved_mask, out, lse)
        ctx.block_k = int(block_k)
        ctx.span_len_factor = float(span_len_factor)
        ctx.span_power = float(span_power)
        ctx.max_span_blocks_cap = None if max_span_blocks_cap is None else int(max_span_blocks_cap)
        return out, lse

    @staticmethod
    def backward(ctx, grad_out, grad_lse):
        Q2, K, V, qstart, qend, attention_mask_saved, out, lse = ctx.saved_tensors
        attention_mask = None if attention_mask_saved.numel() == 0 else attention_mask_saved

        if grad_out is None:
            grad_out = torch.zeros_like(out)
        if grad_lse is None:
            grad_lse = torch.zeros_like(lse)

        B, H_q, L_Q, D = Q2.shape
        _, H_kv, L_KV, _ = K.shape
        TOPK = int(qstart.shape[-1])
        if H_q % H_kv != 0:
            raise ValueError(f"H_q must be divisible by H_kv (got H_q={H_q}, H_kv={H_kv})")

        block_k = int(ctx.block_k)
        span_len_factor = float(ctx.span_len_factor)
        span_power = float(ctx.span_power)
        max_span_blocks_cap = ctx.max_span_blocks_cap

        if D > 256:
            raise ValueError(f"span backward expects D<=256 (got {D})")

        nb = (int(L_KV) + block_k - 1) // block_k
        max_kblocks = max_span_blocks_upper_bound(
            L_kv=L_KV,
            block_k=block_k,
            span_len_factor=span_len_factor,
            span_power=span_power,
        )
        if max_span_blocks_cap is not None:
            max_kblocks = min(max_kblocks, int(max_span_blocks_cap))
        max_kblocks = max(1, min(int(max_kblocks), int(nb)))

        # Bucket to next power of 2 for metadata only (dynamic loops don't need this for kernel caching)
        bucketed_max_kblocks = _next_pow2_int(max_kblocks)

        Q2c = Q2.contiguous()
        Kc = K.contiguous()
        Vc = V.contiguous()

        dQ = torch.zeros_like(Q2c, dtype=torch.float32)
        dK = torch.zeros_like(Kc, dtype=torch.float32)
        dV = torch.zeros_like(Vc, dtype=torch.float32)

        BH_q = B * H_q
        P = L_Q * TOPK

        kv_for_q = make_kv_for_q_map(B=B, H_q=H_q, H_kv=H_kv, device=Q2.device)
        qs_all = qstart.reshape(-1).contiguous().to(torch.int32)
        qe_all = qend.reshape(-1).contiguous().to(torch.int32)

        if attention_mask is not None:
            attn_mask = attention_mask[:, :L_KV].contiguous().to(torch.int8)
            has_mask = True
        else:
            attn_mask = torch.empty((1,), device=Q2.device, dtype=torch.int8)
            has_mask = False

        Qf = Q2c.view(BH_q, L_Q, D)
        Kf = Kc.view(B * H_kv, L_KV, D)
        Vf = Vc.view(B * H_kv, L_KV, D)

        out_f = out.contiguous().view(BH_q, P, D)
        lse_f = lse.contiguous().view(BH_q, P)

        grad_out_f = grad_out.contiguous().view(BH_q, P, D)
        grad_lse_f = grad_lse.contiguous().to(torch.float32).view(BH_q, P)

        block_d = _next_pow2_int(D)
        grid = (BH_q * P,)

        _span_attn_bucketed_banded_backward_gqa_kernel[grid](
            Qf,
            Kf,
            Vf,
            kv_for_q,
            qs_all,
            qe_all,
            attn_mask,
            out_f,
            lse_f,
            grad_out_f,
            grad_lse_f,
            dQ.view(BH_q, L_Q, D),
            dK.view(B * H_kv, L_KV, D),
            dV.view(B * H_kv, L_KV, D),
            P=P,
            L_Q=L_Q,
            L_KV=L_KV,
            D=D,
            TOPK=TOPK,
            H_q=H_q,
            BLOCK_K=block_k,
            BLOCK_D=block_d,
            SM_SCALE=1.0 / math.sqrt(D),
            HAS_ATTN_MASK=has_mask,
            num_warps=4,
        )

        return dQ.to(Q2.dtype), dK.to(K.dtype), dV.to(V.dtype), None, None, None, None, None, None, None, None, None, None, None


# =============================================================================
# Public API: Bucketed span attention with automatic backward support
# =============================================================================

def span_attention_bucketed_packed_tiles_gqa_independent(
    Q2: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    qstart: torch.Tensor,
    qend: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None,
    block_k: int = DEFAULT_BLOCK_K,
    block_p: int = DEFAULT_BLOCK_P,
    span_len_factor: float,
    span_power: float,
    max_span_blocks_cap: int | None = None,
    num_warps: int = DEFAULT_NUM_WARPS_SPAN,
    num_stages: int = DEFAULT_NUM_STAGES_SPAN,
    use_bf16_pv: bool = True,
):
    """
    Bucketed span attention for GQA with automatic backward support.
    
    When gradients are enabled and inputs require grad, this uses an autograd
    Function with a custom backward kernel. Otherwise, uses the efficient
    forward-only path.
    
    Args:
        Q2: Query tensor [B, H_q, L_Q, D]
        K: Key tensor [B, H_kv, L_KV, D]
        V: Value tensor [B, H_kv, L_KV, D]
        qstart: Start positions [B, H_q, L_Q, TOPK]
        qend: End positions [B, H_q, L_Q, TOPK]
        attention_mask: Optional boolean mask [B, L_KV]
        block_k: Block size for K dimension
        block_p: Block size for processing queries
        span_len_factor: Factor for span length computation
        span_power: Power for span length computation
        max_span_blocks_cap: Optional cap on max span blocks
        num_warps: Number of warps per block
        num_stages: Number of pipeline stages
        use_bf16_pv: Use bf16 for P@V multiplication
        
    Returns:
        out: Output tensor [B, H_q, L_Q, TOPK, D]
        lse: Log-sum-exp tensor [B, H_q, L_Q, TOPK]
    """
    # Use forward-only path when gradients are not needed
    if not (torch.is_grad_enabled() and (Q2.requires_grad or K.requires_grad or V.requires_grad)):
        return _span_attention_bucketed_packed_tiles_gqa_independent_forward(
            Q2,
            K,
            V,
            qstart,
            qend,
            attention_mask=attention_mask,
            block_k=int(block_k),
            block_p=int(block_p),
            span_len_factor=float(span_len_factor),
            span_power=float(span_power),
            max_span_blocks_cap=None if max_span_blocks_cap is None else int(max_span_blocks_cap),
            num_warps=int(num_warps),
            num_stages=int(num_stages),
            use_bf16_pv=bool(use_bf16_pv),
        )

    # Use autograd Function when gradients are needed
    return _SpanAttentionBucketedPackedTilesGQAFn.apply(
        Q2,
        K,
        V,
        qstart,
        qend,
        attention_mask,
        int(block_k),
        int(block_p),
        float(span_len_factor),
        float(span_power),
        None if max_span_blocks_cap is None else int(max_span_blocks_cap),
        int(num_warps),
        int(num_stages),
        bool(use_bf16_pv),
    )


__all__ = [
    "DEFAULT_BLOCK_K",
    "DEFAULT_BLOCK_P",
    "DEFAULT_NUM_WARPS_SPAN",
    "DEFAULT_NUM_STAGES_SPAN",
    "DEFAULT_BLOCK_Q_MERGE",
    "DEFAULT_NUM_WARPS_MERGE",
    "max_span_blocks_upper_bound",
    "make_kv_for_q_map",
    "build_meta_histogram_banded_direct_active",
    "span_attention_bucketed_packed_tiles_gqa_independent",
    "merge_gate_qmajor",
]
