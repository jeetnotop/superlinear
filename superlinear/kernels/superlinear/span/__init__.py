"""Span attention module for superlinear attention patterns."""

from .api import (
    # High-level functions
    fused_span_attention,
    fused_span_attention_gqa,
    span_attention,
    span_attention_gqa,
    decode_span_attention_staged,
    decode_span_attention_staged_gqa,
    decode_span_attention,
    decode_span_attention_gqa,
    # Mask utilities
    create_span_mask,
    create_sort_indices,
    create_sorted_span_mask,
    create_sliding_window_mask,
    invert_sorted_matrix,
    # Adjustment utilities
    compute_qend_from_qanchor,
)

__all__ = [
    # High-level functions
    "fused_span_attention",
    "fused_span_attention_gqa",
    "span_attention",
    "span_attention_gqa",
    "decode_span_attention_staged",
    "decode_span_attention_staged_gqa",
    "decode_span_attention",
    "decode_span_attention_gqa",
    # Mask utilities
    "create_span_mask",
    "create_sort_indices",
    "create_sorted_span_mask",
    "create_sliding_window_mask",
    "invert_sorted_matrix",
    # Adjustment utilities
    "compute_qend_from_qanchor",
]
