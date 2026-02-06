"""Attention module for superlinear attention patterns.

This module provides the main entry points for using superlinear attention:

Prefill (training and prompt processing):
- fused_prefill_with_swtriton_gqa: Main prefill function for GQA models
- fused_prefill_with_swtriton_bucketed_gqa: Optimized bucketed variant

Decode (autoregressive generation):
- decode_span_attention_staged: Basic decode for standard attention
- decode_span_attention_staged_gqa: GQA-optimized decode

Example:
    from superlinear.attention import fused_prefill_with_swtriton_gqa, decode_span_attention_staged_gqa

    # During prefill
    output = fused_prefill_with_swtriton_gqa(
        Q1, Q2, K, V, cache_position,
        attention_mask=mask,
        sw_index=3,
        backward_factor=3.0,
        forward_factor=1.0,
    )

    # During decode
    output = decode_span_attention_staged_gqa(
        Q1, Q2, K, V, cache_position,
        attention_mask=mask,
        sw_index=3,
        enable_gqa=True,
    )
"""

from .api import (
    # Main prefill entry points
    build_sw_blockmask,
    fused_prefill_with_swflex,
    fused_prefill_with_swflex_gqa,
    fused_prefill_with_swtriton,
    fused_prefill_with_swtriton_gqa,
    fused_prefill_with_swtriton_bucketed_gqa,
    prefill,
    prefill_gqa,
    prefill_bucketed_gqa,
    prefill_mha,
    prefill_bucketed,
    # Decode entry points
    decode_span_attention_staged,
    decode_span_attention_staged_gqa,
    decode_attention,
    decode_attention_gqa,
    # Lower-level functions
    full_span_attention_fused_with_search_values,
    full_span_attention_fused_with_search_values_gqa,
)

__all__ = [
    # Main prefill entry points
    "build_sw_blockmask",
    "fused_prefill_with_swflex",
    "fused_prefill_with_swflex_gqa",
    "fused_prefill_with_swtriton",
    "fused_prefill_with_swtriton_gqa",
    "fused_prefill_with_swtriton_bucketed_gqa",
    "prefill",
    "prefill_gqa",
    "prefill_bucketed_gqa",
    "prefill_mha",
    "prefill_bucketed",
    # Decode entry points
    "decode_span_attention_staged",
    "decode_span_attention_staged_gqa",
    "decode_attention",
    "decode_attention_gqa",
    # Lower-level functions
    "full_span_attention_fused_with_search_values",
    "full_span_attention_fused_with_search_values_gqa",
]
