"""
Superlinear Attention - Efficient sparse attention kernels for long-context language models.

This package provides optimized Triton kernels for superlinear attention patterns,
combining power-law span selection with sliding window attention.

Quick Start:
    from superlinear.attention import fused_prefill_with_swtriton_gqa

    # During prefill
    output = fused_prefill_with_swtriton_gqa(
        Q1, Q2, K, V, cache_position,
        attention_mask=mask,
        sw_index=3,
        backward_factor=3.0,
        forward_factor=1.0,
    )

Submodules:
    - superlinear.attention: High-level prefill and decode functions
    - superlinear.search: Span search operations
    - superlinear.span: Low-level span attention kernels
    - superlinear.common: Shared utilities (power params, adjustments)

Environment Variables:
    SPAN_ATTENTION_ENABLE_ASSERTS: Set to "1" to enable validation checks
        (disabled by default for CUDA graph capture compatibility)
"""

from superlinear._version import __version__

# Core attention functions (main entry points)
from superlinear.kernels.superlinear.attention import (
    fused_prefill_with_swtriton_gqa,
    fused_prefill_with_swtriton_bucketed_gqa,
    prefill,
    prefill_gqa,
    prefill_bucketed_gqa,
    prefill_mha,
    prefill_bucketed,
    decode_span_attention_staged,
    decode_span_attention_staged_gqa,
    decode_attention,
    decode_attention_gqa,
)

# Search functions
from superlinear.kernels.superlinear.search import (
    span_search,
    search_spans,
    span_search_with_values,
    span_search_with_values_gqa,
)

# Span attention kernels
from superlinear.kernels.superlinear.span import (
    span_attention,
    span_attention_gqa,
    decode_span_attention,
    decode_span_attention_gqa,
)

# Common utilities
from superlinear.kernels.common import (
    StripePowerParams,
    derive_stripe_power_params,
    window_len_from_sw_index,
    compute_qend_from_qanchor,
)

# Backward-compatible submodule aliases
from superlinear.kernels.superlinear import attention, search, span
from superlinear.kernels import common

# Runtime utilities
from superlinear.runtime import (
    is_triton_available,
    is_cuda_available,
)

__all__ = [
    # Version
    "__version__",
    # Main attention functions
    "fused_prefill_with_swtriton_gqa",
    "fused_prefill_with_swtriton_bucketed_gqa",
    "prefill",
    "prefill_gqa",
    "prefill_bucketed_gqa",
    "prefill_mha",
    "prefill_bucketed",
    "decode_span_attention_staged",
    "decode_span_attention_staged_gqa",
    "decode_attention",
    "decode_attention_gqa",
    "decode_span_attention",
    "decode_span_attention_gqa",
    # Search
    "span_search",
    "search_spans",
    "span_search_with_values",
    "span_search_with_values_gqa",
    # Span attention
    "span_attention",
    "span_attention_gqa",
    # Utilities
    "StripePowerParams",
    "derive_stripe_power_params",
    "window_len_from_sw_index",
    "compute_qend_from_qanchor",
    # Runtime
    "is_triton_available",
    "is_cuda_available",
]
