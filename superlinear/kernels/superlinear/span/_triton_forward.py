"""Compatibility re-exports for span Triton kernels.

Historically, the project had a `span_attention3.py` file that defined the base
span kernels and helper utilities. During the OSS migration, the canonical
implementation moved to [superlinear/span/_triton_impl.py](superlinear/span/_triton_impl.py).

This module remains as a thin re-export layer to avoid stale imports.
"""

from ._triton_impl import (
    _assert_no_span_sw_overlap,
    _next_power_of_two,
    fused_span_backward_kernel,
    fused_span_forward_kernel,
)

__all__ = [
    "_next_power_of_two",
    "_assert_no_span_sw_overlap",
    "fused_span_forward_kernel",
    "fused_span_backward_kernel",
]
