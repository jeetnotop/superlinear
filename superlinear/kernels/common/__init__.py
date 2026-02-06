"""Common internal utilities for superlinear (not part of public API)."""

from __future__ import annotations

from superlinear.kernels.common.power import (
    StripePowerParams,
    derive_stripe_power_params,
    floor_nth_root,
    max_stripe_index_for_token_pos,
    window_len_from_sw_index,
)
from superlinear.kernels.common.adjustment import compute_qend_from_qanchor

__all__ = [
    "StripePowerParams",
    "derive_stripe_power_params",
    "floor_nth_root",
    "max_stripe_index_for_token_pos",
    "window_len_from_sw_index",
    "compute_qend_from_qanchor",
]
