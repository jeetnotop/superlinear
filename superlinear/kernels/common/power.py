"""Stripe power parameters and window length utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


_SUPPORTED_INV_SEARCH_POWER_INTS = (2, 3, 4, 5, 6)


@dataclass(frozen=True)
class StripePowerParams:
    """
    Derived stripe power parameters.

    `inv_p` is the exponent used in `floor(i ** inv_p)` (i.e., inv_p = 1 / p).
    `triton_inv_n` is a small integer specialization used by Triton kernels:
      - 0 => float path (use inv_p)
      - 2..6 => exact integer exponentiation path
    """

    inv_p: float
    p: float
    triton_inv_n: int


def derive_stripe_power_params(
    *,
    search_power: Optional[float],
    inv_search_power_int: Optional[int],
) -> StripePowerParams:
    """
    Enforce the XOR API rule and derive the (inv_p, p) pair.

    Exactly one of `search_power` and `inv_search_power_int` must be provided.
    The default call sites should pass `inv_search_power_int=2, search_power=None`.
    """
    if (inv_search_power_int is None) == (search_power is None):
        raise ValueError("Provide exactly one of inv_search_power_int or search_power")

    if inv_search_power_int is not None:
        inv_n = int(inv_search_power_int)
        if inv_n not in _SUPPORTED_INV_SEARCH_POWER_INTS:
            raise ValueError(
                f"inv_search_power_int must be one of {_SUPPORTED_INV_SEARCH_POWER_INTS} (got {inv_n})"
            )
        inv_p = float(inv_n)
        p = 1.0 / inv_p
        return StripePowerParams(inv_p=inv_p, p=p, triton_inv_n=inv_n)

    p_float = float(search_power)
    if not math.isfinite(p_float) or not (0.0 < p_float < 1.0):
        raise ValueError(f"search_power must be finite and in (0, 1) (got {search_power})")
    inv_p = 1.0 / p_float
    p = p_float

    # Optional specialization: if inv_p is exactly an int in our supported set, use the
    # integer fast path even when the caller chose float mode.
    inv_p_round = int(round(inv_p))
    if inv_p_round in _SUPPORTED_INV_SEARCH_POWER_INTS and float(inv_p_round) == inv_p:
        triton_inv_n = inv_p_round
    else:
        triton_inv_n = 0

    return StripePowerParams(inv_p=inv_p, p=p, triton_inv_n=triton_inv_n)


def floor_nth_root(value: int, n: int) -> int:
    """
    Return floor(value ** (1/n)) using integer arithmetic (exact).
    """
    if n <= 0:
        raise ValueError(f"n must be > 0 (got {n})")
    if value < 0:
        raise ValueError(f"value must be >= 0 (got {value})")
    if value in (0, 1):
        return value

    # Initial guess (may be off by a small amount due to float rounding).
    guess = int(value ** (1.0 / n))
    guess = max(1, guess)

    # Adjust guess to satisfy guess**n <= value < (guess+1)**n.
    while pow(guess, n) > value:
        guess -= 1
    while pow(guess + 1, n) <= value:
        guess += 1
    return guess


def window_len_from_sw_index(
    sw_index: int,
    *,
    search_power: Optional[float],
    inv_search_power_int: Optional[int],
) -> int:
    """
    Compute the sliding-window width:
      window_len = floor((sw_index + 1) ** inv_p) - 1
    """
    if sw_index < 0:
        raise ValueError(f"sw_index must be >= 0 (got {sw_index})")

    params = derive_stripe_power_params(
        search_power=search_power, inv_search_power_int=inv_search_power_int
    )
    base = int(sw_index) + 1

    if params.triton_inv_n != 0:
        return (base ** params.triton_inv_n) - 1

    floor_power = int(base ** params.inv_p)
    return floor_power - 1


def max_stripe_index_for_token_pos(
    token_pos: int,
    *,
    search_power: Optional[float],
    inv_search_power_int: Optional[int],
) -> int:
    """
    Compute the maximum stripe index i such that floor(i ** inv_p) <= token_pos.

    For inv_search_power_int mode, this is floor(token_pos ** p) with exact integer roots.
    """
    if token_pos < 0:
        raise ValueError(f"token_pos must be >= 0 (got {token_pos})")

    params = derive_stripe_power_params(
        search_power=search_power, inv_search_power_int=inv_search_power_int
    )

    if params.triton_inv_n != 0:
        return floor_nth_root(token_pos, params.triton_inv_n)

    if token_pos <= 1:
        return token_pos

    approx = int(token_pos ** params.p)
    approx = max(1, approx)
    # Ensure this is not an under-approx by adding a small cushion and adjusting down.
    approx += 2

    def floor_power(i: int) -> int:
        return int(i ** params.inv_p)

    while approx > 0 and floor_power(approx) > token_pos:
        approx -= 1
    while floor_power(approx + 1) <= token_pos:
        approx += 1
    return approx
