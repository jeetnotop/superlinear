"""Token-level repetition detection for early stopping.

This module implements a high-precision detector for exact token periodicity
in the recent tail of generated token IDs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RepeatHit:
    period: int
    repeats: int
    checked_tail_len: int


@dataclass(frozen=True)
class RepetitionDetectionConfig:
    """Configuration for repetition early-stop.

    Notes:
    - Defaults are tuned for typical long-context Q&A.
    - `enabled` defaults to True to catch repetition loops early.
    - Settings are conservative to avoid false positives on legitimate content.
    """

    enabled: bool = True
    tail_len: int = 1024
    check_every: int = 32
    min_generated_tokens: int = 256
    min_repeats: int = 3
    max_period: int = 512
    min_unique_tokens: int = 5

    def validate(self) -> None:
        if self.tail_len <= 0:
            raise ValueError("'repetition_detection.tail_len' must be > 0.")
        if self.check_every <= 0:
            raise ValueError("'repetition_detection.check_every' must be > 0.")
        if self.min_generated_tokens < 0:
            raise ValueError("'repetition_detection.min_generated_tokens' must be >= 0.")
        if self.min_repeats < 2:
            raise ValueError("'repetition_detection.min_repeats' must be >= 2.")
        if self.max_period <= 0:
            raise ValueError("'repetition_detection.max_period' must be > 0.")
        if self.min_unique_tokens <= 0:
            raise ValueError("'repetition_detection.min_unique_tokens' must be > 0.")

    def merged(self, override: Any | None) -> "RepetitionDetectionConfig":
        """Merge a request-level override (typically request.extra['repetition_detection'])."""
        if override is None:
            return self
        if isinstance(override, RepetitionDetectionConfig):
            override.validate()
            return override
        if not isinstance(override, dict):
            raise ValueError("'repetition_detection' must be an object.")

        data: dict[str, Any] = dict(override)
        if "min_unique_tokens" not in data and "min_unique_tokens_in_period" in data:
            data["min_unique_tokens"] = data["min_unique_tokens_in_period"]

        enabled = self.enabled
        if "enabled" in data:
            raw_enabled = data["enabled"]
            if not isinstance(raw_enabled, bool):
                raise ValueError("'repetition_detection.enabled' must be a boolean.")
            enabled = raw_enabled
        tail_len = self.tail_len if "tail_len" not in data else _coerce_int(data["tail_len"], "tail_len")
        check_every = (
            self.check_every
            if "check_every" not in data
            else _coerce_int(data["check_every"], "check_every")
        )
        min_generated_tokens = (
            self.min_generated_tokens
            if "min_generated_tokens" not in data
            else _coerce_int(data["min_generated_tokens"], "min_generated_tokens", min_value=0)
        )
        min_repeats = (
            self.min_repeats
            if "min_repeats" not in data
            else _coerce_int(data["min_repeats"], "min_repeats", min_value=2)
        )
        max_period = (
            self.max_period
            if "max_period" not in data
            else _coerce_int(data["max_period"], "max_period")
        )
        min_unique_tokens = (
            self.min_unique_tokens
            if "min_unique_tokens" not in data
            else _coerce_int(data["min_unique_tokens"], "min_unique_tokens", min_value=1)
        )

        merged = RepetitionDetectionConfig(
            enabled=enabled,
            tail_len=tail_len,
            check_every=check_every,
            min_generated_tokens=min_generated_tokens,
            min_repeats=min_repeats,
            max_period=max_period,
            min_unique_tokens=min_unique_tokens,
        )
        merged.validate()
        return merged


def _coerce_int(value: Any, name: str, *, min_value: int = 1) -> int:
    if isinstance(value, bool):
        raise ValueError(f"'repetition_detection.{name}' must be an integer.")
    try:
        out = int(value)
    except Exception as exc:
        raise ValueError(f"'repetition_detection.{name}' must be an integer.") from exc
    if out < min_value:
        raise ValueError(f"'repetition_detection.{name}' must be >= {min_value}.")
    return out


def prefix_function(seq: list[int]) -> list[int]:
    """Classic KMP prefix-function (pi array) for a sequence of ints.

    pi[i] = length of the longest proper prefix of seq[:i+1]
            that is also a suffix of seq[:i+1].
    """
    n = len(seq)
    pi = [0] * n
    j = 0
    for i in range(1, n):
        while j > 0 and seq[i] != seq[j]:
            j = pi[j - 1]
        if j < n and seq[i] == seq[j]:
            j += 1
        pi[i] = j
    return pi


def _nontrivial_period(period_tokens: list[int], *, min_unique_tokens: int) -> bool:
    # Avoid stopping on junk like a single token or whitespace/punctuation-only loops.
    return len(set(period_tokens)) >= min_unique_tokens


def detect_repetition_kmp_tail(
    tokens: list[int],
    *,
    tail_len: int = 1024,
    min_generated_tokens: int = 256,
    min_repeats: int = 3,
    max_period: int = 512,
    min_unique_tokens: int = 4,
) -> RepeatHit | None:
    """Detect exact periodic repetition using only the last `tail_len` tokens.

    Strategy:
    - Compute KMP prefix function on the tail.
    - Walk the border chain to derive candidate periods.
    - Validate candidate periods by checking the last `min_repeats` blocks match.
    """
    if tail_len <= 0:
        raise ValueError("'tail_len' must be > 0.")
    if min_generated_tokens < 0:
        raise ValueError("'min_generated_tokens' must be >= 0.")
    if min_repeats < 2:
        raise ValueError("'min_repeats' must be >= 2.")
    if max_period <= 0:
        raise ValueError("'max_period' must be > 0.")
    if min_unique_tokens <= 0:
        raise ValueError("'min_unique_tokens' must be > 0.")

    if len(tokens) < min_generated_tokens:
        return None

    tail = tokens[-tail_len:] if len(tokens) > tail_len else tokens
    L = len(tail)
    if L < min_repeats:
        return None

    pi = prefix_function(tail)
    if not pi:
        return None
    b = pi[-1]

    # Walk border chain: b -> pi[b-1] -> ...
    while b > 0:
        p = L - b
        if 1 <= p <= max_period:
            need = p * min_repeats
            if L >= need:
                a = tail[-p:]
                ok = True
                for r in range(2, min_repeats + 1):
                    if tail[-r * p : -(r - 1) * p] != a:
                        ok = False
                        break
                if ok and _nontrivial_period(a, min_unique_tokens=min_unique_tokens):
                    return RepeatHit(period=p, repeats=min_repeats, checked_tail_len=L)
        b = pi[b - 1]

    return None
