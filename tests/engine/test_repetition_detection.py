import random

from superlinear.engine.repetition import detect_repetition_kmp_tail, prefix_function


def test_prefix_function_basic():
    # Classic "ababa" style prefix-function: [0, 0, 1, 2, 3]
    seq = [1, 2, 1, 2, 1]
    assert prefix_function(seq) == [0, 0, 1, 2, 3]


def test_detect_repetition_period_4():
    pattern = [10, 11, 12, 13]
    tokens = pattern * 20
    hit = detect_repetition_kmp_tail(
        tokens,
        tail_len=64,
        min_generated_tokens=0,
        min_repeats=3,
        max_period=32,
        min_unique_tokens=4,
    )
    assert hit is not None
    assert hit.period == 4


def test_detect_repetition_misaligned_tail():
    pattern = [1, 2, 3, 4]
    tokens = pattern * 30
    hit = detect_repetition_kmp_tail(
        tokens,
        tail_len=30,  # Not a multiple of 4; tail starts mid-period.
        min_generated_tokens=0,
        min_repeats=3,
        max_period=32,
        min_unique_tokens=4,
    )
    assert hit is not None
    assert hit.period == 4


def test_detect_repetition_break_near_end_is_rejected():
    pattern = [1, 2, 3, 4]
    tokens = pattern * 20
    # Break periodicity inside the last 3 repeats (last 12 tokens).
    tokens[-5] = 999
    hit = detect_repetition_kmp_tail(
        tokens,
        tail_len=64,
        min_generated_tokens=0,
        min_repeats=3,
        max_period=32,
        min_unique_tokens=4,
    )
    assert hit is None


def test_detect_repetition_trivial_loop_is_rejected():
    tokens = [7] * 1000
    hit = detect_repetition_kmp_tail(
        tokens,
        tail_len=128,
        min_generated_tokens=0,
        min_repeats=3,
        max_period=64,
        min_unique_tokens=4,
    )
    assert hit is None


def test_detect_repetition_random_tokens_no_false_positive():
    rng = random.Random(0)
    tokens = [rng.randrange(1 << 30) for _ in range(2048)]
    hit = detect_repetition_kmp_tail(
        tokens,
        tail_len=1024,
        min_generated_tokens=0,
        min_repeats=3,
        max_period=512,
        min_unique_tokens=4,
    )
    assert hit is None
