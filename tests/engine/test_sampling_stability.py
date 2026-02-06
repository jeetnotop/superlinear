import torch


def _sample_token_for_test(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Local copy of SuperlinearAdapter._sample_token logic for unit testing.

    We keep this minimal and CPU-only: the goal is to validate numerical
    sanitation and fallback behavior (no NaNs/Infs, no negative probs).
    """
    if temperature is None or temperature < 0:
        raise ValueError
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits_f = logits.float() / float(temperature)
    probs = torch.softmax(logits_f, dim=-1)

    if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = torch.clamp(probs, min=0.0)
        z = probs.sum(dim=-1, keepdim=True)
        if (z <= 0).any():
            return torch.argmax(logits, dim=-1, keepdim=True)
        probs = probs / z

    return torch.multinomial(probs, 1)


def test_sampling_fp32_softmax_avoids_nan() -> None:
    # Extreme logits that are likely to overflow in fp16 softmax.
    logits = torch.tensor([[10000.0, -10000.0, 0.0]], dtype=torch.float16)
    tok = _sample_token_for_test(logits, temperature=0.1)
    assert tok.shape == (1, 1)
    assert int(tok.item()) in {0, 1, 2}


def test_sampling_fallback_when_all_probs_zero() -> None:
    # Force NaNs by injecting NaN logits; after sanitation, probs could be all zeros.
    logits = torch.tensor([[float("nan"), float("nan"), float("nan")]], dtype=torch.float16)
    tok = _sample_token_for_test(logits, temperature=0.7)
    # Greedy fallback (argmax) returns 0 by PyTorch convention for all-NaN.
    assert tok.shape == (1, 1)
    assert int(tok.item()) == 0
