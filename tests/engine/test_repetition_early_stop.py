import asyncio

import pytest


torch = pytest.importorskip("torch", reason="torch not installed")


from superlinear.engine.chat_engine import ChatEngine, EngineConfig
from superlinear.engine.chat_types import ChatMessage, ChatRequest
from superlinear.engine.repetition import RepetitionDetectionConfig


class _FakeTokenizer:
    def __init__(self) -> None:
        self._char_offset = 1000

    def encode(self, text: str, *, add_special_tokens: bool = False):
        _ = add_special_tokens
        return [self._char_offset + ord(ch) for ch in text]

    def decode(self, ids, *, skip_special_tokens: bool = False):
        _ = skip_special_tokens
        out: list[str] = []
        for tid in ids:
            tid = int(tid)
            if tid >= self._char_offset:
                out.append(chr(tid - self._char_offset))
        return "".join(out)

    def apply_chat_template(
        self,
        messages,
        *,
        add_generation_prompt: bool,
        return_tensors: str | None = None,
        enable_thinking: bool = True,
        **kwargs,
    ):
        _ = (enable_thinking, kwargs)
        ids: list[int] = [1]  # BOS-ish
        for msg in messages:
            content = msg.get("content") or ""
            ids.extend(self.encode(content, add_special_tokens=False))
            ids.append(2)  # EOS-ish per message
        if add_generation_prompt:
            ids.append(3)  # assistant marker-ish
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids


class _FakeAdapter:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()

    def stream_generate(self, input_ids, *, max_new_tokens: int, **kwargs):
        _ = (input_ids, kwargs)
        pattern = self.tokenizer.encode("abcd", add_special_tokens=False)

        def _gen():
            for i in range(int(max_new_tokens)):
                yield torch.tensor(pattern[i % len(pattern)])

        return _gen()

    def stream_generate_session(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


def test_chat_engine_early_stops_with_finish_reason_repetition():
    adapter = _FakeAdapter()
    engine = ChatEngine(
        adapter,
        config=EngineConfig(
            repetition_detection=RepetitionDetectionConfig(
                enabled=True,
                tail_len=64,
                check_every=4,
                min_generated_tokens=12,
                min_repeats=3,
                max_period=32,
                min_unique_tokens=4,
            )
        ),
    )
    req = ChatRequest(messages=[ChatMessage(role="user", content="hi")], max_tokens=200)
    result = asyncio.run(engine.generate_chat(req))
    assert result["finish_reason"] == "repetition"
    assert result["usage"].completion_tokens < int(req.max_tokens)


class _FakeAdapterWithThinking:
    """Adapter that simulates repetition inside <think> block then recovers.
    
    When enable_thinking=True, the model generation starts INSIDE the think block
    (the generation prompt already includes <think>), so we don't emit <think> tokens.
    
    Scenario:
    1. Emit repetitive pattern (triggers bailout after ~16 tokens)
    2. Engine injects </think> to parser
    3. Model continues emitting pattern tokens (which now go to content due to parser state)
    4. Eventually model emits </think> and answer
    
    For test: We track when bailout happened (via repetition_tail clear) and after that
    produce different tokens to avoid re-triggering repetition.
    """

    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()
        self._bailout_idx: int | None = None

    def stream_generate(self, input_ids, *, max_new_tokens: int, **kwargs):
        _ = (input_ids, kwargs)
        # Pattern with 5 unique chars for detection
        pattern = self.tokenizer.encode("abcde", add_special_tokens=False)
        think_close = self.tokenizer.encode("</think>", add_special_tokens=False)
        # Answer with high diversity to not trigger repetition again
        answer = self.tokenizer.encode("The final answer is forty-two.", add_special_tokens=False)
        eos = [2]

        def _gen():
            # Phase 1: Repetitive content (will trigger bailout around token 16)
            for i in range(24):  # Enough to trigger detection
                yield torch.tensor(pattern[i % len(pattern)])
            
            # Phase 2: Model's natural output after thinking
            # Emit </think> (model doesn't know bailout happened)
            for tid in think_close:
                yield torch.tensor(tid)
            
            # Phase 3: Answer (diverse, no repetition)
            for tid in answer:
                yield torch.tensor(tid)
            
            # EOS
            for tid in eos:
                yield torch.tensor(tid)

        return _gen()

    def stream_generate_session(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


def test_chat_engine_thinking_bailout_injects_think_close_and_continues():
    """When repetition is detected during thinking, inject </think> and continue generating."""
    adapter = _FakeAdapterWithThinking()
    engine = ChatEngine(
        adapter,
        config=EngineConfig(
            enable_thinking=True,  # Enable thinking mode
            repetition_detection=RepetitionDetectionConfig(
                enabled=True,
                tail_len=64,
                check_every=4,
                min_generated_tokens=12,
                min_repeats=3,
                max_period=32,
                min_unique_tokens=4,
            )
        ),
    )
    req = ChatRequest(
        messages=[ChatMessage(role="user", content="hi")],
        max_tokens=200,
        stream_thinking=True,
    )
    result = asyncio.run(engine.generate_chat(req))

    # Should be "stop" (EOS reached) not "repetition" because we bailed out and continued
    assert result["finish_reason"] == "stop"

    # The content should contain the answer
    # After bailout, post-bailout tokens go to content, then model's </think> appears
    # as literal content, then the answer follows
    content = result.get("content") or ""
    # The content will have: post-bailout pattern chars + "</think>" + answer
    assert "forty-two" in content or "answer" in content

