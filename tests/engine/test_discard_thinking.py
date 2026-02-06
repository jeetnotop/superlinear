import asyncio

import pytest


torch = pytest.importorskip("torch", reason="torch not installed")


from superlinear.engine.chat_engine import ChatEngine, EngineConfig
from superlinear.engine.chat_types import ChatMessage, ChatRequest


class _FakeTokenizer:
    def __init__(self) -> None:
        self.all_special_tokens: list[str] = []
        self._char_offset = 1000
        self._special_to_id = {"<think>": 900, "</think>": 901}
        self._id_to_special = {v: k for k, v in self._special_to_id.items()}

    def encode(self, text: str, *, add_special_tokens: bool = False):
        _ = add_special_tokens
        if text in self._special_to_id:
            return [self._special_to_id[text]]
        return [self._char_offset + ord(ch) for ch in text]

    def decode(self, ids, *, skip_special_tokens: bool = False):
        _ = skip_special_tokens
        out: list[str] = []
        for tid in ids:
            tid = int(tid)
            if tid in self._id_to_special:
                out.append(self._id_to_special[tid])
                continue
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
        _ = kwargs
        ids: list[int] = [1]  # BOS-ish
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content") or ""
            if role == "user":
                ids.append(10)
            elif role == "assistant":
                ids.append(20)
            else:
                ids.append(30)
            ids.extend(self.encode(content, add_special_tokens=False))
            ids.append(2)  # EOS-ish per message

        if add_generation_prompt:
            ids.append(20)  # assistant marker
            if enable_thinking:
                ids.append(self._special_to_id["<think>"])
            else:
                ids.extend([self._special_to_id["<think>"], self._special_to_id["</think>"]])

        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids


class _FakeAdapter:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()
        self._sessions: dict[str, dict[str, int]] = {}
        self.append_lens: list[int] = []
        self.checkpoints: list[int] = []
        self.restores: list[int] = []
        self.max_pos_seen: int = 0

    def create_session(self, *, cache_id: str, max_seq_len: int = 0, **kwargs) -> None:
        _ = (max_seq_len, kwargs)
        self._sessions[cache_id] = {"current_pos": 0}

    def get_session_info(self, cache_id: str):
        if cache_id not in self._sessions:
            raise KeyError(cache_id)
        return {"cache_id": cache_id, "current_pos": self._sessions[cache_id]["current_pos"], "max_seq_len": 1_000_000}

    def append_to_session(self, *, cache_id: str, input_ids, **kwargs) -> None:
        _ = kwargs
        if cache_id not in self._sessions:
            raise KeyError(cache_id)
        n = int(getattr(input_ids, "shape")[1])
        self.append_lens.append(n)
        self._sessions[cache_id]["current_pos"] += n
        self.max_pos_seen = max(self.max_pos_seen, self._sessions[cache_id]["current_pos"])

    def checkpoint_session(self, cache_id: str):
        if cache_id not in self._sessions:
            raise KeyError(cache_id)
        pos = int(self._sessions[cache_id]["current_pos"])
        self.checkpoints.append(pos)
        return pos

    def restore_session_checkpoint(self, *, cache_id: str, checkpoint) -> None:
        if cache_id not in self._sessions:
            raise KeyError(cache_id)
        self._sessions[cache_id]["current_pos"] = int(checkpoint)
        self.restores.append(int(checkpoint))

    def stream_generate_session(self, *, cache_id: str, max_new_tokens: int = 0, **kwargs):
        _ = (max_new_tokens, kwargs)
        if cache_id not in self._sessions:
            raise KeyError(cache_id)

        # Emit: <think>abc</think>Hello
        out = [
            900,
            self.tokenizer.encode("abc", add_special_tokens=False)[0],
            self.tokenizer.encode("bc", add_special_tokens=False)[0],
            901,
            *self.tokenizer.encode("Hello", add_special_tokens=False),
        ]

        def _gen():
            try:
                for tid in out:
                    self._sessions[cache_id]["current_pos"] += 1
                    self.max_pos_seen = max(self.max_pos_seen, self._sessions[cache_id]["current_pos"])
                    yield torch.tensor(tid)
            finally:
                # Mirror the real adapter: state is persisted on close.
                self.max_pos_seen = max(self.max_pos_seen, self._sessions[cache_id]["current_pos"])

        return _gen()

    def stream_generate(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


def test_discard_thinking_commits_stripped_history_tokens():
    adapter = _FakeAdapter()
    adapter.create_session(cache_id="s1")

    engine = ChatEngine(adapter, config=EngineConfig(enable_thinking=True, discard_thinking=True))
    req = ChatRequest(
        messages=[ChatMessage(role="user", content="hi")],
        session_id="s1",
        session_append_from_pos=0,
        max_tokens=16,
    )

    result = asyncio.run(engine.generate_chat(req))
    assert result["content"] == "Hello"

    persisted = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    ids_persisted = adapter.tokenizer.apply_chat_template(
        persisted,
        add_generation_prompt=False,
        return_tensors="pt",
        enable_thinking=True,
    )

    assert adapter.get_session_info("s1")["current_pos"] == int(ids_persisted.shape[1])
    assert adapter.max_pos_seen > adapter.get_session_info("s1")["current_pos"]

    ids_no_gen = adapter.tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        add_generation_prompt=False,
        return_tensors="pt",
        enable_thinking=True,
    )
    boundary_pos = int(ids_no_gen.shape[1])
    ids_with_gen = adapter.tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=True,
    )
    gen_prompt_len = int(ids_with_gen.shape[1]) - boundary_pos

    # Option B (checkpoint-after-user) should prefill user, then gen prompt, then commit assistant delta.
    assert adapter.append_lens[:2] == [boundary_pos, gen_prompt_len]


def test_discard_thinking_false_is_supported_for_sessions():
    adapter = _FakeAdapter()
    adapter.create_session(cache_id="s1")

    engine = ChatEngine(adapter, config=EngineConfig(enable_thinking=True, discard_thinking=True))
    req = ChatRequest(
        messages=[ChatMessage(role="user", content="hi")],
        session_id="s1",
        session_append_from_pos=0,
        max_tokens=16,
        discard_thinking=False,
    )

    result = asyncio.run(engine.generate_chat(req))
    assert result["content"] == "Hello"
    assert isinstance(result.get("raw_content"), str)
    assert "<think>" in result["raw_content"]
    assert "</think>" in result["raw_content"]
