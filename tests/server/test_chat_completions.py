import json
import threading

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")


def _collect_sse_events(raw: str) -> list[str]:
    events: list[str] = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if not block.startswith("data: "):
            continue
        events.append(block[len("data: ") :])
    return events


def test_non_stream_basic_shape():
    from fastapi.testclient import TestClient

    from apps.server.app import create_app
    from superlinear.engine.chat_types import Usage

    class FakeEngine:
        async def generate_chat(self, req):
            return {
                "content": "hello",
                "tool_calls": [],
                "finish_reason": "stop",
                "usage": Usage(prompt_tokens=3, completion_tokens=2),
                "timing": None,
            }

    app = create_app(engine=FakeEngine(), model_id="superlinear-test")
    client = TestClient(app)

    resp = client.post(
        "/v1/chat/completions",
        json={"model": "superlinear-test", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "superlinear-test"
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["message"]["content"] == "hello"
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"]["prompt_tokens"] == 3
    assert data["usage"]["completion_tokens"] == 2
    assert data["usage"]["total_tokens"] == 5


def test_stream_sse_ordering_and_done():
    from fastapi.testclient import TestClient

    from apps.server.app import create_app
    from superlinear.engine.chat_types import DeltaEvent, FinalEvent, Timing, Usage

    class FakeEngine:
        async def astream_chat(self, req):
            yield DeltaEvent("he")
            yield DeltaEvent("llo")
            yield FinalEvent(
                finish_reason="stop",
                usage=Usage(prompt_tokens=1, completion_tokens=2),
                timing=Timing(),
            )

    app = create_app(engine=FakeEngine(), model_id="superlinear-test")
    client = TestClient(app)

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "superlinear-test", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
    ) as resp:
        assert resp.status_code == 200
        resp.read()  # Must read the stream before accessing .text
        raw = resp.text

    events = _collect_sse_events(raw)
    assert events[0] != "[DONE]"
    assert events[-1] == "[DONE]"

    first = json.loads(events[0])
    assert first["object"] == "chat.completion.chunk"
    assert first["choices"][0]["delta"]["role"] == "assistant"

    # Content chunks should concatenate.
    content = ""
    for e in events[1:]:
        if e == "[DONE]":
            break
        obj = json.loads(e)
        delta = obj["choices"][0]["delta"]
        if "content" in delta:
            content += delta["content"]
    assert content == "hello"

    # Terminal chunk includes finish_reason.
    terminal = json.loads(events[-2])
    assert terminal["choices"][0]["finish_reason"] == "stop"


def test_tool_calls_non_stream():
    from fastapi.testclient import TestClient

    from apps.server.app import create_app
    from superlinear.engine.chat_types import ToolCall, Usage

    class FakeEngine:
        async def generate_chat(self, req):
            return {
                "content": None,
                "tool_calls": [
                    ToolCall(id="call_123", name="lookup", arguments={"q": "hi"}),
                ],
                "finish_reason": "tool_calls",
                "usage": Usage(prompt_tokens=3, completion_tokens=5),
                "timing": None,
            }

    app = create_app(engine=FakeEngine(), model_id="superlinear-test")
    client = TestClient(app)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "superlinear-test",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    msg = data["choices"][0]["message"]
    assert msg["content"] is None
    assert msg["tool_calls"][0]["id"] == "call_123"
    assert msg["tool_calls"][0]["function"]["name"] == "lookup"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"q": "hi"}
    assert data["choices"][0]["finish_reason"] == "tool_calls"


def test_tool_calls_streaming_chunk():
    from fastapi.testclient import TestClient

    from apps.server.app import create_app
    from superlinear.engine.chat_types import FinalEvent, Timing, ToolCall, ToolCallEvent, Usage

    class FakeEngine:
        async def astream_chat(self, req):
            yield ToolCallEvent(
                tool_calls=[ToolCall(id="call_123", name="lookup", arguments={"q": "hi"})]
            )
            yield FinalEvent(
                finish_reason="tool_calls",
                usage=Usage(prompt_tokens=3, completion_tokens=5),
                timing=Timing(),
            )

    app = create_app(engine=FakeEngine(), model_id="superlinear-test")
    client = TestClient(app)

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "superlinear-test",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
        },
    ) as resp:
        assert resp.status_code == 200
        resp.read()  # Must read the stream before accessing .text
        raw = resp.text

    events = _collect_sse_events(raw)
    assert events[-1] == "[DONE]"

    # One chunk should include tool_calls delta.
    tool_chunk = None
    for e in events:
        if e in {"[DONE]"}:
            continue
        obj = json.loads(e)
        delta = obj["choices"][0]["delta"]
        if "tool_calls" in delta:
            tool_chunk = delta["tool_calls"]
            break
    assert tool_chunk is not None
    assert tool_chunk[0]["id"] == "call_123"
    assert tool_chunk[0]["function"]["name"] == "lookup"

    terminal = json.loads(events[-2])
    assert terminal["choices"][0]["finish_reason"] == "tool_calls"


def test_max_completion_tokens_alias_and_cap():
    from fastapi.testclient import TestClient

    from apps.server.app import create_app
    from superlinear.engine.chat_types import Usage

    class FakeEngine:
        async def generate_chat(self, req):
            assert req.max_tokens == 7
            return {
                "content": "ok",
                "tool_calls": [],
                "finish_reason": "stop",
                "usage": Usage(prompt_tokens=1, completion_tokens=1),
                "timing": None,
            }

    app = create_app(engine=FakeEngine(), model_id="superlinear-test", http_max_completion_tokens=10)
    client = TestClient(app)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "superlinear-test",
            "max_completion_tokens": 7,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200

    too_big = client.post(
        "/v1/chat/completions",
        json={
            "model": "superlinear-test",
            "max_tokens": 11,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert too_big.status_code == 400

    mismatch = client.post(
        "/v1/chat/completions",
        json={
            "model": "superlinear-test",
            "max_tokens": 7,
            "max_completion_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert mismatch.status_code == 400


def test_session_system_message_dedup_and_replace():
    from fastapi.testclient import TestClient

    from apps.server.app import create_app
    from superlinear.engine.chat_types import Usage

    class FakeAdapter:
        def __init__(self):
            self._sessions: dict[str, dict[str, int]] = {}

        def create_session(self, *, cache_id: str, max_seq_len: int):
            if cache_id in self._sessions:
                raise ValueError("session exists")
            self._sessions[cache_id] = {"current_pos": 0, "max_seq_len": int(max_seq_len)}

        def get_session_info(self, cache_id: str):
            if cache_id not in self._sessions:
                raise KeyError(cache_id)
            return dict(self._sessions[cache_id])

        def list_sessions(self):
            return list(self._sessions.keys())

        def close_session(self, cache_id: str):
            self._sessions.pop(cache_id, None)

    class FakeEngine:
        def __init__(self):
            self.adapter = FakeAdapter()
            self.seen_systems: list[str] = []

        async def generate_chat(self, req):
            systems = [m for m in req.messages if getattr(m, "role", None) == "system"]
            assert len(systems) == 1
            self.seen_systems.append(str(getattr(systems[0], "content", "")))
            return {
                "content": "ok",
                "tool_calls": [],
                "finish_reason": "stop",
                "usage": Usage(prompt_tokens=1, completion_tokens=1),
                "timing": None,
            }

    engine = FakeEngine()
    app = create_app(engine=engine, model_id="superlinear-test")
    client = TestClient(app)

    session_id = "sess_system_dedup"
    resp = client.post("/v1/sessions", json={"session_id": session_id})
    assert resp.status_code == 200

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "superlinear-test",
            "session_id": session_id,
            "messages": [
                {"role": "system", "content": "A"},
                {"role": "user", "content": "u1"},
            ],
        },
    )
    assert resp.status_code == 200

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "superlinear-test",
            "session_id": session_id,
            "messages": [
                {"role": "system", "content": "B"},
                {"role": "user", "content": "u2"},
            ],
        },
    )
    assert resp.status_code == 200

    hist = client.get(f"/v1/sessions/{session_id}/history")
    assert hist.status_code == 200
    msgs = hist.json()["messages"]

    system_msgs = [m for m in msgs if m.get("role") == "system"]
    assert len(system_msgs) == 1
    assert system_msgs[0].get("content") == "B"
    assert engine.seen_systems == ["A", "B"]


@pytest.mark.anyio
async def test_http_concurrency_limit_429():
    import anyio
    import httpx

    from apps.server.app import create_app
    from superlinear.engine.chat_types import Usage

    started = anyio.Event()
    first_done = anyio.Event()
    first_status: int | None = None

    class FakeEngine:
        async def generate_chat(self, req):
            started.set()
            # Block long enough for another request to arrive.
            await anyio.sleep(0.3)
            return {
                "content": "ok",
                "tool_calls": [],
                "finish_reason": "stop",
                "usage": Usage(prompt_tokens=1, completion_tokens=1),
                "timing": None,
            }

    app = create_app(engine=FakeEngine(), model_id="superlinear-test", http_max_concurrency=1)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:

        async def _first_request() -> None:
            nonlocal first_status
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "superlinear-test", "messages": [{"role": "user", "content": "hi"}]},
            )
            first_status = resp.status_code
            first_done.set()

        async with anyio.create_task_group() as tg:
            tg.start_soon(_first_request)
            await started.wait()

            # Make a second request while the first is in-flight.
            resp2 = await client.post(
                "/v1/chat/completions",
                json={"model": "superlinear-test", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert resp2.status_code == 429

            await first_done.wait()

        assert first_status == 200
