import io

from apps.cli.client import iter_sse_json
from apps.cli.main import build_parser
from apps.cli.state import CliState, DocsWorkspaceState, load_state, save_state


def test_state_round_trip(tmp_path):
    path = tmp_path / "state.json"
    state = CliState(
        active_chat_session_id="chat-1",
        chat_checkpoint_snapshot_id="snap-1",
        chat_checkpoints={"chat-1": "snap-1", "chat-2": "snap-2"},
        docs_workspaces={
            "docs": DocsWorkspaceState(
                session_id="docs-1",
                phase="QA",
                base_snapshot_id="base-1",
                sources=[{"path": "a.txt", "bytes": 3, "sha256": "deadbeef", "added_at_unix_s": 123}],
                rag_backend="off",
                light_rag_enabled=False,
                light_rag_k=7,
                light_rag_total_chars=9000,
                light_rag_per_source_chars=2000,
                light_rag_debug=True,
            )
        },
    )
    save_state(state, path=path)
    loaded = load_state(path=path)
    assert loaded.active_chat_session_id == "chat-1"
    assert loaded.chat_checkpoint_snapshot_id == "snap-1"
    assert loaded.chat_checkpoints["chat-1"] == "snap-1"
    assert loaded.chat_checkpoints["chat-2"] == "snap-2"
    assert loaded.docs_workspaces["docs"].session_id == "docs-1"
    assert loaded.docs_workspaces["docs"].phase == "QA"
    assert loaded.docs_workspaces["docs"].base_snapshot_id == "base-1"
    assert loaded.docs_workspaces["docs"].sources[0]["path"] == "a.txt"
    assert loaded.docs_workspaces["docs"].light_rag_enabled is False
    assert loaded.docs_workspaces["docs"].light_rag_k == 7
    assert loaded.docs_workspaces["docs"].light_rag_total_chars == 9000
    assert loaded.docs_workspaces["docs"].light_rag_per_source_chars == 2000
    assert loaded.docs_workspaces["docs"].light_rag_debug is True


def test_sse_json_stops_on_done():
    stream = io.BytesIO(
        b"data: {\"x\": 1}\n\n"
        b"data: {\"y\": 2}\n\n"
        b"data: [DONE]\n\n"
        b"data: {\"z\": 3}\n\n"
    )
    events = list(iter_sse_json(stream))
    assert events == [{"x": 1}, {"y": 2}]


def test_parser_global_url():
    parser = build_parser()
    args = parser.parse_args(["--url", "http://example.invalid:8000", "chat"])
    assert args.url == "http://example.invalid:8000"
    assert args.command == "chat"


def test_parser_server_start():
    parser = build_parser()
    args = parser.parse_args(
        ["--url", "http://127.0.0.1:8000", "server", "start", "--model", "m", "--port", "9000"]
    )
    assert args.command == "server"
    assert args.server_cmd == "start"
    assert args.model == "m"
    assert args.port == 9000


def test_parser_session_ls_json():
    parser = build_parser()
    args = parser.parse_args(["session", "ls", "--json"])
    assert args.command == "session"
    assert args.session_cmd == "ls"
    assert args.json is True


def test_parser_session_close_force():
    parser = build_parser()
    args = parser.parse_args(["session", "close", "sess-1", "--force"])
    assert args.command == "session"
    assert args.session_cmd == "close"
    assert args.session_id == "sess-1"
    assert args.force is True


def test_parser_snapshot_load_force():
    parser = build_parser()
    args = parser.parse_args(["snapshot", "load", "snap-1", "--session", "sess-1", "--force", "--json"])
    assert args.command == "snapshot"
    assert args.snapshot_cmd == "load"
    assert args.snapshot_id == "snap-1"
    assert args.session_id == "sess-1"
    assert args.force is True
    assert args.json is True


def test_parser_chat_new():
    parser = build_parser()
    args = parser.parse_args(["chat", "--new"])
    assert args.command == "chat"
    assert args.new is True


def test_parser_chat_session_attach():
    parser = build_parser()
    args = parser.parse_args(["chat", "--session", "chat-1"])
    assert args.command == "chat"
    assert args.session == "chat-1"


def test_parser_docs_name_required():
    parser = build_parser()
    args = parser.parse_args(["docs", "mydocs"])
    assert args.command == "docs"
    assert args.name == "mydocs"
