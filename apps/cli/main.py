"""`spl` — Superlinear CLI (HTTP client).

This is the CLI entrypoint. Run from source with:
  `python -m apps.cli.main --help`
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from apps.cli.client import DEFAULT_URL
from apps.cli.chat_repl import chat_repl, DEFAULT_SYSTEM_PROMPT
from apps.cli.docs_repl import docs_repl
from apps.cli.server_cmds import ServerCommandError, server_connect, server_start, server_status, server_stop
from apps.cli.state import load_state
from apps.cli.session_cmds import (
    SessionCommandError,
    session_close,
    session_close_all,
    session_history,
    session_info,
    session_ls,
    unified_ls,
    unified_rm,
)
from apps.cli.snapshot_cmds import (
    SnapshotCommandError,
    snapshot_load,
    snapshot_ls,
    snapshot_rm,
    snapshot_rm_all,
    snapshot_save,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="spl", description="Superlinear CLI (HTTP client)")
    p.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Server base URL (default: %(default)s)",
    )

    sub = p.add_subparsers(dest="command")

    server = sub.add_parser("server", help="Manage local inference server")
    server_sub = server.add_subparsers(dest="server_cmd")  # Not required - defaults to status
    server_start_p = server_sub.add_parser("start", help="Start local server")
    server_start_p.add_argument("--model", required=True, help="Model path or HF repo id")
    server_start_p.add_argument("--host", help="Bind host (default: derived from --url)")
    server_start_p.add_argument("--port", type=int, help="Bind port (default: derived from --url)")
    server_start_p.add_argument(
        "--attn-implementation",
        type=str,
        default=None,
        help="Attention implementation (maps to apps.server.main --attn-implementation)",
    )
    server_start_p.add_argument(
        "--decode-kernel",
        type=str,
        default=None,
        help="Decode kernel (maps to apps.server.main --decode-kernel)",
    )
    server_start_p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device / device_map (maps to apps.server.main --device)",
    )
    server_start_p.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Torch dtype: float16|bfloat16|float32 (maps to apps.server.main --dtype)",
    )
    server_start_p.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size for chunked prefill (default: 8192)",
    )
    server_start_p.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=None,
        help="Server-side hard cap for prompt length (maps to apps.server.main --max-prompt-tokens)",
    )
    server_start_p.add_argument(
        "--disable-cuda-graph",
        action="store_true",
        help="Disable CUDA graphs (maps to apps.server.main --disable-cuda-graph)",
    )
    server_start_p.add_argument(
        "--disable-shared-fused-moe",
        action="store_true",
        help="Disable shared fused MoE (maps to apps.server.main --disable-shared-fused-moe)",
    )
    server_start_p.add_argument(
        "--foreground",
        action="store_true",
        help="Run in foreground (default: detached with logs to ~/.config/spl/server)",
    )

    server_sub.add_parser("status", help="Check server status")
    server_stop_p = server_sub.add_parser("stop", help="Stop local server")
    server_stop_p.add_argument(
        "--force",
        action="store_true",
        help="Stop even if active sessions exist",
    )
    server_connect_p = server_sub.add_parser("connect", help="Connect to a remote server")
    server_connect_p.add_argument("server_url", help="Server URL (e.g., http://gpu-server:8787)")

    # Unified list command
    ls_p = sub.add_parser("ls", help="List sessions and snapshots")
    ls_p.add_argument("--json", action="store_true", help="Machine-readable JSON output")

    rm_p = sub.add_parser("rm", help="Remove sessions and snapshots")
    rm_p.add_argument("ids", nargs="+", help="IDs to remove (chat-*, docs-*, snap-*)")

    session = sub.add_parser("session", help="Manage live sessions")
    session_sub = session.add_subparsers(dest="session_cmd", required=True)
    session_ls_p = session_sub.add_parser("ls", help="List active sessions")
    session_ls_p.add_argument("--json", action="store_true", help="Machine-readable JSON output")
    session_info_p = session_sub.add_parser("info", help="Show session info")
    session_info_p.add_argument("session_id", help="Session id")
    session_info_p.add_argument("--json", action="store_true", help="Machine-readable JSON output")
    session_rm_p = session_sub.add_parser("rm", help="Remove session(s)")
    session_rm_p.add_argument("session_ids", nargs="*", help="Session id(s) to remove")
    session_rm_p.add_argument("--all", action="store_true", dest="remove_all", help="Remove all sessions")

    # Back-compat (hidden from help): `spl session close <id> [--force]`
    session_close_p = session_sub.add_parser("close", help="(deprecated; use `rm`)")
    session_close_p.add_argument("session_id", help=argparse.SUPPRESS)
    session_close_p.add_argument("--force", action="store_true", help=argparse.SUPPRESS)
    # Hide `close` from `spl session --help` while still allowing it to parse.
    try:
        session_sub._choices_actions = [
            a for a in session_sub._choices_actions if getattr(a, "dest", None) != "close"
        ]
        session_sub.metavar = "{ls,info,rm,history}"
    except Exception:
        pass
    session_hist_p = session_sub.add_parser("history", help="Print session transcript")
    session_hist_p.add_argument("session_id", help="Session id")
    session_hist_p.add_argument("--tail", type=int, help="Only show the last N messages")
    session_hist_p.add_argument("--json", action="store_true", help="Machine-readable JSON output")

    snapshot = sub.add_parser("snapshot", help="Manage durable snapshots")
    snapshot_sub = snapshot.add_subparsers(dest="snapshot_cmd", required=True)
    snapshot_ls_p = snapshot_sub.add_parser("ls", help="List snapshots")
    snapshot_ls_p.add_argument("--json", action="store_true", help="Machine-readable JSON output")
    snapshot_save_p = snapshot_sub.add_parser("save", help="Save a session to a snapshot")
    snapshot_save_p.add_argument("--session", dest="session_id", required=True, help="Session id to save")
    snapshot_save_p.add_argument("--title", help="Optional snapshot title")
    snapshot_save_p.add_argument("--json", action="store_true", help="Machine-readable JSON output")
    snapshot_load_p = snapshot_sub.add_parser("load", help="Load snapshot into a session")
    snapshot_load_p.add_argument("snapshot_id", help="Snapshot id")
    snapshot_load_p.add_argument("--session", dest="session_id", help="Target session id (default: new)")
    snapshot_load_p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite target session if it already exists",
    )
    snapshot_load_p.add_argument("--json", action="store_true", help="Machine-readable JSON output")
    snapshot_rm_p = snapshot_sub.add_parser("rm", help="Delete snapshot(s)")
    snapshot_rm_p.add_argument("snapshot_ids", nargs="*", help="Snapshot id(s) to delete")
    snapshot_rm_p.add_argument("--all", action="store_true", dest="remove_all", help="Delete all snapshots")
    snapshot_rm_p.add_argument("--json", action="store_true", help="Machine-readable JSON output")

    chat = sub.add_parser("chat", help="Chat REPL")
    chat.add_argument("--new", action="store_true", help="Start a new chat workspace")
    chat.add_argument("--session", help="Attach to a specific session id")
    chat.add_argument(
        "--max-seq-len",
        type=int,
        default=1_048_576,
        help="Session context length (default: 1048576)",
    )
    chat.add_argument(
        "--think-budget",
        type=int,
        default=8192,
        help="Enable thinking mode with this token budget (0 disables thinking)",
    )
    chat.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (default: 0.1)",
    )
    chat.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling (default: 0.95)",
    )
    chat.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Custom system prompt (use --no-system-prompt to disable)",
    )
    chat.add_argument(
        "--no-system-prompt",
        action="store_true",
        help="Disable the default system prompt",
    )

    docs = sub.add_parser("docs", help="Docs REPL")
    docs.add_argument("name", help="Docs workspace name")
    docs.add_argument(
        "-l",
        "--load",
        type=str,
        metavar="SNAPSHOT_ID",
        default=None,
        help="Load from a snapshot (e.g., from `spl snapshot ls`)",
    )
    docs.add_argument(
        "--max-seq-len",
        type=int,
        default=1_048_576,
        help="Session context length (max_seq_len). If the workspace session already exists, the CLI will try to resize it upward.",
    )
    docs.add_argument(
        "--think-budget",
        type=int,
        default=32768,
        help="Enable thinking mode with this token budget (0 disables thinking)",
    )
    docs.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature (default: 0.3)",
    )
    docs.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling (default: 0.95)",
    )
    docs.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Custom system prompt (use --no-system-prompt to disable)",
    )
    docs.add_argument(
        "--no-system-prompt",
        action="store_true",
        help="Disable the default system prompt",
    )

    return p


def _not_implemented(what: str) -> int:
    print(f"{what} is not implemented yet (CLI foundation only).", file=sys.stderr)
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    if argv is None:
        argv_list = list(sys.argv[1:])
    else:
        argv_list = list(argv)

    url_explicit = any(a == "--url" or a.startswith("--url=") for a in argv_list)

    args = parser.parse_args(argv_list)

    # Resolve server URL: --url flag > saved state > default
    # Check if --url was explicitly provided by comparing to the default
    if args.url == DEFAULT_URL:
        state = load_state()
        if state.server_url:
            args.url = state.server_url

    # `spl` defaults to `spl chat`.
    command = args.command or "chat"

    if command == "chat":
        # Determine system prompt: custom > disabled > default
        if getattr(args, "no_system_prompt", False):
            system_prompt = None
        elif getattr(args, "system_prompt", None):
            system_prompt = args.system_prompt
        else:
            system_prompt = DEFAULT_SYSTEM_PROMPT

        return chat_repl(
            url=args.url,
            new=bool(getattr(args, "new", False)),
            session=getattr(args, "session", None),
            max_seq_len=getattr(args, "max_seq_len", 1_048_576),
            think_budget=None if int(getattr(args, "think_budget", 0)) <= 0 else int(getattr(args, "think_budget", 0)),
            temperature=float(getattr(args, "temperature", 0.1)),
            top_p=float(getattr(args, "top_p", 0.95)),
            system_prompt=system_prompt,
        )
    if command == "docs":
        # Determine system prompt for docs: custom > disabled > default (None means use built-in)
        if getattr(args, "no_system_prompt", False):
            docs_system_prompt: str | None = ""
        elif getattr(args, "system_prompt", None):
            docs_system_prompt = args.system_prompt
        else:
            docs_system_prompt = None  # Use built-in default

        return docs_repl(
            url=args.url,
            name=args.name,
            load_snapshot_id=getattr(args, "load", None),
            max_seq_len=getattr(args, "max_seq_len", 1_048_576),
            think_budget=None if int(getattr(args, "think_budget", 0)) <= 0 else int(getattr(args, "think_budget", 0)),
            temperature=float(getattr(args, "temperature", 0.3)),
            top_p=float(getattr(args, "top_p", 0.95)),
            system_prompt=docs_system_prompt,
        )
    if command == "ls":
        try:
            # Get active session info from state
            state = load_state()
            return unified_ls(
                url=args.url,
                json_output=bool(getattr(args, "json", False)),
                active_chat_session_id=state.active_chat_session_id,
                docs_workspaces=state.docs_workspaces,
            )
        except SessionCommandError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    if command == "rm":
        try:
            return unified_rm(
                url=args.url,
                ids=args.ids,
                allow_remote_snapshot_delete=url_explicit,
            )
        except SessionCommandError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    if command == "server":
        try:
            server_cmd = args.server_cmd or "status"
            if server_cmd == "status":
                result = server_status(url=args.url)
                if not args.server_cmd:
                    # Show hints when defaulting to status
                    print()
                    print("commands: spl server start --model <path>")
                    print("          spl server stop")
                    print("          spl server connect <url>")
                return result
            if server_cmd == "start":
                return server_start(
                    url=args.url,
                    model=args.model,
                    host=args.host,
                    port=args.port,
                    chunk_size=getattr(args, "chunk_size", None),
                    attn_implementation=getattr(args, "attn_implementation", None),
                    decode_kernel=getattr(args, "decode_kernel", None),
                    device=getattr(args, "device", None),
                    dtype=getattr(args, "dtype", None),
                    max_prompt_tokens=getattr(args, "max_prompt_tokens", None),
                    disable_cuda_graph=bool(getattr(args, "disable_cuda_graph", False)),
                    disable_shared_fused_moe=bool(getattr(args, "disable_shared_fused_moe", False)),
                    foreground=bool(args.foreground),
                )
            if server_cmd == "stop":
                return server_stop(url=args.url, force=bool(args.force))
            if server_cmd == "connect":
                return server_connect(url=args.server_url)
            parser.error(f"Unknown server subcommand: {server_cmd!r}")
            return 2
        except ServerCommandError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    if command == "session":
        try:
            if args.session_cmd == "ls":
                return session_ls(url=args.url, json_output=bool(args.json))
            if args.session_cmd == "info":
                return session_info(url=args.url, session_id=args.session_id, json_output=bool(args.json))
            if args.session_cmd == "rm":
                if getattr(args, "remove_all", False):
                    return session_close_all(url=args.url)
                if not args.session_ids:
                    print("error: provide session IDs or use --all", file=sys.stderr)
                    return 2
                return session_close(url=args.url, session_ids=args.session_ids)
            if args.session_cmd == "close":
                # Legacy alias; `--force` is accepted but ignored (MVP smoothness).
                return session_close(url=args.url, session_ids=[args.session_id])
            if args.session_cmd == "history":
                return session_history(
                    url=args.url,
                    session_id=args.session_id,
                    tail=args.tail,
                    json_output=bool(args.json),
                )
            parser.error(f"Unknown session subcommand: {args.session_cmd!r}")
            return 2
        except SessionCommandError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    if command == "snapshot":
        try:
            if args.snapshot_cmd == "ls":
                return snapshot_ls(url=args.url, json_output=bool(args.json))
            if args.snapshot_cmd == "save":
                return snapshot_save(
                    url=args.url,
                    session_id=args.session_id,
                    title=args.title,
                    json_output=bool(args.json),
                )
            if args.snapshot_cmd == "load":
                return snapshot_load(
                    url=args.url,
                    snapshot_id=args.snapshot_id,
                    session_id=args.session_id,
                    force=bool(args.force),
                    json_output=bool(args.json),
                )
            if args.snapshot_cmd == "rm":
                if args.remove_all:
                    return snapshot_rm_all(
                        url=args.url,
                        json_output=bool(args.json),
                        allow_remote_delete=url_explicit,
                    )
                if not args.snapshot_ids:
                    parser.error("snapshot rm: either provide snapshot IDs or use --all")
                return snapshot_rm(
                    url=args.url,
                    snapshot_ids=args.snapshot_ids,
                    json_output=bool(args.json),
                    allow_remote_delete=url_explicit,
                )
            parser.error(f"Unknown snapshot subcommand: {args.snapshot_cmd!r}")
            return 2
        except SnapshotCommandError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    parser.error(f"Unknown command: {command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
