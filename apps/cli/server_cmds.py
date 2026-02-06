from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.parse
from dataclasses import asdict, dataclass
from pathlib import Path

from apps.cli.client import DEFAULT_URL, HttpError, SuperlinearClient
from apps.cli.state import CliState, config_dir, load_state, save_state


class ServerCommandError(RuntimeError):
    pass


@dataclass(frozen=True)
class ServerInstancePaths:
    pid_path: Path
    log_path: Path
    meta_path: Path


@dataclass(frozen=True)
class ServerInstanceMeta:
    pid: int
    url: str
    host: str
    port: int
    model: str
    log_path: str
    started_at_unix_s: int


def _normalize_base_url(url: str) -> str:
    url = url.strip()
    if not url:
        raise ServerCommandError("Server URL is empty")
    if "://" not in url:
        url = "http://" + url
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ServerCommandError(f"Unsupported URL scheme: {parsed.scheme!r}")
    if not parsed.netloc:
        raise ServerCommandError(f"Invalid server URL: {url!r}")
    if parsed.path not in {"", "/"} or parsed.params or parsed.query or parsed.fragment:
        raise ServerCommandError("Server URL must not include a path/query/fragment")

    host = parsed.hostname
    port = parsed.port
    if host is None:
        raise ServerCommandError(f"Invalid server host in URL: {url!r}")
    if host.lower() == "localhost":
        host = "127.0.0.1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = f"{host}:{port}" if port is not None else host
    return f"{parsed.scheme}://{netloc}".rstrip("/")


def _parse_host_port(url: str) -> tuple[str, int]:
    parsed = urllib.parse.urlparse(_normalize_base_url(url))
    host = parsed.hostname
    port = parsed.port
    if host is None:
        raise ServerCommandError(f"Invalid server host in URL: {url!r}")
    if port is None:
        port = 443 if parsed.scheme == "https" else 8787
    return host, int(port)


def _server_dir() -> Path:
    return config_dir() / "server"


def _instance_paths(*, host: str, port: int) -> ServerInstancePaths:
    host = host.strip()
    host_l = host.lower()
    if host_l == "localhost":
        host = "127.0.0.1"
    else:
        host = host_l

    safe_host = host.replace(":", "_").replace("/", "_")
    prefix = f"{safe_host}_{int(port)}"
    base = _server_dir()
    return ServerInstancePaths(
        pid_path=base / f"{prefix}.pid",
        log_path=base / f"{prefix}.log",
        meta_path=base / f"{prefix}.json",
    )


def _is_pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _read_pid(path: Path) -> int | None:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except Exception:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def _write_atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            delete=False,
            prefix=path.name + ".",
            suffix=".tmp",
        ) as f:
            f.write(text)
            tmp_path = Path(f.name)
        tmp_path.replace(path)
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


def _port_open(host: str, port: int, *, timeout_s: float = 0.2) -> bool:
    check_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    try:
        with socket.create_connection((check_host, int(port)), timeout=timeout_s):
            return True
    except OSError:
        return False


def _get_model_id(client: SuperlinearClient) -> str | None:
    try:
        models = client.list_models()
    except HttpError:
        return None
    if not models:
        return None
    first = models[0]
    if isinstance(first, dict) and isinstance(first.get("id"), str):
        return first["id"]
    return None


def server_status(*, url: str) -> int:
    url = _normalize_base_url(url)
    client = SuperlinearClient(base_url=url, timeout_s=5.0)

    try:
        client.health()
    except HttpError:
        host, port = _parse_host_port(url)
        paths = _instance_paths(host=host, port=port)
        pid = _read_pid(paths.pid_path)
        if pid is not None and not _is_pid_running(pid):
            try:
                paths.pid_path.unlink()
            except Exception:
                pass
            pid = None

        print(f"stopped url={url}")
        if pid is not None:
            print(f"pid={pid} (not responding)")
        return 1

    model_id = _get_model_id(client) or "unknown"

    host, port = _parse_host_port(url)
    paths = _instance_paths(host=host, port=port)
    pid = _read_pid(paths.pid_path)
    if pid is not None and not _is_pid_running(pid):
        pid = None

    line = f"running url={url} model={model_id}"
    if pid is not None:
        line += f" pid={pid}"
    print(line)
    return 0


def server_start(
    *,
    url: str,
    model: str,
    host: str | None = None,
    port: int | None = None,
    chunk_size: int | None = None,
    attn_implementation: str | None = None,
    decode_kernel: str | None = None,
    device: str | None = None,
    dtype: str | None = None,
    max_prompt_tokens: int | None = None,
    disable_cuda_graph: bool = False,
    disable_shared_fused_moe: bool = False,
    foreground: bool = False,
) -> int:
    url = _normalize_base_url(url)
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "http":
        raise ServerCommandError("spl server start only supports http URLs (use --url http://...)")
    url_host, url_port = _parse_host_port(url)

    target_host = url_host
    target_port = int(port if port is not None else url_port)
    if target_port <= 0 or target_port > 65535:
        raise ServerCommandError(f"Invalid port: {target_port}")

    bind_host = host or url_host
    bind_port = target_port

    base_url = f"http://{target_host}:{target_port}"

    if _port_open(target_host, target_port):
        client = SuperlinearClient(base_url=base_url, timeout_s=2.0)
        try:
            client.health()
        except HttpError:
            raise ServerCommandError(
                f"Port {target_port} is in use at host {target_host!r} (not a Superlinear server). "
                "Use --port or stop the conflicting process."
            )

        model_id = _get_model_id(client)
        if model_id is None:
            raise ServerCommandError(
                f"Port {target_port} is in use at host {target_host!r} (not a Superlinear server). "
                "Use --port or stop the conflicting process."
            )
        print(f"already running url={base_url} model={model_id}")
        return 0

    paths = _instance_paths(host=target_host, port=target_port)
    paths.pid_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "apps.server.main",
        "--model",
        model,
        "--host",
        bind_host,
        "--port",
        str(bind_port),
    ]

    if chunk_size is not None:
        cmd.extend(["--chunk-size", str(int(chunk_size))])
    if device is not None and str(device).strip():
        cmd.extend(["--device", str(device).strip()])
    if dtype is not None and str(dtype).strip():
        cmd.extend(["--dtype", str(dtype).strip()])
    if attn_implementation is not None and str(attn_implementation).strip():
        cmd.extend(["--attn-implementation", str(attn_implementation).strip()])
    if decode_kernel is not None and str(decode_kernel).strip():
        cmd.extend(["--decode-kernel", str(decode_kernel).strip()])

    if max_prompt_tokens is not None:
        cmd.extend(["--max-prompt-tokens", str(int(max_prompt_tokens))])

    if disable_cuda_graph:
        cmd.append("--disable-cuda-graph")
    if disable_shared_fused_moe:
        cmd.append("--disable-shared-fused-moe")

    expected_model_id = os.path.basename(model.rstrip("/")) or "superlinear"

    if foreground:
        print(f"starting (foreground) url={base_url} model={expected_model_id}")
        return subprocess.call(cmd)

    with open(paths.log_path, "ab") as logf:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=logf,
            stderr=logf,
            start_new_session=True,
        )

    # Quick sanity: if the process exited immediately, surface a helpful error.
    time.sleep(0.2)
    if proc.poll() is not None:
        raise ServerCommandError(
            f"Server process exited immediately (code={proc.returncode}). Check logs: {paths.log_path}"
        )

    meta = ServerInstanceMeta(
        pid=int(proc.pid),
        url=base_url,
        host=target_host,
        port=target_port,
        model=model,
        log_path=str(paths.log_path),
        started_at_unix_s=int(time.time()),
    )
    _write_atomic_text(paths.pid_path, f"{proc.pid}\n")
    _write_atomic_text(paths.meta_path, json.dumps(asdict(meta), ensure_ascii=False, indent=2) + "\n")

    print(f"starting url={base_url} model={expected_model_id} logs={paths.log_path}")

    def _read_new_log_lines(*, fp, pos: int) -> tuple[int, list[str]]:
        try:
            fp.seek(pos)
            chunk = fp.read()
        except Exception:
            return pos, []
        if not chunk:
            return fp.tell(), []
        # Splitlines keeps output readable even if the server writes partial lines.
        lines = chunk.splitlines()
        return fp.tell(), [str(l) for l in lines if str(l).strip()]

    def _is_startup_relevant(line: str) -> bool:
        # Keep the CLI output high-signal; the full log remains in the log file.
        s = line.strip()
        if not s:
            return False
        if s.startswith("[server]") or s.startswith("[warmup]"):
            return True
        if "Loading checkpoint shards" in s:
            return True
        if s.startswith("Traceback") or "ERROR" in s or "Exception" in s:
            return True
        return False

    # Wait for server to become ready
    client = SuperlinearClient(base_url=base_url, timeout_s=5.0)
    spinner = ["|", "/", "-", "\\"]
    spin_idx = 0
    poll_interval = 1.0
    max_wait_s = 600  # 10 minutes max wait
    start_wait = time.monotonic()

    log_fp = None
    log_pos = 0
    printed_any_logs = False
    try:
        log_fp = open(paths.log_path, "r", encoding="utf-8", errors="replace")
        # Start from the current end; we only want new lines from this run.
        log_fp.seek(0, os.SEEK_END)
        log_pos = log_fp.tell()
    except Exception:
        log_fp = None

    try:
        while True:
            elapsed = time.monotonic() - start_wait
            if elapsed > max_wait_s:
                print(f"\rtimeout after {int(elapsed)}s waiting for server. check logs: {paths.log_path}", file=sys.stderr)
                return 1

            # Check if process died
            if proc.poll() is not None:
                print(f"\rserver process exited (code={proc.returncode}). check logs: {paths.log_path}", file=sys.stderr)
                return 1

            # Try health check
            try:
                client.health()
                # Server is ready!
                print(f"\rserver ready ({int(elapsed)}s)                    ")
                print(f"openai api-compatible endpoint: {base_url}/v1/chat/completions")
                return 0
            except Exception:
                pass

            # Stream high-signal server logs during startup so users see warmup/load phases
            # even when running detached.
            if log_fp is not None:
                log_pos, new_lines = _read_new_log_lines(fp=log_fp, pos=log_pos)
                out_lines = [ln for ln in new_lines if _is_startup_relevant(ln)]
                if out_lines:
                    # Clear spinner line before printing logs.
                    sys.stdout.write("\r" + (" " * 120) + "\r")
                    for ln in out_lines[-20:]:
                        # Avoid flooding if the server emits many lines at once.
                        print(ln)
                    printed_any_logs = True

            # Show spinner
            sys.stdout.write(f"\rwaiting for model to load... {spinner[spin_idx]} ({int(elapsed)}s)")
            sys.stdout.flush()
            spin_idx = (spin_idx + 1) % len(spinner)
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("\n(cancelled) server still starting in background")
        print(f"check: `spl --url {base_url} server status`")
        return 0
    finally:
        try:
            if log_fp is not None:
                log_fp.close()
        except Exception:
            pass


def server_stop(*, url: str, force: bool = False) -> int:
    url = _normalize_base_url(url)
    host, port = _parse_host_port(url)
    paths = _instance_paths(host=host, port=port)
    pid = _read_pid(paths.pid_path)

    client = SuperlinearClient(base_url=url, timeout_s=5.0)
    reachable = True
    try:
        client.health()
    except HttpError:
        reachable = False

    if reachable:
        try:
            payload = client.request_json("GET", "/v1/sessions", timeout_s=5.0)
        except HttpError as exc:
            raise ServerCommandError(str(exc)) from exc

        sessions = payload.get("sessions") if isinstance(payload, dict) else None
        if not isinstance(sessions, list):
            sessions = []
        active_session_ids = [s for s in sessions if isinstance(s, str)]
        if active_session_ids and not force:
            msg = (
                f"refusing to stop url={url}: {len(active_session_ids)} active session(s) exist; "
                "in-memory sessions may be lost on stop.\n"
                "next steps: `spl session ls`, then `spl snapshot save --session <id>` or `spl session rm <id>`, "
                "then retry; rerun with `--force` to stop anyway."
            )
            print(msg, file=sys.stderr)
            return 2

    if pid is None:
        if reachable:
            print(
                f"cannot stop url={url}: server is reachable but no managed PID file found at {paths.pid_path}",
                file=sys.stderr,
            )
            return 1
        print(f"stopped url={url} (not running)")
        return 0

    if not _is_pid_running(pid):
        try:
            paths.pid_path.unlink()
        except Exception:
            pass
        print(f"stopped url={url} (stale pid={pid})")
        return 0

    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if not _is_pid_running(pid):
            break
        time.sleep(0.1)

    if _is_pid_running(pid):
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.1)

    try:
        paths.pid_path.unlink()
    except Exception:
        pass

    print(f"stopped url={url} pid={pid}")
    return 0


def server_connect(*, url: str) -> int:
    """Save a server URL as the default for future CLI commands."""
    url = _normalize_base_url(url)

    # Validate the server is reachable
    client = SuperlinearClient(base_url=url, timeout_s=10.0)
    try:
        client.health()
    except Exception as exc:
        print(f"error: cannot reach server at {url}: {exc}", file=sys.stderr)
        return 1

    # Save to state
    state = load_state()
    state.server_url = url
    save_state(state)

    print(f"connected to {url}")
    print("future commands will use this server by default")
    return 0
