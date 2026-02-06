"""HTTP client wrapper for communicating with the Superlinear HTTP server.

This module provides small, dependency-free primitives for JSON requests and SSE streaming.
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, BinaryIO, Iterator


DEFAULT_URL = "http://127.0.0.1:8787"


@dataclass(frozen=True)
class HttpError(RuntimeError):
    message: str
    url: str | None = None
    status_code: int | None = None
    body: str | None = None

    def __str__(self) -> str:  # pragma: no cover
        parts = [self.message]
        if self.status_code is not None:
            parts.append(f"status={self.status_code}")
        if self.url:
            parts.append(f"url={self.url}")
        if self.body:
            parts.append(f"body={self.body}")
        return " ".join(parts)


def _join_url(base_url: str, path: str) -> str:
    if not path.startswith("/"):
        path = "/" + path
    # Ensure base_url ends with "/" so urljoin doesn't drop the path.
    return urllib.parse.urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))


def iter_sse_data(stream: BinaryIO) -> Iterator[str]:
    """Yield decoded SSE `data:` payloads, one per event (without the trailing blank line)."""
    data_lines: list[str] = []
    for raw in stream:
        try:
            line = raw.decode("utf-8", errors="replace")
        except Exception:
            continue
        line = line.rstrip("\r\n")
        if not line:
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
            continue

    if data_lines:
        yield "\n".join(data_lines)


def iter_sse_json(stream: BinaryIO) -> Iterator[dict[str, Any]]:
    """Yield JSON-decoded SSE `data:` payloads; stops on `[DONE]`."""
    for payload in iter_sse_data(stream):
        if payload == "[DONE]":
            return
        yield json.loads(payload)


class SuperlinearClient:
    def __init__(self, *, base_url: str = DEFAULT_URL, timeout_s: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)

    def request_json(
        self,
        method: str,
        path: str,
        *,
        payload: Any | None = None,
        timeout_s: float | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        url = _join_url(self.base_url, path)

        body: bytes | None = None
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        req = urllib.request.Request(url=url, method=method.upper(), data=body)
        req.add_header("Accept", "application/json")
        if payload is not None:
            req.add_header("Content-Type", "application/json")
        if headers:
            for k, v in headers.items():
                req.add_header(k, v)

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s if timeout_s is None else timeout_s) as resp:
                raw = resp.read()
                try:
                    return json.loads(raw.decode("utf-8"))
                except Exception as exc:
                    raise HttpError(
                        "Invalid JSON response",
                        url=url,
                        status_code=getattr(resp, "status", None),
                        body=raw.decode("utf-8", errors="replace"),
                    ) from exc
        except urllib.error.HTTPError as exc:
            body_text: str | None
            try:
                body_text = exc.read().decode("utf-8", errors="replace")
            except Exception:
                body_text = None
            raise HttpError(
                "HTTP error",
                url=url,
                status_code=getattr(exc, "code", None),
                body=body_text,
            ) from exc
        except urllib.error.URLError as exc:
            raise HttpError("Failed to reach server", url=url) from exc
        except socket.timeout as exc:
            raise HttpError("Request timed out", url=url) from exc

    def request_sse(
        self,
        method: str,
        path: str,
        *,
        payload: Any | None = None,
        timeout_s: float | None = None,
        headers: dict[str, str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        url = _join_url(self.base_url, path)
        body: bytes | None = None
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        req = urllib.request.Request(url=url, method=method.upper(), data=body)
        req.add_header("Accept", "text/event-stream")
        if payload is not None:
            req.add_header("Content-Type", "application/json")
        if headers:
            for k, v in headers.items():
                req.add_header(k, v)

        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout_s if timeout_s is None else timeout_s)
        except urllib.error.HTTPError as exc:
            body_text: str | None
            try:
                body_text = exc.read().decode("utf-8", errors="replace")
            except Exception:
                body_text = None
            raise HttpError(
                "HTTP error",
                url=url,
                status_code=getattr(exc, "code", None),
                body=body_text,
            ) from exc
        except urllib.error.URLError as exc:
            raise HttpError("Failed to reach server", url=url) from exc
        except socket.timeout as exc:
            raise HttpError("Request timed out", url=url) from exc

        try:
            yield from iter_sse_json(resp)
        finally:
            try:
                resp.close()
            except Exception:
                pass

    def health(self) -> dict[str, Any]:
        result = self.request_json("GET", "/health", timeout_s=min(self.timeout_s, 5.0))
        if not isinstance(result, dict):
            raise HttpError("Invalid /health response", url=_join_url(self.base_url, "/health"))
        return result

    def list_models(self) -> list[dict[str, Any]]:
        result = self.request_json("GET", "/v1/models")
        if not isinstance(result, dict) or not isinstance(result.get("data"), list):
            raise HttpError("Invalid /v1/models response", url=_join_url(self.base_url, "/v1/models"))
        out: list[dict[str, Any]] = []
        for item in result["data"]:
            if isinstance(item, dict):
                out.append(item)
        return out
