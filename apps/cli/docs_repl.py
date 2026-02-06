from __future__ import annotations

import atexit
import glob
import hashlib
import json
import os
import shutil
import shlex
import sys
import time
import textwrap
from pathlib import Path
import re
from typing import Any

# Enable readline for arrow keys, history navigation, and line editing.
try:
    import readline
except ImportError:
    readline = None  # type: ignore[assignment]  # Windows fallback

from apps.cli.bm25_rag import Bm25RagConfig, Bm25RagRetriever
from apps.cli.client import HttpError, SuperlinearClient
from apps.cli.light_rag import LightRagConfig, LightRagRetriever, tokenize_query_terms
from apps.cli.local_snapshots import delete_local_snapshot, list_local_snapshots
from apps.cli.locks import AlreadyLockedError, SessionLock
from apps.cli.output import format_table
from apps.cli.state import DocsWorkspaceState, load_state, save_state


def _docs_history_file_path() -> Path:
    return Path.home() / ".config" / "spl" / "docs_history"


def _setup_readline_history() -> None:
    """Set up persistent command history for the docs REPL."""
    if readline is None:
        return
    history_file = _docs_history_file_path()
    history_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        readline.read_history_file(history_file)
    except FileNotFoundError:
        pass
    readline.set_history_length(1000)
    atexit.register(readline.write_history_file, history_file)


# Commands for tab completion
_DOCS_COMMANDS = [
    "/help", "/exit", "/clear", "/history", "/ls", "/rm", "/head", "/tail",
    "/show", "/add", "/sources", "/rag", "/reset", "/stats", "/save", "/load", "/info",
]


def _setup_completer() -> None:
    """Set up tab completion for REPL commands."""
    if readline is None:
        return

    def completer(text: str, state: int) -> str | None:
        if text.startswith("/"):
            matches = [cmd for cmd in _DOCS_COMMANDS if cmd.startswith(text)]
        else:
            matches = []
        return matches[state] if state < len(matches) else None

    readline.set_completer(completer)
    readline.set_completer_delims(" \t\n")
    readline.parse_and_bind("tab: complete")


def _cmd_history(n: int = 20) -> None:
    """Show the last n entries from readline input history."""
    if readline is None:
        print("history not available (readline not loaded)", file=sys.stderr)
        return
    length = readline.get_current_history_length()
    if length == 0:
        print("(no history)")
        return
    start = max(1, length - n + 1)
    for i in range(start, length + 1):
        item = readline.get_history_item(i)
        if item:
            print(f"{i:4d}  {item}")


def _cmd_history_clear() -> None:
    """Clear readline input history (both in-memory and on disk)."""
    if readline is None:
        print("history not available (readline not loaded)", file=sys.stderr)
        return
    try:
        readline.clear_history()
    except Exception as exc:
        print(f"failed to clear history: {exc}", file=sys.stderr)
        return

    history_file = _docs_history_file_path()
    try:
        history_file.parent.mkdir(parents=True, exist_ok=True)
        readline.write_history_file(history_file)
    except Exception:
        pass
    print("cleared input history")


class DocsReplError(RuntimeError):
    pass


_PROMPT_TOO_LONG_RE = re.compile(r"Prompt too long:\s*(\d+)\s*tokens\s*\(max=(\d+)\)\.")


def _maybe_print_prompt_too_long_hint(*, msg: str, requested_max_seq_len: int | None) -> None:
    m = _PROMPT_TOO_LONG_RE.search(msg or "")
    if not m:
        return
    try:
        max_allowed = int(m.group(2))
    except Exception:
        return
    if requested_max_seq_len is None:
        return
    try:
        requested = int(requested_max_seq_len)
    except Exception:
        return
    if requested <= max_allowed:
        return

    print(
        "hint: the server is rejecting prompts longer than its configured --max-prompt-tokens. "
        f"You requested --max-seq-len={requested}, but the server cap is max_prompt_tokens={max_allowed}.",
        file=sys.stderr,
    )
    print(
        f"hint: restart the server with e.g. `spl server start --model <model> --max-prompt-tokens {requested}` (or higher).",
        file=sys.stderr,
    )


DOCS_INGEST_PROMPT = (
    "You are Superlinear Docs, a stateful long-context assistant for document-grounded Q&A.\n"
    "You will receive documents and later questions.\n"
    "\n"
    "## Global Rules (always)\n"
    "- Use ONLY the ingested documents provided in this session. Do not use external knowledge.\n"
    "- You MAY make logical inferences from document content. If the documents state 'A created the X series' and 'Y is part of the X series', you can conclude 'A created Y'.\n"
    "- Never invent citations or file paths.\n"
    "- CRITICAL: Before quoting anything, VERIFY which document it comes from by checking the [SOURCE path=...] tag.\n"
    "- Always end answers with: Sources: <comma-separated paths>\n"
    "\n"
    "## Ingestion Mode\n"
    "- I will send one or more documents wrapped in [SOURCE path=...] ... [/SOURCE].\n"
    "- Treat each block as a separate source and remember its path.\n"
    "- Do not answer questions, do not summarize, and do not add commentary while ingesting.\n"
    "- Reply with exactly: OK\n"
    "\n"
    "## Q&A Mode\n"
    "When the user asks a question, follow this method in <think>:\n"
    "\n"
    "### Step 1: PARSE THE QUESTION\n"
    "- Does it ask about a SPECIFIC document? (e.g., 'the LSTM article', 'the Transformer document')\n"
    "  → If yes, you MUST only use content from that exact document. Ignore all other documents.\n"
    "- Does it ask 'list ALL', 'every', 'how many'?\n"
    "  → If yes, scan exhaustively. Do not stop at first few examples.\n"
    "- Does it ask about presence/absence? ('does X mention Y', 'which do NOT mention')\n"
    "  → If yes, you MUST search for the exact term in the specific document before answering.\n"
    "\n"
    "### Step 1.5: SOURCE-SPECIFIC REASONING (when question names a specific document)\n"
    "If the question asks about a specific article/document (e.g., 'According to the RNN article...'):\n"
    "\n"
    "**STRICT RULE: You must ONLY use content from that ONE document. Do NOT mention, quote, or reference ANY other document in your answer.**\n"
    "\n"
    "Reason through these steps:\n"
    "1. IDENTIFY: 'The question asks specifically about the [X] article.'\n"
    "2. LOCATE: 'I have the [X] document at [SOURCE path=...]. I will search ONLY within it.'\n"
    "3. SEARCH: 'Looking for [topic/terms] in the [X] document...'\n"
    "4. EXTRACT: 'Found in the [X] document: [exact quote with context]'\n"
    "5. CONFIRM: 'This quote is definitely from [X], not from another document.'\n"
    "6. FINAL CHECK: 'My answer mentions ONLY the [X] article. I have NOT included information from other articles.'\n"
    "\n"
    "**CRITICAL**: Before using ANY quote, check its [SOURCE path=...]. If the path contains a different filename than [X], you MUST NOT use that quote. Find a quote from the correct document or say the information is not in that document.\n"
    "\n"
    "Example: If asked 'What does the Transformer article say about X?' and you find a great quote from attention_machine_learning.txt → REJECT IT. Either find a quote from transformer_deep_learning.txt or say 'The Transformer article does not contain this.'\n"
    "\n"
    "If the answer is not in the named document, say 'The [X] article does not contain this information.' Do NOT supplement with other documents.\n"
    "\n"
    "### Step 2: VERIFICATION (MANDATORY)\n"
    "Before writing your answer, perform these checks in <think>:\n"
    "\n"
    "A) SOURCE VERIFICATION: For each quote you plan to use:\n"
    "   - State: 'This quote appears in [document name] at [SOURCE path=...]'\n"
    "   - If you cannot confirm the source, do not use the quote.\n"
    "   - If asked about 'the X article' and your quote is from a different article, DISCARD IT.\n"
    "\n"
    "B) TERM SEARCH: For presence/absence questions ('does X mention Y'):\n"
    "   - State: 'Searching for \"Y\" in [document]...'\n"
    "   - Search for the term, partial matches, and variations (singular/plural).\n"
    "   - State: 'Found: [quote]' or 'Not found after searching [sections checked]'\n"
    "   - Only conclude absence after thorough search.\n"
    "\n"
    "C) ENUMERATION CHECK: For 'list all' questions:\n"
    "   - After your initial list, ask: 'Did I miss anything?'\n"
    "   - Re-scan the relevant sections.\n"
    "   - Add any missed items.\n"
    "\n"
    "### Step 3: ANSWER\n"
    "- Quote with correct source attribution.\n"
    "- For enumeration: numbered/bulleted list.\n"
    "- End with: Sources: <paths>\n"
)

DOCS_QA_PROMPT = (
    "Answer using ONLY the ingested documents. Do not use external knowledge.\n"
    "You MAY make logical inferences from document content (e.g., if 'A created the X series' and 'Y is in the X series', conclude 'A created Y').\n"
    "\n"
    "## Method (in <think>)\n"
    "\n"
    "### Step 1: PARSE THE QUESTION\n"
    "- Does it ask about a SPECIFIC document? → Only use that document.\n"
    "- Does it ask 'list ALL', 'every', 'how many'? → Scan exhaustively.\n"
    "- Does it ask about presence/absence? → Search for exact term.\n"
    "\n"
    "### Step 1.5: SOURCE-SPECIFIC REASONING (when question names a specific document)\n"
    "If the question asks about a specific article (e.g., 'According to the RNN article...'):\n"
    "\n"
    "**STRICT RULE: ONLY use content from that ONE document. Do NOT mention or quote ANY other document.**\n"
    "\n"
    "Reason through:\n"
    "1. IDENTIFY: 'This question asks about the [X] article specifically.'\n"
    "2. LOCATE: 'I will search ONLY within the [X] document.'\n"
    "3. SEARCH: 'Looking for [topic] in [X]...'\n"
    "4. EXTRACT: 'Found in [X]: [exact quote]'\n"
    "5. CONFIRM: 'This is from [X], not another document.'\n"
    "6. FINAL CHECK: 'My answer mentions ONLY [X]. No other articles are referenced.'\n"
    "\n"
    "If not found in the named document, say 'The [X] article does not contain this.' Do NOT supplement with other documents.\n"
    "\n"
    "### Step 2: USE EXCERPTS (with source filtering)\n"
    "You received 'Retrieved excerpts' with passages from multiple documents.\n"
    "**WARNING: For source-specific questions, IGNORE the excerpts entirely.**\n"
    "**Search your full memory of the named document instead.**\n"
    "The excerpts may contain tempting quotes from OTHER documents - do not use them.\n"
    "\n"
    "### Step 3: VERIFICATION (MANDATORY)\n"
    "\n"
    "A) SOURCE VERIFICATION: For each quote:\n"
    "   - State: 'This quote appears in [document] at [SOURCE path=...]'\n"
    "   - If asked about 'the X article' and quote is from different article, DISCARD IT.\n"
    "\n"
    "B) TERM SEARCH: For presence/absence questions:\n"
    "   - State: 'Searching for \"Y\" in [document]...'\n"
    "   - Search for term and variations.\n"
    "   - State: 'Found: [quote]' or 'Not found after searching [sections]'\n"
    "\n"
    "C) ENUMERATION CHECK: For 'list all':\n"
    "   - After initial list, ask: 'Did I miss anything?'\n"
    "   - Re-scan and add missed items.\n"
    "\n"
    "## Answer\n"
    "- Quote with correct source attribution.\n"
    "- **For source-specific questions: cite ONLY the named document, even if you mention topics that appear in other documents.**\n"
    "- End with: Sources: <paths>\n"
)


DOCS_QA_PRIMER_ASSISTANT = (
    "Understood. I will use any provided excerpts as hints, but verify against my full memory of the documents. "
    "I will search for the pivotal scenes and dialogue, trace the causation chain, "
    "and give a substantive answer with accurately quoted evidence."
)


_FOLLOWUP_ANAPHORA_RE = re.compile(r"\b(this|that|it|above|previous|earlier|last)\b", re.IGNORECASE)
_FOLLOWUP_INTENT_RE = re.compile(
    r"\b(which\s+(article|source|file|doc|document)|where\s+.*\b(from|in)\b|what\s+(article|source|file|doc|document))\b",
    re.IGNORECASE,
)


def _should_augment_rag_query_with_prev_question(*, question: str, prev_question: str | None) -> bool:
    if not prev_question or not isinstance(prev_question, str) or not prev_question.strip():
        return False
    q = (question or "").strip()
    if not q:
        return False

    # Heuristic: for follow-up questions that refer to "this/that/it" or ask "which article/source",
    # include the previous user question to re-anchor lexical retrieval.
    if _FOLLOWUP_ANAPHORA_RE.search(q) or _FOLLOWUP_INTENT_RE.search(q):
        # Avoid hijacking very short single-entity queries like "RoPE?".
        terms = tokenize_query_terms(q, max_terms=32)
        if len(terms) <= 1 and len(q) < 24:
            return False
        return True

    # If the question has no content terms at all, it's likely a follow-up.
    terms = tokenize_query_terms(q, max_terms=32)
    return not terms


def _now_utc_compact() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def _sanitize_for_id(name: str) -> str:
    out = []
    for ch in name.strip():
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out).strip("_")
    return s[:32] if s else "docs"


def _new_session_id(*, workspace_name: str) -> str:
    import secrets

    prefix = f"docs_{_sanitize_for_id(workspace_name)}"
    return f"{prefix}_{_now_utc_compact()}_{secrets.token_hex(3)}"


def _ensure_reachable(client: SuperlinearClient) -> None:
    try:
        client.health()
    except HttpError as exc:
        raise DocsReplError(
            f"Server unreachable at {client.base_url}. Start it with `spl server start --model <model>` "
            f"or pass `--url`.\n{exc}"
        ) from exc


def _session_exists(client: SuperlinearClient, session_id: str) -> bool:
    try:
        client.request_json("GET", f"/v1/sessions/{session_id}", timeout_s=5.0)
        return True
    except HttpError as exc:
        if exc.status_code == 404:
            return False
        raise


def _create_session(client: SuperlinearClient, session_id: str) -> None:
    try:
        client.request_json("POST", "/v1/sessions", payload={"session_id": session_id}, timeout_s=30.0)
    except HttpError as exc:
        if exc.status_code == 409:
            return
        raise


def _create_session_with_max_seq_len(
    client: SuperlinearClient, session_id: str, *, max_seq_len: int | None
) -> None:
    payload: dict[str, Any] = {"session_id": session_id}
    if max_seq_len is not None:
        payload["max_seq_len"] = int(max_seq_len)
    try:
        client.request_json("POST", "/v1/sessions", payload=payload, timeout_s=30.0)
    except HttpError as exc:
        if exc.status_code == 409:
            return
        raise


def _maybe_resize_session(
    client: SuperlinearClient,
    session_id: str,
    *,
    min_max_seq_len: int | None,
    strategy: str = "auto",
) -> None:
    if min_max_seq_len is None:
        return
    try:
        info = client.request_json("GET", f"/v1/sessions/{session_id}", timeout_s=10.0)
    except HttpError:
        return
    if not isinstance(info, dict):
        return
    try:
        cur = int(info.get("max_seq_len") or 0)
    except Exception:
        cur = 0
    target = int(min_max_seq_len)
    if target <= 0 or (cur > 0 and target <= cur):
        return

    # Resize the *existing* session to at least the requested length.
    try:
        client.request_json(
            "POST",
            f"/v1/sessions/{session_id}/resize",
            payload={"max_seq_len": target, "strategy": strategy},
            timeout_s=300.0,
        )
    except HttpError as exc:
        # Provide a more actionable hint for common failure modes.
        raise DocsReplError(
            "Failed to resize session context length. "
            "This can happen if the target is too large for GPU memory. "
            f"(session_id={session_id} target_max_seq_len={target}): {exc}"
        ) from exc


def _get_session_pos(client: SuperlinearClient, session_id: str) -> int | None:
    try:
        info = client.request_json("GET", f"/v1/sessions/{session_id}", timeout_s=10.0)
    except HttpError:
        return None
    if isinstance(info, dict) and isinstance(info.get("current_pos"), int):
        return int(info["current_pos"])
    return None


def _encode_sources_description(sources: list[dict[str, Any]]) -> str:
    return json.dumps({"spl_docs_sources_v1": sources}, ensure_ascii=False, sort_keys=True)


def _decode_sources_description(desc: str | None) -> list[dict[str, Any]] | None:
    if not desc or not isinstance(desc, str):
        return None
    try:
        obj = json.loads(desc)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    raw = obj.get("spl_docs_sources_v1")
    if not isinstance(raw, list):
        return None
    out: list[dict[str, Any]] = []
    for s in raw:
        if not isinstance(s, dict):
            continue
        path = s.get("path")
        if not isinstance(path, str) or not path:
            continue
        item: dict[str, Any] = {"path": path}
        title = s.get("title")
        if isinstance(title, str) and title.strip():
            item["title"] = title.strip()
        source = s.get("source")
        if isinstance(source, str) and source.strip():
            item["source"] = source.strip()
        url = s.get("url")
        if isinstance(url, str) and url.strip():
            item["url"] = url.strip()
        b = s.get("bytes")
        if isinstance(b, int) and b >= 0:
            item["bytes"] = b
        sha = s.get("sha256")
        if isinstance(sha, str) and sha:
            item["sha256"] = sha
        added = s.get("added_at_unix_s")
        if isinstance(added, int) and added > 0:
            item["added_at_unix_s"] = added
        out.append(item)
    return out


def _extract_doc_metadata(*, path: Path, text: str) -> dict[str, str]:
    """Best-effort extraction of title/source/url from a document.

    Supports the wiki test corpus header format:
      Title: ...\nSource: ...\nURL: ...
    and common Markdown titles.
    """

    title: str | None = None
    source: str | None = None
    url: str | None = None

    lines = text.replace("\r", "").split("\n")
    head = lines[:80]

    for ln in head:
        s = ln.strip()
        if not s:
            continue
        if s.lower().startswith("title:") and title is None:
            title = s.split(":", 1)[1].strip()
            continue
        if s.lower().startswith("source:") and source is None:
            source = s.split(":", 1)[1].strip()
            continue
        if s.lower().startswith("url:") and url is None:
            url = s.split(":", 1)[1].strip()
            continue
        if s.startswith("#") and title is None:
            # Markdown heading.
            title = s.lstrip("#").strip()
            continue

    if not title:
        title = path.stem

    out: dict[str, str] = {"title": title}
    if source:
        out["source"] = source
    if url:
        out["url"] = url
    return out


def _build_qa_bootstrap_message(*, sources: list[dict[str, Any]]) -> str:
    """A strong, near-generation instruction + index injected as a USER message.

    This avoids relying on a late system prompt in session mode: the server intentionally
    drops additional system messages once the transcript already has a leading system.
    """

    # Keep the index compact and resilient.
    max_items = 200
    items = sources[:max_items]

    lines: list[str] = []
    lines.append("You are now in docs Q&A mode for this session.")
    lines.append("You MUST answer using ONLY the ingested documents in this session. Do not use external knowledge.")
    lines.append("If the documents do not contain the answer, say you don't know. Do not guess.")
    lines.append("Never invent citations or file paths.")
    lines.append("")
    lines.append("Method (follow this every time):")
    lines.append("1) In <think>, pick the 3–8 most likely sources from the index by title/source/path.")
    lines.append("2) In <think>, thoroughly search those sources for relevant passages and extract concrete facts.")
    lines.append("3) In <think>, reconcile conflicts and consolidate into a coherent answer.")
    lines.append("4) In the final answer, include short quotes (1–3 lines) for key claims when possible.")
    lines.append("5) Always end with a final line exactly: Sources: <comma-separated paths>.")
    lines.append("")
    lines.append("Note: You may receive a 'Retrieved excerpts' message immediately before a question.")
    lines.append("- Use those excerpts as PRIMARY evidence when relevant.")
    lines.append("- They are not exhaustive; consult other sources from the index if needed.")
    lines.append("")
    lines.append("Quality bar:")
    lines.append("- If you cannot find direct support in the docs, respond 'I don't know'.")
    lines.append("  Still include Sources listing the most relevant documents you checked (1–8 paths).")
    lines.append("- If partially supported, clearly separate supported vs missing details.")
    lines.append("- Prefer fewer, higher-confidence claims over broad speculation.")
    lines.append("- IMPORTANT: Do not rush. Use an extended <think> phase to do the source-selection and extraction steps.")
    lines.append("")
    lines.append("Available documents (index):")

    for i, s in enumerate(items, start=1):
        path = str(s.get("path") or "")
        title = str(s.get("title") or "").strip() or Path(path).name
        title = title.replace("\n", " ").strip()
        if len(title) > 120:
            title = title[:117] + "…"
        src = s.get("source")
        url = s.get("url")
        extra: list[str] = []
        if isinstance(src, str) and src.strip():
            extra.append(f"source={src.strip()}")
        if isinstance(url, str) and url.strip():
            extra.append(f"url={url.strip()}")
        extra_s = (" | " + " | ".join(extra)) if extra else ""
        lines.append(f"- {i}. {title} | path={path}{extra_s}")

    if len(sources) > max_items:
        lines.append(f"(and {len(sources) - max_items} more not shown)")

    return "\n".join(lines).strip() + "\n"


def _hydrate_sources_from_snapshot(client: SuperlinearClient, snapshot_id: str) -> list[dict[str, Any]] | None:
    try:
        manifest = client.request_json("GET", f"/v1/snapshots/{snapshot_id}", timeout_s=30.0)
    except HttpError:
        return None
    if not isinstance(manifest, dict):
        return None
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        return None
    return _decode_sources_description(metadata.get("description"))


def _banner(
    *,
    url: str,
    name: str,
    session_id: str,
    resumed: bool,
    phase: str,
    source_count: int,
    base_snapshot_id: str | None,
    rag_status: str,
) -> None:
    mode = "resumed" if resumed else "new"
    print(f"server={url}")
    print(f"workspace={name} ({mode})")
    print(f"session_id={session_id}")
    print(f"phase={phase} sources={source_count}")
    print(rag_status)
    if base_snapshot_id:
        print(f"base_snapshot_id={base_snapshot_id}")
    print("type /help for commands")


def _rag_backend_from_ws(ws: DocsWorkspaceState) -> str:
    backend = getattr(ws, "rag_backend", None)
    if isinstance(backend, str):
        b = backend.strip().lower()
        if b in {"light", "bm25", "off"}:
            return b
    return "light" if bool(ws.light_rag_enabled) else "off"


def _apply_rag_backend_to_ws(ws: DocsWorkspaceState, backend: str) -> None:
    b = (backend or "").strip().lower()
    if b not in {"light", "bm25", "off"}:
        b = "light"
    ws.rag_backend = b
    ws.light_rag_enabled = b != "off"


def _light_rag_config_from_ws(ws: DocsWorkspaceState) -> LightRagConfig:
    enabled = _rag_backend_from_ws(ws) != "off"
    return LightRagConfig(
        enabled=enabled,
        k=int(ws.light_rag_k),
        total_chars=int(ws.light_rag_total_chars),
        per_source_chars=int(ws.light_rag_per_source_chars),
        debug=bool(ws.light_rag_debug),
    ).sanitized()


def _apply_light_rag_config_to_ws(ws: DocsWorkspaceState, cfg: LightRagConfig) -> None:
    ws.light_rag_enabled = bool(cfg.enabled)
    ws.light_rag_k = int(cfg.k)
    ws.light_rag_total_chars = int(cfg.total_chars)
    ws.light_rag_per_source_chars = int(cfg.per_source_chars)
    ws.light_rag_debug = bool(cfg.debug)


def _format_light_rag_status(cfg: LightRagConfig) -> str:
    cfg = cfg.sanitized()
    return (
        f"lightRAG={'on' if cfg.enabled else 'off'} "
        f"k={cfg.k} chars={cfg.total_chars} per_source_chars={cfg.per_source_chars} "
        f"debug={'on' if cfg.debug else 'off'}"
    )


def _bm25_rag_config_from_ws(ws: DocsWorkspaceState) -> Bm25RagConfig:
    common = _light_rag_config_from_ws(ws).sanitized()
    try:
        k_sources = int(ws.bm25_k_sources)
    except Exception:
        k_sources = 0
    if k_sources <= 0:
        k_sources = int(common.k)
    try:
        k_paragraphs = int(ws.bm25_k_paragraphs)
    except Exception:
        k_paragraphs = 40
    return Bm25RagConfig(
        enabled=bool(common.enabled),
        k_sources=k_sources,
        total_chars=int(common.total_chars),
        per_source_chars=int(common.per_source_chars),
        debug=bool(common.debug),
        k_paragraphs=k_paragraphs,
        max_terms=int(common.max_terms),
        max_paragraphs_per_source=int(common.max_paragraphs_per_source),
        max_paragraph_chars=int(common.max_paragraph_chars),
    ).sanitized()


def _format_rag_status(*, ws: DocsWorkspaceState, bm25_available: bool) -> str:
    backend = _rag_backend_from_ws(ws)
    common = _light_rag_config_from_ws(ws).sanitized()
    debug_s = "on" if common.debug else "off"
    if backend == "off" or not common.enabled:
        return f"rag=off debug={debug_s}"
    if backend == "bm25":
        bm25_cfg = _bm25_rag_config_from_ws(ws)
        avail_s = "" if bm25_available else " (unavailable -> fallback light)"
        return (
            f"rag=on backend=bm25 k_sources={bm25_cfg.k_sources} k_paragraphs={bm25_cfg.k_paragraphs} "
            f"chars={common.total_chars} per_source_chars={common.per_source_chars} debug={debug_s}{avail_s}"
        )
    return (
        f"rag=on backend=light k={common.k} chars={common.total_chars} per_source_chars={common.per_source_chars} "
        f"debug={debug_s}"
    )


def _format_sources_table(sources: list[dict[str, Any]]) -> str:
    rows: list[list[str]] = []
    for i, s in enumerate(sources, start=1):
        path = str(s.get("path") or "")
        b = s.get("bytes")
        sha = s.get("sha256")
        rows.append(
            [
                str(i),
                path,
                "" if b is None else str(b),
                "" if not isinstance(sha, str) else sha[:12],
            ]
        )
    return format_table(["#", "path", "bytes", "sha256"], rows)


class _TurnStats:
    def __init__(self) -> None:
        self.finish_reason: str | None = None
        self.ttft_s: float | None = None
        self.total_s: float | None = None
        self.prompt_tokens: int | None = None
        self.completion_tokens: int | None = None
        self.tok_per_s: float | None = None
        self.server_prefill_s: float | None = None
        self.server_decode_s: float | None = None
        self.server_total_s: float | None = None


def _stats_footer(stats: _TurnStats) -> str:
    parts: list[str] = []
    if stats.ttft_s is not None:
        parts.append(f"ttft={stats.ttft_s:.3f}s")
    if stats.tok_per_s is not None:
        parts.append(f"tok/s={stats.tok_per_s:.2f}")
    if stats.prompt_tokens is not None and stats.completion_tokens is not None:
        parts.append(f"tokens={stats.prompt_tokens}+{stats.completion_tokens}")
    if stats.finish_reason is not None:
        parts.append(f"finish={stats.finish_reason}")
    if stats.total_s is not None:
        parts.append(f"wall={stats.total_s:.3f}s")
    return " ".join(parts) if parts else ""


def _stats_detail(stats: _TurnStats) -> str:
    lines: list[str] = []
    if stats.finish_reason is not None:
        lines.append(f"finish_reason={stats.finish_reason}")
    if stats.ttft_s is not None:
        lines.append(f"ttft_s={stats.ttft_s:.6f}")
    if stats.total_s is not None:
        lines.append(f"wall_s={stats.total_s:.6f}")
    if stats.prompt_tokens is not None:
        lines.append(f"prompt_tokens={stats.prompt_tokens}")
    if stats.completion_tokens is not None:
        lines.append(f"completion_tokens={stats.completion_tokens}")
    if stats.tok_per_s is not None:
        lines.append(f"tok_per_s={stats.tok_per_s:.6f}")
    if stats.server_prefill_s is not None:
        lines.append(f"server_prefill_s={stats.server_prefill_s:.6f}")
    if stats.server_decode_s is not None:
        lines.append(f"server_decode_s={stats.server_decode_s:.6f}")
    if stats.server_total_s is not None:
        lines.append(f"server_total_s={stats.server_total_s:.6f}")
    return "\n".join(lines)


def _stream_request(
    *,
    client: SuperlinearClient,
    session_id: str,
    messages: list[dict[str, Any]],
    max_completion_tokens: int = 32768,
    think_budget: int | None = 8192,
    temperature: float = 0.3,
    top_p: float = 0.95,
    print_output: bool = True,
) -> _TurnStats:
    enable_thinking_ui = print_output and think_budget is not None and think_budget > 0

    payload: dict[str, Any] = {
        "stream": True,
        "session_id": session_id,
        "messages": messages,
        "max_completion_tokens": int(max_completion_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
    }

    if enable_thinking_ui:
        payload["reasoning_budget"] = int(think_budget)
        payload["discard_thinking"] = True
        payload["stream_thinking"] = True

    started = time.monotonic()
    ttft_s: float | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    timing: dict[str, Any] | None = None

    started_answer = False
    in_think = False
    thinking_accum: str = ""
    thinking_panel_active = False
    thinking_panel_lines = 0
    content_buf = ""
    saw_thinking_delta = False
    thinking_start_time: float | None = None
    thinking_end_time: float | None = None

    def _thinking_panel_format(text: str) -> list[str]:
        cols = shutil.get_terminal_size(fallback=(120, 24)).columns
        prefix = "thinking: "
        width = max(20, cols - len(prefix) - 1)

        normalized = text.replace("\r", "")
        wrapped: list[str] = []
        for logical in normalized.split("\n"):
            parts = textwrap.wrap(
                logical,
                width=width,
                replace_whitespace=False,
                drop_whitespace=False,
                break_long_words=True,
                break_on_hyphens=False,
            )
            if not parts:
                wrapped.append("")
            else:
                wrapped.extend(parts)

        tail = wrapped[-10:]
        if not tail:
            tail = [""]
        return [prefix + ln for ln in tail]

    def _thinking_panel_move_to_top() -> None:
        nonlocal thinking_panel_lines
        if thinking_panel_lines > 1:
            sys.stdout.write(f"\x1b[{thinking_panel_lines - 1}A")

    def _thinking_panel_render(text: str) -> None:
        nonlocal thinking_panel_active, thinking_panel_lines
        if not print_output:
            return
        lines = _thinking_panel_format(text)

        if not thinking_panel_active:
            sys.stdout.write("\n")
            thinking_panel_active = True
            thinking_panel_lines = 1

        _thinking_panel_move_to_top()

        for i in range(thinking_panel_lines):
            sys.stdout.write("\r\x1b[2K")
            if i < thinking_panel_lines - 1:
                sys.stdout.write("\n")
        _thinking_panel_move_to_top()

        for i, ln in enumerate(lines):
            sys.stdout.write("\r\x1b[2K" + ln)
            if i < len(lines) - 1:
                sys.stdout.write("\n")

        thinking_panel_lines = len(lines)
        sys.stdout.flush()

    def _thinking_panel_clear() -> None:
        nonlocal thinking_panel_active, thinking_panel_lines, thinking_start_time, thinking_end_time
        if not print_output:
            return
        if not thinking_panel_active:
            return

        _thinking_panel_move_to_top()
        for i in range(thinking_panel_lines):
            sys.stdout.write("\r\x1b[2K")
            if i < thinking_panel_lines - 1:
                sys.stdout.write("\n")
        _thinking_panel_move_to_top()

        thinking_panel_active = False
        thinking_panel_lines = 0

        if thinking_start_time is not None and thinking_end_time is not None:
            duration = thinking_end_time - thinking_start_time
            if duration >= 60:
                minutes = int(duration // 60)
                seconds = duration % 60
                sys.stdout.write(
                    f"[thinking complete] duration: {minutes} minute{'s' if minutes != 1 else ''} {seconds:.1f} seconds\n"
                )
            else:
                sys.stdout.write(f"[thinking complete] duration: {duration:.1f} seconds\n")

        sys.stdout.flush()

    def _answer_start_if_needed() -> None:
        nonlocal started_answer
        if not print_output:
            return
        if not started_answer:
            _thinking_panel_clear()
            print("assistant: ", end="", flush=True)
            started_answer = True

    gen = client.request_sse("POST", "/v1/chat/completions", payload=payload, timeout_s=3600.0)
    try:
        for event in gen:
            if isinstance(event, dict) and "error" in event:
                err = event.get("error")
                msg = err.get("message") if isinstance(err, dict) else str(err)
                raise DocsReplError(str(msg))

            if not isinstance(event, dict):
                continue

            choices = event.get("choices")
            if isinstance(choices, list) and choices:
                ch0 = choices[0]
                if isinstance(ch0, dict):
                    delta = ch0.get("delta") if isinstance(ch0.get("delta"), dict) else {}
                    if isinstance(delta, dict):
                        thinking = delta.get("thinking")
                        if enable_thinking_ui and isinstance(thinking, str) and thinking:
                            saw_thinking_delta = True
                            if ttft_s is None:
                                ttft_s = time.monotonic() - started

                            buf = thinking
                            while buf:
                                if not in_think:
                                    start_idx = buf.find("<think>")
                                    if start_idx == -1:
                                        break
                                    buf = buf[start_idx + len("<think>") :]
                                    in_think = True
                                    thinking_accum = ""
                                    if thinking_start_time is None:
                                        thinking_start_time = time.monotonic()
                                    _thinking_panel_render(thinking_accum)
                                    continue

                                end_idx = buf.find("</think>")
                                if end_idx == -1:
                                    thinking_accum += buf
                                    buf = ""
                                    _thinking_panel_render(thinking_accum)
                                    break

                                thinking_accum += buf[:end_idx]
                                buf = buf[end_idx + len("</think>") :]
                                if thinking_start_time is not None:
                                    thinking_end_time = time.monotonic()
                                _thinking_panel_clear()
                                in_think = False
                                break

                        content = delta.get("content")
                        if isinstance(content, str) and content:
                            if ttft_s is None:
                                ttft_s = time.monotonic() - started
                            if not enable_thinking_ui or saw_thinking_delta:
                                _answer_start_if_needed()
                                sys.stdout.write(content)
                                sys.stdout.flush()
                            else:
                                # Fallback: parse <think> tags from content if the server isn't
                                # sending delta.thinking.
                                content_buf += content
                                while content_buf:
                                    if in_think:
                                        end_idx = content_buf.find("</think>")
                                        if end_idx == -1:
                                            thinking_accum += content_buf
                                            content_buf = ""
                                            _thinking_panel_render(thinking_accum)
                                            break

                                        thinking_accum += content_buf[:end_idx]
                                        content_buf = content_buf[end_idx + len("</think>") :]
                                        in_think = False
                                        if thinking_start_time is not None:
                                            thinking_end_time = time.monotonic()
                                        _thinking_panel_clear()
                                        continue

                                    start_idx = content_buf.find("<think>")
                                    if start_idx == -1:
                                        _answer_start_if_needed()
                                        sys.stdout.write(content_buf)
                                        sys.stdout.flush()
                                        content_buf = ""
                                        break

                                    if start_idx > 0:
                                        _answer_start_if_needed()
                                        sys.stdout.write(content_buf[:start_idx])
                                        sys.stdout.flush()

                                    content_buf = content_buf[start_idx + len("<think>") :]
                                    in_think = True
                                    thinking_accum = ""
                                    if thinking_start_time is None:
                                        thinking_start_time = time.monotonic()
                                    _thinking_panel_render(thinking_accum)
                                    continue

                        tool_calls = delta.get("tool_calls")
                        if tool_calls is not None:
                            if ttft_s is None:
                                ttft_s = time.monotonic() - started
                            if print_output:
                                sys.stdout.write(
                                    f"\n<tool_calls {len(tool_calls) if isinstance(tool_calls, list) else 1}>\n"
                                )
                                sys.stdout.flush()

                    fr = ch0.get("finish_reason")
                    if fr is not None:
                        finish_reason = str(fr)

            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            if isinstance(event.get("x_superlinear_timing"), dict):
                timing = event["x_superlinear_timing"]
    except KeyboardInterrupt:
        try:
            gen.close()
        except Exception:
            pass
        raise
    finally:
        try:
            gen.close()
        except Exception:
            pass

        _thinking_panel_clear()
        if enable_thinking_ui and in_think and thinking_start_time is not None and thinking_end_time is None:
            if print_output:
                sys.stdout.write("[thinking incomplete] (no </think> received before stream ended)\n")
                sys.stdout.flush()

    ended = time.monotonic()
    if print_output:
        sys.stdout.write("\n")
        sys.stdout.flush()

    stats = _TurnStats()
    stats.finish_reason = finish_reason
    stats.ttft_s = ttft_s
    stats.total_s = max(ended - started, 0.0)

    if usage is not None:
        pt = usage.get("prompt_tokens")
        ct = usage.get("completion_tokens")
        if isinstance(pt, int):
            stats.prompt_tokens = pt
        if isinstance(ct, int):
            stats.completion_tokens = ct

    if timing is not None:
        prefill_s = timing.get("prefill_s")
        decode_s = timing.get("decode_s")
        total_s = timing.get("total_s")
        tok_per_s = timing.get("tok_per_s")
        if isinstance(prefill_s, (float, int)):
            stats.server_prefill_s = float(prefill_s)
        if isinstance(decode_s, (float, int)):
            stats.server_decode_s = float(decode_s)
        if isinstance(total_s, (float, int)):
            stats.server_total_s = float(total_s)
        if isinstance(tok_per_s, (float, int)):
            stats.tok_per_s = float(tok_per_s)

    return stats


def _expand_files(args: list[str]) -> list[Path]:
    out: list[Path] = []
    for a in args:
        matches = glob.glob(a, recursive=True)
        candidates = matches if matches else [a]
        for c in candidates:
            p = Path(c).expanduser()
            if p.is_dir():
                for f in sorted(p.rglob("*")):
                    if f.is_file():
                        out.append(f)
            else:
                out.append(p)

    seen: set[str] = set()
    deduped: list[Path] = []
    for p in out:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    return deduped


def _read_text_file(path: Path) -> tuple[str, int, str]:
    data = path.read_bytes()
    if b"\x00" in data:
        raise DocsReplError(f"Refusing to ingest binary file (NUL byte found): {path}")
    text = data.decode("utf-8", errors="replace")
    sha = hashlib.sha256(data).hexdigest()
    return text, len(data), sha


def _build_docs_message(contents: list[tuple[str, str]]) -> str:
    # contents: list[(path_str, text)]
    blocks: list[str] = []
    for path_str, text in contents:
        blocks.append(f"[SOURCE path={path_str}]\n{text}\n[/SOURCE]\n")
    return "\n".join(blocks).strip() + "\n"


def _is_prompt_level_message(m: dict) -> bool:
    """Return True if this is a prompt-level message (system, document ingest, or ingest ack)."""
    role = m.get("role")
    content = m.get("content") or ""
    if role == "system":
        return True
    if role == "user" and "[SOURCE path=" in content:
        return True
    if role == "assistant" and content.strip() == "OK":
        return True
    return False


def _cmd_head(*, client: SuperlinearClient, session_id: str, limit: int = 10) -> None:
    """Show first n Q&A messages (excludes prompt-level messages)."""
    try:
        resp = client.request_json("GET", f"/v1/sessions/{session_id}/history", timeout_s=10.0)
    except HttpError as exc:
        raise DocsReplError(str(exc)) from exc

    msgs = resp.get("messages") if isinstance(resp, dict) else None
    if not isinstance(msgs, list):
        raise DocsReplError("Invalid response from server for /head")

    # Filter out prompt-level messages
    qa_msgs = [(i, m) for i, m in enumerate(msgs, 1) if isinstance(m, dict) and not _is_prompt_level_message(m)]

    limit = max(1, min(int(limit), 200))
    head = qa_msgs[:limit]
    if not head:
        print("(no Q&A messages yet)")
        return

    for orig_idx, m in head:
        role = m.get("role")
        content = m.get("content") or ""
        one_line = content.replace("\r", "").replace("\n", " ").strip()
        if len(one_line) > 200:
            one_line = one_line[:197] + "…"
        print(f"{orig_idx:>4} {role}: {one_line}")


def _cmd_tail(*, client: SuperlinearClient, session_id: str, limit: int = 10) -> None:
    """Show last n Q&A messages (excludes prompt-level messages)."""
    try:
        resp = client.request_json("GET", f"/v1/sessions/{session_id}/history", timeout_s=10.0)
    except HttpError as exc:
        raise DocsReplError(str(exc)) from exc

    msgs = resp.get("messages") if isinstance(resp, dict) else None
    if not isinstance(msgs, list):
        raise DocsReplError("Invalid response from server for /tail")

    # Filter out prompt-level messages
    qa_msgs = [(i, m) for i, m in enumerate(msgs, 1) if isinstance(m, dict) and not _is_prompt_level_message(m)]

    limit = max(1, min(int(limit), 200))
    tail = qa_msgs[-limit:]
    if not tail:
        print("(no Q&A messages yet)")
        return

    for orig_idx, m in tail:
        role = m.get("role")
        content = m.get("content") or ""
        one_line = content.replace("\r", "").replace("\n", " ").strip()
        if len(one_line) > 200:
            one_line = one_line[:197] + "…"
        print(f"{orig_idx:>4} {role}: {one_line}")


def _wrap_for_terminal(text: str, *, indent: str = "", width: int | None = None) -> str:
    cols = shutil.get_terminal_size(fallback=(120, 24)).columns
    target_width = cols if width is None else int(width)
    target_width = max(20, target_width)

    normalized = text.replace("\r", "")
    out_lines: list[str] = []
    for logical in normalized.split("\n"):
        if not logical:
            out_lines.append(indent)
            continue
        wrapped = textwrap.wrap(
            logical,
            width=max(10, target_width - len(indent)),
            replace_whitespace=False,
            drop_whitespace=False,
            break_long_words=True,
            break_on_hyphens=False,
        )
        if not wrapped:
            out_lines.append(indent)
        else:
            out_lines.extend([indent + w for w in wrapped])
    return "\n".join(out_lines)


def _cmd_show(*, client: SuperlinearClient, session_id: str, index: int) -> None:
    """Show a single message in full by 1-based index.

    Use the index shown by /head or /tail (it is the original index in the session history).
    """
    try:
        resp = client.request_json("GET", f"/v1/sessions/{session_id}/history", timeout_s=10.0)
    except HttpError as exc:
        raise DocsReplError(str(exc)) from exc

    msgs = resp.get("messages") if isinstance(resp, dict) else None
    if not isinstance(msgs, list):
        raise DocsReplError("Invalid response from server for /show")

    n = len(msgs)
    if n == 0:
        print("(empty)")
        return
    if index < 1 or index > n:
        raise DocsReplError(f"Message index out of range: {index} (1..{n})")

    m = msgs[index - 1]
    if not isinstance(m, dict):
        raise DocsReplError("Invalid message format")

    role = str(m.get("role") or "")
    content = m.get("content")
    tool_calls = m.get("tool_calls")

    if content is None and tool_calls is not None:
        content_str = f"<tool_calls {len(tool_calls) if isinstance(tool_calls, list) else 1}>"
    else:
        content_str = "" if content is None else str(content)

    header = f"{index:>4} {role}:"
    print(header)
    if content_str:
        print(_wrap_for_terminal(content_str, indent="     "))
    else:
        print("     (empty)")


def _cmd_ls(*, client: SuperlinearClient, current_session_id: str, docs_workspaces: dict[str, DocsWorkspaceState]) -> None:
    """List all sessions and snapshots."""
    # Sessions
    try:
        payload = client.request_json("GET", "/v1/sessions", timeout_s=10.0)
    except HttpError as exc:
        raise DocsReplError(str(exc)) from exc

    raw_sessions = payload.get("sessions") if isinstance(payload, dict) else None
    if not isinstance(raw_sessions, list):
        raw_sessions = []
    session_ids = [s for s in raw_sessions if isinstance(s, str)]

    # Build reverse lookup: session_id -> workspace name
    session_to_workspace: dict[str, tuple[str, DocsWorkspaceState]] = {}
    for ws_name, ws_state in docs_workspaces.items():
        session_to_workspace[ws_state.session_id] = (ws_name, ws_state)

    print("sessions:")
    if not session_ids:
        print("  (none)")
    else:
        for sid in session_ids:
            marker = " *" if sid == current_session_id else ""
            # Add workspace info for docs sessions
            if sid in session_to_workspace:
                ws_name, ws_state = session_to_workspace[sid]
                src_count = len(ws_state.sources) if ws_state.sources else 0
                snap_info = f" snap={ws_state.base_snapshot_id[:8]}..." if ws_state.base_snapshot_id else ""
                print(f"  {sid}{marker}  (docs:{ws_name} {src_count} sources{snap_info})")
            else:
                print(f"  {sid}{marker}")

    # Snapshots (local)
    snapshots = list_local_snapshots()
    print("\nsnapshots:")
    if not snapshots:
        print("  (none)")
    else:
        for snap in snapshots:
            sid = snap.get("snapshot_id") or ""
            title = snap.get("title") or ""
            if title:
                print(f"  {sid}  {title}")
            else:
                print(f"  {sid}")


def _cmd_rm(
    *,
    client: SuperlinearClient,
    target_ids: list[str],
    current_session_id: str,
) -> bool:
    """Remove session(s) and/or snapshot(s). Returns True if current session was removed."""
    removed_current = False
    for target_id in target_ids:
        # Check if it's a snapshot ID (32-char hex)
        raw_id = target_id[5:] if target_id.startswith("snap-") else target_id
        is_snapshot = len(raw_id) == 32 and all(c in "0123456789abcdef" for c in raw_id.lower())

        if is_snapshot:
            # Delete snapshot
            deleted = delete_local_snapshot(raw_id)
            if deleted:
                print(f"removed snapshot_id={raw_id}")
            else:
                print(f"error: snapshot not found: {raw_id}", file=sys.stderr)
        else:
            # Delete session
            try:
                client.request_json("DELETE", f"/v1/sessions/{target_id}", timeout_s=10.0)
                print(f"removed session_id={target_id}")
                if target_id == current_session_id:
                    removed_current = True
            except HttpError as exc:
                if exc.status_code == 404:
                    print(f"error: session not found: {target_id}", file=sys.stderr)
                else:
                    print(f"error: failed to remove {target_id}: {exc}", file=sys.stderr)
    return removed_current


def _cmd_help() -> None:
    print(
        "\n".join(
            [
                "commands:",
                "  /help",
                "  /exit [-c]      exit (--clean/-c: delete workspace)",
                "  /clear          clear screen",
                "  /history [n]    show last n input commands (default 20)",
                "  /history clear  clear input command history",
                "  /info           show workspace info",
                "  /ls             list sessions and snapshots",
                "  /rm             delete current session, start fresh",
                "  /rm <id...>     delete session(s) or snapshot(s)",
                "  /head [n]       show first n Q&A messages",
                "  /tail [n]       show last n Q&A messages",
                "  /show <i>       show message i in full (use /tail to find ids)",
                "  /add <paths...> [-s] add documents (--save/-s: save snapshot)",
                "  /sources        list loaded sources",
                "  /rag ...        configure RAG backend",
                "  /reset          reset workspace to base snapshot",
                "  /stats          show last turn metrics",
                "  /save [title]   save snapshot",
                "  /load <snap>    load snapshot into new workspace",
            ]
        )
    )


def docs_repl(
    *,
    url: str,
    name: str,
    load_snapshot_id: str | None = None,
    max_seq_len: int | None = None,
    think_budget: int | None = 32768,
    temperature: float = 0.3,
    top_p: float = 0.95,
    system_prompt: str | None = None,
) -> int:
    _setup_readline_history()
    _setup_completer()
    if not name or not isinstance(name, str):
        print("docs name is required: `spl docs <name>`", file=sys.stderr)
        return 2

    client = SuperlinearClient(base_url=url, timeout_s=3600.0)
    try:
        _ensure_reachable(client)
    except DocsReplError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    state = load_state()
    existing_ws = state.docs_workspaces.get(name)
    resumed = existing_ws is not None

    if existing_ws is None:
        ws = DocsWorkspaceState(
            session_id=_new_session_id(workspace_name=name),
            phase="INGEST",
            base_snapshot_id=None,
            sources=[],
        )
    else:
        ws = existing_ws

    lock = SessionLock(session_id=ws.session_id, kind="docs", label=f"spl docs {name}")
    try:
        try:
            lock.acquire()
        except AlreadyLockedError as exc:
            print(
                f"error: workspace is already open in another REPL (workspace={name} session_id={ws.session_id} pid={exc.info.pid}).",
                file=sys.stderr,
            )
            print("next steps: close the other REPL, or choose a different docs workspace name.", file=sys.stderr)
            return 2

        # Handle --load flag: load from a snapshot
        if load_snapshot_id is not None:
            snap_id = load_snapshot_id.strip()
            # Normalize: accept raw 32-char hex or snap- prefix
            if snap_id.startswith("snap-"):
                snap_id = snap_id[5:]
            if len(snap_id) != 32:
                print(f"error: invalid snapshot id: {load_snapshot_id}", file=sys.stderr)
                return 2

            # Delete existing session if it exists
            if _session_exists(client, ws.session_id):
                try:
                    client.request_json("DELETE", f"/v1/sessions/{ws.session_id}", timeout_s=30.0)
                except HttpError:
                    pass  # Ignore deletion errors

            # Load from snapshot into a fresh session
            try:
                client.request_json(
                    "POST",
                    f"/v1/snapshots/{snap_id}/load",
                    payload={"session_id": ws.session_id},
                    timeout_s=300.0,
                )
            except HttpError as exc:
                if exc.status_code == 404:
                    print(f"error: snapshot not found: {snap_id} (use `spl snapshot ls`)", file=sys.stderr)
                elif exc.status_code == 429:
                    print("Server is busy (429). Try again.", file=sys.stderr)
                else:
                    print(str(exc), file=sys.stderr)
                return 1

            # Update workspace state with the loaded snapshot
            ws.base_snapshot_id = snap_id
            ws.phase = "INGEST"
            ws.sources = _hydrate_sources_from_snapshot(client, snap_id) or []
            resumed = True  # Mark as resumed since we loaded from snapshot

            state.docs_workspaces[name] = ws
            save_state(state)

            _maybe_resize_session(client, ws.session_id, min_max_seq_len=max_seq_len)

        elif existing_ws is None:
            _create_session_with_max_seq_len(client, ws.session_id, max_seq_len=max_seq_len)
            _maybe_resize_session(client, ws.session_id, min_max_seq_len=max_seq_len)
        else:
            if _session_exists(client, ws.session_id):
                _maybe_resize_session(client, ws.session_id, min_max_seq_len=max_seq_len)
            else:
                print(f"note: session not found on server: {ws.session_id}", file=sys.stderr)
                if ws.base_snapshot_id:
                    try:
                        client.request_json(
                            "POST",
                            f"/v1/snapshots/{ws.base_snapshot_id}/load",
                            payload={"session_id": ws.session_id},
                            timeout_s=300.0,
                        )
                    except HttpError as exc:
                        if exc.status_code == 404:
                            print(
                                f"error: base snapshot not found: {ws.base_snapshot_id} (use `spl snapshot ls`).",
                                file=sys.stderr,
                            )
                            ws.base_snapshot_id = None
                            ws.sources = []
                            _create_session(client, ws.session_id)
                        else:
                            raise
                    ws.phase = "INGEST"
                else:
                    ws.phase = "INGEST"
                    ws.sources = []
                    ws.base_snapshot_id = None
                    _create_session_with_max_seq_len(client, ws.session_id, max_seq_len=max_seq_len)
                    _maybe_resize_session(client, ws.session_id, min_max_seq_len=max_seq_len)

        # If we restored from a snapshot above, its max_seq_len may be smaller than the user's
        # requested context length. Apply a best-effort resize now.
        _maybe_resize_session(client, ws.session_id, min_max_seq_len=max_seq_len)

        # Best-effort hydration of sources from the base snapshot metadata.
        if (not ws.sources) and ws.base_snapshot_id:
            snap_sources = _hydrate_sources_from_snapshot(client, ws.base_snapshot_id)
            if snap_sources is not None:
                ws.sources = snap_sources

        state.docs_workspaces[name] = ws
        save_state(state)

        light_retriever = LightRagRetriever()
        bm25_retriever = Bm25RagRetriever()
        bm25_hint_printed = False

        _banner(
            url=client.base_url,
            name=name,
            session_id=ws.session_id,
            resumed=resumed,
            phase=ws.phase,
            source_count=len(ws.sources),
            base_snapshot_id=ws.base_snapshot_id,
            rag_status=_format_rag_status(ws=ws, bm25_available=bm25_retriever.is_available()),
        )

        last_stats: _TurnStats | None = None
        last_substantive_user_question: str | None = None

        while True:
            prompt = f"spl(docs:{name}:{ws.phase})> "
            try:
                line = input(prompt)
            except EOFError:
                print()
                return 0
            except KeyboardInterrupt:
                print("^C")
                continue

            line = line.strip()
            if not line:
                continue

            if line.startswith("/"):
                cmdline = line[1:].strip()
                try:
                    parts = shlex.split(cmdline)
                except ValueError as exc:
                    print(f"parse error: {exc}", file=sys.stderr)
                    continue
                if not parts:
                    continue
                cmd, args = parts[0], parts[1:]

                if cmd in {"exit", "quit"}:
                    clean = "--clean" in args or "-c" in args
                    if clean:
                        try:
                            client.request_json("DELETE", f"/v1/sessions/{ws.session_id}", timeout_s=30.0)
                        except HttpError:
                            pass
                        del state.docs_workspaces[name]
                        save_state(state)
                        print(f"deleted workspace: {name}")
                    return 0
                if cmd == "clear":
                    print("\033[2J\033[H", end="", flush=True)
                    continue
                if cmd == "help":
                    _cmd_help()
                    continue
                if cmd == "session":
                    print(ws.session_id)
                    continue
                if cmd == "info":
                    print(f"workspace={name}")
                    print(f"session_id={ws.session_id}")
                    print(f"phase={ws.phase}")
                    print(f"sources={len(ws.sources) if ws.sources else 0}")
                    if ws.base_snapshot_id:
                        print(f"base_snapshot_id={ws.base_snapshot_id}")
                    print(_format_rag_status(ws=ws, bm25_available=bm25_retriever.is_available()))
                    continue
                if cmd == "stats":
                    if last_stats is None:
                        print("no stats yet")
                    else:
                        print(_stats_detail(last_stats))
                    continue
                if cmd == "head":
                    n = 10
                    if len(args) == 1:
                        try:
                            n = int(args[0])
                        except Exception:
                            print("usage: /head [n]", file=sys.stderr)
                            continue
                    elif len(args) > 1:
                        print("usage: /head [n]", file=sys.stderr)
                        continue
                    try:
                        _cmd_head(client=client, session_id=ws.session_id, limit=n)
                    except DocsReplError as exc:
                        print(str(exc), file=sys.stderr)
                    continue
                if cmd == "tail":
                    n = 10
                    if len(args) == 1:
                        try:
                            n = int(args[0])
                        except Exception:
                            print("usage: /tail [n]", file=sys.stderr)
                            continue
                    elif len(args) > 1:
                        print("usage: /tail [n]", file=sys.stderr)
                        continue
                    try:
                        _cmd_tail(client=client, session_id=ws.session_id, limit=n)
                    except DocsReplError as exc:
                        print(str(exc), file=sys.stderr)
                    continue
                if cmd == "show":
                    if len(args) != 1:
                        print("usage: /show <i>", file=sys.stderr)
                        continue
                    try:
                        i = int(args[0])
                    except Exception:
                        print("usage: /show <i>", file=sys.stderr)
                        continue
                    try:
                        _cmd_show(client=client, session_id=ws.session_id, index=i)
                    except DocsReplError as exc:
                        print(str(exc), file=sys.stderr)
                    continue
                if cmd == "history":
                    if args in [["clear"], ["--clear"], ["-c"]]:
                        _cmd_history_clear()
                        continue
                    n = 20
                    if len(args) == 1:
                        try:
                            n = int(args[0])
                        except Exception:
                            print("usage: /history [n] | /history clear", file=sys.stderr)
                            continue
                    elif len(args) > 1:
                        print("usage: /history [n] | /history clear", file=sys.stderr)
                        continue
                    _cmd_history(n)
                    continue
                if cmd == "ls":
                    try:
                        _cmd_ls(client=client, current_session_id=ws.session_id, docs_workspaces=state.docs_workspaces)
                    except DocsReplError as exc:
                        print(str(exc), file=sys.stderr)
                    continue
                if cmd == "rm":
                    if not args:
                        # /rm with no args = delete current session and reset workspace
                        try:
                            client.request_json("DELETE", f"/v1/sessions/{ws.session_id}", timeout_s=10.0)
                            print(f"removed session_id={ws.session_id}")
                        except HttpError as exc:
                            if exc.status_code != 404:
                                print(f"error: {exc}", file=sys.stderr)
                        # Reset workspace and create new session
                        new_session = _new_session_id(workspace_name=name)
                        _create_session_with_max_seq_len(client, new_session, max_seq_len=max_seq_len)
                        lock.release()
                        lock = SessionLock(session_id=new_session, kind="docs", label=f"spl docs {name}")
                        lock.acquire()
                        ws.session_id = new_session
                        ws.phase = "INGEST"
                        ws.base_snapshot_id = None
                        ws.sources = []
                        state.docs_workspaces[name] = ws
                        save_state(state)
                        light_retriever.clear_cache()
                        bm25_retriever.clear_index()
                        print(f"new session_id={new_session}")
                        continue
                    try:
                        removed_current = _cmd_rm(
                            client=client,
                            target_ids=args,
                            current_session_id=ws.session_id,
                        )
                        if removed_current:
                            # Current session removed - need to recreate
                            print("current session deleted; recreating...")
                            new_session = _new_session_id(workspace_name=name)
                            _create_session_with_max_seq_len(client, new_session, max_seq_len=max_seq_len)
                            ws.session_id = new_session
                            ws.phase = "INGEST"
                            ws.base_snapshot_id = None
                            ws.sources = []
                            state.docs_workspaces[name] = ws
                            save_state(state)
                            lock.release()
                            lock = SessionLock(session_id=new_session, kind="docs", label=f"spl docs {name}")
                            lock.acquire()
                            light_retriever.clear_cache()
                            bm25_retriever.clear_index()
                    except DocsReplError as exc:
                        print(str(exc), file=sys.stderr)
                    continue
                if cmd == "sources":
                    if (not ws.sources) and ws.base_snapshot_id:
                        snap_sources = _hydrate_sources_from_snapshot(client, ws.base_snapshot_id)
                        if snap_sources is not None:
                            ws.sources = snap_sources
                            state.docs_workspaces[name] = ws
                            save_state(state)
                    print(_format_sources_table(ws.sources))
                    continue
                if cmd == "rag":
                    if not args:
                        print(_format_rag_status(ws=ws, bm25_available=bm25_retriever.is_available()))
                        print(
                            "usage: /rag off | /rag backend=light|bm25 | /rag on | /rag k=<int> | /rag chars=<int> | "
                            "/rag per_source_chars=<int> | /rag debug on|off | /rag bm25_k_paragraphs=<int> | /rag bm25_k_sources=<int>"
                        )
                        continue

                    backend = _rag_backend_from_ws(ws)
                    cfg = _light_rag_config_from_ws(ws).sanitized()
                    k = int(cfg.k)
                    total_chars = int(cfg.total_chars)
                    per_source_chars = int(cfg.per_source_chars)
                    debug = bool(cfg.debug)
                    bm25_k_paragraphs = int(getattr(ws, "bm25_k_paragraphs", 40))
                    bm25_k_sources = int(getattr(ws, "bm25_k_sources", 0))
                    backend_requested_bm25 = False

                    ok = True
                    i = 0
                    while i < len(args):
                        a = args[i]
                        if a in {"on", "off"}:
                            if a == "off":
                                backend = "off"
                            elif backend == "off":
                                backend = "light"
                            i += 1
                            continue

                        if a == "backend":
                            if i + 1 >= len(args):
                                print("usage: /rag backend light|bm25|off", file=sys.stderr)
                                ok = False
                                break
                            b = args[i + 1].strip().lower()
                            if b not in {"light", "bm25", "off"}:
                                print(f"invalid /rag backend: {b!r}", file=sys.stderr)
                                ok = False
                                break
                            backend = b
                            backend_requested_bm25 = backend_requested_bm25 or (b == "bm25")
                            i += 2
                            continue

                        if a.startswith("backend="):
                            b = a.split("=", 1)[1].strip().lower()
                            if b not in {"light", "bm25", "off"}:
                                print(f"invalid /rag backend: {b!r}", file=sys.stderr)
                                ok = False
                                break
                            backend = b
                            backend_requested_bm25 = backend_requested_bm25 or (b == "bm25")
                            i += 1
                            continue

                        if a == "debug":
                            if i + 1 >= len(args) or args[i + 1] not in {"on", "off"}:
                                print("usage: /rag debug on|off", file=sys.stderr)
                                ok = False
                                break
                            debug = args[i + 1] == "on"
                            i += 2
                            continue

                        if a.startswith("k="):
                            try:
                                k = int(a.split("=", 1)[1])
                            except Exception:
                                print(f"invalid /rag k: {a!r}", file=sys.stderr)
                                ok = False
                                break
                            i += 1
                            continue

                        if a.startswith("chars="):
                            try:
                                total_chars = int(a.split("=", 1)[1])
                            except Exception:
                                print(f"invalid /rag chars: {a!r}", file=sys.stderr)
                                ok = False
                                break
                            i += 1
                            continue

                        if a.startswith("per_source_chars=") or a.startswith("per-source-chars="):
                            try:
                                per_source_chars = int(a.split("=", 1)[1])
                            except Exception:
                                print(f"invalid /rag per_source_chars: {a!r}", file=sys.stderr)
                                ok = False
                                break
                            i += 1
                            continue

                        if a.startswith("bm25_k_paragraphs=") or a.startswith("bm25-k-paragraphs="):
                            try:
                                bm25_k_paragraphs = int(a.split("=", 1)[1])
                            except Exception:
                                print(f"invalid /rag bm25_k_paragraphs: {a!r}", file=sys.stderr)
                                ok = False
                                break
                            i += 1
                            continue

                        if a.startswith("bm25_k_sources=") or a.startswith("bm25-k-sources="):
                            try:
                                bm25_k_sources = int(a.split("=", 1)[1])
                            except Exception:
                                print(f"invalid /rag bm25_k_sources: {a!r}", file=sys.stderr)
                                ok = False
                                break
                            i += 1
                            continue

                        print(f"unknown /rag option: {a!r}", file=sys.stderr)
                        ok = False
                        break

                    if not ok:
                        continue

                    new_cfg = LightRagConfig(
                        enabled=backend != "off",
                        k=k,
                        total_chars=total_chars,
                        per_source_chars=per_source_chars,
                        debug=debug,
                    ).sanitized()
                    _apply_rag_backend_to_ws(ws, backend)
                    _apply_light_rag_config_to_ws(ws, new_cfg)
                    ws.bm25_k_paragraphs = max(1, min(int(bm25_k_paragraphs), 1000))
                    ws.bm25_k_sources = max(0, min(int(bm25_k_sources), 50))
                    state.docs_workspaces[name] = ws
                    save_state(state)
                    if backend_requested_bm25 and backend == "bm25" and not bm25_retriever.is_available():
                        print(
                            "note: `rank-bm25` is not installed; BM25 backend will fall back to lightRAG "
                            "(install with `pip install rank-bm25`).",
                            file=sys.stderr,
                        )
                        bm25_hint_printed = True
                    if backend == "bm25" and bm25_retriever.is_available():
                        bm25_cfg = _bm25_rag_config_from_ws(ws)
                        for dbg in bm25_retriever.ensure_index(sources=ws.sources, debug=bm25_cfg.debug):
                            print(dbg, file=sys.stderr)
                    print(_format_rag_status(ws=ws, bm25_available=bm25_retriever.is_available()))
                    continue
                if cmd == "add":
                    if ws.phase != "INGEST":
                        print("cannot /add in QA phase; use /clear to return to INGEST", file=sys.stderr)
                        continue
                    # Parse --save / -s flag
                    save_after = "--save" in args or "-s" in args
                    path_args = [a for a in args if a not in ("--save", "-s")]
                    if not path_args:
                        print("usage: /add <paths...> [--save|-s]", file=sys.stderr)
                        continue
                    files = _expand_files(path_args)
                    if not files:
                        print("no files matched", file=sys.stderr)
                        continue

                    contents: list[tuple[str, str]] = []
                    new_sources: list[dict[str, Any]] = []
                    total_bytes = 0
                    now = int(time.time())
                    for p in files:
                        try:
                            text, nbytes, sha = _read_text_file(p)
                        except Exception as exc:
                            print(str(exc), file=sys.stderr)
                            continue
                        path_str = str(p.resolve())
                        meta = _extract_doc_metadata(path=p, text=text)
                        contents.append((path_str, text))
                        total_bytes += nbytes
                        new_sources.append(
                            {
                                "path": path_str,
                                **meta,
                                "bytes": nbytes,
                                "sha256": sha,
                                "added_at_unix_s": now,
                            }
                        )

                    if not contents:
                        print("no ingestible text files found", file=sys.stderr)
                        continue

                    include_ingest_prompt = (not ws.sources) and (ws.base_snapshot_id is None)
                    messages: list[dict[str, Any]] = []
                    if include_ingest_prompt:
                        ingest_prompt = system_prompt if system_prompt else DOCS_INGEST_PROMPT
                        messages.append({"role": "system", "content": ingest_prompt})
                    messages.append({"role": "user", "content": _build_docs_message(contents)})

                    pos_before = _get_session_pos(client, ws.session_id)
                    try:
                        ingest_stats = _stream_request(
                            client=client,
                            session_id=ws.session_id,
                            messages=messages,
                            max_completion_tokens=16,
                            think_budget=None,
                            temperature=temperature,
                            top_p=top_p,
                            print_output=False,
                        )
                        last_stats = ingest_stats
                    except KeyboardInterrupt:
                        print("(cancelled)")
                        continue
                    except DocsReplError as exc:
                        msg = str(exc)
                        _maybe_print_prompt_too_long_hint(msg=msg, requested_max_seq_len=max_seq_len)
                        print(msg, file=sys.stderr)
                        continue
                    except HttpError as exc:
                        if exc.status_code == 429:
                            print("Server is busy (429). Try again.", file=sys.stderr)
                        else:
                            print(str(exc), file=sys.stderr)
                        continue

                    # Update source list (dedupe by path+sha256).
                    existing_keys = {(s.get("path"), s.get("sha256")) for s in ws.sources if isinstance(s, dict)}
                    for s in new_sources:
                        key = (s.get("path"), s.get("sha256"))
                        if key in existing_keys:
                            continue
                        ws.sources.append(s)
                        existing_keys.add(key)

                    state.docs_workspaces[name] = ws
                    save_state(state)

                    if _rag_backend_from_ws(ws) == "bm25" and bm25_retriever.is_available():
                        bm25_cfg = _bm25_rag_config_from_ws(ws)
                        for dbg in bm25_retriever.ensure_index(sources=ws.sources, debug=bm25_cfg.debug):
                            print(dbg, file=sys.stderr)

                    pos_after = _get_session_pos(client, ws.session_id)
                    prompt_delta = None
                    if (
                        pos_before is not None
                        and pos_after is not None
                        and ingest_stats.completion_tokens is not None
                        and pos_after >= pos_before
                    ):
                        prompt_delta = max(pos_after - pos_before - int(ingest_stats.completion_tokens), 0)

                    extra = []
                    if prompt_delta is not None:
                        extra.append(f"prompt_delta_tokens={prompt_delta}")
                    if ingest_stats.server_prefill_s and prompt_delta is not None and ingest_stats.server_prefill_s > 0:
                        extra.append(f"prefill_tok/s={prompt_delta/ingest_stats.server_prefill_s:.2f}")

                    snap_msg = f" base_snapshot_id={ws.base_snapshot_id}" if ws.base_snapshot_id else ""
                    print(
                        f"added files={len(contents)} bytes={total_bytes} sources={len(ws.sources)}{snap_msg}"
                        + ((" " + " ".join(extra)) if extra else "")
                    )
                    footer = _stats_footer(ingest_stats)
                    if footer:
                        print(footer)

                    # Auto-save if --save/-s flag was passed
                    if save_after:
                        payload: dict[str, Any] = {"description": _encode_sources_description(ws.sources)}
                        try:
                            resp = client.request_json(
                                "POST",
                                f"/v1/sessions/{ws.session_id}/save",
                                payload=payload,
                                timeout_s=300.0,
                            )
                        except HttpError as exc:
                            if exc.status_code == 429:
                                print("Server is busy (429). Try again.", file=sys.stderr)
                            else:
                                print(str(exc), file=sys.stderr)
                            continue
                        snap_id = resp.get("snapshot_id") if isinstance(resp, dict) else None
                        if isinstance(snap_id, str) and snap_id:
                            ws.base_snapshot_id = snap_id
                            state.docs_workspaces[name] = ws
                            save_state(state)
                            print(f"saved base_snapshot_id={snap_id}")
                        else:
                            print("saved", file=sys.stderr)
                    continue
                if cmd == "reset":
                    if ws.base_snapshot_id:
                        try:
                            client.request_json(
                                "POST",
                                f"/v1/snapshots/{ws.base_snapshot_id}/load",
                                payload={"session_id": ws.session_id, "force": True},
                                timeout_s=300.0,
                            )
                        except HttpError as exc:
                            if exc.status_code == 404:
                                print(
                                    f"error: base snapshot not found: {ws.base_snapshot_id} (use `spl snapshot ls`).",
                                    file=sys.stderr,
                                )
                            else:
                                print(str(exc), file=sys.stderr)
                            continue
                        ws.phase = "INGEST"
                        state.docs_workspaces[name] = ws
                        save_state(state)
                        last_stats = None
                        last_substantive_user_question = None
                        bm25_hint_printed = False
                        light_retriever.clear_cache()
                        bm25_retriever.clear_index()
                        print(f"cleared to base_snapshot_id={ws.base_snapshot_id}")
                    else:
                        # No checkpoint yet: return to empty ingest state.
                        try:
                            client.request_json("DELETE", f"/v1/sessions/{ws.session_id}", timeout_s=30.0)
                        except HttpError:
                            pass
                        _create_session(client, ws.session_id)
                        ws.phase = "INGEST"
                        ws.sources = []
                        ws.base_snapshot_id = None
                        state.docs_workspaces[name] = ws
                        save_state(state)
                        last_stats = None
                        last_substantive_user_question = None
                        bm25_hint_printed = False
                        light_retriever.clear_cache()
                        bm25_retriever.clear_index()
                        print("cleared (empty)")
                    continue
                if cmd == "save":
                    if ws.phase != "INGEST":
                        print("docs /save is base-only; use /clear to return to INGEST first", file=sys.stderr)
                        continue
                    title = " ".join(args).strip() if args else ""
                    payload: dict[str, Any] = {"description": _encode_sources_description(ws.sources)}
                    if title:
                        payload["title"] = title
                    try:
                        resp = client.request_json(
                            "POST",
                            f"/v1/sessions/{ws.session_id}/save",
                            payload=payload,
                            timeout_s=300.0,
                        )
                    except HttpError as exc:
                        if exc.status_code == 429:
                            print("Server is busy (429). Try again.", file=sys.stderr)
                        else:
                            print(str(exc), file=sys.stderr)
                        continue
                    snap_id = resp.get("snapshot_id") if isinstance(resp, dict) else None
                    if isinstance(snap_id, str) and snap_id:
                        ws.base_snapshot_id = snap_id
                        state.docs_workspaces[name] = ws
                        save_state(state)
                        print(f"saved base_snapshot_id={snap_id}")
                    else:
                        print("saved", file=sys.stderr)
                    continue
                if cmd == "load":
                    if not args:
                        print("usage: /load <snapshot_id> --as <new_name>", file=sys.stderr)
                        continue
                    snap_id = args[0]
                    rest = args[1:]
                    if len(rest) != 2 or rest[0] != "--as":
                        print("usage: /load <snapshot_id> --as <new_name>", file=sys.stderr)
                        continue
                    new_name = rest[1]
                    if not new_name:
                        print("new_name must be non-empty", file=sys.stderr)
                        continue
                    if new_name in state.docs_workspaces:
                        print(f"workspace already exists: {new_name}", file=sys.stderr)
                        continue
                    new_session = _new_session_id(workspace_name=new_name)
                    next_lock = SessionLock(session_id=new_session, kind="docs", label=f"spl docs {new_name}")
                    try:
                        next_lock.acquire()
                    except AlreadyLockedError as exc:
                        print(
                            f"error: target session is already open in another REPL (session_id={new_session} pid={exc.info.pid}).",
                            file=sys.stderr,
                        )
                        continue

                    try:
                        client.request_json(
                            "POST",
                            f"/v1/snapshots/{snap_id}/load",
                            payload={"session_id": new_session},
                            timeout_s=300.0,
                        )
                    except HttpError as exc:
                        next_lock.release()
                        if exc.status_code == 404:
                            print(f"Snapshot not found: {snap_id} (use `spl snapshot ls`).", file=sys.stderr)
                        elif exc.status_code == 409:
                            print(f"Target session already exists: {new_session} (try again).", file=sys.stderr)
                        elif exc.status_code == 429:
                            print("Server is busy (429). Try again.", file=sys.stderr)
                        else:
                            print(str(exc), file=sys.stderr)
                        continue

                    new_sources = _hydrate_sources_from_snapshot(client, snap_id) or []
                    state.docs_workspaces[new_name] = DocsWorkspaceState(
                        session_id=new_session,
                        phase="INGEST",
                        base_snapshot_id=snap_id,
                        sources=new_sources,
                    )
                    save_state(state)

                    lock.release()
                    lock = next_lock
                    name = new_name
                    ws = state.docs_workspaces[name]
                    resumed = False
                    last_stats = None
                    light_retriever.clear_cache()
                    bm25_retriever.clear_index()
                    bm25_hint_printed = False
                    _banner(
                        url=client.base_url,
                        name=name,
                        session_id=ws.session_id,
                        resumed=False,
                        phase=ws.phase,
                        source_count=len(ws.sources),
                        base_snapshot_id=ws.base_snapshot_id,
                        rag_status=_format_rag_status(ws=ws, bm25_available=bm25_retriever.is_available()),
                    )
                    continue

                print(f"unknown command: /{cmd}", file=sys.stderr)
                continue

            # Regular question input.
            qa_messages: list[dict[str, Any]] = []
            if ws.phase == "INGEST":
                # Transition to QA; lock docs base.
                ws.phase = "QA"
                state.docs_workspaces[name] = ws
                save_state(state)
                # IMPORTANT: do not rely on a system message here. In session mode the server
                # drops additional system messages once a leading system already exists.
                qa_messages.append({"role": "user", "content": _build_qa_bootstrap_message(sources=ws.sources)})
                qa_messages.append({"role": "assistant", "content": DOCS_QA_PRIMER_ASSISTANT})

            backend = _rag_backend_from_ws(ws)
            rag_cfg = _light_rag_config_from_ws(ws)

            rag_question = line
            if ws.phase == "QA" and _should_augment_rag_query_with_prev_question(
                question=line, prev_question=last_substantive_user_question
            ):
                rag_question = f"{line}\n\nContext from previous user question: {last_substantive_user_question}".strip()

            rag_msg: str | None = None
            rag_debug: list[str] = []
            if backend == "light":
                rag_msg, rag_debug = light_retriever.build_retrieved_excerpts_message(
                    question=rag_question,
                    sources=ws.sources,
                    config=rag_cfg,
                )
            elif backend == "bm25":
                if not bm25_retriever.is_available():
                    if not bm25_hint_printed:
                        print(
                            "note: BM25 backend requested but `rank-bm25` is not installed; falling back to lightRAG "
                            "(install with `pip install rank-bm25`).",
                            file=sys.stderr,
                        )
                        bm25_hint_printed = True
                    rag_msg, rag_debug = light_retriever.build_retrieved_excerpts_message(
                        question=rag_question,
                        sources=ws.sources,
                        config=rag_cfg,
                    )
                else:
                    bm25_cfg = _bm25_rag_config_from_ws(ws)
                    rag_msg, rag_debug = bm25_retriever.build_retrieved_excerpts_message(
                        question=rag_question,
                        sources=ws.sources,
                        config=bm25_cfg,
                    )

            if rag_debug:
                for dbg in rag_debug:
                    print(dbg, file=sys.stderr)
            if rag_msg:
                qa_messages.append({"role": "user", "content": rag_msg})

            qa_messages.append({"role": "user", "content": line})

            # Track the last substantive user question so follow-up questions can reuse it for retrieval.
            # We intentionally do this after appending the message so the chat itself remains unchanged.
            cur_terms = tokenize_query_terms(line, max_terms=rag_cfg.max_terms)
            if cur_terms and not _should_augment_rag_query_with_prev_question(
                question=line, prev_question=last_substantive_user_question
            ):
                last_substantive_user_question = line

            try:
                stats = _stream_request(
                    client=client,
                    session_id=ws.session_id,
                    messages=qa_messages,
                    max_completion_tokens=32768,
                    think_budget=think_budget,
                    temperature=temperature,
                    top_p=top_p,
                    print_output=True,
                )
                last_stats = stats
                footer = _stats_footer(stats)
                if footer:
                    print(footer)
            except KeyboardInterrupt:
                print("\n(cancelled)")
                continue
            except DocsReplError as exc:
                msg = str(exc)
                _maybe_print_prompt_too_long_hint(msg=msg, requested_max_seq_len=max_seq_len)
                print(msg, file=sys.stderr)
                continue
            except HttpError as exc:
                if exc.status_code == 429:
                    print("Server is busy (429). Try again.", file=sys.stderr)
                else:
                    print(str(exc), file=sys.stderr)
                continue
            except Exception as exc:  # pragma: no cover
                print(f"error: {exc}", file=sys.stderr)
                continue
    except (DocsReplError, HttpError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            lock.release()
        except Exception:
            pass
