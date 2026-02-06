from __future__ import annotations

import importlib
import json
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from apps.cli.light_rag import split_paragraphs, tokenize_query_terms, tokenize_rag_text


_QUOTEY_RE = re.compile(
    r"\b(quote|verbatim|exact|substring|sentence|sentences|fragment|fragments|no\s+ellipses)\b",
    re.IGNORECASE,
)


def _looks_like_quote_task(question: str) -> bool:
    return bool(_QUOTEY_RE.search(question or ""))


def _split_into_sentences(text: str) -> list[str]:
    t = (text or "").replace("\r", "")
    if not t:
        return []

    out: list[str] = []
    start = 0
    i = 0
    n = len(t)
    while i < n:
        ch = t[i]
        if ch == "\n":
            seg = t[start:i].strip()
            if seg:
                out.append(seg)
            start = i + 1
        elif ch in {".", "!", "?"}:
            end = i + 1
            seg = t[start:end].strip()
            if seg:
                out.append(seg)
            j = end
            while j < n and t[j].isspace() and t[j] != "\n":
                j += 1
            start = j
            i = j - 1
        i += 1

    tail = t[start:].strip()
    if tail:
        out.append(tail)
    return out


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars == 1:
        return text[:1]
    return text[: max_chars - 1].rstrip() + "…"


def _select_sentence_snippet(text: str, *, terms: list[str], max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text

    sentences = _split_into_sentences(text)
    if not sentences:
        return _truncate_text(text, max_chars)

    best: tuple[int, int, str] | None = None  # (score, -len, sentence)
    for s in sentences:
        s_l = s.lower()
        score = sum(1 for t in terms if t and t in s_l)
        if score <= 0:
            continue
        cand = (score, -len(s), s)
        if best is None or cand > best:
            best = cand

    chosen = (best[2] if best is not None else sentences[0]).strip()
    if len(chosen) <= max_chars:
        return chosen

    # Clip without adding an ellipsis (to avoid models copying it into "verbatim" quotes).
    return chosen[:max_chars].rstrip()


def _coerce_int(v: Any, *, default: int, min_v: int, max_v: int) -> int:
    try:
        n = int(v)
    except Exception:
        return default
    if n < min_v:
        return min_v
    if n > max_v:
        return max_v
    return n


@dataclass(frozen=True)
class Bm25RagConfig:
    enabled: bool = True
    k_sources: int = 5
    total_chars: int = 12000
    per_source_chars: int = 2600
    debug: bool = False

    k_paragraphs: int = 40
    max_terms: int = 32
    max_paragraphs_per_source: int = 8
    max_paragraph_chars: int = 1200

    def sanitized(self) -> "Bm25RagConfig":
        return replace(
            self,
            k_sources=_coerce_int(self.k_sources, default=5, min_v=1, max_v=50),
            total_chars=_coerce_int(self.total_chars, default=12000, min_v=200, max_v=200000),
            per_source_chars=_coerce_int(self.per_source_chars, default=2600, min_v=50, max_v=50000),
            k_paragraphs=_coerce_int(self.k_paragraphs, default=40, min_v=1, max_v=1000),
            max_terms=_coerce_int(self.max_terms, default=32, min_v=1, max_v=256),
            max_paragraphs_per_source=_coerce_int(
                self.max_paragraphs_per_source, default=8, min_v=1, max_v=64
            ),
            max_paragraph_chars=_coerce_int(self.max_paragraph_chars, default=1200, min_v=50, max_v=20000),
        )


@dataclass(frozen=True)
class _Paragraph:
    path: str
    paragraph_index: int
    text: str


class Bm25RagRetriever:
    def __init__(self) -> None:
        self._bm25_cls: type | None = None
        self._bm25_import_error: str | None = None
        self._index_key: str | None = None

        self._paragraphs: list[_Paragraph] = []
        self._paragraph_tokens: list[list[str]] = []
        self._source_meta: dict[str, dict[str, Any]] = {}
        self._bm25: Any | None = None
        self._last_build_ms: int | None = None

    def is_available(self) -> bool:
        return self._get_bm25_cls() is not None

    def last_build_stats(self) -> dict[str, Any]:
        return {
            "sources": len(self._source_meta),
            "paragraphs": len(self._paragraphs),
            "build_ms": self._last_build_ms,
        }

    def clear_index(self) -> None:
        self._index_key = None
        self._paragraphs = []
        self._paragraph_tokens = []
        self._source_meta = {}
        self._bm25 = None
        self._last_build_ms = None

    def _get_bm25_cls(self) -> type | None:
        if self._bm25_cls is not None:
            return self._bm25_cls
        if self._bm25_import_error is not None:
            return None

        try:
            mod = importlib.import_module("rank_bm25")
            cls = getattr(mod, "BM25Okapi", None)
            if cls is None:
                self._bm25_import_error = "rank_bm25.BM25Okapi not found"
                return None
            self._bm25_cls = cls
            return cls
        except Exception as exc:
            self._bm25_import_error = str(exc)
            return None

    def _sources_key(self, sources: list[dict[str, Any]]) -> str:
        # Use (path, sha256) when available so we can detect content changes across /add.
        # Sort for stability.
        items: list[tuple[str, str]] = []
        for s in sources:
            if not isinstance(s, dict):
                continue
            path = s.get("path")
            if not isinstance(path, str) or not path:
                continue
            sha = s.get("sha256")
            items.append((path, sha if isinstance(sha, str) else ""))
        items.sort()
        return json.dumps({"v": 1, "sources": items}, ensure_ascii=False, sort_keys=True)

    def ensure_index(self, *, sources: list[dict[str, Any]], debug: bool = False) -> list[str]:
        dbg: list[str] = []
        bm25_cls = self._get_bm25_cls()
        if bm25_cls is None:
            if debug:
                hint = (
                    "bm25: unavailable (install `rank-bm25` to enable BM25 retrieval)"
                    if self._bm25_import_error is None
                    else f"bm25: unavailable ({self._bm25_import_error})"
                )
                dbg.append(hint)
            self.clear_index()
            return dbg

        key = self._sources_key(sources)
        if self._index_key == key and self._bm25 is not None:
            return dbg

        t0 = time.perf_counter()

        paragraphs: list[_Paragraph] = []
        paragraph_tokens: list[list[str]] = []
        source_meta: dict[str, dict[str, Any]] = {}

        skipped: list[str] = []

        # Deduplicate by path; keep the last metadata entry for a path.
        seen_paths: set[str] = set()
        unique_sources: list[dict[str, Any]] = []
        for s in reversed(sources):
            if not isinstance(s, dict):
                continue
            path = s.get("path")
            if not isinstance(path, str) or not path:
                continue
            if path in seen_paths:
                continue
            seen_paths.add(path)
            unique_sources.append(s)
        unique_sources.reverse()

        for s in unique_sources:
            path = s.get("path")
            if not isinstance(path, str) or not path:
                continue

            title = s.get("title")
            src = s.get("source")
            url = s.get("url")
            meta: dict[str, Any] = {"path": path}
            if isinstance(title, str) and title.strip():
                meta["title"] = title.strip()
            if isinstance(src, str) and src.strip():
                meta["source"] = src.strip()
            if isinstance(url, str) and url.strip():
                meta["url"] = url.strip()
            source_meta[path] = meta

            try:
                data = Path(path).read_bytes()
                if b"\x00" in data:
                    raise ValueError("refusing to read binary file (NUL byte found)")
                text = data.decode("utf-8", errors="replace")
            except Exception as exc:
                skipped.append(f"{path}: {exc}")
                continue

            for p_idx, para in enumerate(split_paragraphs(text)):
                tokens = tokenize_rag_text(para)
                if not tokens:
                    continue
                paragraphs.append(_Paragraph(path=path, paragraph_index=p_idx, text=para))
                paragraph_tokens.append(tokens)

        if not paragraphs:
            self._index_key = key
            self._paragraphs = []
            self._paragraph_tokens = []
            self._source_meta = source_meta
            self._bm25 = None
            self._last_build_ms = int((time.perf_counter() - t0) * 1000)
            if debug:
                dbg.append(
                    f"bm25: index empty (sources={len(source_meta)} paragraphs=0 build_ms={self._last_build_ms})"
                )
                if skipped:
                    dbg.append("bm25: skipped (read errors):")
                    dbg.extend([f"  - {s}" for s in skipped[:20]])
            return dbg

        bm25 = bm25_cls(paragraph_tokens)

        self._index_key = key
        self._paragraphs = paragraphs
        self._paragraph_tokens = paragraph_tokens
        self._source_meta = source_meta
        self._bm25 = bm25
        self._last_build_ms = int((time.perf_counter() - t0) * 1000)

        if debug:
            dbg.append(
                f"bm25: index built (sources={len(source_meta)} paragraphs={len(paragraphs)} build_ms={self._last_build_ms})"
            )
            if skipped:
                dbg.append("bm25: skipped (read errors):")
                dbg.extend([f"  - {s}" for s in skipped[:20]])

        return dbg

    def build_retrieved_excerpts_message(
        self,
        *,
        question: str,
        sources: list[dict[str, Any]],
        config: Bm25RagConfig,
    ) -> tuple[str | None, list[str]]:
        cfg = config.sanitized()
        if not cfg.enabled:
            return None, []

        terms = tokenize_query_terms(question, max_terms=cfg.max_terms)
        if not terms:
            return None, []

        debug_lines: list[str] = []
        debug_lines.extend(self.ensure_index(sources=sources, debug=cfg.debug))
        if self._bm25 is None or not self._paragraphs:
            return None, debug_lines

        quote_task = _looks_like_quote_task(question)

        try:
            scores_raw = self._bm25.get_scores(terms)
        except Exception as exc:
            if cfg.debug:
                debug_lines.append(f"bm25: scoring failed ({exc}); falling back")
            return None, debug_lines

        try:
            scores = list(scores_raw)
        except Exception:
            scores = [scores_raw[i] for i in range(len(self._paragraphs))]

        scored: list[tuple[float, int]] = []
        for i, s in enumerate(scores[: len(self._paragraphs)]):
            try:
                f = float(s)
            except Exception:
                continue
            if f <= 0:
                continue
            scored.append((f, i))

        if not scored:
            if cfg.debug:
                debug_lines.append(f"bm25: terms={terms!r}")
                debug_lines.append("bm25: no positive-scoring paragraphs")
            return None, debug_lines

        scored.sort(key=lambda x: (-x[0], x[1]))
        top_para = scored[: cfg.k_paragraphs]

        by_path: dict[str, list[tuple[float, int]]] = {}
        for score, pid in top_para:
            path = self._paragraphs[pid].path
            by_path.setdefault(path, []).append((score, pid))

        source_scored: list[tuple[float, str]] = []
        for path, items in by_path.items():
            agg = float(sum(score for score, _ in items))
            source_scored.append((agg, path))
        source_scored.sort(key=lambda x: (-x[0], x[1]))

        selected_sources = source_scored[: cfg.k_sources]
        if not selected_sources:
            return None, debug_lines

        if cfg.debug:
            debug_lines.append(f"bm25: terms={terms!r}")
            debug_lines.append(
                f"bm25: selected_sources={len(selected_sources)} from_paths={len(by_path)} top_paragraphs={len(top_para)}"
            )

        total_remaining = int(cfg.total_chars)
        blocks: list[str] = [
            "Retrieved excerpts (hints for where to look - verify against your full memory of the documents):",
            "",
        ]

        included = 0
        for agg, path in selected_sources:
            if total_remaining <= 0:
                break

            per_remaining = min(int(cfg.per_source_chars), total_remaining)
            if per_remaining <= 0:
                break

            items = by_path.get(path, [])
            items.sort(key=lambda x: (-x[0], x[1]))
            items = items[: cfg.max_paragraphs_per_source]
            items.sort(key=lambda x: self._paragraphs[x[1]].paragraph_index)

            parts: list[str] = []
            used = 0
            for score, pid in items:
                para = self._paragraphs[pid].text.strip()
                if not para:
                    continue

                sep = "\n\n" if parts else ""
                avail = per_remaining - used - len(sep)
                if avail <= 0:
                    break

                clip_limit = min(int(cfg.max_paragraph_chars), avail)
                if quote_task:
                    clipped = _select_sentence_snippet(para, terms=terms, max_chars=clip_limit)
                else:
                    clipped = _truncate_text(para, clip_limit)
                if not clipped:
                    break

                parts.append(sep + clipped)
                used += len(sep) + len(clipped)
                if used >= per_remaining:
                    break

            excerpt = "".join(parts).strip()
            if not excerpt:
                continue

            included += 1
            total_remaining -= used

            meta = self._source_meta.get(path, {"path": path})
            attrs: list[str] = []
            attrs.append(f"path={json.dumps(path, ensure_ascii=False)}")

            title = meta.get("title")
            if isinstance(title, str) and title.strip():
                attrs.append(f"title={json.dumps(title.strip(), ensure_ascii=False)}")
            src = meta.get("source")
            if isinstance(src, str) and src.strip():
                attrs.append(f"source={json.dumps(src.strip(), ensure_ascii=False)}")
            url = meta.get("url")
            if isinstance(url, str) and url.strip():
                attrs.append(f"url={json.dumps(url.strip(), ensure_ascii=False)}")

            blocks.append(f"[SOURCE {' '.join(attrs)}]")
            blocks.append(excerpt)
            blocks.append("[/SOURCE]")
            blocks.append("")

            if cfg.debug:
                top_scores = [f"{score:.3f}" for score, _ in sorted(items, reverse=True)[:3]]
                debug_lines.append(
                    f"bm25: + {path} agg={agg:.3f} paras={len(items)} chars={used} top_scores={top_scores}"
                )

        if included == 0:
            return None, debug_lines

        msg = "\n".join(blocks).rstrip() + "\n"
        return msg, debug_lines

