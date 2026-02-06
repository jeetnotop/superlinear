from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


# Used for normalization/matching where we intentionally work in lowercase.
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
# Used only for extracting original-cased query tokens (e.g. "Go", "AI").
_NON_ALNUM_RE_ORIG = re.compile(r"[^A-Za-z0-9]+")
_MULTISPACE_RE = re.compile(r"\s+")
_PARA_SPLIT_RE = re.compile(r"\n\s*\n+")


_SHORT_QUERY_TERMS_2 = frozenset(
    {
        "ai",
        "go",
        "ml",
        "rl",
    }
)


_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "also",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "with",
        "would",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",

        # Query-instruction / meta words (common in docs REPL prompts)
        "article",
        "articles",
        "mention",
        "mentions",
        "quoted",
        "quote",
        "quotes",
        "exact",
        "exactly",
        "sentence",
        "sentences",
        "substring",
        "end",
        "sources",
        "path",
        "containing",
        "contains",
        "yes",

        # More prompt-instruction words that otherwise pollute retrieval.
        "name",
        "short",
        "shorter",
        "shortest",
        "line",
        "lines",
        "fragment",
        "fragments",
        "sense",
        "verb",
        "verbatim",

        # More meta words common in evaluation prompts
        "ingested",
        "difference",
        "answer",
        "include",
        "available",
        "term",
        "terms",
        "one",

        # Prompt verbs/nouns that shouldn't drive lexical retrieval
        "find",
        "passage",
        "define",
        "defines",
        "definition",
        "imply",
        "implies",
        "say",
        "says",

        "explicitly",
        "explain",
        "extra",
        "detail",
        "details",
        "sentence",
        "sentences",

        "full",
        "ellipsis",
        "ellipses",
        "paraphrase",
        "using",
        "word",
        "words",
        "present",

        # Constraint/policy words frequently used in prompts
        "must",
        "found",

        # Generic question framing words that rarely help lexical retrieval
        "intuition",
        "problem",
        "problems",
        "solve",
        "solves",
        "solving",

        # Common prompt directives that shouldn't influence retrieval
        "use",
        "used",
        "provide",
        "provided",
        "providing",
        "retrieve",
        "retrieved",
        "excerpt",
        "excerpts",
        "copy",
        "copied",
        "given",

        # Conversation glue words (common in multi-turn follow-ups)
        "tell",
        "mentioned",
    }
)


_QUOTEY_RE = re.compile(
    r"\b(quote|verbatim|exact|substring|sentence|sentences|fragment|fragments|no\s+ellipses)\b",
    re.IGNORECASE,
)


def _looks_like_quote_task(question: str) -> bool:
    return bool(_QUOTEY_RE.search(question or ""))


def _split_into_sentences(text: str) -> list[str]:
    """Best-effort sentence splitter for excerpt selection.

    We intentionally keep it simple (no NLP deps). Newlines also act as boundaries.
    """

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
            # End sentence at punctuation; include it.
            end = i + 1
            seg = t[start:end].strip()
            if seg:
                out.append(seg)
            # Skip trailing whitespace.
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


def _select_sentence_snippet(text: str, *, terms: list[str], max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text

    sentences = _split_into_sentences(text)
    if not sentences:
        return _truncate_text(text, max_chars)

    # Prefer the shortest sentence that still matches the most terms.
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

    # Last resort: clip without adding an ellipsis (to avoid models copying it into "verbatim" quotes).
    return chosen[:max_chars].rstrip()


def tokenize_query_terms(text: str, *, max_terms: int = 32) -> list[str]:
    if not text or not isinstance(text, str):
        return []

    # Preserve original casing for acronym detection
    raw_terms_original = _NON_ALNUM_RE_ORIG.sub(" ", text).split()
    cleaned = _NON_ALNUM_RE.sub(" ", text.lower())
    raw_terms = cleaned.split()

    out: list[str] = []
    seen: set[str] = set()
    for i, term in enumerate(raw_terms):
        # Keep short terms (2 chars) only if they look like acronyms/proper nouns
        # (original had uppercase) e.g., "Go", "AI", "ML"
        original = raw_terms_original[i] if i < len(raw_terms_original) else term
        is_likely_name = len(original) >= 2 and original[0].isupper()
        keep_short = term in _SHORT_QUERY_TERMS_2
        
        if len(term) < 2:
            continue
        if len(term) == 2 and not (is_likely_name or keep_short):
            continue
        if term in _STOPWORDS:
            continue
        if term in seen:
            continue
        seen.add(term)
        out.append(term)
        if len(out) >= int(max_terms):
            break
    return out


def tokenize_rag_text(text: str, *, max_tokens: int | None = None) -> list[str]:
    """Tokenize arbitrary text for lexical/BM25 retrieval.

    The tokenization policy matches `tokenize_query_terms`, except:
    - tokens are **not** deduplicated (BM25 needs term frequencies)
    - the output can be optionally limited by `max_tokens`
    """

    if not text or not isinstance(text, str):
        return []

    raw_terms_original = _NON_ALNUM_RE_ORIG.sub(" ", text).split()
    cleaned = _NON_ALNUM_RE.sub(" ", text.lower())
    raw_terms = cleaned.split()

    out: list[str] = []
    for i, term in enumerate(raw_terms):
        original = raw_terms_original[i] if i < len(raw_terms_original) else term
        is_likely_name = len(original) >= 2 and original[0].isupper()
        keep_short = term in _SHORT_QUERY_TERMS_2

        if len(term) < 2:
            continue
        if len(term) == 2 and not (is_likely_name or keep_short):
            continue
        if term in _STOPWORDS:
            continue

        out.append(term)
        if max_tokens is not None and len(out) >= int(max_tokens):
            break

    return out


def split_paragraphs(text: str) -> list[str]:
    if not text or not isinstance(text, str):
        return []
    normalized = text.replace("\r", "").strip()
    if not normalized:
        return []
    parts = _PARA_SPLIT_RE.split(normalized)
    return [p.strip() for p in parts if p and p.strip()]


def _normalize_for_matching(text: str) -> str:
    cleaned = _NON_ALNUM_RE.sub(" ", (text or "").lower())
    cleaned = _MULTISPACE_RE.sub(" ", cleaned).strip()
    return f" {cleaned} " if cleaned else " "


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars == 1:
        return text[:1]
    return text[: max_chars - 1].rstrip() + "…"


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
class LightRagConfig:
    enabled: bool = True
    k: int = 5
    total_chars: int = 12000
    per_source_chars: int = 2600
    debug: bool = False

    candidate_sources: int = 20
    max_terms: int = 32
    max_paragraphs_per_source: int = 8
    max_paragraph_chars: int = 1200
    min_term_matches: int = 2  # Only include excerpts matching at least this many query terms

    def sanitized(self) -> "LightRagConfig":
        return replace(
            self,
            k=_coerce_int(self.k, default=5, min_v=1, max_v=50),
            total_chars=_coerce_int(self.total_chars, default=12000, min_v=200, max_v=200000),
            per_source_chars=_coerce_int(self.per_source_chars, default=2600, min_v=50, max_v=50000),
            candidate_sources=_coerce_int(self.candidate_sources, default=20, min_v=1, max_v=200),
            max_terms=_coerce_int(self.max_terms, default=32, min_v=1, max_v=128),
            max_paragraphs_per_source=_coerce_int(self.max_paragraphs_per_source, default=8, min_v=1, max_v=64),
            max_paragraph_chars=_coerce_int(self.max_paragraph_chars, default=1200, min_v=50, max_v=20000),
        )


@dataclass
class _DocCacheEntry:
    mtime_ns: int
    paragraphs: list[str]
    normalized_paragraphs: list[str]


class LightRagRetriever:
    def __init__(self, *, cache_docs: int = 64) -> None:
        self._cache_docs = int(cache_docs)
        self._cache: OrderedDict[str, _DocCacheEntry] = OrderedDict()

    def clear_cache(self) -> None:
        self._cache.clear()

    def _read_doc(self, path_str: str) -> _DocCacheEntry:
        path = Path(path_str)
        st = path.stat()
        mtime_ns = int(st.st_mtime_ns)

        cached = self._cache.get(path_str)
        if cached is not None and cached.mtime_ns == mtime_ns:
            self._cache.move_to_end(path_str)
            return cached

        data = path.read_bytes()
        if b"\x00" in data:
            raise ValueError("refusing to read binary file (NUL byte found)")
        text = data.decode("utf-8", errors="replace")

        paragraphs = split_paragraphs(text)
        normalized_paragraphs = [_normalize_for_matching(p) for p in paragraphs]
        entry = _DocCacheEntry(mtime_ns=mtime_ns, paragraphs=paragraphs, normalized_paragraphs=normalized_paragraphs)

        self._cache[path_str] = entry
        self._cache.move_to_end(path_str)
        while len(self._cache) > self._cache_docs:
            self._cache.popitem(last=False)

        return entry

    def build_retrieved_excerpts_message(
        self,
        *,
        question: str,
        sources: list[dict[str, Any]],
        config: LightRagConfig,
    ) -> tuple[str | None, list[str]]:
        cfg = config.sanitized()
        if not cfg.enabled:
            return None, []

        quote_task = _looks_like_quote_task(question)

        terms = tokenize_query_terms(question, max_terms=cfg.max_terms)
        if not terms:
            return None, []

        term_pats = [f" {t} " for t in terms]
        # Adaptive threshold: for short queries, require fewer matches.
        # Note: some queries include generic terms (e.g. "articles", "mention") that won't
        # appear in the docs; we apply a fallback later if the threshold filters everything out.
        effective_min_matches = min(cfg.min_term_matches, len(terms))

        meta_scored: list[tuple[int, int, dict[str, Any], str]] = []
        for idx, s in enumerate(sources):
            path = s.get("path")
            if not isinstance(path, str) or not path:
                continue

            title = s.get("title") if isinstance(s.get("title"), str) else ""
            src = s.get("source") if isinstance(s.get("source"), str) else ""
            meta_text = f"{title}\n{Path(path).name}\n{src}"
            meta_norm = _normalize_for_matching(meta_text)
            meta_score = sum(1 for pat in term_pats if pat in meta_norm)
            meta_scored.append((int(meta_score), idx, s, meta_norm))

        if not meta_scored:
            return None, []

        meta_scored.sort(key=lambda x: (-x[0], x[1]))

        # If the query terms do not appear in titles/filenames/source labels, meta scoring
        # provides no useful signal (many/most scores will be 0). In that case, avoid
        # over-pruning to the first N sources, which can miss the relevant doc purely due
        # to ingestion order.
        if meta_scored[0][0] <= 0:
            max_candidates = max(int(cfg.candidate_sources), 200)
            candidates = meta_scored[: min(len(meta_scored), max_candidates)]
        else:
            candidates = meta_scored[: cfg.candidate_sources]

        def _scan(threshold: int) -> tuple[
            list[tuple[int, int, int, int, dict[str, Any], list[tuple[int, int]], _DocCacheEntry]],
            list[str],
        ]:
            matches_local: list[
                tuple[int, int, int, int, dict[str, Any], list[tuple[int, int]], _DocCacheEntry]
            ] = []
            skipped_local: list[str] = []

            for meta_score, idx, s, meta_norm in candidates:
                path = s.get("path")
                if not isinstance(path, str) or not path:
                    continue
                try:
                    doc = self._read_doc(path)
                except Exception as exc:
                    skipped_local.append(f"{path}: {exc}")
                    continue

                para_hits: list[tuple[int, int]] = []

                # If a query term already matches the source metadata (title/filename/source label),
                # don't require it to appear in every paragraph. This prevents broad doc-selection
                # hints (e.g. "Transformer") from excluding the exact paragraph we want.
                term_pats_doc = term_pats
                if meta_score > 0 and meta_norm:
                    filtered = [pat for pat in term_pats if pat not in meta_norm]
                    if filtered:
                        term_pats_doc = filtered

                threshold_doc = min(int(threshold), len(term_pats_doc))
                if threshold_doc < 1:
                    threshold_doc = 1
                for p_idx, norm in enumerate(doc.normalized_paragraphs):
                    score = 0
                    for pat in term_pats_doc:
                        if pat in norm:
                            score += 1
                    if score >= threshold_doc:
                        para_hits.append((score, p_idx))

                if not para_hits:
                    continue

                para_hits.sort(key=lambda x: (-x[0], x[1]))
                top = para_hits[: cfg.max_paragraphs_per_source]

                best = int(top[0][0])
                total = int(sum(score for score, _ in top))
                matches_local.append((best, total, meta_score, idx, s, top, doc))

            return matches_local, skipped_local

        matches, skipped = _scan(effective_min_matches)
        used_threshold = int(effective_min_matches)
        if not matches and effective_min_matches > 1:
            # Fallback: if the threshold filters out everything, relax to 1 so we can still
            # retrieve entity hits like "AlphaGo" even when other query terms are generic.
            matches, skipped = _scan(1)
            used_threshold = 1

        matches.sort(key=lambda x: (-x[0], -x[1], -x[2], x[3]))
        selected = matches[: cfg.k]

        if not selected:
            debug = []
            if cfg.debug:
                debug = [
                    f"lightRAG: terms={terms!r}",
                    f"lightRAG: candidates={len(candidates)} scanned=0 selected=0 skipped={len(skipped)}",
                ]
                if skipped:
                    debug.append("lightRAG: skipped (read errors):")
                    debug.extend([f"  - {s}" for s in skipped[:20]])
            return None, debug

        total_remaining = int(cfg.total_chars)
        blocks: list[str] = ["Retrieved excerpts (hints for where to look - verify against your full memory of the documents):", ""]

        debug_lines: list[str] = []
        if cfg.debug:
            debug_lines.append(f"lightRAG: terms={terms!r}")
            debug_lines.append(f"lightRAG: min_matches_used={used_threshold}")
            debug_lines.append(
                f"lightRAG: candidates={len(candidates)} scanned={len(matches)} selected={min(len(selected), cfg.k)} skipped={len(skipped)}"
            )

        included = 0
        for best, total, meta_score, idx, s, top, doc in selected:
            if total_remaining <= 0:
                break

            per_remaining = min(int(cfg.per_source_chars), total_remaining)
            if per_remaining <= 0:
                break

            para_indices = sorted({p_idx for _, p_idx in top})
            parts: list[str] = []
            used = 0

            for p_idx in para_indices:
                para = doc.paragraphs[p_idx].strip()
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

            attrs: list[str] = []
            path = s.get("path")
            if isinstance(path, str) and path:
                attrs.append(f"path={json.dumps(path, ensure_ascii=False)}")
            title = s.get("title")
            if isinstance(title, str) and title.strip():
                attrs.append(f"title={json.dumps(title.strip(), ensure_ascii=False)}")
            src = s.get("source")
            if isinstance(src, str) and src.strip():
                attrs.append(f"source={json.dumps(src.strip(), ensure_ascii=False)}")
            url = s.get("url")
            if isinstance(url, str) and url.strip():
                attrs.append(f"url={json.dumps(url.strip(), ensure_ascii=False)}")

            blocks.append(f"[SOURCE {' '.join(attrs)}]")
            blocks.append(excerpt)
            blocks.append("[/SOURCE]")
            blocks.append("")

            if cfg.debug:
                debug_lines.append(
                    f"lightRAG: + {path} best={best} total={total} meta={meta_score} paras={len(para_indices)} chars={used}"
                )

        if included == 0:
            return None, debug_lines

        if cfg.debug and skipped:
            debug_lines.append("lightRAG: skipped (read errors):")
            debug_lines.extend([f"  - {s}" for s in skipped[:20]])

        msg = "\n".join(blocks).rstrip() + "\n"
        return msg, debug_lines
