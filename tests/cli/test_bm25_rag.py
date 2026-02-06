from __future__ import annotations

import hashlib
import sys
import types

from apps.cli.bm25_rag import Bm25RagConfig, Bm25RagRetriever
from apps.cli.light_rag import tokenize_query_terms, tokenize_rag_text


def _install_rank_bm25_stub(monkeypatch) -> None:
    mod = types.ModuleType("rank_bm25")

    class BM25Okapi:
        def __init__(self, corpus: list[list[str]]) -> None:
            self._doc_sets = [set(doc) for doc in corpus]

        def get_scores(self, query: list[str]) -> list[float]:
            q = set(query)
            return [float(len(q & s)) for s in self._doc_sets]

    mod.BM25Okapi = BM25Okapi  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rank_bm25", mod)


def test_tokenize_rag_text_matches_query_policy_and_keeps_dupes():
    text = "AI AI and Go go."
    assert tokenize_query_terms(text, max_terms=32) == ["ai", "go"]
    tokens = tokenize_rag_text(text)
    assert tokens.count("ai") == 2
    assert "and" not in tokens


def test_bm25_retrieval_prefers_matching_paragraph(tmp_path, monkeypatch):
    _install_rank_bm25_stub(monkeypatch)

    p1 = tmp_path / "doc1.txt"
    p2 = tmp_path / "doc2.txt"

    p1.write_text("Intro paragraph.\n\nAlpha appears here.\n\nTail.", encoding="utf-8")
    p2.write_text("Unrelated content only.\n\nNothing to see.", encoding="utf-8")

    sources = [
        {"path": str(p1), "title": "Doc 1"},
        {"path": str(p2), "title": "Doc 2"},
    ]

    r = Bm25RagRetriever()
    cfg = Bm25RagConfig(enabled=True, k_sources=1, k_paragraphs=10, total_chars=2000, per_source_chars=2000)
    msg, debug = r.build_retrieved_excerpts_message(question="Where is alpha mentioned?", sources=sources, config=cfg)

    assert debug == []  # debug off by default
    assert msg is not None
    assert str(p1) in msg
    assert "Alpha appears here." in msg
    assert str(p2) not in msg


def test_bm25_rebuilds_on_sha_change(tmp_path, monkeypatch):
    _install_rank_bm25_stub(monkeypatch)

    p = tmp_path / "doc.txt"
    p.write_text("Alpha paragraph.\n\nOther.", encoding="utf-8")
    sha1 = hashlib.sha256(p.read_bytes()).hexdigest()

    r = Bm25RagRetriever()
    cfg = Bm25RagConfig(enabled=True, k_sources=1, k_paragraphs=10, total_chars=2000, per_source_chars=2000)

    msg1, _ = r.build_retrieved_excerpts_message(
        question="alpha?", sources=[{"path": str(p), "sha256": sha1}], config=cfg
    )
    assert msg1 is not None
    assert "Alpha paragraph." in msg1

    p.write_text("Beta paragraph.\n\nOther.", encoding="utf-8")
    sha2 = hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha2 != sha1

    msg2, _ = r.build_retrieved_excerpts_message(
        question="beta?", sources=[{"path": str(p), "sha256": sha2}], config=cfg
    )
    assert msg2 is not None
    assert "Beta paragraph." in msg2
