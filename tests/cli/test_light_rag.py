from __future__ import annotations

from pathlib import Path

from apps.cli.light_rag import LightRagConfig, LightRagRetriever, split_paragraphs, tokenize_query_terms


def _first_source_excerpt(msg: str) -> str:
    lines = msg.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("[SOURCE "))
    end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("[/SOURCE]"))
    return "\n".join(lines[start + 1 : end])


def test_tokenize_query_terms_filters_and_dedupes():
    terms = tokenize_query_terms("What does the design doc say about concurrency and concurrency?")
    assert "what" not in terms
    assert "does" not in terms
    assert "the" not in terms
    assert "about" not in terms
    assert "design" in terms
    assert "doc" in terms
    assert terms.count("concurrency") == 1


def test_tokenize_query_terms_keeps_short_acronyms_and_names():
    # 2-letter acronyms/proper nouns should survive tokenization.
    terms = tokenize_query_terms("which article mentioned the game of Go?")
    assert "go" in terms

    terms = tokenize_query_terms("what is AI?")
    assert "ai" in terms


def test_tokenize_query_terms_drops_quote_directives():
    q = (
        "Which article mentions the game of Go (the board game)? "
        "Name the article and quote the shortest line/fragment that contains Go in that sense (not the verb)."
    )
    terms = tokenize_query_terms(q)
    # Keep content terms
    assert "go" in terms
    assert "game" in terms
    assert "board" in terms
    # Drop directive words
    for bad in ["name", "quote", "shortest", "line", "fragment", "sense", "verb", "article", "mentions"]:
        assert bad not in terms


def test_tokenize_query_terms_drops_substring_directive():
    q = (
        "Which article mentions \"reinforcement learning\"? "
        "Name the article and quote the shortest line/fragment that contains the exact substring \"reinforcement learning\"."
    )
    terms = tokenize_query_terms(q)
    assert "reinforcement" in terms
    assert "learning" in terms
    assert "substring" not in terms


def test_tokenize_query_terms_drops_sources_format_directives():
    q = (
        "Which article mentions \"self-attention\"? "
        "Name the article, quote the shortest line/fragment containing the exact substring \"self-attention\", "
        "and end with Sources: <path>."
    )
    terms = tokenize_query_terms(q)
    assert "self" in terms
    assert "attention" in terms
    for bad in ["end", "sources", "path", "substring"]:
        assert bad not in terms


def test_tokenize_query_terms_drops_concept_question_meta_words():
    q = (
        "In the ingested articles, what is the difference between self-attention and cross-attention? "
        "Answer in 3–6 sentences, and include one short quote for each term if available."
    )
    terms = tokenize_query_terms(q)
    assert "self" in terms
    assert "attention" in terms
    assert "cross" in terms
    for bad in ["ingested", "difference", "answer", "include", "available", "term", "one"]:
        assert bad not in terms


def test_tokenize_query_terms_drops_definition_prompt_meta_words():
    q = "Find a passage that defines cross-attention (not just implies it). Quote 1–2 sentences verbatim and say which article it’s from."
    terms = tokenize_query_terms(q)
    assert "cross" in terms
    assert "attention" in terms
    for bad in ["find", "passage", "defines", "implies", "say"]:
        assert bad not in terms


def test_tokenize_query_terms_drops_quote_only_meta_words():
    q = (
        "Quote 1–2 sentences that explicitly mention \"cross-attention\" by name, "
        "and explain in one sentence what those sentences say (no extra details)."
    )
    terms = tokenize_query_terms(q)
    assert "cross" in terms
    assert "attention" in terms
    for bad in ["explicitly", "explain", "extra", "details", "sentence", "sentences"]:
        assert bad not in terms


def test_tokenize_query_terms_drops_paraphrase_meta_words():
    q = (
        "Quote the full sentence(s) that include the exact substring cross-attention (no ellipses), "
        "then paraphrase using only words present in the quote."
    )
    terms = tokenize_query_terms(q)
    assert "cross" in terms
    assert "attention" in terms
    for bad in ["full", "ellipses", "paraphrase", "using", "words", "present", "quote"]:
        assert bad not in terms


def test_tokenize_query_terms_drops_must_found_meta_words():
    q = (
        "Find and quote exactly one full sentence from the ingested articles that contains the exact substring "
        "cross-attention (must include that substring verbatim). If you can’t find one, say not found"
    )
    terms = tokenize_query_terms(q)
    assert "cross" in terms
    assert "attention" in terms
    assert "must" not in terms
    assert "found" not in terms


def test_tokenize_query_terms_drops_generic_framing_words():
    q = "What is the intuition behind RoPE and what problem does it solve?"
    terms = tokenize_query_terms(q)
    assert "rope" in terms
    for bad in ["intuition", "problem", "solve"]:
        assert bad not in terms


def test_retriever_does_not_overprune_when_meta_scores_zero(tmp_path: Path):
    # Build many sources where titles/filenames don't include query terms, and the only
    # matching doc is beyond the default candidate_sources cutoff.
    sources = []
    for i in range(50):
        p = tmp_path / f"doc_{i}.txt"
        p.write_text("Unrelated content about something else.\n\nMore text.\n", encoding="utf-8")
        sources.append({"path": str(p), "title": f"Doc {i}"})

    # Put the only match at the end (index 49).
    Path(sources[-1]["path"]).write_text(
        "This document defines RoPE (rotary positional embedding) in transformers.\n",
        encoding="utf-8",
    )

    cfg = LightRagConfig(
        enabled=True,
        k=1,
        total_chars=2000,
        per_source_chars=2000,
        debug=True,
        candidate_sources=20,
        min_term_matches=2,
    )
    retriever = LightRagRetriever()

    msg, debug = retriever.build_retrieved_excerpts_message(
        question="What is RoPE?",
        sources=sources,
        config=cfg,
    )
    assert msg is not None
    assert "RoPE" in msg
    assert any("min_matches_used=" in d for d in debug)


def test_tokenize_query_terms_drops_retrieved_excerpts_directives():
    q = (
        "Use ONLY the provided Retrieved excerpts. In 4–7 sentences, explain what RoPE is and "
        "the intuition given in the Transformer article. Include exactly one short quote copied verbatim from the excerpts."
    )
    terms = tokenize_query_terms(q)
    assert "rope" in terms
    assert "transformer" in terms
    for bad in ["use", "provided", "retrieved", "excerpts", "given", "include", "copied", "verbatim"]:
        assert bad not in terms


def test_retriever_meta_matched_terms_not_required_in_paragraph(tmp_path: Path):
    # The title matches "Transformer", but the paragraph doesn't need to contain that word.
    p = tmp_path / "t.txt"
    p.write_text(
        "RoPE (rotary positional embedding) rotates pairs of features by an angle depending on position.\n",
        encoding="utf-8",
    )
    sources = [{"path": str(p), "title": "Transformer (deep learning)"}]
    cfg = LightRagConfig(
        enabled=True,
        k=1,
        total_chars=2000,
        per_source_chars=2000,
        debug=True,
        candidate_sources=1,
        min_term_matches=2,
    )
    retriever = LightRagRetriever()

    msg, _debug = retriever.build_retrieved_excerpts_message(
        question="Explain RoPE in the Transformer article.",
        sources=sources,
        config=cfg,
    )
    assert msg is not None
    assert "RoPE" in msg


def test_quote_task_excerpt_prefers_full_sentence(tmp_path: Path):
    p = tmp_path / "t.txt"
    p.write_text(
        "Intro text that is long and irrelevant. "
        "Second sentence about attention. "
        "If the attention head is used in a cross-attention fashion, then usually it uses encoder outputs. "
        "Tail.\n",
        encoding="utf-8",
    )

    sources = [{"path": str(p), "title": "T"}]
    cfg = LightRagConfig(enabled=True, k=1, total_chars=4000, per_source_chars=500, debug=False)
    retriever = LightRagRetriever()

    msg, _ = retriever.build_retrieved_excerpts_message(
        question="Find and quote exactly one full sentence that contains the exact substring cross-attention.",
        sources=sources,
        config=cfg,
    )
    assert msg is not None
    excerpt = _first_source_excerpt(msg)
    assert "cross-attention" in excerpt
    # Should include the full sentence ending with a period.
    assert "cross-attention fashion, then usually it uses encoder outputs." in excerpt


def test_split_paragraphs_blank_lines():
    assert split_paragraphs("a\n\nb\n\n\nc\n") == ["a", "b", "c"]


def test_retriever_respects_budgets_and_truncates(tmp_path: Path):
    p = tmp_path / "a.md"
    long_para = "Concurrency and caching: " + ("x" * 400)
    p.write_text(f"{long_para}\n\nUnrelated.\n", encoding="utf-8")

    sources = [{"path": str(p), "title": "Alpha"}]
    cfg = LightRagConfig(enabled=True, k=1, total_chars=200, per_source_chars=80, debug=True)
    retriever = LightRagRetriever()

    msg, debug = retriever.build_retrieved_excerpts_message(
        question="How do concurrency and caching interact?",
        sources=sources,
        config=cfg,
    )
    assert msg is not None
    assert msg.startswith("Retrieved excerpts")
    assert "[SOURCE" in msg and "[/SOURCE]" in msg
    assert any(d.startswith("lightRAG: terms=") for d in debug)
    assert any(d.startswith("lightRAG: + ") for d in debug)

    excerpt = _first_source_excerpt(msg)
    assert len(excerpt) <= cfg.sanitized().per_source_chars
    assert excerpt.endswith("…")


def test_retriever_fallback_allows_single_entity_hits(tmp_path: Path):
    p = tmp_path / "ai.txt"
    p.write_text(
        "AlphaGo won 4 out of 5 games of Go in a match with Lee Sedol.\n\nOther text.\n",
        encoding="utf-8",
    )

    sources = [{"path": str(p), "title": "Artificial intelligence"}]
    # min_term_matches=2 would normally exclude paragraphs that only match 'alphago'
    cfg = LightRagConfig(enabled=True, k=1, total_chars=4000, per_source_chars=4000, debug=True, min_term_matches=2)
    retriever = LightRagRetriever()

    msg, debug = retriever.build_retrieved_excerpts_message(
        question="Does any of the articles mention AlphaGo?",
        sources=sources,
        config=cfg,
    )
    assert msg is not None
    assert "AlphaGo" in msg
    assert any("min_matches_used=" in d for d in debug)


def test_retriever_skips_missing_files(tmp_path: Path):
    missing = tmp_path / "missing.md"
    sources = [{"path": str(missing), "title": "Missing"}]
    cfg = LightRagConfig(enabled=True, k=1, total_chars=200, per_source_chars=80, debug=True)
    retriever = LightRagRetriever()

    msg, debug = retriever.build_retrieved_excerpts_message(
        question="concurrency",
        sources=sources,
        config=cfg,
    )
    assert msg is None
    assert any("skipped" in d for d in debug)

