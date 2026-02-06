from __future__ import annotations

from apps.cli.docs_repl import _should_augment_rag_query_with_prev_question


def test_followup_anaphora_triggers_augmentation():
    assert _should_augment_rag_query_with_prev_question(
        question="which article mentioned this?",
        prev_question="tell me about lti",
    )


def test_short_entity_query_does_not_trigger_augmentation():
    assert not _should_augment_rag_query_with_prev_question(
        question="RoPE?",
        prev_question="tell me about lti",
    )


def test_no_prev_question_no_augmentation():
    assert not _should_augment_rag_query_with_prev_question(
        question="which article mentioned this?",
        prev_question=None,
    )
