"""Span search module for finding top-k anchor positions along power-law stripes."""

from .api import (
    span_search,
    search_spans,
    span_search_with_values,
    span_search_with_values_gqa,
    span_search_triton,
    span_search_triton_gqa,
    search_spans_triton,
    search_spans_triton_gqa,
    span_search_triton_with_values,
    span_search_triton_with_values_gqa,
    get_search_mask,
    get_search_scores,
    get_spans,
)

__all__ = [
    "span_search",
    "search_spans",
    "span_search_with_values",
    "span_search_with_values_gqa",
    "span_search_triton",
    "span_search_triton_gqa",
    "search_spans_triton",
    "search_spans_triton_gqa",
    "span_search_triton_with_values",
    "span_search_triton_with_values_gqa",
    "get_search_mask",
    "get_search_scores",
    "get_spans",
]
