"""Tests for deterministic, LLM-free tool-schema filtering."""

from __future__ import annotations

import pytest

from memahead import Step, ToolFilter, filter_tools
from memahead.tool_filter import _tool_description, _tool_name


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web and gather raw research facts.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "theme_extractor",
            "description": "Identify key themes across research findings.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "image_generator",
            "description": "Generate decorative images and artwork.",
        },
    },
]

FLAT_TOOLS = [
    {"name": "web_search", "description": "Search the web and gather research facts."},
    {"name": "image_generator", "description": "Generate images and artwork."},
]


# -- schema extraction -----------------------------------------------------


def test_extracts_name_and_description_from_function_envelope():
    assert _tool_name(TOOLS[0]) == "web_search"
    assert "Search the web" in _tool_description(TOOLS[0])


def test_extracts_name_and_description_from_flat_schema():
    assert _tool_name(FLAT_TOOLS[0]) == "web_search"
    assert "Search the web" in _tool_description(FLAT_TOOLS[0])


# -- semantic filtering with embeddings ------------------------------------


def test_keeps_relevant_tool_drops_irrelevant(keyword_embedder):
    tf = ToolFilter(embedder=keyword_embedder, threshold=0.55)
    step = Step("synthesize", "identify themes across research")

    matches = {m.name: m.kept for m in tf.match(TOOLS, step)}
    assert matches["theme_extractor"] is True
    assert matches["image_generator"] is False


def test_filter_returns_only_kept(keyword_embedder):
    tf = ToolFilter(embedder=keyword_embedder, threshold=0.55)
    step = Step("research", "search and gather research facts")
    kept = tf.filter(TOOLS, step)
    names = [_tool_name(t) for t in kept]
    assert "web_search" in names
    assert "image_generator" not in names


# -- lexical fallback (no embeddings) --------------------------------------


def test_lexical_fallback_matches_overlap():
    kept = filter_tools(
        TOOLS,
        Step("research", "gather research facts"),
        use_embeddings=False,
        threshold=0.05,
    )
    names = [_tool_name(t) for t in kept]
    assert "web_search" in names
    assert "image_generator" not in names


# -- behaviors -------------------------------------------------------------


def test_min_tools_guarantees_minimum(keyword_embedder):
    # An impossible threshold would normally drop everything...
    tf = ToolFilter(embedder=keyword_embedder, threshold=1.0, min_tools=1)
    step = Step("synthesize", "identify themes")
    kept = tf.filter(TOOLS, step)
    assert len(kept) == 1  # min_tools rescues the single best match


def test_empty_tools_returns_empty(keyword_embedder):
    tf = ToolFilter(embedder=keyword_embedder)
    assert tf.filter([], Step("a", "x")) == []
    assert tf.match([], Step("a", "x")) == []


def test_threshold_validation():
    with pytest.raises(ValueError):
        ToolFilter(threshold=1.5)
    with pytest.raises(ValueError):
        ToolFilter(min_tools=-1)


def test_match_preserves_input_order(keyword_embedder):
    tf = ToolFilter(embedder=keyword_embedder, threshold=0.0)
    matches = tf.match(TOOLS, Step("research", "research facts"))
    assert [m.name for m in matches] == ["web_search", "theme_extractor", "image_generator"]


def test_filter_tools_function_shortcut(keyword_embedder):
    kept = filter_tools(
        TOOLS,
        "identify themes across research",
        embedder=keyword_embedder,
        threshold=0.55,
    )
    names = [_tool_name(t) for t in kept]
    assert "theme_extractor" in names
