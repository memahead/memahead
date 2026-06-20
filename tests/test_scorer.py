"""Tests for the RetentionScorer — the forward-looking scoring core."""

from __future__ import annotations

import numpy as np
import pytest

from memahead import RetentionScorer, Step
from memahead.scorer import ChunkScore


def test_scores_in_unit_interval(keyword_embedder):
    scorer = RetentionScorer(embedder=keyword_embedder)
    chunks = ["research facts about the topic", "weather lunch chit chat"]
    steps = [Step("synthesize", "identify themes from research")]

    results = scorer.score(chunks, steps)
    assert len(results) == len(chunks)
    for r in results:
        assert isinstance(r, ChunkScore)
        assert 0.0 <= r.score <= 1.0


def test_relevant_chunk_scores_higher_than_irrelevant(keyword_embedder):
    scorer = RetentionScorer(embedder=keyword_embedder)
    steps = [
        Step("synthesize", "identify themes from research"),
        Step("draft", "write draft and polish"),
    ]
    chunks = [
        "raw research facts and themes",  # overlaps future steps
        "weather and lunch small talk",   # overlaps nothing future
    ]

    scores = scorer.score_values(chunks, steps)
    assert scores[0] > scores[1]


def test_per_step_alignment(keyword_embedder):
    scorer = RetentionScorer(embedder=keyword_embedder)
    steps = [Step("a", "research facts"), Step("b", "draft write"), Step("c", "polish")]
    results = scorer.score(["research facts everywhere"], steps)

    assert len(results[0].per_step) == len(steps)
    # The chunk overlaps step "a" most strongly.
    per_step = results[0].per_step
    assert per_step[0] == max(per_step)


def test_empty_remaining_steps_gives_zero(keyword_embedder):
    scorer = RetentionScorer(embedder=keyword_embedder)
    results = scorer.score(["anything at all"], [])
    assert results[0].score == 0.0
    assert results[0].per_step == []


def test_empty_chunks_returns_empty(keyword_embedder):
    scorer = RetentionScorer(embedder=keyword_embedder)
    assert scorer.score([], [Step("a", "x")]) == []


def test_max_vs_mean_aggregation(keyword_embedder):
    steps = [Step("a", "research facts"), Step("b", "weather lunch")]
    chunk = ["research facts only"]

    max_score = RetentionScorer(embedder=keyword_embedder, aggregate="max").score_values(
        chunk, steps
    )[0]
    mean_score = RetentionScorer(embedder=keyword_embedder, aggregate="mean").score_values(
        chunk, steps
    )[0]

    # The chunk strongly matches one step but not the other, so max > mean.
    assert max_score > mean_score


def test_invalid_aggregate_raises():
    with pytest.raises(ValueError):
        RetentionScorer(aggregate="median")


def test_accepts_plain_string_steps(keyword_embedder):
    scorer = RetentionScorer(embedder=keyword_embedder)
    scores = scorer.score_values(["research facts"], ["research facts ahead"])
    assert scores[0] > 0.5


def test_embedder_with_encode_method(keyword_embedder):
    class Wrapper:
        def encode(self, texts):
            return keyword_embedder(texts)

    scorer = RetentionScorer(embedder=Wrapper())
    scores = scorer.score_values(["research facts"], [Step("s", "research themes")])
    assert 0.0 <= scores[0] <= 1.0


def test_bad_embedder_shape_raises():
    def bad_embedder(texts):
        return np.zeros(len(texts), dtype=np.float32)  # 1-D, wrong

    scorer = RetentionScorer(embedder=bad_embedder)
    with pytest.raises(ValueError):
        scorer.score(["x"], [Step("a", "y")])
