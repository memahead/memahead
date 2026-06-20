"""Sanity checks for the benchmark suite (fast, no model download)."""

from __future__ import annotations

from benchmarks.quality.scorer import QualityScorer
from benchmarks.tokens import count_context_tokens
from benchmarks.workflows import code_review, data_analysis, research_synthesis


def test_workflows_meet_minimum_token_budget():
    for mod, step in (
        (research_synthesis, "synthesize"),
        (code_review, "summarize"),
        (data_analysis, "interpret"),
    ):
        wf = mod.get_workflow(step)
        tokens = count_context_tokens(wf["history"], wf["tools"])
        assert len(wf["history"]) >= 15
        assert tokens >= 3000, f"{mod.__name__} has only {tokens} tokens"
        assert len(mod.CRITICAL_FACTS) >= 5


def test_quality_scorer_fraction():
    scorer = QualityScorer()
    assert scorer.score("abc", "abc xyz", ["abc", "missing"]) == 0.5


def test_workflow_exports():
    wf = research_synthesis.get_workflow()
    assert "plan" in wf and "history" in wf and "tools" in wf
