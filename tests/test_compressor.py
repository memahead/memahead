"""Tests for the PlanAwareCompressor pipeline and result containers."""

from __future__ import annotations

import sys
import types

import pytest

from memahead import (
    BudgetExceededError,
    CompressedContext,
    Plan,
    PlanAwareCompressor,
    PlanGraph,
    Step,
    TokenReport,
    count_tokens,
)
from memahead.context import DroppedChunk


@pytest.fixture
def plan() -> Plan:
    return Plan(
        [
            Step("research", "Search and gather raw research facts"),
            Step("synthesize", "Identify key themes"),
            Step("draft", "Write a structured draft"),
            Step("revise", "Polish the final output"),
        ]
    )


@pytest.fixture
def history():
    return [
        {"role": "system", "content": "You are a research assistant."},
        {"role": "assistant", "content": "research facts and themes gathered"},
        {"role": "user", "content": "weather lunch small talk"},
        {"role": "user", "content": "please continue"},
    ]


@pytest.fixture
def tools():
    return [
        {"function": {"name": "web_search", "description": "Search and gather research facts."}},
        {"function": {"name": "theme_extractor", "description": "Identify key themes research."}},
        {"function": {"name": "image_generator", "description": "Generate decorative artwork."}},
    ]


def make_compressor(keyword_embedder, **kwargs):
    kwargs.setdefault("use_headroom", False)
    kwargs.setdefault("retention_threshold", 0.6)
    kwargs.setdefault("tool_threshold", 0.55)
    return PlanAwareCompressor(embedder=keyword_embedder, **kwargs)


# -- retention -------------------------------------------------------------


def test_drops_irrelevant_keeps_relevant(keyword_embedder, plan, history, tools):
    comp = make_compressor(keyword_embedder)
    result = comp.compress(history, tools, plan, "research")

    assert isinstance(result, CompressedContext)
    contents = [m["content"] for m in result.messages]
    # The future-relevant chunk survives; the small-talk chunk is gone.
    assert "research facts and themes gathered" in contents
    assert "weather lunch small talk" not in contents

    dropped_sources = result.report.dropped_sources()
    assert "message[2]" in dropped_sources


def test_always_keeps_system_and_last_message(keyword_embedder, plan, history, tools):
    comp = make_compressor(keyword_embedder)
    result = comp.compress(history, tools, plan, "research")

    roles = [m["role"] for m in result.messages]
    assert roles[0] == "system"  # system preserved
    assert result.messages[-1]["content"] == "please continue"  # last preserved


def test_filters_tools_to_current_step(keyword_embedder, plan, history, tools):
    comp = make_compressor(keyword_embedder)
    result = comp.compress(history, tools, plan, "research")

    kept_names = [t["function"]["name"] for t in result.tools]
    assert "web_search" in kept_names
    assert "theme_extractor" in kept_names
    assert "image_generator" not in kept_names


def test_token_report_shows_savings(keyword_embedder, plan, history, tools):
    comp = make_compressor(keyword_embedder)
    result = comp.compress(history, tools, plan, "research")
    report = result.report

    assert report.before > report.after
    assert report.saved == report.before - report.after
    assert 0.0 < report.compression_ratio <= 1.0
    assert report.saved > 0


def test_retained_scores_recorded(keyword_embedder, plan, history, tools):
    comp = make_compressor(keyword_embedder)
    result = comp.compress(history, tools, plan, "research")
    assert "message[1]" in result.retained_scores
    assert 0.0 <= result.retained_scores["message[1]"] <= 1.0


def test_dropped_tool_recorded(keyword_embedder, plan, history, tools):
    comp = make_compressor(keyword_embedder)
    result = comp.compress(history, tools, plan, "research")
    tool_drops = [d for d in result.report.dropped if d.kind == "tool"]
    assert any("image_generator" in d.source for d in tool_drops)


def test_last_step_has_no_future_keeps_all_messages(keyword_embedder, plan, history, tools):
    comp = make_compressor(keyword_embedder)
    result = comp.compress(history, tools, plan, "revise")

    # No remaining steps => nothing pruned by retention.
    message_drops = [d for d in result.report.dropped if d.kind == "message"]
    assert message_drops == []
    assert len(result.messages) == len(history)
    # Tools are still filtered even on the last step.
    assert "image_generator" not in [t["function"]["name"] for t in result.tools]


def test_quality_controls_aggressiveness(keyword_embedder, plan, history, tools):
    # Without an absolute threshold, lower quality should drop at least as much.
    gentle = PlanAwareCompressor(
        embedder=keyword_embedder, use_headroom=False, quality=0.95, tool_threshold=0.55
    ).compress(history, tools, plan, "research")
    aggressive = PlanAwareCompressor(
        embedder=keyword_embedder, use_headroom=False, quality=0.1, tool_threshold=0.55
    ).compress(history, tools, plan, "research")

    gentle_msgs = len([d for d in gentle.report.dropped if d.kind == "message"])
    aggressive_msgs = len([d for d in aggressive.report.dropped if d.kind == "message"])
    assert aggressive_msgs >= gentle_msgs


def test_works_with_plangraph(keyword_embedder, history, tools):
    g = PlanGraph()
    g.add_step(Step("research", "gather research facts"))
    g.add_step(Step("synthesize", "identify themes"), depends_on=["research"])
    g.add_step(Step("draft", "write draft"), depends_on=["synthesize"])

    comp = make_compressor(keyword_embedder)
    result = comp.compress(history, tools, g, "research")
    assert isinstance(result, CompressedContext)
    assert "weather lunch small talk" not in [m["content"] for m in result.messages]


# -- validation ------------------------------------------------------------


def test_invalid_quality_raises():
    with pytest.raises(ValueError):
        PlanAwareCompressor(quality=2.0)
    with pytest.raises(ValueError):
        PlanAwareCompressor(retention_threshold=5.0)
    with pytest.raises(ValueError):
        PlanAwareCompressor(budget_tokens=0)


# -- budget enforcement ----------------------------------------------------


@pytest.fixture
def sample_workflow(plan, history, tools):
    return {
        "history": history,
        "tools": tools,
        "plan": plan,
        "current_step": "research",
    }


def test_budget_respected(keyword_embedder, sample_workflow):
    compressor = PlanAwareCompressor(
        embedder=keyword_embedder,
        use_headroom=False,
        retention_threshold=0.6,
        tool_threshold=0.55,
        budget_tokens=500,
    )
    result = compressor.compress(**sample_workflow)
    assert result.report.after <= 500
    assert result.report.budget_tokens == 500
    assert result.report.budget_utilization is not None
    assert result.report.budget_utilization <= 1.0


def test_budget_none_unchanged_behavior(keyword_embedder, sample_workflow):
    common = dict(
        embedder=keyword_embedder,
        use_headroom=False,
        retention_threshold=0.6,
        tool_threshold=0.55,
    )
    result_no_budget = PlanAwareCompressor(budget_tokens=None, **common).compress(
        **sample_workflow
    )
    result_legacy = PlanAwareCompressor(**common).compress(**sample_workflow)
    assert result_no_budget.report.after == result_legacy.report.after


def test_budget_drops_lowest_scores_first(keyword_embedder, sample_workflow):
    compressor = PlanAwareCompressor(
        embedder=keyword_embedder,
        use_headroom=False,
        quality=0.5,
        tool_threshold=0.55,
        budget_tokens=300,
    )
    result = compressor.compress(**sample_workflow)
    assert result.report.chunks_dropped_for_budget >= 0


def test_budget_never_drops_system_messages(keyword_embedder, sample_workflow):
    compressor = PlanAwareCompressor(
        embedder=keyword_embedder,
        use_headroom=False,
        quality=0.3,
        tool_threshold=0.55,
        budget_tokens=200,
    )
    result = compressor.compress(**sample_workflow)
    system_messages = [m for m in result.messages if m.get("role") == "system"]
    assert len(system_messages) > 0


def test_budget_exceeded_error(keyword_embedder, sample_workflow):
    compressor = PlanAwareCompressor(
        embedder=keyword_embedder,
        use_headroom=False,
        quality=0.99,
        tool_threshold=0.55,
        budget_tokens=10,
    )
    with pytest.raises(BudgetExceededError) as exc_info:
        compressor.compress(**sample_workflow)
    assert exc_info.value.requested_budget == 10
    assert exc_info.value.minimum_achievable > 10


def test_token_report_budget_fields(keyword_embedder, sample_workflow):
    compressor = PlanAwareCompressor(
        embedder=keyword_embedder,
        use_headroom=False,
        retention_threshold=0.6,
        tool_threshold=0.55,
        budget_tokens=1000,
    )
    result = compressor.compress(**sample_workflow)
    assert result.report.budget_tokens == 1000
    assert result.report.budget_utilization is not None
    assert 0.0 <= result.report.budget_utilization <= 1.0


# -- headroom integration --------------------------------------------------


def test_headroom_is_invoked_when_available(keyword_embedder, plan, history, tools, monkeypatch):
    calls = {}

    def fake_compress(messages, model=None):
        calls["invoked"] = True
        calls["model"] = model
        return [{"role": m["role"], "content": "HR:" + m.get("content", "")} for m in messages]

    fake_module = types.ModuleType("headroom")
    fake_module.compress = fake_compress
    monkeypatch.setitem(sys.modules, "headroom", fake_module)

    comp = PlanAwareCompressor(
        embedder=keyword_embedder,
        use_headroom=True,
        retention_threshold=0.6,
        tool_threshold=0.55,
    )
    result = comp.compress(history, tools, plan, "research")

    assert calls.get("invoked") is True
    assert all(m["content"].startswith("HR:") for m in result.messages)


def test_headroom_failure_falls_back(keyword_embedder, plan, history, tools, monkeypatch):
    def broken_compress(messages, model=None):
        raise RuntimeError("boom")

    fake_module = types.ModuleType("headroom")
    fake_module.compress = broken_compress
    monkeypatch.setitem(sys.modules, "headroom", fake_module)

    comp = PlanAwareCompressor(
        embedder=keyword_embedder,
        use_headroom=True,
        retention_threshold=0.6,
        tool_threshold=0.55,
    )
    result = comp.compress(history, tools, plan, "research")
    # Falls back to retained (uncompressed) messages without raising.
    assert "research facts and themes gathered" in [m["content"] for m in result.messages]


def test_normalize_headroom_result_shapes():
    comp = PlanAwareCompressor(use_headroom=False)
    fallback = [{"role": "user", "content": "x"}]

    assert comp._normalize_headroom_result(None, fallback) is fallback
    assert comp._normalize_headroom_result(["a"], fallback) == ["a"]

    class Obj:
        messages = ["m"]

    assert comp._normalize_headroom_result(Obj(), fallback) == ["m"]
    assert comp._normalize_headroom_result({"messages": ["d"]}, fallback) == ["d"]
    assert comp._normalize_headroom_result(42, fallback) is fallback


# -- result containers -----------------------------------------------------


def test_token_report_repr_and_ratio():
    report = TokenReport(before=12400, after=3100)
    assert report.saved == 9300
    assert report.compression_ratio == 0.75
    assert "before=12400" in repr(report)
    assert "after=3100" in repr(report)
    assert "saved=9300" in repr(report)


def test_token_report_zero_before():
    report = TokenReport(before=0, after=0)
    assert report.compression_ratio == 0.0
    assert report.saved == 0


def test_dropped_chunk_tokens_saved():
    d = DroppedChunk(source="message[0]", kind="message", tokens_before=10, tokens_after=3)
    assert d.tokens_saved == 7


def test_count_tokens_behavior():
    assert count_tokens("") == 0
    short = count_tokens("hello world")
    longer = count_tokens("hello world this is a much longer sentence with more tokens")
    assert longer > short
    assert short > 0
