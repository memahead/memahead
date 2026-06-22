"""Tests for the optional LangGraph integration."""

from __future__ import annotations

import warnings
from typing import TypedDict

import pytest

pytest.importorskip("langgraph")

from langgraph.graph import END, START, StateGraph

from memahead import PlanAwareCompressor
from memahead.context import CompressedContext, TokenReport
from memahead.integrations.langgraph import (
    compress_graph,
    compress_node,
    plan_from_graph,
)


class WorkflowState(TypedDict, total=False):
    messages: list
    tools: list
    _memahead_report: object


class RecordingCompressor:
    """Stub compressor that records inputs and optionally shrinks context."""

    def __init__(self, *, shrink: bool = True) -> None:
        self.calls: list[dict] = []
        self.shrink = shrink

    def compress(self, history, tools, plan, current_step):
        self.calls.append(
            {
                "history": list(history),
                "tools": list(tools or []),
                "current_step": current_step,
                "plan": plan,
            }
        )
        messages = list(history)
        if self.shrink and messages:
            messages = messages[:1]
        tool_list = list(tools or [])
        if self.shrink and tool_list:
            tool_list = tool_list[:1]
        return CompressedContext(
            messages=messages,
            tools=tool_list,
            report=TokenReport(before=100, after=10),
        )


def _linear_builder():
    def step_a(state):
        """First step."""
        return {"messages": state["messages"] + ["a"]}

    def step_b(state):
        """Second step."""
        return {"messages": state["messages"] + ["b"]}

    def step_c(state):
        """Third step."""
        return {"messages": state["messages"] + ["c"]}

    builder = StateGraph(WorkflowState)
    builder.add_node("step_a", step_a)
    builder.add_node("step_b", step_b)
    builder.add_node("step_c", step_c)
    builder.add_edge(START, "step_a")
    builder.add_edge("step_a", "step_b")
    builder.add_edge("step_b", "step_c")
    builder.add_edge("step_c", END)
    return builder, step_a, step_b, step_c


def test_plan_from_graph_linear():
    builder, *_ = _linear_builder()
    plan = plan_from_graph(builder)
    assert [step.name for step in plan.steps] == ["step_a", "step_b", "step_c"]


def test_description_from_docstring():
    builder, *_ = _linear_builder()
    plan = plan_from_graph(builder)
    assert plan.get("step_b").description == "Second step"


def test_description_explicit_override():
    builder, *_ = _linear_builder()
    plan = plan_from_graph(builder, step_descriptions={"step_b": "explicit"})
    assert plan.get("step_b").description == "explicit"


def test_description_name_fallback():
    def unnamed(state):
        return state

    builder = StateGraph(WorkflowState)
    builder.add_node("fallback", unnamed)
    builder.add_edge(START, "fallback")
    builder.add_edge("fallback", END)
    plan = plan_from_graph(builder)
    assert plan.get("fallback").description == "fallback"


def test_cyclic_graph_warns_and_degrades():
    def loop_a(state):
        return state

    def loop_b(state):
        return state

    builder = StateGraph(WorkflowState)
    builder.add_node("loop_a", loop_a)
    builder.add_node("loop_b", loop_b)
    builder.add_edge(START, "loop_a")
    builder.add_edge("loop_a", "loop_b")
    builder.add_edge("loop_b", "loop_a")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plan = plan_from_graph(builder)

    assert any("Cyclic LangGraph detected" in str(w.message) for w in caught)
    assert getattr(plan, "_cyclic", False) is True
    assert {step.name for step in plan.steps} == {"loop_a", "loop_b"}


def test_compress_node_preserves_signature():
    builder, _, step_b, _ = _linear_builder()
    recorder = RecordingCompressor()
    wrapped = compress_node(step_b, builder, compressor=recorder, node_name="step_b")

    result = wrapped({"messages": ["one", "two"], "tools": [{"name": "t"}]})
    assert isinstance(result, dict)
    assert "messages" in result
    assert result["messages"] == ["one", "b"]
    assert "_memahead_report" in result


def test_compress_node_compresses_history():
    builder, _, step_b, _ = _linear_builder()
    recorder = RecordingCompressor()
    wrapped = compress_node(step_b, builder, compressor=recorder, node_name="step_b")

    wrapped({"messages": ["one", "two", "three"], "tools": [{"name": "tool"}]})
    assert recorder.calls[0]["current_step"] == "step_b"
    assert recorder.calls[0]["history"] == ["one", "two", "three"]


def test_compress_graph_preserves_topology():
    builder, step_a, step_b, step_c = _linear_builder()
    recorder = RecordingCompressor(shrink=False)
    compress_graph(builder, compressor=recorder, exclude={"step_a"})

    compiled = builder.compile()
    result = compiled.invoke({"messages": [], "tools": []})
    assert result["messages"] == ["a", "b", "c"]
    assert [call["current_step"] for call in recorder.calls] == ["step_b", "step_c"]


def test_compress_graph_exclude():
    builder, step_a, step_b, step_c = _linear_builder()
    recorder = RecordingCompressor()
    compress_graph(builder, compressor=recorder, exclude={"step_a", "step_c"})

    compiled = builder.compile()
    compiled.invoke({"messages": ["seed"], "tools": []})
    assert len(recorder.calls) == 1
    assert recorder.calls[0]["current_step"] == "step_b"


def test_conditional_edges_union_remaining():
    def router(state):
        """Route to a branch."""
        return state

    def branch_a(state):
        """Branch A."""
        return state

    def branch_b(state):
        """Branch B."""
        return state

    builder = StateGraph(WorkflowState)
    builder.add_node("router", router)
    builder.add_node("branch_a", branch_a)
    builder.add_node("branch_b", branch_b)
    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        lambda state: "branch_a",
        {"branch_a": "branch_a", "branch_b": "branch_b"},
    )
    builder.add_edge("branch_a", END)
    builder.add_edge("branch_b", END)

    plan = plan_from_graph(builder)
    names = {step.name for step in plan.remaining_from("router")}
    assert names == {"branch_a", "branch_b"}
