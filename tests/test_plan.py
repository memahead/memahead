"""Tests for plan modeling: Step, Plan, and PlanGraph."""

from __future__ import annotations

import pytest

from memahead import Plan, PlanGraph, Step


# -- Step ------------------------------------------------------------------


def test_step_requires_name():
    with pytest.raises(ValueError):
        Step("")
    with pytest.raises(ValueError):
        Step("   ")


def test_step_as_text_includes_description():
    assert Step("research", "gather facts").as_text() == "research: gather facts"
    assert Step("research").as_text() == "research"


# -- Plan ------------------------------------------------------------------


@pytest.fixture
def plan() -> Plan:
    return Plan(
        [
            Step("research", "Search and gather raw facts"),
            Step("synthesize", "Identify key themes"),
            Step("draft", "Write a structured first draft"),
            Step("revise", "Produce the final polished output"),
        ]
    )


def test_plan_basic_collection_protocol(plan):
    assert len(plan) == 4
    assert plan.names() == ["research", "synthesize", "draft", "revise"]
    assert plan[0].name == "research"
    assert "draft" in plan
    assert "missing" not in plan
    assert [s.name for s in plan] == plan.names()


def test_plan_rejects_empty():
    with pytest.raises(ValueError):
        Plan([])


def test_plan_rejects_duplicate_names():
    with pytest.raises(ValueError):
        Plan([Step("a", "x"), Step("a", "y")])


def test_plan_coerces_tuples_and_strings():
    plan = Plan([("a", "first"), "b", Step("c", "third")])
    assert plan.names() == ["a", "b", "c"]
    assert plan.get("a").description == "first"
    assert plan.get("b").description == ""


def test_remaining_from_returns_future_steps(plan):
    remaining = plan.remaining_from("synthesize")
    assert [s.name for s in remaining] == ["draft", "revise"]


def test_remaining_from_last_step_is_empty(plan):
    assert plan.remaining_from("revise") == []


def test_remaining_from_inclusive(plan):
    remaining = plan.remaining_from("draft", inclusive=True)
    assert [s.name for s in remaining] == ["draft", "revise"]


def test_remaining_from_unknown_step_raises(plan):
    with pytest.raises(KeyError):
        plan.remaining_from("nope")


def test_completed_before(plan):
    done = plan.completed_before("draft")
    assert [s.name for s in done] == ["research", "synthesize"]


def test_index_of_and_get(plan):
    assert plan.index_of("draft") == 2
    assert plan.get("draft").description == "Write a structured first draft"
    with pytest.raises(KeyError):
        plan.index_of("ghost")


def test_plan_equality(plan):
    same = Plan(
        [
            Step("research", "Search and gather raw facts"),
            Step("synthesize", "Identify key themes"),
            Step("draft", "Write a structured first draft"),
            Step("revise", "Produce the final polished output"),
        ]
    )
    assert plan == same
    assert plan != Plan([Step("research", "x")])


# -- PlanGraph -------------------------------------------------------------


def test_plangraph_linear_remaining():
    g = PlanGraph()
    g.add_step(Step("research", "gather"))
    g.add_step(Step("draft", "write"), depends_on=["research"])
    g.add_step(Step("revise", "polish"), depends_on=["draft"])

    names = [s.name for s in g.remaining_from("research")]
    assert names == ["draft", "revise"]
    assert g.remaining_from("revise") == []


def test_plangraph_branching_remaining_is_transitive():
    g = PlanGraph()
    g.add_step(Step("research", "gather facts"))
    g.add_step(Step("draft", "write draft"), depends_on=["research"])
    g.add_step(Step("cite", "add citations"), depends_on=["research"])
    g.add_step(Step("publish", "ship it"), depends_on=["draft", "cite"])

    downstream = sorted(s.name for s in g.remaining_from("research"))
    assert downstream == ["cite", "draft", "publish"]

    # A leaf-adjacent node only sees its own downstream.
    assert [s.name for s in g.remaining_from("draft")] == ["publish"]


def test_plangraph_unknown_dependency_raises():
    g = PlanGraph()
    with pytest.raises(KeyError):
        g.add_step(Step("draft", "write"), depends_on=["nonexistent"])


def test_plangraph_duplicate_step_raises():
    g = PlanGraph()
    g.add_step(Step("a", "x"))
    with pytest.raises(ValueError):
        g.add_step(Step("a", "y"))


def test_plangraph_topological_order():
    g = PlanGraph()
    g.add_step(Step("research", "gather"))
    g.add_step(Step("draft", "write"), depends_on=["research"])
    g.add_step(Step("cite", "cite"), depends_on=["research"])
    g.add_step(Step("publish", "ship"), depends_on=["draft", "cite"])

    order = [s.name for s in g.topological_order()]
    # research must come first; publish must come last.
    assert order[0] == "research"
    assert order[-1] == "publish"
    assert order.index("draft") < order.index("publish")
    assert order.index("cite") < order.index("publish")


def test_plangraph_dependencies_of():
    g = PlanGraph()
    g.add_step(Step("research", "gather"))
    g.add_step(Step("draft", "write"), depends_on=["research"])
    assert [s.name for s in g.dependencies_of("draft")] == ["research"]
    assert g.dependencies_of("research") == []
