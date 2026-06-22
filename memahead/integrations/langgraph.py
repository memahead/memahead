"""LangGraph integration for plan-aware context compression.

Install with ``pip install memahead[langgraph]``.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from memahead import Plan, PlanAwareCompressor, Step
from memahead.context import CompressedContext, TokenReport

try:
    from langgraph.graph import END, START, StateGraph
    from langgraph._internal._runnable import RunnableCallable
except ImportError as exc:  # pragma: no cover - exercised via import test
    raise ImportError(
        "LangGraph integration requires the optional langgraph dependency. "
        "Install it with: pip install memahead[langgraph]"
    ) from exc

__all__ = [
    "plan_from_graph",
    "compress_node",
    "compress_graph",
]

GraphLike = Union[StateGraph, Any]
VIRTUAL_NODES = frozenset({START, END, "__start__", "__end__"})
_PLAN_CACHE_ATTR = "_memahead_plan_cache"


class _CyclicPlanAdapter:
    """Plan-like object for cyclic graphs with all-other-steps remaining."""

    def __init__(self, plan: Plan) -> None:
        self._plan = plan
        self._cyclic = True

    @property
    def steps(self) -> List[Step]:
        return self._plan.steps

    def get(self, step_name: str) -> Step:
        return self._plan.get(step_name)

    def __contains__(self, item: object) -> bool:
        return item in self._plan

    def remaining_from(self, step_name: str, *, inclusive: bool = False) -> List[Step]:
        if inclusive:
            return list(self._plan.steps)
        return [step for step in self._plan.steps if step.name != step_name]


def _resolve_builder(graph: GraphLike) -> StateGraph:
    if isinstance(graph, StateGraph):
        return graph
    builder = getattr(graph, "builder", None)
    if isinstance(builder, StateGraph):
        return builder
    raise TypeError(
        "graph must be a langgraph StateGraph builder or a compiled StateGraph; "
        f"got {type(graph)!r}"
    )


def _node_function(node_spec: Any) -> Callable[..., Any]:
    runnable = getattr(node_spec, "runnable", node_spec)
    func = getattr(runnable, "func", None)
    if callable(func):
        return func
    if callable(runnable):
        return runnable
    raise TypeError(f"could not resolve node function from {node_spec!r}")


def _first_sentence(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped:
        return ""
    for sep in (".", "\n"):
        if sep in stripped:
            return stripped.split(sep, 1)[0].strip()
    return stripped


def _step_description(
    node_name: str,
    node_fn: Callable[..., Any],
    step_descriptions: Optional[Dict[str, str]],
) -> str:
    if step_descriptions and node_name in step_descriptions:
        return step_descriptions[node_name]
    doc = getattr(node_fn, "__doc__", None)
    if doc:
        first = _first_sentence(doc)
        if first:
            return first
    return node_name


def _collect_edges(builder: StateGraph) -> Set[Tuple[str, str]]:
    edges: Set[Tuple[str, str]] = set(getattr(builder, "edges", set()) or set())
    branches = getattr(builder, "branches", {}) or {}
    for source, branch_map in branches.items():
        for branch_spec in branch_map.values():
            ends = getattr(branch_spec, "ends", None) or {}
            for target in ends.values():
                edges.add((source, target))
    return edges


def _real_nodes(builder: StateGraph) -> List[str]:
    return [name for name in builder.nodes.keys() if name not in VIRTUAL_NODES]


def _topological_order(nodes: List[str], edges: Set[Tuple[str, str]]) -> Tuple[List[str], bool]:
    node_set = set(nodes)
    indegree = {name: 0 for name in nodes}
    adjacency: Dict[str, List[str]] = {name: [] for name in nodes}

    for source, target in edges:
        if target not in node_set or target in VIRTUAL_NODES:
            continue
        if source in node_set:
            adjacency[source].append(target)
            indegree[target] += 1

    ready = sorted(name for name, degree in indegree.items() if degree == 0)
    ordered: List[str] = []
    while ready:
        current = ready.pop(0)
        ordered.append(current)
        for neighbor in sorted(adjacency.get(current, [])):
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                ready.append(neighbor)
        ready.sort()

    cyclic = len(ordered) != len(nodes)
    if cyclic:
        seen = set(ordered)
        ordered.extend(name for name in nodes if name not in seen)
    return ordered, cyclic


def plan_from_graph(
    graph: GraphLike,
    step_descriptions: Optional[Dict[str, str]] = None,
) -> Plan:
    """Derive a memahead :class:`Plan` from a LangGraph ``StateGraph``.

    Args:
        graph: A ``StateGraph`` builder or compiled graph.
        step_descriptions: Optional explicit mapping of node name to description.

    Returns:
        A :class:`Plan` whose step order follows graph execution order when
        possible. Cyclic graphs set ``plan._cyclic = True`` and use best-effort
        node order.
    """

    builder = _resolve_builder(graph)
    nodes = _real_nodes(builder)
    if not nodes:
        raise ValueError("graph has no compressible nodes")

    edges = _collect_edges(builder)
    order, cyclic = _topological_order(nodes, edges)

    steps: List[Step] = []
    for name in order:
        node_fn = _node_function(builder.nodes[name])
        steps.append(
            Step(
                name=name,
                description=_step_description(name, node_fn, step_descriptions),
            )
        )

    plan = Plan(steps)
    if cyclic:
        warnings.warn(
            "Cyclic LangGraph detected: plan-aware retention cannot rely on a "
            "linear step order. memahead will treat all non-current nodes as "
            "remaining steps for compression.",
            UserWarning,
            stacklevel=2,
        )
        plan._cyclic = True  # type: ignore[attr-defined]
    else:
        plan._cyclic = False  # type: ignore[attr-defined]
    return plan


def _plan_cache_key(step_descriptions: Optional[Dict[str, str]]) -> Tuple[Tuple[str, str], ...]:
    if not step_descriptions:
        return ()
    return tuple(sorted(step_descriptions.items()))


def _get_cached_plan(
    graph: GraphLike,
    step_descriptions: Optional[Dict[str, str]],
) -> Plan:
    builder = _resolve_builder(graph)
    cache: Dict[Tuple[Tuple[str, str], ...], Plan] = getattr(
        builder, _PLAN_CACHE_ATTR, {}
    )
    if not cache:
        cache = {}
        setattr(builder, _PLAN_CACHE_ATTR, cache)
    key = _plan_cache_key(step_descriptions)
    if key not in cache:
        cache[key] = plan_from_graph(builder, step_descriptions)
    return cache[key]


def _plan_for_compression(plan: Plan, current_step: str) -> Union[Plan, _CyclicPlanAdapter]:
    if getattr(plan, "_cyclic", False):
        return _CyclicPlanAdapter(plan)
    return plan


def _attach_report(result: Dict[str, Any], report: TokenReport) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return result
    updated = dict(result)
    updated["_memahead_report"] = report
    return updated


def compress_node(
    node_fn: Callable[..., Any],
    graph: GraphLike,
    *,
    compressor: Optional[PlanAwareCompressor] = None,
    history_key: str = "messages",
    tools_key: str = "tools",
    step_descriptions: Optional[Dict[str, str]] = None,
    node_name: Optional[str] = None,
) -> Callable[..., Any]:
    """Wrap a LangGraph node to compress its input state before execution."""

    actual_name = node_name or getattr(node_fn, "__name__", "node")
    comp = compressor or PlanAwareCompressor(quality=0.85)

    def wrapped(state: Dict[str, Any]) -> Any:
        plan = _get_cached_plan(graph, step_descriptions)
        plan_for_compress = _plan_for_compression(plan, actual_name)

        history = list(state.get(history_key, []) or [])
        tools = list(state.get(tools_key, []) or [])

        compressed: CompressedContext = comp.compress(
            history=history,
            tools=tools,
            plan=plan_for_compress,
            current_step=actual_name,
        )

        compressed_state = dict(state)
        compressed_state[history_key] = compressed.messages
        if tools_key in state or compressed.tools:
            compressed_state[tools_key] = compressed.tools

        result = node_fn(compressed_state)
        if result is None:
            return {"_memahead_report": compressed.report}
        if isinstance(result, dict):
            return _attach_report(result, compressed.report)
        return result

    wrapped.__name__ = getattr(node_fn, "__name__", actual_name)
    wrapped.__doc__ = getattr(node_fn, "__doc__", None)
    wrapped.__memahead_node_name__ = actual_name  # type: ignore[attr-defined]
    return wrapped


def compress_graph(
    builder: StateGraph,
    *,
    compressor: Optional[PlanAwareCompressor] = None,
    history_key: str = "messages",
    tools_key: str = "tools",
    step_descriptions: Optional[Dict[str, str]] = None,
    exclude: Optional[Set[str]] = None,
) -> StateGraph:
    """Wrap every node in a ``StateGraph`` builder with :func:`compress_node`."""

    excluded = set(exclude or ())
    _get_cached_plan(builder, step_descriptions)

    for name, spec in list(builder.nodes.items()):
        if name in VIRTUAL_NODES or name in excluded:
            continue
        node_fn = _node_function(spec)
        wrapped = compress_node(
            node_fn,
            builder,
            compressor=compressor,
            history_key=history_key,
            tools_key=tools_key,
            step_descriptions=step_descriptions,
            node_name=name,
        )
        builder.nodes[name] = dataclasses.replace(
            spec,
            runnable=RunnableCallable(wrapped),
        )

    return builder
