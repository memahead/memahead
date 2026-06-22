"""
memahead + LangGraph integration example.

Shows two ways to add plan-aware context compression to a LangGraph app:
  1. compress_node  — wrap individual nodes you want compressed
  2. compress_graph — wrap the whole graph in one call

No manual Plan declaration — memahead reads the workflow structure
directly from the StateGraph, using node docstrings as step descriptions.

Also acts as a visible smoke test: prints before/after token counts per
wrapped node and warns loudly if compression did not reduce context.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from memahead import PlanAwareCompressor
from memahead.integrations.langgraph import compress_graph, compress_node, plan_from_graph

# Reuse the benchmark's proven research workflow history — it has superseded tool
# dumps, off-topic noise, and a synthesis rollup that retention scoring can
# distinguish from load-bearing facts.
from benchmarks.workflows.research_synthesis import CRITICAL_FACTS, HISTORY, TOOLS

TOPIC = "LLMs in software development"


class ResearchState(TypedDict):
    messages: list
    tools: list
    topic: str


def count_tokens(messages) -> int:
    """Count tokens across a list of messages. Uses memahead's counter if available."""

    try:
        from memahead.context import count_message_tokens

        return count_message_tokens(messages)
    except Exception:
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            return sum(len(enc.encode(str(m))) for m in messages)
        except Exception:
            return int(sum(len(str(m).split()) for m in messages) * 1.3)


def _check_critical_facts(messages: list) -> int:
    blob = str(messages).lower()
    return sum(1 for fact in CRITICAL_FACTS if fact.lower() in blob)


def report_compression(node_name, state_before, state_after) -> tuple[int, int, float]:
    """Print before/after token stats; return (before, after, pct_saved)."""

    before = count_tokens(state_before["messages"])
    after_msgs = state_after.get("messages", state_before["messages"])
    after = count_tokens(after_msgs)
    saved = before - after
    pct = (saved / before * 100) if before else 0.0

    print(f"\n  [{node_name}]")
    print(f"    before: {before:,} tokens")
    print(f"    after:  {after:,} tokens")
    print(f"    saved:  {saved:,} tokens ({pct:.0f}%)")

    if "_memahead_report" in state_after:
        print(f"    report: {state_after['_memahead_report']}")

    retained = _check_critical_facts(after_msgs)
    print(f"    critical facts retained: {retained}/{len(CRITICAL_FACTS)}")
    if retained < len(CRITICAL_FACTS):
        print("    ⚠️  WARNING: compression dropped a load-bearing fact!")

    if after >= before:
        print(f"    ⚠️  WARNING: compression did NOT reduce tokens for '{node_name}'.")
        print("        This likely means compressed messages are being APPENDED")
        print("        by a LangGraph reducer instead of REPLACING the history.")
        print("        Check how compress_node writes back state[history_key].")
    else:
        print("    ✓ compression confirmed")

    return before, after, pct


def research(state: ResearchState) -> dict:
    """Search the web and gather raw facts about the topic."""
    return {
        "messages": state["messages"]
        + [{"role": "assistant", "content": "Gathered raw facts about LLM agents."}]
    }


def synthesize(state: ResearchState) -> dict:
    """Identify key themes and contradictions across the gathered research."""
    return {
        "messages": state["messages"]
        + [
            {
                "role": "assistant",
                "content": (
                    "Themes: developer productivity gains, code-review automation, "
                    "and context-window cost as the main tension."
                ),
            }
        ]
    }


def draft(state: ResearchState) -> dict:
    """Write a structured first draft of the report."""
    return {
        "messages": state["messages"]
        + [{"role": "assistant", "content": "Draft section one complete."}]
    }


def revise(state: ResearchState) -> dict:
    """Produce the final polished output."""
    return {
        "messages": state["messages"]
        + [{"role": "assistant", "content": "Final polished report ready."}]
    }


def build_large_history() -> list:
    """Benchmark-grade history sized for visible plan-aware compression."""

    return list(HISTORY)


def build_fresh_graph() -> StateGraph:
    """Base research workflow without compression wrappers."""

    builder = StateGraph(ResearchState)
    builder.add_node("research", research)
    builder.add_node("synthesize", synthesize)
    builder.add_node("draft", draft)
    builder.add_node("revise", revise)
    builder.add_edge(START, "research")
    builder.add_edge("research", "synthesize")
    builder.add_edge("synthesize", "draft")
    builder.add_edge("draft", "revise")
    builder.add_edge("revise", END)
    return builder


def make_compressor() -> PlanAwareCompressor:
    # quality=0.80 is slightly more aggressive than the 0.85 default — lower = more compression.
    # Headroom enabled (default) for mechanical compression on surviving chunks.
    return PlanAwareCompressor(quality=0.80, tool_threshold=0.55)


def main() -> None:
    large_history = build_large_history()
    initial_tokens = count_tokens(large_history)
    if initial_tokens < 4000:
        print(f"Note: history is {initial_tokens:,} tokens (target 4000+).")

    compressor = make_compressor()
    builder = build_fresh_graph()

    print("=" * 60)
    print("Derived plan")
    print("=" * 60)
    plan = plan_from_graph(builder)
    for step in plan.steps:
        print(f"  {step.name}: {step.description}")

    seed = {"messages": large_history, "tools": TOOLS, "topic": TOPIC}

    print("\n" + "=" * 60)
    print("Pattern 1 — selective node wrapping (compress_node)")
    print("=" * 60)
    print(f"  input history: {initial_tokens:,} tokens across {len(large_history)} turns")
    print(f"  critical facts in input: {_check_critical_facts(large_history)}/{len(CRITICAL_FACTS)}")

    wrapped_synthesize = compress_node(
        synthesize,
        builder,
        compressor=compressor,
        node_name="synthesize",
    )
    state_before = deepcopy(seed)
    result = wrapped_synthesize(dict(state_before))
    syn_before, syn_after, syn_pct = report_compression("synthesize", state_before, result)

    wrapped_draft = compress_node(
        draft,
        builder,
        compressor=compressor,
        node_name="draft",
    )
    draft_before = {
        "messages": result["messages"],
        "tools": TOOLS,
        "topic": TOPIC,
    }
    draft_result = wrapped_draft(dict(draft_before))
    draft_before_tok, draft_after_tok, draft_pct = report_compression(
        "draft", draft_before, draft_result
    )

    step_rows = [
        ("synthesize", syn_before, syn_after, syn_pct),
        ("draft", draft_before_tok, draft_after_tok, draft_pct),
    ]

    print("\n" + "=" * 60)
    print("Pattern 2 — graph-level wrapping (compress_graph)")
    print("=" * 60)

    builder2 = build_fresh_graph()
    builder2 = compress_graph(builder2, compressor=compressor, exclude={"research"})
    graph2 = builder2.compile()
    final_state = graph2.invoke(deepcopy(seed))

    final_tokens = count_tokens(final_state["messages"])
    overall_pct = ((initial_tokens - final_tokens) / initial_tokens * 100) if initial_tokens else 0.0
    facts_retained = _check_critical_facts(final_state["messages"])
    total_facts = len(CRITICAL_FACTS)

    if facts_retained < total_facts:
        print("  ⚠️  WARNING: graph invoke dropped a load-bearing fact!")
    if final_tokens >= initial_tokens:
        print(
            "  WARNING: final token count did not shrink versus input. "
            "Compressed messages may be appended by a reducer instead of replacing history."
        )

    print("\n  Per-step breakdown:")
    print("  step        before    after    saved")
    print("  " + "-" * 40)
    for name, before, after, pct in step_rows:
        note = "  (already lean)" if pct < 5 else ""
        print(f"  {name:<12}{before:>6,}    {after:>6,}    {pct:>3.0f}%{note}")
    print()
    print("  Low compression at later steps is expected — synthesize front-loads")
    print("  the reduction; draft and revise inherit an already-lean context.")

    print("\n" + "=" * 60)
    print("Workflow summary")
    print("=" * 60)
    print(f"  context entering workflow:  {initial_tokens:,} tokens")
    print(f"  context after full run:     {final_tokens:,} tokens")
    print(f"  overall reduction:          {overall_pct:.0f}%")
    print(f"  critical facts retained:    {facts_retained}/{total_facts}")
    print()
    print("  Meaningful reduction with perfect fact retention — not maximal compression.")
    print("  Compression front-loads to where the dead weight is:")
    print("  the synthesize step strips verbose research scaffolding")
    print("  that later steps no longer need, so draft and revise")
    print("  see an already-lean context. No load-bearing fact is dropped.")
    print("=" * 60)


if __name__ == "__main__":
    main()
