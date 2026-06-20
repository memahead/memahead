"""
Run: python -m benchmarks.run_benchmark

Outputs: benchmarks/results/README.md
"""

from __future__ import annotations

import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from memahead import Plan, PlanAwareCompressor, Step, __version__

from benchmarks.baselines import headroom_only, no_compression
from benchmarks.quality.scorer import QualityScorer
from benchmarks.tokens import count_context_tokens, messages_to_text
from benchmarks.workflows import code_review, data_analysis, research_synthesis

WORKFLOWS: List[Tuple[str, Any, str]] = [
    ("Research & Synthesis", research_synthesis, "synthesize"),
    ("Code Review", code_review, "summarize"),
    ("Data Analysis", data_analysis, "interpret"),
]

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "README.md"


def pct_reduction(before: int, after: int) -> str:
    """Format percent change; negative means fewer tokens (reduction)."""

    if before <= 0:
        return "0%"
    value = round((after - before) / before * 100)
    return f"{value}%"


def display_pct_change(value: str) -> str:
    """Use Unicode minus for negative reductions in markdown tables."""

    return value.replace("-", "−") if value.startswith("-") else value


def avg(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def avg_int(values: Sequence[int]) -> int:
    return int(round(avg(values))) if values else 0


def _warmup_compressor(compressor: PlanAwareCompressor) -> None:
    """Load embedding model and Headroom before timed runs."""

    warmup_plan = Plan([Step("warmup", "warmup step for model loading")])
    wf = {
        "history": [{"role": "user", "content": "warmup"}],
        "tools": [],
        "plan": warmup_plan,
        "current_step": "warmup",
    }
    compressor.compress(**wf)
    headroom_only.compress(**wf)


def _latency_platform() -> str:
    if platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64"):
        return "Apple M-series"
    return platform.processor() or platform.machine()


def run() -> List[Dict[str, Any]]:
    compressor = PlanAwareCompressor(quality=0.85, tool_threshold=0.55)
    _warmup_compressor(compressor)
    scorer = QualityScorer()
    results: List[Dict[str, Any]] = []

    for name, workflow_module, current_step in WORKFLOWS:
        wf = workflow_module.get_workflow(current_step)
        history = wf["history"]
        tools = wf["tools"]
        plan = wf["plan"]

        base = no_compression.compress(history=history, tools=tools)
        hr = headroom_only.compress(history=history, tools=tools)

        t0 = time.perf_counter()
        ma = compressor.compress(
            history=history,
            tools=tools,
            plan=plan,
            current_step=current_step,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        original_text = messages_to_text(history)
        quality_ma = scorer.score(
            original=original_text,
            compressed=messages_to_text(ma.messages),
            critical_facts=workflow_module.CRITICAL_FACTS,
        )
        quality_hr = scorer.score(
            original=original_text,
            compressed=messages_to_text(hr["messages"]),
            critical_facts=workflow_module.CRITICAL_FACTS,
        )

        tokens_baseline = base["tokens_used"]
        tokens_headroom = hr["tokens_used"]
        tokens_memahead = count_context_tokens(ma.messages, ma.tools)

        results.append(
            {
                "workflow": name,
                "tokens_baseline": tokens_baseline,
                "tokens_headroom": tokens_headroom,
                "tokens_memahead": tokens_memahead,
                "reduction_vs_baseline": pct_reduction(tokens_baseline, tokens_memahead),
                "reduction_vs_headroom": pct_reduction(tokens_headroom, tokens_memahead),
                "quality_headroom": quality_hr,
                "quality_memahead": quality_ma,
                "latency_ms": round(latency_ms),
            }
        )

    return results


def format_int(n: int) -> str:
    return f"{n:,}"


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _reduction_magnitude(value: str) -> int:
    """Absolute percent reduction from a value like '-87%'."""

    return abs(int(value.rstrip("%")))


def _interpretation_workflows(
    results: List[Dict[str, Any]],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Return (largest vs Headroom, smallest vs Headroom) workflow rows."""

    ordered = sorted(results, key=lambda r: _reduction_magnitude(r["reduction_vs_headroom"]))
    return ordered[-1], ordered[0]


def write_results_table(results: List[Dict[str, Any]]) -> None:
    """Write benchmarks/results/README.md with a clean markdown table."""

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    avg_baseline = avg_int([r["tokens_baseline"] for r in results])
    avg_headroom = avg_int([r["tokens_headroom"] for r in results])
    avg_memahead = avg_int([r["tokens_memahead"] for r in results])
    avg_red_base = pct_reduction(avg_baseline, avg_memahead)
    avg_red_hr = pct_reduction(avg_headroom, avg_memahead)
    avg_q_hr = avg([r["quality_headroom"] for r in results])
    avg_q_ma = avg([r["quality_memahead"] for r in results])

    token_rows = []
    for r in results:
        token_rows.append(
            f"| {r['workflow']} | {format_int(r['tokens_baseline'])} | "
            f"{format_int(r['tokens_headroom'])} | {format_int(r['tokens_memahead'])} | "
            f"{display_pct_change(r['reduction_vs_baseline'])} | "
            f"{display_pct_change(r['reduction_vs_headroom'])} |"
        )
    token_rows.append(
        f"| **Average** | **{format_int(avg_baseline)}** | **{format_int(avg_headroom)}** | "
        f"**{format_int(avg_memahead)}** | **{display_pct_change(avg_red_base)}** | "
        f"**{display_pct_change(avg_red_hr)}** |"
    )

    quality_rows = []
    for r in results:
        quality_rows.append(
            f"| {r['workflow']} | {format_pct(r['quality_headroom'])} | "
            f"{format_pct(r['quality_memahead'])} |"
        )
    quality_rows.append(
        f"| **Average** | **{format_pct(avg_q_hr)}** | **{format_pct(avg_q_ma)}** |"
    )

    latency_rows = [
        f"| {r['workflow']} | {r['latency_ms']}ms |" for r in results
    ]

    max_vs_baseline = max(_reduction_magnitude(r["reduction_vs_baseline"]) for r in results)
    max_vs_headroom = max(_reduction_magnitude(r["reduction_vs_headroom"]) for r in results)
    min_quality = min(r["quality_memahead"] for r in results)
    quality_callout = (
        "100% critical fact retention"
        if min_quality >= 1.0
        else f"{format_pct(min_quality)} critical fact retention"
    )

    largest_gain, smallest_gain = _interpretation_workflows(results)

    content = f"""# memahead Benchmark Results

> Agent memory, optimized for what's ahead.  
> Comparing plan-aware compression vs baselines across realistic agent workflows.

> **memahead reduces token consumption by up to {max_vs_baseline}% vs no compression  
> and up to {max_vs_headroom}% vs Headroom alone — with {quality_callout}.**

**memahead v{__version__}** · Python {py_version} · Run: {now}

## Token Reduction

| Workflow | No Compression | Headroom Only | memahead | vs Baseline | vs Headroom |
|----------|---------------|---------------|----------|-------------|-------------|
{chr(10).join(token_rows)}

## Quality Retention

| Workflow | Headroom Only | memahead |
|----------|---------------|----------|
{chr(10).join(quality_rows)}

## When plan-awareness helps most

Plan-aware compression delivers the largest gains on workflows where
early steps produce verbose output that later steps don't need
({largest_gain['workflow']}: {display_pct_change(largest_gain['reduction_vs_headroom'])} vs Headroom). The gains are smaller on
workflows where most context remains relevant across all steps
({smallest_gain['workflow']}: {display_pct_change(smallest_gain['reduction_vs_headroom'])} vs Headroom).

## Latency Overhead

| Workflow | memahead overhead |
|----------|------------------|
{chr(10).join(latency_rows)}

> Latency measured on {_latency_platform()}. Overhead includes retention
> scoring and tool filtering. No LLM calls in the compression path.

## Methodology

- Token counts use `tiktoken` (cl100k_base encoding)
- Quality measured by retention of critical facts defined per workflow
- Headroom baseline uses default compression settings, no plan object
- memahead uses `quality=0.85`, `all-MiniLM-L6-v2` embeddings
- All workflows use realistic simulated agent histories (15-25 turns)

## Reproduce

```bash
pip install memahead headroom-ai
python -m benchmarks.run_benchmark
```
"""

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(content, encoding="utf-8")


def print_results_table(results: List[Dict[str, Any]]) -> None:
    print("\nmemahead benchmark results\n" + "=" * 40)
    for r in results:
        print(
            f"{r['workflow']}: "
            f"baseline={r['tokens_baseline']:,} "
            f"headroom={r['tokens_headroom']:,} "
            f"memahead={r['tokens_memahead']:,} "
            f"quality={r['quality_memahead']:.1%} "
            f"latency={r['latency_ms']}ms"
        )
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    benchmark_results = run()
    write_results_table(benchmark_results)
    print_results_table(benchmark_results)
