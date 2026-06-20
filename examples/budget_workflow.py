"""Budget-constrained compression example.

Shows how to use ``budget_tokens`` to enforce a hard token ceiling —
useful when you know your LLM provider's context limit or want to
control costs precisely.

Inspired by ContextBudget (arXiv:2604.01664): budget-free compression
can over-compress (losing evidence) or under-compress (overflowing limits).
``budget_tokens`` targets a specific ceiling.
"""

from __future__ import annotations

from benchmarks.workflows.research_synthesis import get_workflow
from memahead import BudgetExceededError, PlanAwareCompressor


def main() -> None:
    wf = get_workflow("synthesize")

    compressor = PlanAwareCompressor(quality=0.85, use_headroom=False)
    result = compressor.compress(
        history=wf["history"],
        tools=wf["tools"],
        plan=wf["plan"],
        current_step=wf["current_step"],
    )
    print("Quality-only:", result.report)

    compressor_budgeted = PlanAwareCompressor(
        quality=0.5,
        budget_tokens=2000,
        use_headroom=False,
    )
    result_budgeted = compressor_budgeted.compress(
        history=wf["history"],
        tools=wf["tools"],
        plan=wf["plan"],
        current_step=wf["current_step"],
    )
    print("Budget-constrained:", result_budgeted.report)

    compressor_tight = PlanAwareCompressor(quality=0.99, budget_tokens=10)
    try:
        compressor_tight.compress(
            history=wf["history"],
            tools=wf["tools"],
            plan=wf["plan"],
            current_step=wf["current_step"],
        )
    except BudgetExceededError as exc:
        print(f"Budget too tight: {exc}")
        print(f"Minimum achievable: {exc.minimum_achievable} tokens")


if __name__ == "__main__":
    main()
