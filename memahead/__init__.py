"""memahead — agent memory optimized for what's ahead.

memahead compresses an LLM agent's context at each step of a multi-step
workflow using *forward-looking* plan awareness. Instead of compressing
greedily based on what already happened, memahead scores each chunk of context
against the *remaining* steps of the plan and drops what future steps won't
need — far fewer tokens per call without losing what matters downstream.

It builds on Headroom (``pip install headroom-ai``) for the underlying
compression mechanics and adds the plan-aware retention scoring layer on top.

Academic foundations:
    - PAACE: Yuksel et al., arXiv:2512.16970 (Dec 2025)
    - ACON:  Kang et al., Microsoft, arXiv:2510.00615 (2025)

Quick start::

    from memahead import Plan, Step, PlanAwareCompressor

    plan = Plan([
        Step("research", "Search and gather raw facts about the topic"),
        Step("synthesize", "Identify key themes across the research"),
        Step("draft", "Write a structured first draft"),
        Step("revise", "Produce the final polished output"),
    ])

    compressor = PlanAwareCompressor(quality=0.85)
    compressed = compressor.compress(
        history=prior_messages,
        tools=all_tool_schemas,
        plan=plan,
        current_step="synthesize",
    )
    print(compressed.report)
"""

from __future__ import annotations

from .compressor import BudgetExceededError, PlanAwareCompressor
from .context import (
    CompressedContext,
    DroppedChunk,
    TokenReport,
    count_tokens,
)
from .plan import Plan, PlanGraph, Step
from .scorer import ChunkScore, RetentionScorer
from .tool_filter import ToolFilter, ToolMatch, filter_tools

__version__ = "0.3.0"

__all__ = [
    "__version__",
    # plan
    "Step",
    "Plan",
    "PlanGraph",
    # scoring (core novelty)
    "RetentionScorer",
    "ChunkScore",
    # tool filtering
    "ToolFilter",
    "ToolMatch",
    "filter_tools",
    # compression pipeline
    "PlanAwareCompressor",
    "BudgetExceededError",
    # result containers
    "CompressedContext",
    "TokenReport",
    "DroppedChunk",
    "count_tokens",
]
