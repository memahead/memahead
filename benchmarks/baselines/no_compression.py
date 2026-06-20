"""Naive baseline: return context completely unchanged."""

from __future__ import annotations

from typing import Any, List, Sequence

from benchmarks.tokens import count_context_tokens


def compress(
    history: Sequence[Any],
    tools: Sequence[Any],
    **kwargs: Any,
) -> dict:
    """Return context completely unchanged — the naive baseline."""

    messages = list(history)
    tool_list = list(tools or [])
    return {
        "messages": messages,
        "tools": tool_list,
        "tokens_used": count_context_tokens(messages, tool_list),
    }
