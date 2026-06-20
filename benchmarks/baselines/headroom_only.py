"""Headroom compression without plan awareness."""

from __future__ import annotations

from typing import Any, List, Sequence

from benchmarks.tokens import count_context_tokens


def _normalize_headroom_result(result: Any, fallback: List[Any]) -> List[Any]:
    if result is None:
        return fallback
    if isinstance(result, list):
        return result
    for attr in ("messages", "compressed", "output", "result"):
        value = getattr(result, attr, None)
        if isinstance(value, list):
            return value
    if isinstance(result, dict):
        for key in ("messages", "compressed", "output", "result"):
            value = result.get(key)
            if isinstance(value, list):
                return value
    return fallback


def compress(
    history: Sequence[Any],
    tools: Sequence[Any],
    **kwargs: Any,
) -> dict:
    """Headroom compression with no plan object and no retention scoring."""

    messages = list(history)
    tool_list = list(tools or [])
    compressed = messages

    try:
        from headroom import compress as hr_compress

        try:
            result = hr_compress(messages)
        except TypeError:
            result = hr_compress(messages, model=kwargs.get("model"))
        compressed = _normalize_headroom_result(result, messages)
    except Exception:
        compressed = messages

    return {
        "messages": compressed,
        "tools": tool_list,
        "tokens_used": count_context_tokens(compressed, tool_list),
    }
