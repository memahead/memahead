"""Token counting utilities for benchmarks."""

from __future__ import annotations

import json
from typing import Any, List, Optional, Sequence


def _get_encoding():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


_ENCODING = None


def count_text_tokens(text: str) -> int:
    """Count tokens using tiktoken cl100k_base, or heuristic fallback."""

    global _ENCODING
    if not text:
        return 0
    if _ENCODING is None:
        _ENCODING = _get_encoding()
    if _ENCODING is not None:
        return len(_ENCODING.encode(text))
    return int(len(text.split()) * 1.3)


def message_text(message: Any) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    parts.append(str(part.get("text", "")))
            return "\n".join(parts)
        return str(content)
    return str(message)


def tool_text(tool: Any) -> str:
    if isinstance(tool, str):
        return tool
    if isinstance(tool, dict):
        return json.dumps(tool, sort_keys=True, default=str)
    return str(tool)


def count_messages_tokens(messages: Sequence[Any]) -> int:
    return sum(count_text_tokens(message_text(m)) for m in messages)


def count_tools_tokens(tools: Sequence[Any]) -> int:
    return sum(count_text_tokens(tool_text(t)) for t in (tools or []))


def count_context_tokens(
    messages: Sequence[Any],
    tools: Optional[Sequence[Any]] = None,
) -> int:
    """Total tokens for messages plus tool schemas."""

    return count_messages_tokens(messages) + count_tools_tokens(tools or [])


def messages_to_text(messages: Sequence[Any]) -> str:
    """Flatten messages into one string for quality scoring."""

    return "\n".join(message_text(m) for m in messages)
