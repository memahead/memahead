"""Result containers for compression: :class:`CompressedContext` and
:class:`TokenReport`, plus a small token-estimation helper.

These are intentionally dependency-light dataclasses so they can be passed
around, serialized, and inspected without importing heavy ML packages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["count_tokens", "DroppedChunk", "TokenReport", "CompressedContext"]


def count_tokens(text: str, model: Optional[str] = None) -> int:
    """Estimate the number of tokens in ``text``.

    Uses :mod:`tiktoken` when available for accuracy; otherwise falls back to
    a fast heuristic (~4 characters per token). The heuristic keeps the
    library usable with zero extra dependencies while still giving a stable,
    monotonic measure for reporting savings.

    Args:
        text: The text to measure.
        model: Optional model name used to pick a tiktoken encoding.
    """

    if not text:
        return 0
    try:
        import tiktoken

        try:
            encoding = (
                tiktoken.encoding_for_model(model)
                if model
                else tiktoken.get_encoding("cl100k_base")
            )
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # Heuristic fallback: ~4 chars/token, with a floor of one token per word.
        char_estimate = (len(text) + 3) // 4
        word_estimate = len(text.split())
        return max(char_estimate, word_estimate)


def _message_text(message: Any) -> str:
    """Extract the textual content from a chat message (dict or str)."""

    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        # Content can be a list of parts (OpenAI-style multimodal blocks).
        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    parts.append(str(part.get("text", "")))
            return "\n".join(parts)
        return str(content)
    return str(message)


def _tool_text(tool: Any) -> str:
    """Extract a stable textual representation of a tool schema for counting."""

    if isinstance(tool, str):
        return tool
    if isinstance(tool, dict):
        # Support both the bare schema and the OpenAI {"type","function":{...}}
        # envelope so token counts reflect what is actually sent.
        import json

        return json.dumps(tool, sort_keys=True, default=str)
    return str(tool)


def count_message_tokens(messages: List[Any], model: Optional[str] = None) -> int:
    """Total estimated tokens across a list of chat messages."""

    return sum(count_tokens(_message_text(m), model) for m in messages)


def count_tool_tokens(tools: List[Any], model: Optional[str] = None) -> int:
    """Total estimated tokens across a list of tool schemas."""

    return sum(count_tokens(_tool_text(t), model) for t in (tools or []))


@dataclass
class DroppedChunk:
    """Record of a single context chunk that was dropped or shrunk.

    Attributes:
        source: A human-readable origin (e.g. ``"message[3]"`` or a tool name).
        kind: ``"message"`` or ``"tool"``.
        score: The forward-looking retention score (0.0–1.0), if applicable.
        tokens_before: Tokens the chunk occupied before compression.
        tokens_after: Tokens remaining after compression (0 if fully dropped).
        reason: Why it was dropped (e.g. ``"below retention threshold"``).
    """

    source: str
    kind: str
    tokens_before: int
    tokens_after: int = 0
    score: Optional[float] = None
    reason: str = ""

    @property
    def tokens_saved(self) -> int:
        return max(self.tokens_before - self.tokens_after, 0)


@dataclass
class TokenReport:
    """Summary of how many tokens compression saved.

    Attributes:
        before: Total tokens before compression (messages + tools).
        after: Total tokens after compression (messages + tools).
        dropped: Per-chunk records of what was removed or shrunk.
    """

    before: int
    after: int
    dropped: List[DroppedChunk] = field(default_factory=list)

    @property
    def saved(self) -> int:
        """Absolute number of tokens saved."""

        return max(self.before - self.after, 0)

    @property
    def compression_ratio(self) -> float:
        """Fraction of tokens removed, in ``[0.0, 1.0]``.

        ``0.0`` means nothing was saved; ``0.75`` means 75% fewer tokens.
        """

        if self.before <= 0:
            return 0.0
        return round(self.saved / self.before, 4)

    def dropped_sources(self) -> List[str]:
        """Return the sources of fully-dropped chunks."""

        return [d.source for d in self.dropped if d.tokens_after == 0]

    def __repr__(self) -> str:
        return (
            f"TokenReport(before={self.before}, after={self.after}, "
            f"saved={self.saved}, compression_ratio={self.compression_ratio})"
        )


@dataclass
class CompressedContext:
    """The lean, ready-to-send context produced by the compressor.

    Attributes:
        messages: Compressed chat messages, ready to pass to an LLM call.
        tools: Filtered tool schemas relevant to the current step.
        report: A :class:`TokenReport` describing what was saved.
        retained_scores: Mapping of retained message source -> retention score,
            useful for debugging and evaluation.
    """

    messages: List[Any]
    tools: List[Any]
    report: TokenReport
    retained_scores: Dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"CompressedContext(messages={len(self.messages)}, "
            f"tools={len(self.tools)}, report={self.report!r})"
        )
