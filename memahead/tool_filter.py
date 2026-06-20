"""Deterministic, LLM-free tool-schema filtering.

Agents often carry a large catalog of tool schemas, most of which are
irrelevant to the step at hand. This module keeps only the schemas that
semantically match the current step description. It is fully deterministic
(same inputs -> same outputs) and requires no LLM call: matching is done with
embedding cosine similarity, with a transparent lexical fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union

import numpy as np

from ._embeddings import Embedder, cosine_similarity_matrix, resolve_embedder
from .plan import Step

__all__ = ["ToolMatch", "filter_tools", "ToolFilter"]


def _tool_name(tool: Any) -> str:
    """Extract a tool name from common schema shapes."""

    if isinstance(tool, dict):
        if "function" in tool and isinstance(tool["function"], dict):
            return str(tool["function"].get("name", ""))
        return str(tool.get("name", ""))
    return getattr(tool, "name", "") or ""


def _tool_description(tool: Any) -> str:
    """Extract a tool description from common schema shapes."""

    if isinstance(tool, dict):
        if "function" in tool and isinstance(tool["function"], dict):
            return str(tool["function"].get("description", ""))
        return str(tool.get("description", ""))
    return getattr(tool, "description", "") or ""


def _tool_text(tool: Any) -> str:
    name = _tool_name(tool)
    description = _tool_description(tool)
    return f"{name}: {description}".strip(": ").strip()


@dataclass
class ToolMatch:
    """How well a single tool matched the current step.

    Attributes:
        tool: The original tool schema object.
        name: The tool's name.
        score: Match score in ``[0.0, 1.0]``.
        kept: Whether the tool passed the threshold and was retained.
    """

    tool: Any
    name: str
    score: float
    kept: bool


def _lexical_score(step_text: str, tool_text: str) -> float:
    """Token-overlap (Jaccard) similarity used as a no-embedding fallback."""

    step_tokens = {t for t in _tokenize(step_text)}
    tool_tokens = {t for t in _tokenize(tool_text)}
    if not step_tokens or not tool_tokens:
        return 0.0
    intersection = step_tokens & tool_tokens
    union = step_tokens | tool_tokens
    return len(intersection) / len(union)


def _tokenize(text: str) -> List[str]:
    return [t for t in "".join(
        c.lower() if c.isalnum() else " " for c in text
    ).split() if len(t) > 1]


class ToolFilter:
    """Reusable, configurable tool filter.

    Args:
        embedder: Optional custom embedder (see :class:`memahead.RetentionScorer`).
            If ``None``, the default sentence-transformers model is used lazily.
            Pass ``use_embeddings=False`` to skip embeddings entirely and rely
            on the lexical fallback (handy for tests and offline use).
        threshold: Minimum match score in ``[0.0, 1.0]`` for a tool to be kept.
        min_tools: Always keep at least this many top-scoring tools, even if
            none clear the threshold (avoids stripping every tool by accident).
        use_embeddings: Whether to use semantic embeddings (default ``True``).
    """

    def __init__(
        self,
        embedder: Optional[Union[Embedder, object]] = None,
        *,
        threshold: float = 0.3,
        min_tools: int = 0,
        use_embeddings: bool = True,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0.0, 1.0]")
        if min_tools < 0:
            raise ValueError("min_tools must be >= 0")
        self.threshold = threshold
        self.min_tools = min_tools
        self.use_embeddings = use_embeddings
        self._model_name = model_name
        self._explicit_embedder = embedder
        self._embedder: Optional[Embedder] = (
            resolve_embedder(embedder) if (embedder is not None) else None
        )

    def _embedder_or_default(self) -> Embedder:
        if self._embedder is None:
            from ._embeddings import default_embedder

            self._embedder = default_embedder(self._model_name)
        return self._embedder

    def _score_tools(
        self, tools: Sequence[Any], step_text: str
    ) -> List[float]:
        tool_texts = [_tool_text(t) for t in tools]

        if self.use_embeddings:
            try:
                embedder = self._embedder_or_default()
                tool_vecs = np.asarray(embedder(tool_texts), dtype=np.float32)
                step_vec = np.asarray(embedder([step_text]), dtype=np.float32)
                sims = cosine_similarity_matrix(tool_vecs, step_vec)[:, 0]
                sims = np.clip((sims + 1.0) / 2.0, 0.0, 1.0)
                return [float(v) for v in sims.tolist()]
            except ImportError:
                # No embedding backend available: degrade to lexical matching.
                pass
        return [_lexical_score(step_text, tt) for tt in tool_texts]

    def match(
        self,
        tools: Sequence[Any],
        current_step: Union[Step, str],
    ) -> List[ToolMatch]:
        """Score every tool against the current step and mark keep/drop.

        Returns a list of :class:`ToolMatch` aligned with the input order.
        """

        tools = list(tools)
        if not tools:
            return []

        step_text = current_step.as_text() if isinstance(current_step, Step) else str(current_step)
        scores = self._score_tools(tools, step_text)

        kept_flags = [score >= self.threshold for score in scores]

        # Guarantee a minimum number of tools survive if requested.
        if self.min_tools > 0 and sum(kept_flags) < self.min_tools:
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            for i in order[: self.min_tools]:
                kept_flags[i] = True

        return [
            ToolMatch(
                tool=tools[i],
                name=_tool_name(tools[i]),
                score=scores[i],
                kept=kept_flags[i],
            )
            for i in range(len(tools))
        ]

    def filter(
        self,
        tools: Sequence[Any],
        current_step: Union[Step, str],
    ) -> List[Any]:
        """Return only the tool schemas relevant to the current step."""

        return [m.tool for m in self.match(tools, current_step) if m.kept]


def filter_tools(
    tools: Sequence[Any],
    current_step: Union[Step, str],
    *,
    embedder: Optional[Union[Embedder, object]] = None,
    threshold: float = 0.3,
    min_tools: int = 0,
    use_embeddings: bool = True,
) -> List[Any]:
    """Functional shortcut for :class:`ToolFilter`.

    Args:
        tools: The full list of tool schemas.
        current_step: The current step (a :class:`Step` or a description).
        embedder: Optional custom embedder.
        threshold: Minimum match score to keep a tool.
        min_tools: Always keep at least this many top tools.
        use_embeddings: Whether to use semantic embeddings.

    Returns:
        The subset of ``tools`` relevant to the current step.
    """

    return ToolFilter(
        embedder=embedder,
        threshold=threshold,
        min_tools=min_tools,
        use_embeddings=use_embeddings,
    ).filter(tools, current_step)
