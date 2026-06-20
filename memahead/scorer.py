"""Forward-looking retention scoring — memahead's core contribution.

Existing context compressors score a chunk by how it relates to *what already
happened*. memahead instead scores each chunk against the **remaining** plan
steps: a chunk is valuable if a future step is likely to need it. This is the
plan-aware retention idea drawn from PAACE (arXiv:2512.16970) and the
chunk-level optimization in ACON (arXiv:2510.00615).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np

from ._embeddings import (
    Embedder,
    cosine_similarity_matrix,
    resolve_embedder,
)
from .plan import Step

__all__ = ["ChunkScore", "RetentionScorer"]


@dataclass
class ChunkScore:
    """The retention score for a single context chunk.

    Attributes:
        index: Position of the chunk in the input list.
        text: The chunk text that was scored.
        score: Aggregate retention score in ``[0.0, 1.0]`` — how much the
            remaining plan steps are likely to need this chunk.
        per_step: Similarity (``[0.0, 1.0]``) of the chunk to each remaining
            step, aligned with the order of the steps passed in.
    """

    index: int
    text: str
    score: float
    per_step: List[float]


class RetentionScorer:
    """Scores context chunks by their usefulness to future plan steps.

    The scorer embeds both the context chunks and the remaining step
    descriptions with the same model (default ``all-MiniLM-L6-v2``), computes
    cosine similarity between every chunk and every remaining step, and
    aggregates per chunk into a single ``[0.0, 1.0]`` retention score.

    Args:
        embedder: Optional custom embedder. Anything callable mapping
            ``list[str] -> np.ndarray`` of shape ``(n, dim)``, or an object
            with an ``.encode`` method, or ``None`` to use the default
            sentence-transformers model. Injecting a custom embedder makes
            the scorer fully testable offline.
        aggregate: How to combine per-step similarities into one score.
            ``"max"`` (default) keeps a chunk if *any* future step needs it;
            ``"mean"`` favors chunks broadly useful across the horizon.
        model_name: Model name for the default embedder (ignored if a custom
            ``embedder`` is supplied).
    """

    def __init__(
        self,
        embedder: Optional[Union[Embedder, object]] = None,
        *,
        aggregate: str = "max",
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        if aggregate not in ("max", "mean"):
            raise ValueError("aggregate must be 'max' or 'mean'")
        self.aggregate = aggregate
        self._model_name = model_name
        self._embedder = (
            resolve_embedder(embedder)
            if embedder is not None
            else None  # defer default-model load until first use
        )

    def _embedder_or_default(self) -> Embedder:
        if self._embedder is None:
            from ._embeddings import default_embedder

            self._embedder = default_embedder(self._model_name)
        return self._embedder

    @staticmethod
    def _step_text(step: Union[Step, str]) -> str:
        if isinstance(step, Step):
            return step.as_text()
        return str(step)

    def score(
        self,
        chunks: Sequence[str],
        remaining_steps: Sequence[Union[Step, str]],
    ) -> List[ChunkScore]:
        """Score each chunk against the remaining steps.

        Args:
            chunks: The context chunks to evaluate.
            remaining_steps: The forward horizon — steps still to come. May be
                :class:`Step` objects or plain strings.

        Returns:
            A list of :class:`ChunkScore`, one per input chunk, in order.

        Notes:
            If ``remaining_steps`` is empty (the current step is the last one),
            every chunk receives a neutral score of ``0.0`` — there is no
            future that needs it, so retention should be decided by other
            policy (e.g. always-keep rules in the compressor).
        """

        chunks = list(chunks)
        if not chunks:
            return []

        steps = list(remaining_steps)
        if not steps:
            return [
                ChunkScore(index=i, text=chunk, score=0.0, per_step=[])
                for i, chunk in enumerate(chunks)
            ]

        embedder = self._embedder_or_default()
        chunk_texts = [c if isinstance(c, str) else str(c) for c in chunks]
        step_texts = [self._step_text(s) for s in steps]

        chunk_vecs = np.asarray(embedder(chunk_texts), dtype=np.float32)
        step_vecs = np.asarray(embedder(step_texts), dtype=np.float32)

        if chunk_vecs.ndim != 2 or step_vecs.ndim != 2:
            raise ValueError(
                "embedder must return a 2-D array of shape (n, dim); got "
                f"chunk shape {chunk_vecs.shape} and step shape {step_vecs.shape}"
            )

        sims = cosine_similarity_matrix(chunk_vecs, step_vecs)
        # Map cosine [-1, 1] into [0, 1] so scores read as probabilities.
        sims = np.clip((sims + 1.0) / 2.0, 0.0, 1.0)

        if self.aggregate == "max":
            aggregated = sims.max(axis=1)
        else:
            aggregated = sims.mean(axis=1)

        results: List[ChunkScore] = []
        for i, chunk in enumerate(chunk_texts):
            results.append(
                ChunkScore(
                    index=i,
                    text=chunk,
                    score=float(aggregated[i]),
                    per_step=[float(v) for v in sims[i].tolist()],
                )
            )
        return results

    def score_values(
        self,
        chunks: Sequence[str],
        remaining_steps: Sequence[Union[Step, str]],
    ) -> List[float]:
        """Convenience wrapper returning just the float scores, in order."""

        return [cs.score for cs in self.score(chunks, remaining_steps)]
