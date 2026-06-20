"""Internal embedding utilities shared by the scorer and the tool filter.

This module isolates the (optional, heavyweight) ``sentence-transformers``
dependency behind a tiny, swappable interface. Anything that produces a 2-D
array of row vectors from a list of strings can be used as an *embedder*,
which keeps the rest of the library testable without downloading a model.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Union

import numpy as np

__all__ = [
    "Embedder",
    "SentenceTransformerEmbedder",
    "default_embedder",
    "resolve_embedder",
    "cosine_similarity_matrix",
]

# An embedder is any callable that maps a list of texts to a (n, dim) matrix.
Embedder = Callable[[Sequence[str]], np.ndarray]

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class SentenceTransformerEmbedder:
    """Lazy wrapper around a ``sentence-transformers`` model.

    The model is only imported and loaded on first use, so importing
    :mod:`memahead` stays cheap and offline-friendly. The default model is
    ``all-MiniLM-L6-v2`` as described in the PAACE/ACON-inspired design.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ImportError(
                    "sentence-transformers is required for the default embedder. "
                    "Install it with `pip install sentence-transformers`, or pass a "
                    "custom `embedder` callable to RetentionScorer / the tool filter."
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        model = self._ensure_model()
        vectors = model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return np.asarray(vectors, dtype=np.float32)


# Process-wide cache so repeated scorers reuse the loaded weights.
_DEFAULT_EMBEDDER: SentenceTransformerEmbedder | None = None


def default_embedder(model_name: str = DEFAULT_MODEL) -> SentenceTransformerEmbedder:
    """Return a cached default embedder backed by ``sentence-transformers``."""

    global _DEFAULT_EMBEDDER
    if _DEFAULT_EMBEDDER is None or _DEFAULT_EMBEDDER.model_name != model_name:
        _DEFAULT_EMBEDDER = SentenceTransformerEmbedder(model_name)
    return _DEFAULT_EMBEDDER


def resolve_embedder(
    embedder: Union[Embedder, "SentenceTransformerEmbedder", None],
) -> Embedder:
    """Normalize the many ways a caller can supply an embedder.

    Accepts ``None`` (use the default model), a plain callable, or any object
    exposing an ``encode`` method (e.g. a raw ``SentenceTransformer``).
    """

    if embedder is None:
        return default_embedder()
    if callable(embedder):
        return embedder
    encode = getattr(embedder, "encode", None)
    if callable(encode):
        return lambda texts: np.asarray(encode(list(texts)), dtype=np.float32)
    raise TypeError(
        "embedder must be None, a callable, or expose an `encode` method; "
        f"got {type(embedder)!r}"
    )


def _l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, eps)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the (len(a), len(b)) matrix of cosine similarities.

    Rows correspond to vectors in ``a`` (e.g. context chunks); columns to
    vectors in ``b`` (e.g. remaining plan steps). Inputs need not be
    pre-normalized.
    """

    a_norm = _l2_normalize(a)
    b_norm = _l2_normalize(b)
    return a_norm @ b_norm.T
