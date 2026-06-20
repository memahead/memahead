"""Shared test fixtures.

The tests must exercise real logic without downloading a 90MB transformer or
hitting the network. We do that by injecting a deterministic *keyword
embedder*: each text is embedded as a bag-of-words vector over a fixed
vocabulary, so cosine similarity is fully predictable and we can assert exactly
which chunks/tools should be retained.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pytest

# Fixed vocabulary -> orthogonal dimensions. Texts that share these words are
# similar; texts that share none are orthogonal (cosine 0).
VOCAB = [
    "research",
    "facts",
    "themes",
    "synthesize",
    "draft",
    "write",
    "revise",
    "polish",
    "search",
    "weather",
    "lunch",
    "image",
]
_INDEX = {word: i for i, word in enumerate(VOCAB)}


def _embed(texts: Sequence[str]) -> np.ndarray:
    vectors = np.zeros((len(texts), len(VOCAB)), dtype=np.float32)
    for row, text in enumerate(texts):
        for raw in text.lower().split():
            token = "".join(c for c in raw if c.isalnum())
            idx = _INDEX.get(token)
            if idx is not None:
                vectors[row, idx] += 1.0
    return vectors


@pytest.fixture
def keyword_embedder() -> Callable[[Sequence[str]], np.ndarray]:
    """Deterministic, offline bag-of-words embedder over a fixed vocabulary."""

    return _embed
