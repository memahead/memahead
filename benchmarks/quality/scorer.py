"""Deterministic quality scoring via critical-fact retention."""

from __future__ import annotations

from typing import Sequence


class QualityScorer:
    """Measures how much critical information survived compression.

    Each workflow defines a set of *critical facts* — specific strings or
    concepts that must survive compression because future steps depend on them.
    The scorer checks what fraction of those facts are still present in the
    compressed context.

    This is intentionally simple and deterministic: no LLM judge, no embeddings.
    """

    def score(
        self,
        original: str,
        compressed: str,
        critical_facts: Sequence[str],
    ) -> float:
        """Return the fraction of critical facts retained, in ``[0.0, 1.0]``."""

        if not critical_facts:
            return 1.0
        haystack = compressed.lower()
        retained = sum(1 for fact in critical_facts if fact.lower() in haystack)
        return retained / len(critical_facts)
