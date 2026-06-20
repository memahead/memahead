"""The plan-aware compression pipeline.

:class:`PlanAwareCompressor` ties the pieces together:

    history + tools + plan + current_step
        -> split history into chunks
        -> score chunks against the *remaining* plan steps (RetentionScorer)
        -> drop chunks future steps won't need (plan-aware retention)
        -> filter tool schemas to the current step (tool_filter, no LLM)
        -> hand survivors to Headroom for the actual compression mechanics
        -> return a CompressedContext (+ TokenReport)

memahead owns the *retention policy*; Headroom owns the *compression
mechanics*. If Headroom is not installed the pipeline still works — it simply
skips the mechanical compression step and relies on retention alone.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

from .context import (
    CompressedContext,
    DroppedChunk,
    TokenReport,
    _message_text,
    count_message_tokens,
    count_tool_tokens,
    count_tokens,
)
from .plan import Plan, PlanGraph, Step
from .scorer import RetentionScorer
from .tool_filter import ToolFilter

__all__ = ["PlanAwareCompressor"]

PlanLike = Union[Plan, PlanGraph]


class PlanAwareCompressor:
    """Compress agent context using forward-looking, plan-aware retention.

    Args:
        quality: Information-retention dial in ``[0.0, 1.0]``. Higher keeps
            more context (gentler compression); lower is more aggressive.
            Defaults to ``0.85``.
        retention_threshold: Optional absolute score cutoff in ``[0.0, 1.0]``.
            When set, chunks scoring below it are dropped, overriding the
            ``quality``-derived relative policy. Useful for reproducible runs.
        tool_threshold: Match cutoff for keeping a tool schema.
        scorer: A custom :class:`RetentionScorer` (e.g. with an injected
            embedder). If ``None``, one is created lazily.
        tool_filter: A custom :class:`ToolFilter`. If ``None``, one is created.
        embedder: Convenience way to inject one embedder into both the scorer
            and the tool filter (ignored if explicit ``scorer``/``tool_filter``
            are given).
        use_headroom: Whether to run survivors through Headroom for mechanical
            compression. Defaults to ``True``; silently no-ops if Headroom is
            unavailable.
        model: Optional model name forwarded to Headroom and the tokenizer.
        keep_system: Always retain ``system`` role messages. Defaults to True.
        keep_last: Always retain the final message (the current turn's input).
            Defaults to True.
    """

    def __init__(
        self,
        quality: float = 0.85,
        *,
        retention_threshold: Optional[float] = None,
        tool_threshold: float = 0.3,
        scorer: Optional[RetentionScorer] = None,
        tool_filter: Optional[ToolFilter] = None,
        embedder: Optional[Any] = None,
        use_headroom: bool = True,
        model: Optional[str] = None,
        keep_system: bool = True,
        keep_last: bool = True,
    ) -> None:
        if not 0.0 <= quality <= 1.0:
            raise ValueError("quality must be in [0.0, 1.0]")
        if retention_threshold is not None and not 0.0 <= retention_threshold <= 1.0:
            raise ValueError("retention_threshold must be in [0.0, 1.0]")

        self.quality = quality
        self.retention_threshold = retention_threshold
        self.use_headroom = use_headroom
        self.model = model
        self.keep_system = keep_system
        self.keep_last = keep_last

        self.scorer = scorer or RetentionScorer(embedder=embedder)
        self.tool_filter = tool_filter or ToolFilter(
            embedder=embedder, threshold=tool_threshold
        )

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _resolve_step(plan: PlanLike, current_step: Union[Step, str]) -> Step:
        if isinstance(current_step, Step):
            return current_step
        return plan.get(current_step)

    @staticmethod
    def _is_system(message: Any) -> bool:
        return isinstance(message, dict) and message.get("role") == "system"

    def _always_keep_mask(self, history: List[Any]) -> List[bool]:
        n = len(history)
        mask = [False] * n
        for i, msg in enumerate(history):
            if self.keep_system and self._is_system(msg):
                mask[i] = True
        if self.keep_last and n > 0:
            mask[n - 1] = True
        return mask

    def _decide_retention(
        self,
        scores: List[float],
        always_keep: List[bool],
        has_future: bool,
    ) -> List[bool]:
        """Return a keep/drop flag per chunk from scores + policy."""

        n = len(scores)
        if n == 0:
            return []

        # No future steps -> nothing to prune against; keep everything.
        if not has_future:
            return [True] * n

        if self.retention_threshold is not None:
            keep = [s >= self.retention_threshold for s in scores]
        else:
            # Relative policy: min-max normalize, then keep the top band as
            # governed by `quality`. quality=0.85 -> keep normalized >= 0.15.
            lo = min(scores)
            hi = max(scores)
            cutoff = 1.0 - self.quality
            if hi - lo < 1e-9:
                # All equal: a flat horizon. Keep them all rather than guess.
                keep = [True] * n
            else:
                keep = [((s - lo) / (hi - lo)) >= cutoff for s in scores]

        for i in range(n):
            if always_keep[i]:
                keep[i] = True
        return keep

    def _apply_headroom(self, messages: List[Any]) -> List[Any]:
        """Run messages through Headroom's ``compress`` if available.

        Defensive by design: any import error, signature mismatch, or
        unexpected return shape falls back to the input unchanged so that
        retention-only compression still works.
        """

        if not self.use_headroom or not messages:
            return messages
        try:
            from headroom import compress  # type: ignore
        except Exception:
            return messages

        try:
            result = compress(messages, model=self.model) if self.model else compress(messages)
        except TypeError:
            try:
                result = compress(messages)
            except Exception:
                return messages
        except Exception:
            return messages

        return self._normalize_headroom_result(result, fallback=messages)

    @staticmethod
    def _normalize_headroom_result(result: Any, fallback: List[Any]) -> List[Any]:
        if result is None:
            return fallback
        if isinstance(result, list):
            return result
        # Common attribute names across compression libraries.
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

    # -- public API ---------------------------------------------------------

    def compress(
        self,
        history: Sequence[Any],
        tools: Sequence[Any],
        plan: PlanLike,
        current_step: Union[Step, str],
    ) -> CompressedContext:
        """Compress ``history`` and ``tools`` for the given step of ``plan``.

        Args:
            history: Prior chat messages (dicts with ``role``/``content``, or
                plain strings). Each message is treated as one context chunk.
            tools: The full catalog of tool schemas available to the agent.
            plan: The :class:`Plan` (or :class:`PlanGraph`) being executed.
            current_step: The step about to run — the pivot for "what's ahead".

        Returns:
            A :class:`CompressedContext` with lean ``messages``, filtered
            ``tools``, and a :class:`TokenReport`.
        """

        history = list(history)
        tools = list(tools or [])

        step = self._resolve_step(plan, current_step)
        step_key = step.name if isinstance(current_step, Step) else str(current_step)
        remaining_steps = plan.remaining_from(step_key)
        has_future = len(remaining_steps) > 0

        before_tokens = count_message_tokens(history, self.model) + count_tool_tokens(
            tools, self.model
        )

        # 1) chunk + score against the forward horizon.
        chunk_texts = [_message_text(m) for m in history]
        always_keep = self._always_keep_mask(history)

        if history:
            chunk_scores = self.scorer.score(chunk_texts, remaining_steps)
            scores = [cs.score for cs in chunk_scores]
        else:
            scores = []

        # 2) decide retention.
        keep_flags = self._decide_retention(scores, always_keep, has_future)

        retained_messages: List[Any] = []
        retained_scores: Dict[str, float] = {}
        dropped: List[DroppedChunk] = []
        for i, msg in enumerate(history):
            source = f"message[{i}]"
            tok = count_tokens(chunk_texts[i], self.model)
            score = scores[i] if i < len(scores) else None
            if keep_flags[i]:
                retained_messages.append(msg)
                if score is not None:
                    retained_scores[source] = round(score, 4)
            else:
                dropped.append(
                    DroppedChunk(
                        source=source,
                        kind="message",
                        tokens_before=tok,
                        tokens_after=0,
                        score=round(score, 4) if score is not None else None,
                        reason="below retention threshold for remaining plan steps",
                    )
                )

        # 3) filter tools to the current step (deterministic, no LLM call).
        tool_matches = self.tool_filter.match(tools, step)
        kept_tools = [m.tool for m in tool_matches if m.kept]
        for m in tool_matches:
            if not m.kept:
                dropped.append(
                    DroppedChunk(
                        source=f"tool:{m.name or '?'}",
                        kind="tool",
                        tokens_before=count_tokens(_tool_schema_text(m.tool), self.model),
                        tokens_after=0,
                        score=round(m.score, 4),
                        reason="tool not relevant to current step",
                    )
                )

        # 4) hand survivors to Headroom for mechanical compression.
        compressed_messages = self._apply_headroom(retained_messages)

        after_tokens = count_message_tokens(
            compressed_messages, self.model
        ) + count_tool_tokens(kept_tools, self.model)

        report = TokenReport(
            before=before_tokens,
            after=after_tokens,
            dropped=dropped,
        )

        return CompressedContext(
            messages=compressed_messages,
            tools=kept_tools,
            report=report,
            retained_scores=retained_scores,
        )


def _tool_schema_text(tool: Any) -> str:
    from .context import _tool_text

    return _tool_text(tool)
