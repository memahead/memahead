"""End-to-end example of plan-aware context compression with memahead.

Run it directly::

    python examples/basic_workflow.py

By default this example uses a small, dependency-free hashing embedder so it
runs instantly and offline. In production you would simply omit the
``embedder=`` argument and memahead will use the ``all-MiniLM-L6-v2``
sentence-transformers model for far better semantic matching.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from memahead import Plan, PlanAwareCompressor, Step


# A tiny fixed vocabulary so the demo is fully reproducible and offline.
# Real usage relies on the all-MiniLM-L6-v2 sentence-transformers model instead.
_VOCAB = [
    "research", "facts", "themes", "synthesize", "identify",
    "draft", "write", "structured", "revise", "polish",
    "weather", "lunch", "image", "artwork",
]
_INDEX = {word: i for i, word in enumerate(_VOCAB)}


def vocab_embedder():
    """A small deterministic bag-of-words embedder over a fixed vocabulary.

    This is NOT a good semantic model — it only demonstrates the pipeline
    without downloading anything. In production, omit ``embedder=`` and
    memahead uses ``all-MiniLM-L6-v2``.
    """

    def embed(texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), len(_VOCAB)), dtype=np.float32)
        for row, text in enumerate(texts):
            for raw in text.lower().split():
                token = "".join(c for c in raw if c.isalnum())
                idx = _INDEX.get(token)
                if idx is not None:
                    vectors[row, idx] += 1.0
        return vectors

    return embed


def main() -> None:
    plan = Plan(
        [
            Step("research", "Search and gather raw facts about the topic"),
            Step("synthesize", "Identify key themes across the research findings"),
            Step("draft", "Write a structured first draft of the article"),
            Step("revise", "Produce the final polished output"),
        ]
    )

    # A pile of prior context, only some of which future steps actually need.
    prior_messages = [
        {"role": "system", "content": "You are a meticulous research assistant."},
        {
            "role": "assistant",
            "content": "Raw facts gathered: themes findings research data about "
            "the topic, key statistics and figures.",
        },
        {
            "role": "user",
            "content": "Also here is some unrelated office small talk about "
            "lunch plans and the weather forecast for the weekend.",
        },
        {
            "role": "assistant",
            "content": "Draft outline notes: structured article sections, themes "
            "to synthesize and polish in the final output.",
        },
        {"role": "user", "content": "Now identify the key themes."},
    ]

    all_tool_schemas = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web and gather raw facts about a topic.",
            },
        },
        {
            "type": "function",
            "function": {
                "name": "theme_extractor",
                "description": "Identify key themes across research findings.",
            },
        },
        {
            "type": "function",
            "function": {
                "name": "image_generator",
                "description": "Generate decorative images and artwork.",
            },
        },
    ]

    # In production: PlanAwareCompressor(quality=0.85)  (no embedder argument).
    compressor = PlanAwareCompressor(
        quality=0.85,
        embedder=vocab_embedder(),
        tool_threshold=0.55,
        use_headroom=False,  # demo runs without the optional Headroom backend
    )

    compressed = compressor.compress(
        history=prior_messages,
        tools=all_tool_schemas,
        plan=plan,
        current_step="synthesize",
    )

    print("Remaining steps from 'synthesize':",
          [s.name for s in plan.remaining_from("synthesize")])
    print()
    print("Kept messages:")
    for msg in compressed.messages:
        print("  -", msg["role"], "::", msg["content"][:60])
    print()
    print("Kept tools:", [t["function"]["name"] for t in compressed.tools])
    print()
    print("Dropped:")
    for d in compressed.report.dropped:
        print(f"  - {d.source} ({d.kind}) score={d.score} :: {d.reason}")
    print()
    print(compressed.report)


if __name__ == "__main__":
    main()
