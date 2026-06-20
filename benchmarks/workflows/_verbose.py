"""Generate verbose supplemental benchmark text (deterministic, no LLM)."""

from __future__ import annotations


def _paragraphs(prefix: str, count: int = 40) -> str:
    lines = []
    for i in range(1, count + 1):
        lines.append(
            f"{prefix} item {i}: Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
            f"Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
            f"Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut "
            f"aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in "
            f"voluptate velit esse cillum dolore eu fugiat nulla pariatur."
        )
    return "\n".join(lines)


def research_noise() -> str:
    return (
        "ARCHIVED SEARCH NOISE (superseded by filter step — low relevance to synthesis):\n"
        + _paragraphs("SEO blog", 50)
    )


def code_noise() -> str:
    return (
        "LEGACY LINTER OUTPUT (pre-fix branch, already resolved in scan phase):\n"
        + _paragraphs("ruff warning", 50)
    )


def data_noise() -> str:
    return (
        "RAW WAREHOUSE QUERY TRACE (exploratory only, superseded by clean tables):\n"
        + _paragraphs("row batch", 45)
    )
