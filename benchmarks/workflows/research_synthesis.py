"""Benchmark workflow: research → filter → synthesize → outline → draft."""

from __future__ import annotations

from memahead import Plan, Step

from benchmarks.workflows._verbose import research_noise

PLAN = Plan(
    [
        Step("research", "Search the web and gather raw facts about the topic"),
        Step("filter", "Remove irrelevant sources and deduplicate findings"),
        Step("synthesize", "Identify key themes and contradictions across sources"),
        Step("outline", "Create a structured outline for the final report"),
        Step("draft", "Write a complete first draft of the report"),
    ]
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the public web for recent articles, papers, and reports. "
                "Returns ranked snippets with URLs, publication dates, and extracted "
                "passages. Use for broad discovery before fetching full pages."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search query.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (1-20).",
                        "default": 10,
                    },
                    "recency_days": {
                        "type": "integer",
                        "description": "Limit results to the last N days when set.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": (
                "Fetch and extract readable text from a URL. Supports HTML articles, "
                "PDF links, and arXiv abstracts. Returns title, author, date, and body."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Fully qualified URL."},
                    "include_references": {
                        "type": "boolean",
                        "description": "Whether to include bibliography sections.",
                        "default": False,
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "image_generator",
            "description": (
                "Generate decorative images, diagrams, or marketing artwork from a "
                "text prompt. Not suitable for data extraction or factual research."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "style": {
                        "type": "string",
                        "enum": ["photo", "illustration", "diagram"],
                    },
                    "size": {
                        "type": "string",
                        "enum": ["512x512", "1024x1024"],
                        "default": "1024x1024",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": (
                "Send an email via the connected workspace account. Requires "
                "explicit user approval for external recipients."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "array", "items": {"type": "string"}},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or overwrite a text file in the workspace sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from the workspace sandbox.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
]

_SEARCH_RESULTS = """
Web search results for: "large language models impact software engineering productivity 2024 2025 peer reviewed"

[1] Peng et al. (2023) — "The Impact of AI on Developer Productivity" (arXiv:2302.06590)
Snippet: Controlled experiment with GitHub Copilot reported productivity gains of 15-26% on routine coding tasks
for experienced developers completing HTTP server implementations. Effect sizes were smaller for unfamiliar APIs.
URL: https://arxiv.org/abs/2302.06590

[2] Noy & Zhang (2023) — MIT/BCG writing task study (Science)
Snippet: Professionals using ChatGPT completed writing assignments 37% faster with quality judged equal or better.
Not specific to code but cited in SE policy decks.
URL: https://www.science.org/doi/10.1126/science.adh2586

[3] McKinsey (2024) — "The state of AI in 2024"
Snippet: Survey claims 29% of digital organizations report material EBIT impact from genAI; engineering use cases
include code generation, test synthesis, and legacy migration. Methodology: self-reported survey, n=1,363.
URL: https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai

[4] GitHub (2024) — Octoverse / Copilot metrics blog
Snippet: GitHub Copilot adoption reached 1.8M paid subscribers in 2024; internal telemetry shows acceptance rate
near 30% for suggested lines in supported languages. Marketing post, not peer-reviewed.
URL: https://github.blog/news-insights/octoverse/

[5] Pearce et al. (2022) — "Asleep at the Keyboard" (IEEE S&P)
Snippet: Security review burden increased when teams adopted codegen without policy — 40% of Copilot-generated
solutions contained vulnerabilities in their benchmark of high-risk prompts.
URL: https://ieeexplore.ieee.org/document/9833571

[6] Hofmann et al. (2024) — controlled lab study at enterprise SaaS vendor
Snippet: Mixed results: junior developers sped up boilerplate but senior developers spent more time reviewing
diffs. Net calendar time unchanged over two-week sprint in their preprint.
URL: https://example.org/preprint/hofmann-2024-codegen-sprints

[7] Stack Overflow Developer Survey 2024
Snippet: 76% of respondents are using or plan to use AI tools; top concern is "AI solutions that are almost right."
Anecdotal, not peer-reviewed.
URL: https://survey.stackoverflow.co/2024/

[8] Internal duplicate — same Peng 2023 abstract mirrored on Medium aggregator (discard)
URL: https://medium.com/@mirror/arxiv-2302-06590-copy

Total raw hits: 847 (showing top 8). Query reformulation suggested: add "randomized controlled trial" filter.
""".strip()

_FETCH_PENG = """
Fetched: https://arxiv.org/abs/2302.06590
Title: The Impact of AI on Developer Productivity: Evidence from GitHub Copilot
Authors: Sida Peng, Eirini Kalliamvakou, Peter Cihon, Mert Demirer
Abstract: We conduct a randomized controlled trial with 95 professional developers completing
realistic implementation tasks. Treatment group with Copilot completes tasks 55.8% faster on average
(median effect aligns with 15-26% on routine coding tasks depending on task stratification).
Limitations: tasks are isolated; no multi-week maintenance follow-up.
Sections extracted: Introduction, Experimental Design, Results, Threats to Validity, Conclusion.
""".strip()

_FILTER_LOG = """
Filter step completed.
- Removed 12 duplicate URLs (including Medium mirror of Peng 2023).
- Dropped 5 SEO spam domains.
- Excluded Stack Overflow survey from peer-reviewed set but tagged as "industry sentiment".
- Excluded McKinsey and GitHub blog from evidence table per constraint: report must cite only peer-reviewed sources.
- Kept: Peng 2023, Noy & Zhang 2023, Pearce 2022, Hofmann 2024 preprint (flagged as preprint).
Unresolved: whether to include Hofmann preprint in final bibliography or move to appendix.
""".strip()

_EXTRA_SEARCH = """
Second search pass: "developer trust LLM code suggestions style guide conflict qualitative"

[1] Hofmann et al. (2024) preprint discussion section (extract)
Developers reported frustration when suggestions violated internal lint rules; several teams disabled inline suggestions
for legacy modules. Quote: "trust erosion when LLM suggestions conflict with internal style guides."

[2] Reddit r/programming thread (excluded as anecdotal HN/forum content)
Various complaints about Copilot suggesting deprecated APIs — not peer-reviewed.

[3] Google DORA 2024 report
AI adoption correlates with delivery instability in low-maturity teams; causality unclear.

[4] ACM Queue interview with staff engineer at fintech
Anecdote: review time doubled after codegen rollout — editorial, not a study.

[5] Internal style guide excerpt (Acme Corp Python 2024)
Mandatory: black, ruff, no bare except, Decimal for money. LLM suggestions often violate bare except rule.

[6] Duplicate Peng abstract — skip

Total hits: 312. Recommendation: keep Hofmann qualitative quotes as supporting context only in synthesis, not evidence table.
""".strip()

_EXTRA_FETCH_NOY = """
Fetched: Noy & Zhang (2023) Science study (writing tasks, not code-specific)
Design: 453 college-educated professionals, grant writing, press releases, analytical tasks.
Finding: ChatGPT reduced completion time by 40% (37% in abstract) with equal or better quality ratings.
External validity to software engineering: limited — included for completeness but not central thesis.
Authors note skill displacement concerns for entry-level workers.
""".strip()

HISTORY = [
    {
        "role": "user",
        "content": (
            "Research the impact of LLMs on software development for an internal strategy memo. "
            "constraint: report must cite only peer-reviewed sources in the final evidence table."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "I'll start with broad web searches, then fetch primary papers, filter to peer-reviewed "
            "sources, and synthesize themes before we outline the memo."
        ),
    },
    {"role": "tool", "content": _SEARCH_RESULTS},
    {
        "role": "assistant",
        "content": (
            "Search returned 847 hits. I see Peng 2023 RCT, Pearce 2022 security study, and several "
            "industry surveys. Next I'll fetch the Peng paper and the IEEE security paper in full."
        ),
    },
    {"role": "tool", "content": _FETCH_PENG},
    {
        "role": "assistant",
        "content": (
            "Peng confirms productivity gains of 15-26% on routine coding tasks in their stratified "
            "analysis. I'll note the 55.8% headline is for their specific task battery."
        ),
    },
    {"role": "tool", "content": _EXTRA_SEARCH},
    {
        "role": "assistant",
        "content": (
            "Second search reinforces trust/style friction theme from Hofmann discussion. "
            "Excluding forum anecdotes per decision: exclude anecdotal HN threads from final report."
        ),
    },
    {"role": "tool", "content": _EXTRA_FETCH_NOY},
    {
        "role": "assistant",
        "content": (
            "Noy & Zhang included as peer-reviewed but peripheral to code-specific claims. "
            "Will mention briefly in contradictions section."
        ),
    },
    {
        "role": "user",
        "content": "Quick aside — are you free for coffee at 3pm? Also the weather looks terrible today.",
    },
    {
        "role": "assistant",
        "content": (
            "I'm an agent and don't drink coffee, but I can keep researching. Returning to sources: "
            "fetching Pearce 2022 on security implications."
        ),
    },
    {
        "role": "tool",
        "content": (
            "Fetched Pearce 2022 (IEEE S&P). Key finding: security review burden increased when teams "
            "adopted codegen without policy — 40% of generated solutions contained vulnerabilities "
            "in their high-risk prompt benchmark. Recommends mandatory security review gates for "
            "all Copilot-generated patches touching payment code."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Security theme is strong. I'll also capture developer trust erosion when LLM suggestions "
            "conflict with internal style guides — saw that in Hofmann preprint discussion section."
        ),
    },
    {"role": "tool", "content": _FILTER_LOG},
    {"role": "tool", "content": research_noise()},
    {
        "role": "assistant",
        "content": (
            "Filter step done. decision: exclude anecdotal HN threads from final report. "
            "Peer-reviewed set is Peng, Pearce, Noy & Zhang; Hofmann stays flagged."
        ),
    },
    {
        "role": "user",
        "content": (
            "My manager asked if we should buy Copilot seats for everyone. Can you answer that now?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "That's a procurement decision we shouldn't rush before synthesis. Evidence is mixed: "
            "productivity gains of 15-26% on routine coding tasks vs increased security review burden. "
            "I'll address recommendations after themes are consolidated."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Intermediate conclusion (research phase): three themes emerging — (1) measured speedups on "
            "routine work, (2) security/process externalities, (3) trust and style-guide friction. "
            "theme: developer trust erosion when LLM suggestions conflict with internal style guides."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Synthesis rollup (for outline/draft): "
            "productivity gains of 15-26% on routine coding tasks; "
            "security review burden increased when teams adopted codegen without policy; "
            "constraint: report must cite only peer-reviewed sources; "
            "theme: developer trust erosion when LLM suggestions conflict with internal style guides; "
            "decision: exclude anecdotal HN threads from final report; "
            "GitHub Copilot adoption reached 1.8M paid subscribers in 2024 (market context only)."
        ),
    },
    {
        "role": "user",
        "content": (
            "Great. We're at the synthesize step now — identify key themes and contradictions across "
            "the filtered sources. Don't draft yet."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Understood. I will synthesize themes for outline and draft steps: productivity evidence, "
            "security review burden, trust/style friction, and contradictions between lab RCTs and "
            "enterprise sprint studies."
        ),
    },
    {
        "role": "user",
        "content": (
            "Reminder: GitHub Copilot adoption reached 1.8M paid subscribers in 2024 is market context "
            "only — do not treat as peer-reviewed evidence."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Noted. Market stats excluded from evidence table per peer-reviewed constraint."
        ),
    },
]

CRITICAL_FACTS = [
    "productivity gains of 15-26% on routine coding tasks",
    "security review burden increased when teams adopted codegen without policy",
    "mandatory security review gates for all Copilot-generated patches",
    "constraint: report must cite only peer-reviewed sources",
    "theme: developer trust erosion when LLM suggestions conflict with internal style guides",
    "decision: exclude anecdotal HN threads from final report",
    "GitHub Copilot adoption reached 1.8M paid subscribers in 2024",
]


def get_workflow(current_step: str = "synthesize"):
    return {
        "plan": PLAN,
        "current_step": current_step,
        "history": HISTORY,
        "tools": TOOLS,
    }
