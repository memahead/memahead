# memahead — Project Context

_Stable facts about this project. Rarely changes. Read before making 
any architectural decisions._

---

## What this is
A Python library. Not a framework, not a service, not a wrapper. 
A library that plugs into whatever the user is already using.

## The one-liner
"Compress what your agent remembers based on where it's going, 
not just where it's been."

## The core insight
Every step in a multi-agent workflow receives the full accumulated 
context from all prior steps. Most of that context is irrelevant to 
what the current step needs to do — but nobody has built a library 
that uses knowledge of the REMAINING plan to make smarter keep/drop 
decisions. That's what memahead does.

## What makes it different from existing tools
| Tool | Plan-aware | Usable library | Framework-agnostic |
|------|-----------|----------------|-------------------|
| PAACE | ✅ | ❌ | ❌ |
| ACON (Microsoft) | Partial | ❌ | ❌ |
| Headroom | ❌ | ✅ | ✅ |
| LLMLingua | ❌ | ✅ | ✅ |
| memahead | ✅ | ✅ | ✅ |

## Relationship with Headroom
memahead sits ABOVE Headroom. Headroom handles compression mechanics. 
memahead decides what to compress aggressively vs preserve, based on 
the plan. They are complementary, not competing.

Headroom repo: https://github.com/headroomlabs-ai/headroom
Headroom does NOT know about plans, remaining steps, or future needs.

## Academic foundation
- PAACE (arXiv:2512.16970) — plan-aware retention scoring, Dec 2025
- ACON (arXiv:2510.00615) — compression guideline optimization, Oct 2025, 
  updated June 2026 with GPT-5 results and MEM1 comparison
- ContextBudget (arXiv:2604.01664) — budget-aware compression, April 2026 
  (inspired budget_tokens parameter)
- Focus (arXiv:2601.07190) — agent-driven autonomous compression, Jan 2026
- ACC (arXiv:2601.11653) — cognitive compressor, Jan 2026

## Naming conventions
- `Plan` — ordered list of Steps
- `Step` — has name (slug) and description (natural language)
- `PlanAwareCompressor` — the main public class
- `CompressedContext` — what compress() returns
- `TokenReport` — the cost accounting object
- `RetentionScorer` — the embedding-based scorer
- `BudgetExceededError` — raised when budget_tokens can't be met

## Public API (don't break these without a major version bump)
```python
from memahead import (
    Plan,
    Step,
    PlanAwareCompressor,
    CompressedContext,
    TokenReport,
    BudgetExceededError,
)

compressor = PlanAwareCompressor(quality=0.85, budget_tokens=None)
result = compressor.compress(history, tools, plan, current_step)
result.messages    # compressed messages ready for LLM call
result.tools       # filtered tool schemas
result.report      # TokenReport
```

## Quality parameter guidance
- 0.95+ — very conservative, minimal compression, use for critical workflows
- 0.85 — recommended default, per ACON v3 moderate threshold analysis
- 0.70 — aggressive, good for cost-sensitive workflows with redundant context
- Below 0.60 — not recommended, risk of dropping needed context

## What NOT to do
- Don't add LangGraph/CrewAI as core dependencies — keep them optional extras
- Don't hardcode benchmark numbers — always run and measure
- Don't change the quality parameter to map to absolute thresholds — 
  relative min-max is more robust across embedding models
- Don't make Headroom required — always fall back gracefully
- Don't add a server, API endpoint, or hosted component to this repo — 
  that belongs in a separate commercial repo when the time comes

## Target user
A backend/ML engineer running multi-step agent workflows in production 
who is paying real money for LLM inference and wants to reduce that cost 
without degrading output quality. They are comfortable with pip install 
and reading source code. They are NOT a researcher.

## License
Apache 2.0 — chosen deliberately. Maximizes adoption. 
Commercial hosted layer (future) will be separate repo/product.

## Org structure
- github.com/memahead — GitHub org (personal account, spalakollu)
- github.com/memahead/memahead — core library (this repo)
- github.com/memahead/.github — org profile README
- pypi.org/project/memahead — PyPI package
- memahead.com — domain (not yet built out)

## Contact
spalakollu@gmail.com
