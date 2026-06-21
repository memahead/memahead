# memahead — Project Status

_Update this file at the end of every working session._

---

## Current version
**v0.2.0** — live on PyPI · https://pypi.org/project/memahead/0.2.0/

## Repository
https://github.com/memahead/memahead

## What is memahead
Open source Python library for plan-aware context compression in 
multi-agent LLM workflows. Scores each chunk of agent memory against 
remaining workflow steps and drops what future steps won't need.

Based on:
- PAACE (arXiv:2512.16970) — Yuksel et al., Dec 2025
- ACON (arXiv:2510.00615) — Kang et al., Microsoft, Oct 2025

---

## What's built and working

### Core library (`memahead/`)
- `plan.py` — Plan, Step, PlanGraph
- `scorer.py` — RetentionScorer (sentence-transformers, all-MiniLM-L6-v2)
- `compressor.py` — PlanAwareCompressor (quality + budget_tokens parameters)
- `context.py` — CompressedContext, TokenReport
- `tool_filter.py` — deterministic tool schema stripping
- `_embeddings.py` — swappable embedding backend

### Key parameters
- `quality=0.85` — retention threshold (0.0–1.0), moderate value recommended per ACON v3
- `budget_tokens=None` — hard token ceiling, raises BudgetExceededError if unachievable

### Integrations
- Headroom (`headroom-ai`) — compression backend, defensive fallback if not installed
- Framework-agnostic — works with raw LLM calls, no framework required

### Tests
- 65 tests, all passing
- `pytest tests/ -v`

### Benchmarks
- 3 workflows: Research & Synthesis, Code Review, Data Analysis
- Real measured numbers (not hardcoded)
- `python -m benchmarks.run_benchmark`

### Results (latest run)
| Workflow | Before | After | Saved |
|----------|--------|-------|-------|
| Research & Synthesis | 6,240 | 4,795 | 23% |
| Code Review | 5,386 | 2,113 | 61% |
| Data Analysis | 4,821 | 494 | 90% |

### Published assets
- PyPI: https://pypi.org/project/memahead/
- GitHub org: https://github.com/memahead
- Org profile: github.com/memahead (logo + architecture diagram + benchmark table)
- Domain: memahead.com
- License: Apache 2.0
- CI: GitHub Actions, runs on push to main, matrix Python 3.10/3.11/3.12

---

## In progress
- Nothing currently in progress

---

## Up next (prioritized)

### v0.3.0
1. **LangGraph integration** (`memahead[langgraph]`)
   - Read LangGraph state graph directly as Plan — zero Plan declaration required
   - Highest-leverage technical move: largest production agent user base
   
2. **budget_tokens in benchmarks**
   - Add budget-constrained runs to benchmark suite
   - Show three-way comparison: quality-only vs budget-only vs quality+budget

### v0.4.0
3. **Contrastive failure feedback** (`memahead learn`)
   - Log what was dropped that shouldn't have been
   - Iteratively refine compression guidelines per workflow
   - Inspired by ACON v3 contrastive feedback finding

4. **CrewAI integration** (`memahead[crewai]`)
   - Read crew task list as Plan

### Future
5. **Trained retention classifier**
   - Replace sentence-transformers scorer with fine-tuned model
   - Requires usage data — not buildable until meaningful adoption
   
6. **Opt-in telemetry** (v0.2 infra)
   - Anonymized workflow traces → S3
   - Training data for classifier

---

## Known issues / technical debt
- RetentionScorer uses cosine similarity only — doesn't understand semantic 
  dependencies between steps (a constraint in step 2 may be critical for step 5 
  even if it doesn't look similar to step 5's description)
- Headroom fallback is silent — consider a warning when Headroom not installed
- Benchmark workflows are simulated, not real agent traces

---

## Architecture decisions (don't change without discussion)
- Framework-agnostic by design — no LangGraph/CrewAI dependency in core
- Headroom is optional, not required — always fall back gracefully
- quality parameter maps to relative min-max retention cutoff (not absolute)
- Token counting: tiktoken (cl100k_base) with heuristic fallback
- Embedding model: all-MiniLM-L6-v2 — swappable via injectable backend

---

## Recent changes
- v0.2.0: budget_tokens parameter, BudgetExceededError, TokenReport budget fields
- v0.1.0: initial release — Plan, RetentionScorer, PlanAwareCompressor, Headroom integration
