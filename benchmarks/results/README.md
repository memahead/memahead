# memahead Benchmark Results

> Agent memory, optimized for what's ahead.  
> Comparing plan-aware compression vs baselines across realistic agent workflows.

> **memahead reduces token consumption by up to 90% vs no compression  
> and up to 87% vs Headroom alone — with 100% critical fact retention.**

**memahead v0.1.0** · Python 3.13 · Run: 2026-06-20

## Token Reduction

| Workflow | No Compression | Headroom Only | memahead | vs Baseline | vs Headroom |
|----------|---------------|---------------|----------|-------------|-------------|
| Research & Synthesis | 6,240 | 4,991 | 4,795 | −23% | −4% |
| Code Review | 5,386 | 2,278 | 2,113 | −61% | −7% |
| Data Analysis | 4,821 | 3,744 | 494 | −90% | −87% |
| **Average** | **5,482** | **3,671** | **2,467** | **−55%** | **−33%** |

## Quality Retention

| Workflow | Headroom Only | memahead |
|----------|---------------|----------|
| Research & Synthesis | 100.0% | 100.0% |
| Code Review | 100.0% | 100.0% |
| Data Analysis | 100.0% | 100.0% |
| **Average** | **100.0%** | **100.0%** |

## When plan-awareness helps most

Plan-aware compression delivers the largest gains on workflows where
early steps produce verbose output that later steps don't need
(Data Analysis: −87% vs Headroom). The gains are smaller on
workflows where most context remains relevant across all steps
(Research & Synthesis: −4% vs Headroom).

## Latency Overhead

| Workflow | memahead overhead |
|----------|------------------|
| Research & Synthesis | 1725ms |
| Code Review | 522ms |
| Data Analysis | 243ms |

> Latency measured on Apple M-series. Overhead includes retention
> scoring and tool filtering. No LLM calls in the compression path.

## Methodology

- Token counts use `tiktoken` (cl100k_base encoding)
- Quality measured by retention of critical facts defined per workflow
- Headroom baseline uses default compression settings, no plan object
- memahead uses `quality=0.85`, `all-MiniLM-L6-v2` embeddings
- All workflows use realistic simulated agent histories (15-25 turns)

## Reproduce

```bash
pip install memahead headroom-ai
python -m benchmarks.run_benchmark
```
