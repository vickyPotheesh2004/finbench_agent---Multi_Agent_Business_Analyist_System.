# FinBench Multi-Agent Business Analyst AI — Phase 1 Benchmark

**Date:** 2026-05-11
**Commit:** 794b832
**Eval:** 196 questions across 7 companies' 10-K filings

## Final Results

**🎯 Overall Accuracy: 84.2% (139/165 numerical questions)**

| Company             | Correct | Wrong | No-Ans | Total | Accuracy |
|---------------------|---------|-------|--------|-------|----------|
| Apple Inc.          | 24      | 0     | 0      | 24    | **100.0%** |
| Nvidia              | 21      | 1     | 1      | 23    | 91.3%    |
| Amazon              | 20      | 3     | 0      | 23    | 87.0%    |
| Alphabet (Google)   | 20      | 2     | 2      | 24    | 83.3%    |
| Meta Platforms      | 19      | 2     | 3      | 24    | 79.2%    |
| Microsoft           | 18      | 6     | 0      | 24    | 75.0%    |
| Tesla               | 17      | 3     | 3      | 23    | 73.9%    |

## Pipeline Performance

- **Eval time:** 6.1 minutes (vs 9 hours before Session 12 fixes)
- **Sniper hit rate:** ~80% of questions (0.1-0.3s each)
- **Early-exit rate:** ~15% (3-15s each, no waste)
- **Ingest time:** 16-63s per document (BGE embedding cost)

## Session 12 Fixes Shipped

| Fix | Impact |
|-----|--------|
| Fix 1: SniperRAG iXBRL context-aware retrieval | Apple 4/15 → 15/15 |
| Fix 2: LLM timeout cascade eliminated | 1200s → 30s worst case |
| Fix 3: 25+ iXBRL aliases | Folded into Fix 1 v6 |
| Fix 4: Chunker coverage | 10% → 100% (30 → 220 chunks) |
| Fix 5-7: Polish | Noise reduction, unit display |

## Phase 2 Targets

Cluster analysis of 17 wrong-value cases:
- **Bug X (8 cases):** Wrong cell on multi-total filings
- **Bug Y (4 cases):** Period mismatch (deferred_revenue, long_term_debt)
- **Bug Z (3 cases):** EPS basic vs diluted confusion

Phase 2 should achieve **92-95% accuracy**.