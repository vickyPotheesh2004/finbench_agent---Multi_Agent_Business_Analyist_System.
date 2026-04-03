# CONTEXT.md — FinBench Multi-Agent Business Analyst AI
# PASTE THIS ENTIRE FILE AT THE START OF EVERY NEW CLAUDE SESSION
# ═══════════════════════════════════════════════════════════════

BUILD_STEP: Week 13, Day 1
PHASE: Phase 7A — Live Data Infrastructure
LAST_GATE: M1 ✓ M4 ✓ PASSED | M2 M3 M5 M6 M7 M8 PENDING
PROJECT_GOAL: FB>=90% BB>=80% LFQ>=75% | $0 | 100% local | RLEF

## PROJECT FOLDER
PROJECT FOLDER : D:\projects\finbench_agent
VENV ACTIVATE  : cd "D:\projects\finbench_agent" then venv\scripts\activate
CORRECT PROMPT : (venv) PS D:\projects\finbench_agent>

## AMENDMENTS
A1-A6: Original ✓
A7:  FB target >=90% launch
A8:  BB target >=80% launch
A9:  Live Financial QA benchmark pre-launch
A10: SFT pairs = 2000
A11: Sprint sessions = 10,000
A12: DPO rounds = 3 pre-launch
A13: BGE-M3 Gate M3 MRR>=0.90
A14: LLM Gate M5 >=80% held-out
A15: XGB Gate M6 >=3% improvement
A16: All Phase 7 live data pre-launch
A17: K-Means clusters = 6

## GATE STATUS
M1 PASSED ✓  M4 PASSED ✓
M2 PENDING   M3 PENDING   M5 PENDING
M6 PENDING   M7 PENDING   M8 PENDING

## ALL 19 NODES COMPLETE ✓
N01-N19 all built and tested

## TEST RESULTS
pytest tests\ -q → 475/475 PASSED (114s) [pipeline excluded]
pytest tests\test_pipeline.py -q → 24/24 PASSED fresh session

## PIPELINE
src/pipeline/pipeline.py ✓ FinBenchPipeline
  - ingest() N01-N03
  - query()  N04-N19
  - Parallel pods N11+N12+N13+N14
  - SniperRAG conditional edge
  - Gate M4 PASSED

## pytest.ini
addopts = --ignore=tests/test_pipeline.py
Pipeline tests run separately in fresh PowerShell

## DAILY STARTUP
cd "D:\projects\finbench_agent"
venv\scripts\activate
pytest tests\ -q --tb=no → 475/475
Fresh PS: pytest tests\test_pipeline.py -q --tb=no → 24/24

## NEXT PHASE — Phase 7A Live Data
Build order:
  Session A: src/live_data/cache_manager.py (SQLite TTL cache)
             src/live_data/base_fetcher.py  (replace stub)
             src/live_data/data_shield.py   (replace stub)
             src/live_data/fetch_queue.py   (replace stub)

  Session B: Tier 1 fetchers (6 real APIs)
    edgar.py yfinance.py fred.py fx.py news.py world_bank.py

  Session C: Tier 2 fetchers (10 APIs)
    alpha_vantage treasury bls imf ecb newsapi
    reddit_finance edgar_rss edgar_xbrl coingecko

  Session D: live_context_builder.py + N00.5 node
    BAState new fields:
      live_data_chunks, live_data_summary,
      live_data_freshness, live_data_enabled,
      ticker_symbol, market_cap, current_price, pe_ratio

  Phase 7B: MCP servers (mcp_server_base + per-API)
  Phase 7C: DataShield full (freshness + MNPI guard)
  Phase 7D: Pipeline integration N00.5

AFTER PHASE 7: Streamlit UI → Phase 4 ML training

## CONSTRAINTS
C1:$0 C2:local C3:Llama3.1 8B
C4:14GB(15.4 test) C5:seed=42
C6:DPO beta=0.1 C7:context_first
C8:metadata prefix C9:no _rlef_ C10:ollama pull