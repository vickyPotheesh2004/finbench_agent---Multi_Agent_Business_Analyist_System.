# DECISIONS.md — FinBench Multi-Agent Business Analyst AI
# Permanent record of all architectural decisions made in discussion
# Add this file to: D:\projects\finbench_agent\DECISIONS.md
# ═══════════════════════════════════════════════════════════════

## DECISION LOG — Chronological order

---

## DECISION 1 — Real-Time Data Layer (Phase 7)
**Date:** March 2026
**Status:** LOCKED — build post-launch

### What we decided
Add a complete live data layer (Phase 7) to the project AFTER the core
pipeline is complete and FinanceBench ≥82% is confirmed. This is part of
the same project, same codebase, same repository — not a separate project.

### Why
- No open-source BA system combines static document analysis + live data
- Creates a new benchmark category nobody else occupies
- Architecture already supports it — N01 reads HTML/JSON/CSV which is
  what all APIs return
- Only requires adding N00 Live Data Fetcher before N01

### What we are NOT doing
- Not building all 85+ APIs before launch (scope creep risk)
- Not adding live data to the core pipeline now
- Not using any remote services that violate C2

---

## DECISION 2 — 6 Tier 1 APIs Only at Launch
**Status:** LOCKED

### The 6 APIs that launch with Phase 7A
| Priority | API | Why first |
|----------|-----|-----------|
| 1 | SEC EDGAR | Free, reliable, highest BA value, no key |
| 2 | FRED (Federal Reserve) | 800k series, free key, gold standard |
| 3 | yfinance (Yahoo Finance) | Stocks, no key, no limits |
| 4 | Frankfurter API | FX rates, no key, unlimited, ECB data |
| 5 | RSS Feeds | News, no key, unlimited, feedparser |
| 6 | World Bank | 16k indicators, 200 countries, no key |

### All other APIs
Listed in plugin_registry.yaml as enabled: false
Added incrementally post-launch, one at a time
Community can contribute fetchers as pull requests

---

## DECISION 3 — DataShield Architecture (Solves Data Quality)
**Status:** LOCKED — implement in Phase 7A

### Freshness tags on every live data chunk
Every chunk from a live API gets these extra metadata fields:

```
data_source:      "FRED" / "SEC_EDGAR" / "YFINANCE" etc
fetched_at:       ISO timestamp
freshness_status: one of 4 values below
cross_validated:  True / False
```

### Freshness status values
- LIVE_VERIFIED   — < 1 hour old AND cross-validated against 2nd source
- RECENT_24H      — < 24 hours old, not cross-validated
- STALE_7D        — < 7 days old → analyst warned
- OLD             — > 7 days old → HITL review triggered
- UNVERIFIED      — failed cross-validation → HITL review triggered

### Cross-validation rule
When 2 sources report same metric (e.g. FRED GDP + World Bank GDP),
compare values. If they disagree by > 2% → mark UNVERIFIED → HITL fires.

### Where DataShield plugs in
N03 Chunker assigns freshness_status to every live chunk.
Validator V6 checks cross_validated flag.
DOCX output always shows data_source + fetched_at in citations.
Analyst never sees a number without knowing its age and validation status.

---

## DECISION 4 — RAM Protection (Process Isolation)
**Status:** LOCKED — implement in Phase 7A

### The problem
- BGE model: ~500MB RAM
- ChromaDB grows with each document
- Already at 13GB in tests
- Live fetching adds more pressure
- C4 hard cap: 14GB

### The solution
Fetchers run in a completely separate Python subprocess.
Sequence:
  1. Main process requests fetch
  2. Subprocess spawned → fetches API → saves result to disk → exits
  3. Subprocess RAM freed completely
  4. Main process reads result from disk file
  5. N01 ingests the disk file normally

BGE model and fetcher NEVER share RAM.
Peak RAM stays under 12GB.

### Implementation
```python
# fetch_queue.py
import multiprocessing

def fetch_in_isolation(fetcher_name, params):
    process = multiprocessing.Process(
        target=_fetch_worker,
        args=(fetcher_name, params)
    )
    process.start()
    process.join()

def _fetch_worker(fetcher_name, params):
    fetcher = load_fetcher(fetcher_name)
    result  = fetcher.fetch(**params)
    result.save_to_disk()
    # exits here — RAM freed
```

---

## DECISION 5 — BaseAPIFetcher (Solves Maintenance)
**Status:** LOCKED — implement in Phase 7A

### The pattern
Every fetcher inherits from BaseAPIFetcher.
BaseAPIFetcher provides automatically:
  - health_check() — pings API, returns True/False
  - retry logic — 3 attempts with exponential backoff
  - error logging — structured log entry on failure
  - fallback — if API fails, return cached last result from disk
  - save_to_disk() — saves fetched data as JSON
  - validate() — basic schema validation on response

### Daily health checker
Runs at 2am via schedule library.
Pings every enabled API.
Logs failures to logs/api_health_{date}.json
If API fails 3 days in a row → auto-disabled in plugin_registry.yaml
Analyst sees dashboard of API health in Streamlit.

### Result
When an API breaks → one file to fix (the fetcher).
Nothing else in the pipeline changes.
Fix time: 20 minutes, not 2-3 hours.

---

## DECISION 6 — Plugin Registry (Solves Scope Creep)
**Status:** LOCKED — implement in Phase 7A

### The YAML config
src/live_data/plugin_registry.yaml controls everything.
Enable/disable any API with one line change.
No code changes needed to add or remove an API.

### Adding a new API
1. Write one Python file inheriting BaseAPIFetcher
2. Set enabled: true in plugin_registry.yaml
3. Done — it works automatically

### Community contributions
External developers can add fetchers as pull requests.
They write one file. You review. Merge. Done.
You never build all 85+ APIs yourself.

---

## DECISION 7 — MCP Strategy (Future-Proofing)
**Status:** LOCKED

### What we use NOW
- mcp-server-sqlite: Query RLEF database during development
- mcp-server-filesystem: CONTEXT.md management during development
- mcp-server-fetch: Generic URL fetching as fallback

### What we build FOR THE FUTURE
Build fetchers AS local MCP servers from day one.
They run on localhost — C2 compliant, no network calls.
LangGraph calls them via Python today.
When Ollama adds MCP support → zero migration needed.

### What we do NOT do
No remote MCP servers — violates C2.
Ollama/Llama 3.1 8B has no MCP support today.
MCP is for future-proofing, not current inference.

---

## DECISION 8 — Live Financial Intelligence Benchmark (Phase 8)
**Status:** LOCKED — Month 3-4 post-launch

### What we build
eval/run_live_eval.py — new eval script for live data
eval/live_benchmark/questions.json — 200 questions requiring live data
eval/live_benchmark/scoring.py — scoring logic

### Question types
- "What is Apple's current P/E ratio vs their 10-year average?"
- "What did the Fed announce last week and how does it affect JPMorgan?"
- "What is the current India GDP growth rate vs 2019 pre-COVID baseline?"
- "Compare current EUR/USD rate vs rate on the day of Apple FY2023 filing"

### Why this is valuable
No benchmark exists for live financial intelligence.
We define the category. We score first. We submit to Papers With Code.
This is worth more than being #2 on FinanceBench.

### Submission plan
Papers With Code — new benchmark category submission
HuggingFace leaderboard — new dataset + eval script
Academic paper — architecture + benchmark results

---

## IMPLEMENTATION MAP — Where Everything Gets Built

| Decision | Files | Phase | Week |
|----------|-------|-------|------|
| D4 RAM protection | src/live_data/fetch_queue.py | 7A | Post W1 |
| D5 BaseAPIFetcher | src/live_data/base_fetcher.py | 7A | Post W1 |
| D3 DataShield | src/live_data/data_shield.py | 7A | Post W1 |
| D6 Plugin Registry | src/live_data/plugin_registry.yaml | 7A | Post W1 |
| D5 Health Checker | src/live_data/health_checker.py | 7A | Post W1 |
| D7 MCP Bridge | src/live_data/mcp_bridge.py | 7A | Post W1 |
| D2 SEC EDGAR | src/live_data/fetchers/edgar_fetcher.py | 7A | Post W1 |
| D2 FRED | src/live_data/fetchers/fred_fetcher.py | 7B | Post W2 |
| D2 yfinance | src/live_data/fetchers/yfinance_fetcher.py | 7D | Post W5 |
| D2 Frankfurter | src/live_data/fetchers/frankfurter_fetcher.py | 7D | Post W5 |
| D2 RSS News | src/live_data/fetchers/rss_fetcher.py | 7C | Post W4 |
| D2 World Bank | src/live_data/fetchers/worldbank_fetcher.py | 7B | Post W2-3 |
| D8 Live Benchmark | eval/run_live_eval.py | 8 | Month 3 |
| D8 Questions | eval/live_benchmark/questions.json | 8 | Month 3 |
| D8 Paper submission | Papers With Code + HuggingFace | 8 | Month 4 |

---

## COMPLETE FILE STRUCTURE — Phase 7 + 8

```
src/live_data/
├── __init__.py
├── base_fetcher.py          ← BaseAPIFetcher class
├── data_shield.py           ← DataShield freshness tags
├── fetch_queue.py           ← Process isolation (RAM fix)
├── health_checker.py        ← Daily API health checks
├── plugin_registry.yaml     ← Enable/disable per API
├── mcp_bridge.py            ← Local MCP server wrapper
└── fetchers/
    ├── __init__.py
    ├── edgar_fetcher.py     ← SEC EDGAR (Tier 1)
    ├── fred_fetcher.py      ← FRED macro (Tier 1)
    ├── yfinance_fetcher.py  ← Stock prices (Tier 1)
    ├── frankfurter_fetcher.py ← FX rates (Tier 1)
    ├── rss_fetcher.py       ← News feeds (Tier 1)
    ├── worldbank_fetcher.py ← World Bank (Tier 1)
    ├── rbi_fetcher.py       ← RBI India (Tier 2, enabled: false)
    ├── oecd_fetcher.py      ← OECD (Tier 2, enabled: false)
    ├── bis_fetcher.py       ← BIS (Tier 2, enabled: false)
    ├── boe_fetcher.py       ← Bank of England (Tier 2, enabled: false)
    └── [79 more as stubs]   ← enabled: false, community contributes

eval/
├── run_eval.py              ← existing FinanceBench eval
├── run_live_eval.py         ← new live data benchmark (Month 3)
└── live_benchmark/
    ├── __init__.py
    ├── questions.json       ← 200 live-data questions
    └── scoring.py           ← scoring logic
```

---

## WHAT CHANGES IN THE CORE PIPELINE

Almost nothing. Only 2 additions:

1. N00 Live Data Fetcher node added BEFORE N01 in LangGraph pipeline
   - Checks plugin_registry.yaml for enabled fetchers
   - Fetches data via process isolation
   - Saves results as files to data/live_cache/
   - N01 picks up these files alongside uploaded documents

2. N03 Chunker extended to add freshness tags to live chunks
   - Live chunks get DataShield metadata
   - Static document chunks unchanged

Everything else — N04 through N19 — is completely unchanged.
The analysis engine does not know or care whether chunks came from
a PDF or a live API. It just sees text with C8 metadata.

---

## CORE PIPELINE FIRST — NON-NEGOTIABLE RULE

Current build (Weeks 1-18) continues exactly as planned.
Gate M8 (≥82% FinanceBench confirmed) must pass first.
Phase 7 starts ONLY after Gate M8 passes.
NEVER build Phase 7 components before Gate M8.
This rule cannot be changed.

```
CURRENT STATUS:
Week 5 Day 2
N01 ✓  N02 ✓  N03 ✓  N06 ✓  N07 ✓  N08 ✓
N09 ← NEXT (resume building now)
```
