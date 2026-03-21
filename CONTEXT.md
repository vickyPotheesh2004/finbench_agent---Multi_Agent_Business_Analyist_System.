# CONTEXT.md — FinBench Multi-Agent Business Analyst AI
# PASTE THIS ENTIRE FILE AT THE START OF EVERY NEW CLAUDE SESSION
# ═══════════════════════════════════════════════════════════════

BUILD_STEP: Week 6, Day 1
PHASE: Phase 2 — Retrieval (Weeks 3-7)
PHASE_GOAL: N04-N09 retrieval cascade complete
LAST_GATE: M1 PASSED — Week 1
THIS_SESSION_TASK: [REPLACE EACH SESSION — one sentence only]
PROJECT_GOAL: FinanceBench >=82% launch → 91-93% full stack
$0 cost forever | 100% local | Self-improving via RLEF/DPO

## !! PROJECT FOLDER !!
PROJECT FOLDER : D:\projects\finbench_agent
VENV ACTIVATE  : cd "D:\projects\finbench_agent" then venv\scripts\activate
CORRECT PROMPT : (venv) PS D:\projects\finbench_agent>

## AMENDMENTS
A1: PIV REJECT → goes back to PLANNER (not Implementor)
A2: max_retries = 5 per pod (not 3)
A3: After 5 failures → Clarification Engine fires
A4: Target = Top 1 open-source, >=93% full stack
A5: Phase 7 live data layer added (post-launch)
A6: Phase 8 live benchmark + Papers With Code submission

## GATE_STATUS
M1 Schema+Eval     PASSED       Week 1 ✓
M2 Retrieval       PENDING      Week 7
M3 BGE-M3          PENDING      Week 6
M4 Full Pipeline   PENDING      Week 9
M5 LLM SFT         PENDING      Week 12
M6 XGB-Arbiter     PENDING      Week 14
M7 Pre-Sprint      PENDING      Week 15
M8 Launch          PENDING      Sprint End
M9 RLEF Active     PENDING      Post-Launch

## FILES_WRITTEN
src/state/ba_state.py                    ✓
src/utils/seed_manager.py                ✓
src/utils/resource_governor.py           ✓  PYTEST_RUNNING env var support
eval/run_eval.py                         ✓  150 questions
tests/test_ci_gate.py                    ✓  17/17
.github/workflows/tests.yml              ✓
pytest.ini                               ✓  pytest-env, PYTEST_RUNNING=1
src/ingestion/pdf_ingestor.py            ✓  N01
tests/test_pdf_ingestor.py               ✓  24/24
src/ingestion/section_tree_builder.py    ✓  N02
tests/test_section_tree.py               ✓  24/24
src/ingestion/chunker.py                 ✓  N03
tests/test_chunker.py                    ✓  24/24
src/retrieval/sniper_rag.py              ✓  N06 22 patterns
tests/test_sniper_rag.py                 ✓  24/24
src/retrieval/bm25_retriever.py          ✓  N07 LangChain
tests/test_bm25.py                       ✓  24/24
src/retrieval/bge_retriever.py           ✓  N08 384-dim
tests/test_bge_m3.py                     ✓  24/24
src/retrieval/rrf_reranker.py            ✓  N09 RRF+CrossEncoder
tests/test_rrf.py                        ✓  24/24
DECISIONS.md                             ✓  All discussion decisions
src/live_data/base_fetcher.py            ✓  STUB Phase 7A
src/live_data/data_shield.py             ✓  STUB Phase 7A
src/live_data/fetch_queue.py             ✓  STUB Phase 7A
src/live_data/health_checker.py          ✓  STUB Phase 7A
src/live_data/mcp_bridge.py              ✓  STUB Phase 7A
src/live_data/plugin_registry.yaml       ✓  18 APIs configured
src/live_data/fetchers/[18 stubs]        ✓  All Tier 1+2 stubs
eval/run_live_eval.py                    ✓  STUB Phase 8
eval/live_benchmark/questions.json       ✓  STUB Phase 8

## TEST RESULTS
pytest tests\ -q → 185/185 PASSED (38.67s)

## RAM FIX — CRITICAL KNOWLEDGE
Problem: BGE (500MB) + CrossEncoder (90MB) both loaded in same pytest
         process → peak 14.2GB → C4 halt on 15.7GB machine.
Fix:     PYTEST_RUNNING=1 env var → ResourceGovernor uses 15.4GB threshold
         in tests. Production always uses 14GB. C4 not violated.
Requires: pip install pytest-env (already installed)
pytest.ini sets: env = PYTEST_RUNNING=1 automatically.

## CONSTRAINTS C1-C10 (NEVER VIOLATE)
C1: $0 cost. No paid APIs.
C2: 100% local. Zero network calls during inference.
C3: Llama 3.1 8B Q4_K_M via Ollama localhost:11434.
C4: 14GB RAM hard cap (15.4GB in pytest only — see RAM FIX above).
C5: seed=42 everywhere.
C6: DPO beta=0.1 always.
C7: Context BEFORE question in all prompts.
C8: COMPANY/DOCTYPE/FISCAL_YEAR/SECTION/PAGE on every chunk.
C9: _rlef_ fields NEVER in output.
C10: ollama pull distribution.

## TECH_STACK_INSTALLED
pydantic, pytest, pytest-env, numpy, psutil, scipy, datasets
rich, python-dotenv, pyyaml
pdfplumber, pymupdf, python-docx, openpyxl, pandas
Pillow, pytesseract, pdf2image
bm25s, rank-bm25, chromadb, sentence-transformers
langchain==1.2.12, langchain-community
BGE model cached: BAAI/bge-small-en-v1.5 (384-dim)
CrossEncoder cached: cross-encoder/ms-marco-MiniLM-L-6-v2 (90MB)
Tesseract-OCR installed

## RETRIEVAL CASCADE — COMPLETE ✓
N06 SniperRAG    ✓  22 regex patterns, ~50ms, zero GPU
N07 BM25         ✓  bm25s + LangChain, <100ms, zero GPU
N08 BGE-M3       ✓  384-dim semantic, ChromaDB
N09 RRF+Reranker ✓  RRF k=60 merge → CrossEncoder top-3

## PIPELINE PROGRESS
N01 ✓  N02 ✓  N03 ✓  N06 ✓  N07 ✓  N08 ✓  N09 ✓
N04 CART Router     ← NEXT SESSION (Week 6)
N05 LR Difficulty   PENDING Week 7
N10-N19             PENDING Week 8+

## NEXT SESSION — N04 CART Query Router
File: src/routing/cart_router.py
Test: tests/test_cart_router.py
What: sklearn DecisionTree — classifies query into 5 types
      numerical / ratio / multi_doc / text / forensic
      Trained on 200 labelled questions, saved with joblib
      Fires FIRST at query time — controls all downstream routing
Install needed: pip install langgraph (Week 7 — not yet)

## DAILY STARTUP
cd "D:\projects\finbench_agent"
venv\scripts\activate
pytest tests\ -q --tb=no   → must show 185/185

## EVAL COMMAND
python eval/run_eval.py --dataset financebench --seed 42

## SCORE PROGRESSION REFERENCE
Raw Llama 8B:                        ~52%
+ Context-first + Metadata chunks:   ~74%
+ BM25 hybrid search:                ~76%
+ Fine-tuned BGE-M3 + SectionTree:   ~83%
+ SniperRAG direct table extraction: ~85%
+ LLM SFT fine-tune:                 ~87%
+ DPO Cycle 1 (beta=0.1):           ~88%
+ XGB-Arbiter (Gate M6):             ~91%
+ K-Means DPO + Sprint:              ~92%
+ Post-launch RLEF 3 cycles:         ~93%
HARD CEILING Llama 8B: 94-95%