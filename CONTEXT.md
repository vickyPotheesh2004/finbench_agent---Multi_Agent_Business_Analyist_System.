# CONTEXT.md — FinBench Multi-Agent Business Analyst AI
# PASTE THIS ENTIRE FILE AT THE START OF EVERY NEW CLAUDE SESSION
# ═══════════════════════════════════════════════════════════════

BUILD_STEP: Week 6, Day 2
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
src/utils/resource_governor.py           ✓  PYTEST_RUNNING env var
eval/run_eval.py                         ✓  150 questions
tests/test_ci_gate.py                    ✓  17/17
.github/workflows/tests.yml              ✓
pytest.ini                               ✓  pytest-env PYTEST_RUNNING=1
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
src/routing/cart_router.py               ✓  N04 CART 200 questions
tests/test_cart_router.py                ✓  24/24
DECISIONS.md                             ✓  All discussion decisions
src/live_data/[stubs]                    ✓  Phase 7A stubs
src/live_data/plugin_registry.yaml       ✓  18 APIs configured
eval/run_live_eval.py                    ✓  STUB Phase 8

## TEST RESULTS
pytest tests\ -q → 209/209 PASSED (74.02s)

## RAM FIX — CRITICAL
PYTEST_RUNNING=1 in pytest.ini → ResourceGovernor uses 15.4GB in tests
Production always uses 14GB hard cap — C4 not violated
Requires: pytest-env (installed)

## CONSTRAINTS C1-C10 (NEVER VIOLATE)
C1: $0 cost. No paid APIs.
C2: 100% local. Zero network calls during inference.
C3: Llama 3.1 8B Q4_K_M via Ollama localhost:11434.
C4: 14GB RAM hard cap (15.4GB in pytest only).
C5: seed=42 everywhere.
C6: DPO beta=0.1 always.
C7: Context BEFORE question in all prompts.
C8: COMPANY/DOCTYPE/FISCAL_YEAR/SECTION/PAGE on every chunk.
C9: _rlef_ fields NEVER in output.
C10: ollama pull distribution.

## TECH_STACK_INSTALLED
pydantic, pytest, pytest-env, numpy, psutil, scipy, datasets
rich, python-dotenv, pyyaml, joblib, scikit-learn
pdfplumber, pymupdf, python-docx, openpyxl, pandas
Pillow, pytesseract, pdf2image
bm25s, rank-bm25, chromadb, sentence-transformers
langchain==1.2.12, langchain-community
BGE model cached: BAAI/bge-small-en-v1.5 (384-dim)
CrossEncoder cached: cross-encoder/ms-marco-MiniLM-L-6-v2 (90MB)
CART model saved: models/cart_router.pkl
Tesseract-OCR installed

## PIPELINE PROGRESS
N01 ✓  N02 ✓  N03 ✓
N04 ✓  CART Router — 5 classes, 200 questions, 100% accuracy
N06 ✓  N07 ✓  N08 ✓  N09 ✓
N05 LR Difficulty   ← NEXT SESSION
N10-N19             PENDING Week 8+

## ROUTING BUILT — N04 complete
numerical  → SniperRAG first → BM25 → BGE → RRF
ratio      → BM25 → BGE → RRF → Quant pod
multi_doc  → BM25 → BGE → RRF → wide context (5 chunks)
text       → BGE lead → BM25 → RRF → narrative
forensic   → BM25 → BGE → RRF → TriGuard activated

## NEXT SESSION — N05 LR Difficulty Predictor
File: src/routing/lr_difficulty.py
Test: tests/test_lr_difficulty.py
What: sklearn LogisticRegression — classifies query into 3 difficulty levels
      easy / medium / hard
      hard → wider context + 2 extra PIV rounds + lower HITL threshold
      Trained on 150 labelled questions (50 per class)
      Saved with joblib alongside cart_router.pkl

## DAILY STARTUP
cd "D:\projects\finbench_agent"
venv\scripts\activate
pytest tests\ -q --tb=no  → must show 209/209

## EVAL COMMAND
python eval/run_eval.py --dataset financebench --seed 42