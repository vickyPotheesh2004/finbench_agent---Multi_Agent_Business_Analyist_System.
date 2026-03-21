# CONTEXT.md — FinBench Multi-Agent Business Analyst AI
# PASTE THIS ENTIRE FILE AT THE START OF EVERY NEW CLAUDE SESSION
# ═══════════════════════════════════════════════════════════════

BUILD_STEP: Week 7, Day 2
PHASE: Phase 3 — Analysis Engine (Weeks 8-11)
PHASE_GOAL: N11 Analyst Pod + PIV Loop
LAST_GATE: M1 PASSED — Week 1 | M2 PENDING
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
M2 Retrieval       PENDING      Check this week
M3 BGE-M3          PENDING      Week 6
M4 Full Pipeline   PENDING      Week 9
M5 LLM SFT         PENDING      Week 12
M6 XGB-Arbiter     PENDING      Week 14
M7 Pre-Sprint      PENDING      Week 15
M8 Launch          PENDING      Sprint End
M9 RLEF Active     PENDING      Post-Launch

## FILES_WRITTEN
src/state/ba_state.py                    ✓  C7 fix — prompt_template str not enum
src/utils/seed_manager.py                ✓
src/utils/resource_governor.py           ✓  Runtime halt fix — _halt_gb() fn
eval/run_eval.py                         ✓  150 questions
tests/test_ci_gate.py                    ✓  19 tests — PromptTemplate removed
.github/workflows/tests.yml              ✓
pytest.ini                               ✓  filterwarnings — zero warnings now
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
src/routing/lr_difficulty.py             ✓  N05 LR 150 questions
tests/test_lr_difficulty.py              ✓  24/24
src/prompts/assembler.py                 ✓  N10 5 Jinja2 templates C7
tests/test_prompt_assembler.py           ✓  24/24
DECISIONS.md                             ✓  All discussion decisions
src/live_data/[stubs]                    ✓  Phase 7A stubs
src/live_data/plugin_registry.yaml       ✓  18 APIs configured
eval/run_live_eval.py                    ✓  STUB Phase 8

## TEST RESULTS
pytest tests\ -q → 259/259 PASSED — ZERO warnings (75.65s)

## CRITICAL FIXES MADE THIS SESSION
1. ba_state.py: PromptTemplate enum REMOVED
   prompt_template is now plain str = "context_first"
   C7 validator enforces this — never change
   PromptTemplate enum import removed from test_ci_gate.py

2. resource_governor.py: Runtime halt threshold fix
   _halt_gb() function checks PYTEST_RUNNING at RUNTIME
   Previously was computed at class definition time — pytest-env
   could not set it in time
   Production: 14.0GB always | Test: 15.4GB when PYTEST_RUNNING=1

3. pytest.ini: filterwarnings added
   ChromaDB SwigPy C++ deprecation warnings suppressed
   Invalid escape sequence warnings suppressed
   Result: zero warnings in test output

## CONSTRAINTS C1-C10 (NEVER VIOLATE)
C1: $0 cost. No paid APIs.
C2: 100% local. Zero network calls during inference.
C3: Llama 3.1 8B Q4_K_M via Ollama localhost:11434.
C4: 14GB RAM hard cap (15.4GB in pytest — checked at runtime).
C5: seed=42 everywhere.
C6: DPO beta=0.1 always.
C7: prompt_template MUST be "context_first". Context BEFORE question.
C8: COMPANY/DOCTYPE/FISCAL_YEAR/SECTION/PAGE on every chunk.
C9: _rlef_ fields NEVER in output.
C10: ollama pull distribution.

## TECH_STACK_INSTALLED
pydantic, pytest, pytest-env, numpy, psutil, scipy, datasets
rich, python-dotenv, pyyaml, joblib, scikit-learn, jinja2
pdfplumber, pymupdf, python-docx, openpyxl, pandas
Pillow, pytesseract, pdf2image
bm25s, rank-bm25, chromadb, sentence-transformers
langchain==1.2.12, langchain-community
BGE model cached: BAAI/bge-small-en-v1.5 (384-dim)
CrossEncoder cached: cross-encoder/ms-marco-MiniLM-L-6-v2 (90MB)
CART model saved: models/cart_router.pkl
LR model saved:   models/lr_difficulty.pkl
Tesseract-OCR installed

## PIPELINE PROGRESS
N01 ✓  N02 ✓  N03 ✓  Ingestion
N04 ✓  N05 ✓         Routing
N06 ✓  N07 ✓  N08 ✓  N09 ✓  Retrieval cascade
N10 ✓                Prompt Assembler — 5 templates C7 enforced
N11 ← NEXT          Analyst Pod + PIV Loop

## NEXT SESSION — N11 Analyst Pod + PIV Loop
Files:
  src/agents/planner.py          StrategicPlanner — 6 curiosity questions
  src/agents/implementor.py      ContextImplementor — context-only execution
  src/agents/validator.py        CuriousValidator — 8 checks V1-V8
  src/agents/piv_loop.py         PIVLoopController — retry until PASS
  src/agents/analyst_pod.py      N11 — LeadAnalyst using PIVLoopController
  tests/test_analyst_pod.py      24 tests

Install needed: pip install langgraph (first use in Phase 3)
Ollama needed: ollama serve + ollama pull llama3.1:8b

PIV loop per Amendment A1+A2:
  REJECT → back to PLANNER (not Implementor)
  max_retries = 5
  After 5 → Clarification Engine (A3)

## DAILY STARTUP
cd "D:\projects\finbench_agent"
venv\scripts\activate
pytest tests\ -q --tb=no  → must show 259/259 zero warnings

## EVAL COMMAND
python eval/run_eval.py --dataset financebench --seed 42

## SCORE PROGRESSION
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