# CONTEXT.md — FinBench Multi-Agent Business Analyst AI
# PASTE THIS ENTIRE FILE AT THE START OF EVERY NEW CLAUDE SESSION
# ═══════════════════════════════════════════════════════════════

BUILD_STEP: Week 8, Day 2
PHASE: Phase 3 — Analysis Engine (Weeks 8-11)
PHASE_GOAL: N12 CFO/Quant Pod + N13 TriGuard + N14 Auditor Pod
LAST_GATE: M1 PASSED — Week 1 | M2 PENDING | M4 PENDING
THIS_SESSION_TASK: [REPLACE EACH SESSION — one sentence only]
PROJECT_GOAL: FinanceBench >=82% launch → 91-93% full stack
$0 cost forever | 100% local | Self-improving via RLEF/DPO

## !! PROJECT FOLDER !!
PROJECT FOLDER : D:\projects\finbench_agent
VENV ACTIVATE  : cd "D:\projects\finbench_agent" then venv\scripts\activate
CORRECT PROMPT : (venv) PS D:\projects\finbench_agent>

## AMENDMENTS
A1: PIV REJECT → goes back to PLANNER (not Implementor) ✓ IMPLEMENTED
A2: max_retries = 5 per pod ✓ IMPLEMENTED
A3: After 5 failures → low_confidence=True → Clarification Engine ✓ IMPLEMENTED
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
src/state/ba_state.py                    ✓  C7 fix
src/utils/seed_manager.py                ✓
src/utils/resource_governor.py           ✓  Runtime halt fix
eval/run_eval.py                         ✓
tests/test_ci_gate.py                    ✓  19 tests
.github/workflows/tests.yml              ✓
pytest.ini                               ✓  Zero warnings
src/ingestion/pdf_ingestor.py            ✓  N01
tests/test_pdf_ingestor.py               ✓  24/24
src/ingestion/section_tree_builder.py    ✓  N02
tests/test_section_tree.py               ✓  24/24
src/ingestion/chunker.py                 ✓  N03
tests/test_chunker.py                    ✓  24/24
src/retrieval/sniper_rag.py              ✓  N06
tests/test_sniper_rag.py                 ✓  24/24
src/retrieval/bm25_retriever.py          ✓  N07
tests/test_bm25.py                       ✓  24/24
src/retrieval/bge_retriever.py           ✓  N08
tests/test_bge_m3.py                     ✓  24/24
src/retrieval/rrf_reranker.py            ✓  N09
tests/test_rrf.py                        ✓  24/24
src/routing/cart_router.py               ✓  N04
tests/test_cart_router.py                ✓  24/24
src/routing/lr_difficulty.py             ✓  N05
tests/test_lr_difficulty.py              ✓  24/24
src/prompts/assembler.py                 ✓  N10 5 templates C7
tests/test_prompt_assembler.py           ✓  24/24
src/agents/planner.py                    ✓  StrategicPlanner 6Q
src/agents/implementor.py                ✓  ContextImplementor decay
src/agents/validator.py                  ✓  CuriousValidator 8 checks
src/agents/piv_loop.py                   ✓  PIVLoopController A1+A2+A3
src/agents/analyst_pod.py                ✓  N11 LeadAnalyst
tests/test_analyst_pod.py                ✓  24/24
DECISIONS.md                             ✓
src/live_data/[stubs]                    ✓  Phase 7A
src/live_data/plugin_registry.yaml       ✓

## TEST RESULTS
pytest tests\ -q → 283/283 PASSED — zero warnings (49.02s)

## PIV LOOP — FULLY IMPLEMENTED
Planner    → 6 curiosity questions (Q1-Q6) + emotional identity
Implementor→ context-only + confidence decay (1.0→0.95→0.85→0.70→0.60)
Validator  → 8 checks (V1_SCOPE→V8_GROUNDING) + emotional escalation
PIVLoop    → A1 REJECT→PLANNER | A2 max=5 | A3 low_confidence=True
All tests  → mocked Ollama — fast, deterministic, no RAM needed

## CONSTRAINTS C1-C10
C1: $0. C2: local. C3: Llama3.1 8B. C4: 14GB (15.4 test).
C5: seed=42. C6: DPO beta=0.1. C7: context_first always.
C8: metadata prefix. C9: no _rlef_ output. C10: ollama pull.

## TECH_STACK_INSTALLED
pydantic, pytest, pytest-env, numpy, psutil, scipy, datasets
rich, python-dotenv, pyyaml, joblib, scikit-learn, jinja2
pdfplumber, pymupdf, python-docx, openpyxl, pandas
Pillow, pytesseract, pdf2image
bm25s, rank-bm25, chromadb, sentence-transformers, langgraph
langchain, langchain-community, requests
BGE cached, CrossEncoder cached, CART+LR models saved
Ollama: llama3.1:8b (4.9GB) — serve with: ollama serve

## PIPELINE PROGRESS
N01 ✓  N02 ✓  N03 ✓  Ingestion
N04 ✓  N05 ✓         Routing
N06 ✓  N07 ✓  N08 ✓  N09 ✓  Retrieval
N10 ✓                Prompt Assembler
N11 ✓                Analyst Pod + PIV Loop
N12 ←  NEXT         CFO/Quant Pod
N13                  TriGuard Forensics
N14                  Auditor Pod (Blind)
N15-N19              Pending

## NEXT SESSION — N12 CFO/Quant Pod
Files:
  src/agents/quant_pod.py     CFO/Quant Pod — formula + Monte Carlo
  tests/test_quant_pod.py     24 tests

N12 specialisation vs N11:
  Reuses same PIVLoopController (planner+impl+validator)
  Different system role: formula-first quantitative specialist
  Adds: Monte Carlo 10k scenarios, Historical VaR, GARCH(1,1)
  Writes to: quant_result, quant_confidence, quant_citations
  Uses: numpy + scipy (already installed)

## DAILY STARTUP
cd "D:\projects\finbench_agent"
venv\scripts\activate
pytest tests\ -q --tb=no  → 283/283

## EVAL COMMAND
python eval/run_eval.py --dataset financebench --seed 42