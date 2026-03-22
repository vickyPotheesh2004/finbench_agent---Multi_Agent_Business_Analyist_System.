# CONTEXT.md — FinBench Multi-Agent Business Analyst AI
# PASTE THIS ENTIRE FILE AT THE START OF EVERY NEW CLAUDE SESSION
# ═══════════════════════════════════════════════════════════════

BUILD_STEP: Week 11, Day 1
PHASE: Phase 3 — Analysis Engine (Weeks 8-11)
PHASE_GOAL: N18 RLEF JEE Engine + N19 Output Generator
LAST_GATE: M1 PASSED | M4 PENDING — needs N18+N19 + pipeline wire-up
THIS_SESSION_TASK: [REPLACE EACH SESSION — one sentence only]
PROJECT_GOAL: FinanceBench >=82% launch → 91-93% full stack
$0 cost forever | 100% local | Self-improving via RLEF/DPO

## !! PROJECT FOLDER !!
PROJECT FOLDER : D:\projects\finbench_agent
VENV ACTIVATE  : cd "D:\projects\finbench_agent" then venv\scripts\activate
CORRECT PROMPT : (venv) PS D:\projects\finbench_agent>

## AMENDMENTS
A1: PIV REJECT → goes back to PLANNER ✓
A2: max_retries = 5 per pod ✓
A3: After 5 failures → low_confidence=True ✓
A4: Target = Top 1 open-source, >=93% full stack
A5: Phase 7 live data layer (post-launch)
A6: Phase 8 live benchmark + Papers With Code

## GATE_STATUS
M1 Schema+Eval     PASSED       Week 1 ✓
M2 Retrieval       PENDING      Week 7
M3 BGE-M3          PENDING      Week 6
M4 Full Pipeline   PENDING      After N18+N19 + pipeline
M5 LLM SFT         PENDING      Week 12
M6 XGB-Arbiter     PENDING      Week 14 — needs >=300 DPO pairs
M7 Pre-Sprint      PENDING      Week 15
M8 Launch          PENDING      Sprint End
M9 RLEF Active     PENDING      Post-Launch

## FILES_WRITTEN
src/state/ba_state.py                    ✓  All fields complete
src/utils/seed_manager.py                ✓
src/utils/resource_governor.py           ✓
eval/run_eval.py                         ✓
tests/test_ci_gate.py                    ✓  19 tests
pytest.ini                               ✓  Zero warnings
src/ingestion/pdf_ingestor.py            ✓  N01
tests/test_pdf_ingestor.py               ✓  24/24
src/ingestion/section_tree_builder.py    ✓  N02
src/ingestion/chunker.py                 ✓  N03
src/retrieval/sniper_rag.py              ✓  N06
src/retrieval/bm25_retriever.py          ✓  N07
src/retrieval/bge_retriever.py           ✓  N08
src/retrieval/rrf_reranker.py            ✓  N09
src/routing/cart_router.py               ✓  N04
src/routing/lr_difficulty.py             ✓  N05
src/prompts/assembler.py                 ✓  N10
src/agents/planner.py                    ✓  StrategicPlanner
src/agents/implementor.py                ✓  ContextImplementor
src/agents/validator.py                  ✓  CuriousValidator
src/agents/piv_loop.py                   ✓  PIVLoopController
src/agents/analyst_pod.py                ✓  N11
tests/test_analyst_pod.py                ✓  24/24
src/agents/quant_pod.py                  ✓  N12 MC+VaR+GARCH
tests/test_quant_pod.py                  ✓  24/24
src/agents/triguard.py                   ✓  N13 Benford+IF
tests/test_triguard.py                   ✓  24/24
src/agents/auditor_pod.py                ✓  N14 BLIND+contradiction
tests/test_auditor_pod.py                ✓  24/24
src/agents/piv_mediator.py               ✓  N15 unanimous/majority/disagree
tests/test_piv_mediator.py               ✓  24/24
src/explainability/shap_dag.py           ✓  N16 SHAP+CausalDAG
tests/test_shap_dag.py                   ✓  24/24
src/ml/xgb_arbiter.py                    ✓  N17 stub+active+Gate M6
tests/test_xgb_arbiter.py               ✓  24/24
DECISIONS.md                             ✓
models/xgb_arbiter.pkl                   ✓  Trained model saved
src/live_data/[stubs]                    ✓

## TEST RESULTS
pytest tests\ -q → 427/427 PASSED zero warnings (113s)
NOTE: 113s due to XGBoost training in trained_arbiter fixture

## PIPELINE PROGRESS
N01 ✓  N02 ✓  N03 ✓  Ingestion
N04 ✓  N05 ✓         Routing
N06 ✓  N07 ✓  N08 ✓  N09 ✓  Retrieval
N10 ✓                Prompt Assembler
N11 ✓                Analyst Pod + PIV Loop
N12 ✓                CFO/Quant Pod
N13 ✓                TriGuard Forensics
N14 ✓                Auditor Pod BLIND
N15 ✓                PIV Mediator
N16 ✓                SHAP + Causal DAG
N17 ✓                XGBoost Arbiter
N18 ←  NEXT         RLEF JEE Engine
N19                  Output Generator
Pipeline ←  AFTER   Full 19-node wire-up → Gate M4

## NEXT SESSION — N18 RLEF JEE Engine
File: src/rlef/jee_engine.py
Test: tests/test_jee_engine.py
What: 3-validator grader + SQLite DPO pair storage
  3 validators:
    V-A: Numerical precision (+4 if exact, +2 if close, -1 if wrong)
    V-B: Citation quality  (+4 if cited, +2 if partial, -1 if none)
    V-C: Completeness      (+4 if complete, +2 if partial, -1 if missing)
  Final grade: sum of V-A + V-B + V-C scores
  Stores chosen/rejected DPO pairs in SQLite
  All _rlef_ fields PRIVATE — never in output (C9)
  DB: data/rlef_training.db
  Writes to BAState: _rlef_grade, _rlef_chosen, _rlef_rejected
    (via get_rlef_fields() only — never directly)

## CONSTRAINTS C1-C10
C1: $0. C2: local. C3: Llama3.1 8B.
C4: 14GB (15.4 test). C5: seed=42.
C6: DPO beta=0.1. C7: context_first.
C8: metadata prefix. C9: no _rlef_. C10: ollama pull.

## DAILY STARTUP
cd "D:\projects\finbench_agent"
venv\scripts\activate
pytest tests\ -q --tb=no  → must show 427/427