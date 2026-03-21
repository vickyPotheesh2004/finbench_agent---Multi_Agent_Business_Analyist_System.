# CONTEXT.md — FinBench Multi-Agent Business Analyst AI
# PASTE THIS ENTIRE FILE AT THE START OF EVERY NEW CLAUDE SESSION
# ═══════════════════════════════════════════════════════════════

BUILD_STEP: Week 9, Day 2
PHASE: Phase 3 — Analysis Engine (Weeks 8-11)
PHASE_GOAL: N14 Auditor Pod + N15 PIV Mediator
LAST_GATE: M1 PASSED | M4 PENDING — check after N15
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
M4 Full Pipeline   PENDING      After N15
M5 LLM SFT         PENDING      Week 12
M6 XGB-Arbiter     PENDING      Week 14
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
src/agents/triguard.py                   ✓  N13 Benford+IF+Risk
tests/test_triguard.py                   ✓  24/24
DECISIONS.md                             ✓
src/live_data/[stubs]                    ✓

## TEST RESULTS
pytest tests\ -q → 331/331 PASSED zero warnings (34.08s)

## PIPELINE PROGRESS
N01 ✓  N02 ✓  N03 ✓  Ingestion
N04 ✓  N05 ✓         Routing
N06 ✓  N07 ✓  N08 ✓  N09 ✓  Retrieval
N10 ✓                Prompt Assembler
N11 ✓                Analyst Pod + PIV Loop
N12 ✓                CFO/Quant Pod
N13 ✓                TriGuard Forensics
N14 ←  NEXT         Auditor Pod (BLIND)
N15                  PIV Mediator
N16-N19              Pending

## NEXT SESSION — N14 Auditor Pod (BLIND)
File: src/agents/auditor_pod.py
Test: tests/test_auditor_pod.py
What: Blind independent auditor — NEVER sees N11 or N12 output
      Completely separate PIV loop instance
      Re-retrieves independently to prevent anchoring bias
      Checks for contradictions between document sections
      Writes: auditor_output, auditor_confidence,
              auditor_citations, contradiction_flags
Key: BLIND = architecturally enforced by separate BAState fields
     Never shares state with N11 or N12
     Uses same PIVLoopController but independent context

## CONSTRAINTS C1-C10
C1: $0. C2: local. C3: Llama3.1 8B.
C4: 14GB (15.4 test). C5: seed=42.
C6: DPO beta=0.1. C7: context_first.
C8: metadata prefix. C9: no _rlef_. C10: ollama pull.

## DAILY STARTUP
cd "D:\projects\finbench_agent"
venv\scripts\activate
pytest tests\ -q --tb=no  → must show 331/331