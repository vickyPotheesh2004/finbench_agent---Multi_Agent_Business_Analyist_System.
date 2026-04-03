# CONTEXT.md — FinBench Multi-Agent Business Analyst AI
# PASTE THIS ENTIRE FILE AT THE START OF EVERY NEW CLAUDE SESSION
# ═══════════════════════════════════════════════════════════════

BUILD_STEP: Week 12, Day 1
PHASE: Phase 3 FINISH — Pipeline Wire-up → Gate M4
LAST_GATE: M1 PASSED | M4 PENDING THIS SESSION
THIS_SESSION_TASK: [REPLACE EACH SESSION — one sentence only]
PROJECT_GOAL: FinanceBench >=90% launch | $0 | 100% local | RLEF

## !! PROJECT FOLDER !!
PROJECT FOLDER : D:\projects\finbench_agent
VENV ACTIVATE  : cd "D:\projects\finbench_agent" then venv\scripts\activate
CORRECT PROMPT : (venv) PS D:\projects\finbench_agent>

## AMENDMENTS
A1-A6:  Original amendments ✓
A7:  FinanceBench target >=90% at launch
A8:  BizBench target >=80% at launch
A9:  Live Financial QA benchmark pre-launch
A10: SFT pairs raised to 2000
A11: Pre-sprint sessions raised to 10,000
A12: DPO rounds raised to 3 pre-launch
A13: BGE-M3 Gate M3 MRR>=0.90
A14: LLM Gate M5 >=80% held-out
A15: XGB Gate M6 >=3% improvement
A16: All Phase 7 live data pre-launch
A17: K-Means clusters raised to 6

## GATE_STATUS
M1 Schema+Eval     PASSED  Week 1 ✓
M2 Retrieval       PENDING Week 7
M3 BGE-M3          PENDING Phase 4
M4 Full Pipeline   PENDING THIS SESSION ←
M5 LLM SFT         PENDING Phase 4
M6 XGB-Arbiter     PENDING Phase 4
M7 Pre-Sprint      PENDING Phase 5
M8 Launch          PENDING Phase 6
M9 RLEF Active     PENDING Post-Launch

## ALL 19 NODES — COMPLETE ✓
N01 ✓  PDF Ingestor
N02 ✓  Section Tree Builder
N03 ✓  Chunker + Indexer
N04 ✓  CART Router
N05 ✓  LR Difficulty
N06 ✓  SniperRAG
N07 ✓  BM25 Retriever
N08 ✓  BGE-M3 Retrieval
N09 ✓  RRF + Reranker
N10 ✓  Prompt Assembler
N11 ✓  Analyst Pod
N12 ✓  CFO/Quant Pod
N13 ✓  TriGuard Forensics
N14 ✓  Auditor Pod BLIND
N15 ✓  PIV Mediator
N16 ✓  SHAP + Causal DAG
N17 ✓  XGBoost Arbiter
N18 ✓  RLEF JEE Engine
N19 ✓  Output Generator

## TEST RESULTS
pytest tests\ -q → 475/475 PASSED zero warnings (111s)

## NEXT SESSION — Pipeline Wire-up → Gate M4
File: src/pipeline/pipeline.py
Test: tests/test_pipeline.py
What: LangGraph StateGraph connecting all 19 nodes
  Single .invoke() call runs entire pipeline
  Parallel branches: N11+N12+N13+N14 run in parallel
  Conditional edges: SniperRAG short-circuit
  MemorySaver: checkpoint on RAM halt
  Gate M4: 100% of 10 test questions answered
           All 19 nodes fire
           iteration_count never > 5
           DOCX report generated

## PHASE SEQUENCE AFTER GATE M4
Phase 7A: Cache + 16 API fetchers
Phase 7B: MCP servers
Phase 7C: DataShield full
Phase 7D: Pipeline integration N00.5
Streamlit UI
Phase 4: BGE-M3 + SFT + DPO (Colab/Kaggle)
Phase 5: Proof — FB>=90% BB>=80%
Phase 8: Live benchmark build
Phase 6: Launch

## CONSTRAINTS C1-C10
C1:$0 C2:local C3:Llama3.1 8B
C4:14GB(15.4 test) C5:seed=42
C6:DPO beta=0.1 C7:context_first
C8:metadata prefix C9:no _rlef_ C10:ollama pull

## DAILY STARTUP
cd "D:\projects\finbench_agent"
venv\scripts\activate
pytest tests\ -q --tb=no → must show 475/475