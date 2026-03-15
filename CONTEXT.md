# CONTEXT.md — FinBench Multi-Agent Business Analyst AI
# PASTE THIS ENTIRE FILE AT THE START OF EVERY NEW CLAUDE SESSION
# ═══════════════════════════════════════════════════════════════

BUILD_STEP: Week 3, Day 2
PHASE: Phase 2 — Retrieval (Weeks 3-7)
PHASE_GOAL: N04-N09 retrieval cascade complete
LAST_GATE: M1 PASSED — Week 1
THIS_SESSION_TASK: [REPLACE EACH SESSION — one sentence only]
PROJECT_GOAL: FinanceBench >=82% launch → 91-93% full stack
$0 cost forever | 100% local | Self-improving via RLEF/DPO

## !! PROJECT FOLDER !!
PROJECT FOLDER : D:\projects\finbench_agent
VENV ACTIVATE  : cd "D:\projects\finbench_agent" then venv/scripts/activate
CORRECT PROMPT : (venv) PS D:\projects\finbench_agent>

## AMENDMENTS (upgrades to original PDR-BAAAI-001 spec)
A1: PIV REJECT → goes back to PLANNER (not Implementor)
A2: max_retries = 5 per pod (not 3)
A3: After 5 failures → Clarification Engine fires (5 questions + free text → loop resets)
A4: Target = Top 1 open-source, >=93% full stack

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
.gitignore                               ✓
src/*/__init__.py                        ✓  16 files
CONTEXT.md                               ✓
src/state/ba_state.py                    ✓  Pydantic v2, 50+ fields, C4 C5 C8 C9 A1-A4
src/utils/seed_manager.py                ✓  C5 enforced
src/utils/resource_governor.py           ✓  C4 enforced
eval/run_eval.py                         ✓  150 questions, results.json
tests/test_ci_gate.py                    ✓  17/17 PASSED
.github/workflows/tests.yml              ✓  GitHub Actions CI
src/ingestion/pdf_ingestor.py            ✓  N01 — PDF/DOCX/CSV/XLSX/PNG/JPG + OCR
tests/test_pdf_ingestor.py               ✓  24/24 PASSED
src/ingestion/section_tree_builder.py    ✓  N02 — hierarchical section map
tests/test_section_tree.py               ✓  24/24 PASSED
src/ingestion/chunker.py                 ✓  N03 — section-boundary chunks, BM25+ChromaDB
tests/test_chunker.py                    ✓  24/24 PASSED
src/retrieval/sniper_rag.py              ✓  N06 — 22 regex patterns, FY prefix, ~50ms
tests/test_sniper_rag.py                 ✓  24/24 PASSED

## TEST RESULTS
pytest tests/ -q → 113/113 PASSED
python eval/run_eval.py --dataset financebench --seed 42 → 0.0% stub correct

## KNOWN_ISSUES
RAM running at 13.25GB during tests — close unused apps before sessions.
Still safe — below 14GB hard cap. ResourceGovernor ALERT is correct behaviour.

## CONSTRAINTS C1-C10 (ACTIVE — NEVER VIOLATE)
C1: $0 cost permanently. No paid APIs ever.
C2: 100% local. Documents never leave the machine.
C3: Llama 3.1 8B Q4_K_M via Ollama at localhost:11434.
C4: 14GB RAM hard cap. warn@12GB, alert@13GB, halt@14GB.
C5: seed=42 everywhere — SeedManager wraps all calls.
C6: DPO beta=0.1 always. Never >0.15, never <0.05.
C7: Context-first prompts. Document text BEFORE question, 100%.
C8: Mandatory metadata: COMPANY/DOCTYPE/FISCAL_YEAR/SECTION/PAGE.
C9: _rlef_ fields NEVER in DOCX, Streamlit, or logs.
C10: Distribution via ollama pull YOUR_USERNAME/financebench-expert

## TECH_STACK_INSTALLED
pydantic==2.7.1, pytest==8.2.0, pytest-cov
numpy, psutil==5.9.8, scipy, datasets
rich==13.7.1, python-dotenv==1.0.1, pyyaml
pdfplumber, pymupdf, python-docx
openpyxl, pandas, Pillow, pytesseract, pdf2image
bm25s, chromadb, sentence-transformers
Tesseract-OCR: C:\Program Files\Tesseract-OCR\tesseract.exe
NOTE: langchain + langgraph install at Week 4 (N07 session)

## PIPELINE PROGRESS
N01 PDF Ingestor          ✓ DONE — pdf_ingestor.py
N02 Section Tree Builder  ✓ DONE — section_tree_builder.py
N03 Chunker + Indexer     ✓ DONE — chunker.py
N06 SniperRAG             ✓ DONE — sniper_rag.py (22 patterns, FY prefix)
N07 BM25 Retriever        ← NEXT (Week 4 — installs langchain)
N08 BGE-M3 Semantic       PENDING Week 6
N09 RRF + Reranker        PENDING Week 7
N04 CART Router           PENDING Week 7
N05 LR Difficulty         PENDING Week 7
N10-N19                   PENDING Week 8+

## N06 SNIPER RAG — KEY FACTS
- 22 financial metric patterns (income statement + balance sheet + cash flow)
- Fires only for NUMERICAL and RATIO query types
- Confidence threshold: 0.95 (below this → cascade to N07)
- Year extraction handles: 2023, FY2023, FY 2023, fiscal year 2023
- Speed: ~50ms, zero GPU, zero LLM call
- Handles ~40% of FinanceBench numerical questions directly

## NEXT SESSION — N07 BM25 Retriever
Install: pip install langchain langchain-community
File: src/retrieval/bm25_retriever.py
Test: tests/test_bm25.py

## DAILY STARTUP — RUN THESE FIRST EVERY SESSION
cd "D:\projects\finbench_agent"
venv/scripts/activate
You should see: (venv) PS D:\projects\finbench_agent>

## SCORE PROGRESSION (reference — never post projections)
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

## EVAL COMMAND (THE ONLY SCORE THAT MATTERS)
python eval/run_eval.py --dataset financebench --seed 42