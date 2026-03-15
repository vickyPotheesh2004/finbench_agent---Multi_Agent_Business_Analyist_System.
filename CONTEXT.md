<<<<<<< HEAD
\# CONTEXT.md — FinBench Multi-Agent Business Analyst AI

\# PASTE THIS ENTIRE FILE AT THE START OF EVERY NEW CLAUDE SESSION

\# ═══════════════════════════════════════════════════════════════



BUILD\_STEP: Week 1, Day 2

PHASE: Phase 1 — Foundation (Weeks 1-3)

PHASE\_GOAL: BA\_State + run\_eval.py + CI/CD + N01-N03 all passing

LAST\_GATE: M1 PASSED — Week 1

THIS\_SESSION\_TASK: \[REPLACE EACH SESSION — one sentence only]

PROJECT\_GOAL: FinanceBench >=82% launch → 91-93% full stack

$0 cost forever | 100% local | Self-improving via RLEF/DPO



\## !! PROJECT FOLDER !!

PROJECT FOLDER : D:\\projects\\finbench\_agent

VENV ACTIVATE  : cd "D:\\projects\\finbench\_agent" then venv/scripts/activate

CORRECT PROMPT : (venv) PS D:\\projects\\finbench\_agent>



\## AMENDMENTS (upgrades to original PDR-BAAAI-001 spec)

A1: PIV REJECT → goes back to PLANNER (not Implementor)

A2: max\_retries = 5 per pod (not 3)

A3: After 5 failures → Clarification Engine fires (5 questions + free text → loop resets)

A4: Target = Top 1 open-source, >=93% full stack



\## GATE\_STATUS

M1 Schema+Eval     PASSED       Week 1 ✓

M2 Retrieval       PENDING      Week 7

M3 BGE-M3          PENDING      Week 6

M4 Full Pipeline   PENDING      Week 9

M5 LLM SFT         PENDING      Week 12

M6 XGB-Arbiter     PENDING      Week 14

M7 Pre-Sprint      PENDING      Week 15

M8 Launch          PENDING      Sprint End

M9 RLEF Active     PENDING      Post-Launch



\## FILES\_WRITTEN

.gitignore                           ✓

src/\_\_init\_\_.py + all subfolders     ✓  16 files

tests/\_\_init\_\_.py                    ✓

CONTEXT.md                           ✓

src/state/ba\_state.py                ✓  Pydantic v2, 50+ fields, C4 C5 C8 C9 A1 A2 A3

src/utils/seed\_manager.py            ✓  C5 enforced, wraps all random calls

src/utils/resource\_governor.py       ✓  C4 enforced, warn/alert/halt

eval/run\_eval.py                     ✓  150 questions loaded, 0% stub, results.json

tests/test\_ci\_gate.py                ✓  17/17 PASSED

.github/workflows/tests.yml         ✓  GitHub Actions CI



\## TEST RESULTS

pytest tests/ -v → 17/17 PASSED

python eval/run\_eval.py --dataset financebench --seed 42 → 0.0% (stub correct)



\## KNOWN\_ISSUES

None — all clean.



\## CONSTRAINTS C1-C10 (ACTIVE — NEVER VIOLATE)

C1: $0 cost permanently. No paid APIs ever.

C2: 100% local. Documents never leave the machine.

C3: Llama 3.1 8B Q4\_K\_M via Ollama at localhost:11434.

C4: 14GB RAM hard cap. warn@12GB, alert@13GB, halt@14GB.

C5: seed=42 everywhere — SeedManager wraps all calls.

C6: DPO beta=0.1 always. Never >0.15, never <0.05.

C7: Context-first prompts. Document text BEFORE question, 100%.

C8: Mandatory metadata: COMPANY/DOCTYPE/FISCAL\_YEAR/SECTION/PAGE.

C9: \_rlef\_ fields NEVER in DOCX, Streamlit, or logs.

C10: Distribution via ollama pull YOUR\_USERNAME/financebench-expert



\## TECH\_STACK\_INSTALLED

pydantic==2.7.1, pytest==8.2.0, pytest-cov

numpy, psutil==5.9.8, scipy, datasets

rich==13.7.1, python-dotenv==1.0.1, pyyaml



\## NEXT FILES TO BUILD — Week 2

1\. src/ingestion/pdf\_ingestor.py       ← N01 (next session)

2\. tests/test\_pdf\_ingestor.py

3\. src/ingestion/section\_tree\_builder.py  ← N02

4\. tests/test\_section\_tree.py

5\. src/ingestion/chunker.py            ← N03

6\. tests/test\_chunker.py



\## DAILY STARTUP — RUN THESE FIRST EVERY SESSION

cd "D:\\projects\\finbench\_agent"

venv/scripts/activate

You should see: (venv) PS D:\\projects\\finbench\_agent>



\## SCORE PROGRESSION (reference — never post projections)

Raw Llama 8B:                        \~52%

\+ Context-first + Metadata chunks:   \~74%

\+ BM25 hybrid search:                \~76%

\+ Fine-tuned BGE-M3 + SectionTree:   \~83%

\+ SniperRAG direct table extraction: \~85%

\+ LLM SFT fine-tune:                 \~87%

\+ DPO Cycle 1 (beta=0.1):           \~88%

\+ XGB-Arbiter (Gate M6):             \~91%

\+ K-Means DPO + Sprint:              \~92%

\+ Post-launch RLEF 3 cycles:         \~93%

HARD CEILING Llama 8B: 94-95%



\## EVAL COMMAND (THE ONLY SCORE THAT MATTERS)

python eval/run\_eval.py --dataset financebench --seed 42

=======
# CONTEXT.md — FinBench Multi-Agent Business Analyst AI
# PASTE THIS ENTIRE FILE AT THE START OF EVERY NEW CLAUDE SESSION
# ═══════════════════════════════════════════════════════════════

BUILD_STEP: Week 1, Day 2
PHASE: Phase 1 — Foundation (Weeks 1-3)
PHASE_GOAL: BA_State + run_eval.py + CI/CD + N01-N03 all passing
LAST_GATE: None — M1 in progress
THIS_SESSION_TASK: [REPLACE EACH SESSION — one sentence only]
PROJECT_GOAL: FinanceBench >=82% launch → 91-93% full stack
$0 cost forever | 100% local | Self-improving via RLEF/DPO

## !! IMPORTANT — CORRECT FOLDER PATH !!
PROJECT FOLDER : D:\projects\finbench_agent
VENV ACTIVATE  : cd "D:\projects\finbench_agent" then .\venv\Scripts\Activate.ps1
CORRECT PROMPT : (venv) PS D:\projects\finbench_agent>
NOTE: Build guide says "finbench_agent System" but working folder is "finbench_agent"
      All code and tests live here. Do NOT use "finbench_agent System".

## AMENDMENTS (upgrades to PDR-BAAAI-001 original spec)
A1: PIV REJECT → goes back to PLANNER (not Implementor)
A2: max_retries = 5 per pod (not 3)
A3: After 5 failures → Clarification Engine fires (5 questions + free text → loop resets)
A4: Target = Top 1 open-source, >=93% full stack

## GATE_STATUS
M1 Schema+Eval     IN PROGRESS  Week 1
M2 Retrieval       PENDING      Week 7
M3 BGE-M3          PENDING      Week 6
M4 Full Pipeline   PENDING      Week 9
M5 LLM SFT         PENDING      Week 12
M6 XGB-Arbiter     PENDING      Week 14
M7 Pre-Sprint      PENDING      Week 15
M8 Launch          PENDING      Sprint End
M9 RLEF Active     PENDING      Post-Launch

## FILES_WRITTEN
ba_state.py                     ✓  Pydantic v2 BAState, 50+ fields, all constraints + A1-A4
tests/test_ba_state.py          ✓  11 tests — ALL PASSING
CONTEXT.md                      ✓  Session memory

## KNOWN_ISSUES
None — all 11 tests passing cleanly.

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
numpy, pydantic>=2.0, pytest, pytest-cov, scipy
datasets, psutil, rich, python-dotenv, pyyaml
pandas, pyarrow (installed as dataset dependencies)

## FOLDER STRUCTURE (inside D:\projects\finbench_agent)
ba_state.py              ← BAState lives at ROOT for now
tests\test_ba_state.py   ← 11 tests passing
src\state\               ← future home of ba_state.py (Week 2 refactor)
src\utils\               ← SeedManager + ResourceGovernor (next session)
src\ingestion\           ← N01, N02, N03 (Week 3)
src\routing\             ← N04, N05 (Week 7)
src\retrieval\           ← N06-N09 (Week 4-7)
src\agents\              ← PIV loop pods (Week 9)
eval\                    ← run_eval.py (next session)
tests\                   ← all test files

## NEXT SESSION TASK — Week 1 Day 2
Build: src/utils/seed_manager.py + src/utils/resource_governor.py
Then: eval/run_eval.py (FinanceBench eval stub)
Then: tests/test_ci_gate.py (12 CI gate tests)

## DAILY STARTUP — RUN THESE FIRST EVERY SESSION
cd "D:\projects\finbench_agent"
.\venv\Scripts\Activate.ps1
You should see: (venv) PS D:\projects\finbench_agent>

## SCORE_PROGRESSION (reference — never post projections)
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
python eval\run_eval.py --dataset financebench --seed 42
>>>>>>> fa813f75657d2aa83a91648ceb56cac8da2ebd2b
