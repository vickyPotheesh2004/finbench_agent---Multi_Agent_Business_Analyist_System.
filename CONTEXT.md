\# CONTEXT.md — FinBench Multi-Agent Business Analyst AI

\# PASTE THIS ENTIRE FILE AT THE START OF EVERY NEW CLAUDE SESSION

\# ═══════════════════════════════════════════════════════════════



BUILD\_STEP: Week 1, Day 1

PHASE: Phase 1 — Foundation (Weeks 1-3)

PHASE\_GOAL: BA\_State + run\_eval.py + CI/CD + N01-N03 all passing

LAST\_GATE: None — starting fresh

THIS\_SESSION\_TASK: Build BAState schema + eval stub + CI gate tests

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

M1 Schema+Eval     IN PROGRESS  Week 1

M2 Retrieval       PENDING      Week 7

M3 BGE-M3          PENDING      Week 6

M4 Full Pipeline   PENDING      Week 9

M5 LLM SFT         PENDING      Week 12

M6 XGB-Arbiter     PENDING      Week 14

M7 Pre-Sprint      PENDING      Week 15

M8 Launch          PENDING      Sprint End

M9 RLEF Active     PENDING      Post-Launch



\## FILES\_WRITTEN

.gitignore                      ✓

src/\*/\_\_init\_\_.py               ✓  all 16 files

tests/\_\_init\_\_.py               ✓

CONTEXT.md                      ✓  (this file)



\## KNOWN\_ISSUES

None — clean start.



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

pydantic==2.7.1

pytest==8.2.0

pytest-cov

numpy

psutil==5.9.8

scipy

datasets

rich==13.7.1

python-dotenv==1.0.1

pyyaml



\## NEXT FILES TO BUILD (in order)

1\. src/state/ba\_state.py          ← NEXT

2\. src/utils/seed\_manager.py

3\. src/utils/resource\_governor.py

4\. eval/run\_eval.py

5\. tests/test\_ci\_gate.py

6\. .github/workflows/tests.yml



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

