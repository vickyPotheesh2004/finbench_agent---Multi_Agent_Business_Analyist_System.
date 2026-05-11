"""Fix 2 — pipeline early-exit verification."""
import time
from src.pipeline.pipeline import FinBenchPipeline

print("Loading pipeline + ingesting Apple 10-K...")
t0 = time.time()
pipeline = FinBenchPipeline()
state = pipeline.ingest(
    document_path="documents/sec_filings/AAPL_FY2023_10-K.html",
    session_id="fix2-test",
    company_name="Apple Inc.",
    doc_type="10-K",
    fiscal_year="FY2023",
)
ingest_time = time.time() - t0
print(f"Ingest: {ingest_time:.1f}s, {state.chunk_count} chunks, {len(state.table_cells)} cells\n")

QUESTIONS = [
    ("Sniper hit (revenue)",   "What was Apple total revenue in FY2023?"),
    ("Sniper hit (NI)",        "What was Apple net income in FY2023?"),
    ("Sniper hit (OCF)",       "What was Apple operating cash flow in FY2023?"),
    ("Garbage (0 chunks)",     "asdfqwerty xyzzy nonsense unrelated"),
    ("Narrative (LLM dead)",   "What is Apple's strategy for emerging markets?"),
    ("Sniper hit (cash)",      "How much cash did Apple have at end of FY2023?"),
]

print("=" * 95)
print(f"{'#':<3} {'Type':<25} {'Time':<8} {'Pod':<25} {'Answer preview'}")
print("-" * 95)

total_query_time = 0
for i, (qtype, q) in enumerate(QUESTIONS, 1):
    t0 = time.time()
    state = pipeline.query(state, q)
    elapsed = time.time() - t0
    total_query_time += elapsed
    
    pod = (getattr(state, "winning_pod", "?") or "?")[:24]
    ans = (getattr(state, "final_answer", "") or "")[:40].replace("\n", " ")
    print(f"{i:<3} {qtype:<25} {elapsed:>5.1f}s   {pod:<25} {ans}")

print("=" * 95)
print(f"\nTotal query time: {total_query_time:.1f}s for {len(QUESTIONS)} questions")
print(f"Average per question: {total_query_time/len(QUESTIONS):.1f}s")
print(f"\nExpected behavior:")
print(f"  Sniper hits should be < 1s each (4 of them)")
print(f"  Garbage / Narrative should early-exit < 5s each (2 of them)")
print(f"  Old behavior would be 60-1200s on the 2 misses")