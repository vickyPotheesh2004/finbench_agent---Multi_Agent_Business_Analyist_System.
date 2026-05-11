"""Fix 4 verification — chunker produces more chunks for Apple."""
import time
from src.ingestion.pdf_ingestor import PDFIngestor
from src.ingestion.section_tree_builder import SectionTreeBuilder
from src.ingestion.chunker import Chunker
from src.state.ba_state import BAState

print("Ingesting Apple 10-K...")
t0 = time.time()
ing = PDFIngestor().ingest("documents/sec_filings/AAPL_FY2023_10-K.html")
print(f"PDF ingest: {time.time()-t0:.1f}s, raw_text={len(ing['raw_text'])} chars")

state = BAState(
    session_id="test",
    raw_text=ing["raw_text"],
    company_name="Apple Inc.",
    doc_type="10-K",
    fiscal_year="FY2023",
)

t0 = time.time()
state = SectionTreeBuilder().run(state)
sections = state.section_tree.get("children", []) if state.section_tree else []
print(f"Section tree: {time.time()-t0:.1f}s, sections={len(sections)}")

t0 = time.time()
chunker = Chunker(bm25_dir="data/bm25_index", chromadb_dir="data/chromadb")
chunks = chunker.chunk(state.raw_text, state.section_tree, "Apple Inc.", "10-K", "FY2023")
elapsed = time.time() - t0

print(f"\n=== FIX 4 RESULTS ===")
print(f"Chunker: {elapsed:.1f}s")
print(f"Total chunks: {len(chunks)}")
if chunks:
    char_counts = [c.char_count for c in chunks]
    avg = sum(char_counts) // len(chunks)
    min_c = min(char_counts)
    max_c = max(char_counts)
    print(f"Avg chunk: {avg} chars")
    print(f"Min chunk: {min_c} chars")
    print(f"Max chunk: {max_c} chars")
    print(f"Total content: {sum(char_counts):,} chars (vs raw_text {len(state.raw_text):,})")
    print(f"Coverage: {100*sum(char_counts)/len(state.raw_text):.1f}%")
    
    # Show section variety
    sections_seen = set(c.section for c in chunks)
    print(f"\nUnique sections in chunks: {len(sections_seen)}")
    for s in list(sections_seen)[:10]:
        n = sum(1 for c in chunks if c.section == s)
        print(f"  '{s[:50]}': {n} chunks")

print(f"\nBefore Fix 4: 30 chunks at 7000 chars each = ~210K chars (with 3000-char cap dropping content)")
print(f"After Fix 4:  {len(chunks)} chunks at {avg if chunks else 0} chars each")
print(f"\n  Target: >100 chunks, broad section coverage")
print(f"  Status: {'✅ PASS' if len(chunks) > 100 else '❌ FAIL' if len(chunks) < 50 else '🟡 PARTIAL'}")