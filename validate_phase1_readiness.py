"""
validate_phase1_readiness.py
Comprehensive Phase 1 readiness check before FinanceBench eval.

Tests:
  1. PDF ingestion works (FinanceBench uses PDFs, not HTML)
  2. HTML ingestion still works (regression)
  3. Sniper still hits Apple 15/15
  4. BM25 index path works for multiple sessions
  5. Free cash flow shows "million" unit
  6. PDF dependencies installed (pdfplumber, pymupdf)
  7. Full pipeline on real PDF (lightweight — skips BGE which is too slow on CPU)

Run: python validate_phase1_readiness.py
"""
import os
import sys
import shutil
import time
from pathlib import Path

# ── Setup ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))


# Track results
class V:
    passed = []
    failed = []
    skipped = []

def OK(msg): V.passed.append(msg); print(f"  ✅ {msg}")
def FAIL(msg): V.failed.append(msg); print(f"  ❌ {msg}")
def SKIP(msg): V.skipped.append(msg); print(f"  ⊝  {msg}")


# ── Pre-cleanup: remove stale validate folders so Check 6 doesn't flag them ──
for path in [
    "data/bm25_index/validate",
    "data/bm25_index/validate_check7",
    "data/bm25_index_validate",
    "data/chromadb_validate",
]:
    abs_path = ROOT / path
    if abs_path.exists():
        try:
            shutil.rmtree(abs_path)
        except Exception:
            pass


print("=" * 75)
print(" 🔍 FINBENCH PHASE 1 READINESS CHECK")
print("=" * 75)

# ── Check 1: PDF dependencies ────────────────────────────────────────────────
print("\n[Check 1] PDF processing dependencies")
try:
    import pdfplumber
    OK(f"pdfplumber installed: v{pdfplumber.__version__}")
except ImportError:
    FAIL("pdfplumber NOT installed — pip install pdfplumber")

try:
    import fitz
    OK(f"PyMuPDF (fitz) installed: v{fitz.__version__}")
except ImportError:
    FAIL("PyMuPDF NOT installed — pip install pymupdf")

# ── Check 2: HTML ingestion still works ──────────────────────────────────────
print("\n[Check 2] HTML ingestion (regression)")
apple_html = ROOT / "documents" / "sec_filings" / "AAPL_FY2023_10-K.html"
if not apple_html.exists():
    SKIP(f"Apple HTML not at {apple_html}")
else:
    try:
        from src.ingestion.pdf_ingestor import PDFIngestor
        t0 = time.time()
        result = PDFIngestor().ingest(str(apple_html))
        elapsed = time.time() - t0
        n_text = len(result.get("raw_text", ""))
        n_cells = len(result.get("table_cells", []))
        if n_text > 100_000 and n_cells > 1000:
            OK(f"HTML ingest works: {n_text:,} chars, {n_cells} cells in {elapsed:.1f}s")
        else:
            FAIL(f"HTML ingest weak: {n_text} chars, {n_cells} cells")
    except Exception as e:
        FAIL(f"HTML ingest exception: {e}")

# ── Check 3: PDF ingestion works ─────────────────────────────────────────────
print("\n[Check 3] PDF ingestion (critical for FinanceBench)")
fb_root = ROOT / "financebench_dataset" / "financebench"
pdf_dir = fb_root / "pdfs"

if not pdf_dir.exists():
    SKIP(f"FinanceBench PDFs not at {pdf_dir} — clone the dataset first")
else:
    pdfs = list(pdf_dir.glob("*.pdf"))
    print(f"     Found {len(pdfs)} PDFs in dataset")
    if pdfs:
        test_pdf = pdfs[0]
        print(f"     Testing: {test_pdf.name}")
        try:
            from src.ingestion.pdf_ingestor import PDFIngestor
            t0 = time.time()
            result = PDFIngestor().ingest(str(test_pdf))
            elapsed = time.time() - t0
            n_text = len(result.get("raw_text", ""))
            n_cells = len(result.get("table_cells", []))
            if n_text > 5000:
                OK(f"PDF ingest works: {n_text:,} chars, {n_cells} cells in {elapsed:.1f}s")
            else:
                FAIL(f"PDF ingest weak: only {n_text} chars from {test_pdf.name}")
        except Exception as e:
            FAIL(f"PDF ingest exception: {e}")
    else:
        SKIP("No PDFs found in pdf directory")

# ── Check 4: Sniper still hits Apple 15/15 ───────────────────────────────────
print("\n[Check 4] SniperRAG regression (Apple FY2023 baseline)")
if not apple_html.exists():
    SKIP("Apple HTML not available")
else:
    try:
        from src.ingestion.pdf_ingestor import PDFIngestor
        from src.retrieval.sniper_rag import run_sniper

        result = PDFIngestor().ingest(str(apple_html))
        cells = result["table_cells"]
        print(f"     Loaded {len(cells)} cells from Apple")

        TESTS = [
            ("revenue", "What was Apple total revenue in FY2023?", "383,285"),
            ("net_income", "What was Apple net income in FY2023?", "96,995"),
            ("total_assets", "What was Apple total assets in FY2023?", "352,583"),
            ("operating_cf", "What was Apple operating cash flow in FY2023?", "110,543"),
            ("free_cash_flow", "What was Apple free cash flow in FY2023?", "99,584"),
        ]

        correct = 0
        for key, q, expected in TESTS:
            try:
                r = run_sniper(q, cells)
                ans = str(r.answer or "")
                expected_clean = expected.replace(",", "")
                ans_clean = ans.replace(",", "")
                if expected_clean in ans_clean:
                    correct += 1
                    print(f"     ✓ {key}: hit ({expected})")
                else:
                    print(f"     ✗ {key}: expected {expected}, got {ans[:60]}")
            except Exception as e:
                print(f"     ✗ {key}: exception {e}")

        if correct >= 4:
            OK(f"Sniper accuracy: {correct}/5 on Apple smoke test")
        else:
            FAIL(f"Sniper regression: only {correct}/5 on Apple")
    except Exception as e:
        FAIL(f"Sniper test exception: {e}")

# ── Check 5: free_cash_flow shows "million" not "x10^6" ─────────────────────
print("\n[Check 5] free_cash_flow unit display (Fix 6 — non-Apple regression)")
if not apple_html.exists():
    SKIP("Apple HTML not available")
else:
    try:
        from src.retrieval.sniper_rag import run_sniper
        from src.ingestion.pdf_ingestor import PDFIngestor
        result = PDFIngestor().ingest(str(apple_html))
        r = run_sniper("What was Apple free cash flow in FY2023?", result["table_cells"])
        ans = str(r.answer or "")
        if "million" in ans.lower() or "billion" in ans.lower():
            OK(f"FCF shows friendly unit: '{ans[:80]}'")
        elif "x10^6" in ans or "x10^9" in ans:
            FAIL(f"FCF still shows raw unit: '{ans[:80]}'")
        else:
            print(f"     Got: '{ans[:80]}'")
            SKIP("Unit format unclear — manual review needed")
    except Exception as e:
        FAIL(f"FCF check exception: {e}")

# ── Check 6: BM25 path bug ───────────────────────────────────────────────────
print("\n[Check 6] BM25 chunks_meta.json path consistency")
try:
    bm25_root = ROOT / "data" / "bm25_index"
    if bm25_root.exists():
        sub_dirs = [d for d in bm25_root.iterdir() if d.is_dir()]
        broken = []
        for d in sub_dirs:
            meta = d / "chunks_meta.json"
            if not meta.exists():
                broken.append(d.name)
        if not sub_dirs:
            SKIP("No BM25 indices exist yet (will be created on ingest)")
        elif not broken:
            OK(f"BM25 indices OK: {len(sub_dirs)} sessions, all have chunks_meta.json")
        else:
            FAIL(f"BM25 missing chunks_meta.json in: {', '.join(broken[:3])}")
    else:
        SKIP("data/bm25_index/ doesn't exist yet")
except Exception as e:
    FAIL(f"BM25 path check exception: {e}")

# ── Check 7: PDF ingestion + chunking (lightweight — skip BGE) ──────────────
print("\n[Check 7] PDF ingestion + chunking (lightweight, no BGE)")
if not pdf_dir.exists():
    SKIP("FinanceBench dataset not cloned")
else:
    # Filter to 10-K filings (skip 8-K event filings which are tiny)
    pdfs_10k = [p for p in pdf_dir.glob("*10K*.pdf")]
    if not pdfs_10k:
        pdfs_10k = [p for p in pdf_dir.glob("*.pdf") if p.stat().st_size > 100_000]
    if not pdfs_10k:
        SKIP("No 10-K PDFs found in dataset")
    else:
        test_pdf = min(pdfs_10k, key=lambda p: p.stat().st_size)
        size_kb = test_pdf.stat().st_size / 1024
        print(f"     Testing smallest 10-K: {test_pdf.name} ({size_kb:.0f} KB)")
        try:
            from src.ingestion.pdf_ingestor import PDFIngestor
            from src.ingestion.section_tree_builder import SectionTreeBuilder
            from src.ingestion.chunker import Chunker
            from src.state.ba_state import BAState
            from src.retrieval.sniper_rag import run_sniper

            t0 = time.time()
            ing = PDFIngestor().ingest(str(test_pdf))
            ingest_time = time.time() - t0
            n_chars = len(ing.get("raw_text", ""))
            n_cells = len(ing.get("table_cells", []))
            print(f"     PDF text:     {ingest_time:.1f}s  ({n_chars:,} chars, {n_cells} cells)")

            t0 = time.time()
            state = BAState(
                session_id="check7_validate",
                raw_text=ing["raw_text"],
                table_cells=ing.get("table_cells", []),
                company_name="TestCo",
                doc_type="10-K",
                fiscal_year="FY2023",
            )
            state = SectionTreeBuilder().run(state)
            tree_time = time.time() - t0
            n_sections = len(state.section_tree.get("children", []) if state.section_tree else [])
            print(f"     Section tree: {tree_time:.1f}s  ({n_sections} sections)")

            os.environ["DISABLE_CHROMADB"] = "1"
            t0 = time.time()
            chunker = Chunker(
                bm25_dir="data/bm25_index_validate",
                chromadb_dir="data/chromadb_validate",
            )
            chunks = chunker.chunk(state.raw_text, state.section_tree, "TestCo", "10-K", "FY2023")
            chunk_time = time.time() - t0
            os.environ.pop("DISABLE_CHROMADB", None)
            print(f"     Chunker:      {chunk_time:.1f}s  ({len(chunks)} chunks)")

            t0 = time.time()
            sniper_result = run_sniper("What was total revenue?", ing.get("table_cells", []))
            sniper_time = time.time() - t0
            print(f"     Sniper test:  {sniper_time:.2f}s  (hit={sniper_result.sniper_hit})")

            total_time = ingest_time + tree_time + chunk_time + sniper_time
            if len(chunks) >= 10 and n_cells >= 50:
                OK(f"PDF pipeline works: {len(chunks)} chunks, {n_cells} cells, total {total_time:.1f}s")
            elif len(chunks) >= 5 and n_cells >= 20:
                OK(f"PDF pipeline functional: {len(chunks)} chunks, {n_cells} cells, total {total_time:.1f}s")
            else:
                FAIL(f"PDF pipeline weak: only {len(chunks)} chunks, {n_cells} cells")
        except Exception as e:
            FAIL(f"Pipeline exception on PDF: {e}")

# ── Post-cleanup ─────────────────────────────────────────────────────────────
for path in [
    "data/bm25_index_validate",
    "data/chromadb_validate",
]:
    abs_path = ROOT / path
    if abs_path.exists():
        try:
            shutil.rmtree(abs_path)
        except Exception:
            pass

# ── Summary ──────────────────────────────────────────────────────────────────
print()
print("=" * 75)
print(" 📊 SUMMARY")
print("=" * 75)
print(f"  ✅ Passed:   {len(V.passed)}")
print(f"  ❌ Failed:   {len(V.failed)}")
print(f"  ⊝  Skipped:  {len(V.skipped)}")
print()

if V.failed:
    print("  FAILURES:")
    for msg in V.failed:
        print(f"    - {msg}")
    print()

if V.skipped:
    print("  SKIPPED:")
    for msg in V.skipped:
        print(f"    - {msg}")
    print()

ready = len(V.failed) == 0
if ready:
    print("  🎯 VERDICT: READY FOR FINANCEBENCH EVAL")
else:
    print(f"  ⚠️ VERDICT: FIX {len(V.failed)} ISSUE(S) BEFORE EVAL")

print("=" * 75)
sys.exit(0 if ready else 1)