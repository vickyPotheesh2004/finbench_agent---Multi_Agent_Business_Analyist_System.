"""Quick check that Fix 6 (million unit display) is working."""
from src.ingestion.pdf_ingestor import PDFIngestor
from src.retrieval.sniper_rag import run_sniper

print("Loading Apple...")
r = PDFIngestor().ingest("documents/sec_filings/AAPL_FY2023_10-K.html")
cells = r["table_cells"]

result = run_sniper("What was Apple total revenue in FY2023?", cells)
print()
print("Full answer text:")
print(f"  '{result.answer}'")
print()

if "million" in result.answer.lower():
    print("✅ Fix 6 WORKING: 'million' appears in answer")
elif "x10^6" in result.answer:
    print("❌ Fix 6 NOT applied: still showing 'x10^6'")
else:
    print(f"🟡 Unit display unclear — full: {result.answer}")
    
print()
print("Cell unit field:", result.unit)