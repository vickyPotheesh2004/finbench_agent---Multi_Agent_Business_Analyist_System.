"""
Throwaway diagnostic — run after applying S11 fix to verify real Apple 10-K
extracts properly. Delete after running.
"""
from src.ingestion.pdf_ingestor import PDFIngestor

PATH = "documents/sec_filings/AAPL_FY2023_10-K.html"

print(f"Ingesting: {PATH}")
print("=" * 70)

r = PDFIngestor().ingest(PATH)

print(f"chars:        {len(r['raw_text']):,}")
print(f"headings:     {len(r['heading_positions']):,}")
print(f"table_cells:  {len(r['table_cells']):,}")
print(f"company:      {r['company_name']!r}")
print(f"doc_type:     {r['doc_type']!r}")
print(f"fiscal_year:  {r['fiscal_year']!r}")
print()

print("=" * 70)
print("First 10 headings (sorted by font size desc, then page asc):")
print("=" * 70)
sorted_h = sorted(
    r["heading_positions"],
    key=lambda h: (-h["font_size"], h["page"]),
)
for h in sorted_h[:10]:
    print(f"  {h['font_size']:5.1f}pt  page={h['page']:3d}  {h['text'][:70]}")
print()

print("=" * 70)
print("iXBRL fact breakdown:")
print("=" * 70)
ixbrl = [c for c in r["table_cells"] if c.get("section", "").startswith("iXBRL")]
html_tables = [c for c in r["table_cells"] if not c.get("section", "").startswith("iXBRL")]
print(f"  iXBRL facts       : {len(ixbrl):,}")
print(f"  HTML table cells  : {len(html_tables):,}")
print()

print("Sample 10 iXBRL numeric facts:")
numeric = [c for c in ixbrl if c.get("section") == "iXBRL_NUMERIC"]
for c in numeric[:10]:
    name  = c["row_header"][:50]
    value = c["value"][:25]
    print(f"  {name:50s}  {value}")
print()

# Look for the key facts FinanceBench cares about
print("=" * 70)
print("Key FinanceBench facts (Apple FY2023):")
print("=" * 70)

def find_fact(name_substr: str):
    matches = [c for c in r["table_cells"]
               if name_substr.lower() in (c.get("row_header") or "").lower()]
    return matches

for query in ["Revenues", "NetIncomeLoss", "EarningsPerShareDiluted",
              "Assets", "GrossProfit", "OperatingIncome"]:
    found = find_fact(query)
    if found:
        # Show first 3 distinct values
        seen = set()
        unique = []
        for c in found:
            if c["value"] not in seen:
                seen.add(c["value"])
                unique.append(c)
            if len(unique) >= 3:
                break
        print(f"  {query}:")
        for c in unique:
            print(f"     {c['value'][:30]:30s}  ctx={c['col_header'][:20]}")
    else:
        print(f"  {query}: NOT FOUND")

print()
print("=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)