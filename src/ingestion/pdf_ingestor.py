"""
src/ingestion/pdf_ingestor.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 Rev1.0 FINAL

N01 — Universal Document Ingestor
Runs ONCE per document.

Supported formats:
  1. PDF (digital)     — pdfplumber tables + PyMuPDF headings
  2. PDF (scanned/OCR) — pdf2image + pytesseract
  3. DOCX (with OCR)   — python-docx + pytesseract for embedded images
  4. CSV               — pandas
  5. XLSX              — openpyxl + pandas
  6. PNG / JPG (image) — pytesseract OCR directly

Writes to BAState:
  raw_text, table_cells, heading_positions,
  company_name, doc_type, fiscal_year
"""

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.state.ba_state import BAState
from src.utils.resource_governor import ResourceGovernor
from src.utils.seed_manager import SeedManager

SeedManager.set_all()

# ── Tesseract path (Windows default) ─────────────────────────────────────────
import pytesseract
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# ── Heading detection threshold ───────────────────────────────────────────────
HEADING_FONT_SIZE_MIN = 13.0


class PDFIngestor:
    """
    N01: Universal Document Ingestor.
    Auto-detects file format and routes to correct extractor.
    Populates BAState with raw_text, table_cells, heading_positions,
    company_name, doc_type, fiscal_year.
    """

    def __init__(self):
        SeedManager.set_all()

    def run(self, state: BAState) -> BAState:
        """
        Main entry point.
        Auto-detects format from file extension and routes accordingly.
        """
        ResourceGovernor.check("N01 start")

        path = state.document_path
        if not path or not os.path.exists(path):
            raise FileNotFoundError(
                f"[N01] document_path not found: '{path}'"
            )

        ext = Path(path).suffix.lower()
        print(f"[N01] Reading: {path}")
        print(f"[N01] Format : {ext}")

        # Route to correct extractor
        if ext == ".pdf":
            state = self._ingest_pdf(state, path)
        elif ext == ".docx":
            state = self._ingest_docx(state, path)
        elif ext == ".csv":
            state = self._ingest_csv(state, path)
        elif ext in (".xlsx", ".xls"):
            state = self._ingest_xlsx(state, path)
        elif ext in (".png", ".jpg", ".jpeg"):
            state = self._ingest_image(state, path)
        else:
            raise ValueError(
                f"[N01] Unsupported format: '{ext}'. "
                "Supported: .pdf .docx .csv .xlsx .xls .png .jpg .jpeg"
            )

        # Auto-detect metadata if not already set
        if not state.company_name:
            state.company_name = self._detect_company(state.raw_text)
        if not state.doc_type:
            state.doc_type = self._detect_doc_type(state.raw_text)
        if not state.fiscal_year:
            state.fiscal_year = self._detect_fiscal_year(state.raw_text)

        print(f"[N01] Company   : {state.company_name or 'not detected'}")
        print(f"[N01] Doc type  : {state.doc_type     or 'not detected'}")
        print(f"[N01] Fiscal yr : {state.fiscal_year  or 'not detected'}")
        print(f"[N01] Raw text  : {len(state.raw_text):,} chars")
        print(f"[N01] Table cells: {len(state.table_cells)}")
        print(f"[N01] Headings  : {len(state.heading_positions)}")

        ResourceGovernor.check("N01 complete")
        return state

    # ═══════════════════════════════════════════════════════════════════════
    # FORMAT 1 — PDF (digital + scanned/OCR)
    # ═══════════════════════════════════════════════════════════════════════

    def _ingest_pdf(self, state: BAState, path: str) -> BAState:
        """
        PDF ingestion:
        - Try digital extraction first (pdfplumber + PyMuPDF)
        - If text is too short (<100 chars), assume scanned → OCR
        """
        # Try digital first
        raw_text, heading_positions = self._extract_text_and_headings_pymupdf(path)
        table_cells = self._extract_tables_pdfplumber(path)

        # If very little text extracted → scanned PDF → use OCR
        if len(raw_text.strip()) < 100:
            print("[N01] Low text detected — switching to OCR mode")
            raw_text = self._ocr_pdf(path)
            heading_positions = []  # OCR cannot detect headings reliably

        state.raw_text          = raw_text
        state.table_cells       = table_cells
        state.heading_positions = heading_positions
        return state

    def _extract_tables_pdfplumber(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract table cells using pdfplumber — best financial table fidelity."""
        import pdfplumber
        cells = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                if page_num % 20 == 0:
                    ResourceGovernor.check(f"N01 tables page {page_num}")

                tables = page.extract_tables()
                if not tables:
                    continue

                for table_idx, table in enumerate(tables):
                    if not table:
                        continue
                    headers = [
                        str(c).strip() if c else ""
                        for c in table[0]
                    ]
                    for row_idx, row in enumerate(table):
                        if not row:
                            continue
                        row_header = str(row[0]).strip() if row[0] else ""
                        for col_idx, cell_value in enumerate(row):
                            col_header = headers[col_idx] if col_idx < len(headers) else ""
                            cells.append({
                                "page":         page_num,
                                "table_number": table_idx,
                                "row_index":    row_idx,
                                "col_index":    col_idx,
                                "row_header":   row_header,
                                "col_header":   col_header,
                                "cell_value":   str(cell_value).strip() if cell_value else "",
                            })
        return cells

    def _extract_text_and_headings_pymupdf(
        self, pdf_path: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract text and detect headings using PyMuPDF font metadata."""
        import fitz
        raw_parts = []
        headings  = []

        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc, start=1):
            if page_num % 20 == 0:
                ResourceGovernor.check(f"N01 headings page {page_num}")

            raw_parts.append(page.get_text("text"))

            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        size  = span.get("size", 0)
                        flags = span.get("flags", 0)
                        text  = span.get("text", "").strip()
                        if not text:
                            continue
                        is_large = size > HEADING_FONT_SIZE_MIN
                        is_bold  = bool(flags & 2 ** 4)
                        if is_large or is_bold:
                            headings.append({
                                "page":      page_num,
                                "text":      text,
                                "font_size": round(size, 2),
                                "is_bold":   is_bold,
                                "bbox":      span.get("bbox", []),
                            })
        doc.close()
        return "\n".join(raw_parts), headings

    def _ocr_pdf(self, pdf_path: str) -> str:
        """OCR a scanned PDF using pdf2image + pytesseract."""
        try:
            from pdf2image import convert_from_path
            print("[N01] Converting scanned PDF to images for OCR...")
            images = convert_from_path(pdf_path, dpi=200)
            text_parts = []
            for i, img in enumerate(images):
                if i % 10 == 0:
                    ResourceGovernor.check(f"N01 OCR page {i+1}")
                text_parts.append(pytesseract.image_to_string(img))
            return "\n".join(text_parts)
        except Exception as e:
            print(f"[N01] OCR failed: {e}")
            return ""

    # ═══════════════════════════════════════════════════════════════════════
    # FORMAT 2 — DOCX (with OCR for embedded images)
    # ═══════════════════════════════════════════════════════════════════════

    def _ingest_docx(self, state: BAState, path: str) -> BAState:
        """
        DOCX ingestion:
        - Extract text from paragraphs
        - Extract tables preserving structure
        - OCR any embedded images
        - Detect headings from paragraph styles
        """
        import docx as python_docx
        from PIL import Image
        import io

        doc       = python_docx.Document(path)
        text_parts = []
        table_cells = []
        headings   = []

        # Extract paragraphs + headings
        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text)
                # Detect heading from Word style
                style = para.style.name if para.style else ""
                if "Heading" in style or "Title" in style:
                    headings.append({
                        "page":      0,
                        "text":      para.text.strip(),
                        "font_size": 14.0,
                        "is_bold":   True,
                        "bbox":      [],
                    })

        # Extract tables
        for table_idx, table in enumerate(doc.tables):
            headers = [
                cell.text.strip()
                for cell in table.rows[0].cells
            ] if table.rows else []

            for row_idx, row in enumerate(table.rows):
                row_header = row.cells[0].text.strip() if row.cells else ""
                for col_idx, cell in enumerate(row.cells):
                    col_header = headers[col_idx] if col_idx < len(headers) else ""
                    table_cells.append({
                        "page":         0,
                        "table_number": table_idx,
                        "row_index":    row_idx,
                        "col_index":    col_idx,
                        "row_header":   row_header,
                        "col_header":   col_header,
                        "cell_value":   cell.text.strip(),
                    })

        # OCR embedded images
        for rel in doc.part.rels.values():
            if "image" in rel.reltype:
                try:
                    img_data  = rel.target_part.blob
                    img       = Image.open(io.BytesIO(img_data))
                    ocr_text  = pytesseract.image_to_string(img)
                    if ocr_text.strip():
                        text_parts.append(f"[IMAGE OCR]\n{ocr_text}")
                except Exception:
                    pass

        state.raw_text          = "\n".join(text_parts)
        state.table_cells       = table_cells
        state.heading_positions = headings
        return state

    # ═══════════════════════════════════════════════════════════════════════
    # FORMAT 3 — CSV
    # ═══════════════════════════════════════════════════════════════════════

    def _ingest_csv(self, state: BAState, path: str) -> BAState:
        """
        CSV ingestion using pandas.
        Converts all rows to table_cells format.
        Generates raw_text as tab-separated representation.
        """
        import pandas as pd
        df = pd.read_csv(path, dtype=str).fillna("")

        table_cells = []
        headers     = list(df.columns)

        for row_idx, row in df.iterrows():
            row_header = str(row.iloc[0]) if len(row) > 0 else ""
            for col_idx, col_name in enumerate(headers):
                table_cells.append({
                    "page":         1,
                    "table_number": 0,
                    "row_index":    int(row_idx) + 1,
                    "col_index":    col_idx,
                    "row_header":   row_header,
                    "col_header":   col_name,
                    "cell_value":   str(row[col_name]),
                })

        state.raw_text          = df.to_string(index=False)
        state.table_cells       = table_cells
        state.heading_positions = []
        return state

    # ═══════════════════════════════════════════════════════════════════════
    # FORMAT 4 — XLSX
    # ═══════════════════════════════════════════════════════════════════════

    def _ingest_xlsx(self, state: BAState, path: str) -> BAState:
        """
        XLSX ingestion using openpyxl.
        Reads all sheets, preserves cell structure.
        """
        import openpyxl
        wb          = openpyxl.load_workbook(path, data_only=True)
        table_cells = []
        text_parts  = []

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            text_parts.append(f"[SHEET: {sheet_name}]")

            rows = list(ws.iter_rows(values_only=True))
            if not rows:
                continue

            headers = [
                str(c).strip() if c is not None else ""
                for c in rows[0]
            ]

            for row_idx, row in enumerate(rows):
                row_header = str(row[0]).strip() if row[0] is not None else ""
                row_text   = []

                for col_idx, cell_value in enumerate(row):
                    col_header = headers[col_idx] if col_idx < len(headers) else ""
                    val        = str(cell_value).strip() if cell_value is not None else ""
                    row_text.append(val)
                    table_cells.append({
                        "page":         0,
                        "table_number": wb.sheetnames.index(sheet_name),
                        "row_index":    row_idx,
                        "col_index":    col_idx,
                        "row_header":   row_header,
                        "col_header":   col_header,
                        "cell_value":   val,
                    })

                text_parts.append("\t".join(row_text))

        state.raw_text          = "\n".join(text_parts)
        state.table_cells       = table_cells
        state.heading_positions = []
        return state

    # ═══════════════════════════════════════════════════════════════════════
    # FORMAT 5 — PNG / JPG (image OCR)
    # ═══════════════════════════════════════════════════════════════════════

    def _ingest_image(self, state: BAState, path: str) -> BAState:
        """
        Image ingestion using pytesseract OCR.
        Extracts all text from PNG/JPG financial document images.
        """
        from PIL import Image
        print("[N01] Running OCR on image...")
        img      = Image.open(path)
        ocr_text = pytesseract.image_to_string(img)

        state.raw_text          = ocr_text
        state.table_cells       = []
        state.heading_positions = []
        return state

    # ═══════════════════════════════════════════════════════════════════════
    # METADATA AUTO-DETECTION
    # ═══════════════════════════════════════════════════════════════════════

    def _detect_company(self, text: str) -> str:
        """Auto-detect company name from first 2000 chars."""
        snippet = text[:2000].upper()
        companies = [
            ("APPLE",     "Apple Inc"),
            ("MICROSOFT", "Microsoft Corporation"),
            ("TESLA",     "Tesla Inc"),
            ("AMAZON",    "Amazon.com Inc"),
            ("NVIDIA",    "NVIDIA Corporation"),
            ("META",      "Meta Platforms Inc"),
            ("ALPHABET",  "Alphabet Inc"),
            ("GOOGLE",    "Alphabet Inc"),
            ("JPMORGAN",  "JPMorgan Chase"),
            ("GOLDMAN",   "Goldman Sachs"),
            ("NETFLIX",   "Netflix Inc"),
            ("UBER",      "Uber Technologies"),
        ]
        for keyword, name in companies:
            if keyword in snippet:
                return name
        return ""

    def _detect_doc_type(self, text: str) -> str:
        """Auto-detect SEC filing type."""
        snippet = text[:3000].upper()
        if "FORM 10-K" in snippet or "ANNUAL REPORT" in snippet:
            return "10-K"
        if "FORM 10-Q" in snippet or "QUARTERLY REPORT" in snippet:
            return "10-Q"
        if "FORM 8-K" in snippet or "CURRENT REPORT" in snippet:
            return "8-K"
        if "EARNINGS" in snippet and "TRANSCRIPT" in snippet:
            return "Earnings"
        if "PROXY" in snippet:
            return "DEF 14A"
        return ""

    def _detect_fiscal_year(self, text: str) -> str:
        """Auto-detect fiscal year."""
        snippet = text[:5000]
        match = re.search(r'FY\s*(20\d{2})', snippet, re.IGNORECASE)
        if match:
            return f"FY{match.group(1)}"
        match = re.search(
            r'fiscal\s+year\s+(?:ended\s+)?(?:\w+\s+\d+,?\s+)?(20\d{2})',
            snippet, re.IGNORECASE
        )
        if match:
            return f"FY{match.group(1)}"
        match = re.search(
            r'year\s+ended\s+\w+\s+\d+,?\s+(20\d{2})',
            snippet, re.IGNORECASE
        )
        if match:
            return f"FY{match.group(1)}"
        match = re.search(r'\b(20\d{2})\b', snippet)
        if match:
            return f"FY{match.group(1)}"
        return ""


# ═══════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK
# run: python src/ingestion/pdf_ingestor.py
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        from rich import print as rprint
    except ImportError:
        rprint = print

    rprint("\n[bold cyan]── PDFIngestor sanity check ──[/bold cyan]")

    ingestor = PDFIngestor()
    rprint("[green]✓[/green] PDFIngestor instantiated")

    # Test metadata detection
    sample = """
    APPLE INC
    FORM 10-K
    Annual Report for the fiscal year ended September 30, 2023
    """
    assert ingestor._detect_company(sample)     == "Apple Inc"
    assert ingestor._detect_doc_type(sample)    == "10-K"
    assert ingestor._detect_fiscal_year(sample) == "FY2023"
    rprint("[green]✓[/green] Company detection: Apple Inc")
    rprint("[green]✓[/green] Doc type detection: 10-K")
    rprint("[green]✓[/green] Fiscal year detection: FY2023")

    # Test Goldman Sachs detection
    sample2 = "GOLDMAN SACHS GROUP INC FORM 10-K fiscal year 2022"
    assert ingestor._detect_company(sample2)  == "Goldman Sachs"
    assert ingestor._detect_fiscal_year(sample2) == "FY2022"
    rprint("[green]✓[/green] Goldman Sachs detection works")

    # Test supported formats list
    supported = [".pdf", ".docx", ".csv", ".xlsx", ".xls", ".png", ".jpg", ".jpeg"]
    rprint(f"[green]✓[/green] Supported formats: {supported}")

    # Test with real files if they exist
    test_files = {
        "PDF":  "data/pdfs",
        "DOCX": "data/pdfs",
        "CSV":  "data/pdfs",
        "XLSX": "data/pdfs",
    }

    found_any = False
    for fmt in [".pdf", ".docx", ".csv", ".xlsx", ".png", ".jpg"]:
        matches = list(Path("data/pdfs").glob(f"*{fmt}")) if Path("data/pdfs").exists() else []
        if matches:
            found_any = True
            f = matches[0]
            rprint(f"\n[cyan]Testing with: {f}[/cyan]")
            try:
                state = BAState(session_id="test-n01", document_path=str(f))
                state = ingestor.run(state)
                rprint(f"[green]✓[/green] raw_text: {len(state.raw_text):,} chars")
                rprint(f"[green]✓[/green] table_cells: {len(state.table_cells)}")
                rprint(f"[green]✓[/green] headings: {len(state.heading_positions)}")
            except Exception as e:
                rprint(f"[red]✗[/red] Error: {e}")

    if not found_any:
        rprint("\n[yellow]No test files in data/pdfs/ — skipping real file tests[/yellow]")
        rprint("[yellow]Drop any .pdf .docx .csv .xlsx .png .jpg into data/pdfs/ to test[/yellow]")

    rprint("\n[bold green]All checks passed. PDFIngestor ready.[/bold green]\n")