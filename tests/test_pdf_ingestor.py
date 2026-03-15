"""
tests/test_pdf_ingestor.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 Rev1.0 FINAL

Tests for N01 — Universal Document Ingestor
Run: pytest tests/test_pdf_ingestor.py -v
"""

import csv
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.ingestion.pdf_ingestor import PDFIngestor
from src.state.ba_state import BAState


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def ingestor():
    return PDFIngestor()


@pytest.fixture
def fresh_state():
    return BAState(session_id="test-n01")


@pytest.fixture
def sample_csv(tmp_path):
    """Create a temporary CSV file for testing."""
    csv_file = tmp_path / "test_financials.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "FY2022", "FY2023"])
        writer.writerow(["Revenue", "394.33B", "383.29B"])
        writer.writerow(["Net Income", "99.80B", "96.99B"])
        writer.writerow(["EPS Diluted", "6.11", "6.13"])
    return str(csv_file)


@pytest.fixture
def sample_xlsx(tmp_path):
    """Create a temporary XLSX file for testing."""
    import openpyxl
    xlsx_file = tmp_path / "test_financials.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Income Statement"
    ws.append(["Metric", "FY2022", "FY2023"])
    ws.append(["Revenue", "394.33B", "383.29B"])
    ws.append(["Net Income", "99.80B", "96.99B"])
    wb.save(str(xlsx_file))
    return str(xlsx_file)


@pytest.fixture
def sample_docx(tmp_path):
    """Create a temporary DOCX file for testing."""
    import docx
    docx_file = tmp_path / "test_report.docx"
    doc = docx.Document()
    doc.add_heading("Apple Inc — Annual Report FY2023", level=1)
    doc.add_paragraph("FORM 10-K")
    doc.add_paragraph(
        "Apple Inc fiscal year ended September 30, 2023. "
        "Total net revenue was $383.29 billion."
    )
    table = doc.add_table(rows=3, cols=3)
    table.cell(0, 0).text = "Metric"
    table.cell(0, 1).text = "FY2022"
    table.cell(0, 2).text = "FY2023"
    table.cell(1, 0).text = "Revenue"
    table.cell(1, 1).text = "394.33B"
    table.cell(1, 2).text = "383.29B"
    table.cell(2, 0).text = "Net Income"
    table.cell(2, 1).text = "99.80B"
    table.cell(2, 2).text = "96.99B"
    doc.save(str(docx_file))
    return str(docx_file)


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 1 — Instantiation
# ═══════════════════════════════════════════════════════════════════════════

class TestInstantiation:

    def test_01_ingestor_instantiates(self, ingestor):
        """N01: PDFIngestor must instantiate without error"""
        assert ingestor is not None

    def test_02_missing_file_raises(self, ingestor, fresh_state):
        """N01: Missing file must raise FileNotFoundError"""
        fresh_state.document_path = "nonexistent/file.pdf"
        with pytest.raises(FileNotFoundError):
            ingestor.run(fresh_state)

    def test_03_unsupported_format_raises(self, ingestor, fresh_state, tmp_path):
        """N01: Unsupported format must raise ValueError"""
        bad_file = tmp_path / "file.xyz"
        bad_file.write_text("test")
        fresh_state.document_path = str(bad_file)
        with pytest.raises(ValueError, match="Unsupported format"):
            ingestor.run(fresh_state)


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 2 — Metadata detection
# ═══════════════════════════════════════════════════════════════════════════

class TestMetadataDetection:

    def test_04_detect_apple(self, ingestor):
        """N01: Must detect Apple Inc"""
        text = "APPLE INC FORM 10-K Annual Report 2023"
        assert ingestor._detect_company(text) == "Apple Inc"

    def test_05_detect_goldman(self, ingestor):
        """N01: Must detect Goldman Sachs"""
        text = "GOLDMAN SACHS GROUP fiscal year 2022"
        assert ingestor._detect_company(text) == "Goldman Sachs"

    def test_06_detect_microsoft(self, ingestor):
        """N01: Must detect Microsoft"""
        text = "MICROSOFT CORPORATION FORM 10-K"
        assert ingestor._detect_company(text) == "Microsoft Corporation"

    def test_07_detect_10k(self, ingestor):
        """N01: Must detect 10-K filing"""
        text = "FORM 10-K Annual Report for the fiscal year"
        assert ingestor._detect_doc_type(text) == "10-K"

    def test_08_detect_10q(self, ingestor):
        """N01: Must detect 10-Q filing"""
        text = "FORM 10-Q Quarterly Report for the period"
        assert ingestor._detect_doc_type(text) == "10-Q"

    def test_09_detect_fiscal_year_fy(self, ingestor):
        """N01: Must detect FY2023 pattern"""
        text = "Results for FY2023 were strong"
        assert ingestor._detect_fiscal_year(text) == "FY2023"

    def test_10_detect_fiscal_year_ended(self, ingestor):
        """N01: Must detect year ended pattern"""
        text = "for the year ended December 31, 2022"
        assert ingestor._detect_fiscal_year(text) == "FY2022"

    def test_11_detect_fiscal_year_standalone(self, ingestor):
        """N01: Must detect standalone year"""
        text = "Annual Report 2021"
        assert ingestor._detect_fiscal_year(text) == "FY2021"


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 3 — CSV ingestion
# ═══════════════════════════════════════════════════════════════════════════

class TestCSVIngestion:

    def test_12_csv_loads(self, ingestor, fresh_state, sample_csv):
        """N01: CSV must load without error"""
        fresh_state.document_path = sample_csv
        state = ingestor.run(fresh_state)
        assert state.raw_text != ""

    def test_13_csv_table_cells_populated(self, ingestor, fresh_state, sample_csv):
        """N01: CSV must produce table_cells"""
        fresh_state.document_path = sample_csv
        state = ingestor.run(fresh_state)
        assert len(state.table_cells) > 0

    def test_14_csv_table_cells_have_required_keys(self, ingestor, fresh_state, sample_csv):
        """N01: Every table cell must have all required keys"""
        fresh_state.document_path = sample_csv
        state = ingestor.run(fresh_state)
        required = {"page", "table_number", "row_index", "col_index",
                    "row_header", "col_header", "cell_value"}
        for cell in state.table_cells:
            assert required.issubset(cell.keys()), f"Missing keys in: {cell}"

    def test_15_csv_col_headers_correct(self, ingestor, fresh_state, sample_csv):
        """N01: CSV column headers must be detected correctly"""
        fresh_state.document_path = sample_csv
        state = ingestor.run(fresh_state)
        headers = {c["col_header"] for c in state.table_cells}
        assert "FY2022" in headers
        assert "FY2023" in headers


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 4 — XLSX ingestion
# ═══════════════════════════════════════════════════════════════════════════

class TestXLSXIngestion:

    def test_16_xlsx_loads(self, ingestor, fresh_state, sample_xlsx):
        """N01: XLSX must load without error"""
        fresh_state.document_path = sample_xlsx
        state = ingestor.run(fresh_state)
        assert state.raw_text != ""

    def test_17_xlsx_table_cells_populated(self, ingestor, fresh_state, sample_xlsx):
        """N01: XLSX must produce table_cells"""
        fresh_state.document_path = sample_xlsx
        state = ingestor.run(fresh_state)
        assert len(state.table_cells) > 0

    def test_18_xlsx_sheet_name_in_raw_text(self, ingestor, fresh_state, sample_xlsx):
        """N01: Sheet name must appear in raw_text"""
        fresh_state.document_path = sample_xlsx
        state = ingestor.run(fresh_state)
        assert "Income Statement" in state.raw_text


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 5 — DOCX ingestion
# ═══════════════════════════════════════════════════════════════════════════

class TestDOCXIngestion:

    def test_19_docx_loads(self, ingestor, fresh_state, sample_docx):
        """N01: DOCX must load without error"""
        fresh_state.document_path = sample_docx
        state = ingestor.run(fresh_state)
        assert state.raw_text != ""

    def test_20_docx_table_cells_populated(self, ingestor, fresh_state, sample_docx):
        """N01: DOCX must produce table_cells"""
        fresh_state.document_path = sample_docx
        state = ingestor.run(fresh_state)
        assert len(state.table_cells) > 0

    def test_21_docx_headings_detected(self, ingestor, fresh_state, sample_docx):
        """N01: DOCX headings must be detected"""
        fresh_state.document_path = sample_docx
        state = ingestor.run(fresh_state)
        assert len(state.heading_positions) > 0

    def test_22_docx_metadata_detected(self, ingestor, fresh_state, sample_docx):
        """N01: DOCX must auto-detect company + doc_type + fiscal_year"""
        fresh_state.document_path = sample_docx
        state = ingestor.run(fresh_state)
        assert state.company_name == "Apple Inc"
        assert state.doc_type     == "10-K"
        assert state.fiscal_year  == "FY2023"


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 6 — BAState integration
# ═══════════════════════════════════════════════════════════════════════════

class TestBAStateIntegration:

    def test_23_state_written_correctly(self, ingestor, fresh_state, sample_csv):
        """N01: All BAState fields must be written after ingestion"""
        fresh_state.document_path = sample_csv
        state = ingestor.run(fresh_state)
        assert isinstance(state.raw_text,          str)
        assert isinstance(state.table_cells,        list)
        assert isinstance(state.heading_positions,  list)

    def test_24_c8_chunk_prefix_works_after_ingestion(
        self, ingestor, fresh_state, sample_docx
    ):
        """C8: chunk_metadata_prefix must work after N01 populates state"""
        fresh_state.document_path = sample_docx
        state = ingestor.run(fresh_state)
        prefix = state.chunk_metadata_prefix("MD&A", "1")
        assert state.company_name in prefix
        assert state.doc_type     in prefix
        assert state.fiscal_year  in prefix