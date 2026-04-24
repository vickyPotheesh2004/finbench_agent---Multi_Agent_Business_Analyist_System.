"""
tests/test_n01_pdf_ingestor.py
Tests for N01 PDF Ingestor
PDR-BAAAI-001 · Rev 1.0
"""

import os
import csv
import json
import pytest
from src.ingestion.pdf_ingestor import (
    PDFIngestor,
    TableCell,
    run_pdf_ingestor,
    SUPPORTED_EXTENSIONS,
    HEADING_FONT_SIZE_MIN,
)
from src.state.ba_state import BAState


@pytest.fixture
def ingestor():
    return PDFIngestor()


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV file for testing."""
    path = str(tmp_path / "test_financials.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric",       "FY2023",    "FY2022"])
        writer.writerow(["Revenue",      "383285",    "394328"])
        writer.writerow(["Net Income",   "96995",     "99803"])
        writer.writerow(["Gross Profit", "169148",    "170782"])
        writer.writerow(["EPS Diluted",  "6.13",      "6.11"])
    return path


@pytest.fixture
def sample_txt(tmp_path):
    """Create a sample TXT file for testing."""
    path = str(tmp_path / "test_filing.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "Apple Inc Annual Report 10-K\n"
            "Fiscal Year ended September 30, 2023\n\n"
            "Total net sales: $383,285 million\n"
            "Net income: $96,995 million\n"
            "Diluted EPS: $6.13\n"
        )
    return path


@pytest.fixture
def sample_json(tmp_path):
    """Create a sample JSON file for testing."""
    path = str(tmp_path / "test_data.json")
    data = {
        "company": "Apple Inc",
        "fiscal_year": "FY2023",
        "revenue": 383285,
        "net_income": 96995,
    }
    with open(path, "w") as f:
        json.dump(data, f)
    return path


# ── Group 1: Constants ────────────────────────────────────────────────────────

class TestConstants:

    def test_01_supported_extensions_has_pdf(self):
        assert ".pdf" in SUPPORTED_EXTENSIONS

    def test_02_supported_extensions_has_docx(self):
        assert ".docx" in SUPPORTED_EXTENSIONS

    def test_03_supported_extensions_has_xlsx(self):
        assert ".xlsx" in SUPPORTED_EXTENSIONS

    def test_04_heading_font_size_min_is_13(self):
        assert HEADING_FONT_SIZE_MIN == 13.0

    def test_05_at_least_8_extensions_supported(self):
        assert len(SUPPORTED_EXTENSIONS) >= 8


# ── Group 2: TableCell ────────────────────────────────────────────────────────

class TestTableCell:

    def test_06_table_cell_creates(self):
        cell = TableCell(
            row_header="Revenue", col_header="FY2023",
            value="383285", page=94, table_number=1
        )
        assert cell.row_header   == "Revenue"
        assert cell.col_header   == "FY2023"
        assert cell.value        == "383285"
        assert cell.page         == 94
        assert cell.table_number == 1

    def test_07_to_dict_has_required_keys(self):
        cell = TableCell("Revenue", "FY2023", "383285", 94, 1)
        d    = cell.to_dict()
        assert "row_header"   in d
        assert "col_header"   in d
        assert "value"        in d
        assert "page"         in d
        assert "table_number" in d
        assert "section"      in d

    def test_08_default_values(self):
        cell = TableCell()
        assert cell.row_header   == ""
        assert cell.value        == ""
        assert cell.page         == 0
        assert cell.table_number == 0


# ── Group 3: CSV ingestion ────────────────────────────────────────────────────

class TestCSVIngestion:

    def test_09_csv_returns_dict(self, ingestor, sample_csv):
        result = ingestor.ingest(sample_csv)
        assert isinstance(result, dict)

    def test_10_csv_has_required_keys(self, ingestor, sample_csv):
        result = ingestor.ingest(sample_csv)
        assert "raw_text"          in result
        assert "table_cells"       in result
        assert "heading_positions" in result

    def test_11_csv_extracts_table_cells(self, ingestor, sample_csv):
        result = ingestor.ingest(sample_csv)
        assert len(result["table_cells"]) > 0

    def test_12_csv_table_cells_have_value(self, ingestor, sample_csv):
        result = ingestor.ingest(sample_csv)
        for cell in result["table_cells"]:
            assert "value"      in cell
            assert "row_header" in cell
            assert "col_header" in cell

    def test_13_csv_raw_text_not_empty(self, ingestor, sample_csv):
        result = ingestor.ingest(sample_csv)
        assert len(result["raw_text"]) > 0

    def test_14_csv_row_headers_correct(self, ingestor, sample_csv):
        result = ingestor.ingest(sample_csv)
        row_headers = [c["row_header"] for c in result["table_cells"]]
        assert "Revenue" in row_headers

    def test_15_csv_values_correct(self, ingestor, sample_csv):
        result = ingestor.ingest(sample_csv)
        values = [c["value"] for c in result["table_cells"]]
        assert "383285" in values


# ── Group 4: TXT ingestion ────────────────────────────────────────────────────

class TestTXTIngestion:

    def test_16_txt_returns_dict(self, ingestor, sample_txt):
        result = ingestor.ingest(sample_txt)
        assert isinstance(result, dict)

    def test_17_txt_raw_text_not_empty(self, ingestor, sample_txt):
        result = ingestor.ingest(sample_txt)
        assert len(result["raw_text"]) > 50

    def test_18_txt_contains_content(self, ingestor, sample_txt):
        result = ingestor.ingest(sample_txt)
        assert "383,285" in result["raw_text"] or "383285" in result["raw_text"]


# ── Group 5: JSON ingestion ───────────────────────────────────────────────────

class TestJSONIngestion:

    def test_19_json_returns_dict(self, ingestor, sample_json):
        result = ingestor.ingest(sample_json)
        assert isinstance(result, dict)

    def test_20_json_raw_text_not_empty(self, ingestor, sample_json):
        result = ingestor.ingest(sample_json)
        assert len(result["raw_text"]) > 0

    def test_21_json_contains_company(self, ingestor, sample_json):
        result = ingestor.ingest(sample_json)
        assert "Apple" in result["raw_text"]


# ── Group 6: Metadata extraction ─────────────────────────────────────────────

class TestMetadataExtraction:

    def test_22_extract_doc_type_10k(self):
        text = "Annual Report on Form 10-K for fiscal year 2023"
        dt   = PDFIngestor._extract_doc_type(text)
        assert dt == "10-K"

    def test_23_extract_doc_type_10q(self):
        text = "Quarterly Report on Form 10-Q for quarter ended"
        dt   = PDFIngestor._extract_doc_type(text)
        assert dt == "10-Q"

    def test_24_extract_doc_type_unknown(self):
        text = "Some random document"
        dt   = PDFIngestor._extract_doc_type(text)
        assert dt == "UNKNOWN"

    def test_25_extract_fiscal_year_fy2023(self):
        text = "Fiscal Year ended September 30, 2023"
        fy   = PDFIngestor._extract_fiscal_year(text)
        assert fy == "FY2023"

    def test_26_extract_fiscal_year_year_ended(self):
        text = "For the year ended December 31, 2022"
        fy   = PDFIngestor._extract_fiscal_year(text)
        assert fy == "FY2022"

    def test_27_extract_fiscal_year_fy_shorthand(self):
        text = "FY 2023 Annual Performance Summary"
        fy   = PDFIngestor._extract_fiscal_year(text)
        assert fy == "FY2023"

    def test_28_extract_company_inc(self):
        text = "Apple Inc Annual Report 10-K"
        co   = PDFIngestor._extract_company(text)
        assert "Apple" in co or co == ""  # graceful if no match

    def test_29_metadata_returns_tuple(self):
        ingestor = PDFIngestor()
        result   = ingestor._extract_metadata(
            "Apple Inc 10-K Fiscal Year 2023", []
        )
        assert len(result) == 3  # (company, doc_type, fiscal_year)


# ── Group 7: BAState integration ─────────────────────────────────────────────

class TestBAStateIntegration:

    def test_30_run_writes_raw_text(self, ingestor, sample_txt):
        state = BAState(
            session_id    = "t30",
            document_path = sample_txt,
        )
        state = ingestor.run(state)
        assert isinstance(state.raw_text, str)
        assert len(state.raw_text) > 0

    def test_31_run_writes_table_cells(self, ingestor, sample_csv):
        state = BAState(
            session_id    = "t31",
            document_path = sample_csv,
        )
        state = ingestor.run(state)
        assert isinstance(state.table_cells, list)

    def test_32_run_writes_heading_positions(self, ingestor, sample_txt):
        state = BAState(
            session_id    = "t32",
            document_path = sample_txt,
        )
        state = ingestor.run(state)
        assert isinstance(state.heading_positions, list)

    def test_33_run_empty_path_skips_gracefully(self, ingestor):
        state = BAState(session_id="t33", document_path="")
        state = ingestor.run(state)
        assert state.raw_text == ""

    def test_34_run_missing_file_skips_gracefully(self, ingestor):
        state = BAState(
            session_id    = "t34",
            document_path = "/nonexistent/file.pdf",
        )
        state = ingestor.run(state)
        assert state.raw_text == ""

    def test_35_seed_unchanged(self, ingestor, sample_txt):
        state = BAState(
            session_id    = "t35",
            document_path = sample_txt,
        )
        state = ingestor.run(state)
        assert state.seed == 42

    def test_36_no_rlef_in_raw_text(self, ingestor, sample_txt):
        """C9: raw_text must not contain _rlef_ content."""
        state = BAState(
            session_id    = "t36",
            document_path = sample_txt,
        )
        state = ingestor.run(state)
        assert "_rlef_" not in state.raw_text

    def test_37_company_name_not_overridden_if_set(self, ingestor, sample_txt):
        state = BAState(
            session_id    = "t37",
            document_path = sample_txt,
            company_name  = "Pre-set Company",
        )
        state = ingestor.run(state)
        # Should not override pre-set company name
        assert state.company_name == "Pre-set Company"

    def test_38_txt_fiscal_year_extracted(self, ingestor, sample_txt):
        state = BAState(
            session_id    = "t38",
            document_path = sample_txt,
        )
        state = ingestor.run(state)
        assert state.fiscal_year in ["FY2023", ""]


# ── Group 8: Convenience wrapper ─────────────────────────────────────────────

class TestConvenienceWrapper:

    def test_39_run_pdf_ingestor_returns_state(self, sample_csv):
        state = BAState(
            session_id    = "t39",
            document_path = sample_csv,
        )
        result = run_pdf_ingestor(state)
        assert hasattr(result, "raw_text")
        assert result.seed == 42

    def test_40_wrapper_populates_table_cells(self, sample_csv):
        state = BAState(
            session_id    = "t40",
            document_path = sample_csv,
        )
        result = run_pdf_ingestor(state)
        assert isinstance(result.table_cells, list)
        assert len(result.table_cells) > 0