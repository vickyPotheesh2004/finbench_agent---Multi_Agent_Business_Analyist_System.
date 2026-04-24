"""
tests/test_pdf_report_generator.py
Tests for PDF Report Generator (N19b)
PDR-BAAAI-001 Rev 1.0
30 tests - tests generation without a real LLM
"""
import os
import pytest
from pathlib import Path
from src.output.pdf_report_generator import (
    PDFReportGenerator,
    generate_pdf_report,
    run_pdf_report_generator,
    DEFAULT_OUTPUT_DIR,
    COMPANY_NAME,
    REPORT_VERSION,
    COL_PRIMARY,
    COL_SECONDARY,
)
from src.state.ba_state import BAState


@pytest.fixture
def out_dir(tmp_path):
    return str(tmp_path / "reports")


@pytest.fixture
def gen(out_dir):
    return PDFReportGenerator(output_dir=out_dir)


def _make_state(**kwargs):
    defaults = {
        "session_id":          "test-pdf-001",
        "query":               "What was Apple total net sales in FY2023?",
        "company_name":        "Apple Inc",
        "doc_type":            "10-K",
        "fiscal_year":         "FY2023",
        "final_answer":        "Total net sales were $383,285 million in fiscal 2023.",
        "confidence_score":    0.87,
        "risk_score":          15.0,
        "agreement_status":    "majority_agree",
        "winning_pod":         "analyst",
        "iteration_count":     2,
        "chunk_count":         247,
        "analyst_output":      "Total net sales were $383,285 million.",
        "analyst_confidence":  0.87,
        "analyst_citations":   ["INCOME_STATEMENT / p.94"],
        "analyst_attempt_count": 1,
        "quant_result":        "Revenue computed: $383,285M.",
        "quant_confidence":    0.89,
        "quant_citations":     ["INCOME_STATEMENT / p.94"],
        "auditor_output":      "Confirmed $383,285M independently.",
        "auditor_confidence":  0.85,
        "auditor_citations":   ["INCOME_STATEMENT / p.94"],
        "low_confidence":      False,
        "benford_chi2":        2.3,
        "benford_p_value":     0.87,
        "forensic_flags":      [],
        "sniper_hit":          True,
        "sniper_confidence":   0.97,
        "retrieval_stage_2": [
            {"text": "Total net sales were $383,285 million.",
             "section": "INCOME_STATEMENT", "page": 94},
        ],
    }
    defaults.update(kwargs)
    return BAState(**defaults)


# Group 1: Constants
class TestConstants:

    def test_01_default_output_dir_defined(self):
        assert DEFAULT_OUTPUT_DIR

    def test_02_company_name_defined(self):
        assert "FinBench" in COMPANY_NAME

    def test_03_version_defined(self):
        assert "PDR-BAAAI" in REPORT_VERSION

    def test_04_colours_defined(self):
        assert COL_PRIMARY.startswith("#")
        assert COL_SECONDARY.startswith("#")


# Group 2: Instantiation
class TestInstantiation:

    def test_05_creates_with_defaults(self, out_dir):
        gen = PDFReportGenerator(output_dir=out_dir)
        assert gen is not None

    def test_06_output_dir_created(self, out_dir):
        PDFReportGenerator(output_dir=out_dir)
        assert os.path.isdir(out_dir)


# Group 3: Utility methods
class TestUtilities:

    def test_07_safe_escapes_html(self):
        assert PDFReportGenerator._safe("<script>") == "&lt;script&gt;"

    def test_08_safe_handles_none(self):
        assert PDFReportGenerator._safe(None) == ""

    def test_09_safe_handles_ampersand(self):
        assert "&amp;" in PDFReportGenerator._safe("A & B")

    def test_10_truncate_short_text_unchanged(self):
        assert PDFReportGenerator._truncate("short", 100) == "short"

    def test_11_truncate_long_text_ellipsis(self):
        result = PDFReportGenerator._truncate("a" * 200, 50)
        assert result.endswith("...")
        assert len(result) == 50

    def test_12_confidence_label_high(self):
        assert "HIGH" in PDFReportGenerator._confidence_label(0.92)

    def test_13_confidence_label_medium(self):
        assert "MEDIUM" in PDFReportGenerator._confidence_label(0.72)

    def test_14_confidence_label_low(self):
        assert "LOW" in PDFReportGenerator._confidence_label(0.40)

    def test_15_risk_label_low(self):
        assert "Low" in PDFReportGenerator._risk_label(15)

    def test_16_risk_label_moderate(self):
        assert "Moderate" in PDFReportGenerator._risk_label(45)

    def test_17_risk_label_elevated(self):
        assert "Elevated" in PDFReportGenerator._risk_label(80)

    def test_18_retrieval_path_sniper(self):
        s = _make_state(sniper_hit=True)
        path = PDFReportGenerator._retrieval_path(s)
        assert "SniperRAG" in path

    def test_19_retrieval_path_cascade(self):
        s = _make_state(sniper_hit=False)
        path = PDFReportGenerator._retrieval_path(s)
        assert "BM25" in path


# Group 4: Full generation
class TestFullGeneration:

    def test_20_generates_pdf_file(self, gen):
        state = _make_state()
        path  = gen.generate(state)
        assert os.path.exists(path)
        assert path.endswith(".pdf")

    def test_21_pdf_has_content(self, gen):
        state = _make_state()
        path  = gen.generate(state)
        size  = os.path.getsize(path)
        # A multi-page PDF with charts is >30KB minimum
        assert size > 30_000

    def test_22_filename_includes_company(self, gen):
        state = _make_state(company_name="MyTestCo")
        path  = gen.generate(state)
        assert "MyTestCo" in os.path.basename(path)

    def test_23_filename_includes_session_id(self, gen):
        state = _make_state(session_id="unique-abc-123")
        path  = gen.generate(state)
        # PDF generator truncates session_id to 12 chars
        assert "unique-abc-1" in os.path.basename(path)

    def test_24_pdf_starts_with_header(self, gen):
        state = _make_state()
        path  = gen.generate(state)
        with open(path, "rb") as f:
            header = f.read(4)
        assert header == b"%PDF"

    def test_25_handles_empty_state(self, gen):
        state = BAState(session_id="t25")
        # Should not crash even with sparse state
        path = gen.generate(state)
        assert os.path.exists(path)

    def test_26_handles_low_confidence(self, gen):
        state = _make_state(confidence_score=0.45, low_confidence=True)
        path = gen.generate(state)
        assert os.path.exists(path)

    def test_27_handles_forensic_flags(self, gen):
        state = _make_state(forensic_flags=[
            "Benford chi-square unusually high",
            "Isolation forest flagged 3 outliers",
        ])
        path = gen.generate(state)
        assert os.path.exists(path)


# Group 5: BAState integration
class TestBAStateIntegration:

    def test_28_run_sets_final_report_path(self, gen):
        state  = _make_state()
        result = gen.run(state)
        assert result.final_report_path
        assert result.final_report_path.endswith(".pdf")

    def test_29_seed_unchanged(self, gen):
        state  = _make_state()
        result = gen.run(state)
        assert result.seed == 42


# Group 6: Convenience wrapper
class TestWrapper:

    def test_30_wrapper_generates_pdf(self, out_dir):
        state = _make_state()
        path  = generate_pdf_report(state, output_dir=out_dir)
        assert os.path.exists(path)