"""
tests/test_output_generator.py
FinBench Multi-Agent Business Analyst AI

Tests for N19 -- Output Generator

24 tests covering:
  - Instantiation (tests 01-02)
  - DOCX generation (tests 03-08)
  - Report sections (tests 09-14)
  - C9 enforcement (tests 15-17)
  - BAState integration (tests 18-21)
  - Edge cases (tests 22-24)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from src.output.output_generator import OutputGenerator
from src.state.ba_state import (
    BAState, QueryType, Difficulty, PIVStatus
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_generator(tmp_path) -> OutputGenerator:
    return OutputGenerator(output_dir=tmp_path)

def make_full_state() -> BAState:
    return BAState(
        session_id           = "test-n19-full",
        query                = "What was Apple net income FY2023?",
        query_type           = QueryType.NUMERICAL,
        company_name         = "Apple Inc",
        doc_type             = "10-K",
        fiscal_year          = "FY2023",
        final_answer         = "Net income was $96,995 million in FY2023 "
                               "[Financial Statements/P42].",
        xgb_ranked_answer    = "Net income was $96,995 million in FY2023.",
        confidence_score     = 0.92,
        agreement_status     = "unanimous|analyst",
        analyst_output       = "Net income $96,995M FY2023 [FS/42].",
        analyst_confidence   = 0.92,
        analyst_citations    = ["Financial Statements / Page 42",
                                "Income Statement / Page 44"],
        quant_result         = "Net income $96,995M [FS/42].",
        quant_confidence     = 0.88,
        quant_citations      = ["Financial Statements / Page 42"],
        auditor_output       = "Net income $96,995M FY2023 confirmed.",
        auditor_confidence   = 0.90,
        auditor_citations    = ["Financial Statements / Page 42"],
        monte_carlo_results  = {
            "mean": 96995.0, "std": 4849.0,
            "p5": 88876.0,   "p25": 93500.0,
            "p50": 96995.0,  "p75": 100400.0,
            "p95": 104962.0, "n": 10000,
        },
        var_result           = {"var_95": 64793.0, "var_99": 58887.0},
        risk_score           = 18.5,
        anomaly_severity     = "low",
        anomaly_detected     = False,
        forensic_flags       = ["ISOLATION_FOREST_MEDIUM: 2/15 outliers"],
        benford_chi2         = 2.93,
        benford_p_value      = 0.938,
        shap_values          = {
            "bm25_score":       0.42,
            "cosine_sim":       0.38,
            "section_relevance":0.12,
            "citation_present": 0.08,
        },
        contradiction_flags  = [],
        piv_round            = 0,
    )


# ════════════════════════════════════════════════════════════════════════════
# GROUP 1 -- INSTANTIATION (tests 01-02)
# ════════════════════════════════════════════════════════════════════════════

class TestInstantiation:

    def test_01_generator_instantiates(self, tmp_path):
        """N19: OutputGenerator must instantiate without error"""
        gen = make_generator(tmp_path)
        assert gen is not None

    def test_02_output_dir_created(self, tmp_path):
        """N19: Output directory must be created on instantiation"""
        gen = make_generator(tmp_path)
        assert gen.output_dir.exists()


# ════════════════════════════════════════════════════════════════════════════
# GROUP 2 -- DOCX GENERATION (tests 03-08)
# ════════════════════════════════════════════════════════════════════════════

class TestDOCXGeneration:

    def test_03_run_returns_bastate(self, tmp_path):
        """N19: run() must return BAState"""
        gen   = make_generator(tmp_path)
        state = make_full_state()
        result = gen.run(state)
        assert isinstance(result, BAState)

    def test_04_report_path_set(self, tmp_path):
        """N19: run() must set final_report_path"""
        gen   = make_generator(tmp_path)
        state = make_full_state()
        state = gen.run(state)
        assert state.final_report_path is not None
        assert state.final_report_path != ""

    def test_05_report_file_exists(self, tmp_path):
        """N19: Generated report file must exist on disk"""
        gen   = make_generator(tmp_path)
        state = make_full_state()
        state = gen.run(state)
        assert Path(state.final_report_path).exists()

    def test_06_report_is_docx_or_txt(self, tmp_path):
        """N19: Report must be .docx or .txt format"""
        gen   = make_generator(tmp_path)
        state = make_full_state()
        state = gen.run(state)
        suffix = Path(state.final_report_path).suffix
        assert suffix in [".docx", ".txt"]

    def test_07_report_not_empty(self, tmp_path):
        """N19: Report file must not be empty"""
        gen   = make_generator(tmp_path)
        state = make_full_state()
        state = gen.run(state)
        size  = Path(state.final_report_path).stat().st_size
        assert size > 0

    def test_08_generate_report_convenience_method(self, tmp_path):
        """N19: generate_report() must return path string"""
        gen   = make_generator(tmp_path)
        state = make_full_state()
        path  = gen.generate_report(state)
        assert isinstance(path, str)
        assert path != ""


# ════════════════════════════════════════════════════════════════════════════
# GROUP 3 -- REPORT SECTIONS (tests 09-14)
# ════════════════════════════════════════════════════════════════════════════

class TestReportSections:

    def test_09_plain_text_contains_answer(self, tmp_path):
        """N19: Report must contain the final answer"""
        gen   = make_generator(tmp_path)
        state = make_full_state()
        state = gen.run(state)
        path  = Path(state.final_report_path)
        if path.suffix == ".txt":
            content = path.read_text(encoding="utf-8")
            assert "96,995" in content or "96995" in content

    def test_10_plain_text_fallback_works(self, tmp_path):
        """N19: Plain text fallback must work"""
        gen   = make_generator(tmp_path)
        state = make_full_state()
        path  = gen._save_plain_text(state)
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert len(content) > 50

    def test_11_plain_text_no_rlef(self, tmp_path):
        """C9: Plain text fallback must not contain _rlef_"""
        gen   = make_generator(tmp_path)
        state = make_full_state()
        path  = gen._save_plain_text(state)
        content = path.read_text(encoding="utf-8")
        assert "_rlef_" not in content

    def test_12_report_name_contains_session_id(self, tmp_path):
        """N19: Report filename must contain session ID"""
        gen   = make_generator(tmp_path)
        state = make_full_state()
        state = gen.run(state)
        name  = Path(state.final_report_path).name
        assert "test-n19" in name or "report_" in name

    def test_13_low_confidence_state_generates_report(self, tmp_path):
        """N19: Low confidence state must still generate report"""
        gen   = make_generator(tmp_path)
        state = make_full_state()
        state.low_confidence  = True
        state.confidence_score = 0.45
        state = gen.run(state)
        assert state.final_report_path is not None

    def test_14_empty_forensic_flags_handled(self, tmp_path):
        """N19: Empty forensic_flags must not crash generator"""
        gen   = make_generator(tmp_path)
        state = make_full_state()
        state.forensic_flags = []
        state = gen.run(state)
        assert Path(state.final_report_path).exists()


# ════════════════════════════════════════════════════════════════════════════
# GROUP 4 -- C9 ENFORCEMENT (tests 15-17)
# ════════════════════════════════════════════════════════════════════════════

class TestC9Enforcement:

    def test_15_no_rlef_in_txt_output(self, tmp_path):
        """C9: _rlef_ must never appear in plain text output"""
        gen   = make_generator(tmp_path)
        state = make_full_state()
        path  = gen._save_plain_text(state)
        content = path.read_text(encoding="utf-8")
        assert "_rlef_" not in content

    def test_16_assert_no_rlef_passes_on_clean_file(self, tmp_path):
        """C9: _assert_no_rlef must pass on clean file"""
        gen       = make_generator(tmp_path)
        clean_file = tmp_path / "clean.txt"
        clean_file.write_text("This is a clean answer with no private fields.")
        gen._assert_no_rlef(str(clean_file))   # must not raise

    def test_17_assert_no_rlef_raises_on_violation(self, tmp_path):
        """C9: _assert_no_rlef must raise on _rlef_ in file"""
        gen        = make_generator(tmp_path)
        bad_file   = tmp_path / "bad.txt"
        bad_file.write_text("Answer: good. _rlef_grade: 4")
        with pytest.raises(AssertionError):
            gen._assert_no_rlef(str(bad_file))


# ════════════════════════════════════════════════════════════════════════════
# GROUP 5 -- BASTATE INTEGRATION (tests 18-21)
# ════════════════════════════════════════════════════════════════════════════

class TestBAStateIntegration:

    def test_18_run_writes_final_report_path(self, tmp_path):
        """N19: run() must write final_report_path to BAState"""
        gen   = make_generator(tmp_path)
        state = make_full_state()
        state = gen.run(state)
        assert state.final_report_path is not None
        assert isinstance(state.final_report_path, str)

    def test_19_seed_unchanged_after_run(self, tmp_path):
        """C5: BAState seed must still be 42 after N19"""
        gen   = make_generator(tmp_path)
        state = make_full_state()
        state = gen.run(state)
        assert state.seed == 42

    def test_20_final_answer_unchanged(self, tmp_path):
        """N19: run() must not modify final_answer"""
        gen          = make_generator(tmp_path)
        state        = make_full_state()
        original_ans = state.final_answer
        state        = gen.run(state)
        assert state.final_answer == original_ans

    def test_21_empty_state_handled(self, tmp_path):
        """N19: Minimal BAState must produce a report"""
        gen   = make_generator(tmp_path)
        state = BAState(
            session_id   = "t21-minimal",
            query        = "What was revenue?",
            final_answer = "Revenue was $383,285 million FY2023.",
        )
        state = gen.run(state)
        assert state.final_report_path is not None


# ════════════════════════════════════════════════════════════════════════════
# GROUP 6 -- EDGE CASES (tests 22-24)
# ════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_22_multiple_reports_different_names(self, tmp_path):
        """N19: Two different sessions must produce different filenames"""
        gen    = make_generator(tmp_path)
        state1 = make_full_state()
        state2 = make_full_state()
        state2.session_id = "test-n19-second"
        state1 = gen.run(state1)
        state2 = gen.run(state2)
        assert state1.final_report_path != state2.final_report_path

    def test_23_state_with_no_quant_results(self, tmp_path):
        """N19: State without quant results must not crash"""
        gen   = make_generator(tmp_path)
        state = make_full_state()
        state.monte_carlo_results = None
        state.var_result          = None
        state.garch_result        = None
        state = gen.run(state)
        assert Path(state.final_report_path).exists()

    def test_24_state_with_no_shap_values(self, tmp_path):
        """N19: State without SHAP values must not crash"""
        gen   = make_generator(tmp_path)
        state = make_full_state()
        state.shap_values        = None
        state.feature_importance = None
        state.causal_dag_path    = None
        state = gen.run(state)
        assert Path(state.final_report_path).exists()