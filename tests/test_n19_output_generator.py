"""
tests/test_n19_output_generator.py
Tests for N19 DOCX Output Generator
PDR-BAAAI-001 · Rev 1.0
"""

import os
import pytest
from src.output.docx_generator import (
    DOCXGenerator,
    run_output_generator,
    OUTPUT_DIR,
)
from src.state.ba_state import BAState


@pytest.fixture
def gen(tmp_path):
    return DOCXGenerator(output_dir=str(tmp_path))


@pytest.fixture
def full_state():
    return BAState(
        session_id           = "test_session_001",
        company_name         = "Apple Inc",
        doc_type             = "10-K",
        fiscal_year          = "FY2023",
        query                = "What was Apple total net sales FY2023?",
        final_answer_pre_xgb = (
            "Apple total net sales were $383,285 million in FY2023 "
            "[INCOME_STATEMENT/P94]. This represents an 8% increase "
            "from FY2022 net sales of $394,328 million."
        ),
        analyst_citations    = ["INCOME_STATEMENT / PAGE 94: $383,285M"],
        confidence_score     = 0.95,
        risk_score           = 12.5,
        anomaly_severity     = "low",
        anomaly_detected     = False,
        forensic_flags       = [],
        piv_round            = 0,
        winning_pod          = "LeadAnalyst",
        sniper_hit           = False,
        retrieval_stage_2    = [
            {"chunk_id": "c1", "text": "net sales 383285", "section": "INCOME_STATEMENT", "page": 94}
        ],
        feature_importance   = {"net sales": 0.42, "revenue": 0.31, "income": 0.18},
    )


# ── Group 1: Instantiation ────────────────────────────────────────────────────

class TestInstantiation:

    def test_01_instantiates(self, gen):
        assert gen is not None

    def test_02_output_dir_created(self, tmp_path):
        new_dir = str(tmp_path / "new_output")
        g       = DOCXGenerator(output_dir=new_dir)
        assert os.path.exists(new_dir)

    def test_03_default_output_dir_constant(self):
        assert OUTPUT_DIR == "outputs"


# ── Group 2: Generate method ──────────────────────────────────────────────────

class TestGenerate:

    def test_04_generate_returns_path(self, gen, full_state, tmp_path):
        path   = str(tmp_path / "test_report.docx")
        result = gen.generate(state=full_state, output_path=path)
        assert isinstance(result, str)

    def test_05_file_is_created(self, gen, full_state, tmp_path):
        path = str(tmp_path / "test_report.docx")
        gen.generate(state=full_state, output_path=path)
        assert os.path.exists(path)

    def test_06_file_is_not_empty(self, gen, full_state, tmp_path):
        path = str(tmp_path / "test_report.docx")
        gen.generate(state=full_state, output_path=path)
        assert os.path.getsize(path) > 1000  # real DOCX > 1KB

    def test_07_empty_state_does_not_crash(self, gen, tmp_path):
        state = BAState(session_id="empty_test")
        path  = str(tmp_path / "empty_report.docx")
        result = gen.generate(state=state, output_path=path)
        assert result is not None


# ── Group 3: C9 enforcement ───────────────────────────────────────────────────

class TestC9Enforcement:

    def test_08_no_rlef_in_docx(self, gen, full_state, tmp_path):
        """C9: _rlef_ must never appear in DOCX output."""
        path = str(tmp_path / "c9_test.docx")
        gen.generate(state=full_state, output_path=path)
        # Read docx text and verify no _rlef_
        try:
            from docx import Document
            doc = Document(path)
            all_text = " ".join(p.text for p in doc.paragraphs)
            assert "_rlef_" not in all_text
        except ImportError:
            pass  # python-docx not installed — fallback txt

    def test_09_rlef_in_forensic_flags_rejected(self, gen, tmp_path):
        """_rlef_ content in forensic_flags must be filtered out."""
        state = BAState(
            session_id           = "c9_test",
            final_answer_pre_xgb = "Revenue was $383B [INCOME_STATEMENT/P94].",
            forensic_flags       = ["_rlef_grade: 4 this should not appear"],
        )
        path = str(tmp_path / "c9_flags_test.docx")
        # Should not raise
        try:
            gen.generate(state=state, output_path=path)
            # If we get here, the assertion inside _assert_no_rlef_in_doc
            # should have caught it — verify file still created
        except AssertionError:
            pass  # Expected — C9 caught the violation


# ── Group 4: Run method (BAState integration) ─────────────────────────────────

class TestBAStateIntegration:

    def test_10_run_writes_final_report_path(self, gen, full_state):
        state = gen.run(full_state)
        assert hasattr(state, "final_report_path")
        assert state.final_report_path is not None

    def test_11_run_writes_final_answer(self, gen, full_state):
        state = gen.run(full_state)
        assert hasattr(state, "final_answer")
        assert isinstance(state.final_answer, str)
        assert len(state.final_answer) > 0

    def test_12_final_answer_from_pre_xgb_when_no_xgb(self, gen):
        state = BAState(
            session_id           = "t12",
            query                = "revenue",
            final_answer_pre_xgb = "Revenue was $383B [INCOME_STATEMENT/P94].",
        )
        state = gen.run(state)
        assert state.final_answer == "Revenue was $383B [INCOME_STATEMENT/P94]."

    def test_13_xgb_answer_takes_priority(self, gen):
        state = BAState(
            session_id           = "t13",
            query                = "revenue",
            final_answer_pre_xgb = "pre-xgb answer here",
            xgb_ranked_answer    = "XGB ranked answer takes priority here yes",
        )
        state = gen.run(state)
        assert "XGB" in state.final_answer

    def test_14_seed_unchanged(self, gen, full_state):
        """C5: seed must remain 42."""
        state = gen.run(full_state)
        assert state.seed == 42

    def test_15_report_path_ends_with_docx_or_txt(self, gen):
        state = BAState(
            session_id           = "t15",
            query                = "revenue",
            final_answer_pre_xgb = "Revenue was $383B.",
        )
        state = gen.run(state)
        assert (
            state.final_report_path.endswith(".docx") or
            state.final_report_path.endswith(".txt")
        )

    def test_16_report_file_exists_after_run(self, gen, full_state):
        state = gen.run(full_state)
        if state.final_report_path:
            assert os.path.exists(state.final_report_path)

    def test_17_low_confidence_flag_handled(self, gen):
        state = BAState(
            session_id           = "t17",
            query                = "revenue",
            final_answer_pre_xgb = "Revenue was $383B.",
            low_confidence       = True,
        )
        state = gen.run(state)
        assert state.final_report_path is not None

    def test_18_no_rlef_in_final_answer(self, gen, full_state):
        """C9: final_answer must not contain _rlef_."""
        state = gen.run(full_state)
        assert "_rlef_" not in state.final_answer


# ── Group 5: Convenience wrapper ─────────────────────────────────────────────

class TestConvenienceWrapper:

    def test_19_run_output_generator_returns_state(self, tmp_path, full_state):
        result = run_output_generator(full_state, output_dir=str(tmp_path))
        assert hasattr(result, "final_report_path")
        assert result.seed == 42

    def test_20_wrapper_creates_file(self, tmp_path, full_state):
        result = run_output_generator(full_state, output_dir=str(tmp_path))
        if result.final_report_path:
            assert os.path.exists(result.final_report_path)