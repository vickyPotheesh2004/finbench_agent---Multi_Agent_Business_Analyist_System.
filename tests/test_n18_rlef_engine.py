"""
tests/test_n18_rlef_engine.py
Tests for N18 RLEF JEE Engine
PDR-BAAAI-001 · Rev 1.0
"""

import os
import sqlite3
import pytest
from src.rlef.jee_engine import (
    RLEFJEEEngine,
    RLEFGrade,
    DPOPair,
    run_rlef_engine,
    GRADE_CORRECT,
    GRADE_PARTIAL,
    GRADE_WRONG,
    MIN_GRADE_FOR_DPO,
    COMPLETENESS_CORRECT_CHARS,
    COMPLETENESS_PARTIAL_CHARS,
)
from src.state.ba_state import BAState


@pytest.fixture
def engine(tmp_path):
    return RLEFJEEEngine(db_path=str(tmp_path / "test_rlef.db"))


@pytest.fixture
def populated_engine(tmp_path):
    """Engine with 10 graded sessions."""
    eng = RLEFJEEEngine(db_path=str(tmp_path / "populated.db"))
    for i in range(10):
        eng.grade(
            session_id = f"sess_{i:03d}",
            query      = "What was Apple net income FY2023?",
            answer     = (
                f"Net income was $96,995 million in FY2023 "
                f"[INCOME_STATEMENT/P94]. Session {i} grade test."
            ),
            citations  = ["INCOME_STATEMENT / PAGE 94"],
            confidence = 0.9,
        )
    return eng


# ── Group 1: Constants ────────────────────────────────────────────────────────

class TestConstants:

    def test_01_grade_correct_is_4(self):
        assert GRADE_CORRECT == 4

    def test_02_grade_partial_is_2(self):
        assert GRADE_PARTIAL == 2

    def test_03_grade_wrong_is_minus_1(self):
        assert GRADE_WRONG == -1

    def test_04_min_grade_for_dpo_is_0(self):
        assert MIN_GRADE_FOR_DPO == 0

    def test_05_completeness_thresholds_defined(self):
        assert COMPLETENESS_CORRECT_CHARS > COMPLETENESS_PARTIAL_CHARS
        assert COMPLETENESS_CORRECT_CHARS <= 100


# ── Group 2: Database setup ───────────────────────────────────────────────────

class TestDatabaseSetup:

    def test_06_db_created_on_init(self, tmp_path):
        db  = str(tmp_path / "new.db")
        eng = RLEFJEEEngine(db_path=db)
        assert os.path.exists(db)

    def test_07_table_exists(self, engine):
        conn   = sqlite3.connect(engine.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='rlef_sessions'"
        )
        result = cursor.fetchone()
        conn.close()
        assert result is not None

    def test_08_session_count_starts_zero(self, engine):
        assert engine.get_session_count() == 0


# ── Group 3: Grading ──────────────────────────────────────────────────────────

class TestGrading:

    def test_09_grade_returns_rlefgrade(self, engine):
        result = engine.grade(
            session_id = "s001",
            query      = "What was net income?",
            answer     = "Net income was $96,995 million [INCOME_STATEMENT/P94].",
            citations  = ["INCOME_STATEMENT / PAGE 94"],
        )
        assert isinstance(result, RLEFGrade)

    def test_10_correct_answer_gets_high_grade(self, engine):
        result = engine.grade(
            session_id = "s002",
            query      = "What was Apple revenue FY2023?",
            answer     = (
                "Total net sales were $383,285 million in FY2023 "
                "[INCOME_STATEMENT/P94]. This represents an 8 percent "
                "increase compared to the prior fiscal year."
            ),
            citations  = ["INCOME_STATEMENT / PAGE 94"],
        )
        assert result.total_grade >= 6

    def test_11_empty_answer_gets_wrong_grade(self, engine):
        result = engine.grade(
            session_id = "s003",
            query      = "What was revenue?",
            answer     = "",
        )
        assert result.grade_label == "wrong"

    def test_12_retrieval_miss_gets_wrong(self, engine):
        result = engine.grade(
            session_id = "s004",
            query      = "What was revenue?",
            answer     = "RETRIEVAL_MISS: revenue data not found.",
        )
        assert result.vc_score == GRADE_WRONG

    def test_13_grade_has_all_fields(self, engine):
        result = engine.grade("s005", "query", "answer with text here ok")
        assert hasattr(result, "va_score")
        assert hasattr(result, "vb_score")
        assert hasattr(result, "vc_score")
        assert hasattr(result, "total_grade")
        assert hasattr(result, "grade_label")
        assert hasattr(result, "session_id")
        assert hasattr(result, "timestamp_utc")

    def test_14_grade_label_is_valid(self, engine):
        result = engine.grade("s006", "query", "answer text here ok")
        assert result.grade_label in ["correct", "partial", "wrong"]

    def test_15_va_score_is_valid(self, engine):
        result = engine.grade("s007", "revenue query", "answer")
        assert result.va_score in [GRADE_CORRECT, GRADE_PARTIAL, GRADE_WRONG]

    def test_16_vb_score_is_valid(self, engine):
        result = engine.grade("s008", "query", "answer")
        assert result.vb_score in [GRADE_CORRECT, GRADE_PARTIAL, GRADE_WRONG]

    def test_17_vc_score_is_valid(self, engine):
        result = engine.grade("s009", "query", "answer")
        assert result.vc_score in [GRADE_CORRECT, GRADE_PARTIAL, GRADE_WRONG]

    def test_18_grade_stores_in_db(self, engine):
        """grade() must store result automatically."""
        before = engine.get_session_count()
        engine.grade("s010", "query", "answer text here is long enough ok yes")
        after  = engine.get_session_count()
        assert after == before + 1


# ── Group 4: Numerical precision validator ────────────────────────────────────

class TestNumericalPrecision:

    def test_19_number_with_units_gets_correct(self):
        score = RLEFJEEEngine._grade_numerical_precision(
            "Net income was $96,995 million", "what was net income", ""
        )
        assert score == GRADE_CORRECT

    def test_20_number_without_units_gets_partial(self):
        score = RLEFJEEEngine._grade_numerical_precision(
            "Net income was 96995", "what was net income", ""
        )
        assert score == GRADE_PARTIAL

    def test_21_no_number_numerical_question_gets_wrong(self):
        score = RLEFJEEEngine._grade_numerical_precision(
            "The company performed well", "what was total revenue", ""
        )
        assert score == GRADE_WRONG

    def test_22_gold_match_gets_correct(self):
        score = RLEFJEEEngine._grade_numerical_precision(
            "Revenue was 383285 million", "revenue", "383285"
        )
        assert score == GRADE_CORRECT


# ── Group 5: Citation quality validator ───────────────────────────────────────

class TestCitationQuality:

    def test_23_section_page_citation_gets_correct(self):
        score = RLEFJEEEngine._grade_citation_quality(
            "Revenue was $383B [INCOME_STATEMENT/P94].",
            ["INCOME_STATEMENT / PAGE 94"],
        )
        assert score == GRADE_CORRECT

    def test_24_no_citation_gets_wrong(self):
        score = RLEFJEEEngine._grade_citation_quality(
            "Revenue was 383 billion.", []
        )
        assert score == GRADE_WRONG

    def test_25_bracket_without_section_gets_partial(self):
        score = RLEFJEEEngine._grade_citation_quality(
            "Revenue [see filing].", []
        )
        assert score == GRADE_PARTIAL


# ── Group 6: Completeness validator ──────────────────────────────────────────

class TestCompleteness:

    def test_26_long_answer_gets_correct(self):
        # >= COMPLETENESS_CORRECT_CHARS (75) chars → CORRECT
        answer = "Net income was $96,995 million in FY2023 per income statement on page 94 of the annual report."
        assert len(answer) >= COMPLETENESS_CORRECT_CHARS
        score  = RLEFJEEEngine._grade_completeness(answer, "net income")
        assert score == GRADE_CORRECT

    def test_27_medium_answer_gets_partial(self):
        # >= 30 but < 75 chars → PARTIAL
        answer = "Net income was $96,995 million."
        assert COMPLETENESS_PARTIAL_CHARS <= len(answer) < COMPLETENESS_CORRECT_CHARS
        score  = RLEFJEEEngine._grade_completeness(answer, "net income")
        assert score == GRADE_PARTIAL

    def test_28_short_answer_gets_wrong(self):
        score = RLEFJEEEngine._grade_completeness("Yes.", "what was revenue")
        assert score == GRADE_WRONG

    def test_29_empty_answer_gets_wrong(self):
        score = RLEFJEEEngine._grade_completeness("", "query")
        assert score == GRADE_WRONG

    def test_30_retrieval_miss_gets_wrong(self):
        score = RLEFJEEEngine._grade_completeness(
            "RETRIEVAL_MISS: not found", "query"
        )
        assert score == GRADE_WRONG


# ── Group 7: DPO pair extraction ─────────────────────────────────────────────

class TestDPOExtraction:

    def test_31_extract_returns_list(self, populated_engine):
        pairs = populated_engine.extract_dpo_pairs()
        assert isinstance(pairs, list)

    def test_32_pairs_have_required_fields(self, populated_engine):
        pairs = populated_engine.extract_dpo_pairs()
        for p in pairs:
            assert hasattr(p, "session_id")
            assert hasattr(p, "query")
            assert hasattr(p, "chosen")
            assert hasattr(p, "rejected")
            assert hasattr(p, "grade_delta")

    def test_33_session_count_increases_on_grade(self, engine):
        before = engine.get_session_count()
        engine.grade(
            "s_new", "query",
            "answer text here is long enough ok yes it is",
        )
        after = engine.get_session_count()
        assert after == before + 1

    def test_34_grade_distribution_returns_dict(self, populated_engine):
        dist = populated_engine.get_grade_distribution()
        assert isinstance(dist, dict)

    def test_35_same_session_id_overwrites(self, engine):
        engine.grade(
            "same_id", "query",
            "answer one two three four five six seven eight nine"
        )
        engine.grade(
            "same_id", "query",
            "answer updated version with more text here ok yes"
        )
        assert engine.get_session_count() == 1


# ── Group 8: BAState integration ─────────────────────────────────────────────

class TestBAStateIntegration:

    def test_36_run_writes_rlef_grade(self, tmp_path):
        eng   = RLEFJEEEngine(db_path=str(tmp_path / "t36.db"))
        state = BAState(
            session_id           = "t36",
            query                = "What was net income FY2023?",
            final_answer_pre_xgb = (
                "Net income was $96,995 million [INCOME_STATEMENT/P94]. "
                "This is the figure from the income statement page 94."
            ),
            analyst_citations    = ["INCOME_STATEMENT / PAGE 94"],
            confidence_score     = 0.95,
        )
        state = eng.run(state)
        assert state._rlef_grade != 0

    def test_37_run_writes_va_score(self, tmp_path):
        eng   = RLEFJEEEngine(db_path=str(tmp_path / "t37.db"))
        state = BAState(
            session_id           = "t37",
            query                = "revenue",
            final_answer_pre_xgb = "Revenue was $383,285 million [P94].",
        )
        state = eng.run(state)
        assert state._rlef_va_score in [
            float(GRADE_CORRECT), float(GRADE_PARTIAL), float(GRADE_WRONG)
        ]

    def test_38_rlef_fields_not_in_public_fields(self, tmp_path):
        """C9: _rlef_ fields must never appear in non-private output."""
        eng   = RLEFJEEEngine(db_path=str(tmp_path / "t38.db"))
        state = BAState(
            session_id           = "t38",
            query                = "revenue",
            final_answer_pre_xgb = "Revenue was $383,285 million [P94].",
        )
        state = eng.run(state)
        public = state.model_dump(exclude={
            "_rlef_grade", "_rlef_va_score", "_rlef_vb_score",
            "_rlef_vc_score", "_rlef_chosen", "_rlef_rejected",
            "_rlef_user_consented", "_rlef_stored_global",
        })
        for key in public:
            assert not key.startswith("_rlef_"), f"_rlef_ leaked: {key}"

    def test_39_seed_unchanged(self, tmp_path):
        eng   = RLEFJEEEngine(db_path=str(tmp_path / "t39.db"))
        state = BAState(
            session_id           = "t39",
            query                = "revenue",
            final_answer_pre_xgb = "Revenue was $383B.",
        )
        state = eng.run(state)
        assert state.seed == 42

    def test_40_empty_answer_skips_grading(self, tmp_path):
        eng    = RLEFJEEEngine(db_path=str(tmp_path / "t40.db"))
        state  = BAState(session_id="t40", query="revenue")
        before = eng.get_session_count()
        state  = eng.run(state)
        after  = eng.get_session_count()
        assert after == before


# ── Group 9: Convenience wrapper ─────────────────────────────────────────────

class TestConvenienceWrapper:

    def test_41_run_rlef_engine_returns_state(self, tmp_path):
        db    = str(tmp_path / "wrap.db")
        state = BAState(
            session_id           = "t41",
            query                = "What was net income?",
            final_answer_pre_xgb = (
                "Net income was $96,995 million [INCOME_STATEMENT/P94]. "
                "Sourced from the consolidated income statement."
            ),
            analyst_citations    = ["INCOME_STATEMENT / PAGE 94"],
        )
        result = run_rlef_engine(state, db_path=db)
        assert hasattr(result, "_rlef_grade")
        assert result.seed == 42