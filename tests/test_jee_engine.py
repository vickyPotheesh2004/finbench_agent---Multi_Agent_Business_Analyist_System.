"""
tests/test_jee_engine.py
FinBench Multi-Agent Business Analyst AI

Tests for N18 -- RLEF JEE Engine

No LLM needed -- pure grading logic + SQLite.
All tests use tmp_path for isolated SQLite databases.

24 tests covering:
  - Instantiation (tests 01-02)
  - Validator A numerical (tests 03-06)
  - Validator B citations (tests 07-10)
  - Validator C completeness (tests 11-14)
  - DPO pair management (tests 15-18)
  - BAState integration (tests 19-22)
  - Edge cases (tests 23-24)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from src.rlef.jee_engine import (
    JEEEngine,
    MIN_GRADE_FOR_DPO,
    SCORE_EXCELLENT,
    SCORE_PARTIAL,
    SCORE_WRONG,
)
from src.state.ba_state import BAState, QueryType, Difficulty, PIVStatus


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_engine(tmp_path) -> JEEEngine:
    return JEEEngine(db_path=tmp_path / "test_rlef.db")

def make_state(
    session_id:   str       = "test-n18",
    query:        str       = "What was Apple net income FY2023?",
    query_type:   QueryType = QueryType.NUMERICAL,
    final_answer: str       = "Net income was $96,995 million in FY2023 [FS/42].",
    confidence:   float     = 0.92,
) -> BAState:
    return BAState(
        session_id         = session_id,
        query              = query,
        query_type         = query_type,
        query_difficulty   = Difficulty.MEDIUM,
        company_name       = "Apple Inc",
        fiscal_year        = "FY2023",
        final_answer       = final_answer,
        xgb_ranked_answer  = final_answer,
        confidence_score   = confidence,
        analyst_output     = final_answer,
        analyst_confidence = confidence,
        analyst_citations  = ["Financial Statements / Page 42",
                               "Income Statement / Page 44"],
        quant_result       = "Net income $57,411M FY2022.",
        quant_confidence   = 0.60,
        auditor_output     = "Net income $96,995M FY2023 confirmed.",
        auditor_confidence = 0.85,
        retrieval_stage_2  = [{
            "text":    "Net income $96,995 million FY2023. "
                       "Revenue $383,285 million.",
            "section": "Financial Statements",
            "page":    "42",
        }],
    )


# ════════════════════════════════════════════════════════════════════════════
# GROUP 1 -- INSTANTIATION (tests 01-02)
# ════════════════════════════════════════════════════════════════════════════

class TestInstantiation:

    def test_01_engine_instantiates(self, tmp_path):
        """N18: JEEEngine must instantiate without error"""
        engine = make_engine(tmp_path)
        assert engine is not None

    def test_02_db_created_on_init(self, tmp_path):
        """N18: SQLite DB must be created on instantiation"""
        engine  = make_engine(tmp_path)
        db_file = tmp_path / "test_rlef.db"
        assert db_file.exists()


# ════════════════════════════════════════════════════════════════════════════
# GROUP 2 -- VALIDATOR A NUMERICAL (tests 03-06)
# ════════════════════════════════════════════════════════════════════════════

class TestValidatorA:

    def test_03_exact_match_returns_4(self, tmp_path):
        """V-A: Exact numerical match must return +4"""
        engine = make_engine(tmp_path)
        state  = make_state()
        score  = engine.grade_numerical(
            "Net income was $96,995 million FY2023.", state
        )
        assert score == SCORE_EXCELLENT

    def test_04_wrong_number_returns_minus1(self, tmp_path):
        """V-A: Clearly wrong number must return -1"""
        engine = make_engine(tmp_path)
        state  = make_state()
        score  = engine.grade_numerical(
            "Net income was $1,000 million FY2023.", state
        )
        assert score == SCORE_WRONG

    def test_05_no_number_in_answer_returns_minus1(self, tmp_path):
        """V-A: Answer with no number must return -1 for numerical query"""
        engine = make_engine(tmp_path)
        state  = make_state(query_type=QueryType.NUMERICAL)
        score  = engine.grade_numerical(
            "The net income was substantial.", state
        )
        assert score == SCORE_WRONG

    def test_06_text_query_returns_non_wrong(self, tmp_path):
        """V-A: Text query type must return neutral score (not -1)"""
        engine = make_engine(tmp_path)
        state  = make_state(query_type=QueryType.TEXT)
        score  = engine.grade_numerical(
            "The company performed well in FY2023.", state
        )
        # Text queries get neutral treatment — not penalised for no number
        assert score in [SCORE_PARTIAL, SCORE_EXCELLENT]


# ════════════════════════════════════════════════════════════════════════════
# GROUP 3 -- VALIDATOR B CITATIONS (tests 07-10)
# ════════════════════════════════════════════════════════════════════════════

class TestValidatorB:

    def test_07_two_citations_returns_4(self, tmp_path):
        """V-B: Two valid citations must return +4"""
        engine = make_engine(tmp_path)
        state  = make_state()
        # state has 2 analyst_citations
        score  = engine.grade_citations(
            "Net income $96,995M [FS/42].", state
        )
        assert score == SCORE_EXCELLENT

    def test_08_no_citations_returns_minus1(self, tmp_path):
        """V-B: No citations must return -1"""
        engine = make_engine(tmp_path)
        state  = BAState(
            session_id   = "t08",
            query        = "test",
            final_answer = "Net income was high.",
        )
        score = engine.grade_citations("Net income was high.", state)
        assert score == SCORE_WRONG

    def test_09_inline_citation_with_page_counts(self, tmp_path):
        """V-B: Inline citation with page number must be counted"""
        engine = make_engine(tmp_path)
        state  = BAState(
            session_id        = "t09",
            query             = "test",
            analyst_citations = ["Financial Statements / Page 42"],
            final_answer      = "Net income $96,995M.",
        )
        score = engine.grade_citations(
            "Net income $96,995M [Financial Statements / Page 42].", state
        )
        assert score in [SCORE_PARTIAL, SCORE_EXCELLENT]

    def test_10_citation_with_page_counts(self, tmp_path):
        """V-B: Citation with page number must be valid"""
        engine = make_engine(tmp_path)
        state  = BAState(
            session_id        = "t10",
            query             = "test",
            analyst_citations = ["Financial Statements / Page 42"],
            final_answer      = "x",
        )
        score = engine.grade_citations("x", state)
        assert score in [SCORE_PARTIAL, SCORE_EXCELLENT]


# ════════════════════════════════════════════════════════════════════════════
# GROUP 4 -- VALIDATOR C COMPLETENESS (tests 11-14)
# ════════════════════════════════════════════════════════════════════════════

class TestValidatorC:

    def test_11_complete_answer_scores_well(self, tmp_path):
        """V-C: Complete answer with FY and keywords must score >= +2"""
        engine = make_engine(tmp_path)
        state  = make_state()
        score  = engine.grade_completeness(
            "Apple net income was $96,995 million in FY2023 "
            "[Financial Statements/P42].",
            state,
        )
        # Must score well — either partial or excellent
        assert score in [SCORE_PARTIAL, SCORE_EXCELLENT]

    def test_12_empty_answer_returns_minus1(self, tmp_path):
        """V-C: Empty answer must return -1"""
        engine = make_engine(tmp_path)
        state  = make_state()
        score  = engine.grade_completeness("", state)
        assert score == SCORE_WRONG

    def test_13_retrieval_miss_returns_minus1(self, tmp_path):
        """V-C: RETRIEVAL_MISS answer must return -1"""
        engine = make_engine(tmp_path)
        state  = make_state()
        score  = engine.grade_completeness(
            "RETRIEVAL_MISS: no relevant data found.", state
        )
        assert score == SCORE_WRONG

    def test_14_short_answer_returns_wrong(self, tmp_path):
        """V-C: Very short answer must return -1"""
        engine = make_engine(tmp_path)
        state  = make_state()
        score  = engine.grade_completeness("Yes.", state)
        assert score == SCORE_WRONG


# ════════════════════════════════════════════════════════════════════════════
# GROUP 5 -- DPO PAIR MANAGEMENT (tests 15-18)
# ════════════════════════════════════════════════════════════════════════════

class TestDPOPairManagement:

    def test_15_good_grade_stored_in_db(self, tmp_path):
        """N18: Grade >= threshold must be stored in SQLite"""
        engine = make_engine(tmp_path)
        state  = make_state()
        engine.run(state)
        count = engine.get_session_count()
        assert count >= 1

    def test_16_get_dpo_pairs_returns_list(self, tmp_path):
        """N18: get_dpo_pairs() must return list"""
        engine = make_engine(tmp_path)
        state  = make_state()
        engine.run(state)
        pairs = engine.get_dpo_pairs()
        assert isinstance(pairs, list)

    def test_17_dpo_pair_has_required_keys(self, tmp_path):
        """N18: DPO pair must have chosen + rejected keys"""
        engine = make_engine(tmp_path)
        state  = make_state()
        engine.run(state)
        pairs = engine.get_dpo_pairs()
        if pairs:
            pair = pairs[0]
            assert "chosen"   in pair
            assert "rejected" in pair
            assert "grade"    in pair

    def test_18_low_grade_not_stored(self, tmp_path):
        """N18: Grade below threshold must not create DPO pair"""
        engine = make_engine(tmp_path)
        state  = BAState(
            session_id   = "low-grade",
            query        = "What was revenue?",
            query_type   = QueryType.NUMERICAL,
            final_answer = "Yes.",
        )
        engine.run(state)
        pairs = engine.get_dpo_pairs()
        assert all(p["session_id"] != "low-grade" for p in pairs)


# ════════════════════════════════════════════════════════════════════════════
# GROUP 6 -- BASTATE INTEGRATION (tests 19-22)
# ════════════════════════════════════════════════════════════════════════════

class TestBAStateIntegration:

    def test_19_run_writes_rlef_grade(self, tmp_path):
        """N18: run() must write _rlef_grade to BAState"""
        engine = make_engine(tmp_path)
        state  = make_state()
        state  = engine.run(state)
        rlef   = state.get_rlef_fields()
        assert "_rlef_grade" in rlef
        assert isinstance(rlef["_rlef_grade"], int)

    def test_20_run_writes_all_validator_scores(self, tmp_path):
        """N18: run() must write VA + VB + VC scores"""
        engine = make_engine(tmp_path)
        state  = make_state()
        state  = engine.run(state)
        rlef   = state.get_rlef_fields()
        assert "_rlef_va_score" in rlef
        assert "_rlef_vb_score" in rlef
        assert "_rlef_vc_score" in rlef

    def test_21_c9_rlef_fields_private(self, tmp_path):
        """C9: _rlef_ fields must never appear in public BAState dict"""
        engine = make_engine(tmp_path)
        state  = make_state()
        state  = engine.run(state)
        public = state.model_dump()
        for key in public:
            assert not key.startswith("_rlef_"), \
                f"_rlef_ field leaked to public: {key}"

    def test_22_seed_unchanged_after_run(self, tmp_path):
        """C5: BAState seed must still be 42 after N18"""
        engine = make_engine(tmp_path)
        state  = make_state()
        state  = engine.run(state)
        assert state.seed == 42


# ════════════════════════════════════════════════════════════════════════════
# GROUP 7 -- EDGE CASES (tests 23-24)
# ════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_23_empty_final_answer_skips_grading(self, tmp_path):
        """N18: Empty final_answer must skip grading gracefully"""
        engine = make_engine(tmp_path)
        state  = BAState(
            session_id   = "t23-empty",
            query        = "test",
            final_answer = "",
        )
        state = engine.run(state)
        rlef  = state.get_rlef_fields()
        assert rlef["_rlef_grade"] == 0

    def test_24_grade_range_correct(self, tmp_path):
        """N18: Grade must be between -3 and +12"""
        engine = make_engine(tmp_path)
        state  = make_state()
        state  = engine.run(state)
        rlef   = state.get_rlef_fields()
        assert -3 <= rlef["_rlef_grade"] <= 12