"""
tests/test_ba_state.py
Automated tests for BAState — enforce all constraints
Run: pytest tests/test_ba_state.py -v
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ba_state import BAState, PIVStatus, QueryType, Difficulty, ClarificationStatus

class TestConstraints:

    def test_c5_seed_is_42(self):
        """C5: Default seed must be 42"""
        s = BAState(session_id="t1")
        assert s.seed == 42

    def test_c5_wrong_seed_rejected(self):
        """C5: Any seed other than 42 must raise ValueError"""
        with pytest.raises(ValueError, match="C5 VIOLATION"):
            BAState(session_id="t2", seed=0)

    def test_a2_max_attempts_is_5(self):
        """A2: piv_max_attempts must be 5"""
        s = BAState(session_id="t3")
        assert s.piv_max_attempts == 5

    def test_a2_attempt_overflow_rejected(self):
        """A2: attempt_count > 5 must raise ValueError"""
        with pytest.raises(ValueError, match="A2 VIOLATION"):
            BAState(session_id="t4", analyst_attempt_count=6)

    def test_c9_rlef_not_in_safe_dict(self):
        """C9: safe_dict() must contain zero _rlef_ keys"""
        s = BAState(session_id="t5")
        safe = s.safe_dict()
        rlef_keys = [k for k in safe if k.startswith("_rlef_")]
        assert len(rlef_keys) == 0, f"C9 VIOLATION: {rlef_keys}"

    def test_c8_chunk_prefix_format(self):
        """C8: chunk prefix must contain all 5 mandatory fields"""
        s = BAState(
            session_id="t6",
            company_name="Apple Inc",
            doc_type="10-K",
            fiscal_year="FY2023"
        )
        prefix = s.chunk_metadata_prefix()
        assert "Apple Inc" in prefix
        assert "10-K" in prefix
        assert "FY2023" in prefix

    def test_a3_needs_clarification_false_by_default(self):
        """A3: needs_clarification() is False on fresh state"""
        s = BAState(session_id="t7")
        assert s.needs_clarification() is False

    def test_a3_needs_clarification_true_when_all_pods_exhausted(self):
        """A3: needs_clarification() is True when all 3 pods hit 5 REJECTs"""
        s = BAState(
            session_id="t8",
            analyst_attempt_count=5,
            analyst_piv_status=PIVStatus.REJECT,
            quant_attempt_count=5,
            quant_piv_status=PIVStatus.REJECT,
            auditor_attempt_count=5,
            auditor_piv_status=PIVStatus.REJECT,
        )
        assert s.needs_clarification() is True

    def test_a3_reset_for_clarification(self):
        """A3: reset_for_clarification() must reset all attempt counters"""
        s = BAState(
            session_id="t9",
            analyst_attempt_count=5,
            analyst_piv_status=PIVStatus.REJECT,
            quant_attempt_count=5,
            quant_piv_status=PIVStatus.REJECT,
            auditor_attempt_count=5,
            auditor_piv_status=PIVStatus.REJECT,
        )
        s.reset_for_clarification("Use FY2023 GAAP figures")
        assert s.analyst_attempt_count == 0
        assert s.quant_attempt_count   == 0
        assert s.auditor_attempt_count == 0
        assert s.analyst_piv_status    == PIVStatus.PENDING
        assert s.clarification_answer  == "Use FY2023 GAAP figures"
        assert s.clarification_round   == 1

    def test_query_type_enum(self):
        """Query type must be one of 5 valid values"""
        s = BAState(session_id="t10", query_type=QueryType.NUMERICAL)
        assert s.query_type == QueryType.NUMERICAL

    def test_difficulty_enum(self):
        """Difficulty must be easy/medium/hard"""
        s = BAState(session_id="t11", query_difficulty=Difficulty.HARD)
        assert s.query_difficulty == Difficulty.HARD