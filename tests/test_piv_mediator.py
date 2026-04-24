"""
tests/test_piv_mediator.py
FinBench Multi-Agent Business Analyst AI

Tests for N15 -- PIV Mediator

No Ollama needed -- fallback mediation tested directly.
All LLM calls mocked where needed.

24 tests covering:
  - Instantiation (tests 01-02)
  - Agreement detection (tests 03-08)
  - Candidate collection (tests 09-11)
  - Mediation outcomes (tests 12-17)
  - BAState integration (tests 18-22)
  - Edge cases (tests 23-24)
"""

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from src.agents.piv_mediator import (
    PIVMediator,
    CONFIDENCE_UNANIMOUS,
    CONFIDENCE_MAJORITY,
    CONFIDENCE_MEDIATED,
    CONFIDENCE_FALLBACK,
    NUMERICAL_AGREEMENT_PCT,
)
from src.state.ba_state import BAState, PIVStatus


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_state_unanimous() -> BAState:
    return BAState(
        session_id         = "t-unanimous",
        query              = "What was Apple net income FY2023?",
        company_name       = "Apple Inc",
        analyst_output     = "Net income $96,995M [FS/42].",
        analyst_confidence = 0.92,
        analyst_piv_status = PIVStatus.PASS,
        quant_result       = "Net income $96,995 million [FS/42].",
        quant_confidence   = 0.88,
        quant_piv_status   = PIVStatus.PASS,
        auditor_output     = "Net income was $96,995M [FS/42].",
        auditor_confidence = 0.90,
        auditor_piv_status = PIVStatus.PASS,
        retrieval_stage_2  = [{
            "text":    "Net income $96,995M FY2023.",
            "section": "Financial Statements",
            "page":    "42",
        }],
    )

def make_state_majority() -> BAState:
    return BAState(
        session_id         = "t-majority",
        query              = "What was Apple net income FY2023?",
        analyst_output     = "Net income $96,995M FY2023.",
        analyst_confidence = 0.92,
        analyst_piv_status = PIVStatus.PASS,
        quant_result       = "Net income $96,995 million FY2023.",
        quant_confidence   = 0.88,
        quant_piv_status   = PIVStatus.PASS,
        auditor_output     = "Net income $57,411M FY2022.",
        auditor_confidence = 0.75,
        auditor_piv_status = PIVStatus.PASS,
    )

def make_state_disagree() -> BAState:
    return BAState(
        session_id         = "t-disagree",
        query              = "What was Apple net income FY2023?",
        analyst_output     = "Net income $96,995M.",
        analyst_confidence = 0.92,
        analyst_piv_status = PIVStatus.PASS,
        quant_result       = "Net income $57,411M.",
        quant_confidence   = 0.80,
        quant_piv_status   = PIVStatus.PASS,
        auditor_output     = "Net income $29,998M.",
        auditor_confidence = 0.70,
        auditor_piv_status = PIVStatus.PASS,
    )

@pytest.fixture(scope="module")
def mediator():
    return PIVMediator()


# ════════════════════════════════════════════════════════════════════════════
# GROUP 1 -- INSTANTIATION (tests 01-02)
# ════════════════════════════════════════════════════════════════════════════

class TestInstantiation:

    def test_01_mediator_instantiates(self, mediator):
        """N15: PIVMediator must instantiate without error"""
        assert mediator is not None

    def test_02_confidence_constants_correct(self, mediator):
        """N15: Confidence constants must have correct values"""
        assert CONFIDENCE_UNANIMOUS == 1.0
        assert CONFIDENCE_MAJORITY  == 0.85
        assert CONFIDENCE_MEDIATED  == 0.70
        assert CONFIDENCE_FALLBACK  == 0.55


# ════════════════════════════════════════════════════════════════════════════
# GROUP 2 -- AGREEMENT DETECTION (tests 03-08)
# ════════════════════════════════════════════════════════════════════════════

class TestAgreementDetection:

    def test_03_same_number_agrees(self, mediator):
        """N15: Same numerical value must agree"""
        assert mediator._answers_agree("$96,995M", "$96,995 million") is True

    def test_04_within_5pct_agrees(self, mediator):
        """N15: Numbers within 5% tolerance must agree"""
        assert mediator._answers_agree("$96,995M", "$99,500M") is True

    def test_05_beyond_5pct_disagrees(self, mediator):
        """N15: Numbers beyond 5% tolerance must disagree"""
        assert mediator._answers_agree("$96,995M", "$57,411M") is False

    def test_06_text_agreement_high_overlap(self, mediator):
        """N15: High word overlap must agree"""
        a = "Apple gross margin was 44.1 percent"
        b = "gross margin was 44.1 percent for Apple"
        assert mediator._answers_agree(a, b) is True

    def test_07_primary_number_extraction(self, mediator):
        """N15: _extract_primary_number must find first number"""
        assert mediator._extract_primary_number("Net income $96,995M") == 96995.0
        assert mediator._extract_primary_number("no numbers here")     is None

    def test_08_zero_values_agree(self, mediator):
        """N15: Two zero answers must agree"""
        assert mediator._answers_agree("0", "0.0") is True


# ════════════════════════════════════════════════════════════════════════════
# GROUP 3 -- CANDIDATE COLLECTION (tests 09-11)
# ════════════════════════════════════════════════════════════════════════════

class TestCandidateCollection:

    def test_09_collects_pass_candidates_only(self, mediator):
        """N15: Only PASS pods should be collected as candidates"""
        state = BAState(
            session_id         = "t09",
            analyst_output     = "answer A",
            analyst_confidence = 0.90,
            analyst_piv_status = PIVStatus.PASS,
            quant_result       = "answer Q",
            quant_confidence   = 0.80,
            quant_piv_status   = PIVStatus.REJECT,
            auditor_output     = "",
            auditor_piv_status = PIVStatus.REJECT,
        )
        candidates = mediator._collect_candidates(state)
        pods       = [c["pod"] for c in candidates]
        assert "analyst" in pods
        assert "quant"   not in pods

    def test_10_empty_state_returns_no_candidates(self, mediator):
        """N15: Empty BAState must return empty candidate list"""
        state      = BAState(session_id="t10")
        candidates = mediator._collect_candidates(state)
        assert candidates == []

    def test_11_fallback_to_reject_candidates_if_none_pass(self, mediator):
        """N15: If no PASS candidates, include REJECT ones as fallback"""
        state = BAState(
            session_id         = "t11",
            analyst_output     = "fallback answer",
            analyst_confidence = 0.60,
            analyst_piv_status = PIVStatus.REJECT,
        )
        candidates = mediator._collect_candidates(state)
        assert len(candidates) >= 1


# ════════════════════════════════════════════════════════════════════════════
# GROUP 4 -- MEDIATION OUTCOMES (tests 12-17)
# ════════════════════════════════════════════════════════════════════════════

class TestMediationOutcomes:

    def test_12_unanimous_sets_correct_status(self, mediator):
        """N15: All 3 agree → agreement_status contains unanimous"""
        state = make_state_unanimous()
        state = mediator.run(state)
        assert "unanimous" in state.agreement_status

    def test_13_unanimous_high_confidence(self, mediator):
        """N15: Unanimous agreement → confidence >= 0.8"""
        state = make_state_unanimous()
        state = mediator.run(state)
        assert state.confidence_score >= 0.80

    def test_14_majority_sets_correct_status(self, mediator):
        """N15: 2 of 3 agree → agreement_status contains majority"""
        state = make_state_majority()
        state = mediator.run(state)
        assert "majority" in state.agreement_status

    def test_15_majority_medium_confidence(self, mediator):
        """N15: Majority agreement → confidence < unanimous"""
        state_u = make_state_unanimous()
        state_m = make_state_majority()
        state_u = mediator.run(state_u)
        state_m = mediator.run(state_m)
        assert state_m.confidence_score < state_u.confidence_score

    def test_16_full_disagree_sets_correct_status(self, mediator):
        """N15: All 3 disagree → agreement_status contains full_disagree"""
        state = make_state_disagree()
        with patch.object(mediator, "_call_ollama", return_value=None):
            state = mediator.run(state)
        assert "full_disagree" in state.agreement_status

    def test_17_final_answer_always_populated(self, mediator):
        """N15: final_answer_pre_xgb must not be empty after mediation"""
        for make_fn in [make_state_unanimous, make_state_majority,
                        make_state_disagree]:
            state = make_fn()
            with patch.object(mediator, "_call_ollama", return_value=None):
                state = mediator.run(state)
            assert state.final_answer_pre_xgb != "", \
                f"Empty answer for {make_fn.__name__}"


# ════════════════════════════════════════════════════════════════════════════
# GROUP 5 -- BASTATE INTEGRATION (tests 18-22)
# ════════════════════════════════════════════════════════════════════════════

class TestBAStateIntegration:

    def test_18_run_writes_final_answer(self, mediator):
        """N15: run() must write final_answer_pre_xgb"""
        state = make_state_unanimous()
        state = mediator.run(state)
        assert state.final_answer_pre_xgb != ""
        assert isinstance(state.final_answer_pre_xgb, str)

    def test_19_run_writes_agreement_status(self, mediator):
        """N15: run() must write agreement_status"""
        state = make_state_unanimous()
        state = mediator.run(state)
        assert state.agreement_status != ""

    def test_20_run_writes_confidence_score(self, mediator):
        """N15: run() must write confidence_score 0-1"""
        state = make_state_unanimous()
        state = mediator.run(state)
        assert 0.0 <= state.confidence_score <= 1.0

    def test_21_seed_unchanged_after_run(self, mediator):
        """C5: BAState seed must still be 42 after N15"""
        state = make_state_unanimous()
        state = mediator.run(state)
        assert state.seed == 42

    def test_22_empty_state_handled(self, mediator):
        """N15: Empty BAState must set empty final_answer"""
        state = BAState(session_id="t22-empty")
        state = mediator.run(state)
        assert state.final_answer_pre_xgb == ""
        assert state.confidence_score     == 0.0


# ════════════════════════════════════════════════════════════════════════════
# GROUP 6 -- EDGE CASES (tests 23-24)
# ════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_23_single_candidate_unanimous(self, mediator):
        """N15: Single candidate must be treated as unanimous"""
        state = BAState(
            session_id         = "t23",
            query              = "test",
            analyst_output     = "Net income $96,995M",
            analyst_confidence = 0.90,
            analyst_piv_status = PIVStatus.PASS,
        )
        state = mediator.run(state)
        assert "unanimous" in state.agreement_status
        assert state.final_answer_pre_xgb == "Net income $96,995M"

    def test_24_llm_mediation_parsed_correctly(self, mediator):
        """N15: LLM response must be parsed into final answer"""
        mock_response = (
            "AGREEMENT_STATUS: full_disagree\n"
            "WINNING_POD: LeadAnalyst\n"
            "FINAL_ANSWER: Net income was $96,995 million "
            "[Financial Statements / Page 42: $96,995M].\n"
            "RESOLUTION_REASONING: LeadAnalyst answer matches context.\n"
            "CONFIDENCE: 0.85\n"
        )
        state      = make_state_disagree()
        candidates = mediator._collect_candidates(state)
        with patch.object(mediator, "_call_ollama",
                          return_value=mock_response):
            result = mediator._mediate_with_llm(state, candidates)
        assert result["answer"]      != ""
        assert result["winning_pod"] == "analyst"
        assert 0.0 <= result["confidence"] <= 1.0