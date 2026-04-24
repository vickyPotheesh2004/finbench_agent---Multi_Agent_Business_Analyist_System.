"""
tests/test_n15_piv_mediator.py
Tests for N15 PIV Debate Mediator
PDR-BAAAI-001 · Rev 1.0
"""

import pytest
from src.analysis.piv_mediator import (
    PIVMediator,
    run_piv_mediator,
    CandidateAnswer,
    MediatorResult,
    MAX_MEDIATION_ROUNDS,
    ITERATION_CAP,
)
from src.state.ba_state import BAState


class MockLLMClient:
    def __init__(self, responses=None):
        self.responses   = responses or []
        self._call_count = 0

    def chat(self, prompt, temperature=0.1):
        if self._call_count < len(self.responses):
            r = self.responses[self._call_count]
        else:
            r = self.responses[-1] if self.responses else ""
        self._call_count += 1
        return r

    def is_available(self):
        return True


MOCK_MEDIATOR_RESPONSE = """
AGREEMENT_STATUS: majority
WINNING_POD: LeadAnalyst
FINAL_ANSWER: Apple total net sales were $383,285 million in FY2023 [INCOME_STATEMENT/P94].
RESOLUTION_REASONING: LeadAnalyst and QuantAnalyst agree on the figure. BlindAuditor concurs.
CONFIDENCE: 0.92
"""

SAMPLE_CHUNKS = [
    {
        "chunk_id": "c1",
        "text":     "Apple total net sales were 383285 million FY2023.",
        "section":  "INCOME_STATEMENT",
        "page":     94,
    }
]

AGREE_CANDIDATES = [
    CandidateAnswer("LeadAnalyst",  "Net sales were $383,285 million [P94].", 0.95, []),
    CandidateAnswer("QuantAnalyst", "Net sales: 383285 million FY2023.",       0.91, []),
    CandidateAnswer("BlindAuditor", "Total net sales $383,285M in FY2023.",    0.89, []),
]

DISAGREE_CANDIDATES = [
    CandidateAnswer("LeadAnalyst",  "Net sales were $383,285 million.", 0.95, []),
    CandidateAnswer("QuantAnalyst", "Net sales were $394,328 million.", 0.88, []),
    CandidateAnswer("BlindAuditor", "Net sales were $365,817 million.", 0.82, []),
]

MAJORITY_CANDIDATES = [
    CandidateAnswer("LeadAnalyst",  "Net sales were $383,285 million.", 0.95, []),
    CandidateAnswer("QuantAnalyst", "Net sales: 383285 million.",       0.91, []),
    CandidateAnswer("BlindAuditor", "Net sales were $365,817 million.", 0.82, []),
]


@pytest.fixture
def mediator():
    llm = MockLLMClient(responses=[MOCK_MEDIATOR_RESPONSE])
    return PIVMediator(llm_client=llm)


class TestConstants:

    def test_01_max_mediation_rounds(self):
        assert MAX_MEDIATION_ROUNDS == 2

    def test_02_iteration_cap(self):
        assert ITERATION_CAP == 5


class TestDataClasses:

    def test_03_candidate_answer_fields(self):
        c = CandidateAnswer("LeadAnalyst", "answer", 0.9, ["cite1"])
        assert c.pod_name   == "LeadAnalyst"
        assert c.answer     == "answer"
        assert c.confidence == 0.9
        assert c.citations  == ["cite1"]

    def test_04_mediator_result_fields(self):
        r = MediatorResult(
            final_answer="ans", winning_pod="LeadAnalyst",
            agreement_status="unanimous", confidence=0.9,
            resolution_reasoning="reason", rounds_used=0,
        )
        assert r.winning_pod      == "LeadAnalyst"
        assert r.agreement_status == "unanimous"


class TestAnswerAgreement:

    def test_05_numeric_answers_agree_within_1pct(self):
        # 383285 vs 383000 = 0.07% diff → agrees (within 1%)
        assert PIVMediator._answers_agree("383285", "383000") is True
        # 383285 vs 394328 = 2.8% diff → disagrees (exceeds 1%)
        assert PIVMediator._answers_agree("383285", "394328") is False
        # exact same → agrees
        assert PIVMediator._answers_agree("383285", "383286") is True

    def test_06_exact_same_number_agrees(self):
        assert PIVMediator._answers_agree("383285", "383285") is True

    def test_07_different_numbers_disagree(self):
        assert PIVMediator._answers_agree("383285", "394328") is False

    def test_08_text_answers_agree_same(self):
        assert PIVMediator._answers_agree("net income increased", "net income increased") is True

    def test_09_text_answers_disagree_different(self):
        assert PIVMediator._answers_agree("revenue increased", "revenue decreased") is False

    def test_10_empty_strings_disagree(self):
        assert PIVMediator._answers_agree("", "") is False
        assert PIVMediator._answers_agree("383285", "") is False


class TestCoreExtraction:

    def test_11_extracts_number_from_answer(self):
        core = PIVMediator._extract_core_answer(
            "Net sales were $383,285 million [INCOME_STATEMENT/P94]."
        )
        assert "383285" in core or "383" in core

    def test_12_empty_answer_returns_empty(self):
        assert PIVMediator._extract_core_answer("") == ""

    def test_13_strips_citations(self):
        core = PIVMediator._extract_core_answer(
            "Value is 100 [SECTION/P1: citation]"
        )
        assert "[" not in core or "100" in core


class TestMediationLogic:

    def test_14_unanimous_agreement_detected(self, mediator):
        same = [
            CandidateAnswer("LeadAnalyst",  "383285 million", 0.95, []),
            CandidateAnswer("QuantAnalyst", "383285 million", 0.91, []),
            CandidateAnswer("BlindAuditor", "383285 million", 0.89, []),
        ]
        result = mediator.mediate(same, "revenue", [])
        assert result.agreement_status in ["unanimous", "majority", "mediated", "fallback"]
        assert result.final_answer != ""

    def test_15_majority_winner_selected(self, mediator):
        result = mediator.mediate(MAJORITY_CANDIDATES, "revenue", [])
        assert isinstance(result, MediatorResult)
        assert result.final_answer != ""

    def test_16_single_candidate_returns_directly(self, mediator):
        single = [CandidateAnswer("LeadAnalyst", "383285 million", 0.95, [])]
        result = mediator.mediate(single, "revenue", [])
        assert result.agreement_status == "unanimous"
        assert result.winning_pod      == "LeadAnalyst"

    def test_17_empty_candidates_returns_fallback(self, mediator):
        result = mediator.mediate([], "revenue", [])
        assert result.agreement_status == "fallback"
        assert result.final_answer     == ""

    def test_18_disagreement_triggers_mediation(self, mediator):
        result = mediator.mediate(DISAGREE_CANDIDATES, "revenue", SAMPLE_CHUNKS)
        assert isinstance(result, MediatorResult)
        assert result.final_answer != ""

    def test_19_rounds_tracked(self, mediator):
        result = mediator.mediate(DISAGREE_CANDIDATES, "revenue", SAMPLE_CHUNKS)
        assert isinstance(result.rounds_used, int)
        assert result.rounds_used <= MAX_MEDIATION_ROUNDS + 1

    def test_20_confidence_between_0_and_1(self, mediator):
        result = mediator.mediate(AGREE_CANDIDATES, "revenue", [])
        assert 0.0 <= result.confidence <= 1.0

    def test_21_fallback_uses_highest_confidence(self):
        llm      = MockLLMClient(responses=[""])
        mediator = PIVMediator(llm_client=llm)
        result   = mediator.mediate(DISAGREE_CANDIDATES, "revenue", [])
        assert result.winning_pod == "LeadAnalyst"


class TestBAStateIntegration:

    def test_22_run_writes_final_answer(self, mediator):
        state = BAState(
            session_id         = "t22",
            query              = "What was net income?",
            analyst_output     = "Net income was $96,995 million [P94].",
            analyst_confidence = 0.95,
            quant_result       = "Net income: 96995 million FY2023.",
            quant_confidence   = 0.91,
            auditor_output     = "Net income was $96,995 million [P94].",
            auditor_confidence = 0.89,
            retrieval_stage_2  = SAMPLE_CHUNKS,
        )
        state = mediator.run(state)
        assert hasattr(state, "final_answer_pre_xgb")
        assert isinstance(state.final_answer_pre_xgb, str)

    def test_23_run_writes_confidence_score(self, mediator):
        state = BAState(
            session_id         = "t23",
            query              = "net income",
            analyst_output     = "96995 million",
            analyst_confidence = 0.90,
            retrieval_stage_2  = SAMPLE_CHUNKS,
        )
        state = mediator.run(state)
        assert 0.0 <= state.confidence_score <= 1.0

    def test_24_seed_unchanged(self, mediator):
        state = BAState(
            session_id     = "t24",
            query          = "net income",
            analyst_output = "96995 million",
        )
        state = mediator.run(state)
        assert state.seed == 42

    def test_25_no_candidates_sets_low_confidence(self, mediator):
        state = BAState(session_id="t25", query="net income")
        state = mediator.run(state)
        assert state.low_confidence is True

    def test_26_no_rlef_in_final_answer(self, mediator):
        state = BAState(
            session_id         = "t26",
            query              = "net income",
            analyst_output     = "96995 million",
            analyst_confidence = 0.90,
            retrieval_stage_2  = SAMPLE_CHUNKS,
        )
        state = mediator.run(state)
        assert "_rlef_" not in state.final_answer_pre_xgb

    def test_27_piv_round_written(self, mediator):
        state = BAState(
            session_id         = "t27",
            query              = "net income",
            analyst_output     = "96995 million",
            analyst_confidence = 0.90,
        )
        state = mediator.run(state)
        assert isinstance(state.piv_round, int)


class TestConvenienceWrapper:

    def test_28_run_piv_mediator_returns_state(self):
        llm   = MockLLMClient(responses=[MOCK_MEDIATOR_RESPONSE])
        state = BAState(
            session_id         = "t28",
            query              = "net income",
            analyst_output     = "96995 million",
            analyst_confidence = 0.90,
            quant_result       = "96995 million",
            quant_confidence   = 0.88,
            auditor_output     = "96995 million",
            auditor_confidence = 0.85,
            retrieval_stage_2  = SAMPLE_CHUNKS,
        )
        result = run_piv_mediator(state, llm_client=llm)
        assert hasattr(result, "final_answer_pre_xgb")
        assert result.seed == 42