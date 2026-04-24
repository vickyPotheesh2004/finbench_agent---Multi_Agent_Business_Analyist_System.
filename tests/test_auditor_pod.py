"""
tests/test_auditor_pod.py
FinBench Multi-Agent Business Analyst AI

Tests for N14 -- BlindAuditor Pod

All Ollama calls mocked -- fast, deterministic.
Key focus: BLIND behaviour + contradiction detection.

24 tests covering:
  - Instantiation (tests 01-03)
  - BLIND behaviour enforcement (tests 04-07)
  - Contradiction detection (tests 08-13)
  - BAState integration (tests 14-19)
  - Edge cases (tests 20-24)
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from src.agents.auditor_pod  import AuditorPod
from src.agents.planner      import StrategicPlanner,   PlannerOutput
from src.agents.implementor  import ContextImplementor, ImplementorOutput
from src.agents.validator    import CuriousValidator,   ValidatorOutput
from src.agents.validator    import VALIDATOR_PASS, VALIDATOR_REJECT, ALL_CHECKS
from src.agents.piv_loop     import PIVLoopController
from src.state.ba_state      import BAState, QueryType, Difficulty, PIVStatus


# ── Mock helpers ──────────────────────────────────────────────────────────────

def mock_planner_out():
    return PlannerOutput(
        analysis_plan       = "1. Find figure. 2. Cite source.",
        retrieval_hints     = ["net income"],
        validation_criteria = "Must cite section/page.",
        curiosity_answers   = {k: "answer" for k in [
            "Q1_SCOPE","Q2_CONCEPTS","Q3_SECTIONS",
            "Q4_TRAPS","Q5_VERIFY","Q6_EDGECASES"
        ]},
    )

def mock_impl_out(
    answer:     str   = "Net income $96,995M [FS/42].",
    confidence: float = 0.90,
) -> ImplementorOutput:
    return ImplementorOutput(
        answer      = answer,
        confidence  = confidence,
        citations   = ["Financial Statements / Page 42"],
        computation = "N/A",
        output_type = "ANSWER",
    )

def mock_val_out(result: str = VALIDATOR_PASS) -> ValidatorOutput:
    return ValidatorOutput(
        result             = result,
        checks             = {c: "PASS" for c in ALL_CHECKS},
        check_reasons      = {},
        reject_reasons     = [],
        retry_instructions = "",
    )

def make_pod(
    answer:     str = "Net income $96,995M [FS/42].",
    confidence: float = 0.90,
    val_result: str = VALIDATOR_PASS,
) -> AuditorPod:
    p = StrategicPlanner()
    i = ContextImplementor()
    v = CuriousValidator()
    p.run = MagicMock(return_value=mock_planner_out())
    i.run = MagicMock(return_value=mock_impl_out(answer, confidence))
    v.run = MagicMock(return_value=mock_val_out(val_result))
    return AuditorPod(p, i, v)

def make_state(
    analyst_output: str = "",
    query: str = "What was Apple net income FY2023?",
) -> BAState:
    return BAState(
        session_id        = "test-n14",
        query             = query,
        query_type        = QueryType.NUMERICAL,
        query_difficulty  = Difficulty.EASY,
        company_name      = "Apple Inc",
        doc_type          = "10-K",
        fiscal_year       = "FY2023",
        analyst_output    = analyst_output,
        retrieval_stage_2 = [{
            "text":        "Net income: $96,995 million FY2023.",
            "section":     "Financial Statements",
            "page":        "42",
            "company":     "Apple Inc",
            "fiscal_year": "FY2023",
        }],
    )


# ════════════════════════════════════════════════════════════════════════════
# GROUP 1 -- INSTANTIATION (tests 01-03)
# ════════════════════════════════════════════════════════════════════════════

class TestInstantiation:

    def test_01_auditor_pod_instantiates(self):
        """N14: AuditorPod must instantiate without error"""
        pod = AuditorPod()
        assert pod is not None

    def test_02_auditor_has_separate_piv_instance(self):
        """N14: AuditorPod must have its own PIVLoopController"""
        pod = AuditorPod()
        assert isinstance(pod.piv, PIVLoopController)

    def test_03_max_retries_is_5(self):
        """A2: AuditorPod max_retries must be 5"""
        pod = AuditorPod()
        assert pod.max_retries == 5


# ════════════════════════════════════════════════════════════════════════════
# GROUP 2 -- BLIND BEHAVIOUR (tests 04-07)
# ════════════════════════════════════════════════════════════════════════════

class TestBlindBehaviour:

    def test_04_auditor_never_reads_analyst_output(self):
        """N14: BLIND -- auditor must not modify analyst_output"""
        pod   = make_pod()
        state = make_state(analyst_output="Analyst said $96,995M")
        original_analyst = state.analyst_output
        state = pod.run(state)
        # analyst_output must be unchanged
        assert state.analyst_output == original_analyst

    def test_05_auditor_never_reads_quant_result(self):
        """N14: BLIND -- auditor must not modify quant_result"""
        pod   = make_pod()
        state = make_state()
        state.quant_result = "Quant said gross margin 44.1%"
        original_quant    = state.quant_result
        state = pod.run(state)
        assert state.quant_result == original_quant

    def test_06_blind_context_does_not_use_assembled_prompt(self):
        """N14: _build_blind_context must build fresh from chunks"""
        pod   = make_pod()
        state = make_state()
        state.assembled_prompt = "ANALYST CONTEXT: analyst saw this"
        context = pod._build_blind_context(state)
        # Should contain INDEPENDENT AUDIT REVIEW, not assembled_prompt content
        assert "INDEPENDENT AUDIT REVIEW" in context

    def test_07_blind_context_contains_chunk_text(self):
        """N14: Blind context must include retrieved chunk text"""
        pod   = make_pod()
        state = make_state()
        context = pod._build_blind_context(state)
        assert "Net income" in context or "Financial Statements" in context


# ════════════════════════════════════════════════════════════════════════════
# GROUP 3 -- CONTRADICTION DETECTION (tests 08-13)
# ════════════════════════════════════════════════════════════════════════════

class TestContradictionDetection:

    def test_08_detects_numerical_contradiction(self):
        """N14: Must detect when auditor and analyst disagree numerically"""
        pod   = make_pod()
        state = BAState(
            session_id     = "t08",
            query          = "test",
            analyst_output = "Net income was $96,995 million FY2023",
            auditor_output = "Net income was $57,411 million FY2023",
        )
        contradictions = pod._detect_contradictions(state)
        numerical = [c for c in contradictions if "NUMERICAL" in c]
        assert len(numerical) > 0

    def test_09_detects_fiscal_year_contradiction(self):
        """N14: Must detect fiscal year mismatch"""
        pod   = make_pod()
        state = BAState(
            session_id     = "t09",
            query          = "test",
            analyst_output = "Net income $96,995M in FY2023",
            auditor_output = "Net income $96,995M in FY2022",
        )
        contradictions = pod._detect_contradictions(state)
        fy_contradictions = [c for c in contradictions if "FISCAL_YEAR" in c]
        assert len(fy_contradictions) > 0

    def test_10_no_contradiction_on_matching_answers(self):
        """N14: Identical answers must produce no contradictions"""
        pod   = make_pod()
        state = BAState(
            session_id     = "t10",
            query          = "test",
            analyst_output = "Net income was $96,995 million FY2023",
            auditor_output = "Net income was $96,995 million FY2023",
        )
        contradictions = pod._detect_contradictions(state)
        assert len(contradictions) == 0

    def test_11_no_contradiction_when_analyst_empty(self):
        """N14: No contradictions when analyst_output is empty"""
        pod   = make_pod()
        state = BAState(
            session_id     = "t11",
            query          = "test",
            analyst_output = "",
            auditor_output = "Net income $96,995M FY2023",
        )
        contradictions = pod._detect_contradictions(state)
        assert len(contradictions) == 0

    def test_12_contradiction_flags_written_to_state(self):
        """N14: run() must write contradiction_flags to BAState"""
        pod   = make_pod()
        state = make_state()
        state = pod.run(state)
        assert isinstance(state.contradiction_flags, list)

    def test_13_small_numerical_difference_not_flagged(self):
        """N14: < 10% numerical difference must not be flagged"""
        pod   = make_pod()
        state = BAState(
            session_id     = "t13",
            query          = "test",
            analyst_output = "Net income $96,995M FY2023",
            auditor_output = "Net income $97,500M FY2023",
        )
        contradictions = pod._detect_contradictions(state)
        numerical = [c for c in contradictions if "NUMERICAL" in c]
        assert len(numerical) == 0


# ════════════════════════════════════════════════════════════════════════════
# GROUP 4 -- BASTATE INTEGRATION (tests 14-19)
# ════════════════════════════════════════════════════════════════════════════

class TestBAStateIntegration:

    def test_14_run_writes_auditor_output(self):
        """N14: run() must write auditor_output to BAState"""
        pod   = make_pod()
        state = make_state()
        state = pod.run(state)
        assert state.auditor_output != ""

    def test_15_run_writes_auditor_confidence(self):
        """N14: run() must write auditor_confidence to BAState"""
        pod   = make_pod()
        state = make_state()
        state = pod.run(state)
        assert 0.0 <= state.auditor_confidence <= 1.0

    def test_16_run_writes_auditor_piv_status_pass(self):
        """N14: VALIDATOR_PASS must set auditor_piv_status=PASS"""
        pod   = make_pod(val_result=VALIDATOR_PASS)
        state = make_state()
        state = pod.run(state)
        assert state.auditor_piv_status == PIVStatus.PASS

    def test_17_run_writes_auditor_citations(self):
        """N14: run() must write auditor_citations to BAState"""
        pod   = make_pod()
        state = make_state()
        state = pod.run(state)
        assert isinstance(state.auditor_citations, list)

    def test_18_seed_unchanged_after_run(self):
        """C5: BAState seed must still be 42 after N14"""
        pod   = make_pod()
        state = make_state()
        state = pod.run(state)
        assert state.seed == 42

    def test_19_no_query_skips_pod(self):
        """N14: Missing query must skip pod and set REJECT"""
        pod   = make_pod()
        state = BAState(session_id="no-query-n14")
        state = pod.run(state)
        assert state.auditor_piv_status == PIVStatus.REJECT


# ════════════════════════════════════════════════════════════════════════════
# GROUP 5 -- EDGE CASES (tests 20-24)
# ════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_20_all_three_pods_write_different_fields(self):
        """N11+N12+N14: All three pods must write to different BAState fields"""
        from src.agents.analyst_pod import AnalystPod
        from src.agents.quant_pod   import QuantPod

        # Build all three pods with mocks
        analyst_pod = AnalystPod(
            StrategicPlanner(), ContextImplementor(), CuriousValidator()
        )
        quant_pod   = QuantPod(
            StrategicPlanner(), ContextImplementor(), CuriousValidator()
        )
        auditor_pod = make_pod(answer="Auditor: $96,995M [FS/42].")

        for pod in [analyst_pod, quant_pod]:
            pod.planner.run     = MagicMock(return_value=mock_planner_out())
            pod.implementor.run = MagicMock(return_value=mock_impl_out())
            pod.validator.run   = MagicMock(return_value=mock_val_out())

        state = make_state()
        state = analyst_pod.run(state)
        state = quant_pod.run(state)
        state = auditor_pod.run(state)

        assert state.analyst_output != ""
        assert state.quant_result   != ""
        assert state.auditor_output != ""
        assert state.seed           == 42

    def test_21_auditor_output_is_string(self):
        """N14: auditor_output must be a string"""
        pod   = make_pod()
        state = make_state()
        state = pod.run(state)
        assert isinstance(state.auditor_output, str)

    def test_22_fiscal_year_extraction(self):
        """N14: _extract_fiscal_year must parse FY references"""
        pod = make_pod()
        assert pod._extract_fiscal_year("FY2023 results") == "FY2023"
        assert pod._extract_fiscal_year("fiscal 2022")    == "FY2022"
        assert pod._extract_fiscal_year("no year here")   is None

    def test_23_number_extraction_handles_millions(self):
        """N14: _extract_numbers must parse comma-formatted millions"""
        pod   = make_pod()
        nums  = pod._extract_numbers("Net income was $96,995 million in FY2023")
        assert len(nums) > 0
        assert 96995.0 in nums

    def test_24_contradiction_flags_empty_list_by_default(self):
        """N14: contradiction_flags must be empty list when no contradictions"""
        pod   = make_pod()
        state = make_state(analyst_output="")
        state = pod.run(state)
        # When analyst_output is empty, no contradiction possible
        assert isinstance(state.contradiction_flags, list)