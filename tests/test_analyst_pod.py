"""
tests/test_analyst_pod.py
FinBench Multi-Agent Business Analyst AI

Tests for N11 -- Analyst Pod + PIV Loop

All Ollama calls are mocked -- no real LLM needed.
Tests run fast, deterministic, no RAM pressure.

24 tests covering:
  - Instantiation (tests 01-03)
  - Planner unit tests (tests 04-06)
  - Implementor unit tests (tests 07-09)
  - Validator unit tests (tests 10-12)
  - PIV loop controller (tests 13-17)
  - AnalystPod BAState integration (tests 18-22)
  - Edge cases (tests 23-24)
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from src.agents.planner      import StrategicPlanner,   PlannerOutput
from src.agents.implementor  import ContextImplementor, ImplementorOutput
from src.agents.validator    import (
    CuriousValidator, ValidatorOutput,
    VALIDATOR_PASS, VALIDATOR_REJECT, ALL_CHECKS,
)
from src.agents.piv_loop     import PIVLoopController, PIVResult
from src.agents.analyst_pod  import AnalystPod
from src.state.ba_state      import BAState, QueryType, Difficulty, PIVStatus


# ── Mock helpers ──────────────────────────────────────────────────────────────

def mock_planner_output(plan: str = "1. Extract figure. 2. Cite source.") -> PlannerOutput:
    return PlannerOutput(
        analysis_plan       = plan,
        retrieval_hints     = ["net income", "financial statements"],
        validation_criteria = "Must cite section/page. Must state units.",
        curiosity_answers   = {
            "Q1_SCOPE":     "Find Apple net income FY2023",
            "Q2_CONCEPTS":  "Net income, earnings",
            "Q3_SECTIONS":  "Financial Statements",
            "Q4_TRAPS":     "FY vs CY confusion",
            "Q5_VERIFY":    "Cross-check with MD&A",
            "Q6_EDGECASES": "Restatements",
        },
    )


def mock_implementor_output(
    answer: str = "Net income was $96,995 million [Financial Statements / Page 42].",
    confidence: float = 0.92,
    output_type: str = "ANSWER",
) -> ImplementorOutput:
    return ImplementorOutput(
        answer      = answer,
        confidence  = confidence,
        citations   = ["Financial Statements / Page 42: $96,995M"],
        computation = "N/A",
        output_type = output_type,
    )


def mock_validator_output(
    result: str = VALIDATOR_PASS,
    failed: list = None,
) -> ValidatorOutput:
    checks = {c: "PASS" for c in ALL_CHECKS}
    if failed:
        for c in failed:
            checks[c] = "FAIL"
    return ValidatorOutput(
        result             = result,
        checks             = checks,
        check_reasons      = {},
        reject_reasons     = [f"{c}: failed" for c in (failed or [])],
        retry_instructions = "Fix the issues above.",
    )


def make_test_state(query: str = "What was Apple net income FY2023?") -> BAState:
    return BAState(
        session_id        = "test-n11",
        query             = query,
        query_type        = QueryType.NUMERICAL,
        query_difficulty  = Difficulty.EASY,
        company_name      = "Apple Inc",
        doc_type          = "10-K",
        fiscal_year       = "FY2023",
        assembled_prompt  = (
            "RETRIEVED CONTEXT:\n"
            "Apple / 10-K / FY2023 / Financial Statements / 42\n"
            "Net income: $96,995 million FY2023.\n\n"
            "QUESTION: What was Apple net income FY2023?"
        ),
        retrieval_stage_2 = [{
            "chunk_id":    "chunk_000001",
            "text":        "Net income: $96,995 million FY2023.",
            "section":     "Financial Statements",
            "page":        "42",
            "company":     "Apple Inc",
            "doc_type":    "10-K",
            "fiscal_year": "FY2023",
        }],
    )


# ════════════════════════════════════════════════════════════════════════════
# GROUP 1 -- INSTANTIATION (tests 01-03)
# ════════════════════════════════════════════════════════════════════════════

class TestInstantiation:

    def test_01_analyst_pod_instantiates(self):
        """N11: AnalystPod must instantiate without error"""
        pod = AnalystPod()
        assert pod is not None

    def test_02_piv_max_retries_is_5(self):
        """A2: PIVLoopController max_retries must be 5"""
        piv = PIVLoopController()
        assert piv.max_retries == 5

    def test_03_all_agents_instantiate(self):
        """N11: All 4 agents must instantiate correctly"""
        planner     = StrategicPlanner()
        implementor = ContextImplementor()
        validator   = CuriousValidator()
        piv         = PIVLoopController(planner, implementor, validator)
        assert planner     is not None
        assert implementor is not None
        assert validator   is not None
        assert piv         is not None


# ════════════════════════════════════════════════════════════════════════════
# GROUP 2 -- PLANNER UNIT TESTS (tests 04-06)
# ════════════════════════════════════════════════════════════════════════════

class TestPlanner:

    def test_04_planner_fallback_produces_output(self):
        """Planner: fallback must produce valid PlannerOutput"""
        planner = StrategicPlanner()
        result  = planner._fallback_plan("What was Apple net income FY2023?")
        assert result.analysis_plan       != ""
        assert result.validation_criteria != ""
        assert isinstance(result.retrieval_hints,   list)
        assert isinstance(result.curiosity_answers, dict)
        assert result.fallback_used is True

    def test_05_planner_curiosity_has_6_keys(self):
        """Planner: curiosity_answers must have Q1-Q6 keys"""
        planner = StrategicPlanner()
        result  = planner._fallback_plan("test query")
        expected_keys = {
            "Q1_SCOPE", "Q2_CONCEPTS", "Q3_SECTIONS",
            "Q4_TRAPS", "Q5_VERIFY",   "Q6_EDGECASES"
        }
        assert set(result.curiosity_answers.keys()) == expected_keys

    def test_06_planner_run_with_mocked_ollama(self):
        """Planner: run() with mocked Ollama must return PlannerOutput"""
        planner = StrategicPlanner()
        mock_response = (
            "Q1_SCOPE: Find Apple net income FY2023\n"
            "Q2_CONCEPTS: Net income, earnings per share\n"
            "Q3_SECTIONS: Financial Statements, MD&A\n"
            "Q4_TRAPS: FY vs CY, unit confusion, segment mismatch\n"
            "Q5_VERIFY: Cross-check MD&A narrative\n"
            "Q6_EDGECASES: Restatements, non-GAAP adjustments\n"
            "ANALYSIS_PLAN: 1. Find net income. 2. Cite section and page.\n"
            "RETRIEVAL_HINTS: net income, financial statements, earnings\n"
            "VALIDATION_CRITERIA: Must cite section/page and state units.\n"
        )
        with patch.object(planner, "_call_ollama", return_value=mock_response):
            result = planner.run("What was Apple net income FY2023?")
        assert result.fallback_used    is False
        assert result.analysis_plan    != ""
        assert len(result.retrieval_hints) > 0


# ════════════════════════════════════════════════════════════════════════════
# GROUP 3 -- IMPLEMENTOR UNIT TESTS (tests 07-09)
# ════════════════════════════════════════════════════════════════════════════

class TestImplementor:

    def test_07_implementor_fallback_produces_output(self):
        """Implementor: fallback must produce valid ImplementorOutput"""
        impl   = ContextImplementor()
        result = impl._fallback_output(
            "What was Apple net income?",
            "Net income: $96,995 million",
            retry_count=0,
        )
        assert result.output_type in ["ANSWER", "RETRIEVAL_MISS"]
        assert 0.0 <= result.confidence <= 1.0
        assert result.fallback_used is True

    def test_08_implementor_confidence_decays_on_retry(self):
        """Implementor: confidence must decay on retry attempts"""
        impl = ContextImplementor()
        mock_response = (
            "ANSWER: Net income was $96,995 million.\n"
            "COMPUTATION: N/A\n"
            "CONFIDENCE: 0.9 because answer is in context.\n"
            "CITATIONS: Financial Statements / Page 42\n"
        )
        with patch.object(impl, "_call_ollama", return_value=mock_response):
            r0 = impl.run("test", "context", "plan", "criteria", retry_count=0)
            r2 = impl.run("test", "context", "plan", "criteria", retry_count=2)
        assert r0.confidence > r2.confidence

    def test_09_implementor_detects_retrieval_miss(self):
        """Implementor: RETRIEVAL_MISS in response must set output_type"""
        impl = ContextImplementor()
        mock_response = (
            "RETRIEVAL_MISS: Net income figure not found in retrieved context."
        )
        with patch.object(impl, "_call_ollama", return_value=mock_response):
            result = impl.run("test", "no relevant context", "plan", "criteria")
        assert result.output_type == "RETRIEVAL_MISS"
        assert result.needed_info != ""


# ════════════════════════════════════════════════════════════════════════════
# GROUP 4 -- VALIDATOR UNIT TESTS (tests 10-12)
# ════════════════════════════════════════════════════════════════════════════

class TestValidator:

    def test_10_validator_pass_when_all_checks_pass(self):
        """Validator: PASS when all 8 checks return PASS"""
        validator = CuriousValidator()
        mock_response = "\n".join([
            f"{check}: Result: PASS Reason: OK"
            for check in ALL_CHECKS
        ]) + "\nVALIDATOR_PASS\nRETRY_INSTRUCTIONS: None needed."
        with patch.object(validator, "_call_ollama", return_value=mock_response):
            result = validator.run("test", "good answer", "context", "criteria")
        assert result.result == VALIDATOR_PASS
        assert result.reject_reasons == []

    def test_11_validator_reject_when_check_fails(self):
        """Validator: REJECT when any check returns FAIL"""
        validator = CuriousValidator()
        mock_response = (
            "V1_SCOPE: Result: PASS Reason: OK\n"
            "V2_UNITS: Result: FAIL Reason: Units not stated explicitly.\n"
            "V3_SIGN: Result: PASS Reason: OK\n"
            "V4_CITATION: Result: PASS Reason: OK\n"
            "V5_FISCAL_YEAR: Result: PASS Reason: OK\n"
            "V6_CONSISTENCY: Result: PASS Reason: OK\n"
            "V7_COMPLETENESS: Result: PASS Reason: OK\n"
            "V8_GROUNDING: Result: PASS Reason: OK\n"
            "VALIDATOR_REJECT\n"
            "REJECT_REASONS: 1. V2_UNITS: Units not stated.\n"
            "RETRY_INSTRUCTIONS: Please state units as millions or billions.\n"
        )
        with patch.object(validator, "_call_ollama", return_value=mock_response):
            result = validator.run("test", "answer without units", "ctx", "crit")
        assert result.result == VALIDATOR_REJECT
        assert len(result.reject_reasons) > 0

    def test_12_validator_has_8_checks(self):
        """Validator: ALL_CHECKS must contain exactly 8 check names"""
        assert len(ALL_CHECKS) == 8
        expected = {
            "V1_SCOPE", "V2_UNITS", "V3_SIGN", "V4_CITATION",
            "V5_FISCAL_YEAR", "V6_CONSISTENCY", "V7_COMPLETENESS", "V8_GROUNDING"
        }
        assert set(ALL_CHECKS) == expected


# ════════════════════════════════════════════════════════════════════════════
# GROUP 5 -- PIV LOOP CONTROLLER (tests 13-17)
# ════════════════════════════════════════════════════════════════════════════

class TestPIVLoop:

    def _make_piv_with_mocks(
        self,
        impl_answer:   str   = "Net income $96,995M [FS/42].",
        impl_conf:     float = 0.92,
        val_result:    str   = VALIDATOR_PASS,
        val_failed:    list  = None,
    ) -> PIVLoopController:
        """Create PIVLoopController with mocked agents."""
        planner     = StrategicPlanner()
        implementor = ContextImplementor()
        validator   = CuriousValidator()

        planner.run     = MagicMock(return_value=mock_planner_output())
        implementor.run = MagicMock(return_value=mock_implementor_output(
            answer=impl_answer, confidence=impl_conf
        ))
        validator.run   = MagicMock(return_value=mock_validator_output(
            result=val_result, failed=val_failed
        ))

        return PIVLoopController(planner, implementor, validator)

    def test_13_piv_pass_on_first_attempt(self):
        """PIV: VALIDATOR_PASS on first attempt must return retries_used=0"""
        piv    = self._make_piv_with_mocks(val_result=VALIDATOR_PASS)
        result = piv.run("test query", "context")
        assert result.retries_used   == 0
        assert result.low_confidence is False

    def test_14_piv_retry_on_reject(self):
        """PIV: VALIDATOR_REJECT must trigger retry"""
        call_count = [0]
        planner     = StrategicPlanner()
        implementor = ContextImplementor()
        validator   = CuriousValidator()

        def val_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 2:
                return mock_validator_output(
                    result=VALIDATOR_REJECT, failed=["V2_UNITS"]
                )
            return mock_validator_output(result=VALIDATOR_PASS)

        planner.run     = MagicMock(return_value=mock_planner_output())
        implementor.run = MagicMock(return_value=mock_implementor_output())
        validator.run   = MagicMock(side_effect=val_side_effect)

        piv    = PIVLoopController(planner, implementor, validator)
        result = piv.run("test", "context")
        assert result.retries_used   >= 1
        assert result.low_confidence is False

    def test_15_piv_low_confidence_after_max_retries(self):
        """A2+A3: Exhausting max retries must set low_confidence=True"""
        piv = self._make_piv_with_mocks(
            val_result = VALIDATOR_REJECT,
            val_failed = ["V2_UNITS"],
        )
        result = piv.run("test", "context")
        assert result.low_confidence is True
        assert result.retries_used   == piv.max_retries

    def test_16_piv_planner_reruns_on_reject(self):
        """A1: Planner must re-run on VALIDATOR_REJECT"""
        planner     = StrategicPlanner()
        implementor = ContextImplementor()
        validator   = CuriousValidator()

        planner.run     = MagicMock(return_value=mock_planner_output())
        implementor.run = MagicMock(return_value=mock_implementor_output())
        validator.run   = MagicMock(return_value=mock_validator_output(
            result=VALIDATOR_REJECT, failed=["V2_UNITS"]
        ))

        piv = PIVLoopController(planner, implementor, validator, max_retries=2)
        piv.run("test", "context")

        # A1: Planner must be called once per attempt (max_retries+1 times)
        assert planner.run.call_count == 3   # 1 + 2 retries

    def test_17_piv_result_has_required_fields(self):
        """PIV: PIVResult must have all required fields"""
        piv    = self._make_piv_with_mocks()
        result = piv.run("test", "context", pod_role="analyst")
        assert hasattr(result, "answer")
        assert hasattr(result, "confidence")
        assert hasattr(result, "citations")
        assert hasattr(result, "computation")
        assert hasattr(result, "retries_used")
        assert hasattr(result, "low_confidence")
        assert hasattr(result, "validator_checks")
        assert hasattr(result, "reject_reasons")
        assert result.pod_role == "analyst"


# ════════════════════════════════════════════════════════════════════════════
# GROUP 6 -- ANALYST POD BASTATE INTEGRATION (tests 18-22)
# ════════════════════════════════════════════════════════════════════════════

class TestAnalystPodIntegration:

    def _make_pod_with_mocks(
        self,
        val_result: str = VALIDATOR_PASS,
    ) -> AnalystPod:
        planner     = StrategicPlanner()
        implementor = ContextImplementor()
        validator   = CuriousValidator()
        planner.run     = MagicMock(return_value=mock_planner_output())
        implementor.run = MagicMock(return_value=mock_implementor_output())
        validator.run   = MagicMock(return_value=mock_validator_output(
            result=val_result
        ))
        return AnalystPod(planner, implementor, validator)

    def test_18_run_writes_analyst_output(self):
        """N11: run() must write analyst_output to BAState"""
        pod   = self._make_pod_with_mocks()
        state = make_test_state()
        state = pod.run(state)
        assert state.analyst_output != ""

    def test_19_run_writes_analyst_confidence(self):
        """N11: run() must write analyst_confidence to BAState"""
        pod   = self._make_pod_with_mocks()
        state = make_test_state()
        state = pod.run(state)
        assert 0.0 <= state.analyst_confidence <= 1.0

    def test_20_run_writes_piv_status_pass(self):
        """N11: VALIDATOR_PASS must set analyst_piv_status=PASS"""
        pod   = self._make_pod_with_mocks(val_result=VALIDATOR_PASS)
        state = make_test_state()
        state = pod.run(state)
        assert state.analyst_piv_status == PIVStatus.PASS

    def test_21_seed_unchanged_after_run(self):
        """C5: BAState seed must still be 42 after N11"""
        pod   = self._make_pod_with_mocks()
        state = make_test_state()
        state = pod.run(state)
        assert state.seed == 42

    def test_22_no_query_skips_pod(self):
        """N11: Missing query must skip pod and set REJECT status"""
        pod   = self._make_pod_with_mocks()
        state = BAState(session_id="no-query")
        state = pod.run(state)
        assert state.analyst_piv_status == PIVStatus.REJECT


# ════════════════════════════════════════════════════════════════════════════
# GROUP 7 -- EDGE CASES (tests 23-24)
# ════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_23_piv_handles_retrieval_miss(self):
        """PIV: RETRIEVAL_MISS output_type must be handled gracefully"""
        planner     = StrategicPlanner()
        implementor = ContextImplementor()
        validator   = CuriousValidator()

        planner.run     = MagicMock(return_value=mock_planner_output())
        implementor.run = MagicMock(return_value=ImplementorOutput(
            answer      = "",
            confidence  = 0.0,
            citations   = [],
            computation = "N/A",
            output_type = "RETRIEVAL_MISS",
            needed_info = "Net income figure not in context",
        ))
        validator.run = MagicMock(return_value=mock_validator_output(
            result=VALIDATOR_REJECT, failed=["V8_GROUNDING"]
        ))

        piv    = PIVLoopController(planner, implementor, validator, max_retries=1)
        result = piv.run("test", "no relevant context")
        # Should handle gracefully without crashing
        assert result is not None
        assert isinstance(result.low_confidence, bool)

    def test_24_validator_checks_always_8(self):
        """Validator: Output must always have exactly 8 check entries"""
        validator = CuriousValidator()
        result    = validator._fallback_verdict(
            "Some answer with sufficient length here"
        )
        assert len(result.checks) == 8
        assert all(v in ["PASS", "FAIL"] for v in result.checks.values())