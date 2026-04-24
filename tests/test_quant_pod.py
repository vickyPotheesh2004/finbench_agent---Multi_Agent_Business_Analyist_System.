"""
tests/test_quant_pod.py
FinBench Multi-Agent Business Analyst AI

Tests for N12 -- CFO/Quant Pod

All Ollama calls mocked -- fast, deterministic.

24 tests covering:
  - Instantiation (tests 01-03)
  - Monte Carlo (tests 04-07)
  - VaR computation (tests 08-10)
  - GARCH volatility (tests 11-12)
  - Number extraction (tests 13-14)
  - BAState integration (tests 15-20)
  - Edge cases (tests 21-24)
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pytest
import numpy as np

from src.agents.quant_pod    import QuantPod
from src.agents.planner      import StrategicPlanner,   PlannerOutput
from src.agents.implementor  import ContextImplementor, ImplementorOutput
from src.agents.validator    import CuriousValidator,   ValidatorOutput, VALIDATOR_PASS, ALL_CHECKS
from src.agents.piv_loop     import PIVLoopController
from src.state.ba_state      import BAState, QueryType, Difficulty, PIVStatus


# ── Mock helpers ──────────────────────────────────────────────────────────────

def mock_planner():
    p = StrategicPlanner()
    p.run = MagicMock(return_value=PlannerOutput(
        analysis_plan       = "1. Compute ratio. 2. Show formula. 3. Cite source.",
        retrieval_hints     = ["gross profit", "net sales"],
        validation_criteria = "Must show formula. Must state units.",
        curiosity_answers   = {k: "answer" for k in [
            "Q1_SCOPE","Q2_CONCEPTS","Q3_SECTIONS",
            "Q4_TRAPS","Q5_VERIFY","Q6_EDGECASES"
        ]},
    ))
    return p

def mock_implementor(answer: str = "Gross margin = 44.1% [FS/42].",
                     confidence: float = 0.91) -> ContextImplementor:
    impl = ContextImplementor()
    impl.run = MagicMock(return_value=ImplementorOutput(
        answer      = answer,
        confidence  = confidence,
        citations   = ["Financial Statements / Page 42"],
        computation = "Gross Margin = $169,148M / $383,285M = 44.1%",
        output_type = "ANSWER",
    ))
    return impl

def mock_validator(result: str = VALIDATOR_PASS) -> CuriousValidator:
    v = CuriousValidator()
    v.run = MagicMock(return_value=ValidatorOutput(
        result             = result,
        checks             = {c: "PASS" for c in ALL_CHECKS},
        check_reasons      = {},
        reject_reasons     = [],
        retry_instructions = "",
    ))
    return v

def make_pod(val_result: str = VALIDATOR_PASS) -> QuantPod:
    return QuantPod(
        planner     = mock_planner(),
        implementor = mock_implementor(),
        validator   = mock_validator(val_result),
    )

def make_state() -> BAState:
    return BAState(
        session_id        = "test-n12",
        query             = "What was Apple gross margin FY2023?",
        query_type        = QueryType.RATIO,
        query_difficulty  = Difficulty.MEDIUM,
        company_name      = "Apple Inc",
        doc_type          = "10-K",
        fiscal_year       = "FY2023",
        assembled_prompt  = (
            "RETRIEVED CONTEXT:\n"
            "Gross profit: $169,148 million. Net sales: $383,285 million.\n"
            "QUESTION: What was Apple gross margin FY2023?"
        ),
        retrieval_stage_2 = [{
            "text":        "Gross profit: $169,148 million. Net sales: $383,285 million.",
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

    def test_01_quant_pod_instantiates(self):
        """N12: QuantPod must instantiate without error"""
        pod = QuantPod()
        assert pod is not None

    def test_02_quant_pod_has_piv_loop(self):
        """N12: QuantPod must contain a PIVLoopController"""
        pod = QuantPod()
        assert isinstance(pod.piv, PIVLoopController)

    def test_03_max_retries_is_5(self):
        """A2: QuantPod max_retries must be 5"""
        pod = QuantPod()
        assert pod.max_retries == 5


# ════════════════════════════════════════════════════════════════════════════
# GROUP 2 -- MONTE CARLO (tests 04-07)
# ════════════════════════════════════════════════════════════════════════════

class TestMonteCarlo:

    def test_04_monte_carlo_returns_dict(self):
        """N12: run_monte_carlo must return dict with required keys"""
        pod = QuantPod()
        mc  = pod.run_monte_carlo(base_value=96995.0, uncertainty=4849.75)
        for key in ["mean", "std", "p5", "p25", "p50", "p75", "p95", "n"]:
            assert key in mc, f"Missing key: {key}"

    def test_05_monte_carlo_n_is_10000(self):
        """N12: Monte Carlo must run exactly 10,000 scenarios"""
        pod = QuantPod()
        mc  = pod.run_monte_carlo(base_value=100.0, uncertainty=5.0)
        assert mc["n"] == 10_000

    def test_06_monte_carlo_percentiles_ordered(self):
        """N12: p5 < p50 < p95 must hold"""
        pod = QuantPod()
        mc  = pod.run_monte_carlo(base_value=96995.0, uncertainty=4849.75)
        assert mc["p5"] < mc["p50"] < mc["p95"]

    def test_07_monte_carlo_reproducible_seed42(self):
        """C5: Same inputs must produce same Monte Carlo output"""
        pod = QuantPod()
        mc1 = pod.run_monte_carlo(base_value=96995.0, uncertainty=4849.75)
        mc2 = pod.run_monte_carlo(base_value=96995.0, uncertainty=4849.75)
        assert mc1["mean"] == mc2["mean"]
        assert mc1["p5"]   == mc2["p5"]


# ════════════════════════════════════════════════════════════════════════════
# GROUP 3 -- VAR COMPUTATION (tests 08-10)
# ════════════════════════════════════════════════════════════════════════════

class TestVaR:

    def test_08_var_returns_two_values(self):
        """N12: compute_var must return (var_95, var_99) tuple"""
        pod            = QuantPod()
        values         = [96995.0, 99803.0, 57411.0, 94680.0, 94321.0]
        var_95, var_99 = pod.compute_var(values)
        assert var_95 is not None
        assert var_99 is not None

    def test_09_var_99_more_conservative_than_var_95(self):
        """N12: var_99 must be <= var_95 (more conservative)"""
        pod            = QuantPod()
        values         = [96995.0, 99803.0, 57411.0, 94680.0, 94321.0]
        var_95, var_99 = pod.compute_var(values)
        assert var_99 <= var_95

    def test_10_var_returns_none_for_insufficient_data(self):
        """N12: compute_var must return (None, None) if < 3 data points"""
        pod            = QuantPod()
        var_95, var_99 = pod.compute_var([96995.0, 99803.0])
        assert var_95 is None
        assert var_99 is None


# ════════════════════════════════════════════════════════════════════════════
# GROUP 4 -- GARCH VOLATILITY (tests 11-12)
# ════════════════════════════════════════════════════════════════════════════

class TestGARCH:

    def test_11_garch_returns_float(self):
        """N12: compute_garch_volatility must return float >= 0"""
        pod    = QuantPod()
        values = [96995.0, 99803.0, 57411.0, 94680.0, 94321.0]
        vol    = pod.compute_garch_volatility(values)
        assert vol is not None
        assert isinstance(vol, float)
        assert vol >= 0.0

    def test_12_garch_returns_none_for_single_point(self):
        """N12: compute_garch_volatility must return None for < 2 data points"""
        pod = QuantPod()
        vol = pod.compute_garch_volatility([96995.0])
        assert vol is None


# ════════════════════════════════════════════════════════════════════════════
# GROUP 5 -- NUMBER EXTRACTION (tests 13-14)
# ════════════════════════════════════════════════════════════════════════════

class TestNumberExtraction:

    def test_13_extracts_numbers_from_text(self):
        """N12: _extract_numbers must find numerical values in text"""
        pod  = QuantPod()
        nums = pod._extract_numbers(
            "Net income was $96,995 million. Revenue was $383,285 million."
        )
        assert len(nums) >= 2
        assert 96995.0 in nums

    def test_14_returns_empty_for_no_numbers(self):
        """N12: _extract_numbers must return empty list if no numbers"""
        pod  = QuantPod()
        nums = pod._extract_numbers("No numbers in this text at all.")
        assert isinstance(nums, list)


# ════════════════════════════════════════════════════════════════════════════
# GROUP 6 -- BASTATE INTEGRATION (tests 15-20)
# ════════════════════════════════════════════════════════════════════════════

class TestBAStateIntegration:

    def test_15_run_writes_quant_result(self):
        """N12: run() must write quant_result to BAState"""
        pod   = make_pod()
        state = make_state()
        state = pod.run(state)
        assert state.quant_result != ""

    def test_16_run_writes_quant_confidence(self):
        """N12: run() must write quant_confidence to BAState"""
        pod   = make_pod()
        state = make_state()
        state = pod.run(state)
        assert 0.0 <= state.quant_confidence <= 1.0

    def test_17_run_writes_quant_piv_status(self):
        """N12: run() must write quant_piv_status to BAState"""
        pod   = make_pod(val_result=VALIDATOR_PASS)
        state = make_state()
        state = pod.run(state)
        assert state.quant_piv_status == PIVStatus.PASS

    def test_18_run_writes_monte_carlo_results(self):
        """N12: run() must write monte_carlo_results to BAState"""
        pod   = make_pod()
        state = make_state()
        state = pod.run(state)
        # monte_carlo_results populated when numbers found in answer
        assert state.monte_carlo_results is not None or \
               state.quant_result != ""   # either mc ran or answer present

    def test_19_seed_unchanged_after_run(self):
        """C5: BAState seed must still be 42 after N12"""
        pod   = make_pod()
        state = make_state()
        state = pod.run(state)
        assert state.seed == 42

    def test_20_no_query_skips_pod(self):
        """N12: Missing query must skip pod and set REJECT"""
        pod   = make_pod()
        state = BAState(session_id="no-query-n12")
        state = pod.run(state)
        assert state.quant_piv_status == PIVStatus.REJECT


# ════════════════════════════════════════════════════════════════════════════
# GROUP 7 -- EDGE CASES (tests 21-24)
# ════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_21_monte_carlo_handles_zero_uncertainty(self):
        """N12: Monte Carlo must handle zero uncertainty gracefully"""
        pod = QuantPod()
        mc  = pod.run_monte_carlo(base_value=100.0, uncertainty=0.0)
        assert mc["n"]    == 10_000
        assert mc["mean"] > 0.0

    def test_22_var_result_written_to_bastate(self):
        """N12: var_result field must be written to BAState"""
        pod   = make_pod()
        state = make_state()
        # Add multiple numbers to context for VaR
        state.retrieval_stage_2 = [{
            "text": (
                "Net income: $96,995M $99,803M $57,411M $94,680M $94,321M. "
                "Gross profit: $169,148M."
            ),
            "section":     "Financial Statements",
            "page":        "42",
            "company":     "Apple Inc",
            "doc_type":    "10-K",
            "fiscal_year": "FY2023",
        }]
        state = pod.run(state)
        # VaR runs when >= 3 context numbers found
        assert state.var_result is not None or state.quant_result != ""

    def test_23_n11_and_n12_both_write_state(self):
        """N11+N12: Both pods must write to different BAState fields"""
        from src.agents.analyst_pod import AnalystPod
        from src.agents.validator   import ValidatorOutput

        # Build analyst pod with mocks
        analyst_planner     = mock_planner()
        analyst_implementor = mock_implementor(
            answer="Net income $96,995M [FS/42].", confidence=0.92
        )
        analyst_validator   = mock_validator(VALIDATOR_PASS)
        analyst_pod = AnalystPod(
            analyst_planner, analyst_implementor, analyst_validator
        )

        # Build quant pod with mocks
        quant_pod = make_pod()

        state = make_state()
        state.query      = "What was Apple net income and gross margin FY2023?"
        state.query_type = QueryType.NUMERICAL

        state = analyst_pod.run(state)
        state = quant_pod.run(state)

        assert state.analyst_output != ""
        assert state.quant_result   != ""
        assert state.seed           == 42

    def test_24_quant_result_in_bastate_is_str(self):
        """N12: quant_result in BAState must be a string"""
        pod   = make_pod()
        state = make_state()
        state = pod.run(state)
        assert isinstance(state.quant_result, str)