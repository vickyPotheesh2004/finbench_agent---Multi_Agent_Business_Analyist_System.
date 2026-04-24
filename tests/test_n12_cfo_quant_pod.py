"""
tests/test_n12_cfo_quant_pod.py
Tests for N12 CFO/Quant Pod
PDR-BAAAI-001 · Rev 1.0
"""

import pytest
import numpy as np
from src.analysis.cfo_quant_pod import (
    CFOQuantPod,
    run_cfo_quant_pod,
    run_monte_carlo,
    compute_var,
    compute_garch,
    compute_ratio,
    MonteCarloResult,
    VaRResult,
    GARCHResult,
    MONTE_CARLO_SCENARIOS,
    VAR_CONFIDENCE_95,
    VAR_CONFIDENCE_99,
    MIN_DATA_POINTS,
    SEED,
)
from src.state.ba_state import BAState


# ── Mock LLM (same pattern as N11) ───────────────────────────────────────────

class MockLLMClient:
    def __init__(self, responses=None):
        self.responses   = responses or []
        self._call_count = 0

    def chat(self, prompt, temperature=0.1):
        if self._call_count < len(self.responses):
            resp = self.responses[self._call_count]
        else:
            resp = self.responses[-1] if self.responses else ""
        self._call_count += 1
        return resp

    def is_available(self):
        return True


MOCK_PLANNER = """
CURIOSITY_Q1: Gross margin ratio calculation.
CURIOSITY_Q2: Gross profit divided by revenue.
CURIOSITY_Q3: Income statement.
CURIOSITY_Q4: GAAP vs non-GAAP.
CURIOSITY_Q5: Prior year for comparison.
CURIOSITY_Q6: Non-recurring items.
ANALYSIS_PLAN: Compute gross profit / revenue from income statement.
RETRIEVAL_HINTS: gross profit, revenue
VALIDATION_CRITERIA: Show formula with inputs and result as percentage.
"""

MOCK_IMPLEMENTOR = """
ANSWER: Gross margin was 44.1% in FY2023. Gross profit $169,148M / Revenue $383,285M = 44.1% [INCOME_STATEMENT/P94].
COMPUTATION: 169148 / 383285 = 0.441 = 44.1%
CONFIDENCE: 0.93 because formula verified with cited values.
CITATIONS: [INCOME_STATEMENT / PAGE 94: Gross profit $169,148M, Revenue $383,285M]
"""

MOCK_VALIDATOR_PASS = """
V1_SCOPE: Result: PASS Reason: Formula shown.
V2_UNITS: Result: PASS Reason: Percentage stated.
V3_SIGN: Result: PASS Reason: Positive margin correct.
V4_CITATION: Result: PASS Reason: Citations valid.
V5_FISCAL_YEAR: Result: PASS Reason: FY2023 correct.
V6_CONSISTENCY: Result: PASS Reason: 169148/383285 = 44.1% verified.
V7_COMPLETENESS: Result: PASS Reason: Complete.
V8_GROUNDING: Result: PASS Reason: All from context.
VALIDATOR_PASS: ALL 8 checks are PASS
REJECT_REASONS:
RETRY_INSTRUCTIONS:
"""

SAMPLE_CHUNKS = [
    {
        "chunk_id":    "c1",
        "text":        "Apple gross profit was 169148 million. Revenue was 383285 million in FY2023.",
        "section":     "INCOME_STATEMENT",
        "page":        94,
        "company":     "Apple Inc",
        "doc_type":    "10-K",
        "fiscal_year": "FY2023",
    },
    {
        "chunk_id":    "c2",
        "text":        "Net income was 96995 million. Total assets 352583 million.",
        "section":     "INCOME_STATEMENT",
        "page":        94,
        "company":     "Apple Inc",
        "doc_type":    "10-K",
        "fiscal_year": "FY2023",
    },
]


@pytest.fixture
def mock_pod():
    llm = MockLLMClient(responses=[MOCK_PLANNER, MOCK_IMPLEMENTOR, MOCK_VALIDATOR_PASS])
    return CFOQuantPod(llm_client=llm, seed=SEED)


# ── Group 1: Constants ────────────────────────────────────────────────────────

class TestConstants:

    def test_01_monte_carlo_scenarios(self):
        assert MONTE_CARLO_SCENARIOS == 10_000

    def test_02_var_confidence_95(self):
        assert VAR_CONFIDENCE_95 == 0.05

    def test_03_var_confidence_99(self):
        assert VAR_CONFIDENCE_99 == 0.01

    def test_04_min_data_points(self):
        assert MIN_DATA_POINTS == 2

    def test_05_seed_is_42(self):
        assert SEED == 42


# ── Group 2: Monte Carlo ──────────────────────────────────────────────────────

class TestMonteCarlo:

    def test_06_returns_result(self):
        result = run_monte_carlo(base_value=383285.0)
        assert isinstance(result, MonteCarloResult)

    def test_07_correct_scenario_count(self):
        result = run_monte_carlo(base_value=100.0, n_scenarios=1000)
        assert result.n_scenarios == 1000

    def test_08_mean_near_base_value(self):
        result = run_monte_carlo(
            base_value=100.0, growth_rate=0.0,
            volatility=0.01, n_scenarios=10000,
        )
        assert abs(result.mean - 100.0) < 5.0

    def test_09_percentile_5_below_mean(self):
        result = run_monte_carlo(base_value=100.0)
        assert result.percentile_5 < result.mean

    def test_10_percentile_95_above_mean(self):
        result = run_monte_carlo(base_value=100.0)
        assert result.percentile_95 > result.mean

    def test_11_std_is_positive(self):
        result = run_monte_carlo(base_value=100.0)
        assert result.std > 0

    def test_12_base_value_stored(self):
        result = run_monte_carlo(base_value=383285.0)
        assert result.base_value == 383285.0

    def test_13_seed_42_reproducible(self):
        r1 = run_monte_carlo(base_value=100.0, seed=42)
        r2 = run_monte_carlo(base_value=100.0, seed=42)
        assert r1.mean == r2.mean


# ── Group 3: Historical VaR ───────────────────────────────────────────────────

class TestHistoricalVaR:

    def test_14_returns_result(self):
        returns = [0.05, -0.03, 0.02, -0.01, 0.04, -0.02, 0.03]
        result  = compute_var(returns)
        assert isinstance(result, VaRResult)

    def test_15_var_95_below_var_99(self):
        returns = [0.05, -0.10, 0.02, -0.08, 0.04, -0.15, 0.03, -0.05, 0.01]
        result  = compute_var(returns)
        assert result.var_95 >= result.var_99

    def test_16_method_is_historical(self):
        returns = [0.05, -0.03, 0.02, -0.01]
        result  = compute_var(returns)
        assert result.method == "historical"

    def test_17_n_periods_correct(self):
        returns = [0.01, -0.02, 0.03, -0.01, 0.02]
        result  = compute_var(returns)
        assert result.n_periods == 5

    def test_18_empty_returns_returns_none(self):
        result = compute_var([])
        assert result is None

    def test_19_single_return_returns_none(self):
        result = compute_var([0.05])
        assert result is None

    def test_20_negative_returns_give_negative_var(self):
        returns = [-0.10, -0.05, -0.08, -0.12, -0.03]
        result  = compute_var(returns)
        assert result.var_95 < 0


# ── Group 4: GARCH ────────────────────────────────────────────────────────────

class TestGARCH:

    def test_21_returns_result_or_none(self):
        returns = [0.05, -0.03, 0.02, -0.01, 0.04]
        result  = compute_garch(returns)
        assert result is None or isinstance(result, GARCHResult)

    def test_22_empty_returns_none(self):
        result = compute_garch([])
        assert result is None

    def test_23_single_point_returns_none(self):
        result = compute_garch([0.05])
        assert result is None

    def test_24_volatility_is_positive(self):
        returns = [0.05, -0.03, 0.02, -0.01, 0.04, -0.02, 0.03]
        result  = compute_garch(returns)
        if result is not None:
            assert result.conditional_volatility >= 0

    def test_25_n_obs_correct(self):
        returns = [0.05, -0.03, 0.02, -0.01]
        result  = compute_garch(returns)
        if result is not None:
            assert result.n_obs == 4


# ── Group 5: Ratio computation ────────────────────────────────────────────────

class TestRatioComputation:

    def test_26_basic_ratio(self):
        result = compute_ratio(169148.0, 383285.0, "gross_margin")
        assert result is not None
        assert abs(result - 0.441) < 0.001

    def test_27_zero_denominator_returns_none(self):
        result = compute_ratio(100.0, 0.0, "test")
        assert result is None

    def test_28_negative_numerator(self):
        result = compute_ratio(-10000.0, 100000.0, "loss_margin")
        assert result is not None
        assert result < 0

    def test_29_result_rounded(self):
        result = compute_ratio(1.0, 3.0, "one_third")
        assert result is not None
        assert len(str(result).split(".")[-1]) <= 6


# ── Group 6: CFOQuantPod ──────────────────────────────────────────────────────

class TestCFOQuantPod:

    def test_30_instantiates(self, mock_pod):
        assert mock_pod is not None

    def test_31_run_quant_returns_dict(self, mock_pod):
        result = mock_pod.run_quant(
            query      = "What was gross margin FY2023?",
            chunks     = SAMPLE_CHUNKS,
            query_type = "ratio",
        )
        assert isinstance(result, dict)

    def test_32_result_has_answer(self, mock_pod):
        result = mock_pod.run_quant("gross margin", SAMPLE_CHUNKS)
        assert "answer" in result
        assert isinstance(result["answer"], str)

    def test_33_result_has_confidence(self, mock_pod):
        result = mock_pod.run_quant("gross margin", SAMPLE_CHUNKS)
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0

    def test_34_monte_carlo_runs_when_values_found(self, mock_pod):
        result = mock_pod.run_quant("revenue", SAMPLE_CHUNKS)
        if result.get("monte_carlo"):
            mc = result["monte_carlo"]
            assert mc.n_scenarios == MONTE_CARLO_SCENARIOS

    def test_35_result_has_expected_keys(self, mock_pod):
        result = mock_pod.run_quant("gross margin", SAMPLE_CHUNKS)
        for key in ["answer", "confidence", "citations", "computation"]:
            assert key in result

    def test_36_empty_chunks_still_returns_dict(self, mock_pod):
        result = mock_pod.run_quant("gross margin", [])
        assert isinstance(result, dict)
        assert "answer" in result


# ── Group 7: BAState integration ─────────────────────────────────────────────

class TestBAStateIntegration:

    def test_37_run_writes_quant_result(self, mock_pod):
        state = BAState(
            session_id        = "t37",
            query             = "What was gross margin FY2023?",
            query_type        = "ratio",
            retrieval_stage_2 = SAMPLE_CHUNKS,
        )
        state = mock_pod.run(state)
        assert hasattr(state, "quant_result")
        assert isinstance(state.quant_result, str)

    def test_38_run_writes_quant_confidence(self, mock_pod):
        state = BAState(
            session_id        = "t38",
            query             = "gross margin",
            retrieval_stage_2 = SAMPLE_CHUNKS,
        )
        state = mock_pod.run(state)
        assert 0.0 <= state.quant_confidence <= 1.0

    def test_39_seed_unchanged(self, mock_pod):
        """C5: seed must remain 42."""
        state = BAState(
            session_id        = "t39",
            query             = "gross margin",
            retrieval_stage_2 = SAMPLE_CHUNKS,
        )
        state = mock_pod.run(state)
        assert state.seed == 42

    def test_40_empty_query_skips_pod(self, mock_pod):
        state = BAState(session_id="t40", query="")
        state = mock_pod.run(state)
        assert state.quant_result   == ""
        assert state.quant_confidence == 0.0

    def test_41_no_rlef_in_quant_result(self, mock_pod):
        """C9: quant_result must never contain _rlef_."""
        state = BAState(
            session_id        = "t41",
            query             = "gross margin",
            retrieval_stage_2 = SAMPLE_CHUNKS,
        )
        state = mock_pod.run(state)
        assert "_rlef_" not in state.quant_result

    def test_42_monte_carlo_stored_in_state(self, mock_pod):
        state = BAState(
            session_id        = "t42",
            query             = "revenue forecast",
            retrieval_stage_2 = SAMPLE_CHUNKS,
        )
        state = mock_pod.run(state)
        if state.monte_carlo_results:
            assert "mean"        in state.monte_carlo_results
            assert "n_scenarios" in state.monte_carlo_results


# ── Group 8: Convenience wrapper ─────────────────────────────────────────────

class TestConvenienceWrapper:

    def test_43_run_cfo_quant_pod_returns_state(self):
        llm   = MockLLMClient(responses=[MOCK_PLANNER, MOCK_IMPLEMENTOR, MOCK_VALIDATOR_PASS])
        state = BAState(
            session_id        = "t43",
            query             = "What was gross margin?",
            retrieval_stage_2 = SAMPLE_CHUNKS,
        )
        result = run_cfo_quant_pod(state, llm_client=llm)
        assert hasattr(result, "quant_result")
        assert result.seed == 42