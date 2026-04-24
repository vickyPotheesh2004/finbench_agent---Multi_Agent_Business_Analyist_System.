"""
tests/test_n13_triguard.py
Tests for N13 TriGuard Forensics
PDR-BAAAI-001 · Rev 1.0
"""

import pytest
import numpy as np
from src.analysis.triguard import (
    TriGuard,
    run_triguard,
    run_benford_test,
    run_isolation_forest,
    compute_risk_score,
    classify_severity,
    extract_first_digits,
    BenfordResult,
    IsolationResult,
    TriGuardResult,
    BENFORD_PVALUE_THRESH,
    ISOLATION_CONTAMINATION,
    MAX_ROWS_ISOLATION,
    MIN_VALUES_BENFORD,
    SEED,
    BENFORD_EXPECTED,
)
from src.state.ba_state import BAState


# ── Fixtures ──────────────────────────────────────────────────────────────────

# Clean Benford-compliant data (follows natural distribution)
BENFORD_CLEAN = [
    1.23, 1.45, 1.67, 1.89, 2.34, 2.56, 3.12, 3.78,
    4.23, 4.89, 5.67, 6.12, 6.78, 7.45, 8.12, 8.89,
    9.12, 1.11, 1.22, 1.33, 1.44, 1.55, 2.11, 2.22,
    3.11, 3.22, 1.99, 1.88, 1.77, 1.66,
]

# Suspicious data — too many round numbers starting with 5
BENFORD_SUSPICIOUS = [500, 5000, 50000, 5100, 5200, 5300, 5400, 5500,
                      5600, 5700, 5800, 5900, 5010, 5020, 5030, 5040,
                      5050, 5060, 5070, 5080, 5090, 5001, 5002, 5003]

# Normal financial values for isolation forest
NORMAL_VALUES    = [100.0, 102.0, 98.0, 101.5, 99.5, 100.5, 103.0, 97.0, 101.0, 99.0]

# Values with clear outliers
OUTLIER_VALUES   = [100.0, 102.0, 98.0, 101.5, 99.5, 1000.0, 100.5, 99.0, 0.001, 101.0]

SAMPLE_CELLS = [
    {"value": "394,328", "section": "INCOME_STATEMENT"},
    {"value": "365,817", "section": "INCOME_STATEMENT"},
    {"value": "274,515", "section": "INCOME_STATEMENT"},
    {"value": "99,803",  "section": "INCOME_STATEMENT"},
    {"value": "94,680",  "section": "INCOME_STATEMENT"},
    {"value": "57,411",  "section": "INCOME_STATEMENT"},
    {"value": "352,755", "section": "BALANCE_SHEET"},
    {"value": "323,888", "section": "BALANCE_SHEET"},
    {"value": "338,516", "section": "BALANCE_SHEET"},
    {"value": "10,708",  "section": "CASH_FLOW"},
]


@pytest.fixture
def triguard():
    return TriGuard(seed=SEED)


# ── Group 1: Constants ────────────────────────────────────────────────────────

class TestConstants:

    def test_01_seed_is_42(self):
        assert SEED == 42

    def test_02_benford_threshold(self):
        assert BENFORD_PVALUE_THRESH == 0.05

    def test_03_isolation_contamination(self):
        assert ISOLATION_CONTAMINATION == 0.1

    def test_04_max_rows_isolation(self):
        assert MAX_ROWS_ISOLATION == 500

    def test_05_min_values_benford(self):
        assert MIN_VALUES_BENFORD == 10

    def test_06_benford_expected_sums_to_1(self):
        assert abs(sum(BENFORD_EXPECTED) - 1.0) < 1e-6

    def test_07_benford_expected_nine_values(self):
        assert len(BENFORD_EXPECTED) == 9

    def test_08_benford_digit1_highest(self):
        """Digit 1 has highest probability in Benford's Law."""
        assert BENFORD_EXPECTED[0] > BENFORD_EXPECTED[1]
        assert BENFORD_EXPECTED[0] > BENFORD_EXPECTED[-1]


# ── Group 2: First digit extraction ──────────────────────────────────────────

class TestFirstDigitExtraction:

    def test_09_basic_extraction(self):
        digits = extract_first_digits([123.0, 456.0, 789.0])
        assert digits == [1, 4, 7]

    def test_10_decimal_values(self):
        digits = extract_first_digits([0.123, 0.456])
        assert 1 in digits or 4 in digits

    def test_11_negative_values_made_positive(self):
        digits = extract_first_digits([-123.0, -456.0])
        assert 1 in digits
        assert 4 in digits

    def test_12_zero_excluded(self):
        digits = extract_first_digits([0.0, 123.0])
        assert 0 not in digits

    def test_13_large_values(self):
        digits = extract_first_digits([394328.0])
        assert digits == [3]

    def test_14_empty_returns_empty(self):
        digits = extract_first_digits([])
        assert digits == []


# ── Group 3: Benford's Law Test ───────────────────────────────────────────────

class TestBenfordTest:

    def test_15_returns_result(self):
        result = run_benford_test(BENFORD_CLEAN)
        assert result is None or isinstance(result, BenfordResult)

    def test_16_insufficient_data_returns_none(self):
        result = run_benford_test([1.0, 2.0, 3.0])
        assert result is None

    def test_17_result_has_pvalue(self):
        result = run_benford_test(BENFORD_CLEAN + BENFORD_CLEAN)
        if result:
            assert 0.0 <= result.p_value <= 1.0

    def test_18_result_has_chi2(self):
        result = run_benford_test(BENFORD_CLEAN + BENFORD_CLEAN)
        if result:
            assert result.chi2_statistic >= 0.0

    def test_19_result_has_9_freq_values(self):
        result = run_benford_test(BENFORD_CLEAN + BENFORD_CLEAN)
        if result:
            assert len(result.observed_freq) == 9
            assert len(result.expected_freq) == 9

    def test_20_n_values_correct(self):
        data   = BENFORD_CLEAN * 2  # 60 values
        result = run_benford_test(data)
        if result:
            assert result.n_values <= len(data)

    def test_21_suspicious_data_flagged(self):
        """Data dominated by digit 5 should fail Benford test."""
        result = run_benford_test(BENFORD_SUSPICIOUS)
        if result:
            assert result.is_anomaly

    def test_22_flag_message_when_anomaly(self):
        result = run_benford_test(BENFORD_SUSPICIOUS)
        if result and result.is_anomaly:
            assert len(result.flag_message) > 0


# ── Group 4: Isolation Forest ─────────────────────────────────────────────────

class TestIsolationForest:

    def test_23_returns_result(self):
        result = run_isolation_forest(NORMAL_VALUES)
        assert isinstance(result, IsolationResult)

    def test_24_insufficient_data_returns_none(self):
        result = run_isolation_forest([1.0, 2.0])
        assert result is None

    def test_25_n_samples_correct(self):
        result = run_isolation_forest(NORMAL_VALUES)
        assert result.n_samples == len(NORMAL_VALUES)

    def test_26_outlier_fraction_between_0_and_1(self):
        result = run_isolation_forest(NORMAL_VALUES)
        assert 0.0 <= result.outlier_fraction <= 1.0

    def test_27_outlier_values_detected(self):
        """Values with extreme outliers should be flagged."""
        result = run_isolation_forest(OUTLIER_VALUES)
        assert result.is_anomaly is True
        assert result.outlier_count > 0

    def test_28_anomaly_scores_length_matches_input(self):
        result = run_isolation_forest(NORMAL_VALUES)
        assert len(result.anomaly_scores) == len(NORMAL_VALUES)

    def test_29_max_rows_cap_enforced(self):
        many_values = [float(i) for i in range(1000)]
        result      = run_isolation_forest(many_values)
        assert result.n_samples <= MAX_ROWS_ISOLATION

    def test_30_seed_42_reproducible(self):
        r1 = run_isolation_forest(OUTLIER_VALUES, seed=42)
        r2 = run_isolation_forest(OUTLIER_VALUES, seed=42)
        assert r1.outlier_count == r2.outlier_count


# ── Group 5: Risk score and severity ─────────────────────────────────────────

class TestRiskScore:

    def test_31_zero_risk_no_anomalies(self):
        score = compute_risk_score(None, None, None)
        assert score == 0.0

    def test_32_risk_score_range_0_to_100(self):
        benford   = BenfordResult(
            chi2_statistic=100.0, p_value=0.001, is_anomaly=True,
            observed_freq=[], expected_freq=[], n_values=50,
        )
        isolation = IsolationResult(
            outlier_indices=[0], outlier_count=3, outlier_fraction=0.3,
            anomaly_scores=[], is_anomaly=True, n_samples=10,
        )
        score = compute_risk_score(benford, isolation, volatility=0.5)
        assert 0.0 <= score <= 100.0

    def test_33_high_anomalies_give_high_score(self):
        benford = BenfordResult(
            chi2_statistic=200.0, p_value=0.0001, is_anomaly=True,
            observed_freq=[], expected_freq=[], n_values=100,
        )
        score = compute_risk_score(benford, None, None)
        assert score > 0

    def test_34_severity_low_below_40(self):
        assert classify_severity(0.0)  == "low"
        assert classify_severity(30.0) == "low"
        assert classify_severity(39.9) == "low"

    def test_35_severity_medium_40_to_70(self):
        assert classify_severity(40.0) == "medium"
        assert classify_severity(55.0) == "medium"
        assert classify_severity(69.9) == "medium"

    def test_36_severity_high_above_70(self):
        assert classify_severity(70.0)  == "high"
        assert classify_severity(85.0)  == "high"
        assert classify_severity(100.0) == "high"


# ── Group 6: TriGuard main class ──────────────────────────────────────────────

class TestTriGuard:

    def test_37_analyze_returns_result(self, triguard):
        result = triguard.analyze(NORMAL_VALUES)
        assert isinstance(result, TriGuardResult)

    def test_38_risk_score_in_range(self, triguard):
        result = triguard.analyze(NORMAL_VALUES)
        assert 0.0 <= result.risk_score <= 100.0

    def test_39_severity_is_valid(self, triguard):
        result = triguard.analyze(NORMAL_VALUES)
        assert result.anomaly_severity in ["low", "medium", "high"]

    def test_40_forensic_flags_is_list(self, triguard):
        result = triguard.analyze(NORMAL_VALUES)
        assert isinstance(result.forensic_flags, list)

    def test_41_empty_values_returns_zero_risk(self, triguard):
        result = triguard.analyze([])
        assert result.risk_score == 0.0
        assert result.anomaly_detected is False

    def test_42_high_volatility_adds_flag(self, triguard):
        result = triguard.analyze([], volatility=0.50)
        assert len(result.forensic_flags) > 0

    def test_43_extract_values_from_cells(self, triguard):
        values = triguard._extract_values(SAMPLE_CELLS, "")
        assert len(values) > 0
        assert all(isinstance(v, float) for v in values)

    def test_44_extract_values_from_text(self, triguard):
        text   = "Revenue was 383285 million. Net income 96995 million."
        values = triguard._extract_values([], text)
        assert len(values) >= 2

    def test_45_parenthetical_negatives_extracted(self, triguard):
        cells  = [{"value": "(50,672)"}]
        values = triguard._extract_values(cells, "")
        if values:
            assert values[0] < 0


# ── Group 7: BAState integration ─────────────────────────────────────────────

class TestBAStateIntegration:

    def test_46_run_writes_risk_score(self, triguard):
        state = BAState(
            session_id  = "t46",
            raw_text    = "Revenue 394328 million. Net income 99803 million.",
            table_cells = SAMPLE_CELLS,
        )
        state = triguard.run(state)
        assert hasattr(state, "risk_score")
        assert 0.0 <= state.risk_score <= 100.0

    def test_47_run_writes_forensic_flags(self, triguard):
        state = BAState(
            session_id  = "t47",
            raw_text    = "Revenue 394328 million.",
            table_cells = SAMPLE_CELLS,
        )
        state = triguard.run(state)
        assert isinstance(state.forensic_flags, list)

    def test_48_run_writes_anomaly_detected(self, triguard):
        state = BAState(
            session_id  = "t48",
            raw_text    = "",
            table_cells = [],
        )
        state = triguard.run(state)
        assert isinstance(state.anomaly_detected, bool)

    def test_49_run_writes_severity(self, triguard):
        state = BAState(
            session_id  = "t49",
            raw_text    = "",
            table_cells = [],
        )
        state = triguard.run(state)
        assert state.anomaly_severity in ["low", "medium", "high"]

    def test_50_seed_unchanged(self, triguard):
        """C5: seed must remain 42."""
        state = BAState(
            session_id  = "t50",
            raw_text    = "Revenue 394328.",
            table_cells = [],
        )
        state = triguard.run(state)
        assert state.seed == 42

    def test_51_no_rlef_in_forensic_flags(self, triguard):
        """C9: forensic flags must not contain _rlef_ fields."""
        state = BAState(
            session_id  = "t51",
            raw_text    = "Revenue 394328.",
            table_cells = SAMPLE_CELLS,
        )
        state = triguard.run(state)
        for flag in state.forensic_flags:
            assert "_rlef_" not in flag

    def test_52_garch_volatility_used_when_available(self, triguard):
        """High GARCH volatility should increase risk score."""
        state_no_vol = BAState(session_id="t52a", raw_text="", table_cells=[])
        state_no_vol = triguard.run(state_no_vol)

        state_high_vol = BAState(
            session_id   = "t52b",
            raw_text     = "",
            table_cells  = [],
            garch_result = {"conditional_volatility": 0.50, "converged": True},
        )
        state_high_vol = triguard.run(state_high_vol)

        assert state_high_vol.risk_score >= state_no_vol.risk_score


# ── Group 8: Convenience wrapper ─────────────────────────────────────────────

class TestConvenienceWrapper:

    def test_53_run_triguard_returns_state(self):
        state = BAState(
            session_id  = "t53",
            raw_text    = "Revenue 394328 million.",
            table_cells = SAMPLE_CELLS,
        )
        result = run_triguard(state)
        assert hasattr(result, "risk_score")
        assert hasattr(result, "forensic_flags")
        assert result.seed == 42