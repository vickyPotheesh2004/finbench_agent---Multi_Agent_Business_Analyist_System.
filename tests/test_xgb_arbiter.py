"""
tests/test_xgb_arbiter.py
FinBench Multi-Agent Business Analyst AI

Tests for N17 -- XGBoost Arbiter

No LLM needed -- pure ML, deterministic with seed=42.

24 tests covering:
  - Instantiation (tests 01-03)
  - Stub mode Gate M6 (tests 04-07)
  - Feature extraction (tests 08-13)
  - Active mode (tests 14-17)
  - BAState integration (tests 18-22)
  - Edge cases (tests 23-24)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pytest
import numpy as np

from src.ml.xgb_arbiter import (
    XGBArbiter,
    GATE_M6_MIN_PAIRS,
    FEATURE_NAMES,
    MIN_ANSWER_LEN,
    MAX_ANSWER_LEN,
)
from src.state.ba_state import BAState, PIVStatus, QueryType


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_state_full() -> BAState:
    return BAState(
        session_id           = "test-n17-full",
        query                = "What was Apple net income FY2023?",
        company_name         = "Apple Inc",
        fiscal_year          = "FY2023",
        final_answer_pre_xgb = "Net income $96,995M [FS/42].",
        confidence_score     = 0.88,
        analyst_output       = "Net income $96,995M [FS/42].",
        analyst_confidence   = 0.92,
        analyst_piv_status   = PIVStatus.PASS,
        analyst_citations    = ["Financial Statements / Page 42"],
        quant_result         = "Net income $96,995 million [FS/42].",
        quant_confidence     = 0.85,
        quant_piv_status     = PIVStatus.PASS,
        quant_citations      = ["Financial Statements / Page 42"],
        auditor_output       = "Net income was $96,995M [FS/42].",
        auditor_confidence   = 0.88,
        auditor_piv_status   = PIVStatus.PASS,
        auditor_citations    = ["Financial Statements / Page 42"],
        retrieval_stage_2    = [{
            "text":    "Net income $96,995M FY2023. Revenue $383,285M.",
            "section": "Financial Statements",
            "page":    "42",
        }],
    )

def make_state_simple() -> BAState:
    return BAState(
        session_id           = "test-n17-simple",
        query                = "What was Apple net income FY2023?",
        final_answer_pre_xgb = "Net income $96,995M [FS/42].",
        confidence_score     = 0.88,
        analyst_output       = "Net income $96,995M [FS/42].",
        analyst_confidence   = 0.92,
        analyst_piv_status   = PIVStatus.PASS,
        analyst_citations    = ["Financial Statements / Page 42"],
        retrieval_stage_2    = [{
            "text":    "Net income $96,995M FY2023.",
            "section": "Financial Statements",
            "page":    "42",
        }],
    )

@pytest.fixture(scope="module")
def stub_arbiter():
    """Arbiter in stub mode -- Gate M6 not passed."""
    return XGBArbiter(dpo_pair_count=0)

@pytest.fixture(scope="module")
def trained_arbiter(tmp_path_factory):
    """Arbiter with trained model -- simulates Gate M6 passed."""
    model_path = tmp_path_factory.mktemp("models") / "xgb_arbiter.pkl"
    arbiter    = XGBArbiter(dpo_pair_count=0, model_path=model_path)
    X, y       = arbiter.build_synthetic_training_data(200)
    arbiter.train(X, y)
    arbiter.gate_m6_passed = True
    return arbiter


# ════════════════════════════════════════════════════════════════════════════
# GROUP 1 -- INSTANTIATION (tests 01-03)
# ════════════════════════════════════════════════════════════════════════════

class TestInstantiation:

    def test_01_arbiter_instantiates(self, stub_arbiter):
        """N17: XGBArbiter must instantiate without error"""
        assert stub_arbiter is not None

    def test_02_gate_m6_not_passed_at_zero_pairs(self, stub_arbiter):
        """N17: Gate M6 must not be passed at 0 DPO pairs"""
        assert stub_arbiter.gate_m6_passed is False

    def test_03_gate_m6_passed_at_300_pairs(self):
        """N17: Gate M6 must be passed at >= 300 DPO pairs"""
        arbiter = XGBArbiter(dpo_pair_count=300)
        assert arbiter.gate_m6_passed is True


# ════════════════════════════════════════════════════════════════════════════
# GROUP 2 -- STUB MODE (tests 04-07)
# ════════════════════════════════════════════════════════════════════════════

class TestStubMode:

    def test_04_stub_passes_final_answer_through(self, stub_arbiter):
        """N17: Stub mode must pass final_answer_pre_xgb through unchanged"""
        state = make_state_simple()
        state = stub_arbiter.run(state)
        assert state.final_answer == "Net income $96,995M [FS/42]."

    def test_05_stub_sets_xgb_score(self, stub_arbiter):
        """N17: Stub mode must set xgb_score from confidence_score"""
        state = make_state_simple()
        state = stub_arbiter.run(state)
        assert state.xgb_score == pytest.approx(0.88, abs=0.01)

    def test_06_stub_sets_xgb_ranked_answer(self, stub_arbiter):
        """N17: Stub mode must set xgb_ranked_answer"""
        state = make_state_simple()
        state = stub_arbiter.run(state)
        assert state.xgb_ranked_answer != ""

    def test_07_stub_final_answer_equals_pre_xgb(self, stub_arbiter):
        """N17: In stub mode final_answer must equal final_answer_pre_xgb"""
        state = make_state_simple()
        state = stub_arbiter.run(state)
        assert state.final_answer == state.xgb_ranked_answer


# ════════════════════════════════════════════════════════════════════════════
# GROUP 3 -- FEATURE EXTRACTION (tests 08-13)
# ════════════════════════════════════════════════════════════════════════════

class TestFeatureExtraction:

    def test_08_extracts_8_features(self, stub_arbiter):
        """N17: _extract_features must return list of 8 values"""
        state     = make_state_simple()
        candidate = {
            "pod":        "analyst",
            "answer":     "Net income $96,995M [FS/42].",
            "confidence": 0.92,
            "retries":    0,
            "citations":  ["Financial Statements / Page 42"],
        }
        features = stub_arbiter._extract_features(candidate, state)
        assert len(features) == 8

    def test_09_features_in_0_1_range(self, stub_arbiter):
        """N17: All features must be in 0-1 range"""
        state     = make_state_simple()
        candidate = {
            "pod":        "analyst",
            "answer":     "Net income $96,995M [FS/42].",
            "confidence": 0.92,
            "retries":    0,
            "citations":  ["Financial Statements / Page 42"],
        }
        features = stub_arbiter._extract_features(candidate, state)
        assert all(0.0 <= f <= 1.0 for f in features)

    def test_10_citation_present_feature_correct(self, stub_arbiter):
        """N17: citation_present (F2) must be 1.0 when citations exist"""
        state     = make_state_simple()
        candidate = {
            "pod":        "analyst",
            "answer":     "Net income $96,995M.",
            "confidence": 0.9,
            "retries":    0,
            "citations":  ["Financial Statements / Page 42"],
        }
        features = stub_arbiter._extract_features(candidate, state)
        assert features[1] == 1.0

    def test_11_no_citation_feature_zero(self, stub_arbiter):
        """N17: citation_present (F2) must be 0.0 when no citations"""
        state     = make_state_simple()
        candidate = {
            "pod":        "analyst",
            "answer":     "Net income $96,995M.",
            "confidence": 0.9,
            "retries":    0,
            "citations":  [],
        }
        features = stub_arbiter._extract_features(candidate, state)
        assert features[1] == 0.0

    def test_12_retry_penalty_applied(self, stub_arbiter):
        """N17: retry_penalty (F8) must decrease with more retries"""
        state = make_state_simple()
        cand0 = {"pod":"a","answer":"x $100M","confidence":0.9,
                 "retries":0,"citations":[]}
        cand3 = {"pod":"a","answer":"x $100M","confidence":0.9,
                 "retries":3,"citations":[]}
        f0 = stub_arbiter._extract_features(cand0, state)
        f3 = stub_arbiter._extract_features(cand3, state)
        assert f0[7] > f3[7]

    def test_13_feature_names_count(self, stub_arbiter):
        """N17: Must have exactly 8 feature names"""
        assert len(FEATURE_NAMES) == 8


# ════════════════════════════════════════════════════════════════════════════
# GROUP 4 -- ACTIVE MODE (tests 14-17)
# ════════════════════════════════════════════════════════════════════════════

class TestActiveMode:

    def test_14_trained_model_produces_answer(self, trained_arbiter):
        """N17: Active mode must produce non-empty final_answer"""
        state = make_state_full()
        state = trained_arbiter.run(state)
        assert state.final_answer != ""

    def test_15_xgb_score_in_range(self, trained_arbiter):
        """N17: Active mode xgb_score must be 0-1"""
        state = make_state_full()
        state = trained_arbiter.run(state)
        assert 0.0 <= state.xgb_score <= 1.0

    def test_16_synthetic_training_data_shape(self, stub_arbiter):
        """N17: Synthetic training data must have shape (n, 8)"""
        X, y = stub_arbiter.build_synthetic_training_data(100)
        assert X.shape == (100, 8)
        assert len(y)  == 100

    def test_17_training_returns_accuracy(self, tmp_path):
        """N17: train() must return float accuracy > 0"""
        arbiter = XGBArbiter(
            dpo_pair_count = 0,
            model_path     = tmp_path / "test_model.pkl",
        )
        X, y    = arbiter.build_synthetic_training_data(100)
        val_acc = arbiter.train(X, y)
        assert isinstance(val_acc, float)
        assert val_acc > 0.0


# ════════════════════════════════════════════════════════════════════════════
# GROUP 5 -- BASTATE INTEGRATION (tests 18-22)
# ════════════════════════════════════════════════════════════════════════════

class TestBAStateIntegration:

    def test_18_run_writes_final_answer(self, stub_arbiter):
        """N17: run() must write final_answer to BAState"""
        state = make_state_simple()
        state = stub_arbiter.run(state)
        assert isinstance(state.final_answer, str)
        assert state.final_answer != ""

    def test_19_run_writes_xgb_ranked_answer(self, stub_arbiter):
        """N17: run() must write xgb_ranked_answer to BAState"""
        state = make_state_simple()
        state = stub_arbiter.run(state)
        assert isinstance(state.xgb_ranked_answer, str)

    def test_20_run_writes_xgb_score(self, stub_arbiter):
        """N17: run() must write xgb_score 0-1 to BAState"""
        state = make_state_simple()
        state = stub_arbiter.run(state)
        assert 0.0 <= state.xgb_score <= 1.0

    def test_21_seed_unchanged_after_run(self, stub_arbiter):
        """C5: BAState seed must still be 42 after N17"""
        state = make_state_simple()
        state = stub_arbiter.run(state)
        assert state.seed == 42

    def test_22_empty_state_handled(self, stub_arbiter):
        """N17: Empty BAState must produce empty final_answer"""
        state = BAState(session_id="t22-empty")
        state = stub_arbiter.run(state)
        assert state.final_answer      == ""
        assert state.xgb_ranked_answer == ""
        assert state.xgb_score         == 0.0


# ════════════════════════════════════════════════════════════════════════════
# GROUP 6 -- EDGE CASES (tests 23-24)
# ════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_23_gate_m6_constant_correct(self, stub_arbiter):
        """N17: GATE_M6_MIN_PAIRS must be 300"""
        assert GATE_M6_MIN_PAIRS == 300

    def test_24_number_extraction(self, stub_arbiter):
        """N17: _extract_numbers must parse financial figures"""
        nums = stub_arbiter._extract_numbers(
            "Net income was $96,995 million. Revenue $383,285M."
        )
        assert len(nums) >= 2
        assert 96995.0 in nums