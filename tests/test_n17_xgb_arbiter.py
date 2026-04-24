"""
tests/test_n17_xgb_arbiter.py
Tests for N17 XGB Arbiter (Gate M6 protected)
PDR-BAAAI-001 Rev 1.0
35 tests - no trained model required (tests no-op path)
"""
import os
import pickle
import sqlite3
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.ml.xgb_arbiter import (
    XGBArbiter,
    run_xgb_arbiter,
    GATE_M6_MIN_DPO_PAIRS,
    SEED,
    N_FEATURES,
    MODEL_PATH,
    RLEF_DB_PATH,
)
from src.state.ba_state import BAState


@pytest.fixture
def tmp_model_path(tmp_path):
    return str(tmp_path / "xgb_model.pkl")


@pytest.fixture
def tmp_db_path(tmp_path):
    return str(tmp_path / "rlef.db")


@pytest.fixture
def arb(tmp_model_path, tmp_db_path):
    return XGBArbiter(
        model_path    = tmp_model_path,
        db_path       = tmp_db_path,
        min_dpo_pairs = 5,  # low threshold for tests
    )


def _seed_dpo_db(db_path: str, n_pairs: int) -> None:
    """Create minimal RLEF DB with N pairs for gate testing."""
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS dpo_pairs (
            id INTEGER PRIMARY KEY,
            prompt TEXT,
            chosen TEXT,
            rejected TEXT
        )
    """)
    for i in range(n_pairs):
        cur.execute(
            "INSERT INTO dpo_pairs (prompt, chosen, rejected) VALUES (?, ?, ?)",
            (f"q{i}", f"good{i}", f"bad{i}"),
        )
    conn.commit()
    conn.close()


def _make_state(**kwargs):
    defaults = {
        "session_id":         "t-test",
        "query":              "What was revenue?",
        "analyst_output":     "Revenue was $383B.",
        "analyst_confidence": 0.85,
        "analyst_citations":  ["INCOME_STATEMENT p.94"],
    }
    defaults.update(kwargs)
    return BAState(**defaults)


# Group 1: Constants
class TestConstants:

    def test_01_gate_m6_threshold_is_300(self):
        assert GATE_M6_MIN_DPO_PAIRS == 300

    def test_02_seed_is_42(self):
        assert SEED == 42

    def test_03_n_features_is_7(self):
        assert N_FEATURES == 7

    def test_04_model_path_defined(self):
        assert MODEL_PATH.endswith(".pkl")


# Group 2: Instantiation
class TestInstantiation:

    def test_05_creates_with_defaults(self):
        arb = XGBArbiter()
        assert arb is not None

    def test_06_custom_threshold(self, arb):
        assert arb.min_dpo_pairs == 5

    def test_07_model_none_initially(self, arb):
        assert arb._model is None


# Group 3: Gate M6 check
class TestGateM6:

    def test_08_not_ready_without_model(self, arb):
        assert arb.is_ready() is False

    def test_09_not_ready_without_dpo_pairs(self, arb, tmp_model_path):
        Path(tmp_model_path).write_bytes(b"placeholder")
        assert arb.is_ready() is False

    def test_10_ready_when_both_met(self, arb, tmp_model_path, tmp_db_path):
        # Fake model file with a pickleable placeholder
        with open(tmp_model_path, "wb") as f:
            pickle.dump({"model": "placeholder"}, f)
        # Seed DB with enough pairs
        _seed_dpo_db(tmp_db_path, n_pairs=10)
        assert arb.is_ready() is True

    def test_11_gate_status_returns_dict(self, arb):
        status = arb.gate_m6_status()
        assert isinstance(status, dict)
        assert "gate_name"      in status
        assert "dpo_pairs"      in status
        assert "dpo_required"   in status
        assert "overall_passed" in status

    def test_12_gate_status_failed_initially(self, arb):
        status = arb.gate_m6_status()
        assert status["overall_passed"] is False

    def test_13_blocker_describes_issue(self, arb):
        status = arb.gate_m6_status()
        assert "DPO" in status["blocker"] or "training" in status["blocker"]


# Group 4: No-op when gate not met
class TestNoOpMode:

    def test_14_run_returns_state_unchanged(self, arb):
        state   = _make_state()
        result  = arb.run(state)
        assert result is state

    def test_15_run_does_not_set_xgb_ranked(self, arb):
        state  = _make_state()
        result = arb.run(state)
        assert result.xgb_ranked_answer is None

    def test_16_run_does_not_set_xgb_score(self, arb):
        state  = _make_state()
        result = arb.run(state)
        assert result.xgb_score == 0.0

    def test_17_seed_unchanged_in_noop(self, arb):
        state  = _make_state()
        result = arb.run(state)
        assert result.seed == 42


# Group 5: Candidate collection
class TestCandidateCollection:

    def test_18_collects_analyst_candidate(self, arb):
        state = _make_state()
        cands = arb._collect_candidates(state)
        assert len(cands) >= 1
        assert any(c["pod"] == "analyst" for c in cands)

    def test_19_collects_quant_candidate(self, arb):
        state = _make_state(
            quant_result     = "Revenue = $383B per calc.",
            quant_confidence = 0.90,
        )
        cands = arb._collect_candidates(state)
        assert any(c["pod"] == "quant" for c in cands)

    def test_20_collects_auditor_candidate(self, arb):
        state = _make_state(
            auditor_output     = "Verified: $383B.",
            auditor_confidence = 0.88,
        )
        cands = arb._collect_candidates(state)
        assert any(c["pod"] == "auditor" for c in cands)

    def test_21_empty_outputs_no_candidates(self, arb):
        state = _make_state(
            analyst_output = "",
            quant_result   = "",
            auditor_output = "",
        )
        cands = arb._collect_candidates(state)
        assert len(cands) == 0


# Group 6: Feature extraction
class TestFeatureExtraction:

    def test_22_returns_seven_features(self, arb):
        state = _make_state()
        cand  = arb._collect_candidates(state)[0]
        feats = arb._extract_features(cand, state)
        assert len(feats) == N_FEATURES

    def test_23_all_features_are_floats(self, arb):
        state = _make_state()
        cand  = arb._collect_candidates(state)[0]
        feats = arb._extract_features(cand, state)
        for f in feats:
            assert isinstance(f, float)

    def test_24_confidence_feature_first(self, arb):
        state = _make_state(analyst_confidence=0.77)
        cand  = arb._collect_candidates(state)[0]
        feats = arb._extract_features(cand, state)
        assert feats[0] == 0.77

    def test_25_citation_feature_normalised(self, arb):
        state = _make_state(analyst_citations=["a", "b", "c"])
        cand  = arb._collect_candidates(state)[0]
        feats = arb._extract_features(cand, state)
        assert 0.0 <= feats[1] <= 1.0

    def test_26_length_feature_normalised(self, arb):
        state = _make_state()
        cand  = arb._collect_candidates(state)[0]
        feats = arb._extract_features(cand, state)
        assert 0.0 <= feats[2] <= 1.0


# Group 7: Helpers
class TestHelpers:

    def test_27_is_numeric_detects_number(self):
        assert XGBArbiter._is_numeric("383")
        assert XGBArbiter._is_numeric("383,285")
        assert XGBArbiter._is_numeric("$383")
        assert XGBArbiter._is_numeric("44.1%")

    def test_28_is_numeric_rejects_words(self):
        assert not XGBArbiter._is_numeric("revenue")
        assert not XGBArbiter._is_numeric("the")

    def test_29_word_overlap_identical(self):
        overlap = XGBArbiter._word_overlap(
            "revenue income earnings",
            "revenue income earnings statement",
        )
        assert overlap == 1.0

    def test_30_word_overlap_zero(self):
        overlap = XGBArbiter._word_overlap(
            "revenue income",
            "completely unrelated content",
        )
        assert overlap == 0.0

    def test_31_word_overlap_empty_safe(self):
        assert XGBArbiter._word_overlap("", "anything") == 0.0
        assert XGBArbiter._word_overlap("anything", "") == 0.0


# Group 8: DPO counting
class TestDPOCounting:

    def test_32_zero_when_db_missing(self, arb):
        assert arb._count_dpo_pairs() == 0

    def test_33_counts_correctly(self, tmp_db_path, tmp_model_path):
        _seed_dpo_db(tmp_db_path, n_pairs=12)
        arb = XGBArbiter(model_path=tmp_model_path, db_path=tmp_db_path)
        assert arb._count_dpo_pairs() == 12


# Group 9: Convenience wrapper
class TestWrapper:

    def test_34_wrapper_returns_state(self):
        state  = _make_state()
        result = run_xgb_arbiter(state)
        assert hasattr(result, "xgb_score")

    def test_35_no_rlef_in_output(self, arb):
        state  = _make_state()
        result = arb.run(state)
        assert "_rlef_" not in str(result.xgb_ranked_answer)