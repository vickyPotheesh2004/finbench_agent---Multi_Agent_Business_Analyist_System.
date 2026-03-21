"""
tests/test_lr_difficulty.py
FinBench Multi-Agent Business Analyst AI

Tests for N05 — LR Difficulty Predictor

24 tests covering:
  - Instantiation (tests 01-03)
  - Training (tests 04-06)
  - All 3 difficulty predictions (tests 07-09)
  - Confidence and config (tests 10-13)
  - BAState integration (tests 14-19)
  - Edge cases and N04+N05 pipeline (tests 20-24)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from src.routing.lr_difficulty import (
    LRDifficultyPredictor,
    DIFFICULTY_CONFIG,
    TRAINING_DATA,
)
from src.routing.cart_router import CARTRouter
from src.state.ba_state import BAState, Difficulty


# ── Module-level fixture — predictor trained once ────────────────────────────

@pytest.fixture(scope="module")
def trained_predictor(tmp_path_factory):
    """Train predictor once, reuse across all tests in module."""
    tmp        = tmp_path_factory.mktemp("lr_test")
    model_path = tmp / "lr_difficulty_test.pkl"
    predictor  = LRDifficultyPredictor(model_path=model_path)
    predictor.train()
    return predictor


@pytest.fixture(scope="module")
def trained_router(tmp_path_factory):
    """Train CART router once for pipeline tests."""
    tmp        = tmp_path_factory.mktemp("cart_for_lr")
    model_path = tmp / "cart_router_test.pkl"
    router     = CARTRouter(model_path=model_path)
    router.train()
    return router


# ════════════════════════════════════════════════════════════════════════════
# GROUP 1 — INSTANTIATION (tests 01-03)
# ════════════════════════════════════════════════════════════════════════════

class TestInstantiation:

    def test_01_predictor_instantiates(self):
        """N05: LRDifficultyPredictor must instantiate without error"""
        predictor = LRDifficultyPredictor()
        assert predictor is not None

    def test_02_not_loaded_by_default(self):
        """N05: Model must not be loaded until train() or load() called"""
        predictor = LRDifficultyPredictor()
        assert predictor.is_loaded() is False

    def test_03_training_data_has_150_questions(self):
        """N05: Built-in training data must have exactly 150 questions"""
        assert len(TRAINING_DATA) == 150


# ════════════════════════════════════════════════════════════════════════════
# GROUP 2 — TRAINING (tests 04-06)
# ════════════════════════════════════════════════════════════════════════════

class TestTraining:

    def test_04_train_returns_metrics(self, trained_predictor):
        """N05: train() must return metrics dict with required keys"""
        tmp_path = trained_predictor.model_path.parent / "retrain_test.pkl"
        p        = LRDifficultyPredictor(model_path=tmp_path)
        metrics  = p.train()
        assert "accuracy"     in metrics
        assert "n_samples"    in metrics
        assert "class_counts" in metrics
        assert "model_path"   in metrics
        assert "classes"      in metrics

    def test_05_training_accuracy_above_80(self, trained_predictor):
        """N05: Training accuracy must be >= 80% on 150 questions"""
        tmp_path = trained_predictor.model_path.parent / "acc_test.pkl"
        p        = LRDifficultyPredictor(model_path=tmp_path)
        metrics  = p.train()
        assert metrics["accuracy"] >= 0.80, \
            f"Training accuracy {metrics['accuracy']:.1%} below 80%"

    def test_06_class_balance_50_each(self, trained_predictor):
        """N05: Each class must have exactly 50 training examples"""
        tmp_path = trained_predictor.model_path.parent / "balance_test.pkl"
        p        = LRDifficultyPredictor(model_path=tmp_path)
        metrics  = p.train()
        counts   = metrics["class_counts"]
        for cls, count in counts.items():
            assert count == 50, \
                f"Class '{cls}' has {count} examples, expected 50"


# ════════════════════════════════════════════════════════════════════════════
# GROUP 3 — ALL 3 DIFFICULTY PREDICTIONS (tests 07-09)
# ════════════════════════════════════════════════════════════════════════════

class TestDifficultyPredictions:

    def test_07_easy_query_classified(self, trained_predictor):
        """N05: Single fact lookup must be classified as easy"""
        pred, conf = trained_predictor.predict(
            "What was Apple net income in FY2023?"
        )
        assert pred == "easy"
        assert conf > 0.5

    def test_08_medium_query_classified(self, trained_predictor):
        """N05: Ratio computation must be classified as medium"""
        pred, conf = trained_predictor.predict(
            "What was Apple gross margin percentage in FY2023?"
        )
        assert pred == "medium"
        assert conf > 0.5

    def test_09_hard_query_classified(self, trained_predictor):
        """N05: Multi-year comparison must be classified as hard"""
        pred, conf = trained_predictor.predict(
            "Compare Apple revenue growth across FY2021 FY2022 and FY2023 "
            "and explain all drivers in detail"
        )
        assert pred == "hard"
        assert conf > 0.5


# ════════════════════════════════════════════════════════════════════════════
# GROUP 4 — CONFIDENCE AND CONFIG (tests 10-13)
# ════════════════════════════════════════════════════════════════════════════

class TestConfidenceAndConfig:

    def test_10_confidence_is_float_between_0_and_1(self, trained_predictor):
        """N05: Confidence must be float between 0 and 1"""
        _, conf = trained_predictor.predict("What was Apple net income?")
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_11_hard_config_has_wider_context(self, trained_predictor):
        """N05: Hard config must have context_window_size=5"""
        config = trained_predictor.get_difficulty_config("hard")
        assert config["context_window_size"] == 5
        assert config["top_k"]               == 5

    def test_12_hard_config_has_more_retries(self, trained_predictor):
        """N05: Hard config must have piv_max_retries=5"""
        config = trained_predictor.get_difficulty_config("hard")
        assert config["piv_max_retries"] == 5

    def test_13_hard_config_lower_hitl_threshold(self, trained_predictor):
        """N05: Hard config must have lower hitl_threshold than easy"""
        easy_config = trained_predictor.get_difficulty_config("easy")
        hard_config = trained_predictor.get_difficulty_config("hard")
        assert hard_config["hitl_threshold"] < easy_config["hitl_threshold"]


# ════════════════════════════════════════════════════════════════════════════
# GROUP 5 — BASTATE INTEGRATION (tests 14-19)
# ════════════════════════════════════════════════════════════════════════════

class TestBAStateIntegration:

    def test_14_run_writes_query_difficulty(self, trained_predictor):
        """N05: run() must write query_difficulty to BAState"""
        state = BAState(
            session_id = "t14",
            query      = "What was Apple net income FY2023?",
        )
        state = trained_predictor.run(state)
        assert state.query_difficulty in [
            Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD
        ]

    def test_15_easy_query_does_not_change_context_window(self, trained_predictor):
        """N05: Easy query must NOT override context_window_size"""
        state = BAState(
            session_id          = "t15",
            query               = "What was Apple net income FY2023?",
            context_window_size = 3,
        )
        state = trained_predictor.run(state)
        # Easy should not override — context stays at 3
        if state.query_difficulty == Difficulty.EASY:
            assert state.context_window_size == 3

    def test_16_hard_query_widens_context_window(self, trained_predictor):
        """N05: Hard query must set context_window_size=5"""
        state = BAState(
            session_id          = "t16",
            query               = "Compare Apple revenue growth across FY2021 "
                                  "FY2022 FY2023 explain all drivers in detail",
            context_window_size = 3,
        )
        state = trained_predictor.run(state)
        if state.query_difficulty == Difficulty.HARD:
            assert state.context_window_size == 5

    def test_17_no_query_defaults_to_medium(self, trained_predictor):
        """N05: Missing query must default to medium difficulty"""
        state = BAState(session_id="t17")
        state = trained_predictor.run(state)
        assert state.query_difficulty == Difficulty.MEDIUM

    def test_18_seed_unchanged_after_run(self, trained_predictor):
        """C5: BAState seed must still be 42 after N05"""
        state = BAState(
            session_id = "t18",
            query      = "What was Apple net income?",
        )
        state = trained_predictor.run(state)
        assert state.seed == 42

    def test_19_three_classes_in_bastate(self, trained_predictor):
        """N05: All 3 Difficulty enum values must be reachable"""
        queries = [
            "What was Apple net income FY2023?",
            "What was Apple gross margin FY2023?",
            "Compare Apple revenue across FY2021 FY2022 FY2023 explain drivers",
        ]
        difficulties = set()
        for q in queries:
            state = BAState(session_id="t19", query=q)
            state = trained_predictor.run(state)
            difficulties.add(state.query_difficulty)
        # Should see all 3 difficulty levels
        assert len(difficulties) == 3


# ════════════════════════════════════════════════════════════════════════════
# GROUP 6 — EDGE CASES AND PIPELINE (tests 20-24)
# ════════════════════════════════════════════════════════════════════════════

class TestEdgeCasesAndPipeline:

    def test_20_batch_predict_returns_list(self, trained_predictor):
        """N05: predict_batch() must return list of (difficulty, conf) tuples"""
        queries = [
            "What was Apple net income?",
            "Calculate Apple gross margin",
            "Compare Apple across five years",
        ]
        results = trained_predictor.predict_batch(queries)
        assert isinstance(results, list)
        assert len(results) == 3
        for pred, conf in results:
            assert pred  in ["easy", "medium", "hard"]
            assert 0.0 <= conf <= 1.0

    def test_21_model_saves_to_disk(self, tmp_path):
        """N05: train() must save model file to disk"""
        model_path = tmp_path / "test_save_lr.pkl"
        p          = LRDifficultyPredictor(model_path=model_path)
        p.train()
        assert model_path.exists()
        assert model_path.stat().st_size > 0

    def test_22_model_loads_from_disk(self, tmp_path):
        """N05: load() must restore model and allow predictions"""
        model_path = tmp_path / "test_load_lr.pkl"
        p1         = LRDifficultyPredictor(model_path=model_path)
        p1.train()

        p2     = LRDifficultyPredictor(model_path=model_path)
        loaded = p2.load()
        assert loaded is True
        assert p2.is_loaded() is True

        pred, conf = p2.predict("What was Apple net income FY2023?")
        assert pred in ["easy", "medium", "hard"]
        assert conf > 0.0

    def test_23_n04_n05_pipeline_both_write_state(
        self, trained_router, trained_predictor
    ):
        """N04+N05: Both nodes must write their fields to BAState"""
        state = BAState(
            session_id = "t23",
            query      = "What was Apple total net sales FY2023?",
        )
        state = trained_router.run(state)
        state = trained_predictor.run(state)

        assert state.query_type       is not None
        assert state.query_difficulty is not None
        assert state.routing_path     != ""
        assert state.seed             == 42

    def test_24_n04_n05_numerical_easy_routing(
        self, trained_router, trained_predictor
    ):
        """N04+N05: Simple revenue question → numerical + easy"""
        state = BAState(
            session_id = "t24",
            query      = "What was Apple net income FY2023?",
        )
        state = trained_router.run(state)
        state = trained_predictor.run(state)

        # N04 should say numerical
        from src.state.ba_state import QueryType
        assert state.query_type == QueryType.NUMERICAL
        # N05 should say easy
        assert state.query_difficulty == Difficulty.EASY