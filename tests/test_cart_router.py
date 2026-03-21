"""
tests/test_cart_router.py
FinBench Multi-Agent Business Analyst AI

Tests for N04 — CART Query Router

24 tests covering:
  - Instantiation (tests 01-03)
  - Training (tests 04-06)
  - All 5 query type predictions (tests 07-11)
  - Confidence and routing config (tests 12-15)
  - BAState integration (tests 16-20)
  - Edge cases (tests 21-24)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pytest
import tempfile
from src.routing.cart_router import CARTRouter, ROUTING_CONFIG, TRAINING_DATA
from src.state.ba_state import BAState, QueryType


# ── Module-level fixture — router trained once ───────────────────────────────

@pytest.fixture(scope="module")
def trained_router(tmp_path_factory):
    """Train router once, reuse across all tests in module."""
    tmp = tmp_path_factory.mktemp("cart_test")
    model_path = tmp / "cart_router_test.pkl"
    router = CARTRouter(model_path=model_path)
    router.train()
    return router


# ════════════════════════════════════════════════════════════════════════════
# GROUP 1 — INSTANTIATION (tests 01-03)
# ════════════════════════════════════════════════════════════════════════════

class TestInstantiation:

    def test_01_router_instantiates(self):
        """N04: CARTRouter must instantiate without error"""
        router = CARTRouter()
        assert router is not None

    def test_02_not_loaded_by_default(self):
        """N04: Model must not be loaded until train() or load() called"""
        router = CARTRouter()
        assert router.is_loaded() is False

    def test_03_training_data_has_200_questions(self):
        """N04: Built-in training data must have exactly 200 questions"""
        assert len(TRAINING_DATA) == 200


# ════════════════════════════════════════════════════════════════════════════
# GROUP 2 — TRAINING (tests 04-06)
# ════════════════════════════════════════════════════════════════════════════

class TestTraining:

    def test_04_train_returns_metrics(self, trained_router):
        """N04: train() must return metrics dict with required keys"""
        # Re-train to get metrics (router already trained in fixture)
        tmp_path = trained_router.model_path.parent / "retrain_test.pkl"
        r = CARTRouter(model_path=tmp_path)
        metrics = r.train()
        assert "accuracy"     in metrics
        assert "n_samples"    in metrics
        assert "class_counts" in metrics
        assert "model_path"   in metrics
        assert "classes"      in metrics

    def test_05_training_accuracy_above_95(self, trained_router):
        """N04: Training accuracy must be >= 95% on 200 labelled questions"""
        tmp_path = trained_router.model_path.parent / "acc_test.pkl"
        r = CARTRouter(model_path=tmp_path)
        metrics = r.train()
        assert metrics["accuracy"] >= 0.95, \
            f"Training accuracy {metrics['accuracy']:.1%} below 95%"

    def test_06_class_balance_40_each(self, trained_router):
        """N04: Each class must have exactly 40 training examples"""
        tmp_path = trained_router.model_path.parent / "balance_test.pkl"
        r = CARTRouter(model_path=tmp_path)
        metrics = r.train()
        counts  = metrics["class_counts"]
        for cls, count in counts.items():
            assert count == 40, \
                f"Class '{cls}' has {count} examples, expected 40"


# ════════════════════════════════════════════════════════════════════════════
# GROUP 3 — ALL 5 QUERY TYPE PREDICTIONS (tests 07-11)
# ════════════════════════════════════════════════════════════════════════════

class TestQueryTypePredictions:

    def test_07_numerical_query_classified(self, trained_router):
        """N04: Numerical query must be classified as numerical"""
        pred, conf = trained_router.predict(
            "What was Apple total net sales in FY2023?"
        )
        assert pred == "numerical"
        assert conf > 0.5

    def test_08_ratio_query_classified(self, trained_router):
        """N04: Ratio query must be classified as ratio"""
        pred, conf = trained_router.predict(
            "What was Apple gross margin percentage in FY2023?"
        )
        assert pred == "ratio"
        assert conf > 0.5

    def test_09_multi_doc_query_classified(self, trained_router):
        """N04: Multi-doc query must be classified as multi_doc"""
        pred, conf = trained_router.predict(
            "Compare Apple and Microsoft revenue growth from 2021 to 2023"
        )
        assert pred == "multi_doc"
        assert conf > 0.5

    def test_10_text_query_classified(self, trained_router):
        """N04: Text query must be classified as text"""
        pred, conf = trained_router.predict(
            "What are Apple main risk factors disclosed in the 10-K?"
        )
        assert pred == "text"
        assert conf > 0.5

    def test_11_forensic_query_classified(self, trained_router):
        """N04: Forensic query must be classified as forensic"""
        pred, conf = trained_router.predict(
            "Are there any anomalies in Apple revenue recognition patterns?"
        )
        assert pred == "forensic"
        assert conf > 0.5


# ════════════════════════════════════════════════════════════════════════════
# GROUP 4 — CONFIDENCE AND ROUTING CONFIG (tests 12-15)
# ════════════════════════════════════════════════════════════════════════════

class TestConfidenceAndRouting:

    def test_12_confidence_is_float_between_0_and_1(self, trained_router):
        """N04: Confidence must be float between 0 and 1"""
        _, conf = trained_router.predict("What was Apple net income FY2023?")
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_13_numerical_routing_config_correct(self, trained_router):
        """N04: Numerical routing must set sniper_first=True"""
        config = trained_router.get_routing_config("numerical")
        assert config["sniper_first"]        is True
        assert config["context_window_size"] == 3
        assert "sniper" in config["routing_path"]

    def test_14_multi_doc_wider_context(self, trained_router):
        """N04: Multi-doc must use context_window_size=5"""
        config = trained_router.get_routing_config("multi_doc")
        assert config["context_window_size"] == 5

    def test_15_forensic_activates_triguard(self, trained_router):
        """N04: Forensic query must set triguard=True in routing config"""
        config = trained_router.get_routing_config("forensic")
        assert config["triguard"] is True


# ════════════════════════════════════════════════════════════════════════════
# GROUP 5 — BASTATE INTEGRATION (tests 16-20)
# ════════════════════════════════════════════════════════════════════════════

class TestBAStateIntegration:

    def test_16_run_writes_query_type(self, trained_router):
        """N04: run() must write query_type to BAState"""
        state = BAState(
            session_id = "t16",
            query      = "What was Apple net income FY2023?",
        )
        state = trained_router.run(state)
        assert state.query_type == QueryType.NUMERICAL

    def test_17_run_writes_routing_path(self, trained_router):
        """N04: run() must write routing_path to BAState"""
        state = BAState(
            session_id = "t17",
            query      = "What was Apple net income FY2023?",
        )
        state = trained_router.run(state)
        assert state.routing_path != ""
        assert isinstance(state.routing_path, str)

    def test_18_run_writes_context_window_size(self, trained_router):
        """N04: run() must write context_window_size to BAState"""
        state = BAState(
            session_id = "t18",
            query      = "Compare Apple and Microsoft revenue",
        )
        state = trained_router.run(state)
        assert state.context_window_size >= 3

    def test_19_seed_unchanged_after_run(self, trained_router):
        """C5: BAState seed must still be 42 after N04"""
        state = BAState(
            session_id = "t19",
            query      = "What was Apple revenue?",
        )
        state = trained_router.run(state)
        assert state.seed == 42

    def test_20_no_query_defaults_to_text(self, trained_router):
        """N04: Missing query must default to text type"""
        state = BAState(session_id="t20")
        state = trained_router.run(state)
        assert state.query_type == QueryType.TEXT


# ════════════════════════════════════════════════════════════════════════════
# GROUP 6 — EDGE CASES (tests 21-24)
# ════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_21_batch_predict_returns_list(self, trained_router):
        """N04: predict_batch() must return list of (type, conf) tuples"""
        queries = [
            "What was Apple net income?",
            "What was gross margin?",
            "Compare Apple and Microsoft",
        ]
        results = trained_router.predict_batch(queries)
        assert isinstance(results, list)
        assert len(results) == 3
        for pred, conf in results:
            assert isinstance(pred,  str)
            assert isinstance(conf,  float)
            assert 0.0 <= conf <= 1.0

    def test_22_model_saves_to_disk(self, tmp_path):
        """N04: train() must save model file to disk"""
        model_path = tmp_path / "test_save.pkl"
        router     = CARTRouter(model_path=model_path)
        router.train()
        assert model_path.exists()
        assert model_path.stat().st_size > 0

    def test_23_model_loads_from_disk(self, tmp_path):
        """N04: load() must restore model and allow predictions"""
        model_path = tmp_path / "test_load.pkl"
        r1 = CARTRouter(model_path=model_path)
        r1.train()

        r2     = CARTRouter(model_path=model_path)
        loaded = r2.load()
        assert loaded is True
        assert r2.is_loaded() is True

        pred, conf = r2.predict("What was Apple net income FY2023?")
        assert pred  == "numerical"
        assert conf  >  0.5

    def test_24_five_classes_present(self, trained_router):
        """N04: Model must output all 5 query type classes"""
        queries = [
            "What was Apple net income?",
            "What was gross margin?",
            "Compare Apple and Microsoft",
            "What are the main risk factors?",
            "Are there anomalies in the data?",
        ]
        preds = {trained_router.predict(q)[0] for q in queries}
        expected = {"numerical", "ratio", "multi_doc", "text", "forensic"}
        assert preds == expected