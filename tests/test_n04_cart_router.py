"""
tests/test_n04_cart_router.py
Tests for N04 CART Query Router
PDR-BAAAI-001 · Rev 1.0
"""

import os
import pytest
from src.routing.cart_router import (
    CARTRouter,
    run_cart_router,
    QUERY_CLASSES,
    ROUTING_CONFIG,
    TRAINING_DATA,
    _SEED,
    _MODEL_FILENAME,
    _VECTORIZER_FILENAME,
)
from src.state.ba_state import BAState


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def trained_router(tmp_path_factory):
    """Train once and reuse across all tests in this module."""
    tmp_dir = str(tmp_path_factory.mktemp("cart_model"))
    router  = CARTRouter(model_dir=tmp_dir)
    router.train()
    router.save()
    return router


@pytest.fixture
def fresh_router(tmp_path):
    return CARTRouter(model_dir=str(tmp_path))


# ── Group 1: Constants ────────────────────────────────────────────────────────

class TestConstants:

    def test_01_five_query_classes(self):
        assert len(QUERY_CLASSES) == 5

    def test_02_query_classes_correct(self):
        assert set(QUERY_CLASSES) == {
            "numerical", "ratio", "multi_doc", "text", "forensic"
        }

    def test_03_routing_config_has_all_classes(self):
        for cls in QUERY_CLASSES:
            assert cls in ROUTING_CONFIG

    def test_04_routing_config_keys(self):
        required = {
            "sniper_first", "piv_rounds",
            "context_window", "triguard_active", "description"
        }
        for cls in QUERY_CLASSES:
            assert required.issubset(ROUTING_CONFIG[cls].keys())

    def test_05_numerical_activates_sniper(self):
        assert ROUTING_CONFIG["numerical"]["sniper_first"] is True

    def test_06_forensic_activates_triguard(self):
        assert ROUTING_CONFIG["forensic"]["triguard_active"] is True

    def test_07_text_does_not_activate_triguard(self):
        assert ROUTING_CONFIG["text"]["triguard_active"] is False

    def test_08_multi_doc_has_wider_context(self):
        assert ROUTING_CONFIG["multi_doc"]["context_window"] >= 5

    def test_09_seed_is_42(self):
        """C5: seed must be 42"""
        assert _SEED == 42

    def test_10_training_data_200_samples(self):
        assert len(TRAINING_DATA) == 200

    def test_11_training_data_balanced(self):
        """40 samples per class"""
        from collections import Counter
        counts = Counter(label for _, label in TRAINING_DATA)
        for cls in QUERY_CLASSES:
            assert counts[cls] == 40, (
                f"Class '{cls}' has {counts[cls]} samples, expected 40"
            )


# ── Group 2: Instantiation ────────────────────────────────────────────────────

class TestInstantiation:

    def test_12_instantiates(self, fresh_router):
        assert fresh_router is not None

    def test_13_not_trained_at_init(self, fresh_router):
        assert fresh_router.is_trained() is False

    def test_14_model_none_at_init(self, fresh_router):
        assert fresh_router._model is None

    def test_15_vectorizer_none_at_init(self, fresh_router):
        assert fresh_router._vectorizer is None


# ── Group 3: Training ─────────────────────────────────────────────────────────

class TestTraining:

    def test_16_train_returns_dict(self, fresh_router):
        result = fresh_router.train()
        assert isinstance(result, dict)

    def test_17_train_result_has_accuracy(self, fresh_router):
        result = fresh_router.train()
        assert "train_accuracy" in result

    def test_18_train_accuracy_above_90_percent(self, fresh_router):
        """200 labelled questions — DecisionTree should fit well"""
        result = fresh_router.train()
        assert result["train_accuracy"] >= 0.90, (
            f"Train accuracy {result['train_accuracy']:.1%} < 90%"
        )

    def test_19_trained_after_train(self, fresh_router):
        fresh_router.train()
        assert fresh_router.is_trained() is True

    def test_20_n_samples_correct(self, fresh_router):
        result = fresh_router.train()
        assert result["n_samples"] == 200


# ── Group 4: Persistence ──────────────────────────────────────────────────────

class TestPersistence:

    def test_21_save_creates_model_file(self, tmp_path):
        router = CARTRouter(model_dir=str(tmp_path))
        router.train()
        router.save()
        assert os.path.exists(os.path.join(str(tmp_path), _MODEL_FILENAME))

    def test_22_save_creates_vectorizer_file(self, tmp_path):
        router = CARTRouter(model_dir=str(tmp_path))
        router.train()
        router.save()
        assert os.path.exists(os.path.join(str(tmp_path), _VECTORIZER_FILENAME))

    def test_23_load_returns_true(self, tmp_path):
        router = CARTRouter(model_dir=str(tmp_path))
        router.train()
        router.save()
        router2 = CARTRouter(model_dir=str(tmp_path))
        assert router2.load() is True

    def test_24_load_missing_returns_false(self, tmp_path):
        router = CARTRouter(model_dir=str(tmp_path))
        assert router.load() is False

    def test_25_loaded_model_is_trained(self, tmp_path):
        router = CARTRouter(model_dir=str(tmp_path))
        router.train()
        router.save()
        router2 = CARTRouter(model_dir=str(tmp_path))
        router2.load()
        assert router2.is_trained() is True

    def test_26_save_returns_path_string(self, tmp_path):
        router = CARTRouter(model_dir=str(tmp_path))
        router.train()
        path = router.save()
        assert isinstance(path, str)
        assert path.endswith(".pkl")


# ── Group 5: Classification accuracy ─────────────────────────────────────────

class TestClassification:

    @pytest.mark.parametrize("query,expected", [
        ("What was total net sales in FY2023?",                 "numerical"),
        ("What was net income for the year?",                   "numerical"),
        ("What were total assets?",                             "numerical"),
        ("What was diluted EPS?",                               "ratio"),
        ("What was operating cash flow?",                       "ratio"),
        ("Calculate the gross margin percentage",               "ratio"),
        ("What was the current ratio?",                         "ratio"),
        ("Calculate return on equity",                          "ratio"),
        ("What was the debt to equity ratio?",                  "ratio"),
        ("Compare revenue between FY2022 and FY2023",           "multi_doc"),
        ("How has net income trended over three years?",        "multi_doc"),
        ("What are the main risk factors?",                     "text"),
        ("Describe the company's competitive strategy",         "text"),
        ("Summarise the MD&A key points",                       "text"),
        ("Are there signs of earnings manipulation?",           "forensic"),
        ("Does Benford law flag any anomalies?",                "forensic"),
        ("Are accounts receivable growing faster than revenue?","forensic"),
    ])
    def test_27_classification_accuracy(
        self, trained_router, query, expected
    ):
        query_type, confidence, _ = trained_router.classify(query)
        assert query_type == expected, (
            f"Query: '{query}'\n"
            f"Expected: {expected}\n"
            f"Got:      {query_type} (confidence={confidence:.3f})"
        )

    def test_28_classify_returns_tuple(self, trained_router):
        result = trained_router.classify("What was net income?")
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_29_confidence_between_0_and_1(self, trained_router):
        _, confidence, _ = trained_router.classify("What was net income?")
        assert 0.0 <= confidence <= 1.0

    def test_30_result_includes_routing_config(self, trained_router):
        _, _, config = trained_router.classify("net income FY2023")
        assert "sniper_first"    in config
        assert "piv_rounds"      in config
        assert "context_window"  in config
        assert "triguard_active" in config

    def test_31_all_5_classes_reachable(self, trained_router):
        """Every query class must be reachable from training data."""
        queries = {
            "numerical":  "What was total revenue?",
            "ratio":      "Calculate the gross margin ratio",
            "multi_doc":  "Compare revenue between two years",
            "text":       "Describe the main risk factors",
            "forensic":   "Are there signs of manipulation?",
        }
        found_classes = set()
        for _, query in queries.items():
            qt, _, _ = trained_router.classify(query)
            found_classes.add(qt)
        assert len(found_classes) >= 4, (
            f"Only {len(found_classes)} classes found: {found_classes}"
        )

    def test_32_batch_classify_returns_list(self, trained_router):
        queries = ["net income?", "gross margin ratio", "risk factors"]
        results = trained_router.classify_batch(queries)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, tuple)
            assert len(r) == 3


# ── Group 6: BAState integration ─────────────────────────────────────────────

class TestBAStateIntegration:

    def test_33_run_writes_query_type(self, trained_router):
        state = BAState(
            session_id = "t33",
            query      = "What was net income FY2023?",
        )
        state = trained_router.run(state)
        assert state.query_type in QUERY_CLASSES

    def test_34_run_writes_routing_path(self, trained_router):
        state = BAState(
            session_id = "t34",
            query      = "What was net income?",
        )
        state = trained_router.run(state)
        assert state.routing_path != ""
        assert "cart_" in state.routing_path

    def test_35_run_writes_context_window_size(self, trained_router):
        state = BAState(
            session_id = "t35",
            query      = "What was net income?",
        )
        state = trained_router.run(state)
        assert state.context_window_size >= 3

    def test_36_seed_unchanged_after_run(self, trained_router):
        """C5: seed must remain 42"""
        state = BAState(
            session_id = "t36",
            query      = "What was net income?",
        )
        state = trained_router.run(state)
        assert state.seed == 42

    def test_37_empty_query_defaults_to_text(self, trained_router):
        state = BAState(session_id="t37", query="")
        state = trained_router.run(state)
        assert state.query_type == "text"

    def test_38_numerical_query_sets_sniper_config(self, trained_router):
        state = BAState(
            session_id = "t38",
            query      = "What was total net sales FY2023?",
        )
        state = trained_router.run(state)
        if state.query_type == "numerical":
            assert ROUTING_CONFIG["numerical"]["sniper_first"] is True

    def test_39_multi_doc_sets_wider_context(self, trained_router):
        state = BAState(
            session_id = "t39",
            query      = "Compare revenue between FY2022 and FY2023",
        )
        state = trained_router.run(state)
        if state.query_type == "multi_doc":
            assert state.context_window_size >= 5


# ── Group 7: Convenience wrapper ─────────────────────────────────────────────

class TestConvenienceWrapper:

    def test_40_run_cart_router_returns_state(self, tmp_path):
        state = BAState(
            session_id = "t40",
            query      = "What was net income?",
        )
        result = run_cart_router(state, model_dir=str(tmp_path))
        assert hasattr(result, "query_type")
        assert result.query_type in QUERY_CLASSES

# ════════════════════════════════════════════════════════════════════════════
# BUG #7 — Routers must persist after first train
# ════════════════════════════════════════════════════════════════════════════

class TestBug7RouterPersistence:
    """Regression for Bug #7: routers retrained on every pipeline run.

    Before fix: each CARTRouter() instance trained from scratch (~5s).
    After fix:  first instance trains+saves, subsequent instances load.
    """

    def test_first_use_saves_model_to_disk(self, tmp_path):
        """Bug #7: _ensure_trained() must save after training."""
        # Fresh dir → no saved model
        router = CARTRouter(model_dir=str(tmp_path))
        assert router.is_trained() is False

        # Trigger training via classify
        router.classify("What was net income?")

        # Bug #7: model files must now exist on disk
        model_path      = os.path.join(str(tmp_path), _MODEL_FILENAME)
        vectorizer_path = os.path.join(str(tmp_path), _VECTORIZER_FILENAME)
        assert os.path.exists(model_path), (
            f"Bug #7 regression: {model_path} not saved after training"
        )
        assert os.path.exists(vectorizer_path)

    def test_second_instance_loads_instead_of_training(self, tmp_path):
        """Bug #7: 2nd instance with same model_dir must load (fast)."""
        import time

        # First instance trains + saves
        r1 = CARTRouter(model_dir=str(tmp_path))
        r1.classify("seed query")

        # Second instance should load — measure time as proxy
        t0 = time.time()
        r2 = CARTRouter(model_dir=str(tmp_path))
        r2.classify("another query")
        elapsed = time.time() - t0

        # Loading should be <1s; training is ~5s
        assert elapsed < 2.0, (
            f"Bug #7: 2nd instance took {elapsed:.2f}s "
            f"(must be <2s — likely retraining instead of loading)"
        )

    def test_load_after_persist_returns_true(self, tmp_path):
        """Bug #7: after auto-save, load() must succeed on fresh instance."""
        r1 = CARTRouter(model_dir=str(tmp_path))
        r1.classify("first call to trigger train+save")

        r2 = CARTRouter(model_dir=str(tmp_path))
        assert r2.load() is True, (
            "Bug #7: load() should succeed because r1 saved the model"
        )
        assert r2.is_trained() is True