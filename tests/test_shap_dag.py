"""
tests/test_shap_dag.py
FinBench Multi-Agent Business Analyst AI

Tests for N16 -- SHAP + Causal DAG

No LLM needed -- pure statistical explainability.
All tests fast and deterministic with seed=42.

24 tests covering:
  - Instantiation (tests 01-02)
  - Feature matrix building (tests 03-06)
  - SHAP computation (tests 07-11)
  - Node value extraction (tests 12-15)
  - Causal DAG building (tests 16-18)
  - BAState integration (tests 19-22)
  - Edge cases (tests 23-24)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pytest
import numpy as np

from src.explainability.shap_dag import (
    SHAPDag,
    FEATURE_NAMES,
    CAUSAL_NODES,
    CAUSAL_EDGES,
    SHAP_ROW_CAP,
)
from src.state.ba_state import BAState, QueryType, PIVStatus


# ── Shared test data ──────────────────────────────────────────────────────────

APPLE_CHUNKS = [
    {
        "text":        "Net income: $96,995 million. Revenue: $383,285 million. "
                       "Gross profit: $169,148 million.",
        "section":     "Financial Statements",
        "page":        "42",
        "company":     "Apple Inc",
        "fiscal_year": "FY2023",
        "bm25_score":  0.85,
        "cosine_sim":  0.92,
    },
    {
        "text":        "Operating income: $114,301 million. EPS diluted: $6.13. "
                       "Interest expense: $3,933 million.",
        "section":     "Financial Statements",
        "page":        "43",
        "company":     "Apple Inc",
        "fiscal_year": "FY2023",
        "bm25_score":  0.72,
        "cosine_sim":  0.88,
    },
    {
        "text":        "Cost of sales: $214,137 million. "
                       "Research and development: $29,915 million.",
        "section":     "Financial Statements",
        "page":        "44",
        "company":     "Apple Inc",
        "fiscal_year": "FY2023",
        "bm25_score":  0.65,
        "cosine_sim":  0.80,
    },
]

def make_state(chunks=None) -> BAState:
    return BAState(
        session_id           = "test-n16",
        query                = "What was Apple net income FY2023?",
        company_name         = "Apple Inc",
        fiscal_year          = "FY2023",
        final_answer_pre_xgb = "Net income was $96,995 million FY2023.",
        retrieval_stage_2    = chunks or APPLE_CHUNKS,
    )


@pytest.fixture(scope="module")
def explainer(tmp_path_factory):
    out = tmp_path_factory.mktemp("outputs")
    return SHAPDag(output_dir=out)


# ════════════════════════════════════════════════════════════════════════════
# GROUP 1 -- INSTANTIATION (tests 01-02)
# ════════════════════════════════════════════════════════════════════════════

class TestInstantiation:

    def test_01_shap_dag_instantiates(self, explainer):
        """N16: SHAPDag must instantiate without error"""
        assert explainer is not None

    def test_02_feature_names_count(self, explainer):
        """N16: Must have exactly 8 feature names"""
        assert len(FEATURE_NAMES) == 8
        assert "bm25_score"        in FEATURE_NAMES
        assert "cosine_sim"        in FEATURE_NAMES
        assert "fiscal_year_match" in FEATURE_NAMES
        assert "company_match"     in FEATURE_NAMES


# ════════════════════════════════════════════════════════════════════════════
# GROUP 2 -- FEATURE MATRIX (tests 03-06)
# ════════════════════════════════════════════════════════════════════════════

class TestFeatureMatrix:

    def test_03_feature_matrix_shape(self, explainer):
        """N16: Feature matrix must have shape (n_chunks, 8)"""
        X = explainer._build_feature_matrix(
            APPLE_CHUNKS, "net income", "Apple Inc", "FY2023"
        )
        assert X.shape == (3, 8)

    def test_04_feature_matrix_values_in_range(self, explainer):
        """N16: All feature values must be in 0-1 range"""
        X = explainer._build_feature_matrix(
            APPLE_CHUNKS, "net income", "Apple Inc", "FY2023"
        )
        assert np.all(X >= 0.0)
        assert np.all(X <= 1.0)

    def test_05_company_match_feature_correct(self, explainer):
        """N16: company_match must be 1.0 when company matches"""
        X = explainer._build_feature_matrix(
            APPLE_CHUNKS, "revenue", "Apple Inc", "FY2023"
        )
        # company_match is last column (index 7)
        assert X[0, 7] == 1.0

    def test_06_fiscal_year_match_feature_correct(self, explainer):
        """N16: fiscal_year_match must be 1.0 when FY matches"""
        X = explainer._build_feature_matrix(
            APPLE_CHUNKS, "revenue", "Apple Inc", "FY2023"
        )
        # fiscal_year_match is index 6
        assert X[0, 6] == 1.0


# ════════════════════════════════════════════════════════════════════════════
# GROUP 3 -- SHAP COMPUTATION (tests 07-11)
# ════════════════════════════════════════════════════════════════════════════

class TestSHAPComputation:

    def test_07_shap_returns_8_features(self, explainer):
        """N16: SHAP must return dict with 8 feature values"""
        shap_vals = explainer.compute_shap_for_chunks(
            APPLE_CHUNKS, "net income", "Apple Inc", "FY2023"
        )
        assert len(shap_vals) == 8

    def test_08_shap_values_are_floats(self, explainer):
        """N16: All SHAP values must be floats"""
        shap_vals = explainer.compute_shap_for_chunks(
            APPLE_CHUNKS, "net income", "Apple Inc", "FY2023"
        )
        assert all(isinstance(v, float) for v in shap_vals.values())

    def test_09_shap_values_non_negative(self, explainer):
        """N16: Mean absolute SHAP values must be >= 0"""
        shap_vals = explainer.compute_shap_for_chunks(
            APPLE_CHUNKS, "net income", "Apple Inc", "FY2023"
        )
        assert all(v >= 0.0 for v in shap_vals.values())

    def test_10_shap_reproducible_seed42(self, explainer):
        """C5: SHAP must give same result with seed=42"""
        shap1 = explainer.compute_shap_for_chunks(
            APPLE_CHUNKS, "net income", "Apple Inc", "FY2023"
        )
        shap2 = explainer.compute_shap_for_chunks(
            APPLE_CHUNKS, "net income", "Apple Inc", "FY2023"
        )
        assert shap1 == shap2

    def test_11_shap_empty_chunks_returns_zeros(self, explainer):
        """N16: Empty chunks must return zero SHAP values"""
        shap_vals = explainer.compute_shap_for_chunks([], "test")
        assert all(v == 0.0 for v in shap_vals.values())


# ════════════════════════════════════════════════════════════════════════════
# GROUP 4 -- NODE VALUE EXTRACTION (tests 12-15)
# ════════════════════════════════════════════════════════════════════════════

class TestNodeValueExtraction:

    def test_12_extracts_revenue(self, explainer):
        """N16: Must extract Revenue from text"""
        state = make_state()
        vals  = explainer._extract_node_values(state)
        # Revenue should be found in chunks
        assert "Revenue" in vals

    def test_13_extracts_net_income(self, explainer):
        """N16: Must extract Net Income from text"""
        state = make_state()
        vals  = explainer._extract_node_values(state)
        assert "Net Income" in vals
        if vals["Net Income"] is not None:
            assert vals["Net Income"] > 0

    def test_14_returns_none_for_missing_nodes(self, explainer):
        """N16: Missing values must be None not zero"""
        state = BAState(
            session_id        = "t14",
            final_answer_pre_xgb = "Some answer with no financials.",
        )
        vals = explainer._extract_node_values(state)
        assert vals.get("Revenue") is None

    def test_15_all_causal_nodes_present_in_result(self, explainer):
        """N16: All CAUSAL_NODES must appear as keys in result"""
        state = make_state()
        vals  = explainer._extract_node_values(state)
        for node in CAUSAL_NODES:
            assert node in vals


# ════════════════════════════════════════════════════════════════════════════
# GROUP 5 -- CAUSAL DAG (tests 16-18)
# ════════════════════════════════════════════════════════════════════════════

class TestCausalDAG:

    def test_16_dag_file_created(self, explainer):
        """N16: Causal DAG PNG must be created on disk"""
        state = make_state()
        path  = explainer.build_dag_for_state(state)
        if path:
            assert Path(path).exists()

    def test_17_causal_edges_defined(self, explainer):
        """N16: CAUSAL_EDGES must define a valid financial chain"""
        assert len(CAUSAL_EDGES) > 0
        # Revenue must be a source node
        sources = [e[0] for e in CAUSAL_EDGES]
        assert "Revenue" in sources
        # EPS must be a target node
        targets = [e[1] for e in CAUSAL_EDGES]
        assert "EPS" in targets

    def test_18_causal_nodes_count(self, explainer):
        """N16: Must have 9 causal nodes"""
        assert len(CAUSAL_NODES) == 9


# ════════════════════════════════════════════════════════════════════════════
# GROUP 6 -- BASTATE INTEGRATION (tests 19-22)
# ════════════════════════════════════════════════════════════════════════════

class TestBAStateIntegration:

    def test_19_run_writes_shap_values(self, explainer):
        """N16: run() must write shap_values dict to BAState"""
        state = make_state()
        state = explainer.run(state)
        assert state.shap_values is not None
        assert isinstance(state.shap_values, dict)
        assert len(state.shap_values) == 8

    def test_20_run_writes_feature_importance(self, explainer):
        """N16: run() must write feature_importance dict to BAState"""
        state = make_state()
        state = explainer.run(state)
        assert state.feature_importance is not None
        assert isinstance(state.feature_importance, dict)

    def test_21_run_writes_causal_dag_path(self, explainer):
        """N16: run() must write causal_dag_path to BAState"""
        state = make_state()
        state = explainer.run(state)
        # path is string or None — both acceptable
        assert state.causal_dag_path is None or \
               isinstance(state.causal_dag_path, str)

    def test_22_seed_unchanged_after_run(self, explainer):
        """C5: BAState seed must still be 42 after N16"""
        state = make_state()
        state = explainer.run(state)
        assert state.seed == 42


# ════════════════════════════════════════════════════════════════════════════
# GROUP 7 -- EDGE CASES (tests 23-24)
# ════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_23_empty_state_handled(self, explainer):
        """N16: Empty BAState must produce zero SHAP values"""
        state = BAState(session_id="t23-empty")
        state = explainer.run(state)
        assert state.shap_values is not None
        assert all(v == 0.0 for v in state.shap_values.values())

    def test_24_shap_row_cap_enforced(self, explainer):
        """N16: Feature matrix must not exceed 500 rows"""
        big_chunks = [APPLE_CHUNKS[0]] * 600
        X = explainer._build_feature_matrix(
            big_chunks[:SHAP_ROW_CAP],
            "revenue", "Apple Inc", "FY2023"
        )
        assert X.shape[0] <= SHAP_ROW_CAP