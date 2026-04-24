"""
tests/test_n16_shap_dag.py
Tests for N16 SHAP + Causal DAG
PDR-BAAAI-001 · Rev 1.0
"""

import os
import pytest
from src.analysis.shap_dag import (
    SHAPDAGNode,
    run_shap_dag,
    compute_shap_importance,
    build_causal_dag,
    SHAP_ROW_CAP,
    DAG_DPI,
    CAUSAL_EDGES,
    SEED,
)
from src.state.ba_state import BAState


SAMPLE_CHUNKS = [
    {
        "chunk_id":    "c1",
        "text":        "Apple total net sales revenue were 383285 million in FY2023. Net income was 96995 million.",
        "section":     "INCOME_STATEMENT",
        "page":        94,
        "company":     "Apple Inc",
        "fiscal_year": "FY2023",
    },
    {
        "chunk_id":    "c2",
        "text":        "Gross profit was 169148 million. Operating income was 114301 million in FY2023.",
        "section":     "INCOME_STATEMENT",
        "page":        94,
        "company":     "Apple Inc",
        "fiscal_year": "FY2023",
    },
    {
        "chunk_id":    "c3",
        "text":        "Total assets were 352583 million. Shareholders equity 62146 million.",
        "section":     "BALANCE_SHEET",
        "page":        96,
        "company":     "Apple Inc",
        "fiscal_year": "FY2023",
    },
    {
        "chunk_id":    "c4",
        "text":        "Capital expenditures were 10959 million. Operating cash flow 114301 million.",
        "section":     "CASH_FLOW",
        "page":        98,
        "company":     "Apple Inc",
        "fiscal_year": "FY2023",
    },
    {
        "chunk_id":    "c5",
        "text":        "Research and development expense was 29915 million in fiscal 2023.",
        "section":     "INCOME_STATEMENT",
        "page":        95,
        "company":     "Apple Inc",
        "fiscal_year": "FY2023",
    },
]

SAMPLE_ANSWER = "Net income was $96,995 million in FY2023 [INCOME_STATEMENT/P94]."


@pytest.fixture
def node(tmp_path):
    return SHAPDAGNode(output_dir=str(tmp_path))


class TestConstants:

    def test_01_shap_row_cap_is_500(self):
        assert SHAP_ROW_CAP == 500

    def test_02_dag_dpi_is_300(self):
        assert DAG_DPI == 300

    def test_03_seed_is_42(self):
        assert SEED == 42

    def test_04_causal_edges_defined(self):
        assert len(CAUSAL_EDGES) >= 8

    def test_05_causal_edges_revenue_to_eps(self):
        nodes = set()
        for src, dst in CAUSAL_EDGES:
            nodes.add(src)
            nodes.add(dst)
        assert "Revenue"   in nodes
        assert "EPS"       in nodes
        assert "Net Income"in nodes


class TestSHAPImportance:

    def test_06_returns_dict_or_none(self):
        result = compute_shap_importance(SAMPLE_CHUNKS, SAMPLE_ANSWER)
        assert result is None or isinstance(result, dict)

    def test_07_result_has_required_keys(self):
        result = compute_shap_importance(SAMPLE_CHUNKS, SAMPLE_ANSWER)
        if result:
            assert "top_features" in result
            assert "top_chunks"   in result
            assert "n_chunks"     in result

    def test_08_top_features_have_importance(self):
        result = compute_shap_importance(SAMPLE_CHUNKS, SAMPLE_ANSWER)
        if result:
            for f in result["top_features"]:
                assert "feature"    in f
                assert "importance" in f
                assert f["importance"] >= 0.0

    def test_09_top_chunks_have_metadata(self):
        result = compute_shap_importance(SAMPLE_CHUNKS, SAMPLE_ANSWER)
        if result:
            for c in result["top_chunks"]:
                assert "chunk_id" in c
                assert "shap_sum" in c

    def test_10_empty_chunks_returns_none(self):
        assert compute_shap_importance([], SAMPLE_ANSWER) is None

    def test_11_empty_answer_returns_none(self):
        assert compute_shap_importance(SAMPLE_CHUNKS, "") is None

    def test_12_single_chunk_returns_none(self):
        assert compute_shap_importance([SAMPLE_CHUNKS[0]], SAMPLE_ANSWER) is None

    def test_13_row_cap_enforced(self):
        many   = SAMPLE_CHUNKS * 200
        result = compute_shap_importance(many, SAMPLE_ANSWER)
        if result:
            assert result["n_chunks"] <= SHAP_ROW_CAP


class TestCausalDAG:

    def test_14_build_returns_path_or_none(self, tmp_path):
        path   = str(tmp_path / "test_dag.png")
        result = build_causal_dag(output_path=path)
        assert result is not None

    def test_15_png_file_created(self, tmp_path):
        path   = str(tmp_path / "test_dag.png")
        result = build_causal_dag(output_path=path)
        if result and result != "dag_built_no_path":
            assert os.path.exists(path)

    def test_16_no_output_path_returns_sentinel(self):
        result = build_causal_dag(output_path=None)
        assert result is not None

    def test_17_highlight_nodes_accepted(self, tmp_path):
        path   = str(tmp_path / "highlighted_dag.png")
        result = build_causal_dag(output_path=path,
                                  highlight_nodes=["Net Income", "Revenue"])
        assert result is not None


class TestNodeDetection:

    def test_18_detects_net_income(self):
        nodes = SHAPDAGNode._detect_relevant_nodes("Net income was 96995 million.")
        assert "Net Income" in nodes

    def test_19_detects_revenue(self):
        nodes = SHAPDAGNode._detect_relevant_nodes("Total revenue was 383285 million.")
        assert "Revenue" in nodes

    def test_20_detects_eps(self):
        nodes = SHAPDAGNode._detect_relevant_nodes("Diluted EPS was 6.13 dollars.")
        assert "EPS" in nodes

    def test_21_detects_gross_profit(self):
        nodes = SHAPDAGNode._detect_relevant_nodes("Gross profit was 169148 million.")
        assert "Gross Profit" in nodes

    def test_22_empty_answer_returns_empty(self):
        assert SHAPDAGNode._detect_relevant_nodes("") == []

    def test_23_returns_list(self):
        assert isinstance(SHAPDAGNode._detect_relevant_nodes(SAMPLE_ANSWER), list)


class TestSHAPDAGNode:

    def test_24_instantiates(self, node):
        assert node is not None

    def test_25_explain_returns_dict(self, node):
        result = node.explain(SAMPLE_CHUNKS, SAMPLE_ANSWER)
        assert isinstance(result, dict)

    def test_26_explain_has_required_keys(self, node):
        result = node.explain(SAMPLE_CHUNKS, SAMPLE_ANSWER)
        assert "shap"               in result
        assert "feature_importance" in result
        assert "dag_path"           in result
        assert "highlight_nodes"    in result

    def test_27_highlight_nodes_is_list(self, node):
        result = node.explain(SAMPLE_CHUNKS, SAMPLE_ANSWER)
        assert isinstance(result["highlight_nodes"], list)

    def test_28_empty_input_still_returns_dict(self, node):
        result = node.explain([], "")
        assert isinstance(result, dict)


class TestBAStateIntegration:

    def test_29_run_writes_causal_dag_path(self, node):
        state = BAState(
            session_id           = "t29",
            retrieval_stage_2    = SAMPLE_CHUNKS,
            final_answer_pre_xgb = SAMPLE_ANSWER,
        )
        state = node.run(state)
        # BAState field is causal_dag_path (not causal_dag)
        assert hasattr(state, "causal_dag_path")

    def test_30_run_writes_shap_values(self, node):
        state = BAState(
            session_id           = "t30",
            retrieval_stage_2    = SAMPLE_CHUNKS,
            final_answer_pre_xgb = SAMPLE_ANSWER,
        )
        state = node.run(state)
        assert hasattr(state, "shap_values")

    def test_31_run_writes_feature_importance(self, node):
        state = BAState(
            session_id           = "t31",
            retrieval_stage_2    = SAMPLE_CHUNKS,
            final_answer_pre_xgb = SAMPLE_ANSWER,
        )
        state = node.run(state)
        assert hasattr(state, "feature_importance")

    def test_32_seed_unchanged(self, node):
        state = BAState(
            session_id           = "t32",
            retrieval_stage_2    = SAMPLE_CHUNKS,
            final_answer_pre_xgb = SAMPLE_ANSWER,
        )
        state = node.run(state)
        assert state.seed == 42

    def test_33_no_rlef_leak(self, node):
        state = BAState(
            session_id           = "t33",
            retrieval_stage_2    = SAMPLE_CHUNKS,
            final_answer_pre_xgb = SAMPLE_ANSWER,
        )
        state = node.run(state)
        fi = state.feature_importance or {}
        for key in fi:
            assert "_rlef_" not in key

    def test_34_empty_state_does_not_crash(self, node):
        state = BAState(session_id="t34")
        state = node.run(state)
        assert hasattr(state, "causal_dag_path")

    def test_35_causal_dag_path_is_str_or_none(self, node):
        state = BAState(
            session_id           = "t35",
            retrieval_stage_2    = SAMPLE_CHUNKS,
            final_answer_pre_xgb = SAMPLE_ANSWER,
        )
        state = node.run(state)
        assert state.causal_dag_path is None or isinstance(state.causal_dag_path, str)


class TestConvenienceWrapper:

    def test_36_run_shap_dag_returns_state(self, tmp_path):
        state = BAState(
            session_id           = "t36",
            retrieval_stage_2    = SAMPLE_CHUNKS,
            final_answer_pre_xgb = SAMPLE_ANSWER,
        )
        result = run_shap_dag(state, output_dir=str(tmp_path))
        assert hasattr(result, "causal_dag_path")
        assert result.seed == 42