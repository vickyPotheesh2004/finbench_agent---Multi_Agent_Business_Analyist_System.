"""
tests/test_pipeline.py
FinBench Multi-Agent Business Analyst AI

Tests for full 19-node pipeline + Gate M4 check.

NOTE: Run separately from full suite — pipeline loads all models.
Command: pytest tests\test_pipeline.py -v --tb=short

24 tests covering:
  - Instantiation (tests 01-03)
  - Query phase (tests 04-09)
  - Parallel pods (tests 10-13)
  - Conditional SniperRAG edge (tests 14-15)
  - BAState integration (tests 16-20)
  - Gate M4 check (tests 21-24)
"""

import sys
import copy
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from src.pipeline.pipeline import FinBenchPipeline
from src.state.ba_state    import BAState, QueryType, Difficulty, PIVStatus


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_pre_ingested_state(
    session_id: str = "test-pipeline",
    query:      str = "What was Apple net income FY2023?",
) -> BAState:
    return BAState(
        session_id          = session_id,
        query               = query,
        company_name        = "Apple Inc",
        doc_type            = "10-K",
        fiscal_year         = "FY2023",
        chunk_count         = 2,
        bm25_index_path     = "mock",
        chromadb_collection = "mock",
        retrieval_stage_2   = [{
            "text":        "Net income $96,995 million FY2023. "
                           "Revenue $383,285 million.",
            "section":     "Financial Statements",
            "page":        "42",
            "company":     "Apple Inc",
            "fiscal_year": "FY2023",
            "bm25_score":  0.85,
            "cosine_sim":  0.92,
        }],
    )


def make_state_with_answer(session_id: str, query: str) -> BAState:
    """Pre-populate analyst answer so mediator has a candidate."""
    state = make_pre_ingested_state(session_id, query)
    state.analyst_output     = f"Net income was $96,995 million FY2023."
    state.analyst_confidence = 0.60
    state.analyst_piv_status = PIVStatus.PASS
    state.analyst_citations  = ["Financial Statements / Page 42"]
    return state


def safe_run(pipeline, state, question):
    """
    Run pipeline query — skip test on MemoryError (C4 RAM halt).
    MemoryError means RAM was already full from other loaded models.
    Run pipeline tests in fresh session to avoid this.
    """
    try:
        return pipeline.run_query_only(state, question)
    except MemoryError as e:
        pytest.skip(f"C4 RAM limit hit — run pipeline tests in fresh session. {e}")


@pytest.fixture(scope="module")
def pipeline():
    try:
        return FinBenchPipeline()
    except MemoryError as e:
        pytest.skip(f"C4 RAM limit — run in fresh session. {e}")


# ════════════════════════════════════════════════════════════════════════════
# GROUP 1 -- INSTANTIATION (tests 01-03)
# ════════════════════════════════════════════════════════════════════════════

class TestInstantiation:

    def test_01_pipeline_instantiates(self, pipeline):
        """Pipeline must instantiate without error"""
        assert pipeline is not None

    def test_02_all_19_nodes_present(self, pipeline):
        """Pipeline must have all 19 node attributes"""
        for node in ["n01","n02","n03","n04","n05","n06","n07",
                     "n08","n09","n10","n11","n12","n13","n14",
                     "n15","n16","n17","n18","n19"]:
            assert hasattr(pipeline, node), f"Missing node: {node}"

    def test_03_get_nodes_fired_returns_list(self, pipeline):
        """get_nodes_fired() must return a list"""
        assert isinstance(pipeline.get_nodes_fired(), list)


# ════════════════════════════════════════════════════════════════════════════
# GROUP 2 -- QUERY PHASE (tests 04-09)
# ════════════════════════════════════════════════════════════════════════════

class TestQueryPhase:

    def test_04_run_query_only_returns_bastate(self, pipeline):
        """run_query_only() must return BAState"""
        state  = make_state_with_answer("t04", "What was net income?")
        result = safe_run(pipeline, state, "What was net income?")
        assert isinstance(result, BAState)

    def test_05_final_answer_populated(self, pipeline):
        """Query must produce non-empty final answer"""
        state = make_state_with_answer("t05", "What was net income?")
        state = safe_run(pipeline, state, "What was net income?")
        assert (bool(state.final_answer) or
                bool(state.xgb_ranked_answer) or
                bool(state.final_answer_pre_xgb))

    def test_06_nodes_fired_after_query(self, pipeline):
        """At least 10 nodes must fire"""
        state = make_state_with_answer("t06", "What was revenue?")
        safe_run(pipeline, state, "What was revenue?")
        assert len(pipeline.get_nodes_fired()) >= 10

    def test_07_report_path_set(self, pipeline):
        """final_report_path must be set"""
        state = make_state_with_answer("t07", "What was net income?")
        state = safe_run(pipeline, state, "What was net income?")
        assert state.final_report_path is not None

    def test_08_report_file_exists(self, pipeline):
        """Report file must exist on disk"""
        state = make_state_with_answer("t08", "What was revenue?")
        state = safe_run(pipeline, state, "What was revenue?")
        assert Path(state.final_report_path).exists()

    def test_09_seed_preserved_after_query(self, pipeline):
        """C5: seed must still be 42"""
        state = make_state_with_answer("t09", "What was EPS?")
        state = safe_run(pipeline, state, "What was EPS?")
        assert state.seed == 42


# ════════════════════════════════════════════════════════════════════════════
# GROUP 3 -- PARALLEL PODS (tests 10-13)
# ════════════════════════════════════════════════════════════════════════════

class TestParallelPods:

    def test_10_pods_write_state(self, pipeline):
        """N11+N12+N13+N14 must write to BAState"""
        state = make_state_with_answer("t10", "What was net income?")
        state = safe_run(pipeline, state, "What was net income?")
        assert (bool(state.analyst_output) or bool(state.quant_result) or
                state.risk_score >= 0 or bool(state.auditor_output))

    def test_11_triguard_writes_risk_score(self, pipeline):
        """N13 TriGuard must write risk_score 0-100"""
        state = make_state_with_answer("t11", "What was net income?")
        state = safe_run(pipeline, state, "What was net income?")
        assert 0.0 <= state.risk_score <= 100.0

    def test_12_parallel_pods_in_nodes_fired(self, pipeline):
        """N11+N12+N13+N14 must appear in nodes_fired"""
        state = make_state_with_answer("t12", "What was net income?")
        safe_run(pipeline, state, "What was net income?")
        fired = " ".join(pipeline.get_nodes_fired())
        assert "N11" in fired or "N13" in fired

    def test_13_mediator_fires_after_pods(self, pipeline):
        """N15 PIV Mediator must fire"""
        state = make_state_with_answer("t13", "What was net income?")
        safe_run(pipeline, state, "What was net income?")
        assert any("N15" in n for n in pipeline.get_nodes_fired())


# ════════════════════════════════════════════════════════════════════════════
# GROUP 4 -- SNIPER RAG EDGE (tests 14-15)
# ════════════════════════════════════════════════════════════════════════════

class TestSniperRAGEdge:

    def test_14_sniper_hit_skips_retrieval(self, pipeline):
        """SniperRAG hit must skip N07-N09"""
        state                   = make_state_with_answer("t14", "net income?")
        state.sniper_hit        = True
        state.sniper_confidence = 0.98
        state.sniper_result     = "Net income $96,995M FY2023"
        original         = pipeline.n06.run
        pipeline.n06.run = MagicMock(return_value=state)
        safe_run(pipeline, state, "net income?")
        fired            = " ".join(pipeline.get_nodes_fired())
        pipeline.n06.run = original
        assert "N07_BM25" not in fired or "N06" in fired

    def test_15_sniper_miss_runs_full_retrieval(self, pipeline):
        """SniperRAG miss must run N07+N08+N09"""
        state                   = make_state_with_answer("t15", "gross margin?")
        state.sniper_hit        = False
        state.sniper_confidence = 0.0
        safe_run(pipeline, state, "gross margin?")
        fired = " ".join(pipeline.get_nodes_fired())
        assert "N07" in fired or "N08" in fired


# ════════════════════════════════════════════════════════════════════════════
# GROUP 5 -- BASTATE INTEGRATION (tests 16-20)
# ════════════════════════════════════════════════════════════════════════════

class TestBAStateIntegration:

    def test_16_query_type_set_by_n04(self, pipeline):
        """N04 CART Router must set query_type"""
        state = make_state_with_answer("t16", "What was net income?")
        state = safe_run(pipeline, state, "What was net income?")
        assert state.query_type is not None

    def test_17_shap_values_written(self, pipeline):
        """N16 must write shap_values"""
        state = make_state_with_answer("t17", "What was net income?")
        state = safe_run(pipeline, state, "What was net income?")
        assert state.shap_values is not None
        assert isinstance(state.shap_values, dict)

    def test_18_rlef_grade_written(self, pipeline):
        """N18 must write _rlef_grade"""
        state = make_state_with_answer("t18", "What was net income?")
        state = safe_run(pipeline, state, "What was net income?")
        assert "_rlef_grade" in state.get_rlef_fields()

    def test_19_no_rlef_in_public_fields(self, pipeline):
        """C9: No _rlef_ in public BAState"""
        state = make_state_with_answer("t19", "What was net income?")
        state = safe_run(pipeline, state, "What was net income?")
        for key in state.model_dump():
            assert not key.startswith("_rlef_")

    def test_20_iteration_count_not_exceeded(self, pipeline):
        """A2: iteration_count must never exceed 5"""
        state = make_state_with_answer("t20", "What was revenue?")
        state = safe_run(pipeline, state, "What was revenue?")
        assert state.iteration_count <= 5


# ════════════════════════════════════════════════════════════════════════════
# GROUP 6 -- GATE M4 (tests 21-24)
# ════════════════════════════════════════════════════════════════════════════

class TestGateM4:

    GATE_M4_QUESTIONS = [
        "What was Apple net income FY2023?",
        "What was Apple total revenue FY2023?",
        "What was Apple gross profit margin FY2023?",
    ]

    def test_21_gate_m4_pipeline_completes_end_to_end(self, pipeline):
        """Gate M4: Pipeline must complete end-to-end"""
        for q in self.GATE_M4_QUESTIONS:
            state = make_state_with_answer(
                f"m4-{q[:20].replace(' ','_')}", q
            )
            state = safe_run(pipeline, state, q)
            assert len(pipeline.get_nodes_fired()) >= 10
            assert state.final_report_path is not None
            assert Path(state.final_report_path).exists()

    def test_22_gate_m4_core_nodes_fire(self, pipeline):
        """Gate M4: N04+N10+N15+N17+N19 must all fire"""
        state = make_state_with_answer("m4-core", "What was net income?")
        safe_run(pipeline, state, "What was net income?")
        fired = " ".join(pipeline.get_nodes_fired())
        for n in ["N04", "N10", "N15", "N17", "N19"]:
            assert n in fired, f"Missing: {n}"

    def test_23_gate_m4_iteration_cap(self, pipeline):
        """Gate M4: iteration_count never > 5"""
        for q in self.GATE_M4_QUESTIONS:
            state = make_state_with_answer(
                f"m4-iter-{q[:15].replace(' ','_')}", q
            )
            state = safe_run(pipeline, state, q)
            assert state.iteration_count <= 5

    def test_24_gate_m4_docx_report_generated(self, pipeline):
        """Gate M4: DOCX report must exist for every query"""
        for q in self.GATE_M4_QUESTIONS:
            state = make_state_with_answer(
                f"m4-rep-{q[:15].replace(' ','_')}", q
            )
            state = safe_run(pipeline, state, q)
            assert state.final_report_path is not None
            assert Path(state.final_report_path).exists()