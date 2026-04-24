"""
tests/test_n07_bm25.py
Tests for N07 BM25 Retriever
PDR-BAAAI-001 Rev 1.0
"""
import json
import os
import pytest
from src.retrieval.bm25_retriever import BM25Retriever, TOP_K
from src.state.ba_state import BAState


def _build_test_index(tmp_path):
    """Build a real bm25s index using N03 Chunker for test use."""
    from src.ingestion.chunker import Chunker

    raw_text = (
        "Apple Inc Annual Report 10-K Fiscal Year 2023\n\n"
        "Business Overview\n"
        "Apple Inc designs manufactures and markets smartphones worldwide.\n\n"
        "Financial Statements\n"
        "Total net sales were 383285 million dollars in fiscal year 2023.\n"
        "Net income was 96995 million dollars.\n"
        "Diluted earnings per share was 6.13 dollars.\n\n"
        "Balance Sheet\n"
        "Total assets were 352583 million dollars as of September 30 2023.\n"
        "Total liabilities were 290437 million dollars.\n\n"
        "Risk Factors\n"
        "The company faces competition from Samsung Google and Microsoft.\n\n"
        "Notes to Financial Statements\n"
        "Note 1 Summary of Significant Accounting Policies revenue recognition.\n"
    )

    section_tree = {
        "document": "root",
        "total_sections": 3,
        "children": [
            {"name": "Business Overview",    "start_page": 3,  "end_page": 5,  "level": 1, "children": []},
            {"name": "Financial Statements", "start_page": 60, "end_page": 79, "level": 1, "children": []},
            {"name": "Balance Sheet",        "start_page": 63, "end_page": 70, "level": 1, "children": []},
        ],
    }

    bm25_dir = str(tmp_path / "bm25")
    chunker  = Chunker(bm25_dir=bm25_dir, chromadb_dir=str(tmp_path / "chromadb"))

    state = BAState(
        session_id   = "test-n07",
        raw_text     = raw_text,
        section_tree = section_tree,
        company_name = "Apple Inc",
        doc_type     = "10-K",
        fiscal_year  = "FY2023",
    )
    state = chunker.run(state)

    index_path = state.bm25_index_path
    return index_path


@pytest.fixture(scope="module")
def index_path(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bm25_test")
    return _build_test_index(tmp)


@pytest.fixture
def retriever():
    return BM25Retriever(top_k=10)


# ── Group 1: Constants ────────────────────────────────────────────────────────

class TestConstants:

    def test_01_top_k_default_defined(self):
        assert TOP_K == 10

    def test_02_retriever_default_top_k(self, retriever):
        assert retriever.top_k == 10

    def test_03_custom_top_k(self):
        r = BM25Retriever(top_k=5)
        assert r.top_k == 5


# ── Group 2: Instantiation ────────────────────────────────────────────────────

class TestInstantiation:

    def test_04_instantiates_no_error(self, retriever):
        assert retriever is not None

    def test_05_initial_chunks_empty(self, retriever):
        assert retriever._chunks == []

    def test_06_initial_retriever_none(self, retriever):
        assert retriever._retriever is None


# ── Group 3: BAState — no index path ─────────────────────────────────────────

class TestNoIndex:

    def test_07_no_index_path_returns_empty(self, retriever):
        state = BAState(session_id="t07", query="net sales")
        state = retriever.run(state)
        assert state.bm25_results == []

    def test_08_no_query_returns_empty(self, retriever, index_path):
        state = BAState(session_id="t08", bm25_index_path=index_path)
        state = retriever.run(state)
        assert state.bm25_results == []

    def test_09_missing_index_dir_returns_empty(self, retriever):
        state = BAState(
            session_id      = "t09",
            query           = "net sales",
            bm25_index_path = "data/bm25_index/nonexistent",
        )
        state = retriever.run(state)
        assert state.bm25_results == []

    def test_10_seed_unchanged_on_empty(self, retriever):
        state = BAState(session_id="t10", query="net sales")
        state = retriever.run(state)
        assert state.seed == 42


# ── Group 4: Search with real index ──────────────────────────────────────────

class TestSearchWithIndex:

    def test_11_search_returns_list(self, retriever, index_path):
        state = BAState(
            session_id      = "t11",
            query           = "What was Apple total net sales FY2023?",
            bm25_index_path = index_path,
        )
        state = retriever.run(state)
        assert isinstance(state.bm25_results, list)

    def test_12_search_returns_results(self, retriever, index_path):
        state = BAState(
            session_id      = "t12",
            query           = "net sales revenue income",
            bm25_index_path = index_path,
        )
        state = retriever.run(state)
        assert len(state.bm25_results) > 0

    def test_13_results_have_required_keys(self, retriever, index_path):
        state = BAState(
            session_id      = "t13",
            query           = "net income fiscal year",
            bm25_index_path = index_path,
        )
        state = retriever.run(state)
        for r in state.bm25_results:
            assert "bm25_score" in r

    def test_14_scores_are_numeric(self, retriever, index_path):
        state = BAState(
            session_id      = "t14",
            query           = "total assets balance sheet",
            bm25_index_path = index_path,
        )
        state = retriever.run(state)
        for r in state.bm25_results:
            assert isinstance(r["bm25_score"], (int, float))

    def test_15_scores_sorted_descending(self, retriever, index_path):
        state = BAState(
            session_id      = "t15",
            query           = "diluted earnings per share",
            bm25_index_path = index_path,
        )
        state = retriever.run(state)
        scores = [r["bm25_score"] for r in state.bm25_results]
        assert scores == sorted(scores, reverse=True)

    def test_16_top_k_respected(self, index_path):
        r = BM25Retriever(top_k=3)
        state = BAState(
            session_id      = "t16",
            query           = "apple revenue income",
            bm25_index_path = index_path,
        )
        state = r.run(state)
        assert len(state.bm25_results) <= 3

    def test_17_seed_unchanged_after_search(self, retriever, index_path):
        state = BAState(
            session_id      = "t17",
            query           = "gross margin percentage",
            bm25_index_path = index_path,
        )
        state = retriever.run(state)
        assert state.seed == 42

    def test_18_no_rlef_in_results(self, retriever, index_path):
        state = BAState(
            session_id      = "t18",
            query           = "total liabilities",
            bm25_index_path = index_path,
        )
        state = retriever.run(state)
        for r in state.bm25_results:
            assert "_rlef_" not in str(r)

    def test_19_bm25_confidence_written(self, retriever, index_path):
        state = BAState(
            session_id      = "t19",
            query           = "net income earnings",
            bm25_index_path = index_path,
        )
        state = retriever.run(state)
        assert hasattr(state, "bm25_confidence")
        assert isinstance(state.bm25_confidence, float)

    def test_20_financial_terms_retrieved(self, retriever, index_path):
        state = BAState(
            session_id      = "t20",
            query           = "net sales 383285 million",
            bm25_index_path = index_path,
        )
        state = retriever.run(state)
        assert len(state.bm25_results) > 0
        combined = " ".join(r.get("text", r.get("chunk_text", ""))
                            for r in state.bm25_results).lower()
        assert any(kw in combined for kw in ["net", "sales", "income", "apple"])

    def test_21_empty_query_handled(self, retriever, index_path):
        state = BAState(
            session_id      = "t21",
            query           = "",
            bm25_index_path = index_path,
        )
        state = retriever.run(state)
        assert isinstance(state.bm25_results, list)

    def test_22_chunks_meta_json_exists(self, index_path):
        meta_path = os.path.join(index_path, "chunks_meta.json")
        assert os.path.isfile(meta_path)

    def test_23_chunks_meta_is_valid_json(self, index_path):
        meta_path = os.path.join(index_path, "chunks_meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) > 0

    def test_24_run_twice_same_result(self, index_path):
        r1 = BM25Retriever(top_k=10)
        r2 = BM25Retriever(top_k=10)
        query = "What was total net sales FY2023?"
        s1 = BAState(session_id="t24a", query=query, bm25_index_path=index_path)
        s2 = BAState(session_id="t24b", query=query, bm25_index_path=index_path)
        s1 = r1.run(s1)
        s2 = r2.run(s2)
        assert len(s1.bm25_results) == len(s2.bm25_results)