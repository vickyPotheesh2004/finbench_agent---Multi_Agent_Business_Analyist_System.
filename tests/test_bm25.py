"""
tests/test_bm25.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 Rev1.0 FINAL

Tests for N07 — BM25 Retriever
Run: pytest tests/test_bm25.py -v
"""

import json
import shutil
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.retrieval.bm25_retriever import BM25Retriever
from src.state.ba_state import BAState


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def built_index(tmp_path_factory):
    """
    Build a real BM25 index once for all tests in this module.
    scope=module means it runs once — saves time.
    """
    from src.ingestion.chunker import Chunker

    tmp_dir = tmp_path_factory.mktemp("bm25_test")
    chunker = Chunker(data_dir=str(tmp_dir))

    sample_text = """
Financial Statements

Net sales were 383285 million dollars in fiscal year 2023.
Net income was 96995 million dollars for the year ended September 2023.
Diluted earnings per share were 6.13 dollars in fiscal 2023.
Total assets were 352583 million dollars as of September 2023.
Gross margin was 169148 million representing 44.1 percent of net sales.
Operating income was 114301 million dollars in fiscal year 2023.
Total liabilities were 290437 million dollars in fiscal 2023.
Shareholders equity was 62146 million dollars in fiscal 2023.

Risk Factors

Competition in each of the Company markets is intense and expected to remain so.
The Company faces risks from global economic conditions affecting product demand.
Supply chain disruptions could adversely affect production and delivery timelines.
Regulatory changes in various jurisdictions may impact business operations.

Business Overview

Apple Inc designs and markets smartphones personal computers and tablets worldwide.
iPhone represented 52 percent of total net revenue in fiscal year 2023.
Services revenue grew to 85200 million dollars representing strong recurring income.
International net sales accounted for 58 percent of total net sales in 2023.
    """

    state = BAState(
        session_id   = "test-n07-module",
        company_name = "Apple Inc",
        doc_type     = "10-K",
        fiscal_year  = "FY2023",
        raw_text     = sample_text,
        section_tree = {}
    )
    state = chunker.run(state)

    yield {
        "state":      state,
        "tmp_dir":    tmp_dir,
        "index_path": state.bm25_index_path,
        "chunker":    chunker,
    }

    # Cleanup — close ChromaDB first to release Windows file locks
    try:
        import chromadb
        client = chromadb.PersistentClient(
            path=str(tmp_dir / "chromadb" / "test-n07-module")
        )
        client.reset()
    except Exception:
        pass
    time.sleep(0.3)
    shutil.rmtree(str(tmp_dir), ignore_errors=True)


@pytest.fixture
def retriever():
    return BM25Retriever(top_k=5)


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 1 — Instantiation
# ═══════════════════════════════════════════════════════════════════════════

class TestInstantiation:

    def test_01_retriever_instantiates(self, retriever):
        """N07: BM25Retriever must instantiate without error"""
        assert retriever is not None

    def test_02_default_top_k(self):
        """N07: Default top_k must be 10"""
        r = BM25Retriever()
        assert r.top_k == 10

    def test_03_not_loaded_by_default(self, retriever):
        """N07: is_loaded() must be False before loading"""
        assert retriever.is_loaded() is False

    def test_04_chunk_count_zero_before_load(self, retriever):
        """N07: get_chunk_count() must return 0 before loading"""
        assert retriever.get_chunk_count() == 0


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 2 — Index loading
# ═══════════════════════════════════════════════════════════════════════════

class TestIndexLoading:

    def test_05_loads_index(self, retriever, built_index):
        """N07: Must load BM25 index without error"""
        retriever._load_index(built_index["index_path"])
        assert retriever.is_loaded() is True

    def test_06_chunk_count_after_load(self, retriever, built_index):
        """N07: chunk count must be > 0 after loading"""
        retriever._load_index(built_index["index_path"])
        assert retriever.get_chunk_count() > 0

    def test_07_missing_index_handled(self, retriever):
        """N07: Missing index path must not raise — returns empty"""
        retriever._load_index("nonexistent/path")
        assert retriever.get_chunk_count() == 0

    def test_08_no_index_path_returns_empty(self, retriever):
        """N07: run() with no index path must return empty bm25_results"""
        state = BAState(session_id="t08", query="net income")
        state = retriever.run(state)
        assert state.bm25_results == []

    def test_09_no_query_returns_empty(self, retriever, built_index):
        """N07: run() with no query must return empty bm25_results"""
        state = BAState(
            session_id      = "t09",
            bm25_index_path = built_index["index_path"]
        )
        state = retriever.run(state)
        assert state.bm25_results == []


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 3 — Search results
# ═══════════════════════════════════════════════════════════════════════════

class TestSearchResults:

    def test_10_search_returns_results(self, retriever, built_index):
        """N07: Search must return at least 1 result"""
        results = retriever.search_direct(
            "net income 2023", built_index["index_path"], top_k=5
        )
        assert len(results) > 0

    def test_11_results_have_required_keys(self, retriever, built_index):
        """N07: Every result must have required keys"""
        results = retriever.search_direct(
            "net income", built_index["index_path"], top_k=5
        )
        required = {"bm25_score", "bm25_score_norm", "rank",
                    "retriever", "chunk_id", "text"}
        for r in results:
            assert required.issubset(r.keys()), \
                f"Missing keys: {required - r.keys()}"

    def test_12_scores_are_normalised(self, retriever, built_index):
        """N07: bm25_score_norm must be between 0 and 1"""
        results = retriever.search_direct(
            "net income", built_index["index_path"], top_k=5
        )
        for r in results:
            assert 0.0 <= r["bm25_score_norm"] <= 1.0, \
                f"Score out of range: {r['bm25_score_norm']}"

    def test_13_results_sorted_by_score(self, retriever, built_index):
        """N07: Results must be sorted by bm25_score descending"""
        results = retriever.search_direct(
            "net income", built_index["index_path"], top_k=5
        )
        for i in range(len(results) - 1):
            assert results[i]["bm25_score"] >= results[i+1]["bm25_score"], \
                "Results not sorted by score"

    def test_14_rank_starts_at_1(self, retriever, built_index):
        """N07: First result must have rank=1"""
        results = retriever.search_direct(
            "net income", built_index["index_path"], top_k=5
        )
        assert results[0]["rank"] == 1

    def test_15_retriever_field_is_bm25(self, retriever, built_index):
        """N07: retriever field must be 'bm25'"""
        results = retriever.search_direct(
            "net income", built_index["index_path"], top_k=5
        )
        for r in results:
            assert r["retriever"] == "bm25"

    def test_16_top_k_respected(self, retriever, built_index):
        """N07: Must not return more results than top_k"""
        results = retriever.search_direct(
            "net income", built_index["index_path"], top_k=3
        )
        assert len(results) <= 3


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 4 — BAState integration
# ═══════════════════════════════════════════════════════════════════════════

class TestBAStateIntegration:

    def test_17_run_writes_bm25_results(self, retriever, built_index):
        """N07: run() must write bm25_results to BAState"""
        state = BAState(
            session_id      = "t17",
            query           = "What was net income in 2023?",
            bm25_index_path = built_index["index_path"],
        )
        state = retriever.run(state)
        assert isinstance(state.bm25_results, list)
        assert len(state.bm25_results) > 0

    def test_18_seed_unchanged_after_run(self, retriever, built_index):
        """C5: BAState seed must still be 42 after N07"""
        state = BAState(
            session_id      = "t18",
            query           = "net income",
            bm25_index_path = built_index["index_path"],
        )
        state = retriever.run(state)
        assert state.seed == 42

    def test_19_c8_prefix_present_in_results(self, retriever, built_index):
        """C8: Results must contain C8 metadata prefix in text"""
        state = BAState(
            session_id      = "t19",
            query           = "net income 2023",
            bm25_index_path = built_index["index_path"],
        )
        state = retriever.run(state)
        for r in state.bm25_results:
            assert "Apple Inc" in r.get("text", "") or \
                   "Apple Inc" in r.get("prefix", ""), \
                "C8 company name missing from result"

    def test_20_metadata_fields_in_results(self, retriever, built_index):
        """N07: Results must contain chunk metadata fields"""
        state = BAState(
            session_id      = "t20",
            query           = "net income",
            bm25_index_path = built_index["index_path"],
        )
        state = retriever.run(state)
        for r in state.bm25_results:
            assert "company"     in r
            assert "doc_type"    in r
            assert "fiscal_year" in r


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 5 — LangChain integration
# ═══════════════════════════════════════════════════════════════════════════

class TestLangChainIntegration:

    def test_21_langchain_retriever_not_none(self, retriever, built_index):
        """N07: as_langchain_retriever() must return non-None"""
        lc_ret = retriever.as_langchain_retriever(
            built_index["index_path"]
        )
        assert lc_ret is not None

    def test_22_langchain_retriever_returns_docs(self, retriever, built_index):
        """N07: LangChain retriever must return documents"""
        lc_ret = retriever.as_langchain_retriever(
            built_index["index_path"]
        )
        docs = lc_ret.invoke("net income 2023")
        assert len(docs) > 0

    def test_23_langchain_docs_have_metadata(self, retriever, built_index):
        """N07: LangChain docs must have metadata"""
        lc_ret = retriever.as_langchain_retriever(
            built_index["index_path"]
        )
        docs = lc_ret.invoke("net income")
        for doc in docs:
            assert hasattr(doc, "metadata")
            assert "chunk_id" in doc.metadata


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 6 — N06 + N07 cascade
# ═══════════════════════════════════════════════════════════════════════════

class TestN06N07Cascade:

    def test_24_n06_miss_cascades_to_n07(self, built_index):
        """N07: N06 miss must cascade correctly to N07"""
        from src.retrieval.sniper_rag import SniperRAG
        from src.state.ba_state import QueryType

        sniper    = SniperRAG()
        retriever = BM25Retriever(top_k=5)

        # Query with no matching table cells → N06 miss
        state = BAState(
            session_id      = "t24",
            query           = "What was net income in 2023?",
            query_type      = QueryType.NUMERICAL,
            table_cells     = [],  # empty → N06 miss
            bm25_index_path = built_index["index_path"],
        )

        # N06 fires first
        state = sniper.run(state)
        assert state.sniper_hit is False

        # N07 fires because N06 missed
        state = retriever.run(state)
        assert len(state.bm25_results) > 0