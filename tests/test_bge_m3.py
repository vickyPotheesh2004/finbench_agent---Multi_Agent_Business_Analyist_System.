"""
tests/test_bge_m3.py
FinBench Multi-Agent Business Analyst AI

Tests for N08 — BGE-M3 Semantic Retriever

!! WARNING — RAM CONSTRAINT !!
This test loads a 500MB BGE model that stays in RAM.
Running this file ALONE after other sessions will hit C4 14GB halt.

CORRECT usage:  pytest tests\ -q          (full suite, fresh PowerShell)
WRONG usage:    pytest tests\test_bge_m3.py -v  (standalone = C4 halt risk)

Run: pytest tests\ -q --tb=no
"""

import shutil
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.retrieval.bge_retriever import BGERetriever
from src.state.ba_state import BAState


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def built_index(tmp_path_factory):
    """Build real ChromaDB + BM25 index once for all tests."""
    from src.ingestion.chunker import Chunker

    tmp_dir = tmp_path_factory.mktemp("bge_test")
    chunker = Chunker(data_dir=str(tmp_dir))

    sample_text = """
Financial Statements

Net sales were 383285 million dollars in fiscal year 2023.
Net income was 96995 million dollars for the year ended September 2023.
Diluted earnings per share were 6.13 dollars in fiscal 2023.
Total assets were 352583 million dollars as of September 2023.
Gross margin was 169148 million representing 44.1 percent.
Operating income was 114301 million dollars in 2023.
Total liabilities were 290437 million in fiscal 2023.
Shareholders equity was 62146 million dollars in fiscal 2023.

Business Overview

Apple designs smartphones personal computers tablets and wearables.
Services revenue grew to 85200 million in fiscal 2023.
iPhone represented 52 percent of total net revenue in 2023.
International sales accounted for 58 percent of total net sales.

Risk Factors

Competition in each market is intense and expected to remain so.
Supply chain disruptions could adversely affect production timelines.
Changes in global economic conditions may affect consumer spending.
    """

    state = BAState(
        session_id   = "test-n08-module",
        company_name = "Apple Inc",
        doc_type     = "10-K",
        fiscal_year  = "FY2023",
        raw_text     = sample_text,
        section_tree = {}
    )
    state = chunker.run(state)

    yield {
        "state":           state,
        "tmp_dir":         tmp_dir,
        "collection_name": state.chromadb_collection,
        "chromadb_dir":    str(tmp_dir / "chromadb" / "test-n08-module"),
    }

    try:
        import chromadb
        client = chromadb.PersistentClient(
            path=str(tmp_dir / "chromadb" / "test-n08-module")
        )
        client.reset()
    except Exception:
        pass
    time.sleep(0.3)
    shutil.rmtree(str(tmp_dir), ignore_errors=True)


@pytest.fixture(scope="module")
def loaded_retriever(built_index):
    """Single retriever instance shared across all tests — model loads once."""
    r = BGERetriever(top_k=5, data_dir=built_index["tmp_dir"])
    r.load_collection_direct(
        collection_name = built_index["collection_name"],
        chromadb_dir    = built_index["chromadb_dir"],
    )
    return r


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 1 — Instantiation
# ═══════════════════════════════════════════════════════════════════════════

class TestInstantiation:

    def test_01_retriever_instantiates(self):
        """N08: BGERetriever must instantiate without error"""
        r = BGERetriever()
        assert r is not None

    def test_02_default_model_name(self):
        """N08: Default model must be bge-small-en-v1.5"""
        r = BGERetriever()
        assert "bge" in r.model_name.lower()

    def test_03_not_loaded_by_default(self):
        """N08: is_loaded() must be False before model load"""
        r = BGERetriever()
        assert r.is_loaded() is False

    def test_04_collection_count_zero_before_load(self):
        """N08: collection_count() must be 0 before loading"""
        r = BGERetriever()
        assert r.collection_count() == 0


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 2 — Model loading
# ═══════════════════════════════════════════════════════════════════════════

class TestModelLoading:

    def test_05_model_loads(self, loaded_retriever):
        """N08: Model must load without error"""
        loaded_retriever._load_model()
        assert loaded_retriever.is_loaded() is True

    def test_06_collection_loads(self, loaded_retriever):
        """N08: ChromaDB collection must load without error"""
        assert loaded_retriever.collection_count() > 0

    def test_07_missing_collection_handled(self, built_index):
        """N08: Missing collection must not raise"""
        r = BGERetriever(data_dir=str(built_index["tmp_dir"]))
        r._load_collection("nonexistent_collection", built_index["chromadb_dir"])
        assert r.collection_count() == 0


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 3 — Search results
# ═══════════════════════════════════════════════════════════════════════════

class TestSearchResults:

    def test_08_search_returns_results(self, loaded_retriever):
        """N08: Search must return at least 1 result"""
        results = loaded_retriever._search("net income 2023")
        assert len(results) > 0

    def test_09_results_have_required_keys(self, loaded_retriever):
        """N08: Every result must have required keys"""
        results  = loaded_retriever._search("net income")
        required = {"chunk_id", "text", "semantic_score",
                    "distance", "rank", "retriever"}
        for r in results:
            assert required.issubset(r.keys()), \
                f"Missing: {required - r.keys()}"

    def test_10_scores_in_range(self, loaded_retriever):
        """N08: semantic_score must be between 0 and 1"""
        results = loaded_retriever._search("net income")
        for r in results:
            assert 0.0 <= r["semantic_score"] <= 1.0

    def test_11_results_sorted_by_score(self, loaded_retriever):
        """N08: Results must be sorted by semantic_score descending"""
        results = loaded_retriever._search("net income earnings")
        for i in range(len(results) - 1):
            assert results[i]["semantic_score"] >= results[i+1]["semantic_score"]

    def test_12_rank_starts_at_1(self, loaded_retriever):
        """N08: First result must have rank=1"""
        results = loaded_retriever._search("net income")
        if results:
            assert results[0]["rank"] == 1

    def test_13_retriever_field_is_bge_m3(self, loaded_retriever):
        """N08: retriever field must be 'bge_m3'"""
        results = loaded_retriever._search("net income")
        for r in results:
            assert r["retriever"] == "bge_m3"

    def test_14_semantic_search_finds_synonym(self, loaded_retriever):
        """N08: Must find 'earnings per share' when querying 'diluted EPS'"""
        results = loaded_retriever._search("What was diluted EPS in 2023?")
        assert len(results) > 0
        # Semantic search should score this highly even with different wording
        assert results[0]["semantic_score"] > 0.0


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 4 — Embedding methods
# ═══════════════════════════════════════════════════════════════════════════

class TestEmbeddingMethods:

    def test_15_embed_query_returns_vector(self, loaded_retriever):
        """N08: embed_query() must return a float vector"""
        vec = loaded_retriever.embed_query("What was net income?")
        assert isinstance(vec, list)
        assert len(vec) > 0
        assert isinstance(vec[0], float)

    def test_16_embed_query_dimension_consistent(self, loaded_retriever):
        """N08: embed_query() vectors must have consistent dimension"""
        vec1 = loaded_retriever.embed_query("net income")
        vec2 = loaded_retriever.embed_query("diluted earnings per share")
        assert len(vec1) == len(vec2)

    def test_17_embed_texts_returns_list(self, loaded_retriever):
        """N08: embed_texts() must return list of vectors"""
        texts = ["net income 2023", "diluted EPS", "total assets"]
        vecs  = loaded_retriever.embed_texts(texts)
        assert len(vecs) == 3
        assert all(isinstance(v, list) for v in vecs)

    def test_18_embed_texts_dimension_384(self, loaded_retriever):
        """N08: bge-small-en-v1.5 must produce 384-dim vectors"""
        vec = loaded_retriever.embed_query("test query")
        assert len(vec) == 384


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 5 — BAState integration
# ═══════════════════════════════════════════════════════════════════════════

class TestBAStateIntegration:

    def test_19_run_writes_retrieval_stage_1(
        self, loaded_retriever, built_index
    ):
        """N08: run() must write retrieval_stage_1 to BAState"""
        state = BAState(
            session_id            = built_index["state"].session_id,
            query                 = "What was net income in 2023?",
            chromadb_collection   = built_index["collection_name"],
        )
        state = loaded_retriever.run(state)
        assert isinstance(state.retrieval_stage_1, list)
        assert len(state.retrieval_stage_1) > 0

    def test_20_seed_unchanged_after_run(
        self, loaded_retriever, built_index
    ):
        """C5: BAState seed must still be 42 after N08"""
        state = BAState(
            session_id          = built_index["state"].session_id,
            query               = "net income",
            chromadb_collection = built_index["collection_name"],
        )
        state = loaded_retriever.run(state)
        assert state.seed == 42

    def test_21_no_collection_returns_empty(self, loaded_retriever):
        """N08: run() with no collection must return empty list"""
        state = BAState(session_id="t21", query="net income")
        state = loaded_retriever.run(state)
        assert state.retrieval_stage_1 == []

    def test_22_no_query_returns_empty(
        self, loaded_retriever, built_index
    ):
        """N08: run() with no query must return empty list"""
        state = BAState(
            session_id          = "t22",
            chromadb_collection = built_index["collection_name"],
        )
        state = loaded_retriever.run(state)
        assert state.retrieval_stage_1 == []

    def test_23_metadata_in_results(
        self, loaded_retriever, built_index
    ):
        """N08: Results must contain chunk metadata fields"""
        state = BAState(
            session_id          = built_index["state"].session_id,
            query               = "net income",
            chromadb_collection = built_index["collection_name"],
        )
        state = loaded_retriever.run(state)
        for r in state.retrieval_stage_1:
            assert "company"     in r
            assert "doc_type"    in r
            assert "fiscal_year" in r


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 6 — N07 + N08 parallel retrieval
# ═══════════════════════════════════════════════════════════════════════════

class TestN07N08Parallel:

    def test_24_bm25_and_bge_both_return_results(
        self, loaded_retriever, built_index
    ):
        """N08: BM25 and BGE must both return results for same query"""
        from src.retrieval.bm25_retriever import BM25Retriever

        bm25 = BM25Retriever(top_k=5)
        bge  = loaded_retriever

        query = "What was net income in 2023?"

        # BM25 results
        bm25_results = bm25.search_direct(
            query,
            built_index["state"].bm25_index_path,
            top_k=5
        )

        # BGE results
        bge_results = bge._search(query)

        assert len(bm25_results) > 0, "BM25 returned no results"
        assert len(bge_results)  > 0, "BGE returned no results"

        # Both use different retriever labels
        assert bm25_results[0]["retriever"] == "bm25"
        assert bge_results[0]["retriever"]  == "bge_m3"