"""
tests/test_rrf.py
FinBench Multi-Agent Business Analyst AI

Tests for N09 — RRF Merge + Cross-Encoder Reranker

!! RAM NOTE !!
Cross-encoder loads ~90MB model.
Run as part of full suite only: pytest tests\ -q
Never run standalone after BGE tests in same session.

24 tests covering:
  - Instantiation (tests 01-03)
  - RRF algorithm correctness (tests 04-09)
  - Cross-encoder reranking (tests 10-14)
  - BAState integration (tests 15-20)
  - Edge cases (tests 21-24)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from src.retrieval.rrf_reranker import RRFReranker
from src.state.ba_state import BAState


# ── Shared mock data ──────────────────────────────────────────────────────────

def make_bm25_results():
    return [
        {
            "chunk_id":        "chunk_000001",
            "text":            "Apple / 10-K / FY2023 / Financial Statements / 42\nNet income was 96995 million.",
            "content":         "Apple / 10-K / FY2023 / Financial Statements / 42\nNet income was 96995 million.",
            "section":         "Financial Statements",
            "company":         "Apple Inc",
            "doc_type":        "10-K",
            "fiscal_year":     "FY2023",
            "page":            "42",
            "bm25_score":      0.91,
            "bm25_score_norm": 1.0,
            "rank":            1,
        },
        {
            "chunk_id":        "chunk_000002",
            "text":            "Apple / 10-K / FY2023 / MD&A / 28\nNet sales were 383285 million.",
            "content":         "Apple / 10-K / FY2023 / MD&A / 28\nNet sales were 383285 million.",
            "section":         "MD&A",
            "company":         "Apple Inc",
            "doc_type":        "10-K",
            "fiscal_year":     "FY2023",
            "page":            "28",
            "bm25_score":      0.72,
            "bm25_score_norm": 0.79,
            "rank":            2,
        },
        {
            "chunk_id":        "chunk_000003",
            "text":            "Apple / 10-K / FY2023 / Business Overview / 3\nApple designs iPhones.",
            "content":         "Apple / 10-K / FY2023 / Business Overview / 3\nApple designs iPhones.",
            "section":         "Business Overview",
            "company":         "Apple Inc",
            "doc_type":        "10-K",
            "fiscal_year":     "FY2023",
            "page":            "3",
            "bm25_score":      0.45,
            "bm25_score_norm": 0.49,
            "rank":            3,
        },
    ]


def make_bge_results():
    """BGE results — different order from BM25 (chunk_000002 is rank 1)"""
    return [
        {
            "chunk_id":      "chunk_000002",
            "text":          "Apple / 10-K / FY2023 / MD&A / 28\nNet sales were 383285 million.",
            "content":       "Apple / 10-K / FY2023 / MD&A / 28\nNet sales were 383285 million.",
            "section":       "MD&A",
            "company":       "Apple Inc",
            "doc_type":      "10-K",
            "fiscal_year":   "FY2023",
            "page":          "28",
            "semantic_score": 0.88,
            "distance":      0.24,
            "rank":          1,
        },
        {
            "chunk_id":      "chunk_000001",
            "text":          "Apple / 10-K / FY2023 / Financial Statements / 42\nNet income was 96995 million.",
            "content":       "Apple / 10-K / FY2023 / Financial Statements / 42\nNet income was 96995 million.",
            "section":       "Financial Statements",
            "company":       "Apple Inc",
            "doc_type":      "10-K",
            "fiscal_year":   "FY2023",
            "page":          "42",
            "semantic_score": 0.85,
            "distance":      0.30,
            "rank":          2,
        },
        {
            "chunk_id":      "chunk_000004",
            "text":          "Apple / 10-K / FY2023 / Risk Factors / 8\nCompetition is intense.",
            "content":       "Apple / 10-K / FY2023 / Risk Factors / 8\nCompetition is intense.",
            "section":       "Risk Factors",
            "company":       "Apple Inc",
            "doc_type":      "10-K",
            "fiscal_year":   "FY2023",
            "page":          "8",
            "semantic_score": 0.62,
            "distance":      0.76,
            "rank":          3,
        },
    ]


# ── Module-level fixture — reranker instantiated once ────────────────────────

@pytest.fixture(scope="module")
def reranker():
    return RRFReranker()


# ════════════════════════════════════════════════════════════════════════════
# GROUP 1 — INSTANTIATION (tests 01-03)
# ════════════════════════════════════════════════════════════════════════════

class TestInstantiation:

    def test_01_reranker_instantiates(self, reranker):
        """N09: RRFReranker must instantiate without error"""
        assert reranker is not None

    def test_02_default_params(self, reranker):
        """N09: Default RRF_K=60, top_k_merge=10, final_top_k=3"""
        assert reranker.rrf_k       == 60
        assert reranker.top_k_merge == 10
        assert reranker.final_top_k == 3

    def test_03_cross_encoder_not_loaded_by_default(self, reranker):
        """N09: Cross-encoder must be lazy-loaded (not on instantiation)"""
        r = RRFReranker()
        assert r.is_cross_encoder_loaded() is False


# ════════════════════════════════════════════════════════════════════════════
# GROUP 2 — RRF ALGORITHM (tests 04-09)
# ════════════════════════════════════════════════════════════════════════════

class TestRRFAlgorithm:

    def test_04_rrf_returns_results(self, reranker):
        """N09: RRF must return at least 1 chunk"""
        result = reranker.rrf_only(
            make_bm25_results(),
            make_bge_results(),
            top_k=3
        )
        assert len(result) >= 1

    def test_05_rrf_dual_list_chunk_scores_highest(self, reranker):
        """N09: Chunk appearing in both BM25 and BGE must have highest RRF score"""
        result = reranker.rrf_only(
            make_bm25_results(),
            make_bge_results(),
            top_k=3
        )
        # chunk_000001 appears in both lists
        # chunk_000004 appears only in BGE
        dual_chunks   = [c for c in result if c.get("in_both") is True]
        single_chunks = [c for c in result if c.get("in_both") is False]
        if dual_chunks and single_chunks:
            assert dual_chunks[0]["rrf_score"] > single_chunks[-1]["rrf_score"]

    def test_06_rrf_score_field_present(self, reranker):
        """N09: Every RRF result must have rrf_score field"""
        result = reranker.rrf_only(
            make_bm25_results(),
            make_bge_results(),
            top_k=3
        )
        for chunk in result:
            assert "rrf_score" in chunk
            assert chunk["rrf_score"] > 0

    def test_07_rrf_sorted_by_score_descending(self, reranker):
        """N09: RRF results must be sorted by rrf_score descending"""
        result = reranker.rrf_only(
            make_bm25_results(),
            make_bge_results(),
            top_k=3
        )
        scores = [c["rrf_score"] for c in result]
        assert scores == sorted(scores, reverse=True)

    def test_08_rrf_top_k_respected(self, reranker):
        """N09: rrf_only must return at most top_k results"""
        result = reranker.rrf_only(
            make_bm25_results(),
            make_bge_results(),
            top_k=2
        )
        assert len(result) <= 2

    def test_09_rrf_empty_bge_uses_bm25_only(self, reranker):
        """N09: RRF must handle empty BGE list gracefully"""
        result = reranker.rrf_only(
            make_bm25_results(),
            [],
            top_k=3
        )
        assert len(result) >= 1
        for chunk in result:
            assert "rrf_score" in chunk


# ════════════════════════════════════════════════════════════════════════════
# GROUP 3 — CROSS-ENCODER RERANKING (tests 10-14)
# ════════════════════════════════════════════════════════════════════════════

class TestCrossEncoder:

    def test_10_run_returns_final_chunks(self, reranker):
        """N09: run() must return retrieval_stage_2 with chunks"""
        state = BAState(
            session_id        = "t10",
            query             = "What was Apple net income in FY2023?",
            bm25_results      = make_bm25_results(),
            retrieval_stage_1 = make_bge_results(),
        )
        state = reranker.run(state)
        assert isinstance(state.retrieval_stage_2, list)
        assert len(state.retrieval_stage_2) >= 1

    def test_11_final_top_k_respected(self, reranker):
        """N09: run() must return at most final_top_k=3 chunks"""
        state = BAState(
            session_id        = "t11",
            query             = "net income",
            bm25_results      = make_bm25_results(),
            retrieval_stage_1 = make_bge_results(),
        )
        state = reranker.run(state)
        assert len(state.retrieval_stage_2) <= 3

    def test_12_rerank_score_field_present(self, reranker):
        """N09: Every final chunk must have rerank_score field"""
        state = BAState(
            session_id        = "t12",
            query             = "net income Apple 2023",
            bm25_results      = make_bm25_results(),
            retrieval_stage_1 = make_bge_results(),
        )
        state = reranker.run(state)
        for chunk in state.retrieval_stage_2:
            assert "rerank_score" in chunk

    def test_13_final_rank_field_present(self, reranker):
        """N09: Every final chunk must have final_rank field"""
        state = BAState(
            session_id        = "t13",
            query             = "What was net income?",
            bm25_results      = make_bm25_results(),
            retrieval_stage_1 = make_bge_results(),
        )
        state = reranker.run(state)
        for chunk in state.retrieval_stage_2:
            assert "final_rank" in chunk

    def test_14_results_sorted_by_rerank_score(self, reranker):
        """N09: Final chunks must be sorted by rerank_score descending"""
        state = BAState(
            session_id        = "t14",
            query             = "net income fiscal year 2023",
            bm25_results      = make_bm25_results(),
            retrieval_stage_1 = make_bge_results(),
        )
        state = reranker.run(state)
        scores = [c["rerank_score"] for c in state.retrieval_stage_2]
        assert scores == sorted(scores, reverse=True)


# ════════════════════════════════════════════════════════════════════════════
# GROUP 4 — BASTATE INTEGRATION (tests 15-20)
# ════════════════════════════════════════════════════════════════════════════

class TestBAStateIntegration:

    def test_15_writes_retrieval_stage_2(self, reranker):
        """N09: run() must write to state.retrieval_stage_2"""
        state = BAState(
            session_id        = "t15",
            query             = "net income",
            bm25_results      = make_bm25_results(),
            retrieval_stage_1 = make_bge_results(),
        )
        state = reranker.run(state)
        assert state.retrieval_stage_2 is not None
        assert isinstance(state.retrieval_stage_2, list)

    def test_16_seed_unchanged_after_run(self, reranker):
        """C5: BAState seed must still be 42 after N09"""
        state = BAState(
            session_id        = "t16",
            query             = "net income",
            bm25_results      = make_bm25_results(),
            retrieval_stage_1 = make_bge_results(),
        )
        state = reranker.run(state)
        assert state.seed == 42

    def test_17_empty_both_returns_empty_list(self, reranker):
        """N09: Both empty inputs must return empty retrieval_stage_2"""
        state = BAState(
            session_id        = "t17",
            query             = "net income",
            bm25_results      = [],
            retrieval_stage_1 = [],
        )
        state = reranker.run(state)
        assert state.retrieval_stage_2 == []

    def test_18_no_query_returns_empty(self, reranker):
        """N09: No query must return empty retrieval_stage_2"""
        state = BAState(
            session_id        = "t18",
            bm25_results      = make_bm25_results(),
            retrieval_stage_1 = make_bge_results(),
        )
        state = reranker.run(state)
        assert state.retrieval_stage_2 == []

    def test_19_retriever_field_is_rrf_reranked(self, reranker):
        """N09: Final chunks retriever field must be rrf_reranked"""
        state = BAState(
            session_id        = "t19",
            query             = "What was net income?",
            bm25_results      = make_bm25_results(),
            retrieval_stage_1 = make_bge_results(),
        )
        state = reranker.run(state)
        for chunk in state.retrieval_stage_2:
            assert chunk.get("retriever") == "rrf_reranked"

    def test_20_bm25_only_input_still_works(self, reranker):
        """N09: BM25 results only (no BGE) must still produce output"""
        state = BAState(
            session_id        = "t20",
            query             = "net income 2023",
            bm25_results      = make_bm25_results(),
            retrieval_stage_1 = [],
        )
        state = reranker.run(state)
        assert len(state.retrieval_stage_2) >= 1


# ════════════════════════════════════════════════════════════════════════════
# GROUP 5 — EDGE CASES (tests 21-24)
# ════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_21_single_chunk_input_works(self, reranker):
        """N09: Single chunk from one source must return result"""
        state = BAState(
            session_id        = "t21",
            query             = "net income",
            bm25_results      = [make_bm25_results()[0]],
            retrieval_stage_1 = [],
        )
        state = reranker.run(state)
        assert len(state.retrieval_stage_2) == 1

    def test_22_duplicate_chunks_deduplicated(self, reranker):
        """N09: Same chunk_id appearing in both lists must not duplicate"""
        # chunk_000001 appears in both BM25 and BGE
        result = reranker.rrf_only(
            make_bm25_results(),
            make_bge_results(),
            top_k=10
        )
        chunk_ids = [c["chunk_id"] for c in result]
        assert len(chunk_ids) == len(set(chunk_ids)), \
            "Duplicate chunk_ids found in RRF output"

    def test_23_sources_field_tracks_origin(self, reranker):
        """N09: Chunk in both lists must have both sources listed"""
        result = reranker.rrf_only(
            make_bm25_results(),
            make_bge_results(),
            top_k=10
        )
        # chunk_000001 is rank 1 in BM25, rank 2 in BGE
        dual = next(
            (c for c in result if c["chunk_id"] == "chunk_000001"), None
        )
        assert dual is not None
        assert "bm25"   in dual.get("sources", [])
        assert "bge_m3" in dual.get("sources", [])

    def test_24_c8_metadata_preserved_in_final(self, reranker):
        """C8: Final chunks must preserve company doc_type fiscal_year section page"""
        state = BAState(
            session_id        = "t24",
            query             = "net income Apple",
            bm25_results      = make_bm25_results(),
            retrieval_stage_1 = make_bge_results(),
        )
        state = reranker.run(state)
        required = ["company", "doc_type", "fiscal_year", "section", "page"]
        for chunk in state.retrieval_stage_2:
            for field in required:
                assert field in chunk, \
                    f"C8 metadata field '{field}' missing from final chunk"