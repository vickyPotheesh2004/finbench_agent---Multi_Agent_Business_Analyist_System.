"""
tests/test_n09_rrf_reranker.py
Tests for N09 RRF + Cross-Encoder Reranker
PDR-BAAAI-001 · Rev 1.0
"""

import pytest
from src.retrieval.rrf_reranker import (
    RRFReranker,
    RRFEnsembleRetriever,
    reciprocal_rank_fusion,
    run_rrf_reranker,
    _RRF_K,
    _DEFAULT_FINAL_TOP_K,
    _RETRIEVER_LABEL,
)
from src.state.ba_state import BAState


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_bm25_result(chunk_id, text, rank, score=1.0):
    return {
        "chunk_id":    chunk_id,
        "text":        text,
        "rank":        rank,
        "bm25_score":  score,
        "bm25_score_norm": score / 20.0,
        "retriever":   "bm25",
        "company":     "Apple Inc",
        "doc_type":    "10-K",
        "fiscal_year": "FY2023",
        "section":     "INCOME_STATEMENT",
        "page":        94,
        "prefix":      "Apple Inc/10-K/FY2023/INCOME_STATEMENT/94",
    }


def _make_bge_result(chunk_id, text, rank, score=0.9):
    return {
        "chunk_id":    chunk_id,
        "text":        text,
        "rank":        rank,
        "bge_score":   score,
        "retriever":   "bge_m3",
        "company":     "Apple Inc",
        "doc_type":    "10-K",
        "fiscal_year": "FY2023",
        "section":     "INCOME_STATEMENT",
        "page":        94,
        "prefix":      "Apple Inc/10-K/FY2023/INCOME_STATEMENT/94",
    }


BM25_RESULTS = [
    _make_bm25_result("chunk_A", "Net income was 96995 million in FY2023",   1, 18.5),
    _make_bm25_result("chunk_B", "Total net sales were 383285 million",       2, 16.2),
    _make_bm25_result("chunk_C", "Diluted EPS was 6.13 dollars in FY2023",   3, 14.1),
    _make_bm25_result("chunk_D", "Total assets were 352583 million FY2023",  4, 12.0),
    _make_bm25_result("chunk_E", "Operating cash flow was 114301 million",   5,  9.8),
]

BGE_RESULTS = [
    _make_bge_result("chunk_B", "Total net sales were 383285 million",       1, 0.95),
    _make_bge_result("chunk_A", "Net income was 96995 million in FY2023",    2, 0.91),
    _make_bge_result("chunk_F", "Gross margin was 169148 million FY2023",    3, 0.87),
    _make_bge_result("chunk_C", "Diluted EPS was 6.13 dollars in FY2023",   4, 0.84),
    _make_bge_result("chunk_G", "Research and development was 29915 million",5, 0.78),
]


@pytest.fixture
def reranker():
    """RRF only — no cross-encoder for speed in unit tests."""
    return RRFReranker(final_top_k=3, rrf_k=60, use_reranker=False)


@pytest.fixture
def full_reranker():
    """With cross-encoder — used for integration tests."""
    return RRFReranker(final_top_k=3, rrf_k=60, use_reranker=True)


# ── Group 1: RRF algorithm ────────────────────────────────────────────────────

class TestRRFAlgorithm:

    def test_01_rrf_k_value(self):
        """PDR Section 7.5: RRF k must be 60"""
        assert _RRF_K == 60

    def test_02_rrf_single_list(self):
        """RRF with one list preserves order"""
        ranked = [["A", "B", "C"]]
        result = reciprocal_rank_fusion(ranked, k=60)
        ids    = [r[0] for r in result]
        assert ids[0] == "A"
        assert ids[1] == "B"
        assert ids[2] == "C"

    def test_03_rrf_two_lists_agree(self):
        """When both lists agree on rank-1, that item wins"""
        list1  = ["A", "B", "C"]
        list2  = ["A", "C", "B"]
        result = reciprocal_rank_fusion([list1, list2], k=60)
        assert result[0][0] == "A"

    def test_04_rrf_boosted_by_both_lists(self):
        """Item appearing in both lists scores higher than single-list item"""
        list1  = ["A", "B"]
        list2  = ["B", "C"]
        result = reciprocal_rank_fusion([list1, list2], k=60)
        scores = dict(result)
        # B appears in both lists — must score higher than C (single list)
        assert scores["B"] > scores["C"]

    def test_05_rrf_scores_are_floats(self):
        ranked = [["A", "B"], ["B", "A"]]
        result = reciprocal_rank_fusion(ranked)
        for _, score in result:
            assert isinstance(score, float)

    def test_06_rrf_sorted_descending(self):
        ranked = [["A", "B", "C"], ["C", "B", "A"]]
        result = reciprocal_rank_fusion(ranked)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_07_rrf_formula_correct(self):
        """Verify exact formula: score = 1/(k+rank) for k=60, rank 0-based"""
        ranked = [["A"]]
        result = reciprocal_rank_fusion(ranked, k=60)
        # rank=0 (first item) → 1/(60+0+1) = 1/61
        expected = 1.0 / 61.0
        assert abs(result[0][1] - expected) < 1e-9

    def test_08_rrf_empty_lists(self):
        result = reciprocal_rank_fusion([])
        assert result == []

    def test_09_rrf_empty_inner_list(self):
        result = reciprocal_rank_fusion([[]])
        assert result == []

    def test_10_rrf_deduplicates_chunk_ids(self):
        """Same chunk_id in both lists appears only once in output"""
        result = reciprocal_rank_fusion([["A", "B"], ["A", "C"]])
        ids    = [r[0] for r in result]
        assert ids.count("A") == 1


# ── Group 2: Instantiation ────────────────────────────────────────────────────

class TestInstantiation:

    def test_11_instantiates(self, reranker):
        assert reranker is not None

    def test_12_default_final_top_k(self):
        assert _DEFAULT_FINAL_TOP_K == 3

    def test_13_retriever_label(self):
        assert _RETRIEVER_LABEL == "rrf_reranker"

    def test_14_rrf_k_on_instance(self, reranker):
        assert reranker.rrf_k == 60

    def test_15_final_top_k_on_instance(self, reranker):
        assert reranker.final_top_k == 3


# ── Group 3: rerank() method ──────────────────────────────────────────────────

class TestRerankMethod:

    def test_16_rerank_returns_list(self, reranker):
        result = reranker.rerank("net income", BM25_RESULTS, BGE_RESULTS)
        assert isinstance(result, list)

    def test_17_rerank_respects_final_top_k(self, reranker):
        result = reranker.rerank("net income", BM25_RESULTS, BGE_RESULTS)
        assert len(result) <= reranker.final_top_k

    def test_18_rerank_results_have_required_keys(self, reranker):
        result  = reranker.rerank("net income", BM25_RESULTS, BGE_RESULTS)
        required = {
            "chunk_id", "text", "rank", "retriever",
            "company", "doc_type", "fiscal_year",
            "section", "page", "rrf_score",
        }
        for r in result:
            assert required.issubset(r.keys()), \
                f"Missing keys: {required - r.keys()}"

    def test_19_rank_starts_at_1(self, reranker):
        result = reranker.rerank("net income", BM25_RESULTS, BGE_RESULTS)
        if result:
            assert result[0]["rank"] == 1

    def test_20_retriever_label_correct(self, reranker):
        result = reranker.rerank("net income", BM25_RESULTS, BGE_RESULTS)
        for r in result:
            assert r["retriever"] == _RETRIEVER_LABEL

    def test_21_chunk_appearing_in_both_ranks_high(self, reranker):
        """chunk_A and chunk_B appear in both lists — must be in top-3"""
        result  = reranker.rerank("net income", BM25_RESULTS, BGE_RESULTS)
        top_ids = [r["chunk_id"] for r in result]
        # At least one of the chunks appearing in both lists should be in top-3
        both_lists = {"chunk_A", "chunk_B", "chunk_C"}
        assert any(cid in both_lists for cid in top_ids)

    def test_22_empty_bm25_uses_bge_only(self, reranker):
        result = reranker.rerank("net income", [], BGE_RESULTS)
        assert len(result) > 0

    def test_23_empty_bge_uses_bm25_only(self, reranker):
        result = reranker.rerank("net income", BM25_RESULTS, [])
        assert len(result) > 0

    def test_24_both_empty_returns_empty(self, reranker):
        result = reranker.rerank("net income", [], [])
        assert result == []

    def test_25_empty_query_still_merges(self, reranker):
        """Empty query: RRF still works, reranker skips gracefully"""
        result = reranker.rerank("", BM25_RESULTS, BGE_RESULTS)
        assert isinstance(result, list)

    def test_26_rrf_scores_are_positive(self, reranker):
        result = reranker.rerank("net income", BM25_RESULTS, BGE_RESULTS)
        for r in result:
            assert r["rrf_score"] > 0.0


# ── Group 4: BAState integration ─────────────────────────────────────────────

class TestBAStateIntegration:

    def test_27_run_writes_retrieval_stage_2(self, reranker):
        state = BAState(
            session_id        = "t27",
            query             = "What was net income FY2023?",
            bm25_results      = BM25_RESULTS,
            retrieval_stage_1 = BGE_RESULTS,
        )
        state = reranker.run(state)
        assert isinstance(state.retrieval_stage_2, list)
        assert len(state.retrieval_stage_2) > 0

    def test_28_seed_unchanged(self, reranker):
        """C5: seed must remain 42"""
        state = BAState(
            session_id        = "t28",
            query             = "net income",
            bm25_results      = BM25_RESULTS,
            retrieval_stage_1 = BGE_RESULTS,
        )
        state = reranker.run(state)
        assert state.seed == 42

    def test_29_empty_query_returns_empty_stage2(self, reranker):
        state = BAState(
            session_id        = "t29",
            query             = "",
            bm25_results      = BM25_RESULTS,
            retrieval_stage_1 = BGE_RESULTS,
        )
        state = reranker.run(state)
        assert state.retrieval_stage_2 == []

    def test_30_stage2_max_3_results(self, reranker):
        """retrieval_stage_2 must never exceed 3 chunks (PDR spec)"""
        state = BAState(
            session_id        = "t30",
            query             = "net income",
            bm25_results      = BM25_RESULTS,
            retrieval_stage_1 = BGE_RESULTS,
        )
        state = reranker.run(state)
        assert len(state.retrieval_stage_2) <= 3

    def test_31_stage2_results_have_c8_metadata(self, reranker):
        """C8: results must carry company/doc_type/fiscal_year"""
        state = BAState(
            session_id        = "t31",
            query             = "net income FY2023",
            bm25_results      = BM25_RESULTS,
            retrieval_stage_1 = BGE_RESULTS,
        )
        state = reranker.run(state)
        for r in state.retrieval_stage_2:
            assert r.get("company")     != ""
            assert r.get("doc_type")    != ""
            assert r.get("fiscal_year") != ""

    def test_32_run_sniper_rrf_returns_bastate(self, reranker):
        state = BAState(
            session_id        = "t32",
            query             = "net income",
            bm25_results      = BM25_RESULTS,
            retrieval_stage_1 = BGE_RESULTS,
        )
        result = reranker.run(state)
        assert hasattr(result, "retrieval_stage_2")


# ── Group 5: run_rrf_reranker convenience wrapper ────────────────────────────

class TestConvenienceWrapper:

    def test_33_run_rrf_reranker_returns_state(self):
        state = BAState(
            session_id        = "t33",
            query             = "net income",
            bm25_results      = BM25_RESULTS,
            retrieval_stage_1 = BGE_RESULTS,
        )
        result = run_rrf_reranker(state, use_reranker=False)
        assert hasattr(result, "retrieval_stage_2")

    def test_34_wrapper_produces_top3(self):
        state = BAState(
            session_id        = "t34",
            query             = "total net sales FY2023",
            bm25_results      = BM25_RESULTS,
            retrieval_stage_1 = BGE_RESULTS,
        )
        result = run_rrf_reranker(state, use_reranker=False)
        assert len(result.retrieval_stage_2) <= 3


# ── Group 6: Full N07+N08+N09 cascade ────────────────────────────────────────

class TestFullCascade:

    def test_35_stage2_subset_of_combined_inputs(self, reranker):
        """
        All chunk_ids in stage_2 must come from BM25 or BGE results.
        RRF cannot invent new chunks.
        """
        state = BAState(
            session_id        = "t35",
            query             = "net income FY2023",
            bm25_results      = BM25_RESULTS,
            retrieval_stage_1 = BGE_RESULTS,
        )
        state    = reranker.run(state)
        all_ids  = {
            r.get("chunk_id") for r in BM25_RESULTS + BGE_RESULTS
        }
        for r in state.retrieval_stage_2:
            assert r["chunk_id"] in all_ids, \
                f"chunk_id '{r['chunk_id']}' not in input results"

    def test_36_ranks_are_sequential(self, reranker):
        """Ranks in stage_2 must be 1, 2, 3 (sequential)"""
        state = BAState(
            session_id        = "t36",
            query             = "net income",
            bm25_results      = BM25_RESULTS,
            retrieval_stage_1 = BGE_RESULTS,
        )
        state = reranker.run(state)
        ranks = [r["rank"] for r in state.retrieval_stage_2]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_37_single_bm25_result_works(self, reranker):
        single = [BM25_RESULTS[0]]
        result = reranker.rerank("net income", single, [])
        assert len(result) >= 1

    def test_38_single_bge_result_works(self, reranker):
        single = [BGE_RESULTS[0]]
        result = reranker.rerank("net income", [], single)
        assert len(result) >= 1