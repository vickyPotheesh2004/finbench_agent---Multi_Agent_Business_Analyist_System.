"""
tests/test_n07_bm25_regression.py
Regression suite for Bug #1 (BM25 _search returning 0 results) and
Bug #1.1 (bm25s 0.2+ k=1 reshape crash).

These tests would have FAILED before Session 7+8 fixes. After fixes,
all pass.

Phase 1 of FinBench fix campaign.
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.ingestion.chunker        import Chunker
from src.retrieval.bm25_retriever import BM25Retriever
from src.state.ba_state           import BAState


# ── Long fixture text — guaranteed to produce 8+ chunks ─────────────────────

REGRESSION_RAW_TEXT = """Financial Statements

Net sales were 383285 million dollars in fiscal year 2023 for Apple Inc.

Net income was 96995 million dollars for the year ended September 2023.

Diluted earnings per share were 6.13 dollars in fiscal 2023 fully diluted.

Total assets were 352583 million dollars as of September 30 2023 reporting period.

Gross margin was 169148 million representing 44.1 percent of total net sales.

Operating income was 114301 million dollars in fiscal year 2023 fiscal period.

Cash and cash equivalents were 29965 million dollars at fiscal year end 2023.

Long term debt was 95281 million dollars on the balance sheet at year end.

Risk Factors

Competition in each of the Company markets is intense and substantial across all segments.

The Company faces competition from companies with greater financial resources globally.

Changes in global economic conditions could affect demand for products and services.

Foreign exchange rate fluctuations could materially impact the Company financial results.

Management Discussion and Analysis

The Company sees continued strong demand for its products and services worldwide markets.

Services revenue grew significantly during fiscal 2023 driven by subscriptions growth.

Research and development expenses were 29915 million in fiscal 2023 supporting innovation.
"""


@pytest.fixture
def populated_index(tmp_path):
    """Build a real BM25 index with multi-chunk content."""
    chunker = Chunker(
        bm25_dir     = str(tmp_path / "bm25"),
        chromadb_dir = str(tmp_path / "chromadb"),
    )

    state = BAState(
        session_id   = "regression-bug1",
        company_name = "Apple Inc",
        doc_type     = "10-K",
        fiscal_year  = "FY2023",
        raw_text     = REGRESSION_RAW_TEXT,
        section_tree = {},
    )

    old = os.environ.get("DISABLE_CHROMADB")
    os.environ["DISABLE_CHROMADB"] = "1"
    try:
        state = chunker.run(state)
    finally:
        if old is None:
            os.environ.pop("DISABLE_CHROMADB", None)
        else:
            os.environ["DISABLE_CHROMADB"] = old

    # Sanity: chunker must produce ≥ 5 chunks for these tests to be meaningful
    assert state.chunk_count >= 5, (
        f"Fixture failed: expected ≥5 chunks, got {state.chunk_count}. "
        f"Chunker may be over-consolidating paragraphs."
    )
    return state.bm25_index_path


# ════════════════════════════════════════════════════════════════════════════
# CORE BUG #1 TESTS — non-zero results
# ════════════════════════════════════════════════════════════════════════════

class TestBug1NonZeroResults:

    def test_search_returns_nonzero(self, populated_index):
        retriever = BM25Retriever(top_k=5)
        retriever._load_index(populated_index)
        results = retriever._search("net sales fiscal 2023")
        assert len(results) > 0, (
            "Bug #1 regression: BM25 returned 0 results despite "
            "a corpus that contains 'net sales fiscal 2023'."
        )

    def test_run_via_state_populates_results(self, populated_index):
        retriever = BM25Retriever(top_k=5)
        state = BAState(
            session_id      = "bug1-test",
            query           = "net sales fiscal 2023",
            bm25_index_path = populated_index,
        )
        state = retriever.run(state)
        assert isinstance(state.bm25_results, list)
        assert len(state.bm25_results) > 0, (
            "state.bm25_results is empty after run()."
        )

    def test_run_writes_confidence(self, populated_index):
        retriever = BM25Retriever(top_k=5)
        state = BAState(
            session_id      = "bug1-conf",
            query           = "net income",
            bm25_index_path = populated_index,
        )
        state = retriever.run(state)
        assert hasattr(state, "bm25_confidence")
        assert 0.0 <= state.bm25_confidence <= 1.0
        if state.bm25_results:
            assert state.bm25_confidence > 0.0


# ════════════════════════════════════════════════════════════════════════════
# RESULT STRUCTURE
# ════════════════════════════════════════════════════════════════════════════

class TestBug1ResultStructure:

    def test_results_have_required_keys(self, populated_index):
        retriever = BM25Retriever(top_k=5)
        retriever._load_index(populated_index)
        results = retriever._search("net income")
        assert len(results) > 0
        required = {
            "bm25_score", "bm25_score_norm", "rank", "retriever",
            "chunk_id",   "section",         "page",
        }
        for r in results:
            missing = required - set(r.keys())
            assert not missing, f"Result missing keys: {missing}"
            assert r["retriever"] in ("bm25", "bm25_fallback")

    def test_norm_score_in_range(self, populated_index):
        retriever = BM25Retriever(top_k=5)
        retriever._load_index(populated_index)
        results = retriever._search("net income")
        for r in results:
            assert 0.0 <= r["bm25_score_norm"] <= 1.0

    def test_top_norm_score_is_one(self, populated_index):
        retriever = BM25Retriever(top_k=5)
        retriever._load_index(populated_index)
        results = retriever._search("net income fiscal year")
        if not results:
            pytest.skip("No results — separate failure")
        assert results[0]["bm25_score_norm"] == pytest.approx(1.0, abs=1e-6)

    def test_results_sorted_descending(self, populated_index):
        retriever = BM25Retriever(top_k=5)
        retriever._load_index(populated_index)
        results = retriever._search("net income fiscal year")
        if len(results) < 2:
            pytest.skip("Not enough results to test ordering")
        for i in range(len(results) - 1):
            assert (
                results[i]["bm25_score"]
                >= results[i + 1]["bm25_score"]
            )

    def test_ranks_are_sequential(self, populated_index):
        retriever = BM25Retriever(top_k=5)
        retriever._load_index(populated_index)
        results = retriever._search("net income")
        if not results:
            pytest.skip("No results — separate failure")
        for i, r in enumerate(results):
            assert r["rank"] == i + 1


# ════════════════════════════════════════════════════════════════════════════
# EDGE CASES
# ════════════════════════════════════════════════════════════════════════════

class TestBug1EdgeCases:

    def test_no_crash_on_empty_query(self, populated_index):
        retriever = BM25Retriever(top_k=5)
        retriever._load_index(populated_index)
        assert retriever._search("") == []

    def test_no_crash_on_whitespace_query(self, populated_index):
        retriever = BM25Retriever(top_k=5)
        retriever._load_index(populated_index)
        assert retriever._search("   \n\t   ") == []

    def test_no_crash_on_unicode(self, populated_index):
        retriever = BM25Retriever(top_k=5)
        retriever._load_index(populated_index)
        results = retriever._search("résumé Apple's 财报 净收入")
        assert isinstance(results, list)

    def test_no_crash_on_top_k_larger_than_corpus(self, populated_index):
        retriever = BM25Retriever(top_k=1000)
        retriever._load_index(populated_index)
        results = retriever._search("net income")
        assert isinstance(results, list)

    def test_no_crash_on_top_k_one(self, populated_index):
        """Bug #1.1: top_k=1 must not trigger bm25s reshape crash."""
        retriever = BM25Retriever(top_k=1)
        retriever._load_index(populated_index)
        results = retriever._search("net income")
        assert isinstance(results, list)
        assert len(results) <= 1

    def test_no_crash_on_no_match_query(self, populated_index):
        retriever = BM25Retriever(top_k=5)
        retriever._load_index(populated_index)
        results = retriever._search("xyzzyx pqrstuv abcdefghi")
        assert isinstance(results, list)


# ════════════════════════════════════════════════════════════════════════════
# BUG #1.1 — k=1 reshape regression
# ════════════════════════════════════════════════════════════════════════════

class TestBug11K1ReshapeBug:
    """Regression for: 'cannot reshape array of size N into shape (1,1)'
    when bm25s.retrieve() is called with k=1 on small corpora."""

    def test_single_chunk_corpus_returns_that_chunk(self, tmp_path):
        """When corpus has only 1 chunk, return it directly without
        calling bm25s.retrieve() (which would crash)."""
        chunker = Chunker(
            bm25_dir     = str(tmp_path / "bm25"),
            chromadb_dir = str(tmp_path / "chromadb"),
        )
        # Force a single-chunk corpus by giving short text and no breaks
        single_para = (
            "Net sales were 383285 million dollars in fiscal 2023."
        )
        state = BAState(
            session_id   = "bug11-single",
            company_name = "Apple Inc",
            doc_type     = "10-K",
            fiscal_year  = "FY2023",
            raw_text     = single_para,
            section_tree = {},
        )

        old = os.environ.get("DISABLE_CHROMADB")
        os.environ["DISABLE_CHROMADB"] = "1"
        try:
            state = chunker.run(state)
        finally:
            if old is None:
                os.environ.pop("DISABLE_CHROMADB", None)
            else:
                os.environ["DISABLE_CHROMADB"] = old

        if state.chunk_count != 1:
            pytest.skip(
                f"Chunker produced {state.chunk_count} chunks, "
                f"this test only valid for 1-chunk corpora."
            )

        retriever = BM25Retriever(top_k=5)
        retriever._load_index(state.bm25_index_path)
        results = retriever._search("net sales")
        assert len(results) == 1
        assert results[0]["bm25_score"] == pytest.approx(1.0)
        assert results[0]["rank"] == 1


# ════════════════════════════════════════════════════════════════════════════
# CHUNK INTEGRITY
# ════════════════════════════════════════════════════════════════════════════

class TestBug1ResultIntegrity:

    def test_no_rlef_in_results(self, populated_index):
        retriever = BM25Retriever(top_k=5)
        retriever._load_index(populated_index)
        results = retriever._search("net income")
        for r in results:
            assert "_rlef_" not in str(r), (
                f"C9 violation: _rlef_ found in BM25 result"
            )

    def test_chunks_have_company_metadata(self, populated_index):
        retriever = BM25Retriever(top_k=5)
        retriever._load_index(populated_index)
        results = retriever._search("net income")
        if not results:
            pytest.skip("No results — separate failure")
        for r in results:
            assert r.get("company") == "Apple Inc"

    def test_chunks_have_fiscal_year(self, populated_index):
        retriever = BM25Retriever(top_k=5)
        retriever._load_index(populated_index)
        results = retriever._search("net income")
        if not results:
            pytest.skip("No results — separate failure")
        for r in results:
            assert r.get("fiscal_year") == "FY2023"