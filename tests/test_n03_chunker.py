"""
tests/test_n03_chunker.py
Tests for N03 Chunker + Index Builder
PDR-BAAAI-001 · Rev 1.0
"""

import os
import pytest
from src.ingestion.chunker import (
    Chunker,
    DocumentChunk,
    run_chunker,
    MAX_CHUNK_TOKENS,
    MIN_CHUNK_CHARS,
    MAX_CHUNKS_CAP,
    CHUNK_OVERLAP,
    SEED,
)
from src.state.ba_state import BAState


SAMPLE_RAW_TEXT = """
Apple Inc Annual Report 10-K Fiscal Year 2023

Business Overview
Apple Inc designs, manufactures, and markets smartphones, personal computers,
tablets, wearables, and accessories worldwide. The company sells its products
and resells third-party products in most of the major global markets through
its retail stores, online stores, and direct sales force.

Risk Factors
The company faces intense competition from Samsung, Google, Microsoft and others.
Market conditions can adversely affect demand for the company's products and
services. Supply chain disruptions may impact production schedules and costs.

Management Discussion and Analysis
Total net sales were $383,285 million in fiscal 2023, a decrease of 3 percent
compared to fiscal 2022. Net income was $96,995 million. Gross margin was
44.1 percent compared to 43.3 percent in fiscal 2022.

Financial Statements
Consolidated Statements of Operations for the years ended September 30, 2023.
Net sales: $383,285 million. Cost of sales: $214,137 million.
Gross margin: $169,148 million. Operating income: $114,301 million.

Notes to Financial Statements
Note 1 - Summary of Significant Accounting Policies.
The company prepares its consolidated financial statements in conformity with
GAAP. Revenue is recognized when performance obligations are satisfied.
"""

SAMPLE_SECTION_TREE = {
    "document": "root",
    "total_sections": 5,
    "children": [
        {"name": "Business Overview",        "start_page": 3,  "end_page": 6,  "level": 1, "children": []},
        {"name": "Risk Factors",             "start_page": 7,  "end_page": 19, "level": 1, "children": []},
        {"name": "Management Discussion",    "start_page": 20, "end_page": 59, "level": 1, "children": []},
        {"name": "Financial Statements",     "start_page": 60, "end_page": 79, "level": 1, "children": []},
        {"name": "Notes to Financial Statements", "start_page": 80, "end_page": 120, "level": 1, "children": []},
    ],
}


@pytest.fixture
def chunker(tmp_path):
    return Chunker(
        bm25_dir     = str(tmp_path / "bm25"),
        chromadb_dir = str(tmp_path / "chromadb"),
    )


# ── Group 1: Constants ────────────────────────────────────────────────────────

class TestConstants:

    def test_01_max_chunk_tokens_defined(self):
        assert MAX_CHUNK_TOKENS > 0
        assert MAX_CHUNK_TOKENS <= 1024

    def test_02_min_chunk_chars_defined(self):
        assert MIN_CHUNK_CHARS > 0

    def test_03_max_chunks_cap_defined(self):
        assert MAX_CHUNKS_CAP > 0

    def test_04_chunk_overlap_defined(self):
        assert CHUNK_OVERLAP >= 0

    def test_05_seed_is_42(self):
        assert SEED == 42


# ── Group 2: DocumentChunk ────────────────────────────────────────────────────

class TestDocumentChunk:

    def test_06_creates_with_all_fields(self):
        chunk = DocumentChunk(
            chunk_id    = "chunk_0001",
            text        = "This is the chunk text content.",
            company     = "Apple Inc",
            doc_type    = "10-K",
            fiscal_year = "FY2023",
            section     = "INCOME_STATEMENT",
            page        = 94,
        )
        assert chunk.chunk_id    == "chunk_0001"
        assert chunk.company     == "Apple Inc"
        assert chunk.doc_type    == "10-K"
        assert chunk.fiscal_year == "FY2023"
        assert chunk.section     == "INCOME_STATEMENT"
        assert chunk.page        == 94

    def test_07_prefix_contains_all_5_fields(self):
        """C8: prefix must contain all 5 mandatory fields."""
        chunk = DocumentChunk(
            "c1", "text", "Apple Inc", "10-K", "FY2023", "MD&A", 42
        )
        assert "Apple Inc"  in chunk.prefix
        assert "10-K"       in chunk.prefix
        assert "FY2023"     in chunk.prefix
        assert "MD&A"       in chunk.prefix
        assert "42"         in chunk.prefix

    def test_08_prefixed_text_has_prefix_before_content(self):
        """C7-style: prefix must come before content."""
        chunk = DocumentChunk(
            "c1", "actual content here", "Apple", "10-K", "FY2023", "SEC", 1
        )
        idx_prefix  = chunk.prefixed_text.find("Apple")
        idx_content = chunk.prefixed_text.find("actual content")
        assert idx_prefix < idx_content

    def test_09_to_dict_has_required_keys(self):
        chunk = DocumentChunk("c1", "text", "Co", "10-K", "FY2023", "SEC", 1)
        d     = chunk.to_dict()
        assert "chunk_id"       in d
        assert "text"           in d
        assert "prefix"         in d
        assert "company"        in d
        assert "doc_type"       in d
        assert "fiscal_year"    in d
        assert "section"        in d
        assert "page"           in d
        assert "char_count"     in d
        assert "token_estimate" in d

    def test_10_char_count_correct(self):
        text  = "Hello World this is test"
        chunk = DocumentChunk("c1", text, "Co", "10-K", "FY2023", "SEC", 1)
        assert chunk.char_count == len(text)

    def test_11_no_rlef_in_prefix(self):
        """C9: prefix must never contain _rlef_."""
        chunk = DocumentChunk("c1", "text", "Co", "10-K", "FY2023", "SEC", 1)
        assert "_rlef_" not in chunk.prefix


# ── Group 3: Chunk method ─────────────────────────────────────────────────────

class TestChunkMethod:

    def test_12_returns_list(self, chunker):
        result = chunker.chunk(SAMPLE_RAW_TEXT, SAMPLE_SECTION_TREE, "Apple", "10-K", "FY2023")
        assert isinstance(result, list)

    def test_13_returns_document_chunks(self, chunker):
        result = chunker.chunk(SAMPLE_RAW_TEXT, SAMPLE_SECTION_TREE, "Apple", "10-K", "FY2023")
        assert len(result) > 0
        assert all(isinstance(c, DocumentChunk) for c in result)

    def test_14_all_chunks_have_c8_prefix(self, chunker):
        """C8: all chunks must have 5-field metadata prefix."""
        result = chunker.chunk(SAMPLE_RAW_TEXT, SAMPLE_SECTION_TREE, "Apple", "10-K", "FY2023")
        for chunk in result:
            assert chunk.company     != ""
            assert chunk.doc_type    != ""
            assert chunk.fiscal_year != ""
            assert chunk.section     != ""
            assert chunk.page        is not None

    def test_15_no_chunk_exceeds_max_tokens(self, chunker):
        result = chunker.chunk(SAMPLE_RAW_TEXT, SAMPLE_SECTION_TREE, "Apple", "10-K", "FY2023")
        for chunk in result:
            assert chunk.token_estimate <= MAX_CHUNK_TOKENS + 50  # small tolerance

    def test_16_no_chunk_shorter_than_min(self, chunker):
        result = chunker.chunk(SAMPLE_RAW_TEXT, SAMPLE_SECTION_TREE, "Apple", "10-K", "FY2023")
        for chunk in result:
            assert chunk.char_count >= MIN_CHUNK_CHARS

    def test_17_empty_text_returns_empty_list(self, chunker):
        result = chunker.chunk("", {}, "Apple", "10-K", "FY2023")
        assert result == []

    def test_18_chunks_capped_at_max(self, chunker):
        long_text = ("Apple Inc revenue was 383285 million. " * 200)
        result = chunker.chunk(long_text, {}, "Apple", "10-K", "FY2023")
        assert len(result) <= MAX_CHUNKS_CAP

    def test_19_fallback_works_without_section_tree(self, chunker):
        result = chunker.chunk(SAMPLE_RAW_TEXT, {}, "Apple", "10-K", "FY2023")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_20_company_propagated_to_all_chunks(self, chunker):
        result = chunker.chunk(SAMPLE_RAW_TEXT, {}, "MyCompany", "10-K", "FY2023")
        for chunk in result:
            assert chunk.company == "MyCompany"

    def test_21_doc_type_propagated(self, chunker):
        result = chunker.chunk(SAMPLE_RAW_TEXT, {}, "Apple", "10-Q", "FY2023")
        for chunk in result:
            assert chunk.doc_type == "10-Q"

    def test_22_fiscal_year_propagated(self, chunker):
        result = chunker.chunk(SAMPLE_RAW_TEXT, {}, "Apple", "10-K", "FY2024")
        for chunk in result:
            assert chunk.fiscal_year == "FY2024"


# ── Group 4: C8 metadata validation ──────────────────────────────────────────

class TestC8Validation:

    def test_23_assert_metadata_passes_valid_chunks(self):
        chunks = [
            DocumentChunk("c1", "valid text content here ok", "Apple", "10-K", "FY2023", "MD&A", 1),
        ]
        Chunker._assert_metadata_prefixes(chunks)  # should not raise

    def test_24_empty_company_raises(self):
        chunks = [
            DocumentChunk("c1", "text", "", "10-K", "FY2023", "SEC", 1),
        ]
        with pytest.raises(ValueError):
            Chunker._assert_metadata_prefixes(chunks)

    def test_25_empty_doc_type_raises(self):
        chunks = [
            DocumentChunk("c1", "text", "Apple", "", "FY2023", "SEC", 1),
        ]
        with pytest.raises(ValueError):
            Chunker._assert_metadata_prefixes(chunks)

    def test_26_empty_fiscal_year_raises(self):
        chunks = [
            DocumentChunk("c1", "text", "Apple", "10-K", "", "SEC", 1),
        ]
        with pytest.raises(ValueError):
            Chunker._assert_metadata_prefixes(chunks)

    def test_27_empty_section_raises(self):
        chunks = [
            DocumentChunk("c1", "text", "Apple", "10-K", "FY2023", "", 1),
        ]
        with pytest.raises(ValueError):
            Chunker._assert_metadata_prefixes(chunks)


# ── Group 5: BAState integration ─────────────────────────────────────────────

class TestBAStateIntegration:

    def test_28_run_writes_chunk_count(self, chunker):
        state = BAState(
            session_id   = "t28",
            raw_text     = SAMPLE_RAW_TEXT,
            section_tree = SAMPLE_SECTION_TREE,
            company_name = "Apple Inc",
            doc_type     = "10-K",
            fiscal_year  = "FY2023",
        )
        state = chunker.run(state)
        assert state.chunk_count > 0

    def test_29_run_writes_bm25_path(self, chunker):
        state = BAState(
            session_id   = "t29",
            raw_text     = SAMPLE_RAW_TEXT,
            section_tree = SAMPLE_SECTION_TREE,
            company_name = "Apple Inc",
            doc_type     = "10-K",
            fiscal_year  = "FY2023",
        )
        state = chunker.run(state)
        assert isinstance(state.bm25_index_path, str)
        assert len(state.bm25_index_path) > 0

    def test_30_run_writes_chromadb_collection(self, chunker):
        state = BAState(
            session_id   = "t30",
            raw_text     = SAMPLE_RAW_TEXT,
            section_tree = SAMPLE_SECTION_TREE,
            company_name = "Apple Inc",
            doc_type     = "10-K",
            fiscal_year  = "FY2023",
        )
        state = chunker.run(state)
        assert isinstance(state.chromadb_collection, str)

    def test_31_empty_raw_text_skips_gracefully(self, chunker):
        state = BAState(session_id="t31", raw_text="")
        state = chunker.run(state)
        assert state.chunk_count == 0

    def test_32_seed_unchanged(self, chunker):
        """C5: seed must remain 42."""
        state = BAState(
            session_id   = "t32",
            raw_text     = SAMPLE_RAW_TEXT,
            section_tree = SAMPLE_SECTION_TREE,
            company_name = "Apple",
            doc_type     = "10-K",
            fiscal_year  = "FY2023",
        )
        state = chunker.run(state)
        assert state.seed == 42

    def test_33_no_rlef_in_chunks(self, chunker):
        """C9: chunks must not contain _rlef_ content."""
        result = chunker.chunk(
            SAMPLE_RAW_TEXT, SAMPLE_SECTION_TREE, "Apple", "10-K", "FY2023"
        )
        for chunk in result:
            assert "_rlef_" not in chunk.text
            assert "_rlef_" not in chunk.prefix


# ── Group 6: Convenience wrapper ─────────────────────────────────────────────

class TestConvenienceWrapper:

    def test_34_run_chunker_returns_state(self, tmp_path):
        state = BAState(
            session_id   = "t34",
            raw_text     = SAMPLE_RAW_TEXT,
            section_tree = SAMPLE_SECTION_TREE,
            company_name = "Apple Inc",
            doc_type     = "10-K",
            fiscal_year  = "FY2023",
        )
        result = run_chunker(
            state,
            bm25_dir     = str(tmp_path / "bm25"),
            chromadb_dir = str(tmp_path / "chromadb"),
        )
        assert hasattr(result, "chunk_count")
        assert result.seed == 42

    def test_35_wrapper_creates_chunks(self, tmp_path):
        state = BAState(
            session_id   = "t35",
            raw_text     = SAMPLE_RAW_TEXT,
            section_tree = SAMPLE_SECTION_TREE,
            company_name = "Apple Inc",
            doc_type     = "10-K",
            fiscal_year  = "FY2023",
        )
        result = run_chunker(
            state,
            bm25_dir     = str(tmp_path / "bm25"),
            chromadb_dir = str(tmp_path / "chromadb"),
        )
        assert result.chunk_count > 0