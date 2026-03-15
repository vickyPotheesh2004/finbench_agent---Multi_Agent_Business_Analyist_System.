"""
tests/test_chunker.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 Rev1.0 FINAL

Tests for N03 — Chunker + Indexer
Run: pytest tests/test_chunker.py -v
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.ingestion.chunker import Chunker
from src.state.ba_state import BAState


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def chunker(tmp_path):
    return Chunker(data_dir=str(tmp_path))


@pytest.fixture
def fresh_state():
    return BAState(session_id="test-n03")


@pytest.fixture
def state_with_content():
    """BAState with realistic financial document content."""
    return BAState(
        session_id   = "test-n03-content",
        company_name = "Apple Inc",
        doc_type     = "10-K",
        fiscal_year  = "FY2023",
        raw_text     = """
Business Overview

Apple Inc designs, manufactures, and markets smartphones, personal computers,
tablets, wearables and accessories worldwide. The Company sells a range of
related services. iPhone is the Company primary product line and represented
52 percent of total net revenue in fiscal 2023.

Risk Factors

The Company operates in highly competitive markets. Competition in each of
the Company markets is intense and the Company expects it to remain so.
A failure to obtain or create the content or services our customers desire
could impact our business and financial performance significantly.

Management Discussion and Analysis

The Company total net revenue decreased 3 percent or 14.3 billion during 2023
compared to 2022. Products segment net sales were 298.1 billion in 2023.
Services net sales were 85.2 billion in fiscal 2023 representing growth.
International net sales accounted for 58 percent of total net sales.

Financial Statements

Net sales: 383285 million dollars
Cost of sales: 214137 million dollars
Gross margin: 169148 million dollars
Net income: 96995 million dollars
Earnings per share diluted: 6.13 dollars

Notes to Financial Statements

Note 1 Summary of Significant Accounting Policies.
The consolidated financial statements include accounts of Apple Inc
and its wholly owned subsidiaries across all regions worldwide.
        """,
        section_tree = {
            "sections": [
                {"id": "sec_0000", "title": "Business Overview",
                 "level": 1, "page_start": 3, "page_end": 7,
                 "is_bold": True, "font_size": 16.0,
                 "summary": "", "children": [], "key_section": None},
                {"id": "sec_0001", "title": "Risk Factors",
                 "level": 1, "page_start": 8, "page_end": 23,
                 "is_bold": True, "font_size": 16.0,
                 "summary": "", "children": [], "key_section": None},
                {"id": "sec_0002",
                 "title": "Management Discussion and Analysis",
                 "level": 1, "page_start": 24, "page_end": 41,
                 "is_bold": True, "font_size": 16.0,
                 "summary": "", "children": [], "key_section": None},
                {"id": "sec_0003", "title": "Financial Statements",
                 "level": 1, "page_start": 42, "page_end": 45,
                 "is_bold": True, "font_size": 16.0,
                 "summary": "", "children": [], "key_section": None},
                {"id": "sec_0004",
                 "title": "Notes to Financial Statements",
                 "level": 1, "page_start": 46, "page_end": 60,
                 "is_bold": True, "font_size": 16.0,
                 "summary": "", "children": [], "key_section": None},
            ],
            "key_sections": {},
            "total_sections": 5,
            "page_count": 60,
        }
    )


@pytest.fixture
def state_no_sections():
    """BAState with text but no section tree."""
    return BAState(
        session_id   = "test-n03-nosec",
        company_name = "Goldman Sachs",
        doc_type     = "10-K",
        fiscal_year  = "FY2022",
        raw_text     = """
Goldman Sachs Annual Report 2022.

The firm reported net revenues of 47.4 billion for the full year 2022.
Global Markets generated net revenues of 21.6 billion in 2022.
Investment Banking net revenues were 7.3 billion in 2022.
Consumer and Wealth Management revenues were 6.5 billion.
Total assets were 1.44 trillion at year end December 2022.

Net earnings applicable to common shareholders were 10.8 billion.
Diluted earnings per common share were 30.06 for the full year.
Return on average common shareholders equity was 10.2 percent.
Book value per common share increased to 301.30 at year end.
        """,
        section_tree={}
    )


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 1 — Instantiation
# ═══════════════════════════════════════════════════════════════════════════

class TestInstantiation:

    def test_01_chunker_instantiates(self, chunker):
        """N03: Chunker must instantiate without error"""
        assert chunker is not None

    def test_02_empty_state_returns_zero_chunks(self, chunker, fresh_state):
        """N03: Empty BAState must return chunk_count=0"""
        state = chunker.run(fresh_state)
        assert state.chunk_count == 0

    def test_03_data_dirs_created(self, chunker):
        """N03: data/bm25_index and data/chromadb must exist"""
        assert chunker.bm25_dir.exists()
        assert chunker.chromadb_dir.exists()


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 2 — Chunk creation
# ═══════════════════════════════════════════════════════════════════════════

class TestChunkCreation:

    def test_04_chunks_created(self, chunker, state_with_content):
        """N03: Must create at least 1 chunk from document"""
        state = chunker.run(state_with_content)
        assert state.chunk_count > 0

    def test_05_chunk_count_written_to_state(self, chunker, state_with_content):
        """N03: chunk_count must be written to BAState"""
        state = chunker.run(state_with_content)
        assert isinstance(state.chunk_count, int)
        assert state.chunk_count > 0

    def test_06_chunks_without_sections(self, chunker, state_no_sections):
        """N03: Must chunk document even without section tree"""
        state = chunker.run(state_no_sections)
        assert state.chunk_count > 0

    def test_07_all_chunk_ids_unique(self, chunker, state_with_content):
        """N03: All chunk IDs must be globally unique"""
        state       = chunker.run(state_with_content)
        chunks_meta = chunker.load_chunks_meta(state.bm25_index_path)
        all_ids     = [c["chunk_id"] for c in chunks_meta]
        assert len(all_ids) == len(set(all_ids)), "Duplicate chunk IDs found"

    def test_08_chunks_not_empty(self, chunker, state_with_content):
        """N03: No chunk must have empty text"""
        state       = chunker.run(state_with_content)
        chunks_meta = chunker.load_chunks_meta(state.bm25_index_path)
        for chunk in chunks_meta:
            assert chunk["text"].strip() != "", \
                f"Empty chunk found: {chunk['chunk_id']}"


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 3 — C8 metadata enforcement
# ═══════════════════════════════════════════════════════════════════════════

class TestC8MetadataEnforcement:

    def test_09_every_chunk_has_prefix(self, chunker, state_with_content):
        """C8: Every chunk must have a metadata prefix"""
        state       = chunker.run(state_with_content)
        chunks_meta = chunker.load_chunks_meta(state.bm25_index_path)
        for chunk in chunks_meta:
            assert "prefix" in chunk
            assert chunk["prefix"] != ""

    def test_10_prefix_contains_company(self, chunker, state_with_content):
        """C8: Prefix must contain company name"""
        state       = chunker.run(state_with_content)
        chunks_meta = chunker.load_chunks_meta(state.bm25_index_path)
        for chunk in chunks_meta:
            assert "Apple Inc" in chunk["prefix"], \
                f"Company missing from prefix: {chunk['prefix']}"

    def test_11_prefix_contains_doc_type(self, chunker, state_with_content):
        """C8: Prefix must contain doc_type"""
        state       = chunker.run(state_with_content)
        chunks_meta = chunker.load_chunks_meta(state.bm25_index_path)
        for chunk in chunks_meta:
            assert "10-K" in chunk["prefix"], \
                f"doc_type missing from prefix: {chunk['prefix']}"

    def test_12_prefix_contains_fiscal_year(self, chunker, state_with_content):
        """C8: Prefix must contain fiscal_year"""
        state       = chunker.run(state_with_content)
        chunks_meta = chunker.load_chunks_meta(state.bm25_index_path)
        for chunk in chunks_meta:
            assert "FY2023" in chunk["prefix"], \
                f"fiscal_year missing from prefix: {chunk['prefix']}"

    def test_13_prefix_contains_section(self, chunker, state_with_content):
        """C8: Prefix must contain section name"""
        state       = chunker.run(state_with_content)
        chunks_meta = chunker.load_chunks_meta(state.bm25_index_path)
        for chunk in chunks_meta:
            assert chunk["section"] in chunk["prefix"], \
                f"section missing from prefix: {chunk['prefix']}"

    def test_14_prefix_at_start_of_text(self, chunker, state_with_content):
        """C8: Prefix must appear at the start of chunk text"""
        state       = chunker.run(state_with_content)
        chunks_meta = chunker.load_chunks_meta(state.bm25_index_path)
        for chunk in chunks_meta:
            assert chunk["text"].startswith(chunk["prefix"]), \
                f"Prefix not at start of text in chunk {chunk['chunk_id']}"

    def test_15_chunk_has_all_5_metadata_fields(
        self, chunker, state_with_content
    ):
        """C8: Every chunk must have all 5 required metadata fields"""
        state       = chunker.run(state_with_content)
        chunks_meta = chunker.load_chunks_meta(state.bm25_index_path)
        required    = ["company", "doc_type", "fiscal_year", "section", "page"]
        for chunk in chunks_meta:
            for field in required:
                assert field in chunk, \
                    f"Missing field '{field}' in chunk {chunk['chunk_id']}"

    def test_16_c8_violation_raises(self, chunker):
        """C8: _validate_chunks must raise on missing field"""
        bad_chunks = [
            {
                "chunk_id": "chunk_000000",
                "text":     "some text",
                "prefix":   "prefix",
                "company":  "Apple",
                # Missing: doc_type, fiscal_year, section, page
            }
        ]
        with pytest.raises(ValueError, match="C8"):
            chunker._validate_chunks(bad_chunks)


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 4 — BM25 index
# ═══════════════════════════════════════════════════════════════════════════

class TestBM25Index:

    def test_17_bm25_path_written_to_state(self, chunker, state_with_content):
        """N03: bm25_index_path must be written to BAState"""
        state = chunker.run(state_with_content)
        assert state.bm25_index_path != ""

    def test_18_bm25_index_exists_on_disk(self, chunker, state_with_content):
        """N03: BM25 index must exist on disk after building"""
        state = chunker.run(state_with_content)
        assert Path(state.bm25_index_path).exists()

    def test_19_bm25_loads_correctly(self, chunker, state_with_content):
        """N03: BM25 index must load without error"""
        state     = chunker.run(state_with_content)
        retriever = chunker.load_bm25(state.bm25_index_path)
        assert retriever is not None

    def test_20_chunks_meta_json_exists(self, chunker, state_with_content):
        """N03: chunks_meta.json must exist alongside BM25 index"""
        state     = chunker.run(state_with_content)
        meta_path = Path(state.bm25_index_path) / "chunks_meta.json"
        assert meta_path.exists()

    def test_21_chunks_meta_matches_chunk_count(
        self, chunker, state_with_content
    ):
        """N03: chunks_meta.json must contain same count as chunk_count"""
        state       = chunker.run(state_with_content)
        chunks_meta = chunker.load_chunks_meta(state.bm25_index_path)
        assert len(chunks_meta) == state.chunk_count


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 5 — ChromaDB index
# ═══════════════════════════════════════════════════════════════════════════

class TestChromaDBIndex:

    def test_22_chromadb_collection_written_to_state(
        self, chunker, state_with_content
    ):
        """N03: chromadb_collection must be written to BAState"""
        state = chunker.run(state_with_content)
        assert state.chromadb_collection != ""

    def test_23_chromadb_collection_name_format(
        self, chunker, state_with_content
    ):
        """N03: ChromaDB collection name must follow finbench_ prefix"""
        state = chunker.run(state_with_content)
        assert state.chromadb_collection.startswith("finbench_")


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 6 — Full N01+N02+N03 pipeline
# ═══════════════════════════════════════════════════════════════════════════

class TestFullIngestionPipeline:

    def test_24_n01_n02_n03_pipeline(self, chunker, tmp_path):
        """N03: Must work correctly after N01 + N02"""
        import csv
        from src.ingestion.pdf_ingestor import PDFIngestor
        from src.ingestion.section_tree_builder import SectionTreeBuilder

        # Create temp CSV
        csv_file = tmp_path / "financials.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "FY2022", "FY2023"])
            writer.writerow(["Revenue", "394.33B", "383.29B"])
            writer.writerow(["Net Income", "99.80B", "96.99B"])
            writer.writerow(["EPS", "6.11", "6.13"])
            writer.writerow(["Gross Margin", "43.3%", "44.1%"])

        state = BAState(
            session_id   = "pipeline-n03",
            document_path = str(csv_file)
        )

        # N01
        ingestor = PDFIngestor()
        state    = ingestor.run(state)
        assert state.raw_text != ""

        # N02
        builder = SectionTreeBuilder()
        state   = builder.run(state)
        assert state.section_tree is not None

        # N03
        state = chunker.run(state)
        assert state.chunk_count > 0
        assert state.bm25_index_path != ""
        assert state.chromadb_collection != ""

        # Verify C8 on pipeline output
        chunks_meta = chunker.load_chunks_meta(state.bm25_index_path)
        assert len(chunks_meta) > 0
        for chunk in chunks_meta:
            assert "prefix" in chunk
            assert chunk["prefix"] != ""