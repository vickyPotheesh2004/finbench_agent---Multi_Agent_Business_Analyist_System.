"""
tests/test_section_tree.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 Rev1.0 FINAL

Tests for N02 — Section Tree Builder
Run: pytest tests/test_section_tree.py -v
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.ingestion.section_tree_builder import SectionTreeBuilder
from src.state.ba_state import BAState


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def builder():
    return SectionTreeBuilder()


@pytest.fixture
def fresh_state():
    return BAState(session_id="test-n02")


@pytest.fixture
def state_with_headings():
    """BAState with realistic financial document headings."""
    return BAState(
        session_id="test-n02-headings",
        company_name="Apple Inc",
        doc_type="10-K",
        fiscal_year="FY2023",
        raw_text="Apple Inc Annual Report FY2023\n" * 200,
        heading_positions=[
            {"page": 1,  "text": "APPLE INC ANNUAL REPORT",
             "font_size": 20.0, "is_bold": True,  "bbox": []},
            {"page": 3,  "text": "Business Overview",
             "font_size": 16.0, "is_bold": True,  "bbox": []},
            {"page": 5,  "text": "Products and Services",
             "font_size": 14.0, "is_bold": True,  "bbox": []},
            {"page": 8,  "text": "Risk Factors",
             "font_size": 16.0, "is_bold": True,  "bbox": []},
            {"page": 24, "text": "Management Discussion and Analysis",
             "font_size": 16.0, "is_bold": True,  "bbox": []},
            {"page": 28, "text": "Revenue Analysis",
             "font_size": 14.0, "is_bold": True,  "bbox": []},
            {"page": 42, "text": "Financial Statements",
             "font_size": 16.0, "is_bold": True,  "bbox": []},
            {"page": 44, "text": "Consolidated Balance Sheet",
             "font_size": 14.0, "is_bold": True,  "bbox": []},
            {"page": 46, "text": "Notes to Financial Statements",
             "font_size": 14.0, "is_bold": False, "bbox": []},
        ],
    )


@pytest.fixture
def state_with_text_only():
    """BAState with raw text but no headings (CSV/text fallback)."""
    return BAState(
        session_id="test-n02-text",
        raw_text=(
            "ANNUAL REPORT 2023\n"
            "BUSINESS OVERVIEW\n"
            "We make great products.\n"
            "RISK FACTORS\n"
            "There are many risks.\n"
            "ITEM 7 MANAGEMENT DISCUSSION\n"
            "Revenue was strong.\n"
            "FINANCIAL STATEMENTS\n"
            "See attached tables.\n"
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 1 — Instantiation
# ═══════════════════════════════════════════════════════════════════════════

class TestInstantiation:

    def test_01_builder_instantiates(self, builder):
        """N02: SectionTreeBuilder must instantiate without error"""
        assert builder is not None

    def test_02_empty_state_returns_empty_tree(self, builder, fresh_state):
        """N02: Empty BAState must return empty section tree"""
        state = builder.run(fresh_state)
        assert state.section_tree is not None
        assert state.section_tree["total_sections"] == 0
        assert state.section_tree["sections"] == []


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 2 — Section tree structure
# ═══════════════════════════════════════════════════════════════════════════

class TestSectionTreeStructure:

    def test_03_tree_has_required_keys(self, builder, state_with_headings):
        """N02: Section tree must contain all required top-level keys"""
        state = builder.run(state_with_headings)
        tree  = state.section_tree
        for key in ["total_sections", "sections", "key_sections", "page_count"]:
            assert key in tree, f"Missing key: {key}"

    def test_04_sections_list_not_empty(self, builder, state_with_headings):
        """N02: sections list must not be empty for a document with headings"""
        state = builder.run(state_with_headings)
        assert len(state.section_tree["sections"]) > 0

    def test_05_section_node_has_required_fields(self, builder, state_with_headings):
        """N02: Every section node must have all required fields"""
        state    = builder.run(state_with_headings)
        required = {"id", "title", "level", "page_start", "page_end",
                    "is_bold", "font_size", "summary", "children"}
        sections = state.section_tree["sections"]
        assert len(sections) > 0
        for section in sections:
            assert required.issubset(section.keys()), \
                f"Missing fields: {required - section.keys()}"

    def test_06_section_ids_are_unique(self, builder, state_with_headings):
        """N02: All section IDs must be unique"""
        state = builder.run(state_with_headings)

        def collect_ids(sections):
            ids = []
            for s in sections:
                ids.append(s["id"])
                ids.extend(collect_ids(s.get("children", [])))
            return ids

        all_ids = collect_ids(state.section_tree["sections"])
        assert len(all_ids) == len(set(all_ids)), "Duplicate section IDs found"

    def test_07_page_ranges_are_valid(self, builder, state_with_headings):
        """N02: page_start must be <= page_end for all sections"""
        state = builder.run(state_with_headings)

        def check_pages(sections):
            for s in sections:
                assert s["page_start"] <= s["page_end"], \
                    f"Invalid page range in '{s['title']}': " \
                    f"{s['page_start']} > {s['page_end']}"
                check_pages(s.get("children", []))

        check_pages(state.section_tree["sections"])


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 3 — Key section detection
# ═══════════════════════════════════════════════════════════════════════════

class TestKeySectonDetection:

    def test_08_all_5_key_sections_present_in_tree(
        self, builder, state_with_headings
    ):
        """N02: key_sections dict must contain all 5 keys"""
        state = builder.run(state_with_headings)
        ks    = state.section_tree["key_sections"]
        for key in ["business_overview", "risk_factors", "mda",
                    "financial_statements", "notes"]:
            assert key in ks, f"Missing key section: {key}"

    def test_09_business_overview_detected(self, builder, state_with_headings):
        """N02: Business Overview must be detected"""
        state = builder.run(state_with_headings)
        assert state.section_tree["key_sections"]["business_overview"] is not None

    def test_10_risk_factors_detected(self, builder, state_with_headings):
        """N02: Risk Factors must be detected"""
        state = builder.run(state_with_headings)
        assert state.section_tree["key_sections"]["risk_factors"] is not None

    def test_11_mda_detected(self, builder, state_with_headings):
        """N02: MD&A must be detected"""
        state = builder.run(state_with_headings)
        assert state.section_tree["key_sections"]["mda"] is not None

    def test_12_financial_statements_detected(self, builder, state_with_headings):
        """N02: Financial Statements must be detected"""
        state = builder.run(state_with_headings)
        assert state.section_tree["key_sections"]["financial_statements"] is not None

    def test_13_notes_detected(self, builder, state_with_headings):
        """N02: Notes to Financial Statements must be detected"""
        state = builder.run(state_with_headings)
        assert state.section_tree["key_sections"]["notes"] is not None

    def test_14_mda_correct_page_start(self, builder, state_with_headings):
        """N02: MD&A must start on correct page"""
        state = builder.run(state_with_headings)
        mda   = state.section_tree["key_sections"]["mda"]
        assert mda["page_start"] == 24

    def test_15_risk_factors_correct_page_start(self, builder, state_with_headings):
        """N02: Risk Factors must start on correct page"""
        state = builder.run(state_with_headings)
        rf    = state.section_tree["key_sections"]["risk_factors"]
        assert rf["page_start"] == 8


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 4 — Text fallback
# ═══════════════════════════════════════════════════════════════════════════

class TestTextFallback:

    def test_16_builds_from_text_when_no_headings(
        self, builder, state_with_text_only
    ):
        """N02: Must build tree from raw text when heading_positions is empty"""
        state = builder.run(state_with_text_only)
        assert state.section_tree["total_sections"] > 0

    def test_17_text_fallback_detects_item_headings(
        self, builder, state_with_text_only
    ):
        """N02: Text fallback must detect ITEM headings"""
        state    = builder.run(state_with_text_only)
        titles   = [s["title"] for s in state.section_tree["sections"]]
        has_item = any("ITEM" in t.upper() for t in titles)
        assert has_item, f"No ITEM heading detected. Found: {titles}"


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 5 — Helper methods
# ═══════════════════════════════════════════════════════════════════════════

class TestHelperMethods:

    def test_18_summary_returns_string(self, builder, state_with_headings):
        """N02: summary() must return a non-empty string"""
        state   = builder.run(state_with_headings)
        summary = builder.summary(state)
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_19_summary_contains_section_count(
        self, builder, state_with_headings
    ):
        """N02: summary() must mention section count"""
        state   = builder.run(state_with_headings)
        summary = builder.summary(state)
        assert "section" in summary.lower()

    def test_20_get_section_text_returns_string(
        self, builder, state_with_headings
    ):
        """N02: get_section_text() must return a string"""
        state = builder.run(state_with_headings)
        text  = builder.get_section_text(state, "mda")
        assert isinstance(text, str)

    def test_21_get_section_text_missing_key_returns_empty(
        self, builder, fresh_state
    ):
        """N02: get_section_text() with missing key must return empty string"""
        fresh_state.section_tree = {
            "sections": [], "key_sections": {}, "total_sections": 0
        }
        text = builder.get_section_text(fresh_state, "nonexistent")
        assert text == ""


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 6 — BAState integration
# ═══════════════════════════════════════════════════════════════════════════

class TestBAStateIntegration:

    def test_22_section_tree_written_to_state(
        self, builder, state_with_headings
    ):
        """N02: section_tree must be written to BAState"""
        state = builder.run(state_with_headings)
        assert isinstance(state.section_tree, dict)
        assert state.section_tree != {}

    def test_23_state_seed_unchanged(self, builder, state_with_headings):
        """C5: BAState seed must still be 42 after N02 runs"""
        state = builder.run(state_with_headings)
        assert state.seed == 42

    def test_24_n01_n02_pipeline(self, builder):
        """N02: Must work correctly after N01 output"""
        from src.ingestion.pdf_ingestor import PDFIngestor
        import tempfile, csv, os

        # Create a temp CSV to simulate N01 output
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            writer.writerow(["Revenue", "383.29B"])
            tmp_path = f.name

        try:
            state = BAState(
                session_id="pipeline-test",
                document_path=tmp_path
            )
            # Run N01
            ingestor = PDFIngestor()
            state    = ingestor.run(state)
            # Run N02
            state    = builder.run(state)
            assert state.section_tree is not None
            assert "sections" in state.section_tree
        finally:
            os.unlink(tmp_path)