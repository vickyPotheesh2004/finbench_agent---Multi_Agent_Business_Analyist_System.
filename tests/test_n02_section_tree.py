"""
tests/test_n02_section_tree.py
Tests for N02 Section Tree Builder
PDR-BAAAI-001 Rev 1.0
"""
import pytest
from src.ingestion.section_tree_builder import (
    SectionTreeBuilder,
    SectionNode,
    run_section_tree_builder,
    SEC_MAJOR_SECTIONS,
    HEADING_H1_MIN,
    HEADING_H2_MIN,
    MAX_HEADING_LENGTH,
    MIN_HEADING_LENGTH,
)
from src.state.ba_state import BAState

SAMPLE_HEADINGS = [
    {"text": "Business Overview",           "font_size": 18.0, "is_bold": True,  "page": 3},
    {"text": "Products and Services",       "font_size": 14.0, "is_bold": True,  "page": 4},
    {"text": "Competition",                 "font_size": 13.5, "is_bold": False, "page": 5},
    {"text": "Risk Factors",                "font_size": 18.0, "is_bold": True,  "page": 7},
    {"text": "Market Risk",                 "font_size": 14.0, "is_bold": True,  "page": 8},
    {"text": "Liquidity Risk",              "font_size": 13.5, "is_bold": False, "page": 9},
    {"text": "Management Discussion",       "font_size": 18.0, "is_bold": True,  "page": 20},
    {"text": "Revenue Analysis",            "font_size": 14.0, "is_bold": True,  "page": 21},
    {"text": "Financial Statements",        "font_size": 18.0, "is_bold": True,  "page": 60},
    {"text": "Income Statement",            "font_size": 14.0, "is_bold": True,  "page": 61},
    {"text": "Balance Sheet",               "font_size": 14.0, "is_bold": True,  "page": 63},
    {"text": "Notes to Financial Statements", "font_size": 18.0, "is_bold": True, "page": 80},
]

SAMPLE_RAW_TEXT = """
Apple Inc Annual Report 10-K Fiscal Year 2023

Business Overview
Apple Inc designs, manufactures, and markets smartphones worldwide.

Risk Factors
The company faces competition from Samsung, Google, and Microsoft.

Management Discussion and Analysis
Total net sales were 383285 million in fiscal 2023.

Financial Statements
The consolidated financial statements are prepared in accordance with GAAP.

Notes to Financial Statements
Note 1: Summary of Significant Accounting Policies.
"""

EMPTY_HEADINGS = []

MINIMAL_HEADINGS = [
    {"text": "Business Overview", "font_size": 16.0, "is_bold": True, "page": 1},
]

NOISY_HEADINGS = [
    {"text": "",                   "font_size": 13.0, "is_bold": False, "page": 1},
    {"text": "1.",                 "font_size": 13.0, "is_bold": False, "page": 2},
    {"text": "A" * 250,            "font_size": 16.0, "is_bold": True,  "page": 3},
    {"text": "Valid Heading Here", "font_size": 16.0, "is_bold": True,  "page": 4},
]


@pytest.fixture
def builder():
    return SectionTreeBuilder(llm_client=None)


class TestConstants:

    def test_01_sec_major_sections_defined(self):
        assert len(SEC_MAJOR_SECTIONS) >= 5

    def test_02_business_in_sec_sections(self):
        assert "business" in SEC_MAJOR_SECTIONS

    def test_03_risk_in_sec_sections(self):
        assert "risk factor" in SEC_MAJOR_SECTIONS

    def test_04_financial_statements_in_sec(self):
        assert "financial statement" in SEC_MAJOR_SECTIONS

    def test_05_heading_thresholds_defined(self):
        assert HEADING_H1_MIN > HEADING_H2_MIN
        assert HEADING_H2_MIN >= 13.0

    def test_06_length_constants(self):
        assert MAX_HEADING_LENGTH > MIN_HEADING_LENGTH
        assert MIN_HEADING_LENGTH >= 2


class TestSectionNode:

    def test_07_creates_correctly(self):
        node = SectionNode("Business Overview", 1, 3, 10, 18.0, True)
        assert node.name       == "Business Overview"
        assert node.level      == 1
        assert node.start_page == 3
        assert node.end_page   == 10
        assert node.font_size  == 18.0
        assert node.is_bold    is True

    def test_08_to_dict_has_required_keys(self):
        node = SectionNode("Business", 1, 3)
        d    = node.to_dict()
        assert "name"       in d
        assert "level"      in d
        assert "start_page" in d
        assert "end_page"   in d
        assert "summary"    in d
        assert "sec_type"   in d
        assert "children"   in d

    def test_09_children_is_empty_list(self):
        node = SectionNode("Test", 1, 1)
        assert node.children == []

    def test_10_children_to_dict(self):
        parent = SectionNode("Parent", 1, 1)
        child  = SectionNode("Child",  2, 2)
        parent.children.append(child)
        d = parent.to_dict()
        assert len(d["children"]) == 1
        assert d["children"][0]["name"] == "Child"


class TestBuildMethod:

    def test_11_build_returns_dict(self, builder):
        result = builder.build(SAMPLE_HEADINGS, SAMPLE_RAW_TEXT)
        assert isinstance(result, dict)

    def test_12_result_has_required_keys(self, builder):
        result = builder.build(SAMPLE_HEADINGS, SAMPLE_RAW_TEXT)
        assert "document"       in result
        assert "total_sections" in result
        assert "children"       in result

    def test_13_children_is_list(self, builder):
        result = builder.build(SAMPLE_HEADINGS, SAMPLE_RAW_TEXT)
        assert isinstance(result["children"], list)

    def test_14_top_level_sections_found(self, builder):
        result = builder.build(SAMPLE_HEADINGS, SAMPLE_RAW_TEXT)
        assert len(result["children"]) >= 4

    def test_15_total_sections_count(self, builder):
        result = builder.build(SAMPLE_HEADINGS, SAMPLE_RAW_TEXT)
        assert result["total_sections"] >= len(result["children"])

    def test_16_empty_headings_returns_empty_tree(self, builder):
        result = builder.build(EMPTY_HEADINGS, "")
        assert result["total_sections"] == 0
        assert result["children"]       == []

    def test_17_children_have_nested_structure(self, builder):
        result = builder.build(SAMPLE_HEADINGS, SAMPLE_RAW_TEXT)
        has_children = any(
            len(s.get("children", [])) > 0
            for s in result["children"]
        )
        assert has_children

    def test_18_each_section_has_page_info(self, builder):
        result = builder.build(SAMPLE_HEADINGS, SAMPLE_RAW_TEXT)
        for section in result["children"]:
            assert "start_page" in section
            assert section["start_page"] >= 0


class TestSECClassification:

    def test_19_business_overview_classified(self, builder):
        result    = builder.build(SAMPLE_HEADINGS, SAMPLE_RAW_TEXT)
        sec_types = [s["sec_type"] for s in result["children"]]
        assert "Business Overview" in sec_types

    def test_20_risk_factors_classified(self, builder):
        result    = builder.build(SAMPLE_HEADINGS, SAMPLE_RAW_TEXT)
        sec_types = [s["sec_type"] for s in result["children"]]
        assert "Risk Factors" in sec_types

    def test_21_financial_statements_classified(self, builder):
        result    = builder.build(SAMPLE_HEADINGS, SAMPLE_RAW_TEXT)
        sec_types = [s["sec_type"] for s in result["children"]]
        assert "Financial Statements" in sec_types

    def test_22_get_sec_type_business(self):
        assert SectionTreeBuilder._get_sec_type("Business Overview") == "Business Overview"

    def test_23_get_sec_type_risk(self):
        assert SectionTreeBuilder._get_sec_type("Risk Factors and Uncertainties") == "Risk Factors"

    def test_24_get_sec_type_notes(self):
        assert SectionTreeBuilder._get_sec_type("Notes to Financial Statements") == "Notes"

    def test_25_get_sec_type_unknown(self):
        assert SectionTreeBuilder._get_sec_type("Something Unknown Here") == "Other"


class TestHeadingCleaning:

    def test_26_empty_headings_filtered(self, builder):
        result = builder.build(NOISY_HEADINGS, "")
        names  = [s["name"] for s in result["children"]]
        assert "" not in names

    def test_27_number_only_headings_filtered(self, builder):
        result = builder.build(NOISY_HEADINGS, "")
        names  = [s["name"] for s in result["children"]]
        assert "1." not in names

    def test_28_too_long_headings_filtered(self, builder):
        result = builder.build(NOISY_HEADINGS, "")
        for s in result["children"]:
            assert len(s["name"]) <= MAX_HEADING_LENGTH

    def test_29_valid_heading_kept(self, builder):
        result = builder.build(NOISY_HEADINGS, "")
        names  = [s["name"] for s in result["children"]]
        assert "Valid Heading Here" in names


class TestLevelAssignment:

    def test_30_large_font_is_h1(self, builder):
        headings = [
            {"text": "Big Heading",   "font_size": 20.0, "is_bold": True,  "page": 1},
            {"text": "Small Heading", "font_size": 13.0, "is_bold": False, "page": 2},
        ]
        levelled = builder._assign_levels(headings)
        assert levelled[0]["level"] == 1
        assert levelled[1]["level"] == 2

    def test_31_bold_h2_font_becomes_h1(self, builder):
        headings = [{"text": "Bold Section", "font_size": 14.0, "is_bold": True, "page": 1}]
        levelled = builder._assign_levels(headings)
        assert levelled[0]["level"] == 1

    def test_32_page_ranges_assigned(self, builder):
        headings = [
            {"text": "Section A", "font_size": 16.0, "is_bold": True, "page": 1},
            {"text": "Section B", "font_size": 16.0, "is_bold": True, "page": 5},
        ]
        levelled = builder._assign_levels(headings)
        paged    = builder._assign_page_ranges(levelled)
        assert paged[0]["start_page"] == 1
        assert paged[0]["end_page"]   == 5


class TestBAStateIntegration:

    def test_33_run_writes_section_tree(self, builder):
        state = BAState(
            session_id        = "t33",
            heading_positions = SAMPLE_HEADINGS,
            raw_text          = SAMPLE_RAW_TEXT,
        )
        state = builder.run(state)
        assert hasattr(state, "section_tree")
        assert isinstance(state.section_tree, dict)

    def test_34_section_tree_has_children(self, builder):
        state = BAState(
            session_id        = "t34",
            heading_positions = SAMPLE_HEADINGS,
            raw_text          = SAMPLE_RAW_TEXT,
        )
        state = builder.run(state)
        assert "children" in state.section_tree

    def test_35_empty_headings_gives_empty_tree(self, builder):
        state = BAState(
            session_id        = "t35",
            heading_positions = [],
            raw_text          = "",
        )
        state = builder.run(state)
        assert state.section_tree["total_sections"] == 0

    def test_36_seed_unchanged(self, builder):
        state = BAState(
            session_id        = "t36",
            heading_positions = SAMPLE_HEADINGS,
            raw_text          = SAMPLE_RAW_TEXT,
        )
        state = builder.run(state)
        assert state.seed == 42

    def test_37_no_rlef_in_section_tree(self, builder):
        state = BAState(
            session_id        = "t37",
            heading_positions = SAMPLE_HEADINGS,
            raw_text          = SAMPLE_RAW_TEXT,
        )
        state = builder.run(state)
        assert "_rlef_" not in str(state.section_tree)

    def test_38_minimal_headings_works(self, builder):
        state = BAState(
            session_id        = "t38",
            heading_positions = MINIMAL_HEADINGS,
            raw_text          = "Business overview content here.",
        )
        state = builder.run(state)
        assert len(state.section_tree["children"]) >= 1


class TestConvenienceWrapper:

    def test_39_returns_state(self):
        state = BAState(
            session_id        = "t39",
            heading_positions = SAMPLE_HEADINGS,
            raw_text          = SAMPLE_RAW_TEXT,
        )
        result = run_section_tree_builder(state)
        assert hasattr(result, "section_tree")
        assert result.seed == 42

    def test_40_wrapper_populates_tree(self):
        state = BAState(
            session_id        = "t40",
            heading_positions = SAMPLE_HEADINGS,
            raw_text          = SAMPLE_RAW_TEXT,
        )
        result = run_section_tree_builder(state)
        assert result.section_tree["total_sections"] > 0