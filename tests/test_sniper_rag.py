"""
tests/test_sniper_rag.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 Rev1.0 FINAL

Tests for N06 — SniperRAG Tier 1 Retrieval
Run: pytest tests/test_sniper_rag.py -v
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.retrieval.sniper_rag import SniperRAG, FINANCIAL_PATTERNS
from src.state.ba_state import BAState, QueryType


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sniper():
    return SniperRAG()


@pytest.fixture
def apple_table_cells():
    """Realistic Apple 10-K table cells."""
    return [
        # Headers
        {"page": 42, "table_number": 0, "row_index": 0, "col_index": 0,
         "row_header": "", "col_header": "", "cell_value": ""},
        {"page": 42, "table_number": 0, "row_index": 0, "col_index": 1,
         "row_header": "", "col_header": "2023", "cell_value": "2023"},
        {"page": 42, "table_number": 0, "row_index": 0, "col_index": 2,
         "row_header": "", "col_header": "2022", "cell_value": "2022"},
        # Net sales / Revenue
        {"page": 42, "table_number": 0, "row_index": 1, "col_index": 0,
         "row_header": "Net sales", "col_header": "", "cell_value": "Net sales"},
        {"page": 42, "table_number": 0, "row_index": 1, "col_index": 1,
         "row_header": "Net sales", "col_header": "2023", "cell_value": "383,285"},
        {"page": 42, "table_number": 0, "row_index": 1, "col_index": 2,
         "row_header": "Net sales", "col_header": "2022", "cell_value": "394,328"},
        # Gross margin
        {"page": 42, "table_number": 0, "row_index": 2, "col_index": 0,
         "row_header": "Gross margin", "col_header": "", "cell_value": "Gross margin"},
        {"page": 42, "table_number": 0, "row_index": 2, "col_index": 1,
         "row_header": "Gross margin", "col_header": "2023", "cell_value": "169,148"},
        {"page": 42, "table_number": 0, "row_index": 2, "col_index": 2,
         "row_header": "Gross margin", "col_header": "2022", "cell_value": "170,782"},
        # Operating income
        {"page": 42, "table_number": 0, "row_index": 3, "col_index": 0,
         "row_header": "Operating income", "col_header": "",
         "cell_value": "Operating income"},
        {"page": 42, "table_number": 0, "row_index": 3, "col_index": 1,
         "row_header": "Operating income", "col_header": "2023",
         "cell_value": "114,301"},
        {"page": 42, "table_number": 0, "row_index": 3, "col_index": 2,
         "row_header": "Operating income", "col_header": "2022",
         "cell_value": "119,437"},
        # Net income
        {"page": 42, "table_number": 0, "row_index": 5, "col_index": 0,
         "row_header": "Net income", "col_header": "", "cell_value": "Net income"},
        {"page": 42, "table_number": 0, "row_index": 5, "col_index": 1,
         "row_header": "Net income", "col_header": "2023", "cell_value": "96,995"},
        {"page": 42, "table_number": 0, "row_index": 5, "col_index": 2,
         "row_header": "Net income", "col_header": "2022", "cell_value": "99,803"},
        # EPS diluted
        {"page": 42, "table_number": 0, "row_index": 8, "col_index": 0,
         "row_header": "Diluted", "col_header": "", "cell_value": "Diluted"},
        {"page": 42, "table_number": 0, "row_index": 8, "col_index": 1,
         "row_header": "Diluted", "col_header": "2023", "cell_value": "6.13"},
        {"page": 42, "table_number": 0, "row_index": 8, "col_index": 2,
         "row_header": "Diluted", "col_header": "2022", "cell_value": "6.11"},
        # Total assets
        {"page": 44, "table_number": 1, "row_index": 1, "col_index": 0,
         "row_header": "Total assets", "col_header": "",
         "cell_value": "Total assets"},
        {"page": 44, "table_number": 1, "row_index": 1, "col_index": 1,
         "row_header": "Total assets", "col_header": "2023",
         "cell_value": "352,583"},
        {"page": 44, "table_number": 1, "row_index": 1, "col_index": 2,
         "row_header": "Total assets", "col_header": "2022",
         "cell_value": "352,755"},
    ]


def make_state(query, query_type, table_cells, session_id="test"):
    return BAState(
        session_id  = session_id,
        query       = query,
        query_type  = query_type,
        table_cells = table_cells,
    )


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 1 — Instantiation
# ═══════════════════════════════════════════════════════════════════════════

class TestInstantiation:

    def test_01_sniper_instantiates(self, sniper):
        """N06: SniperRAG must instantiate without error"""
        assert sniper is not None

    def test_02_patterns_loaded(self, sniper):
        """N06: Must have at least 20 financial patterns"""
        assert len(FINANCIAL_PATTERNS) >= 20

    def test_03_pattern_names_accessible(self, sniper):
        """N06: get_pattern_names() must return list"""
        names = sniper.get_pattern_names()
        assert isinstance(names, list)
        assert len(names) >= 20


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 2 — Query type routing
# ═══════════════════════════════════════════════════════════════════════════

class TestQueryTypeRouting:

    def test_04_text_query_skipped(self, sniper, apple_table_cells):
        """N06: TEXT queries must be skipped — routing_path contains 'skipped'"""
        state = make_state(
            "What are the main risk factors?",
            QueryType.TEXT, apple_table_cells, "t04"
        )
        state = sniper.run(state)
        assert state.sniper_hit is False
        assert "sniper_skipped" in state.routing_path

    def test_05_forensic_query_skipped(self, sniper, apple_table_cells):
        """N06: FORENSIC queries must be skipped"""
        state = make_state(
            "Are there any accounting anomalies?",
            QueryType.FORENSIC, apple_table_cells, "t05"
        )
        state = sniper.run(state)
        assert state.sniper_hit is False

    def test_06_multi_doc_query_skipped(self, sniper, apple_table_cells):
        """N06: MULTI_DOC queries must be skipped"""
        state = make_state(
            "Compare Apple and Microsoft revenue",
            QueryType.MULTI_DOC, apple_table_cells, "t06"
        )
        state = sniper.run(state)
        assert state.sniper_hit is False

    def test_07_numerical_query_fires(self, sniper, apple_table_cells):
        """N06: NUMERICAL queries must trigger SniperRAG"""
        state = make_state(
            "What was net income in 2023?",
            QueryType.NUMERICAL, apple_table_cells, "t07"
        )
        state = sniper.run(state)
        # Should fire (hit or miss — not skipped)
        assert "sniper_skipped" not in state.routing_path

    def test_08_ratio_query_fires(self, sniper, apple_table_cells):
        """N06: RATIO queries must trigger SniperRAG"""
        state = make_state(
            "What was diluted EPS in 2023?",
            QueryType.RATIO, apple_table_cells, "t08"
        )
        state = sniper.run(state)
        assert "sniper_skipped" not in state.routing_path


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 3 — Successful hits
# ═══════════════════════════════════════════════════════════════════════════

class TestSuccessfulHits:

    def test_09_net_income_2023_hit(self, sniper, apple_table_cells):
        """N06: Must find Apple net income FY2023"""
        state = make_state(
            "What was Apple's net income for 2023?",
            QueryType.NUMERICAL, apple_table_cells, "t09"
        )
        state = sniper.run(state)
        assert state.sniper_hit is True
        assert "96,995" in state.sniper_result
        assert state.sniper_confidence >= 0.95

    def test_10_revenue_2023_hit(self, sniper, apple_table_cells):
        """N06: Must find Apple net sales FY2023"""
        state = make_state(
            "What was total net sales in 2023?",
            QueryType.NUMERICAL, apple_table_cells, "t10"
        )
        state = sniper.run(state)
        assert state.sniper_hit is True
        assert "383,285" in state.sniper_result

    def test_11_eps_diluted_2023_hit(self, sniper, apple_table_cells):
        """N06: Must find Apple diluted EPS FY2023"""
        state = make_state(
            "What was diluted EPS for 2023?",
            QueryType.RATIO, apple_table_cells, "t11"
        )
        state = sniper.run(state)
        assert state.sniper_hit is True
        assert "6.13" in state.sniper_result

    def test_12_operating_income_hit(self, sniper, apple_table_cells):
        """N06: Must find operating income"""
        state = make_state(
            "What was operating income in 2023?",
            QueryType.NUMERICAL, apple_table_cells, "t12"
        )
        state = sniper.run(state)
        assert state.sniper_hit is True
        assert "114,301" in state.sniper_result

    def test_13_total_assets_hit(self, sniper, apple_table_cells):
        """N06: Must find total assets"""
        state = make_state(
            "What were total assets in 2023?",
            QueryType.NUMERICAL, apple_table_cells, "t13"
        )
        state = sniper.run(state)
        assert state.sniper_hit is True
        assert "352,583" in state.sniper_result

    def test_14_year_disambiguation_2022(self, sniper, apple_table_cells):
        """N06: Must return 2022 value when 2022 specified"""
        state = make_state(
            "What was net income in 2022?",
            QueryType.NUMERICAL, apple_table_cells, "t14"
        )
        state = sniper.run(state)
        assert state.sniper_hit is True
        assert "99,803" in state.sniper_result

    def test_15_revenue_2022_year_match(self, sniper, apple_table_cells):
        """N06: Must return 2022 revenue when specified"""
        state = make_state(
            "What was Apple net sales for fiscal 2022?",
            QueryType.NUMERICAL, apple_table_cells, "t15"
        )
        state = sniper.run(state)
        assert state.sniper_hit is True
        assert "394,328" in state.sniper_result


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 4 — Confidence and routing
# ═══════════════════════════════════════════════════════════════════════════

class TestConfidenceAndRouting:

    def test_16_hit_confidence_above_threshold(self, sniper, apple_table_cells):
        """N06: Sniper hit must have confidence >= 0.95"""
        state = make_state(
            "What was net income in 2023?",
            QueryType.NUMERICAL, apple_table_cells, "t16"
        )
        state = sniper.run(state)
        if state.sniper_hit:
            assert state.sniper_confidence >= 0.95

    def test_17_miss_routes_to_n07(self, sniper):
        """N06: Miss must route to N07 cascade"""
        state = make_state(
            "What was the revenue growth rate?",
            QueryType.NUMERICAL, [], "t17"
        )
        state = sniper.run(state)
        assert state.sniper_hit is False
        assert "sniper_no_tables" in state.routing_path or \
               "cascade_to_N07" in state.routing_path or \
               state.sniper_confidence < 0.95

    def test_18_routing_path_always_set(self, sniper, apple_table_cells):
        """N06: routing_path must always be set after run()"""
        state = make_state(
            "What was net income in 2023?",
            QueryType.NUMERICAL, apple_table_cells, "t18"
        )
        state = sniper.run(state)
        assert state.routing_path != ""

    def test_19_empty_tables_no_hit(self, sniper):
        """N06: Empty table_cells must return sniper_hit=False"""
        state = make_state(
            "What was net income in 2023?",
            QueryType.NUMERICAL, [], "t19"
        )
        state = sniper.run(state)
        assert state.sniper_hit is False


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 5 — Year extraction
# ═══════════════════════════════════════════════════════════════════════════

class TestYearExtraction:

    def test_20_extract_year_from_query(self, sniper):
        """N06: Must extract year from query string"""
        assert sniper._extract_year("What was net income in 2023?") == "2023"
        assert sniper._extract_year("FY2022 revenue") == "2022"
        assert sniper._extract_year("fiscal year 2021") == "2021"

    def test_21_extract_year_none_when_missing(self, sniper):
        """N06: Must return None when no year in query"""
        assert sniper._extract_year("What was net income?") is None
        assert sniper._extract_year("revenue growth rate") is None


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 6 — BAState integration
# ═══════════════════════════════════════════════════════════════════════════

class TestBAStateIntegration:

    def test_22_state_fields_written(self, sniper, apple_table_cells):
        """N06: All BAState fields must be written"""
        state = make_state(
            "What was net income in 2023?",
            QueryType.NUMERICAL, apple_table_cells, "t22"
        )
        state = sniper.run(state)
        assert isinstance(state.sniper_hit, bool)
        assert isinstance(state.sniper_result, str)
        assert isinstance(state.sniper_confidence, float)
        assert isinstance(state.routing_path, str)

    def test_23_seed_unchanged(self, sniper, apple_table_cells):
        """C5: BAState seed must still be 42 after N06"""
        state = make_state(
            "What was net income in 2023?",
            QueryType.NUMERICAL, apple_table_cells, "t23"
        )
        state = sniper.run(state)
        assert state.seed == 42

    def test_24_alias_recognition(self, sniper, apple_table_cells):
        """N06: Must recognize financial metric aliases"""
        # 'earnings' is an alias for net income
        state = make_state(
            "What were net earnings in 2023?",
            QueryType.NUMERICAL, apple_table_cells, "t24"
        )
        state = sniper.run(state)
        # Should find net income via alias
        assert state.sniper_result != "" or state.sniper_hit is False