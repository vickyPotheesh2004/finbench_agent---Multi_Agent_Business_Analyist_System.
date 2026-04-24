import pytest
from src.retrieval.sniper_rag import (
    TableCell, TableIndex, SniperRAG, SniperResult,
    _normalise, _parse_numeric, _extract_fy_from_query,
    _detect_unit_from_context, run_sniper, COMPILED_PATTERNS,
    _HIT_THRESHOLD, _CONF_EXACT, _CONF_PREFIX,
    _CONF_CONTAINS, _CONF_UNIT_BONUS,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _cell(row, col, value, unit="millions", page=94,
          section="INCOME_STATEMENT", company="APPLE",
          doc_type="10-K", fiscal_year="FY2022"):
    return {
        "row_header": row, "col_header": col, "value": value,
        "unit": unit, "page": page, "section": section,
        "company": company, "doc_type": doc_type, "fiscal_year": fiscal_year,
    }


APPLE_CELLS = [
    _cell("Total net sales",           "FY2022", "394,328", fiscal_year="FY2022"),
    _cell("Total net sales",           "FY2021", "365,817", fiscal_year="FY2021"),
    _cell("Net income",                "FY2022", "99,803",  fiscal_year="FY2022"),
    _cell("Net income",                "FY2021", "94,680",  fiscal_year="FY2021"),
    _cell("Gross profit",              "FY2022", "170,782", fiscal_year="FY2022"),
    _cell("Operating income",          "FY2022", "119,437", fiscal_year="FY2022"),
    _cell("Diluted earnings per share","FY2022", "6.11",    "USD", fiscal_year="FY2022"),
    _cell("Diluted earnings per share","FY2021", "5.61",    "USD", fiscal_year="FY2021"),
    _cell("Research and development",  "FY2022", "26,251",  fiscal_year="FY2022"),
    _cell("Total assets",              "FY2022", "352,755", page=96, section="BALANCE_SHEET", fiscal_year="FY2022"),
    _cell("Cash and cash equivalents", "FY2022", "23,646",  page=96, section="BALANCE_SHEET", fiscal_year="FY2022"),
    _cell("Long-term debt",            "FY2022", "98,959",  page=97, section="BALANCE_SHEET", fiscal_year="FY2022"),
    _cell("Total stockholders equity", "FY2022", "(50,672)", page=97, section="BALANCE_SHEET", fiscal_year="FY2022"),
    _cell("Capital expenditures",      "FY2022", "10,708",  page=98, section="CASH_FLOW",     fiscal_year="FY2022"),
    _cell("Operating cash flow",       "FY2022", "122,151", page=98, section="CASH_FLOW",     fiscal_year="FY2022"),
]


@pytest.fixture
def apple_sniper():
    return SniperRAG(TableIndex.from_raw_cells(APPLE_CELLS))


@pytest.fixture
def empty_sniper():
    return SniperRAG(TableIndex.from_raw_cells([]))


# ── Group 1: Helper functions ─────────────────────────────────────────────────

class TestNormalise:
    def test_lowercase(self):             assert _normalise("Total Net Sales") == "total net sales"
    def test_strips_whitespace(self):     assert _normalise("  net income  ") == "net income"
    def test_collapses_spaces(self):      assert _normalise("gross  profit") == "gross profit"
    def test_empty_string(self):          assert _normalise("") == ""


class TestParseNumeric:
    def test_simple_integer(self):        assert _parse_numeric("394328") == pytest.approx(394328.0)
    def test_comma_formatted(self):       assert _parse_numeric("394,328") == pytest.approx(394328.0)
    def test_decimal(self):               assert _parse_numeric("6.11") == pytest.approx(6.11)
    def test_parenthetical_negative(self):assert _parse_numeric("(50,672)") == pytest.approx(-50672.0)
    def test_dollar_sign(self):           assert _parse_numeric("$394,328") == pytest.approx(394328.0)
    def test_empty_string(self):          assert _parse_numeric("") is None
    def test_non_numeric(self):           assert _parse_numeric("N/A") is None
    def test_percentage(self):            assert _parse_numeric("42.5%") == pytest.approx(42.5)
    def test_negative_with_minus(self):   assert _parse_numeric("-2,000") == pytest.approx(-2000.0)


class TestExtractFY:
    def test_fy_prefix(self):             assert _extract_fy_from_query("revenue FY2022") == "FY2022"
    def test_fiscal_year_words(self):     assert _extract_fy_from_query("fiscal year 2023") == "FY2023"
    def test_bare_year(self):             assert _extract_fy_from_query("net sales in 2021") == "FY2021"
    def test_no_year(self):               assert _extract_fy_from_query("what was net income?") is None
    def test_fy_short(self):              assert _extract_fy_from_query("revenue FY22") == "FY2022"


class TestDetectUnit:
    def test_millions(self):              assert _detect_unit_from_context("revenue in millions") == "millions"
    def test_billions(self):              assert _detect_unit_from_context("assets in billions") == "billions"
    def test_percentage(self):            assert _detect_unit_from_context("gross margin percentage") == "%"
    def test_no_unit(self):               assert _detect_unit_from_context("what was revenue") == "units"


# ── Group 2: TableCell ────────────────────────────────────────────────────────

class TestTableCell:
    def test_metadata_key_format(self):
        """C8: COMPANY/DOCTYPE/FISCAL_YEAR/SECTION/PAGE"""
        cell = TableCell(
            row_header="total net sales", col_header="fy2022",
            value="394,328", unit="millions", page=94,
            section="INCOME_STATEMENT", company="APPLE",
            doc_type="10-K", fiscal_year="FY2022",
        )
        assert cell.metadata_key == "APPLE/10-K/FY2022/INCOME_STATEMENT/94"

    def test_metadata_key_five_parts(self):
        cell = TableCell(
            row_header="net income", col_header="fy2023",
            value="99,803", unit="millions", page=50,
            section="INCOME_STATEMENT", company="MSFT",
            doc_type="10-Q", fiscal_year="FY2023",
        )
        assert len(cell.metadata_key.split("/")) == 5

    def test_parenthetical_negative_parsed(self):
        raw = _cell("Net income", "FY2022", "(2,000)")
        idx = TableIndex.from_raw_cells([raw])
        assert idx._cells[0].numeric_value == pytest.approx(-2000.0)


# ── Group 3: TableIndex ───────────────────────────────────────────────────────

class TestTableIndex:
    def test_count(self):                 assert len(TableIndex.from_raw_cells(APPLE_CELLS)) == len(APPLE_CELLS)
    def test_not_empty(self):             assert not TableIndex.from_raw_cells(APPLE_CELLS).is_empty()
    def test_empty(self):                 assert TableIndex.from_raw_cells([]).is_empty()
    def test_search_by_row(self):         assert len(TableIndex.from_raw_cells(APPLE_CELLS).search_by_row("total net sales")) >= 2
    def test_search_missing(self):        assert TableIndex.from_raw_cells(APPLE_CELLS).search_by_row("nonexistent") == []
    def test_row_keys_lowercase(self):
        idx = TableIndex.from_raw_cells(APPLE_CELLS)
        for key in idx._row_map:
            assert key == key.lower()


# ── Group 4: Pattern coverage ─────────────────────────────────────────────────

class TestPatternCoverage:
    @pytest.mark.parametrize("name,query", [
        ("revenue",             "total net sales FY2022"),
        ("revenue",             "total revenues for the year"),
        ("net_income",          "net income attributable to shareholders"),
        ("gross_profit",        "gross profit FY2022"),
        ("operating_income",    "operating income from operations"),
        ("ebitda",              "adjusted EBITDA"),
        ("eps_diluted",         "diluted earnings per share"),
        ("eps_diluted",         "diluted EPS"),
        ("eps_basic",           "basic earnings per share"),
        ("r_and_d",             "research and development expense"),
        ("sg_and_a",            "selling general administrative expense"),
        ("cogs",                "cost of revenue"),
        ("interest_expense",    "interest expense"),
        ("income_tax",          "provision for income taxes"),
        ("total_assets",        "total assets"),
        ("total_liabilities",   "total liabilities"),
        ("shareholders_equity", "total stockholders equity"),
        ("cash",                "cash and cash equivalents"),
        ("long_term_debt",      "long-term debt"),
        ("goodwill",            "goodwill impairment"),
        ("deferred_revenue",    "deferred revenue"),
        ("accounts_receivable", "net accounts receivable"),
        ("inventory",           "total inventories"),
        ("current_assets",      "total current assets"),
        ("current_liabilities", "total current liabilities"),
        ("operating_cash_flow", "net cash provided by operating activities"),
        ("capex",               "capital expenditures"),
        ("free_cash_flow",      "free cash flow"),
        ("dividends_paid",      "dividends paid"),
        ("share_repurchase",    "repurchases of common stock"),
    ])
    def test_pattern_matches(self, name, query):
        assert COMPILED_PATTERNS[name].search(query.lower()), (
            f"Pattern '{name}' did not match: '{query}'"
        )

    def test_minimum_20_patterns(self):
        assert len(COMPILED_PATTERNS) >= 20


# ── Group 5: SniperRAG hit/miss ───────────────────────────────────────────────

class TestSniperHit:
    def test_revenue_hit(self, apple_sniper):
        r = apple_sniper.hit("What was Apple total net sales FY2022?")
        assert r.sniper_hit is True
        assert "394" in r.value

    def test_net_income_hit(self, apple_sniper):
        r = apple_sniper.hit("What was net income FY2022?")
        assert r.sniper_hit is True

    def test_eps_diluted_hit(self, apple_sniper):
        r = apple_sniper.hit("What was diluted earnings per share FY2022?")
        assert r.sniper_hit is True

    def test_gross_profit_hit(self, apple_sniper):
        r = apple_sniper.hit("What was gross profit FY2022?")
        assert r.sniper_hit is True

    def test_total_assets_hit(self, apple_sniper):
        r = apple_sniper.hit("total assets FY2022?")
        assert r.sniper_hit is True
        assert "352" in r.value

    def test_capex_hit(self, apple_sniper):
        r = apple_sniper.hit("What were capital expenditures FY2022?")
        assert r.sniper_hit is True

    def test_empty_index_miss(self, empty_sniper):
        r = empty_sniper.hit("What was net income?")
        assert r.sniper_hit is False
        assert r.confidence == 0.0

    def test_narrative_query_miss(self, apple_sniper):
        r = apple_sniper.hit("What is the company's strategic vision for ESG?")
        assert r.sniper_hit is False

    def test_hit_has_five_part_citation(self, apple_sniper):
        r = apple_sniper.hit("What was total net sales FY2022?")
        if r.sniper_hit:
            assert len(r.citation.split("/")) == 5

    def test_answer_contains_value(self, apple_sniper):
        r = apple_sniper.hit("What was total net sales FY2022?")
        if r.sniper_hit:
            assert r.value in r.answer

    def test_reason_populated(self, apple_sniper):
        r = apple_sniper.hit("What was total net sales FY2022?")
        assert r.reason != ""

    def test_confidence_never_exceeds_1(self, apple_sniper):
        for q in ["total net sales in millions FY2022",
                  "diluted earnings per share FY2022",
                  "total assets FY2022"]:
            assert apple_sniper.hit(q).confidence <= 1.0


# ── Group 6: Fiscal year filtering ───────────────────────────────────────────

class TestFiscalYearFiltering:
    def test_fy2022_cell_wins_for_fy2022_query(self, apple_sniper):
        r = apple_sniper.hit("What was net income FY2022?")
        if r.sniper_hit and r.cell:
            assert "2022" in r.cell.fiscal_year

    def test_fy2021_cell_wins_for_fy2021_query(self, apple_sniper):
        r = apple_sniper.hit("What was total net sales in fiscal year 2021?")
        if r.sniper_hit and r.cell:
            assert "2021" in r.cell.fiscal_year


# ── Group 7: Parenthetical negatives ─────────────────────────────────────────

class TestParentheticalNegatives:
    def test_negative_numeric_value(self):
        raw = _cell("Net income", "FY2022", "(2,000)")
        idx = TableIndex.from_raw_cells([raw])
        assert idx._cells[0].numeric_value < 0

    def test_negative_display_in_answer(self):
        raw = _cell("Net income", "FY2022", "(2,000)")
        r   = SniperRAG(TableIndex.from_raw_cells([raw])).hit("net income FY2022")
        if r.sniper_hit:
            assert "-" in r.answer or "(" in r.answer


# ── Group 8: Confidence thresholds ───────────────────────────────────────────

class TestConfidenceThreshold:
    def test_hit_threshold_is_095(self):  assert _HIT_THRESHOLD   == 0.95
    def test_exact_conf_is_098(self):     assert _CONF_EXACT       == 0.98
    def test_prefix_conf_is_092(self):    assert _CONF_PREFIX      == 0.92
    def test_contains_conf_is_085(self):  assert _CONF_CONTAINS    == 0.85
    def test_unit_bonus_is_002(self):     assert _CONF_UNIT_BONUS  == 0.02

    def test_no_hit_below_threshold(self, apple_sniper):
        for q in ["total net sales FY2022", "net income FY2022"]:
            r = apple_sniper.hit(q)
            if r.sniper_hit:
                assert r.confidence >= _HIT_THRESHOLD


# ── Group 9: run_sniper wrapper ───────────────────────────────────────────────

class TestRunSniperWrapper:
    def test_returns_sniper_result(self):
        assert isinstance(run_sniper("total net sales FY2022", APPLE_CELLS), SniperResult)

    def test_hit_on_revenue(self):
        assert run_sniper("What was total net sales FY2022?", APPLE_CELLS).sniper_hit is True

    def test_miss_on_empty(self):
        assert run_sniper("total net sales FY2022", []).sniper_hit is False

    def test_miss_on_narrative(self):
        assert run_sniper("What is the company's competitive advantage?", APPLE_CELLS).sniper_hit is False

    def test_c8_citation_on_hit(self):
        r = run_sniper("total net sales FY2022", APPLE_CELLS)
        if r.sniper_hit:
            parts = r.citation.split("/")
            assert len(parts) == 5
            assert parts[0] == "APPLE"
            assert parts[1] == "10-K"
            assert "FY" in parts[2]
            assert parts[4].isdigit()


# ── Group 10: Gate M2 hit rate ────────────────────────────────────────────────

class TestGateM2HitRate:
    NUMERICAL = [
        "What was total net sales FY2022?",
        "What was net income FY2022?",
        "What was diluted earnings per share FY2022?",
        "What was gross profit FY2022?",
        "What was operating income FY2022?",
        "What was total assets FY2022?",
        "What was cash and cash equivalents FY2022?",
        "What was long-term debt FY2022?",
        "What was research and development FY2022?",
        "What were capital expenditures FY2022?",
        "What was total net sales FY2021?",
        "What was net income FY2021?",
        "What was diluted earnings per share FY2021?",
        "What was operating cash flow FY2022?",
    ]
    NON_NUMERICAL = [
        "What is the company's business strategy?",
        "How does Apple describe its competitive advantage?",
        "What are the main risk factors?",
        "Describe the company's ESG initiatives",
        "What did management say about supply chain?",
        "What is the company's geographic market exposure?",
    ]

    def test_hit_rate_above_55_percent(self, apple_sniper):
        hits     = sum(1 for q in self.NUMERICAL if apple_sniper.hit(q).sniper_hit)
        hit_rate = hits / len(self.NUMERICAL)
        assert hit_rate >= 0.55, (
            f"Hit rate {hit_rate:.1%} below 55% Gate M2 — "
            f"{hits}/{len(self.NUMERICAL)} hits"
        )

    def test_false_positive_rate_below_3_percent(self, apple_sniper):
        fps     = sum(1 for q in self.NON_NUMERICAL if apple_sniper.hit(q).sniper_hit)
        fp_rate = fps / len(self.NON_NUMERICAL)
        assert fp_rate < 0.03, (
            f"False positive rate {fp_rate:.1%} exceeds 3% — "
            f"{fps}/{len(self.NON_NUMERICAL)}"
        )

    def test_all_hits_above_threshold(self, apple_sniper):
        for q in self.NUMERICAL:
            r = apple_sniper.hit(q)
            if r.sniper_hit:
                assert r.confidence >= _HIT_THRESHOLD


# ── Group 11: C8 constraint ───────────────────────────────────────────────────

class TestC8Constraint:
    def test_all_hits_have_5_part_citation(self):
        sniper = SniperRAG(TableIndex.from_raw_cells(APPLE_CELLS))
        for q in [
            "total net sales FY2022",
            "net income FY2022",
            "gross profit FY2022",
            "total assets FY2022",
            "capital expenditures FY2022",
        ]:
            r = sniper.hit(q)
            if r.sniper_hit:
                parts = r.citation.split("/")
                assert len(parts) == 5, f"Bad citation '{r.citation}' for '{q}'"
                assert parts[4].isdigit(), f"Page not numeric in '{r.citation}'"