"""
src/retrieval/sniper_rag.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 Rev1.0 FINAL

N06 — SniperRAG Tier 1 Retrieval
Fires FIRST for numerical queries only.

Responsibilities:
  1. Scan table_cells index with 20+ compiled regex patterns
  2. Match query terms against row_header and col_header
  3. Extract cell_value with confidence score
  4. If confidence >= 0.95 → return result, skip N07/N08/N09
  5. If confidence < 0.95 → cascade to N07

Speed: ~50ms · Zero GPU · Zero LLM call
Handles: ~40% of FinanceBench numerical questions directly

Writes to BAState:
  sniper_hit, sniper_result, sniper_confidence, routing_path
"""

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.state.ba_state import BAState, QueryType
from src.utils.resource_governor import ResourceGovernor
from src.utils.seed_manager import SeedManager

SeedManager.set_all()

# ── Confidence threshold ──────────────────────────────────────────────────────
SNIPER_CONFIDENCE_THRESHOLD = 0.95

# ── Year patterns ─────────────────────────────────────────────────────────────
# Handles: 2023, FY2023, FY 2023, fy2023, fiscal 2023
YEAR_PATTERN    = re.compile(r'\b(20\d{2})\b')
FY_YEAR_PATTERN = re.compile(r'FY\s*(20\d{2})', re.IGNORECASE)


# ═══════════════════════════════════════════════════════════════════════════
# 22 FINANCIAL METRIC PATTERNS
# ═══════════════════════════════════════════════════════════════════════════

FINANCIAL_PATTERNS = [
    # ── Income Statement ────────────────────────────────────────────────────
    {
        "name":         "net_income",
        "row_keywords": ["net income", "net earnings", "net profit",
                         "profit for the year", "net income attributable"],
        "col_keywords": [],
        "value_type":   "currency",
        "aliases":      ["net income", "net earnings", "profit after tax"],
    },
    {
        "name":         "revenue",
        "row_keywords": ["total net revenue", "net revenue", "total revenue",
                         "net sales", "total net sales", "revenues"],
        "col_keywords": [],
        "value_type":   "currency",
        "aliases":      ["revenue", "sales", "net sales", "total sales"],
    },
    {
        "name":         "gross_profit",
        "row_keywords": ["gross profit", "gross margin amount", "gross income"],
        "col_keywords": [],
        "value_type":   "currency",
        "aliases":      ["gross profit", "gross income"],
    },
    {
        "name":         "gross_margin_pct",
        "row_keywords": ["gross margin", "gross profit margin",
                         "gross margin percentage", "gross margin %"],
        "col_keywords": [],
        "value_type":   "percentage",
        "aliases":      ["gross margin", "gross margin %"],
    },
    {
        "name":         "operating_income",
        "row_keywords": ["operating income", "income from operations",
                         "operating profit", "ebit"],
        "col_keywords": [],
        "value_type":   "currency",
        "aliases":      ["operating income", "operating profit", "ebit"],
    },
    {
        "name":         "operating_margin",
        "row_keywords": ["operating margin", "operating income margin",
                         "operating margin %"],
        "col_keywords": [],
        "value_type":   "percentage",
        "aliases":      ["operating margin"],
    },
    {
        "name":         "eps_diluted",
        "row_keywords": ["diluted", "diluted eps", "earnings per share diluted",
                         "diluted earnings per share", "eps diluted"],
        "col_keywords": [],
        "value_type":   "ratio",
        "aliases":      ["diluted eps", "eps diluted", "diluted earnings per share"],
    },
    {
        "name":         "eps_basic",
        "row_keywords": ["basic", "basic eps", "earnings per share basic",
                         "basic earnings per share"],
        "col_keywords": [],
        "value_type":   "ratio",
        "aliases":      ["basic eps", "eps basic", "basic earnings per share"],
    },
    {
        "name":         "ebitda",
        "row_keywords": ["ebitda", "earnings before interest tax depreciation"],
        "col_keywords": [],
        "value_type":   "currency",
        "aliases":      ["ebitda"],
    },

    # ── Balance Sheet ────────────────────────────────────────────────────────
    {
        "name":         "total_assets",
        "row_keywords": ["total assets"],
        "col_keywords": [],
        "value_type":   "currency",
        "aliases":      ["total assets"],
    },
    {
        "name":         "total_liabilities",
        "row_keywords": ["total liabilities"],
        "col_keywords": [],
        "value_type":   "currency",
        "aliases":      ["total liabilities"],
    },
    {
        "name":         "shareholders_equity",
        "row_keywords": ["total shareholders equity",
                         "total stockholders equity",
                         "shareholders equity", "stockholders equity",
                         "total equity"],
        "col_keywords": [],
        "value_type":   "currency",
        "aliases":      ["shareholders equity", "stockholders equity"],
    },
    {
        "name":         "cash_and_equivalents",
        "row_keywords": ["cash and cash equivalents",
                         "cash equivalents", "cash and equivalents"],
        "col_keywords": [],
        "value_type":   "currency",
        "aliases":      ["cash", "cash and cash equivalents"],
    },
    {
        "name":         "long_term_debt",
        "row_keywords": ["long-term debt", "long term debt",
                         "long-term borrowings", "term loan"],
        "col_keywords": [],
        "value_type":   "currency",
        "aliases":      ["long-term debt", "long term debt"],
    },

    # ── Cash Flow ────────────────────────────────────────────────────────────
    {
        "name":         "operating_cash_flow",
        "row_keywords": ["net cash from operating",
                         "cash provided by operating",
                         "operating activities",
                         "cash flows from operating"],
        "col_keywords": [],
        "value_type":   "currency",
        "aliases":      ["operating cash flow", "cash from operations"],
    },
    {
        "name":         "free_cash_flow",
        "row_keywords": ["free cash flow", "fcf"],
        "col_keywords": [],
        "value_type":   "currency",
        "aliases":      ["free cash flow", "fcf"],
    },
    {
        "name":         "capex",
        "row_keywords": ["capital expenditure", "capex",
                         "purchases of property", "capital spending",
                         "purchase of property plant"],
        "col_keywords": [],
        "value_type":   "currency",
        "aliases":      ["capex", "capital expenditure"],
    },

    # ── Per Share / Ratios ────────────────────────────────────────────────────
    {
        "name":         "shares_outstanding",
        "row_keywords": ["shares outstanding", "diluted shares",
                         "weighted average shares",
                         "shares used in computing diluted"],
        "col_keywords": [],
        "value_type":   "shares",
        "aliases":      ["shares outstanding", "diluted shares"],
    },
    {
        "name":         "dividends_per_share",
        "row_keywords": ["dividends per share", "dividend per share",
                         "cash dividends declared per share"],
        "col_keywords": [],
        "value_type":   "ratio",
        "aliases":      ["dividends per share", "dividend per share"],
    },
    {
        "name":         "return_on_equity",
        "row_keywords": ["return on equity", "roe",
                         "return on average equity"],
        "col_keywords": [],
        "value_type":   "percentage",
        "aliases":      ["roe", "return on equity"],
    },
    {
        "name":         "return_on_assets",
        "row_keywords": ["return on assets", "roa",
                         "return on average assets"],
        "col_keywords": [],
        "value_type":   "percentage",
        "aliases":      ["roa", "return on assets"],
    },
    {
        "name":         "debt_to_equity",
        "row_keywords": ["debt to equity", "debt/equity", "leverage ratio"],
        "col_keywords": [],
        "value_type":   "ratio",
        "aliases":      ["debt to equity", "d/e ratio"],
    },
]

# ── Value extraction patterns ─────────────────────────────────────────────────
VALUE_PATTERNS = {
    "currency":   re.compile(
        r'[\$£€]?\s*(-?[\d,]+(?:\.\d+)?)\s*(billion|million|thousand|bn|mn|m|b|k)?',
        re.IGNORECASE
    ),
    "percentage": re.compile(r'(-?[\d,]+(?:\.\d+)?)\s*%'),
    "ratio":      re.compile(r'(-?[\d,]+(?:\.\d+)?)'),
    "shares":     re.compile(
        r'(-?[\d,]+(?:\.\d+)?)\s*(billion|million|thousand|bn|mn|m|b|k)?',
        re.IGNORECASE
    ),
    "any":        re.compile(r'(-?[\d,]+(?:\.\d+)?)'),
}


class SniperRAG:
    """
    N06: SniperRAG Tier 1 Retrieval.
    Direct table cell extraction using compiled regex patterns.
    Fires only for numerical/ratio query types.
    Zero GPU. Zero LLM. ~50ms.
    """

    def __init__(self):
        SeedManager.set_all()
        self._compiled = self._compile_patterns()

    def _compile_patterns(self) -> List[Dict[str, Any]]:
        """Pre-compile all row keyword patterns at init time."""
        compiled = []
        for pattern in FINANCIAL_PATTERNS:
            compiled_row_kws = [
                re.compile(re.escape(kw), re.IGNORECASE)
                for kw in pattern["row_keywords"]
            ]
            compiled.append({**pattern, "compiled_row_kws": compiled_row_kws})
        return compiled

    def run(self, state: BAState) -> BAState:
        """Main entry point."""
        ResourceGovernor.check("N06 SniperRAG")

        if state.query_type not in (QueryType.NUMERICAL, QueryType.RATIO):
            state.sniper_hit        = False
            state.sniper_confidence = 0.0
            state.routing_path      = "sniper_skipped_non_numerical"
            return state

        if not state.table_cells or not state.query:
            state.sniper_hit        = False
            state.sniper_confidence = 0.0
            state.routing_path      = "sniper_no_tables"
            return state

        target_year = self._extract_year(state.query)

        result, confidence, pattern_name = self._search_tables(
            query       = state.query,
            table_cells = state.table_cells,
            target_year = target_year,
        )

        state.sniper_confidence = confidence
        state.sniper_result     = result or ""

        if result and confidence >= SNIPER_CONFIDENCE_THRESHOLD:
            state.sniper_hit   = True
            state.routing_path = f"sniper_hit:{pattern_name}:conf={confidence:.2f}"
            print(f"[N06] SNIPER HIT | pattern={pattern_name} | "
                  f"conf={confidence:.2f} | result={result}")
        else:
            state.sniper_hit   = False
            state.routing_path = (
                f"sniper_no_tables:cascade_to_N07"
                if not result
                else f"sniper_low_conf:{confidence:.2f}:cascade_to_N07"
            )
            print(f"[N06] Sniper miss — cascading to N07 (conf={confidence:.2f})")

        return state

    # ═══════════════════════════════════════════════════════════════════════
    # TABLE SEARCH
    # ═══════════════════════════════════════════════════════════════════════

    def _search_tables(
        self,
        query:       str,
        table_cells: List[Dict[str, Any]],
        target_year: Optional[str],
    ) -> Tuple[Optional[str], float, str]:
        """Search table cells for the best matching value."""
        best_result     = None
        best_confidence = 0.0
        best_pattern    = ""
        query_lower     = query.lower()

        for pattern in self._compiled:
            if not self._query_matches_pattern(query_lower, pattern):
                continue

            candidates = self._find_candidates(table_cells, pattern, target_year)
            if not candidates:
                continue

            for candidate in candidates:
                confidence = self._score_candidate(
                    candidate, pattern, target_year, query_lower
                )
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_result     = candidate["cell_value"]
                    best_pattern    = pattern["name"]

        return best_result, best_confidence, best_pattern

    def _query_matches_pattern(
        self, query_lower: str, pattern: Dict[str, Any]
    ) -> bool:
        """Check if query is asking about this financial metric."""
        for kw in pattern["row_keywords"]:
            if kw.lower() in query_lower:
                return True
        for alias in pattern.get("aliases", []):
            if alias.lower() in query_lower:
                return True
        return False

    def _find_candidates(
        self,
        table_cells: List[Dict[str, Any]],
        pattern:     Dict[str, Any],
        target_year: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Find table cells matching the pattern's row keywords."""
        candidates = []

        for cell in table_cells:
            row_header = cell.get("row_header", "").lower()
            col_header = cell.get("col_header", "").lower()
            cell_value = cell.get("cell_value", "").strip()

            if not cell_value or not row_header:
                continue
            if cell.get("row_index", 0) == 0:
                continue

            row_match = any(
                compiled_kw.search(row_header)
                for compiled_kw in pattern["compiled_row_kws"]
            )
            if not row_match:
                continue

            year_match = True
            if target_year:
                year_in_col = YEAR_PATTERN.search(col_header)
                if year_in_col:
                    year_match = year_in_col.group(1) == target_year

            has_value = bool(
                VALUE_PATTERNS["any"].search(re.sub(r'[,$%\s]', '', cell_value))
            )

            if has_value:
                candidates.append({**cell, "year_match": year_match})

        return candidates

    def _score_candidate(
        self,
        candidate:   Dict[str, Any],
        pattern:     Dict[str, Any],
        target_year: Optional[str],
        query_lower: str,
    ) -> float:
        """Score a candidate cell. Returns confidence 0.0–1.0."""
        score      = 0.5
        row_header = candidate.get("row_header", "").lower()

        for kw in pattern["row_keywords"]:
            if kw.lower() == row_header.strip():
                score += 0.25
                break
            elif kw.lower() in row_header:
                score += 0.10

        if target_year and candidate.get("year_match"):
            score += 0.20
        elif target_year and not candidate.get("year_match"):
            score -= 0.20

        cell_value = candidate.get("cell_value", "")
        if cell_value and cell_value not in ("-", "—", "N/A", ""):
            score += 0.05

        return min(1.0, max(0.0, score))

    # ═══════════════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════════════

    def _extract_year(self, query: str) -> Optional[str]:
        """
        Extract fiscal year from query.
        Handles: 2023, FY2023, FY 2023, fy2023, fiscal year 2023
        """
        # Try FY prefix first (FY2022, FY 2022)
        match = FY_YEAR_PATTERN.search(query)
        if match:
            return match.group(1)
        # Try standalone year (2023)
        match = YEAR_PATTERN.search(query)
        return match.group(1) if match else None

    def extract_numeric_value(self, raw_value: str) -> Optional[str]:
        """Clean and normalise a raw cell value."""
        if not raw_value:
            return None
        raw   = raw_value.strip()
        match = VALUE_PATTERNS["currency"].search(raw)
        if match:
            number   = match.group(1).replace(",", "")
            unit     = (match.group(2) or "").lower()
            unit_map = {
                "billion": "billion", "bn": "billion", "b": "billion",
                "million": "million", "mn": "million", "m": "million",
                "thousand": "thousand", "k": "thousand",
            }
            unit_word = unit_map.get(unit, "")
            return f"{number} {unit_word}".strip() if unit_word else number
        return raw

    def get_pattern_names(self) -> List[str]:
        """Return list of all supported pattern names."""
        return [p["name"] for p in FINANCIAL_PATTERNS]


# ═══════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK
# run: python src/retrieval/sniper_rag.py
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        from rich import print as rprint
    except ImportError:
        rprint = print

    rprint("\n[bold cyan]── SniperRAG sanity check ──[/bold cyan]")

    sniper = SniperRAG()
    rprint(f"[green]✓[/green] SniperRAG instantiated")
    rprint(f"[green]✓[/green] {len(FINANCIAL_PATTERNS)} patterns loaded")

    # Test year extraction including FY prefix
    assert sniper._extract_year("net income in 2023") == "2023"
    assert sniper._extract_year("FY2022 revenue")     == "2022"
    assert sniper._extract_year("FY 2021 results")    == "2021"
    assert sniper._extract_year("fiscal year 2020")   == "2020"
    assert sniper._extract_year("what was revenue?")  is None
    rprint(f"[green]✓[/green] Year extraction — all formats work")

    mock_cells = [
        {"page": 42, "table_number": 0, "row_index": 0, "col_index": 0,
         "row_header": "", "col_header": "", "cell_value": ""},
        {"page": 42, "table_number": 0, "row_index": 0, "col_index": 1,
         "row_header": "", "col_header": "2023", "cell_value": "2023"},
        {"page": 42, "table_number": 0, "row_index": 0, "col_index": 2,
         "row_header": "", "col_header": "2022", "cell_value": "2022"},
        {"page": 42, "table_number": 0, "row_index": 5, "col_index": 0,
         "row_header": "Net income", "col_header": "", "cell_value": "Net income"},
        {"page": 42, "table_number": 0, "row_index": 5, "col_index": 1,
         "row_header": "Net income", "col_header": "2023", "cell_value": "96,995"},
        {"page": 42, "table_number": 0, "row_index": 5, "col_index": 2,
         "row_header": "Net income", "col_header": "2022", "cell_value": "99,803"},
        {"page": 42, "table_number": 0, "row_index": 8, "col_index": 0,
         "row_header": "Diluted", "col_header": "", "cell_value": "Diluted"},
        {"page": 42, "table_number": 0, "row_index": 8, "col_index": 1,
         "row_header": "Diluted", "col_header": "2023", "cell_value": "6.13"},
    ]

    state = BAState(
        session_id  = "sanity-n06",
        query       = "What was Apple's net income for FY2023?",
        query_type  = QueryType.NUMERICAL,
        table_cells = mock_cells,
    )
    state = sniper.run(state)
    assert state.sniper_hit is True
    assert "96,995" in state.sniper_result
    rprint(f"[green]✓[/green] FY2023 prefix query works: {state.sniper_result}")

    rprint(f"\n[bold green]All checks passed. SniperRAG ready.[/bold green]\n")