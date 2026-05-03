"""
N06 SniperRAG — Tier 1 Direct Table Cell Extraction
PDR-BAAAI-001 · Rev 1.0 · Node N06

CHANGELOG:
  2026-05-03 S17  Bug A3: index iXBRL row_headers (us-gaap:NetIncomeLoss)
                  by humanized name + alias map so existing patterns hit.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TableCell:
    row_header:    str
    col_header:    str
    value:         str
    unit:          str
    page:          int
    section:       str
    company:       str
    doc_type:      str
    fiscal_year:   str
    numeric_value: Optional[float] = None

    @property
    def metadata_key(self) -> str:
        """C8: COMPANY/DOCTYPE/FISCAL_YEAR/SECTION/PAGE"""
        return (
            f"{self.company}/{self.doc_type}/{self.fiscal_year}"
            f"/{self.section}/{self.page}"
        )


@dataclass
class SniperResult:
    sniper_hit:      bool
    answer:          str
    value:           str
    unit:            str
    confidence:      float
    matched_pattern: str
    cell:            Optional[TableCell]
    citation:        str
    reason:          str


# ── 20+ compiled regex patterns ───────────────────────────────────────────────

RAW_PATTERNS: Dict[str, str] = {
    "revenue":             r"(?:total\s+)?(?:net\s+)?(?:revenues?|net\s+sales?|total\s+sales?)",
    "net_income":          r"net\s+(?:income|earnings|loss)(?:\s+attributable)?(?:\s+to\s+(?:common\s+)?shareholders?)?",
    "gross_profit":        r"gross\s+(?:profit|margin|income)",
    "operating_income":    r"(?:(?:total\s+)?operating\s+(?:income|loss|profit))|(?:income\s+(?:loss\s+)?from\s+operations)",
    "ebitda":              r"(?:adjusted\s+)?ebitda|earnings\s+before\s+interest[,\s]+(?:taxes[,\s]+)?depreciation",
    "ebit":                r"\bebit\b|earnings\s+before\s+interest\s+and\s+taxes",
    "eps_diluted":         r"(?:diluted\s+)?(?:earnings|loss)\s+per\s+(?:diluted\s+)?(?:common\s+)?share|diluted\s+eps|eps\s+diluted",
    "eps_basic":           r"basic\s+(?:earnings|loss)\s+per\s+(?:common\s+)?share|basic\s+eps",
    "r_and_d":             r"research\s+(?:and|&)\s+development(?:\s+(?:expense|cost))?",
    "sg_and_a":            r"(?:selling[,\s]+)?general\s+(?:and\s+)?administrative(?:\s+(?:expense|cost))?|sg(?:\s*[&and]+\s*)?a",
    "cogs":                r"cost\s+of\s+(?:goods?\s+)?(?:revenue|sales?|products?|services?)|cost\s+of\s+revenues?",
    "interest_expense":    r"interest\s+(?:expense|cost|charges?)|finance\s+(?:cost|charge)",
    "income_tax":          r"(?:provision\s+for\s+)?income\s+tax(?:es)?|tax\s+(?:expense|provision)",
    "total_assets":        r"total\s+assets",
    "total_liabilities":   r"total\s+(?:liabilities(?:\s+and)?|debt)",
    "shareholders_equity": r"(?:total\s+)?(?:stockholders?|shareholders?)\s+(?:equity|deficit)",
    "cash":                r"cash\s+(?:and\s+(?:cash\s+)?equivalents?)?|cash\s+and\s+short.?term\s+investments?",
    "long_term_debt":      r"long.?term\s+(?:debt|notes?\s+payable|borrowings?|obligations?)",
    "goodwill":            r"goodwill(?:\s+impairment)?",
    "deferred_revenue":    r"deferred\s+revenue(?:s)?|unearned\s+revenue",
    "accounts_receivable": r"(?:net\s+)?accounts?\s+receivable|trade\s+receivables?",
    "inventory":           r"(?:total\s+)?inventor(?:y|ies)",
    "current_assets":      r"(?:total\s+)?current\s+assets",
    "current_liabilities": r"(?:total\s+)?current\s+liabilities",
    "operating_cash_flow": r"(?:net\s+)?cash\s+(?:provided\s+by\s+|from\s+)?operating\s+activities|operating\s+cash\s+flow",
    "capex":               r"capital\s+expenditures?|capex|purchases?\s+of\s+property",
    "free_cash_flow":      r"free\s+cash\s+flow|fcf",
    "dividends_paid":      r"dividends?\s+paid|cash\s+dividends?",
    "share_repurchase":    r"(?:repurchases?|buybacks?)\s+(?:of\s+)?(?:common\s+)?(?:stock|shares?)|treasury\s+stock\s+purchases?",
}

COMPILED_PATTERNS: Dict[str, re.Pattern] = {
    name: re.compile(pat, re.IGNORECASE)
    for name, pat in RAW_PATTERNS.items()
}


_FY_PATTERNS = [
    re.compile(r"\bfy\s*(\d{4})\b",                 re.IGNORECASE),
    re.compile(r"\bfiscal\s+year\s+(\d{4})\b",      re.IGNORECASE),
    re.compile(r"\b(?:in|for)\s+(\d{4})\b",         re.IGNORECASE),
    re.compile(r"\b(\d{4})\b"),
    re.compile(r"\bfy\s*'?(\d{2,4})\b",             re.IGNORECASE),
]

_UNIT_PATTERNS: Dict[str, re.Pattern] = {
    "billions":  re.compile(r"\bbillion[s]?\b|\bbn\b", re.IGNORECASE),
    "millions":  re.compile(r"\bmillion[s]?\b|\bmn\b|\bm\b(?!\w)", re.IGNORECASE),
    "thousands": re.compile(r"\bthousand[s]?\b|\bk\b(?!\w)", re.IGNORECASE),
    "%":         re.compile(r"\bpercent(?:age)?\b|%"),
}

_CONF_EXACT      = 0.98
_CONF_PREFIX     = 0.92
_CONF_CONTAINS   = 0.85
_CONF_UNIT_BONUS = 0.02
_HIT_THRESHOLD   = 0.95


# ── Helper functions ──────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[\u2019\u2018\u201c\u201d]", "'", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _parse_numeric(value_str: str) -> Optional[float]:
    if not value_str:
        return None
    s = value_str.strip()
    negative = False
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
        negative = True
    s = re.sub(r"[$,\s]", "", s)
    s = s.rstrip("%")
    try:
        val = float(s)
        return -val if negative else val
    except ValueError:
        return None


def _extract_fy_from_query(query: str) -> Optional[str]:
    norm = _normalise(query)
    for pat in _FY_PATTERNS:
        m = pat.search(norm)
        if m:
            year_str = m.group(1) if m.lastindex >= 1 else m.group(0)
            if len(year_str) == 2:
                year_str = f"20{year_str}"
            return f"FY{year_str}"
    return None


def _detect_unit_from_context(text: str) -> str:
    for unit_name, pat in _UNIT_PATTERNS.items():
        if pat.search(text):
            return unit_name
    return "units"


# ── Bug A3 (S17): iXBRL semantic-name to human-readable conversion ──────────

def _humanize_ixbrl(name: str) -> str:
    """Convert 'us-gaap:NetIncomeLoss' -> 'net income loss'.

    Strips namespace prefix (us-gaap:, dei:, etc.) and converts CamelCase
    to lowercase space-separated words.
    """
    if not name:
        return ""
    if ":" in name:
        name = name.split(":", 1)[1]
    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", name)
    return spaced.lower().strip()


_IXBRL_ALIAS_MAP: Dict[str, List[str]] = {
    "net income loss":       ["net income", "net loss", "net earnings"],
    "revenues":              ["revenue", "net sales", "total revenue"],
    "revenue from contract with customer excluding assessed tax":
                             ["revenue", "net sales", "total revenue"],
    "earnings per share diluted": ["diluted eps", "diluted earnings per share"],
    "earnings per share basic":   ["basic eps", "basic earnings per share"],
    "gross profit":          ["gross margin", "gross profit"],
    "operating income loss": ["operating income", "operating loss"],
    "assets":                ["total assets"],
    "liabilities":           ["total liabilities"],
    "stockholders equity":   ["shareholders equity", "total equity"],
    "stockholders equity including portion attributable to noncontrolling interest":
                             ["shareholders equity", "total equity"],
    "cash and cash equivalents at carrying value":
                             ["cash", "cash and cash equivalents"],
    "long term debt":        ["long-term debt", "long term debt"],
    "research and development expense":
                             ["research and development", "r and d"],
    "selling general and administrative expense":
                             ["sg and a", "selling general administrative"],
    "cost of revenue":       ["cost of sales", "cost of goods sold"],
    "cost of goods and services sold":
                             ["cost of sales", "cost of goods sold"],
    "income tax expense benefit":
                             ["income tax", "tax expense"],
    "interest expense":      ["interest expense"],
}


def _ixbrl_aliases(humanized: str) -> List[str]:
    """Return aliases for an iXBRL humanized name."""
    return _IXBRL_ALIAS_MAP.get(humanized, [])


# ── TableIndex ────────────────────────────────────────────────────────────────

class TableIndex:
    def __init__(self) -> None:
        self._cells:   List[TableCell] = []
        self._row_map: Dict[str, List[TableCell]] = {}

    @classmethod
    def from_raw_cells(cls, raw_cells: List[Dict]) -> "TableIndex":
        """Build TableIndex from raw cells.

        Bug A3 fix (S17): iXBRL cells have row_header like
        'us-gaap:NetIncomeLoss' — we ALSO index by a humanized version
        ('net income loss') so SniperRAG patterns can match them.
        """
        idx = cls()
        for raw in raw_cells:
            row_header_raw = raw.get("row_header", "") or ""
            row_header_norm = _normalise(row_header_raw)

            value_raw = raw.get("value", "") or ""
            cell = TableCell(
                row_header=row_header_norm,
                col_header=_normalise(raw.get("col_header", "")),
                value=value_raw.strip() if isinstance(value_raw, str) else str(value_raw),
                unit=raw.get("unit", "units"),
                page=int(raw.get("page", 0)),
                section=raw.get("section", "UNKNOWN"),
                company=raw.get("company", "UNKNOWN"),
                doc_type=raw.get("doc_type", "UNKNOWN"),
                fiscal_year=raw.get("fiscal_year", "UNKNOWN"),
                numeric_value=_parse_numeric(value_raw if isinstance(value_raw, str) else str(value_raw)),
            )
            idx._cells.append(cell)
            idx._row_map.setdefault(cell.row_header, []).append(cell)

            # Bug A3: index by humanized iXBRL name AND aliases
            humanized = _humanize_ixbrl(row_header_raw)
            if humanized and humanized != cell.row_header:
                idx._row_map.setdefault(humanized, []).append(cell)
                for alias in _ixbrl_aliases(humanized):
                    idx._row_map.setdefault(alias, []).append(cell)

        logger.info(
            "TableIndex built: %d cells, %d unique rows",
            len(idx._cells), len(idx._row_map),
        )
        return idx

    def search_by_row(self, normalised_row: str) -> List[TableCell]:
        return self._row_map.get(normalised_row, [])

    def search_prefix(self, prefix: str) -> List[TableCell]:
        return [
            cell for key, cells in self._row_map.items()
            if key.startswith(prefix)
            for cell in cells
        ]

    def search_contains(self, substring: str) -> List[TableCell]:
        return [
            cell for key, cells in self._row_map.items()
            if substring in key
            for cell in cells
        ]

    def __len__(self) -> int:
        return len(self._cells)

    def is_empty(self) -> bool:
        return len(self._cells) == 0


# ── SniperRAG ─────────────────────────────────────────────────────────────────

class SniperRAG:
    """N06 SniperRAG — Tier 1 Direct Table Cell Extraction."""

    def __init__(self, table_index: TableIndex) -> None:
        self.index = table_index

    def run(self, state) -> object:
        """LangGraph N06 node entry point."""
        if self.index.is_empty() and hasattr(state, "table_cells") and state.table_cells:
            self.index = TableIndex.from_raw_cells(state.table_cells)

        query  = getattr(state, "query", "") or ""
        result = self.hit(query)

        state.sniper_hit        = result.sniper_hit
        state.sniper_result     = result.answer if result.sniper_hit else None
        state.sniper_confidence = result.confidence

        logger.info(
            "N06 SniperRAG: hit=%s | confidence=%.3f | pattern=%s",
            result.sniper_hit, result.confidence, result.matched_pattern,
        )
        return state

    def hit(self, query: str) -> SniperResult:
        """Attempt direct table cell extraction for a query."""
        if self.index.is_empty():
            return self._miss("Table index is empty — no cells to search")

        norm_query = _normalise(query)
        matched_metric, matched_pattern_name = self._identify_metric(norm_query)

        if matched_metric is None:
            return self._miss(
                f"No financial metric pattern matched query: '{query[:80]}'"
            )

        fy_hint    = _extract_fy_from_query(query)
        query_unit = _detect_unit_from_context(norm_query)

        # Strategy 1: Exact match
        candidates = self._score_candidates(
            self.index.search_by_row(matched_metric),
            fy_hint, query_unit, base_confidence=_CONF_EXACT,
        )
        # Strategy 2: Prefix match
        if not candidates:
            candidates = self._score_candidates(
                self.index.search_prefix(matched_metric),
                fy_hint, query_unit, base_confidence=_CONF_PREFIX,
            )
        # Strategy 3: Contains match
        if not candidates:
            first_word = matched_metric.split()[0] if matched_metric else ""
            if len(first_word) >= 4:
                candidates = self._score_candidates(
                    self.index.search_contains(first_word),
                    fy_hint, query_unit, base_confidence=_CONF_CONTAINS,
                )

        if not candidates:
            return self._miss(
                f"No table cells matched metric '{matched_metric}'"
            )

        best_cell, best_conf = max(candidates, key=lambda x: x[1])

        if best_conf >= _HIT_THRESHOLD:
            return self._build_hit(best_cell, best_conf, matched_pattern_name)

        return SniperResult(
            sniper_hit=False,
            answer="",
            value=best_cell.value,
            unit=best_cell.unit,
            confidence=best_conf,
            matched_pattern=matched_pattern_name,
            cell=best_cell,
            citation=best_cell.metadata_key,
            reason=(
                f"Best confidence {best_conf:.3f} < "
                f"threshold {_HIT_THRESHOLD} — cascade to BM25"
            ),
        )

    def _identify_metric(
        self, norm_query: str
    ) -> Tuple[Optional[str], Optional[str]]:
        for name, pattern in COMPILED_PATTERNS.items():
            m = pattern.search(norm_query)
            if m:
                matched_text = _normalise(m.group(0)).strip()
                return matched_text, name
        return None, None

    def _score_candidates(
        self,
        cells:           List[TableCell],
        fy_hint:         Optional[str],
        query_unit:      str,
        base_confidence: float,
    ) -> List[Tuple[TableCell, float]]:
        scored: List[Tuple[TableCell, float]] = []
        for cell in cells:
            conf = base_confidence
            if query_unit != "units" and query_unit == cell.unit:
                conf = min(conf + _CONF_UNIT_BONUS, 1.0)
            if fy_hint is not None:
                cell_fy  = _normalise(cell.fiscal_year)
                query_fy = _normalise(fy_hint)
                if cell_fy and query_fy not in cell_fy and cell_fy not in query_fy:
                    conf -= 0.05
            if not cell.value or cell.value in ("—", "-", "N/A", ""):
                conf -= 0.10
            scored.append((cell, round(conf, 4)))
        return scored

    @staticmethod
    def _build_hit(
        cell: TableCell, confidence: float, pattern_name: str
    ) -> SniperResult:
        display_value = cell.value
        if cell.numeric_value is not None and cell.numeric_value < 0:
            if not display_value.startswith("-"):
                display_value = f"-{display_value.strip('()')}"
        unit_str = f" {cell.unit}" if cell.unit and cell.unit != "units" else ""
        answer   = f"{display_value}{unit_str} [{cell.metadata_key}]"
        return SniperResult(
            sniper_hit=True,
            answer=answer,
            value=cell.value,
            unit=cell.unit,
            confidence=confidence,
            matched_pattern=pattern_name,
            cell=cell,
            citation=cell.metadata_key,
            reason=(
                f"Pattern '{pattern_name}' matched with confidence "
                f"{confidence:.3f} >= {_HIT_THRESHOLD}"
            ),
        )

    @staticmethod
    def _miss(reason: str) -> SniperResult:
        logger.debug("SniperRAG MISS: %s", reason)
        return SniperResult(
            sniper_hit=False,
            answer="",
            value="",
            unit="",
            confidence=0.0,
            matched_pattern="",
            cell=None,
            citation="",
            reason=reason,
        )


# ── Convenience wrapper for LangGraph N06 node ───────────────────────────────

def run_sniper(query: str, table_cells: List[Dict]) -> SniperResult:
    """Convenience wrapper used by the LangGraph pipeline node N06."""
    index  = TableIndex.from_raw_cells(table_cells)
    sniper = SniperRAG(index)
    result = sniper.hit(query)
    logger.info(
        "SniperRAG: hit=%s | confidence=%.3f | pattern=%s",
        result.sniper_hit, result.confidence, result.matched_pattern,
    )
    return result