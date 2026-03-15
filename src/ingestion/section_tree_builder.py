"""
src/ingestion/section_tree_builder.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 Rev1.0 FINAL

N02 — Section Tree Builder
Runs ONCE per document, after N01.

Responsibilities:
  1. Take heading_positions from BAState (written by N01)
  2. Build a hierarchical JSON section tree
  3. Assign page ranges to every section
  4. Identify the 5 key financial sections:
       Business Overview, Risk Factors, MD&A,
       Financial Statements, Notes to Financial Statements
  5. Write section_tree to BAState

The section tree is used by N03 to split document at
section boundaries — never at arbitrary word counts.
"""

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.state.ba_state import BAState
from src.utils.resource_governor import ResourceGovernor
from src.utils.seed_manager import SeedManager

SeedManager.set_all()

# ── Key financial section names to detect ────────────────────────────────────
KEY_SECTIONS = [
    ("business_overview",       ["business overview", "item 1", "our business"]),
    ("risk_factors",            ["risk factors", "item 1a"]),
    ("mda",                     ["management", "discussion", "analysis", "item 7", "md&a"]),
    ("financial_statements",    ["financial statements", "consolidated balance",
                                 "consolidated statements", "item 8"]),
    ("notes",                   ["notes to", "note 1", "note 2", "accounting policies"]),
]


class SectionTreeBuilder:
    """
    N02: Section Tree Builder.
    Builds a hierarchical section map from heading_positions.
    Writes section_tree to BAState.
    """

    def __init__(self):
        SeedManager.set_all()

    def run(self, state: BAState) -> BAState:
        """
        Main entry point.
        Reads state.heading_positions and state.raw_text.
        Writes state.section_tree.
        """
        ResourceGovernor.check("N02 Section Tree Builder start")

        if not state.heading_positions and not state.raw_text:
            print("[N02] No headings or raw text — returning empty section tree")
            state.section_tree = self._empty_tree()
            return state

        # Build tree from headings if available
        if state.heading_positions:
            tree = self._build_from_headings(state.heading_positions, state.raw_text)
        else:
            # Fallback: detect sections from raw text patterns
            tree = self._build_from_text(state.raw_text)

        # Tag key financial sections
        tree = self._tag_key_sections(tree)

        state.section_tree = tree

        print(f"[N02] Built section tree: {len(tree['sections'])} top-level sections")
        print(f"[N02] Key sections found: {[k for k,v in tree['key_sections'].items() if v]}")

        ResourceGovernor.check("N02 Section Tree Builder complete")
        return state

    # ═══════════════════════════════════════════════════════════════════════
    # BUILD FROM HEADINGS
    # ═══════════════════════════════════════════════════════════════════════

    def _build_from_headings(
        self,
        headings: List[Dict[str, Any]],
        raw_text: str
    ) -> Dict[str, Any]:
        """
        Build section tree from heading_positions detected by N01.
        Groups headings by font size to determine hierarchy level.
        """
        if not headings:
            return self._empty_tree()

        # Determine font size tiers for hierarchy
        sizes = sorted(set(h["font_size"] for h in headings), reverse=True)
        size_to_level = {}
        for i, size in enumerate(sizes[:4]):  # max 4 levels
            size_to_level[size] = i + 1

        # Build flat list of section nodes
        sections = []
        for i, heading in enumerate(headings):
            level = size_to_level.get(heading["font_size"], 4)
            text  = heading["text"].strip()

            if not text or len(text) < 2:
                continue

            # Page range: from this heading to next heading of same/higher level
            start_page = heading["page"]
            end_page   = start_page

            for next_h in headings[i + 1:]:
                next_level = size_to_level.get(next_h["font_size"], 4)
                if next_level <= level:
                    end_page = max(start_page, next_h["page"] - 1)
                    break
                end_page = next_h["page"]

            sections.append({
                "id":         f"sec_{i:04d}",
                "title":      text,
                "level":      level,
                "page_start": start_page,
                "page_end":   end_page,
                "is_bold":    heading.get("is_bold", False),
                "font_size":  heading["font_size"],
                "summary":    "",
                "children":   [],
                "key_section": None,
            })

        # Build hierarchy
        tree_sections = self._nest_sections(sections)

        return {
            "total_sections": len(sections),
            "sections":       tree_sections,
            "key_sections":   {},
            "page_count":     max((h["page"] for h in headings), default=0),
        }

    def _nest_sections(
        self, flat: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert flat section list into nested hierarchy.
        Level 1 = top level, level 2 = children of level 1, etc.
        """
        result  = []
        stack   = []  # (level, section)

        for section in flat:
            level = section["level"]

            # Pop stack until parent level found
            while stack and stack[-1][0] >= level:
                stack.pop()

            if not stack:
                result.append(section)
            else:
                stack[-1][1]["children"].append(section)

            stack.append((level, section))

        return result

    # ═══════════════════════════════════════════════════════════════════════
    # BUILD FROM RAW TEXT (fallback)
    # ═══════════════════════════════════════════════════════════════════════

    def _build_from_text(self, raw_text: str) -> Dict[str, Any]:
        """
        Fallback: detect sections from raw text using regex patterns.
        Used when heading_positions is empty (e.g. CSV/XLSX files).
        """
        sections  = []
        lines     = raw_text.split("\n")
        sec_id    = 0

        # Pattern: ALL CAPS lines or lines starting with "ITEM"
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or len(line) < 3:
                continue

            is_heading = (
                line.isupper() and len(line) > 4
                or re.match(r'^ITEM\s+\d+', line, re.IGNORECASE)
                or re.match(r'^PART\s+[IVX]+', line, re.IGNORECASE)
            )

            if is_heading:
                sections.append({
                    "id":          f"sec_{sec_id:04d}",
                    "title":       line,
                    "level":       1,
                    "page_start":  0,
                    "page_end":    0,
                    "is_bold":     False,
                    "font_size":   14.0,
                    "summary":     "",
                    "children":    [],
                    "key_section": None,
                })
                sec_id += 1

        return {
            "total_sections": len(sections),
            "sections":       sections,
            "key_sections":   {},
            "page_count":     0,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # TAG KEY FINANCIAL SECTIONS
    # ═══════════════════════════════════════════════════════════════════════

    def _tag_key_sections(self, tree: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify and tag the 5 key financial sections.
        These are used by N06 SniperRAG and N08 BGE-M3
        to focus retrieval on the most relevant sections.
        """
        key_sections = {k: None for k, _ in KEY_SECTIONS}

        def search_sections(sections: List[Dict]) -> None:
            for section in sections:
                title_lower = section["title"].lower()
                for key, keywords in KEY_SECTIONS:
                    if key_sections[key] is None:
                        if any(kw in title_lower for kw in keywords):
                            key_sections[key] = {
                                "section_id": section["id"],
                                "title":      section["title"],
                                "page_start": section["page_start"],
                                "page_end":   section["page_end"],
                            }
                            section["key_section"] = key
                # Recurse into children
                if section.get("children"):
                    search_sections(section["children"])

        search_sections(tree["sections"])
        tree["key_sections"] = key_sections
        return tree

    # ═══════════════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════════════

    def _empty_tree(self) -> Dict[str, Any]:
        """Return an empty section tree."""
        return {
            "total_sections": 0,
            "sections":       [],
            "key_sections":   {k: None for k, _ in KEY_SECTIONS},
            "page_count":     0,
        }

    def get_section_text(
        self,
        state: BAState,
        section_key: str
    ) -> str:
        """
        Extract raw text for a specific key section.
        Used by N06 SniperRAG to search within a section.
        Returns empty string if section not found.
        """
        tree    = state.section_tree
        section = tree.get("key_sections", {}).get(section_key)

        if not section or not state.raw_text:
            return ""

        # Approximate text extraction by character position
        # Real implementation uses page-based splitting
        lines      = state.raw_text.split("\n")
        title      = section["title"].lower()
        start_idx  = 0
        end_idx    = len(lines)

        for i, line in enumerate(lines):
            if title in line.lower():
                start_idx = i
                break

        return "\n".join(lines[start_idx:min(start_idx + 500, end_idx)])

    def summary(self, state: BAState) -> str:
        """Human-readable summary of the section tree."""
        tree = state.section_tree
        if not tree or not tree.get("sections"):
            return "[N02] Empty section tree"

        lines = [
            f"[N02] Section tree: {tree['total_sections']} sections, "
            f"{tree['page_count']} pages"
        ]
        for key, val in tree.get("key_sections", {}).items():
            if val:
                lines.append(
                    f"  {key}: '{val['title']}' "
                    f"(pp. {val['page_start']}–{val['page_end']})"
                )
            else:
                lines.append(f"  {key}: not detected")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK
# run: python src/ingestion/section_tree_builder.py
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        from rich import print as rprint
    except ImportError:
        rprint = print

    rprint("\n[bold cyan]── SectionTreeBuilder sanity check ──[/bold cyan]")

    builder = SectionTreeBuilder()
    rprint("[green]✓[/green] SectionTreeBuilder instantiated")

    # Test with mock headings
    mock_headings = [
        {"page": 1,  "text": "APPLE INC ANNUAL REPORT",
         "font_size": 20.0, "is_bold": True,  "bbox": []},
        {"page": 3,  "text": "Business Overview",
         "font_size": 16.0, "is_bold": True,  "bbox": []},
        {"page": 8,  "text": "Risk Factors",
         "font_size": 16.0, "is_bold": True,  "bbox": []},
        {"page": 24, "text": "Management Discussion and Analysis",
         "font_size": 16.0, "is_bold": True,  "bbox": []},
        {"page": 42, "text": "Financial Statements",
         "font_size": 16.0, "is_bold": True,  "bbox": []},
        {"page": 44, "text": "Consolidated Balance Sheet",
         "font_size": 14.0, "is_bold": True,  "bbox": []},
        {"page": 46, "text": "Notes to Financial Statements",
         "font_size": 14.0, "is_bold": False, "bbox": []},
    ]

    state = BAState(
        session_id    = "test-n02",
        company_name  = "Apple Inc",
        doc_type      = "10-K",
        fiscal_year   = "FY2023",
        heading_positions = mock_headings,
        raw_text      = "Apple Inc Annual Report FY2023\n" * 100,
    )

    state = builder.run(state)

    assert isinstance(state.section_tree, dict)
    assert "sections"     in state.section_tree
    assert "key_sections" in state.section_tree
    rprint(f"[green]✓[/green] Section tree built: "
           f"{state.section_tree['total_sections']} sections")

    # Check key sections detected
    ks = state.section_tree["key_sections"]
    assert ks["business_overview"]  is not None, "business_overview not found"
    assert ks["risk_factors"]       is not None, "risk_factors not found"
    assert ks["mda"]                is not None, "MD&A not found"
    assert ks["financial_statements"] is not None, "financial_statements not found"
    assert ks["notes"]              is not None, "notes not found"
    rprint("[green]✓[/green] All 5 key sections detected")

    # Check page ranges
    mda = ks["mda"]
    assert mda["page_start"] == 24
    rprint(f"[green]✓[/green] MD&A page range: "
           f"pp. {mda['page_start']}–{mda['page_end']}")

    # Test empty state
    empty_state = BAState(session_id="empty-n02")
    empty_state = builder.run(empty_state)
    assert empty_state.section_tree["total_sections"] == 0
    rprint("[green]✓[/green] Empty state handled correctly")

    # Test summary
    summary = builder.summary(state)
    assert "Section tree" in summary
    rprint(f"[green]✓[/green] Summary:\n{summary}")

    rprint("\n[bold green]All checks passed. SectionTreeBuilder ready.[/bold green]\n")