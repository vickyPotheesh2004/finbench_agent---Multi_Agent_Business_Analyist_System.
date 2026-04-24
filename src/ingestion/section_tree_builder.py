"""
N02 Section Tree Builder — Hierarchical Section Index
PDR-BAAAI-001 · Rev 1.0 · Node N02

Purpose:
    Builds a hierarchical JSON section tree from heading_positions
    produced by N01 PDF Ingestor.

    For each section node:
        - section name (from heading text)
        - start_page / end_page
        - font_size + is_bold
        - 1-sentence Gemma4 summary (local LLM, zero network calls)
        - child sections (nested hierarchy)

    Identifies 5 major SEC filing sections:
        Business Overview, Risk Factors, MD&A,
        Financial Statements, Notes

    Writes: state.section_tree (hierarchical dict)

Constraints satisfied:
    C1  $0 cost — local LLM only
    C2  100% local — localhost:11434, zero network calls
    C5  seed=42
    C9  No _rlef_ fields in output
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Known SEC filing section names for identification
SEC_MAJOR_SECTIONS = {
    "note":                  "Notes",
    "business":              "Business Overview",
    "risk factor":           "Risk Factors",
    "management":            "MD&A",
    "financial statement":   "Financial Statements",
    "quantitative":          "Quantitative Disclosures",
    "controls":              "Controls and Procedures",
    "legal proceeding":      "Legal Proceedings",
    "market":                "Market Information",
    "selected financial":    "Selected Financial Data",
    "executive compensation":"Executive Compensation",
    "directors":             "Directors and Officers",
    "security ownership":    "Security Ownership",
    "properties":            "Properties",
    "mine safety":           "Mine Safety",
}

# Font size thresholds for heading hierarchy
HEADING_H1_MIN = 16.0
HEADING_H2_MIN = 13.0

# Maximum heading text length (filter out noise)
MAX_HEADING_LENGTH = 200
MIN_HEADING_LENGTH = 3

# Max sections to summarise (LLM calls are slow on CPU)
MAX_SECTIONS_TO_SUMMARISE = 20


class SectionNode:
    """A single node in the section tree."""

    __slots__ = (
        "name", "level", "start_page", "end_page",
        "font_size", "is_bold", "summary",
        "sec_type", "children",
    )

    def __init__(
        self,
        name:       str,
        level:      int,
        start_page: int,
        end_page:   int = 0,
        font_size:  float = 13.0,
        is_bold:    bool = False,
        summary:    str = "",
        sec_type:   str = "",
    ) -> None:
        self.name       = name
        self.level      = level
        self.start_page = start_page
        self.end_page   = end_page
        self.font_size  = font_size
        self.is_bold    = is_bold
        self.summary    = summary
        self.sec_type   = sec_type
        self.children:  List[SectionNode] = []

    def to_dict(self) -> Dict:
        return {
            "name":       self.name,
            "level":      self.level,
            "start_page": self.start_page,
            "end_page":   self.end_page,
            "font_size":  self.font_size,
            "is_bold":    self.is_bold,
            "summary":    self.summary,
            "sec_type":   self.sec_type,
            "children":   [c.to_dict() for c in self.children],
        }


class SectionTreeBuilder:
    """
    N02 Section Tree Builder.

    Two usage modes:
        1. builder.build(heading_positions, raw_text) → dict
        2. builder.run(ba_state)                      → BAState
    """

    def __init__(self, llm_client=None) -> None:
        self._llm = llm_client   # Optional — None = skip LLM summaries

    # ── LangGraph pipeline node ───────────────────────────────────────────────

    def run(self, state) -> object:
        """
        LangGraph N02 node entry point.
        Reads:  state.heading_positions, state.raw_text
        Writes: state.section_tree
        """
        headings = getattr(state, "heading_positions", []) or []
        raw_text = getattr(state, "raw_text",          "") or ""

        section_tree = self.build(headings, raw_text)
        state.section_tree = section_tree

        n_sections = len(section_tree.get("children", []))
        logger.info(
            "N02 Section Tree: %d top-level sections | %d total headings",
            n_sections, len(headings),
        )
        return state

    # ── Core build method ─────────────────────────────────────────────────────

    def build(
        self,
        heading_positions: List[Dict],
        raw_text:          str = "",
    ) -> Dict:
        """
        Build hierarchical section tree from heading positions.

        Args:
            heading_positions : List of dicts with text, font_size, page, is_bold
            raw_text          : Full document text for extracting section content

        Returns:
            Dict with 'document' root and nested 'children' sections
        """
        # Clean and filter headings
        headings = self._clean_headings(heading_positions)

        if not headings:
            return self._empty_tree()

        # Assign heading levels based on font size
        levelled = self._assign_levels(headings)

        # Build section pages (start_page → end_page)
        paged = self._assign_page_ranges(levelled)

        # Build nested tree
        tree_nodes = self._build_tree(paged)

        # Generate 1-sentence summaries (if LLM available)
        if self._llm and raw_text:
            self._add_summaries(tree_nodes, raw_text)

        # Identify SEC filing section types
        self._classify_sec_sections(tree_nodes)

        return {
            "document": "root",
            "total_sections": self._count_sections(tree_nodes),
            "children": [node.to_dict() for node in tree_nodes],
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _clean_headings(self, headings: List[Dict]) -> List[Dict]:
        """Filter and clean heading list."""
        cleaned = []
        for h in headings:
            text = h.get("text", "").strip()
            if not text:
                continue
            if len(text) < MIN_HEADING_LENGTH:
                continue
            if len(text) > MAX_HEADING_LENGTH:
                continue
            # Skip headings that are just numbers or single words unlikely to be sections
            if re.match(r'^[\d\s\.\-]+$', text):
                continue
            cleaned.append(h)
        return cleaned

    def _assign_levels(self, headings: List[Dict]) -> List[Dict]:
        """Assign level 1 or 2 based on font size."""
        levelled = []
        for h in headings:
            font_size = h.get("font_size", 13.0)
            is_bold   = h.get("is_bold",   False)

            if font_size >= HEADING_H1_MIN or (font_size >= HEADING_H2_MIN and is_bold):
                level = 1
            else:
                level = 2

            levelled.append({
                **h,
                "level": level,
            })
        return levelled

    def _assign_page_ranges(self, headings: List[Dict]) -> List[Dict]:
        """Assign end_page = next heading's start_page - 1."""
        paged = []
        for i, h in enumerate(headings):
            start_page = h.get("page", 0)
            if i + 1 < len(headings):
                end_page = headings[i + 1].get("page", start_page)
            else:
                end_page = start_page + 50  # estimate for last section
            paged.append({**h, "start_page": start_page, "end_page": end_page})
        return paged

    def _build_tree(self, headings: List[Dict]) -> List[SectionNode]:
        """Build nested hierarchy from flat heading list."""
        roots:   List[SectionNode] = []
        current_h1: Optional[SectionNode] = None

        for h in headings:
            node = SectionNode(
                name       = h.get("text",       ""),
                level      = h.get("level",       1),
                start_page = h.get("start_page",  0),
                end_page   = h.get("end_page",    0),
                font_size  = h.get("font_size",   13.0),
                is_bold    = h.get("is_bold",     False),
            )

            if node.level == 1:
                roots.append(node)
                current_h1 = node
            else:
                if current_h1:
                    current_h1.children.append(node)
                else:
                    roots.append(node)

        return roots

    def _add_summaries(
        self, nodes: List[SectionNode], raw_text: str
    ) -> None:
        """Generate 1-sentence summaries for top sections using local LLM."""
        count = 0
        for node in nodes:
            if count >= MAX_SECTIONS_TO_SUMMARISE:
                break
            if not node.name:
                continue
            try:
                summary = self._generate_summary(node.name, raw_text)
                node.summary = summary
                count += 1
            except Exception as exc:
                logger.debug("Summary failed for '%s': %s", node.name, exc)

    def _generate_summary(self, section_name: str, raw_text: str) -> str:
        """
        Generate 1-sentence section summary using local LLM.
        Extracts a small snippet around the section heading first.
        """
        # Find relevant text snippet (500 chars around section name)
        idx = raw_text.lower().find(section_name.lower()[:30])
        if idx == -1:
            snippet = raw_text[:300]
        else:
            start   = max(0, idx)
            snippet = raw_text[start:start + 500]

        prompt = (
            f"Summarise in exactly one sentence what the '{section_name}' "
            f"section of this financial filing covers, based on this excerpt:\n\n"
            f"{snippet}\n\n"
            f"One sentence only. Be specific and factual."
        )

        response = self._llm.chat(prompt, temperature=0.1)
        # Take first sentence only
        sentences = re.split(r'[.!?]', response)
        summary   = sentences[0].strip() + "." if sentences and sentences[0].strip() else ""
        return summary[:300]

    def _classify_sec_sections(self, nodes: List[SectionNode]) -> None:
        """Identify standard SEC filing section types."""
        for node in nodes:
            node.sec_type = self._get_sec_type(node.name)
            for child in node.children:
                child.sec_type = self._get_sec_type(child.name)

    @staticmethod
    def _get_sec_type(name: str) -> str:
        """Match heading text to known SEC section types."""
        name_lower = name.lower()
        for keyword, sec_type in SEC_MAJOR_SECTIONS.items():
            if keyword in name_lower:
                return sec_type
        return "Other"

    @staticmethod
    def _count_sections(nodes: List[SectionNode]) -> int:
        """Count total sections including children."""
        total = len(nodes)
        for node in nodes:
            total += len(node.children)
        return total

    @staticmethod
    def _empty_tree() -> Dict:
        """Return empty tree when no headings found."""
        return {
            "document":       "root",
            "total_sections": 0,
            "children":       [],
        }


# ── Convenience wrapper ───────────────────────────────────────────────────────

def run_section_tree_builder(state, llm_client=None) -> object:
    """Convenience wrapper for LangGraph N02 node."""
    return SectionTreeBuilder(llm_client=llm_client).run(state)