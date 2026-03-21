"""
src/prompts/assembler.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 Rev1.0 FINAL

N10 — Prompt Assembler
Assembles final LLM prompt from retrieved chunks + analyst question.

CRITICAL RULE C7: retrieved_context MUST appear BEFORE the question
in 100% of LLM prompts. This is enforced by:
  1. Template design — context block always first
  2. Pydantic validator on BAState.prompt_template
  3. CI/CD gate scans every prompt file

5 templates — one per query type from N04:
  numerical  — table extraction focus, unit validation
  ratio      — formula + computation section
  multi_doc  — cross-document comparison structure
  text       — narrative analysis, section citation
  forensic   — anomaly detection, Benford context

Every prompt includes:
  - COMPANY / DOC_TYPE / FISCAL_YEAR from BAState metadata
  - Retrieved chunks with C8 metadata prefix
  - Explicit unit instruction (millions/billions/%)
  - Citation format instruction [SECTION/PAGE: value]
  - Question AFTER context (C7)

Writes to BAState: assembled_prompt, prompt_template
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from jinja2 import Environment, BaseLoader, StrictUndefined

from src.state.ba_state import BAState, QueryType
from src.utils.seed_manager import SeedManager
from src.utils.resource_governor import ResourceGovernor

SeedManager.set_all()

# ═══════════════════════════════════════════════════════════════════════════
# PROMPT TEMPLATES — 5 total, one per query type
# C7: retrieved_context ALWAYS before question — never change this order
# ═══════════════════════════════════════════════════════════════════════════

# ── NUMERICAL template ────────────────────────────────────────────────────
NUMERICAL_TEMPLATE = """You are an expert financial analyst reviewing SEC filings.

DOCUMENT CONTEXT (use ONLY this — never use training memory):
Company:     {{ company }}
Document:    {{ doc_type }}
Fiscal Year: {{ fiscal_year }}

RETRIEVED SECTIONS:
{% for chunk in chunks %}
--- Source {{ loop.index }}: {{ chunk.section }} / Page {{ chunk.page }} ---
{{ chunk.text }}

{% endfor %}

INSTRUCTIONS:
1. Answer ONLY from the retrieved sections above.
2. If the answer is not in the sections above, respond with: RETRIEVAL_MISS: [describe what is missing]
3. State the exact numerical value with its unit (millions, billions, or %).
4. Cite every number: [SECTION / PAGE: value]
5. State the fiscal year for every figure cited.
6. Never use training memory — every claim must trace to the sections above.

QUESTION: {{ question }}

ANSWER FORMAT:
ANSWER: [your complete answer with inline citations]
COMPUTATION: N/A
CONFIDENCE: [0.0-1.0] because [brief reason]
CITATIONS: [list every section/page reference used]"""

# ── RATIO template ────────────────────────────────────────────────────────
RATIO_TEMPLATE = """You are an expert financial analyst reviewing SEC filings.

DOCUMENT CONTEXT (use ONLY this — never use training memory):
Company:     {{ company }}
Document:    {{ doc_type }}
Fiscal Year: {{ fiscal_year }}

RETRIEVED SECTIONS:
{% for chunk in chunks %}
--- Source {{ loop.index }}: {{ chunk.section }} / Page {{ chunk.page }} ---
{{ chunk.text }}

{% endfor %}

INSTRUCTIONS:
1. Answer ONLY from the retrieved sections above.
2. If the answer is not in the sections above, respond with: RETRIEVAL_MISS: [describe what is missing]
3. Show the formula used: e.g. Gross Margin = Gross Profit / Revenue
4. Show all inputs with citations: [SECTION / PAGE: value]
5. Show the computation step by step.
6. State the result with correct units (% for ratios, $ for values).
7. State the fiscal year for every figure cited.
8. Never use training memory — every claim must trace to the sections above.

QUESTION: {{ question }}

ANSWER FORMAT:
ANSWER: [result with units]
COMPUTATION: [formula] = [numerator with citation] / [denominator with citation] = [result]
CONFIDENCE: [0.0-1.0] because [brief reason]
CITATIONS: [list every section/page reference used]"""

# ── MULTI_DOC template ────────────────────────────────────────────────────
MULTI_DOC_TEMPLATE = """You are an expert financial analyst reviewing SEC filings.

DOCUMENT CONTEXT (use ONLY this — never use training memory):
Company:     {{ company }}
Document:    {{ doc_type }}
Fiscal Year: {{ fiscal_year }}

RETRIEVED SECTIONS:
{% for chunk in chunks %}
--- Source {{ loop.index }}: {{ chunk.section }} / Page {{ chunk.page }} ---
{{ chunk.text }}

{% endfor %}

INSTRUCTIONS:
1. Answer ONLY from the retrieved sections above.
2. If the answer is not in the sections above, respond with: RETRIEVAL_MISS: [describe what is missing]
3. Compare figures across time periods or segments systematically.
4. Present data in a structured format: [Period/Segment]: [Value with citation]
5. Calculate year-over-year or period-over-period changes where relevant.
6. Cite every number: [SECTION / PAGE: value]
7. Note any restatements or reclassifications that affect comparability.
8. Never use training memory — every claim must trace to the sections above.

QUESTION: {{ question }}

ANSWER FORMAT:
ANSWER: [structured comparison with inline citations]
COMPUTATION: [any calculations performed, else N/A]
CONFIDENCE: [0.0-1.0] because [brief reason]
CITATIONS: [list every section/page reference used]"""

# ── TEXT template ─────────────────────────────────────────────────────────
TEXT_TEMPLATE = """You are an expert financial analyst reviewing SEC filings.

DOCUMENT CONTEXT (use ONLY this — never use training memory):
Company:     {{ company }}
Document:    {{ doc_type }}
Fiscal Year: {{ fiscal_year }}

RETRIEVED SECTIONS:
{% for chunk in chunks %}
--- Source {{ loop.index }}: {{ chunk.section }} / Page {{ chunk.page }} ---
{{ chunk.text }}

{% endfor %}

INSTRUCTIONS:
1. Answer ONLY from the retrieved sections above.
2. If the answer is not in the sections above, respond with: RETRIEVAL_MISS: [describe what is missing]
3. Provide a complete narrative answer citing specific sections.
4. Use direct evidence from the text — quote key phrases with citations.
5. Cite every claim: [SECTION / PAGE]
6. Do not speculate beyond what the document states.
7. Never use training memory — every claim must trace to the sections above.

QUESTION: {{ question }}

ANSWER FORMAT:
ANSWER: [complete narrative answer with inline citations]
COMPUTATION: N/A
CONFIDENCE: [0.0-1.0] because [brief reason]
CITATIONS: [list every section/page reference used]"""

# ── FORENSIC template ─────────────────────────────────────────────────────
FORENSIC_TEMPLATE = """You are an expert forensic financial analyst reviewing SEC filings.

DOCUMENT CONTEXT (use ONLY this — never use training memory):
Company:     {{ company }}
Document:    {{ doc_type }}
Fiscal Year: {{ fiscal_year }}

RETRIEVED SECTIONS:
{% for chunk in chunks %}
--- Source {{ loop.index }}: {{ chunk.section }} / Page {{ chunk.page }} ---
{{ chunk.text }}

{% endfor %}

FORENSIC ANALYSIS FRAMEWORK:
Apply these checks to the retrieved sections:
F1: Digit frequency — do leading digits follow Benford Law distribution?
F2: Round numbers — unusual concentration of round figures ($X00M)?
F3: Trend breaks — abrupt changes inconsistent with narrative explanation?
F4: Cross-section consistency — do related figures reconcile?
F5: Disclosure completeness — are required disclosures present?

INSTRUCTIONS:
1. Answer ONLY from the retrieved sections above.
2. If the answer is not in the sections above, respond with: RETRIEVAL_MISS: [describe what is missing]
3. Apply the forensic framework systematically.
4. Flag anomalies with severity: HIGH / MEDIUM / LOW.
5. Cite every anomaly: [SECTION / PAGE: specific finding]
6. State clearly: this analysis is based on public filing data only.
7. Never conclude fraud — flag signals for further investigation.
8. Never use training memory — every claim must trace to the sections above.

QUESTION: {{ question }}

ANSWER FORMAT:
ANSWER: [forensic findings with anomaly flags and citations]
COMPUTATION: [any statistical observations, else N/A]
CONFIDENCE: [0.0-1.0] because [brief reason]
CITATIONS: [list every section/page reference used]"""

# ── Template registry ─────────────────────────────────────────────────────
TEMPLATES: Dict[str, str] = {
    QueryType.NUMERICAL: NUMERICAL_TEMPLATE,
    QueryType.RATIO:     RATIO_TEMPLATE,
    QueryType.MULTI_DOC: MULTI_DOC_TEMPLATE,
    QueryType.TEXT:      TEXT_TEMPLATE,
    QueryType.FORENSIC:  FORENSIC_TEMPLATE,
}


class PromptAssembler:
    """
    N10: Prompt Assembler.

    Assembles LLM prompt from retrieved chunks + analyst question.
    C7 enforced: context always before question.
    5 templates — one per query type.
    Writes assembled_prompt to BAState.
    """

    def __init__(self):
        SeedManager.set_all()
        # StrictUndefined raises error on missing variables — catches bugs early
        self._env = Environment(
            loader        = BaseLoader(),
            undefined     = StrictUndefined,
            trim_blocks   = True,
            lstrip_blocks = True,
        )
        self._compiled: Dict[str, Any] = {}
        self._compile_templates()

    def _compile_templates(self) -> None:
        """Pre-compile all 5 templates at startup."""
        for query_type, template_str in TEMPLATES.items():
            self._compiled[query_type] = self._env.from_string(template_str)

    # ═══════════════════════════════════════════════════════════════════════
    # RUN — BAState integration
    # ═══════════════════════════════════════════════════════════════════════

    def run(self, state: BAState) -> BAState:
        """
        Main entry point — N10 node.
        Reads: state.retrieval_stage_2, state.query, state.query_type
               state.company_name, state.doc_type, state.fiscal_year
        Writes: state.assembled_prompt, state.prompt_template
        """
        ResourceGovernor.check("N10 Prompt Assembler")

        # Guard: need at least a query
        if not state.query:
            print("[N10] No query — skipping prompt assembly")
            state.assembled_prompt = ""
            state.prompt_template  = "context_first"
            return state

        # Get chunks — prefer stage_2 (reranked), fall back to stage_1
        chunks = state.retrieval_stage_2 or state.retrieval_stage_1 or []

        # Normalise chunk format for template rendering
        normalised = self._normalise_chunks(chunks)

        # Select template based on query_type
        query_type = state.query_type or QueryType.TEXT
        prompt     = self._assemble(
            query_type  = query_type,
            question    = state.query,
            chunks      = normalised,
            company     = state.company_name  or "Unknown Company",
            doc_type    = state.doc_type      or "10-K",
            fiscal_year = state.fiscal_year   or "Unknown FY",
        )

        state.assembled_prompt = prompt
        state.prompt_template  = "context_first"   # C7 enforced always

        # Validate C7 — context before question
        self._validate_c7(prompt, state.query)

        print(f"[N10] Assembled {query_type} prompt — "
              f"{len(chunks)} chunks — "
              f"{len(prompt)} chars")

        return state

    # ═══════════════════════════════════════════════════════════════════════
    # ASSEMBLE
    # ═══════════════════════════════════════════════════════════════════════

    def _assemble(
        self,
        query_type:  str,
        question:    str,
        chunks:      List[Dict],
        company:     str,
        doc_type:    str,
        fiscal_year: str,
    ) -> str:
        """Render the correct Jinja2 template with provided variables."""
        template = self._compiled.get(
            query_type,
            self._compiled[QueryType.TEXT]   # fallback
        )
        return template.render(
            question    = question,
            chunks      = chunks,
            company     = company,
            doc_type    = doc_type,
            fiscal_year = fiscal_year,
        )

    def assemble_direct(
        self,
        query_type:  str,
        question:    str,
        chunks:      List[Dict],
        company:     str     = "Unknown Company",
        doc_type:    str     = "10-K",
        fiscal_year: str     = "Unknown FY",
    ) -> str:
        """
        Direct assembly without BAState.
        Used by PIV pods to assemble prompts directly.
        """
        normalised = self._normalise_chunks(chunks)
        return self._assemble(
            query_type  = query_type,
            question    = question,
            chunks      = normalised,
            company     = company,
            doc_type    = doc_type,
            fiscal_year = fiscal_year,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════════════

    def _normalise_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Ensure every chunk has section and page fields for templates.
        Handles both BGE format (section, page) and raw format.
        """
        normalised = []
        for chunk in chunks:
            text = chunk.get("text") or chunk.get("content") or ""
            normalised.append({
                "text":    text,
                "section": chunk.get("section", "Unknown Section"),
                "page":    chunk.get("page",    "?"),
                "company": chunk.get("company", ""),
                "doc_type":    chunk.get("doc_type",    ""),
                "fiscal_year": chunk.get("fiscal_year", ""),
            })
        return normalised

    def _validate_c7(self, prompt: str, question: str) -> None:
        """
        C7 validation: context must appear before the question.
        Raises AssertionError if question appears before RETRIEVED SECTIONS.
        """
        retrieved_pos = prompt.find("RETRIEVED SECTIONS")
        question_pos  = prompt.find(f"QUESTION: {question}")

        if retrieved_pos == -1:
            # No chunks — prompt still valid if question is at end
            return

        assert question_pos > retrieved_pos, (
            f"C7 VIOLATION: QUESTION appears before RETRIEVED SECTIONS. "
            f"retrieved_pos={retrieved_pos} question_pos={question_pos}"
        )

    def get_template_names(self) -> List[str]:
        """Return list of available template names."""
        return list(self._compiled.keys())

    def context_before_question(self, prompt: str) -> bool:
        """
        Returns True if context appears before question in prompt.
        Used by CI/CD gate.
        """
        retrieved_pos = prompt.find("RETRIEVED SECTIONS")
        question_pos  = prompt.find("QUESTION:")
        if retrieved_pos == -1 or question_pos == -1:
            return True   # no chunks or no question — not a violation
        return retrieved_pos < question_pos


# ═══════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK
# run: python src/prompts/assembler.py
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        from rich import print as rprint
    except ImportError:
        rprint = print

    rprint("\n[bold cyan]── PromptAssembler sanity check ──[/bold cyan]")

    assembler = PromptAssembler()
    rprint("[green]✓[/green] PromptAssembler instantiated")
    rprint(f"[green]✓[/green] Templates compiled: {assembler.get_template_names()}")

    # Mock chunks
    chunks = [
        {
            "chunk_id":    "chunk_000001",
            "text":        "Apple / 10-K / FY2023 / Financial Statements / 42\n"
                           "Net income: $96,995 million for fiscal year ended "
                           "September 30 2023.",
            "section":     "Financial Statements",
            "page":        "42",
            "company":     "Apple Inc",
            "doc_type":    "10-K",
            "fiscal_year": "FY2023",
        },
        {
            "chunk_id":    "chunk_000002",
            "text":        "Apple / 10-K / FY2023 / MD&A / 28\n"
                           "Total net sales were $383,285 million for fiscal 2023.",
            "section":     "MD&A",
            "page":        "28",
            "company":     "Apple Inc",
            "doc_type":    "10-K",
            "fiscal_year": "FY2023",
        },
    ]

    # Test all 5 templates
    query_types = [
        QueryType.NUMERICAL,
        QueryType.RATIO,
        QueryType.MULTI_DOC,
        QueryType.TEXT,
        QueryType.FORENSIC,
    ]

    for qt in query_types:
        prompt = assembler.assemble_direct(
            query_type  = qt,
            question    = "What was Apple net income in FY2023?",
            chunks      = chunks,
            company     = "Apple Inc",
            doc_type    = "10-K",
            fiscal_year = "FY2023",
        )
        assert len(prompt) > 100
        assert "Apple Inc"    in prompt
        assert "FY2023"       in prompt
        assert "RETRIEVED SECTIONS" in prompt
        assert "QUESTION:"    in prompt
        # C7: context before question
        assert assembler.context_before_question(prompt), \
            f"C7 VIOLATION in {qt} template"
        rprint(f"[green]✓[/green] {qt} template: {len(prompt)} chars — C7 OK")

    # BAState integration
    state = BAState(
        session_id        = "sanity-n10",
        query             = "What was Apple net income in FY2023?",
        query_type        = QueryType.NUMERICAL,
        company_name      = "Apple Inc",
        doc_type          = "10-K",
        fiscal_year       = "FY2023",
        retrieval_stage_2 = chunks,
    )
    state = assembler.run(state)
    assert state.assembled_prompt != ""
    assert state.prompt_template  == "context_first"
    assert "Apple Inc" in state.assembled_prompt
    assert "RETRIEVED SECTIONS" in state.assembled_prompt
    assert assembler.context_before_question(state.assembled_prompt)
    rprint(f"[green]✓[/green] BAState: prompt={len(state.assembled_prompt)} chars "
           f"template={state.prompt_template}")

    # No chunks still assembles
    state2 = BAState(
        session_id  = "sanity-n10-empty",
        query       = "What was Apple net income?",
        query_type  = QueryType.TEXT,
        company_name= "Apple Inc",
        doc_type    = "10-K",
        fiscal_year = "FY2023",
    )
    state2 = assembler.run(state2)
    assert state2.assembled_prompt != ""
    rprint(f"[green]✓[/green] Empty chunks handled — prompt still assembled")

    # No query returns empty
    state3 = BAState(session_id="sanity-n10-noquery")
    state3 = assembler.run(state3)
    assert state3.assembled_prompt == ""
    rprint(f"[green]✓[/green] No query returns empty prompt correctly")

    # C7 validation fires on violation
    try:
        assembler._validate_c7(
            "QUESTION: test\nRETRIEVED SECTIONS:\nsome context",
            "test"
        )
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "C7 VIOLATION" in str(e)
        rprint(f"[green]✓[/green] C7 violation detected correctly")

    rprint(f"\n[bold green]All checks passed. PromptAssembler ready.[/bold green]\n")