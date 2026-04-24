"""
src/agents/auditor_pod.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 Rev1.0 FINAL

N14 — BlindAuditor Pod
Completely independent analysis pod. NEVER sees N11 or N12 output.

BLIND design principle:
  Architecturally enforced — auditor reads only from:
    state.query
    state.assembled_prompt
    state.retrieval_stage_2 (same chunks as N11/N12)

  Never reads from:
    state.analyst_output     (N11)
    state.analyst_confidence (N11)
    state.quant_result       (N12)

  This prevents anchoring bias where the auditor simply confirms
  the analyst's answer rather than forming an independent view.

Contradiction detection:
  After forming its own answer, auditor checks if it contradicts N11.
  Contradictions are flagged in state.contradiction_flags.
  N15 Mediator uses these flags for resolution.

Writes to BAState:
  auditor_output        — independent answer
  auditor_confidence    — 0.0-1.0
  auditor_citations     — list of section/page references
  auditor_attempt_count — retries used
  auditor_piv_status    — PASS or REJECT
  contradiction_flags   — list of contradictions with N11

Emotional identity: External auditor
  Professional scepticism — same as CuriousValidator but applied
  to the document itself, not another agent's answer.
  Moral courage — willing to flag uncomfortable findings.
  Methodical patience — no shortcut is acceptable.
"""

import sys
import re
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.agents.planner     import StrategicPlanner
from src.agents.implementor import ContextImplementor
from src.agents.validator   import CuriousValidator
from src.agents.piv_loop    import PIVLoopController, PIVResult
from src.agents.validator   import VALIDATOR_PASS
from src.state.ba_state     import BAState, QueryType, Difficulty, PIVStatus
from src.utils.seed_manager import SeedManager
from src.utils.resource_governor import ResourceGovernor

SeedManager.set_all()


class AuditorPod:
    """
    N14 — BlindAuditor Pod.

    Completely independent PIV loop.
    NEVER reads analyst_output or quant_result.
    Detects contradictions after forming independent answer.
    """

    def __init__(
        self,
        planner:     Optional[StrategicPlanner]   = None,
        implementor: Optional[ContextImplementor] = None,
        validator:   Optional[CuriousValidator]   = None,
        max_retries: int = 5,
    ):
        SeedManager.set_all()
        self.planner     = planner     or StrategicPlanner()
        self.implementor = implementor or ContextImplementor()
        self.validator   = validator   or CuriousValidator()
        self.max_retries = max_retries

        # Separate PIV instance — never shared with N11 or N12
        self.piv = PIVLoopController(
            planner     = self.planner,
            implementor = self.implementor,
            validator   = self.validator,
            max_retries = self.max_retries,
        )

    def run(self, state: BAState) -> BAState:
        """
        Main entry point — N14 node.

        BLIND: reads query + context only.
        NEVER reads analyst_output or quant_result.

        Reads:
          state.query
          state.assembled_prompt
          state.retrieval_stage_2
          state.query_type
          state.query_difficulty

        Writes:
          state.auditor_output
          state.auditor_confidence
          state.auditor_citations
          state.auditor_attempt_count
          state.auditor_piv_status
          state.contradiction_flags
        """
        ResourceGovernor.check("N14 Auditor Pod")

        if not state.query:
            print("[N14] No query — skipping auditor pod")
            state.auditor_piv_status = PIVStatus.REJECT
            return state

        # ── BLIND: build context independently ────────────────────────────
        retrieved_context = self._build_blind_context(state)
        section_summary   = self._build_section_summary(state)
        query_type        = (state.query_type or QueryType.TEXT).value
        query_difficulty  = (state.query_difficulty or Difficulty.MEDIUM).value

        print(f"[N14] Running BLIND PIV loop — "
              f"query_type={query_type} "
              f"chunks={len(state.retrieval_stage_2)}")

        # Run independent PIV loop
        result: PIVResult = self.piv.run(
            query             = state.query,
            retrieved_context = retrieved_context,
            section_summary   = section_summary,
            query_type        = query_type,
            query_difficulty  = query_difficulty,
            pod_role          = "auditor",
            state             = state,
        )

        # Write results
        state.auditor_output        = result.answer
        state.auditor_confidence    = result.confidence
        state.auditor_citations     = result.citations
        state.auditor_attempt_count = min(result.retries_used, 5)

        # Contradiction detection
        contradictions            = self._detect_contradictions(state)
        state.contradiction_flags = contradictions

        if result.low_confidence:
            state.auditor_piv_status = PIVStatus.REJECT
            print(f"[N14] Low confidence after {result.retries_used} retries")
        else:
            state.auditor_piv_status = PIVStatus.PASS

        print(f"[N14] Complete — "
              f"status={state.auditor_piv_status} "
              f"conf={result.confidence:.2f} "
              f"contradictions={len(contradictions)}")

        return state

    # ═══════════════════════════════════════════════════════════════════════
    # BLIND CONTEXT BUILDER
    # ═══════════════════════════════════════════════════════════════════════

    def _build_blind_context(self, state: BAState) -> str:
        """
        Build context independently from chunks.
        NEVER uses assembled_prompt (which may contain N11 context).
        Always builds fresh from retrieval_stage_2.
        """
        chunks = state.retrieval_stage_2 or state.retrieval_stage_1 or []

        if not chunks:
            return "No context retrieved."

        parts = []
        for i, chunk in enumerate(chunks, 1):
            text    = chunk.get("text") or chunk.get("content") or ""
            section = chunk.get("section", "Unknown Section")
            page    = chunk.get("page",    "?")
            company = chunk.get("company", state.company_name or "")
            fy      = chunk.get("fiscal_year", state.fiscal_year or "")
            parts.append(
                f"--- Source {i}: {section} / Page {page} "
                f"[{company} {fy}] ---\n{text}"
            )

        context = "\n\n".join(parts)

        return (
            f"INDEPENDENT AUDIT REVIEW\n"
            f"Company: {state.company_name or 'Unknown'} | "
            f"Doc: {state.doc_type or '10-K'} | "
            f"FY: {state.fiscal_year or 'Unknown'}\n\n"
            f"RETRIEVED SECTIONS FOR INDEPENDENT REVIEW:\n"
            f"{context}"
        )

    def _build_section_summary(self, state: BAState) -> str:
        """Build section summary for Planner."""
        return (
            f"[BLIND AUDIT] Company: {state.company_name or 'Unknown'} | "
            f"Doc: {state.doc_type or '10-K'} | "
            f"FY: {state.fiscal_year or 'Unknown'} | "
            f"IMPORTANT: Form your own independent view. "
            f"Do not be influenced by prior analysis."
        )

    # ═══════════════════════════════════════════════════════════════════════
    # CONTRADICTION DETECTION
    # ═══════════════════════════════════════════════════════════════════════

    def _detect_contradictions(self, state: BAState) -> List[str]:
        """
        Detect contradictions between auditor answer and analyst answer.
        Returns list of contradiction descriptions.
        """
        contradictions = []

        analyst_answer = state.analyst_output or ""
        auditor_answer = state.auditor_output or ""

        if not analyst_answer or not auditor_answer:
            return contradictions

        # Extract numbers from both answers
        analyst_nums = self._extract_numbers(analyst_answer)
        auditor_nums = self._extract_numbers(auditor_answer)

        # Numerical disagreement > 10%
        if analyst_nums and auditor_nums:
            analyst_primary = analyst_nums[0]
            auditor_primary = auditor_nums[0]

            if analyst_primary > 0 and auditor_primary > 0:
                diff_pct = abs(analyst_primary - auditor_primary) / max(
                    analyst_primary, auditor_primary
                )
                if diff_pct > 0.10:
                    contradictions.append(
                        f"NUMERICAL_CONTRADICTION: "
                        f"Analyst={analyst_primary:.0f} vs "
                        f"Auditor={auditor_primary:.0f} "
                        f"({diff_pct:.1%} difference)"
                    )

        # Fiscal year disagreement
        analyst_fy = self._extract_fiscal_year(analyst_answer)
        auditor_fy  = self._extract_fiscal_year(auditor_answer)

        if analyst_fy and auditor_fy and analyst_fy != auditor_fy:
            contradictions.append(
                f"FISCAL_YEAR_CONTRADICTION: "
                f"Analyst references {analyst_fy} vs "
                f"Auditor references {auditor_fy}"
            )

        # Sign contradiction
        analyst_neg = bool(re.search(r'\(\s*[\d,]+\s*\)', analyst_answer))
        auditor_neg  = bool(re.search(r'\(\s*[\d,]+\s*\)', auditor_answer))

        if analyst_neg != auditor_neg and analyst_nums and auditor_nums:
            contradictions.append(
                f"SIGN_CONTRADICTION: "
                f"Analyst and Auditor disagree on sign "
                f"(positive vs negative)"
            )

        return contradictions

    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numerical values from text."""
        pattern = r'\$?([\d,]+(?:\.\d+)?)\s*(?:million|billion|M|B)?'
        matches = re.findall(pattern, text, re.IGNORECASE)
        numbers = []
        for m in matches:
            try:
                val = float(m.replace(",", ""))
                if val > 0:
                    numbers.append(val)
            except ValueError:
                pass
        return numbers[:5]

    def _extract_fiscal_year(self, text: str) -> Optional[str]:
        """
        Extract fiscal year reference from text.
        Normalises all variants to FY20XX format.
        Handles: FY2023, FY 2023, fiscal 2022, fiscal year 2022, 2023
        """
        match = re.search(
            r'FY\s*(20\d{2})|fiscal\s*(?:year\s*)?(20\d{2})|(20\d{2})',
            text, re.IGNORECASE
        )
        if match:
            year = match.group(1) or match.group(2) or match.group(3)
            return f"FY{year}"
        return None


# ═══════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK
# run: python src/agents/auditor_pod.py
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        from rich import print as rprint
    except ImportError:
        rprint = print

    rprint("\n[bold cyan]-- AuditorPod (N14) sanity check --[/bold cyan]")

    pod = AuditorPod()
    rprint("[green]✓[/green] AuditorPod instantiated")
    rprint("[green]✓[/green] Separate PIV instance from N11/N12")

    state = BAState(
        session_id        = "sanity-n14",
        query             = "What was Apple net income in FY2023?",
        query_type        = QueryType.NUMERICAL,
        query_difficulty  = Difficulty.EASY,
        company_name      = "Apple Inc",
        doc_type          = "10-K",
        fiscal_year       = "FY2023",
        retrieval_stage_2 = [{
            "text":        "Net income: $96,995 million FY2023.",
            "section":     "Financial Statements",
            "page":        "42",
            "company":     "Apple Inc",
            "fiscal_year": "FY2023",
        }],
    )

    state = pod.run(state)

    rprint(f"[green]✓[/green] auditor_piv_status: {state.auditor_piv_status}")
    rprint(f"[green]✓[/green] auditor_confidence: {state.auditor_confidence}")
    rprint(f"[green]✓[/green] auditor_output length: {len(state.auditor_output)}")
    rprint(f"[green]✓[/green] contradiction_flags: {state.contradiction_flags}")
    rprint(f"[green]✓[/green] analyst_output untouched: '{state.analyst_output}'")

    assert state.auditor_piv_status in [PIVStatus.PASS, PIVStatus.REJECT]
    assert 0.0 <= state.auditor_confidence <= 1.0
    assert state.analyst_output == ""
    assert state.seed == 42

    # Test contradiction detection
    state2 = BAState(
        session_id     = "sanity-n14-contradiction",
        query          = "What was net income?",
        analyst_output = "Net income was $96,995 million FY2023",
        auditor_output = "Net income was $57,411 million FY2022",
    )
    contradictions = pod._detect_contradictions(state2)
    rprint(f"[green]✓[/green] Contradictions detected: {contradictions}")
    assert len(contradictions) > 0

    # Test fiscal year extraction normalisation
    assert pod._extract_fiscal_year("FY2023 results")    == "FY2023"
    assert pod._extract_fiscal_year("fiscal 2022")       == "FY2022"
    assert pod._extract_fiscal_year("fiscal year 2021")  == "FY2021"
    assert pod._extract_fiscal_year("no year here")      is None
    rprint(f"[green]✓[/green] Fiscal year extraction normalised correctly")

    # Test no contradiction on same answer
    state3 = BAState(
        session_id     = "sanity-n14-agree",
        query          = "What was net income?",
        analyst_output = "Net income was $96,995 million FY2023",
        auditor_output = "Net income was $96,995 million FY2023",
    )
    no_contradictions = pod._detect_contradictions(state3)
    assert len(no_contradictions) == 0
    rprint(f"[green]✓[/green] No contradictions on matching answers")

    rprint(f"\n[bold green]All checks passed. AuditorPod N14 ready.[/bold green]\n")