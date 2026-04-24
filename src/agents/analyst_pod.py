"""
src/agents/analyst_pod.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 Rev1.0 FINAL

N11 — LeadAnalyst Pod
Primary analysis pod. Handles all 5 query types.
Wires together: StrategicPlanner + ContextImplementor +
                CuriousValidator + PIVLoopController

Role: Primary analysis pod — first pod to run on every query.
      Uses assembled_prompt from N10 as retrieved context.
      Writes analyst_output + analyst_confidence + analyst_citations
      to BAState.

Emotional identity: Senior equity research analyst.
Deep domain expertise. Methodical. Citation-first.
Never guesses. Always defers to retrieved context.

N11 is the LeadAnalyst — it sets the quality bar.
N12 (Quant) and N14 (Auditor) run independently after N11.
N15 (Mediator) arbitrates between all three.
"""

import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.agents.planner      import StrategicPlanner
from src.agents.implementor  import ContextImplementor
from src.agents.validator    import CuriousValidator
from src.agents.piv_loop     import PIVLoopController, PIVResult
from src.state.ba_state      import BAState, QueryType, Difficulty, PIVStatus
from src.utils.seed_manager  import SeedManager
from src.utils.resource_governor import ResourceGovernor

SeedManager.set_all()


class AnalystPod:
    """
    N11 — LeadAnalyst Pod.

    Orchestrates the complete PIV loop for primary analysis.
    Reads assembled_prompt + retrieval_stage_2 from BAState.
    Writes analyst_output, analyst_confidence, analyst_citations.

    In production: real Ollama calls via PIVLoopController.
    In tests: mock _call_ollama methods on sub-agents.
    """

    def __init__(
        self,
        planner:     Optional[StrategicPlanner]   = None,
        implementor: Optional[ContextImplementor] = None,
        validator:   Optional[CuriousValidator]   = None,
        max_retries: int = 5,   # A2
    ):
        SeedManager.set_all()
        self.planner     = planner     or StrategicPlanner()
        self.implementor = implementor or ContextImplementor()
        self.validator   = validator   or CuriousValidator()
        self.max_retries = max_retries

        self.piv = PIVLoopController(
            planner     = self.planner,
            implementor = self.implementor,
            validator   = self.validator,
            max_retries = self.max_retries,
        )

    def run(self, state: BAState) -> BAState:
        """
        Main entry point — N11 node.

        Reads:
          state.query
          state.assembled_prompt   (from N10 — context already first C7)
          state.retrieval_stage_2  (top-3 chunks from N09)
          state.query_type
          state.query_difficulty
          state.section_tree       (for section summary)

        Writes:
          state.analyst_output
          state.analyst_confidence
          state.analyst_citations
          state.analyst_attempt_count
          state.analyst_piv_status
          state.low_confidence
        """
        ResourceGovernor.check("N11 Analyst Pod")

        if not state.query:
            print("[N11] No query — skipping analyst pod")
            state.analyst_piv_status = PIVStatus.REJECT
            return state

        # Build retrieved context string from chunks
        retrieved_context = self._build_context(state)

        # Build section summary for Planner
        section_summary = self._build_section_summary(state)

        # Query type and difficulty for Planner routing
        query_type       = (state.query_type or QueryType.TEXT).value
        query_difficulty = (state.query_difficulty or Difficulty.MEDIUM).value

        print(f"[N11] Running PIV loop — "
              f"query_type={query_type} "
              f"difficulty={query_difficulty} "
              f"chunks={len(state.retrieval_stage_2)}")

        # Run PIV loop
        result: PIVResult = self.piv.run(
            query             = state.query,
            retrieved_context = retrieved_context,
            section_summary   = section_summary,
            query_type        = query_type,
            query_difficulty  = query_difficulty,
            pod_role          = "analyst",
            state             = state,
        )

        # Write results to BAState (piv_loop also writes but we ensure here)
        state.analyst_output     = result.answer
        state.analyst_confidence = result.confidence
        state.analyst_citations  = result.citations
        state.analyst_attempt_count = min(result.retries_used, 5)
        state.low_confidence     = result.low_confidence

        if result.low_confidence:
            state.analyst_piv_status = PIVStatus.REJECT
            print(f"[N11] Low confidence after {result.retries_used} retries "
                  f"— HITL triggered")
        else:
            state.analyst_piv_status = PIVStatus.PASS

        print(f"[N11] Complete — "
              f"status={state.analyst_piv_status} "
              f"conf={result.confidence:.2f} "
              f"retries={result.retries_used}")

        return state

    def _build_context(self, state: BAState) -> str:
        """
        Build retrieved context string from BAState chunks.
        Prefers assembled_prompt (already formatted by N10).
        Falls back to raw chunks if no assembled prompt.
        """
        # Use assembled_prompt if available — it already has C7 format
        if state.assembled_prompt:
            return state.assembled_prompt

        # Fall back to raw chunks
        chunks = state.retrieval_stage_2 or state.retrieval_stage_1 or []
        if not chunks:
            return "No context retrieved."

        parts = []
        for i, chunk in enumerate(chunks, 1):
            text    = chunk.get("text") or chunk.get("content") or ""
            section = chunk.get("section", "Unknown Section")
            page    = chunk.get("page", "?")
            parts.append(
                f"--- Source {i}: {section} / Page {page} ---\n{text}"
            )
        return "\n\n".join(parts)

    def _build_section_summary(self, state: BAState) -> str:
        """Build a brief section summary for the Planner from section_tree."""
        if not state.section_tree:
            return (
                f"Company: {state.company_name or 'Unknown'} | "
                f"Doc: {state.doc_type or '10-K'} | "
                f"FY: {state.fiscal_year or 'Unknown'}"
            )

        sections = state.section_tree.get("sections", [])
        names    = [s.get("title", "") for s in sections[:8] if s.get("title")]
        return (
            f"Company: {state.company_name} | "
            f"Sections: {', '.join(names) if names else 'Not available'}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK
# run: python src/agents/analyst_pod.py
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        from rich import print as rprint
    except ImportError:
        rprint = print

    rprint("\n[bold cyan]-- AnalystPod (N11) sanity check --[/bold cyan]")

    pod = AnalystPod()
    rprint("[green]✓[/green] AnalystPod instantiated")

    # Build test state
    state = BAState(
        session_id        = "sanity-n11",
        query             = "What was Apple net income in FY2023?",
        query_type        = QueryType.NUMERICAL,
        query_difficulty  = Difficulty.EASY,
        company_name      = "Apple Inc",
        doc_type          = "10-K",
        fiscal_year       = "FY2023",
        assembled_prompt  = (
            "RETRIEVED CONTEXT:\n"
            "Apple / 10-K / FY2023 / Financial Statements / 42\n"
            "Net income: $96,995 million for fiscal year ended "
            "September 30 2023.\n"
            "Total net sales: $383,285 million.\n\n"
            "QUESTION: What was Apple net income in FY2023?"
        ),
        retrieval_stage_2 = [
            {
                "chunk_id":    "chunk_000001",
                "text":        "Apple / 10-K / FY2023 / Financial Statements / 42\n"
                               "Net income: $96,995 million FY2023.",
                "section":     "Financial Statements",
                "page":        "42",
                "company":     "Apple Inc",
                "doc_type":    "10-K",
                "fiscal_year": "FY2023",
            }
        ],
    )

    state = pod.run(state)

    rprint(f"[green]✓[/green] analyst_piv_status: {state.analyst_piv_status}")
    rprint(f"[green]✓[/green] analyst_confidence: {state.analyst_confidence}")
    rprint(f"[green]✓[/green] analyst_output length: {len(state.analyst_output)}")
    rprint(f"[green]✓[/green] attempt_count: {state.analyst_attempt_count}")
    rprint(f"[green]✓[/green] seed: {state.seed}")

    assert state.analyst_piv_status in [PIVStatus.PASS, PIVStatus.REJECT]
    assert 0.0 <= state.analyst_confidence <= 1.0
    assert state.analyst_output != ""
    assert state.seed == 42

    rprint(f"\n[bold green]All checks passed. AnalystPod N11 ready.[/bold green]\n")