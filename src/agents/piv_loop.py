"""
src/agents/piv_loop.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 Rev1.0 FINAL

PIVLoopController — Agent 4 of PIV Loop
Orchestrates: Planner → Implementor → Validator → retry until PASS

Amendment A1: REJECT goes back to PLANNER (not Implementor)
Amendment A2: max_retries = 5 per pod
Amendment A3: After 5 failures → Clarification Engine fires

Loop behaviour:
  1. Planner runs ONCE — produces plan + curiosity answers
  2. Implementor runs → Validator checks
  3. If VALIDATOR_REJECT → back to PLANNER (A1) with rejection reasons
  4. New Planner run with rejection context → new Implementor run
  5. Repeat until VALIDATOR_PASS or max_retries (5) exhausted
  6. After 5 failures → return low_confidence=True + best attempt
  7. BAState flags low_confidence=True → A3 Clarification Engine

Emotional identity: Calm persistence under pressure.
Never shortcuts. Runs full retry budget.
Transparent — retry_count always visible in output.

Output: PIVResult dataclass with:
  answer          — final answer string
  confidence      — 0.0-1.0 (decayed on retries)
  citations       — list of section/page references
  computation     — formula if ratio, else N/A
  retries_used    — number of retries consumed
  low_confidence  — True if max retries exhausted
  validator_checks— dict V1-V8 final check results
  reject_reasons  — list of final rejection reasons
  planner_output  — last PlannerOutput used
  pod_role        — which pod this is (analyst/quant/auditor)
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.agents.planner     import StrategicPlanner,    PlannerOutput
from src.agents.implementor import ContextImplementor,  ImplementorOutput
from src.agents.validator   import CuriousValidator,    ValidatorOutput
from src.agents.validator   import VALIDATOR_PASS, VALIDATOR_REJECT
from src.state.ba_state     import BAState
from src.utils.seed_manager import SeedManager

SeedManager.set_all()

# ── Config ────────────────────────────────────────────────────────────────────
MAX_RETRIES          = 5     # A2: max per pod
LOW_CONF_THRESHOLD   = 0.65  # below this → HITL review


@dataclass
class PIVResult:
    """Output from PIVLoopController."""
    answer:           str
    confidence:       float
    citations:        List[str]
    computation:      str
    retries_used:     int
    low_confidence:   bool
    validator_checks: Dict[str, str]
    reject_reasons:   List[str]
    planner_output:   Optional[PlannerOutput] = None
    pod_role:         str = "analyst"
    fallback_used:    bool = False


class PIVLoopController:
    """
    PIV Loop Orchestrator — Amendment A1 + A2 + A3.

    A1: REJECT → back to PLANNER (not just Implementor)
        Planner re-runs with rejection context for fresh perspective.
    A2: max_retries = 5
    A3: After 5 failures → low_confidence=True → BAState flags A3

    Emotional identity: Calm persistence.
    Never shortcuts. Runs full 5-attempt budget.
    """

    def __init__(
        self,
        planner:     Optional[StrategicPlanner]   = None,
        implementor: Optional[ContextImplementor] = None,
        validator:   Optional[CuriousValidator]   = None,
        max_retries: int = MAX_RETRIES,
    ):
        SeedManager.set_all()
        self.planner     = planner     or StrategicPlanner()
        self.implementor = implementor or ContextImplementor()
        self.validator   = validator   or CuriousValidator()
        self.max_retries = max_retries

    def run(
        self,
        query:             str,
        retrieved_context: str,
        section_summary:   str = "",
        query_type:        str = "text",
        query_difficulty:  str = "medium",
        pod_role:          str = "analyst",
        state:             Optional[BAState] = None,
    ) -> PIVResult:
        """
        Run PIV loop until VALIDATOR_PASS or max_retries exhausted.

        A1: Each REJECT → Planner re-runs with rejection context.
        A2: Maximum 5 retries.
        A3: Exhausted → low_confidence=True.

        Returns PIVResult with final answer + all metadata.
        """
        SeedManager.set_all()

        best_attempt:      Optional[ImplementorOutput] = None
        best_validator:    Optional[ValidatorOutput]   = None
        last_planner:      Optional[PlannerOutput]     = None
        retry_instructions = ""
        rejection_context  = ""

        for attempt in range(self.max_retries + 1):

            # ── STEP 1: PLANNER (A1 — runs on every attempt) ──────────────
            # On first attempt: fresh plan
            # On retries: plan WITH rejection context (A1 key behaviour)
            planner_query = query
            if attempt > 0 and rejection_context:
                planner_query = (
                    f"{query}\n\n"
                    f"[PREVIOUS ATTEMPT FAILED — Validator rejected because: "
                    f"{rejection_context}. "
                    f"Please adjust your analysis plan to address these issues.]"
                )

            planner_out = self.planner.run(
                query            = planner_query,
                section_summary  = section_summary,
                query_type       = query_type,
                query_difficulty = query_difficulty,
            )
            last_planner = planner_out

            # ── STEP 2: IMPLEMENTOR ────────────────────────────────────────
            impl_out = self.implementor.run(
                query               = query,
                retrieved_context   = retrieved_context,
                analysis_plan       = planner_out.analysis_plan,
                validation_criteria = planner_out.validation_criteria,
                retry_count         = attempt,
                retry_instructions  = retry_instructions,
            )

            # Track best attempt (highest confidence)
            if (best_attempt is None or
                    impl_out.confidence > best_attempt.confidence):
                best_attempt = impl_out

            # Handle RETRIEVAL_MISS
            if impl_out.output_type == "RETRIEVAL_MISS":
                print(f"[PIV attempt {attempt}] RETRIEVAL_MISS: "
                      f"{impl_out.needed_info}")
                # Still run validator to record the attempt
                val_out = self.validator.run(
                    query               = query,
                    implementor_answer  = (
                        f"RETRIEVAL_MISS: {impl_out.needed_info}"
                    ),
                    retrieved_context   = retrieved_context,
                    validation_criteria = planner_out.validation_criteria,
                    retry_count         = attempt,
                )
            else:
                # ── STEP 3: VALIDATOR ─────────────────────────────────────
                val_out = self.validator.run(
                    query               = query,
                    implementor_answer  = impl_out.answer,
                    retrieved_context   = retrieved_context,
                    validation_criteria = planner_out.validation_criteria,
                    retry_count         = attempt,
                )

            best_validator = val_out

            print(f"[PIV attempt {attempt}] "
                  f"conf={impl_out.confidence:.2f} "
                  f"verdict={val_out.result}")

            # ── STEP 4: CHECK RESULT ───────────────────────────────────────
            if val_out.result == VALIDATOR_PASS:
                # Update BAState if provided
                if state is not None:
                    self._update_state(
                        state, pod_role, impl_out, val_out, attempt
                    )
                return PIVResult(
                    answer           = impl_out.answer,
                    confidence       = impl_out.confidence,
                    citations        = impl_out.citations,
                    computation      = impl_out.computation,
                    retries_used     = attempt,
                    low_confidence   = impl_out.confidence < LOW_CONF_THRESHOLD,
                    validator_checks = val_out.checks,
                    reject_reasons   = [],
                    planner_output   = last_planner,
                    pod_role         = pod_role,
                    fallback_used    = impl_out.fallback_used,
                )

            # ── STEP 5: REJECT → set up next attempt ──────────────────────
            rejection_context  = "; ".join(val_out.reject_reasons)
            retry_instructions = val_out.retry_instructions

        # ── MAX RETRIES EXHAUSTED (A3) ─────────────────────────────────────
        print(f"[PIV] Max retries ({self.max_retries}) exhausted — "
              f"low_confidence=True")

        # Use best attempt found
        final_impl = best_attempt
        final_val  = best_validator

        if final_impl is None:
            # Should never happen — safety fallback
            final_impl = ImplementorOutput(
                answer        = "Unable to answer — retrieval insufficient.",
                confidence    = 0.0,
                citations     = [],
                computation   = "N/A",
                output_type   = "ANSWER",
                fallback_used = True,
            )

        # Apply final confidence penalty (A3)
        final_conf = final_impl.confidence * 0.6

        if state is not None:
            self._update_state(
                state, pod_role, final_impl, final_val, self.max_retries
            )
            state.low_confidence = True

        return PIVResult(
            answer           = final_impl.answer,
            confidence       = round(final_conf, 4),
            citations        = final_impl.citations,
            computation      = final_impl.computation,
            retries_used     = self.max_retries,
            low_confidence   = True,
            validator_checks = final_val.checks if final_val else {},
            reject_reasons   = final_val.reject_reasons if final_val else [],
            planner_output   = last_planner,
            pod_role         = pod_role,
            fallback_used    = final_impl.fallback_used,
        )

    def _update_state(
        self,
        state:    BAState,
        pod_role: str,
        impl_out: ImplementorOutput,
        val_out:  Optional[ValidatorOutput],
        attempt:  int,
    ) -> None:
        """Write PIV results to the correct pod fields in BAState."""
        if pod_role == "analyst":
            state.analyst_output           = impl_out.answer
            state.analyst_confidence       = impl_out.confidence
            state.analyst_citations        = impl_out.citations
            state.analyst_attempt_count    = min(attempt, 5)
            if val_out:
                state.analyst_piv_status   = (
                    "PASS" if val_out.result == VALIDATOR_PASS else "REJECT"  # type: ignore
                )
        elif pod_role == "quant":
            state.quant_result             = impl_out.answer
            state.quant_confidence         = impl_out.confidence
            state.quant_citations          = impl_out.citations
            state.quant_attempt_count      = min(attempt, 5)
            if val_out:
                state.quant_piv_status     = (
                    "PASS" if val_out.result == VALIDATOR_PASS else "REJECT"  # type: ignore
                )
        elif pod_role == "auditor":
            state.auditor_output           = impl_out.answer
            state.auditor_confidence       = impl_out.confidence
            state.auditor_citations        = impl_out.citations
            state.auditor_attempt_count    = min(attempt, 5)
            if val_out:
                state.auditor_piv_status   = (
                    "PASS" if val_out.result == VALIDATOR_PASS else "REJECT"  # type: ignore
                )


# ═══════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK
# run: python src/agents/piv_loop.py
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        from rich import print as rprint
    except ImportError:
        rprint = print

    rprint("\n[bold cyan]-- PIVLoopController sanity check --[/bold cyan]")

    piv = PIVLoopController()
    rprint("[green]✓[/green] PIVLoopController instantiated")
    rprint(f"[green]✓[/green] max_retries={piv.max_retries} (A2)")

    context = """Apple / 10-K / FY2023 / Financial Statements / 42
Net income: $96,995 million for fiscal year ended September 30 2023.
Total net sales: $383,285 million."""

    result = piv.run(
        query             = "What was Apple net income in FY2023?",
        retrieved_context = context,
        section_summary   = "Financial Statements, MD&A, Notes",
        query_type        = "numerical",
        query_difficulty  = "easy",
        pod_role          = "analyst",
    )

    rprint(f"[green]✓[/green] PIV result:")
    rprint(f"  answer length:    {len(result.answer)} chars")
    rprint(f"  confidence:       {result.confidence}")
    rprint(f"  retries_used:     {result.retries_used}")
    rprint(f"  low_confidence:   {result.low_confidence}")
    rprint(f"  validator_checks: {len(result.validator_checks)} checks")
    rprint(f"  pod_role:         {result.pod_role}")

    assert isinstance(result.answer,       str)
    assert 0.0 <= result.confidence <= 1.0
    assert result.retries_used     <= MAX_RETRIES
    assert result.pod_role         == "analyst"
    assert isinstance(result.validator_checks, dict)

    # BAState integration
    state = BAState(session_id="piv-test")
    result2 = piv.run(
        query             = "What was Apple net income?",
        retrieved_context = context,
        pod_role          = "analyst",
        state             = state,
    )
    assert state.analyst_output != "" or result2.fallback_used
    assert state.seed == 42
    rprint(f"[green]✓[/green] BAState updated: "
           f"analyst_output length={len(state.analyst_output)}")

    rprint(f"\n[bold green]All checks passed. PIVLoopController ready.[/bold green]\n")