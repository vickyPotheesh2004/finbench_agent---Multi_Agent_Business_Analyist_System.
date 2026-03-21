"""
src/agents/quant_pod.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 Rev1.0 FINAL

N12 — CFO/Quant Pod
Formula-first quantitative specialist.
Runs in parallel with N11 Analyst Pod.

Specialisation vs N11:
  - Formula-first: shows every computation step explicitly
  - Monte Carlo: 10,000 scenario simulation for uncertainty bounds
  - Historical VaR: 95th/99th percentile risk computation
  - Ratio computation: explicit numerator/denominator with citations
  - Writes to: quant_result, quant_confidence, quant_citations

Reuses: PIVLoopController (same Planner + Implementor + Validator)
Different: quantitative system prompt injected into Implementor

Emotional identity: CFO/Quant analyst
  Primary:  Numerical precision — every figure must be exact
  Secondary: Formula discipline — never skip computation steps
  Tertiary:  Quantitative honesty — state confidence intervals

Monte Carlo:
  - 10,000 scenarios via numpy (seed=42)
  - Returns: mean, std, p5, p50, p95 percentiles
  - Used for: ratio uncertainty, projection bounds

Historical VaR:
  - 95th and 99th percentile
  - Used for: risk quantification on financial ratios

GARCH(1,1) via arch library:
  - Temporal volatility on time series data
  - Minimum 2 data points required
  - Falls back gracefully if insufficient data
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy import stats as scipy_stats

from src.agents.planner      import StrategicPlanner
from src.agents.implementor  import ContextImplementor
from src.agents.validator    import CuriousValidator
from src.agents.piv_loop     import PIVLoopController, PIVResult
from src.state.ba_state      import BAState, QueryType, Difficulty, PIVStatus
from src.utils.seed_manager  import SeedManager
from src.utils.resource_governor import ResourceGovernor

SeedManager.set_all()

# ── Monte Carlo config ────────────────────────────────────────────────────────
MC_SCENARIOS  = 10_000
MC_SEED       = 42       # C5


@dataclass
class QuantResult:
    """Extended quantitative result with statistical measures."""
    answer:              str
    confidence:          float
    citations:           List[str]
    computation:         str
    monte_carlo:         Optional[Dict[str, float]] = None
    var_95:              Optional[float]            = None
    var_99:              Optional[float]            = None
    garch_volatility:    Optional[float]            = None
    retries_used:        int                        = 0
    low_confidence:      bool                       = False
    fallback_used:       bool                       = False


class QuantPod:
    """
    N12 — CFO/Quant Pod.

    Formula-first quantitative analysis.
    Reuses PIVLoopController with quantitative specialisation.
    Adds Monte Carlo, VaR, and GARCH on top of PIV answer.

    Writes to BAState:
      quant_result, quant_confidence, quant_citations
      quant_attempt_count, quant_piv_status
      monte_carlo_results, var_result, garch_result
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

        self.piv = PIVLoopController(
            planner     = self.planner,
            implementor = self.implementor,
            validator   = self.validator,
            max_retries = self.max_retries,
        )

    def run(self, state: BAState) -> BAState:
        """
        Main entry point — N12 node.

        Reads:  state.query, state.assembled_prompt,
                state.retrieval_stage_2, state.query_type
        Writes: state.quant_result, state.quant_confidence,
                state.quant_citations, state.quant_piv_status,
                state.monte_carlo_results, state.var_result
        """
        ResourceGovernor.check("N12 Quant Pod")

        if not state.query:
            print("[N12] No query — skipping quant pod")
            state.quant_piv_status = PIVStatus.REJECT
            return state

        # Build context string
        retrieved_context = self._build_context(state)
        section_summary   = self._build_section_summary(state)
        query_type        = (state.query_type or QueryType.RATIO).value
        query_difficulty  = (state.query_difficulty or Difficulty.MEDIUM).value

        print(f"[N12] Running Quant PIV loop — "
              f"query_type={query_type} "
              f"chunks={len(state.retrieval_stage_2)}")

        # Run PIV loop with quant pod role
        result: PIVResult = self.piv.run(
            query             = state.query,
            retrieved_context = retrieved_context,
            section_summary   = section_summary,
            query_type        = query_type,
            query_difficulty  = query_difficulty,
            pod_role          = "quant",
            state             = state,
        )

        # Write core results to BAState
        state.quant_result      = result.answer
        state.quant_confidence  = result.confidence
        state.quant_citations   = result.citations
        state.quant_attempt_count = min(result.retries_used, 5)
        state.low_confidence    = result.low_confidence

        # Run quantitative analysis on top of PIV answer
        quant_result = self._run_quantitative_analysis(
            state  = state,
            answer = result.answer,
        )

        # Write quantitative metrics to BAState
        if quant_result.monte_carlo:
            state.monte_carlo_results = quant_result.monte_carlo
        if quant_result.var_95 is not None:
            state.var_result = {
                "var_95": quant_result.var_95,
                "var_99": quant_result.var_99,
            }

        if result.low_confidence:
            state.quant_piv_status = PIVStatus.REJECT
            print(f"[N12] Low confidence — HITL triggered")
        else:
            state.quant_piv_status = PIVStatus.PASS

        print(f"[N12] Complete — "
              f"status={state.quant_piv_status} "
              f"conf={result.confidence:.2f} "
              f"retries={result.retries_used}")

        return state

    # ═══════════════════════════════════════════════════════════════════════
    # QUANTITATIVE ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════

    def _run_quantitative_analysis(
        self,
        state:  BAState,
        answer: str,
    ) -> QuantResult:
        """
        Run quantitative methods on top of PIV answer.
        Extracts numbers from answer and computes statistical measures.
        """
        # Extract numerical values from answer and context
        numbers = self._extract_numbers(answer)
        context_numbers = self._extract_numbers(
            " ".join(
                c.get("text", "") for c in state.retrieval_stage_2
            )
        )

        result = QuantResult(
            answer     = answer,
            confidence = state.quant_confidence,
            citations  = state.quant_citations,
            computation= "Quantitative analysis applied.",
        )

        # Monte Carlo — run if we have a primary number
        if numbers:
            primary_value = numbers[0]
            mc = self.run_monte_carlo(
                base_value    = primary_value,
                uncertainty   = primary_value * 0.05,  # 5% uncertainty
                n_scenarios   = MC_SCENARIOS,
            )
            result.monte_carlo = mc

        # Historical VaR — run if we have multiple numbers
        if len(context_numbers) >= 3:
            var_95, var_99 = self.compute_var(context_numbers)
            result.var_95  = var_95
            result.var_99  = var_99

        # GARCH — run if we have time series (2+ data points)
        if len(context_numbers) >= 2:
            result.garch_volatility = self.compute_garch_volatility(
                context_numbers
            )

        return result

    def run_monte_carlo(
        self,
        base_value:  float,
        uncertainty: float,
        n_scenarios: int = MC_SCENARIOS,
    ) -> Dict[str, float]:
        """
        Monte Carlo simulation — 10,000 scenarios.
        Returns: mean, std, p5, p25, p50, p75, p95.
        seed=42 for reproducibility (C5).
        """
        SeedManager.set_all()
        rng       = np.random.default_rng(MC_SEED)
        scenarios = rng.normal(
            loc   = base_value,
            scale = max(uncertainty, abs(base_value) * 0.01),
            size  = n_scenarios,
        )
        return {
            "mean":  float(round(np.mean(scenarios),   4)),
            "std":   float(round(np.std(scenarios),    4)),
            "p5":    float(round(np.percentile(scenarios,  5), 4)),
            "p25":   float(round(np.percentile(scenarios, 25), 4)),
            "p50":   float(round(np.percentile(scenarios, 50), 4)),
            "p75":   float(round(np.percentile(scenarios, 75), 4)),
            "p95":   float(round(np.percentile(scenarios, 95), 4)),
            "n":     n_scenarios,
        }

    def compute_var(
        self,
        values: List[float],
        confidence_95: float = 0.95,
        confidence_99: float = 0.99,
    ) -> tuple:
        """
        Historical VaR at 95th and 99th percentile.
        Returns (var_95, var_99) as negative losses.
        Requires at least 3 data points.
        """
        if len(values) < 3:
            return None, None

        arr     = np.array(values, dtype=float)
        var_95  = float(np.percentile(arr, (1 - confidence_95) * 100))
        var_99  = float(np.percentile(arr, (1 - confidence_99) * 100))
        return round(var_95, 4), round(var_99, 4)

    def compute_garch_volatility(
        self,
        values: List[float],
    ) -> Optional[float]:
        """
        GARCH(1,1) volatility estimate.
        Falls back to simple std dev if arch library unavailable
        or insufficient data points.
        Minimum 2 data points required.
        """
        if len(values) < 2:
            return None

        try:
            from arch import arch_model
            SeedManager.set_all()
            arr     = np.array(values, dtype=float)
            returns = np.diff(arr) / arr[:-1] * 100   # percentage returns
            if len(returns) < 2:
                return float(round(np.std(arr), 4))
            model   = arch_model(returns, vol="Garch", p=1, q=1, rescale=False)
            res     = model.fit(disp="off", show_warning=False)
            vol     = float(round(res.conditional_volatility[-1], 4))
            return vol
        except Exception:
            # Fallback: simple standard deviation
            return float(round(np.std(values), 4))

    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numerical values from text. Returns list of floats."""
        import re
        # Match numbers with optional commas and decimal points
        pattern = r'[\$]?(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)'
        matches = re.findall(pattern, text)
        numbers = []
        for m in matches:
            try:
                val = float(m.replace(",", ""))
                if val > 0:
                    numbers.append(val)
            except ValueError:
                continue
        return numbers[:10]   # cap at 10 to avoid noise

    def _build_context(self, state: BAState) -> str:
        """Build retrieved context from BAState."""
        if state.assembled_prompt:
            return state.assembled_prompt
        chunks = state.retrieval_stage_2 or state.retrieval_stage_1 or []
        if not chunks:
            return "No context retrieved."
        parts = []
        for i, chunk in enumerate(chunks, 1):
            text    = chunk.get("text") or chunk.get("content") or ""
            section = chunk.get("section", "Unknown")
            page    = chunk.get("page",    "?")
            parts.append(f"--- Source {i}: {section} / Page {page} ---\n{text}")
        return "\n\n".join(parts)

    def _build_section_summary(self, state: BAState) -> str:
        """Build section summary for Planner."""
        return (
            f"Company: {state.company_name or 'Unknown'} | "
            f"Doc: {state.doc_type or '10-K'} | "
            f"FY: {state.fiscal_year or 'Unknown'}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK
# run: python src/agents/quant_pod.py
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        from rich import print as rprint
    except ImportError:
        rprint = print

    rprint("\n[bold cyan]-- QuantPod (N12) sanity check --[/bold cyan]")

    pod = QuantPod()
    rprint("[green]✓[/green] QuantPod instantiated")

    # Test Monte Carlo
    mc = pod.run_monte_carlo(base_value=96995.0, uncertainty=4849.75)
    assert "mean" in mc and "p5" in mc and "p95" in mc
    assert mc["n"] == 10_000
    assert mc["p5"] < mc["mean"] < mc["p95"]
    rprint(f"[green]✓[/green] Monte Carlo: mean={mc['mean']:.0f} "
           f"p5={mc['p5']:.0f} p95={mc['p95']:.0f} n={mc['n']}")

    # Test VaR
    values = [96995.0, 99803.0, 57411.0, 94680.0, 94321.0]
    var_95, var_99 = pod.compute_var(values)
    assert var_95 is not None
    assert var_99 <= var_95
    rprint(f"[green]✓[/green] VaR: var_95={var_95:.0f} var_99={var_99:.0f}")

    # Test GARCH
    vol = pod.compute_garch_volatility(values)
    assert vol is not None
    assert vol >= 0.0
    rprint(f"[green]✓[/green] GARCH volatility: {vol:.4f}")

    # Test number extraction
    nums = pod._extract_numbers(
        "Net income was $96,995 million. Revenue was $383,285 million."
    )
    assert len(nums) > 0
    rprint(f"[green]✓[/green] Number extraction: {nums}")

    # Full BAState run
    state = BAState(
        session_id        = "sanity-n12",
        query             = "What was Apple gross margin percentage FY2023?",
        query_type        = QueryType.RATIO,
        query_difficulty  = Difficulty.MEDIUM,
        company_name      = "Apple Inc",
        doc_type          = "10-K",
        fiscal_year       = "FY2023",
        assembled_prompt  = (
            "RETRIEVED CONTEXT:\n"
            "Apple / 10-K / FY2023 / Financial Statements / 42\n"
            "Gross profit: $169,148 million. Net sales: $383,285 million.\n"
            "QUESTION: What was Apple gross margin percentage FY2023?"
        ),
        retrieval_stage_2 = [{
            "text":        "Gross profit: $169,148 million. Net sales: $383,285 million.",
            "section":     "Financial Statements",
            "page":        "42",
            "company":     "Apple Inc",
            "doc_type":    "10-K",
            "fiscal_year": "FY2023",
        }],
    )

    state = pod.run(state)

    rprint(f"[green]✓[/green] quant_piv_status: {state.quant_piv_status}")
    rprint(f"[green]✓[/green] quant_confidence: {state.quant_confidence}")
    rprint(f"[green]✓[/green] quant_result length: {len(state.quant_result)}")
    rprint(f"[green]✓[/green] monte_carlo_results: {state.monte_carlo_results}")
    rprint(f"[green]✓[/green] seed: {state.seed}")

    assert state.quant_piv_status in [PIVStatus.PASS, PIVStatus.REJECT]
    assert 0.0 <= state.quant_confidence <= 1.0
    assert state.seed == 42

    rprint(f"\n[bold green]All checks passed. QuantPod N12 ready.[/bold green]\n")