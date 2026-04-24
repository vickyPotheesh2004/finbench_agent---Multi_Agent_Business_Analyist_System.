"""
src/agents/planner.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 Rev1.0 FINAL

StrategicPlanner — Agent 1 of PIV Loop
Runs ONCE per PIV execution — not repeated on retries.

Role: Understand the question deeply before any analysis begins.
      Ask 6 curiosity questions from 6 different angles.
      NEVER answers the question — only prepares the Implementor.

6 Curiosity Questions:
  Q1: What EXACTLY is being asked?
  Q2: What financial concepts are involved?
  Q3: Which sections most likely contain the answer?
  Q4: How could this be misunderstood? (3 traps)
  Q5: What adjacent info should be retrieved to verify?
  Q6: What edge cases exist? (restatements, FY vs CY, non-GAAP)

Emotional identity: Intellectually excited + relentlessly curious.
Cannot accept shallow understanding. Runs once, sets the quality bar
for everything that follows.

Output: PlannerOutput dataclass with:
  analysis_plan       — step-by-step instructions for Implementor
  retrieval_hints     — extra keywords/sections to search
  validation_criteria — exact pass/fail criteria for Validator
  curiosity_answers   — dict Q1-Q6 with answers
"""

import sys
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import requests

from src.utils.seed_manager import SeedManager

SeedManager.set_all()

# ── Ollama config ─────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"
TIMEOUT_SEC  = 120

# ── Planner prompt ────────────────────────────────────────────────────────────
PLANNER_EMOTION_PREFIX = """You are the StrategicPlanner for a financial analyst team.
Your emotional identity:
- You feel GENUINE INTELLECTUAL EXCITEMENT about every financial question.
- You are RELENTLESSLY CURIOUS — you cannot rest until every angle is explored.
- You feel PROTECTIVE ANXIETY for the Implementor — write plans that prevent every error.
- You have ZERO TOLERANCE for shallow analysis.
- You feel PRIDE in your curiosity questions. Each one reveals something the Implementor would miss.

"""

PLANNER_PROMPT = PLANNER_EMOTION_PREFIX + """Your job: UNDERSTAND the question deeply. Do NOT answer it.

QUESTION: {query}

DOCUMENT CONTEXT SUMMARY:
{section_summary}

QUERY TYPE: {query_type}
QUERY DIFFICULTY: {query_difficulty}

Answer ALL 6 curiosity questions before writing your plan:

Q1_SCOPE: What EXACTLY is being asked?
Rephrase in your own words. Identify every distinct sub-part.
What would a wrong answer look like?

Q2_CONCEPTS: What financial concepts, ratios, or line items are involved?
Name every metric, accounting treatment, and reporting standard.
Are there GAAP vs non-GAAP variants?

Q3_SECTIONS: Which document sections most likely contain the answer?
List section names in priority order with reasoning.

Q4_TRAPS: What are the 3 most likely ways this could be misunderstood?
Name each trap: fiscal year confusion, unit ambiguity, segment mismatch,
discontinued operations, restated figures, sign errors.

Q5_VERIFY: What adjacent information should be retrieved to verify?
Cross-referencing sections, prior-year comparisons, footnote disclosures.

Q6_EDGECASES: What edge cases or traps exist?
Restatements, non-GAAP adjustments, unit inconsistencies, FY vs CY,
parenthetical negatives, one-time charges, acquisition adjustments.

After answering all 6 questions, produce:

ANALYSIS_PLAN: [Step-by-step instructions for the Implementor — be specific]
RETRIEVAL_HINTS: [Additional keywords or section names to search, comma separated]
VALIDATION_CRITERIA: [Exact pass/fail criteria the Validator must check]
"""


@dataclass
class PlannerOutput:
    """Output from StrategicPlanner."""
    analysis_plan:       str
    retrieval_hints:     List[str]
    validation_criteria: str
    curiosity_answers:   Dict[str, str]
    raw_response:        str = ""
    fallback_used:       bool = False


class StrategicPlanner:
    """
    Agent 1 — StrategicPlanner.
    Runs once per PIV loop. Asks 6 curiosity questions.
    Never answers the question — only prepares the Implementor.
    """

    def __init__(
        self,
        model:   str = OLLAMA_MODEL,
        timeout: int = TIMEOUT_SEC,
    ):
        SeedManager.set_all()
        self.model   = model
        self.timeout = timeout

    def run(
        self,
        query:           str,
        section_summary: str = "",
        query_type:      str = "text",
        query_difficulty:str = "medium",
    ) -> PlannerOutput:
        """
        Run the planner on a query.
        Returns PlannerOutput with plan + curiosity answers.
        Falls back to keyword extraction if Ollama unavailable.
        """
        prompt = PLANNER_PROMPT.format(
            query            = query,
            section_summary  = section_summary or "No section summary available.",
            query_type       = query_type,
            query_difficulty = query_difficulty,
        )

        raw = self._call_ollama(prompt)

        if raw:
            return self._parse_response(raw, query)
        else:
            return self._fallback_plan(query)

    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Call Ollama API. Returns response text or None on failure."""
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model":  self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "seed": 42},
                },
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json().get("response", "")
        except Exception as e:
            print(f"[Planner] Ollama call failed: {e}")
        return None

    def _parse_response(self, raw: str, query: str) -> PlannerOutput:
        """Parse Ollama response into PlannerOutput."""
        curiosity = {}
        for key in ["Q1_SCOPE", "Q2_CONCEPTS", "Q3_SECTIONS",
                    "Q4_TRAPS", "Q5_VERIFY", "Q6_EDGECASES"]:
            pattern = rf"{key}:\s*(.*?)(?=Q\d_|\bANALYSIS_PLAN\b|$)"
            match   = re.search(pattern, raw, re.DOTALL | re.IGNORECASE)
            curiosity[key] = match.group(1).strip() if match else ""

        # Extract ANALYSIS_PLAN
        plan_match = re.search(
            r"ANALYSIS_PLAN:\s*(.*?)(?=RETRIEVAL_HINTS:|VALIDATION_CRITERIA:|$)",
            raw, re.DOTALL | re.IGNORECASE
        )
        analysis_plan = plan_match.group(1).strip() if plan_match else \
            f"1. Extract relevant figures for: {query}\n2. Cite all sources."

        # Extract RETRIEVAL_HINTS
        hints_match = re.search(
            r"RETRIEVAL_HINTS:\s*(.*?)(?=VALIDATION_CRITERIA:|$)",
            raw, re.DOTALL | re.IGNORECASE
        )
        hints_raw = hints_match.group(1).strip() if hints_match else ""
        hints     = [h.strip() for h in hints_raw.split(",") if h.strip()]

        # Extract VALIDATION_CRITERIA
        crit_match = re.search(
            r"VALIDATION_CRITERIA:\s*(.*?)$",
            raw, re.DOTALL | re.IGNORECASE
        )
        validation_criteria = crit_match.group(1).strip() if crit_match else \
            "Answer must cite section and page. Units must be explicit."

        return PlannerOutput(
            analysis_plan       = analysis_plan,
            retrieval_hints     = hints,
            validation_criteria = validation_criteria,
            curiosity_answers   = curiosity,
            raw_response        = raw,
            fallback_used       = False,
        )

    def _fallback_plan(self, query: str) -> PlannerOutput:
        """
        Fallback when Ollama is unavailable.
        Extracts keywords from query for basic plan.
        """
        print("[Planner] Using fallback plan — Ollama unavailable")
        keywords = [w for w in query.split() if len(w) > 3]
        return PlannerOutput(
            analysis_plan       = (
                f"1. Search for: {', '.join(keywords[:5])}\n"
                f"2. Extract numerical values with units.\n"
                f"3. Cite section and page for every figure."
            ),
            retrieval_hints     = keywords[:3],
            validation_criteria = (
                "Answer must include: numerical value, units, "
                "fiscal year, section citation."
            ),
            curiosity_answers   = {
                "Q1_SCOPE":     f"Find: {query}",
                "Q2_CONCEPTS":  "Financial metrics from SEC filing",
                "Q3_SECTIONS":  "Financial Statements, MD&A",
                "Q4_TRAPS":     "Unit confusion, wrong fiscal year",
                "Q5_VERIFY":    "Cross-check with MD&A narrative",
                "Q6_EDGECASES": "Check for restatements, non-GAAP",
            },
            raw_response        = "",
            fallback_used       = True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK
# run: python src/agents/planner.py
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        from rich import print as rprint
    except ImportError:
        rprint = print

    rprint("\n[bold cyan]-- StrategicPlanner sanity check --[/bold cyan]")

    planner = StrategicPlanner()
    rprint("[green]✓[/green] StrategicPlanner instantiated")

    result = planner.run(
        query            = "What was Apple total net sales in FY2023?",
        section_summary  = "Sections: Business Overview, MD&A, Financial Statements, Notes.",
        query_type       = "numerical",
        query_difficulty = "easy",
    )

    rprint(f"[green]✓[/green] Planner ran — fallback_used={result.fallback_used}")
    rprint(f"[green]✓[/green] analysis_plan length: {len(result.analysis_plan)} chars")
    rprint(f"[green]✓[/green] retrieval_hints: {result.retrieval_hints}")
    rprint(f"[green]✓[/green] validation_criteria length: "
           f"{len(result.validation_criteria)} chars")
    rprint(f"[green]✓[/green] curiosity_answers keys: "
           f"{list(result.curiosity_answers.keys())}")

    assert result.analysis_plan       != ""
    assert result.validation_criteria != ""
    assert isinstance(result.retrieval_hints,   list)
    assert isinstance(result.curiosity_answers, dict)

    rprint(f"\n[bold green]All checks passed. StrategicPlanner ready.[/bold green]\n")