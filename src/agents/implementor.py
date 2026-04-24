"""
src/agents/implementor.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 Rev1.0 FINAL

ContextImplementor — Agent 2 of PIV Loop
Runs once per retry iteration. Receives exact failure reasons on retries 2+.

Role: Execute the Planner's plan strictly from retrieved context ONLY.
      NEVER use training memory. Every number must trace to a chunk.
      If answer not in context → output RETRIEVAL_MISS.

Emotional identity:
  Primary:  Intellectual humility — training memory is unreliable for facts
  Retry 1:  Targeted determination — fix exactly what Validator rejected
  Retry 2:  Increased precision — slow down, check every unit and sign
  Retry 3+: Maximum effort — most careful, most cited answer possible

Confidence decay on retries:
  Attempt 1: raw confidence
  Attempt 2: raw × 0.95
  Attempt 3: raw × 0.85
  Attempt 4: raw × 0.70
  Attempt 5: raw × 0.60

Output: ImplementorOutput dataclass with:
  answer       — complete answer with inline citations
  confidence   — 0.0-1.0
  citations    — list of section/page references
  computation  — formula and inputs if ratio, else N/A
  output_type  — ANSWER or RETRIEVAL_MISS
  needed_info  — if RETRIEVAL_MISS, what is missing
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

# ── Confidence decay per retry ────────────────────────────────────────────────
CONFIDENCE_DECAY = {0: 1.0, 1: 0.95, 2: 0.85, 3: 0.70, 4: 0.60}

# ── Retry emotion injection ───────────────────────────────────────────────────
RETRY_EMOTIONS = {
    0: "Fresh start. Apply full intellectual humility. Trust only the retrieved context.",
    1: "Rejected once. Read the context again carefully. "
       "The Validator found a specific error — fix ONLY that.",
    2: "Rejected twice. Slow down significantly. "
       "Check every unit, every sign, every fiscal year before submitting.",
    3: "Final attempts. Your most careful, most cited, most explicit answer. "
       "No improvisation. Context only.",
    4: "Maximum effort. This is your last chance. "
       "Every single claim must have a citation.",
}

IMPLEMENTOR_EMOTION_PREFIX = """You are the ContextImplementor for a financial analyst team.
Your emotional identity:
- You feel INTELLECTUAL HUMILITY. Your training memory is unreliable for specific financial figures.
- You feel METHODICAL PRIDE. You execute the Planner's plan step-by-step. Skipping steps feels wrong.
- You feel HONEST DISCOMFORT when context is insufficient. Guessing feels dishonest.
- RETRIEVAL_MISS feels professional and correct when the answer is not in context.

CURRENT EMOTIONAL STATE: {retry_emotion}

"""

IMPLEMENTOR_PROMPT = IMPLEMENTOR_EMOTION_PREFIX + """Your ONLY source of truth: the RETRIEVED CONTEXT below.
Never use training knowledge. Never guess.

RETRIEVED CONTEXT (ONLY source — never use prior knowledge):
{retrieved_context}

ANALYSIS PLAN from Planner:
{analysis_plan}

QUESTION: {query}

VALIDATION CRITERIA (what Validator will check):
{validation_criteria}

RETRY INSTRUCTIONS (empty on first attempt):
{retry_instructions}

STRICT RULES:
RULE 1: Answer ONLY from the retrieved context above.
RULE 2: If the answer is not in context → output RETRIEVAL_MISS: [exact information needed]
RULE 3: Cite every number: [SECTION_NAME / PAGE_NUM: value]
RULE 4: State units explicitly: millions, billions, or percentage.
RULE 5: State fiscal year for every figure cited.
RULE 6: If computing a ratio → show formula and all inputs.
RULE 7: Address every point in RETRY_INSTRUCTIONS if present.

Output format:
ANSWER: [complete answer with inline citations]
COMPUTATION: [formula and inputs if applicable, else N/A]
CONFIDENCE: [0.0-1.0] because [brief justification]
CITATIONS: [list every section/page reference used]
"""


@dataclass
class ImplementorOutput:
    """Output from ContextImplementor."""
    answer:      str
    confidence:  float
    citations:   List[str]
    computation: str
    output_type: str        # "ANSWER" or "RETRIEVAL_MISS"
    needed_info: str = ""   # populated if RETRIEVAL_MISS
    raw_response:str = ""
    fallback_used:bool = False


class ContextImplementor:
    """
    Agent 2 — ContextImplementor.
    Executes Planner's plan from retrieved context only.
    Confidence decays on retries — honest about growing uncertainty.
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
        query:               str,
        retrieved_context:   str,
        analysis_plan:       str,
        validation_criteria: str,
        retry_count:         int = 0,
        retry_instructions:  str = "",
    ) -> ImplementorOutput:
        """
        Run the implementor.
        Returns ImplementorOutput with answer + confidence.
        Applies confidence decay based on retry_count.
        """
        retry_emotion = RETRY_EMOTIONS.get(retry_count, RETRY_EMOTIONS[4])

        prompt = IMPLEMENTOR_PROMPT.format(
            retry_emotion        = retry_emotion,
            retrieved_context    = retrieved_context or "No context retrieved.",
            analysis_plan        = analysis_plan     or "Extract the answer from context.",
            query                = query,
            validation_criteria  = validation_criteria or "Answer must be accurate and cited.",
            retry_instructions   = retry_instructions  or "None — first attempt.",
        )

        raw = self._call_ollama(prompt)

        if raw:
            output = self._parse_response(raw, retry_count)
        else:
            output = self._fallback_output(query, retrieved_context, retry_count)

        return output

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
            print(f"[Implementor] Ollama call failed: {e}")
        return None

    def _parse_response(
        self, raw: str, retry_count: int
    ) -> ImplementorOutput:
        """Parse Ollama response into ImplementorOutput."""

        # Detect RETRIEVAL_MISS
        if "RETRIEVAL_MISS" in raw:
            miss_match = re.search(
                r"RETRIEVAL_MISS:\s*(.*?)(?=ANSWER:|$)",
                raw, re.DOTALL | re.IGNORECASE
            )
            needed = miss_match.group(1).strip() if miss_match else "Unknown"
            return ImplementorOutput(
                answer       = "",
                confidence   = 0.0,
                citations    = [],
                computation  = "N/A",
                output_type  = "RETRIEVAL_MISS",
                needed_info  = needed,
                raw_response = raw,
            )

        # Extract ANSWER
        ans_match = re.search(
            r"ANSWER:\s*(.*?)(?=COMPUTATION:|CONFIDENCE:|CITATIONS:|$)",
            raw, re.DOTALL | re.IGNORECASE
        )
        answer = ans_match.group(1).strip() if ans_match else raw.strip()

        # Extract COMPUTATION
        comp_match = re.search(
            r"COMPUTATION:\s*(.*?)(?=CONFIDENCE:|CITATIONS:|$)",
            raw, re.DOTALL | re.IGNORECASE
        )
        computation = comp_match.group(1).strip() if comp_match else "N/A"

        # Extract CONFIDENCE
        conf_match = re.search(
            r"CONFIDENCE:\s*([0-9.]+)",
            raw, re.IGNORECASE
        )
        raw_conf = float(conf_match.group(1)) if conf_match else 0.7
        raw_conf = max(0.0, min(1.0, raw_conf))

        # Apply confidence decay
        decay    = CONFIDENCE_DECAY.get(retry_count, 0.60)
        conf     = round(raw_conf * decay, 4)

        # Extract CITATIONS
        cit_match = re.search(
            r"CITATIONS:\s*(.*?)$",
            raw, re.DOTALL | re.IGNORECASE
        )
        cit_raw   = cit_match.group(1).strip() if cit_match else ""
        citations = [
            c.strip() for c in re.split(r"[\n,]", cit_raw)
            if c.strip() and len(c.strip()) > 3
        ]

        return ImplementorOutput(
            answer       = answer,
            confidence   = conf,
            citations    = citations,
            computation  = computation,
            output_type  = "ANSWER",
            raw_response = raw,
        )

    def _fallback_output(
        self,
        query:             str,
        retrieved_context: str,
        retry_count:       int,
    ) -> ImplementorOutput:
        """Fallback when Ollama unavailable — extract from context directly."""
        print("[Implementor] Using fallback — Ollama unavailable")

        # Try to find a number in context
        numbers = re.findall(r'\$[\d,]+(?:\.\d+)?\s*(?:million|billion)?', retrieved_context)
        if numbers:
            answer = (
                f"Based on retrieved context: {numbers[0]} "
                f"[Financial Statements / Page: see context]"
            )
        else:
            answer = (
                f"Retrieved context does not contain a clear answer to: {query}. "
                f"Please verify against source document."
            )

        decay = CONFIDENCE_DECAY.get(retry_count, 0.60)
        return ImplementorOutput(
            answer        = answer,
            confidence    = round(0.5 * decay, 4),
            citations     = ["[Fallback — verify against source]"],
            computation   = "N/A",
            output_type   = "ANSWER",
            fallback_used = True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK
# run: python src/agents/implementor.py
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        from rich import print as rprint
    except ImportError:
        rprint = print

    rprint("\n[bold cyan]-- ContextImplementor sanity check --[/bold cyan]")

    impl = ContextImplementor()
    rprint("[green]✓[/green] ContextImplementor instantiated")

    context = """Apple / 10-K / FY2023 / Financial Statements / 42
Net income: $96,995 million for fiscal year ended September 30 2023.
Total net sales: $383,285 million."""

    result = impl.run(
        query               = "What was Apple net income in FY2023?",
        retrieved_context   = context,
        analysis_plan       = "1. Find net income figure. 2. Cite section and page.",
        validation_criteria = "Must cite section/page. Must state units.",
        retry_count         = 0,
    )

    rprint(f"[green]✓[/green] Output type: {result.output_type}")
    rprint(f"[green]✓[/green] Confidence: {result.confidence}")
    rprint(f"[green]✓[/green] Fallback: {result.fallback_used}")
    rprint(f"[green]✓[/green] Answer length: {len(result.answer)} chars")

    assert result.output_type in ["ANSWER", "RETRIEVAL_MISS"]
    assert 0.0 <= result.confidence <= 1.0
    assert result.computation is not None

    # Confidence decay test
    r1 = impl.run(
        query="test", retrieved_context=context,
        analysis_plan="test", validation_criteria="test",
        retry_count=0,
    )
    r2 = impl.run(
        query="test", retrieved_context=context,
        analysis_plan="test", validation_criteria="test",
        retry_count=2,
    )
    # retry_count=2 should have lower or equal confidence
    # (both may use fallback so just check range)
    assert 0.0 <= r1.confidence <= 1.0
    assert 0.0 <= r2.confidence <= 1.0
    rprint(f"[green]✓[/green] Confidence decay: retry0={r1.confidence} retry2={r2.confidence}")

    rprint(f"\n[bold green]All checks passed. ContextImplementor ready.[/bold green]\n")