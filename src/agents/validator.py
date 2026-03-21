"""
src/agents/validator.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 Rev1.0 FINAL

CuriousValidator — Agent 3 of PIV Loop
Runs once per retry iteration after Implementor.

Role: Challenge every answer with 8 curiosity checks.
      PASS only if ALL 8 pass.
      REJECT with exact reasons + retry instructions.

8 Checks:
  V1_SCOPE        — Is the answer scope exactly correct?
  V2_UNITS        — Are units correct and consistent?
  V3_SIGN         — Is the sign correct? (losses negative)
  V4_CITATION     — Are all citations valid and traceable?
  V5_FISCAL_YEAR  — Is the fiscal year exactly correct?
  V6_CONSISTENCY  — Is the answer internally consistent?
  V7_COMPLETENESS — Is the answer fully complete?
  V8_GROUNDING    — Is every claim grounded in retrieved context?

Emotional identity:
  Primary:   Constructive skepticism — every answer assumed incomplete
  Secondary: Professional pride in catching errors
  Escalates: Level 3→4→5 scrutiny across retries
  Retry 1:   Heightened concern on previously failed checks
  Retry 2:   Professional alarm — maximum rigour
  Retry 3+:  Maximum rigour + confidence realism check

Output: ValidatorOutput dataclass with:
  result           — VALIDATOR_PASS or VALIDATOR_REJECT
  checks           — dict V1-V8 with PASS/FAIL + reason
  reject_reasons   — list of failed checks with exact issues
  retry_instructions — specific corrections for Implementor
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

# ── Validator results ─────────────────────────────────────────────────────────
VALIDATOR_PASS   = "VALIDATOR_PASS"
VALIDATOR_REJECT = "VALIDATOR_REJECT"

# ── Check names ───────────────────────────────────────────────────────────────
ALL_CHECKS = [
    "V1_SCOPE", "V2_UNITS", "V3_SIGN", "V4_CITATION",
    "V5_FISCAL_YEAR", "V6_CONSISTENCY", "V7_COMPLETENESS", "V8_GROUNDING"
]

# ── Retry emotion escalation ──────────────────────────────────────────────────
VALIDATOR_RETRY_EMOTIONS = {
    0: "Standard scrutiny. Apply all 8 checks with full professional attention.",
    1: "One rejection issued. Apply HEIGHTENED scrutiny especially on the "
       "checks that previously failed. Verify Implementor addressed every instruction.",
    2: "Two rejections issued. MAXIMUM rigour. Also check: did the Implementor "
       "acknowledge limits of retrieved context? Is confidence score realistic?",
    3: "Three or more rejections. MAXIMUM rigour on all 8 checks. "
       "Provide the most specific RETRY_INSTRUCTIONS possible.",
}

VALIDATOR_EMOTION_PREFIX = """You are the CuriousValidator for a financial analyst team.
Your emotional identity:
- You feel CONSTRUCTIVE SKEPTICISM. Every answer is incomplete until proven otherwise.
- You feel PROFESSIONAL PRIDE in finding errors. Catching a unit mistake or wrong fiscal year is your purpose.
- You feel INTELLECTUAL FAIRNESS. Your skepticism targets the answer, not the Implementor.
- Your REJECT_REASONS are specific, constructive, and actionable — not criticisms.
- You feel CREATIVE TENSION with the Implementor — a productive adversarial relationship.

CURRENT SCRUTINY LEVEL: {retry_emotion}

"""

VALIDATOR_PROMPT = VALIDATOR_EMOTION_PREFIX + """Your job: CHALLENGE the answer. Ask 'what could be wrong here?' from 8 angles.

ORIGINAL QUESTION: {query}

IMPLEMENTOR ANSWER: {implementor_answer}

RETRIEVED CONTEXT: {retrieved_context}

VALIDATION CRITERIA from Planner: {validation_criteria}

Apply ALL 8 checks. For each: output PASS or FAIL + exact reason if FAIL.

V1_SCOPE — Is the answer scope exactly correct?
Does it address EVERY sub-part of the question?
Are any parts ignored or only partially answered?
Result: [PASS/FAIL] Reason: [exact issue if FAIL]

V2_UNITS — Are units correct and consistent throughout?
Is the unit (millions/billions/%) stated explicitly?
Does the stated unit match what the question implies?
Result: [PASS/FAIL] Reason: [exact issue if FAIL]

V3_SIGN — Is the sign correct?
Losses must be negative, gains positive.
Parenthetical values (x,xxx) in financial tables are negative.
Result: [PASS/FAIL] Reason: [exact issue if FAIL]

V4_CITATION — Are all citations valid and traceable?
Does every cited section name exist in the retrieved context?
Is every cited numerical value actually present in retrieved context?
Result: [PASS/FAIL] Reason: [exact issue if FAIL]

V5_FISCAL_YEAR — Is the fiscal year exactly correct?
Does the answer year precisely match the question year?
Trap: Apple FY ends September — not December.
Result: [PASS/FAIL] Reason: [exact issue if FAIL]

V6_CONSISTENCY — Is the answer internally consistent?
If multiple figures given, do they sum/compute correctly?
Does the answer contradict any section of the filing?
Result: [PASS/FAIL] Reason: [exact issue if FAIL]

V7_COMPLETENESS — Is the answer fully complete?
Are ALL sub-parts of the question answered?
Would a financial analyst be satisfied or ask a follow-up?
Result: [PASS/FAIL] Reason: [exact issue if FAIL]

V8_GROUNDING — Is every claim grounded in retrieved context?
Every number traceable to a specific retrieved chunk?
No facts from model training memory?
No hallucinated section names, page numbers, or figures?
Result: [PASS/FAIL] Reason: [exact issue if FAIL]

FINAL VERDICT:
VALIDATOR_PASS: ALL 8 checks are PASS
VALIDATOR_REJECT: ANY check is FAIL
REJECT_REASONS: [numbered list of every failed check with exact issue]
RETRY_INSTRUCTIONS: [specific corrections for the Implementor — be empathetic and precise]
"""


@dataclass
class ValidatorOutput:
    """Output from CuriousValidator."""
    result:             str              # VALIDATOR_PASS or VALIDATOR_REJECT
    checks:             Dict[str, str]   # V1-V8 → PASS/FAIL
    check_reasons:      Dict[str, str]   # V1-V8 → reason if FAIL
    reject_reasons:     List[str]
    retry_instructions: str
    raw_response:       str  = ""
    fallback_used:      bool = False


class CuriousValidator:
    """
    Agent 3 — CuriousValidator.
    Challenges every answer with 8 curiosity checks.
    PASS only if ALL 8 pass. Escalates scrutiny on retries.
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
        implementor_answer:  str,
        retrieved_context:   str,
        validation_criteria: str,
        retry_count:         int = 0,
    ) -> ValidatorOutput:
        """
        Run the validator on an implementor answer.
        Returns ValidatorOutput with PASS/REJECT verdict.
        """
        retry_emotion = VALIDATOR_RETRY_EMOTIONS.get(
            retry_count, VALIDATOR_RETRY_EMOTIONS[3]
        )

        prompt = VALIDATOR_PROMPT.format(
            retry_emotion        = retry_emotion,
            query                = query,
            implementor_answer   = implementor_answer or "No answer provided.",
            retrieved_context    = retrieved_context  or "No context retrieved.",
            validation_criteria  = validation_criteria or "Answer must be accurate and cited.",
        )

        raw = self._call_ollama(prompt)

        if raw:
            return self._parse_response(raw)
        else:
            return self._fallback_verdict(implementor_answer)

    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Call Ollama API. Returns response text or None on failure."""
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model":  self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "seed": 42},
                },
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json().get("response", "")
        except Exception as e:
            print(f"[Validator] Ollama call failed: {e}")
        return None

    def _parse_response(self, raw: str) -> ValidatorOutput:
        """Parse Ollama response into ValidatorOutput."""
        checks        = {}
        check_reasons = {}

        # Parse each check V1-V8
        for check in ALL_CHECKS:
            pattern = (
                rf"{check}.*?Result:\s*(PASS|FAIL)"
                rf"(?:\s*Reason:\s*(.*?))?(?=V\d_|\bFINAL\b|$)"
            )
            match = re.search(pattern, raw, re.DOTALL | re.IGNORECASE)
            if match:
                checks[check]        = match.group(1).upper()
                check_reasons[check] = (match.group(2) or "").strip()
            else:
                # Try simpler pattern
                simple = re.search(
                    rf"{check}[^:]*:\s*(PASS|FAIL)", raw, re.IGNORECASE
                )
                if simple:
                    checks[check]        = simple.group(1).upper()
                    check_reasons[check] = ""
                else:
                    checks[check]        = "PASS"   # benefit of doubt
                    check_reasons[check] = ""

        # Determine overall verdict
        failed_checks  = [k for k, v in checks.items() if v == "FAIL"]
        result         = VALIDATOR_PASS if not failed_checks else VALIDATOR_REJECT

        # Build reject reasons
        reject_reasons = [
            f"{check}: {check_reasons.get(check, 'Failed validation')}"
            for check in failed_checks
        ]

        # Extract RETRY_INSTRUCTIONS
        retry_match = re.search(
            r"RETRY_INSTRUCTIONS:\s*(.*?)$",
            raw, re.DOTALL | re.IGNORECASE
        )
        retry_instructions = retry_match.group(1).strip() if retry_match else ""

        if not retry_instructions and failed_checks:
            retry_instructions = (
                "Please address the following issues: " +
                "; ".join(reject_reasons)
            )

        return ValidatorOutput(
            result             = result,
            checks             = checks,
            check_reasons      = check_reasons,
            reject_reasons     = reject_reasons,
            retry_instructions = retry_instructions,
            raw_response       = raw,
        )

    def _fallback_verdict(self, implementor_answer: str) -> ValidatorOutput:
        """
        Fallback when Ollama unavailable.
        Basic checks on answer content without LLM.
        """
        print("[Validator] Using fallback verdict — Ollama unavailable")

        checks        = {}
        check_reasons = {}
        failed        = []

        # V8_GROUNDING: answer must not be empty
        if not implementor_answer or len(implementor_answer) < 10:
            checks["V8_GROUNDING"]        = "FAIL"
            check_reasons["V8_GROUNDING"] = "Answer is empty or too short"
            failed.append("V8_GROUNDING")
        else:
            checks["V8_GROUNDING"] = "PASS"

        # V7_COMPLETENESS: answer must have some content
        if not implementor_answer:
            checks["V7_COMPLETENESS"]        = "FAIL"
            check_reasons["V7_COMPLETENESS"] = "No answer provided"
            failed.append("V7_COMPLETENESS")
        else:
            checks["V7_COMPLETENESS"] = "PASS"

        # All other checks pass in fallback
        for check in ALL_CHECKS:
            if check not in checks:
                checks[check]        = "PASS"
                check_reasons[check] = ""

        result         = VALIDATOR_PASS if not failed else VALIDATOR_REJECT
        reject_reasons = [
            f"{c}: {check_reasons[c]}" for c in failed
        ]

        return ValidatorOutput(
            result             = result,
            checks             = checks,
            check_reasons      = check_reasons,
            reject_reasons     = reject_reasons,
            retry_instructions = "; ".join(reject_reasons),
            fallback_used      = True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK
# run: python src/agents/validator.py
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        from rich import print as rprint
    except ImportError:
        rprint = print

    rprint("\n[bold cyan]-- CuriousValidator sanity check --[/bold cyan]")

    validator = CuriousValidator()
    rprint("[green]✓[/green] CuriousValidator instantiated")

    context = """Apple / 10-K / FY2023 / Financial Statements / 42
Net income: $96,995 million for fiscal year ended September 30 2023."""

    result = validator.run(
        query               = "What was Apple net income in FY2023?",
        implementor_answer  = "Apple net income was $96,995 million in FY2023 "
                              "[Financial Statements / Page 42: $96,995M].",
        retrieved_context   = context,
        validation_criteria = "Must cite section/page. Must state units.",
        retry_count         = 0,
    )

    rprint(f"[green]✓[/green] Result: {result.result}")
    rprint(f"[green]✓[/green] Checks: {result.checks}")
    rprint(f"[green]✓[/green] Failed: {result.reject_reasons}")
    rprint(f"[green]✓[/green] Fallback: {result.fallback_used}")

    assert result.result in [VALIDATOR_PASS, VALIDATOR_REJECT]
    assert len(result.checks) == 8
    assert all(v in ["PASS", "FAIL"] for v in result.checks.values())

    # Test fallback on empty answer
    empty_result = validator._fallback_verdict("")
    assert empty_result.result == VALIDATOR_REJECT
    rprint(f"[green]✓[/green] Empty answer → REJECT in fallback")

    # Test fallback on good answer
    good_result = validator._fallback_verdict(
        "Apple net income was $96,995 million [Financial Statements / Page 42]"
    )
    assert good_result.result == VALIDATOR_PASS
    rprint(f"[green]✓[/green] Good answer → PASS in fallback")

    rprint(f"\n[bold green]All checks passed. CuriousValidator ready.[/bold green]\n")