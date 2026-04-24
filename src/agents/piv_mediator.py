"""
src/agents/piv_mediator.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 Rev1.0 FINAL

N15 — PIV Debate Mediator
Arbitrates between N11 (Analyst) + N12 (Quant) + N14 (Auditor) candidates.

Resolution steps:
  STEP 1: Extract core numerical/factual answer from each pod
  STEP 2: Check if any 2+ pods agree on the core answer
  STEP 3: unanimous  → all 3 agree → highest confidence wins
          majority   → 2 of 3 agree → majority answer selected
          full_disagree → all different → mediator LLM resolves
  STEP 4: For full_disagree — mediator calls Llama to resolve
           using retrieved context as ground truth
  STEP 5: Write final_answer_pre_xgb + agreement_status + confidence

Max 2 mediation rounds. iteration_count cap = 5 (PDR constraint).

Emotional identity: Senior partner / PIV Mediator
  Trust with accountability — respects all 3 pods equally.
  Accountable to retrieved context, not to any pod's conclusion.
  The correct answer is in the document — not in any agent's opinion.
  Calm, decisive, fair.

Writes to BAState:
  final_answer_pre_xgb — answer ready for XGBoost Arbiter
  agreement_status     — unanimous / majority / full_disagree
  confidence_score     — 0.0-1.0 weighted by agreement
  winning_pod          — analyst / quant / auditor / mediator
"""

import sys
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import requests

from src.state.ba_state     import BAState, PIVStatus
from src.utils.seed_manager import SeedManager
from src.utils.resource_governor import ResourceGovernor

SeedManager.set_all()

# ── Ollama config ─────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"
TIMEOUT_SEC  = 120

# ── Agreement thresholds ─────────────────────────────────────────────────────
NUMERICAL_AGREEMENT_PCT = 0.05   # 5% tolerance for numerical agreement
MAX_MEDIATION_ROUNDS    = 2

# ── Confidence weights by agreement ─────────────────────────────────────────
CONFIDENCE_UNANIMOUS    = 1.0    # all 3 agree
CONFIDENCE_MAJORITY     = 0.85   # 2 of 3 agree
CONFIDENCE_MEDIATED     = 0.70   # mediator resolved
CONFIDENCE_FALLBACK     = 0.55   # could not resolve

# ── Mediator prompt ───────────────────────────────────────────────────────────
MEDIATOR_PROMPT = """You are the PIVMediator — a senior financial analysis partner.
Your role: resolve disagreements between three independent analysts fairly.
You are accountable to the RETRIEVED CONTEXT only — not to any analyst's opinion.

QUESTION: {query}

RETRIEVED CONTEXT:
{retrieved_context}

CANDIDATE 1 — LeadAnalyst (N11):
{analyst_output} [confidence: {analyst_confidence:.2f}]

CANDIDATE 2 — QuantAnalyst (N12):
{quant_result} [confidence: {quant_confidence:.2f}]

CANDIDATE 3 — BlindAuditor (N14) [independent blind review]:
{auditor_output} [confidence: {auditor_confidence:.2f}]

CONTRADICTION FLAGS: {contradiction_flags}

Resolution steps:
STEP 1: Extract core numerical/factual answer from each candidate.
STEP 2: Check if any two candidates agree on the core answer.
STEP 3: If 2+ agree → select winning answer. If all disagree → step 4.
STEP 4: For each disagreement, locate correct answer in retrieved context.
STEP 5: Issue final answer with full citations and resolution reasoning.

Output:
AGREEMENT_STATUS: [unanimous/majority/full_disagree]
WINNING_POD: [LeadAnalyst/QuantAnalyst/BlindAuditor/Mediator_synthesis]
FINAL_ANSWER: [complete answer with citations]
RESOLUTION_REASONING: [why this answer was selected]
CONFIDENCE: [0.0-1.0]
"""


class PIVMediator:
    """
    N15 — PIV Debate Mediator.

    Arbitrates between N11 + N12 + N14.
    Majority vote first, LLM mediation if full disagreement.
    Writes final_answer_pre_xgb to BAState.
    """

    def __init__(
        self,
        model:   str = OLLAMA_MODEL,
        timeout: int = TIMEOUT_SEC,
    ):
        SeedManager.set_all()
        self.model   = model
        self.timeout = timeout

    def run(self, state: BAState) -> BAState:
        """
        Main entry point — N15 node.

        Reads:  state.analyst_output, state.quant_result,
                state.auditor_output + their confidences
        Writes: state.final_answer_pre_xgb, state.agreement_status,
                state.confidence_score
        """
        ResourceGovernor.check("N15 PIV Mediator")

        candidates = self._collect_candidates(state)

        if not candidates:
            print("[N15] No candidates — using empty answer")
            state.final_answer_pre_xgb = ""
            state.agreement_status     = "no_candidates"
            state.confidence_score     = 0.0
            return state

        print(f"[N15] Mediating {len(candidates)} candidates")

        agreement, winner_idx = self._check_agreement(candidates)

        if agreement in ("unanimous", "majority"):
            winner                     = candidates[winner_idx]
            state.final_answer_pre_xgb = winner["answer"]
            state.agreement_status     = agreement
            state.confidence_score     = self._compute_confidence(
                agreement, winner["confidence"]
            )
            winning_pod = winner["pod"]
            print(f"[N15] {agreement} — winner: {winning_pod} "
                  f"conf={state.confidence_score:.2f}")

        else:
            print("[N15] full_disagree — calling LLM mediator")
            mediated = self._mediate_with_llm(state, candidates)

            state.final_answer_pre_xgb = mediated["answer"]
            state.agreement_status     = "full_disagree"
            state.confidence_score     = mediated["confidence"]
            winning_pod                = mediated["winning_pod"]
            print(f"[N15] Mediated — winner: {winning_pod} "
                  f"conf={state.confidence_score:.2f}")

        state.agreement_status = f"{state.agreement_status}|{winning_pod}"
        return state

    # ═══════════════════════════════════════════════════════════════════════
    # CANDIDATE COLLECTION
    # ═══════════════════════════════════════════════════════════════════════

    def _collect_candidates(
        self, state: BAState
    ) -> List[Dict[str, Any]]:
        """Collect non-empty candidates from all 3 pods."""
        candidates = []

        if (state.analyst_output and
                state.analyst_piv_status == PIVStatus.PASS):
            candidates.append({
                "pod":        "analyst",
                "answer":     state.analyst_output,
                "confidence": state.analyst_confidence,
                "citations":  state.analyst_citations,
            })

        if (state.quant_result and
                state.quant_piv_status == PIVStatus.PASS):
            candidates.append({
                "pod":        "quant",
                "answer":     state.quant_result,
                "confidence": state.quant_confidence,
                "citations":  state.quant_citations,
            })

        if (state.auditor_output and
                state.auditor_piv_status == PIVStatus.PASS):
            candidates.append({
                "pod":        "auditor",
                "answer":     state.auditor_output,
                "confidence": state.auditor_confidence,
                "citations":  state.auditor_citations,
            })

        # Fallback: include REJECT candidates if none passed
        if not candidates:
            for pod, answer, conf, cits in [
                ("analyst", state.analyst_output,
                 state.analyst_confidence, state.analyst_citations),
                ("quant",   state.quant_result,
                 state.quant_confidence,  state.quant_citations),
                ("auditor", state.auditor_output,
                 state.auditor_confidence, state.auditor_citations),
            ]:
                if answer:
                    candidates.append({
                        "pod":        pod,
                        "answer":     answer,
                        "confidence": conf,
                        "citations":  cits,
                    })

        return candidates

    # ═══════════════════════════════════════════════════════════════════════
    # AGREEMENT CHECKING
    # ═══════════════════════════════════════════════════════════════════════

    def _check_agreement(
        self, candidates: List[Dict[str, Any]]
    ) -> Tuple[str, int]:
        """
        Check if candidates agree on the core answer.
        Returns (agreement_type, winner_index).
        """
        if len(candidates) == 1:
            return "unanimous", 0

        if len(candidates) == 2:
            agree = self._answers_agree(
                candidates[0]["answer"],
                candidates[1]["answer"],
            )
            if agree:
                winner = 0 if (
                    candidates[0]["confidence"] >=
                    candidates[1]["confidence"]
                ) else 1
                return "majority", winner
            return "full_disagree", 0

        # 3 candidates
        agree_01 = self._answers_agree(
            candidates[0]["answer"], candidates[1]["answer"]
        )
        agree_02 = self._answers_agree(
            candidates[0]["answer"], candidates[2]["answer"]
        )
        agree_12 = self._answers_agree(
            candidates[1]["answer"], candidates[2]["answer"]
        )

        if agree_01 and agree_02 and agree_12:
            best = max(range(3), key=lambda i: candidates[i]["confidence"])
            return "unanimous", best

        if agree_01:
            winner = 0 if (
                candidates[0]["confidence"] >=
                candidates[1]["confidence"]
            ) else 1
            return "majority", winner

        if agree_02:
            winner = 0 if (
                candidates[0]["confidence"] >=
                candidates[2]["confidence"]
            ) else 2
            return "majority", winner

        if agree_12:
            winner = 1 if (
                candidates[1]["confidence"] >=
                candidates[2]["confidence"]
            ) else 2
            return "majority", winner

        return "full_disagree", 0

    def _answers_agree(self, answer_a: str, answer_b: str) -> bool:
        """
        Check if two answers agree on the core numerical value.
        Uses 5% tolerance for numerical comparison.
        Falls back to text similarity for non-numerical answers.
        """
        nums_a = self._extract_primary_number(answer_a)
        nums_b = self._extract_primary_number(answer_b)

        if nums_a is not None and nums_b is not None:
            # Both zero → agree
            if nums_a == 0.0 and nums_b == 0.0:
                return True
            # One zero one nonzero → disagree
            if nums_a == 0.0 or nums_b == 0.0:
                return False
            diff_pct = abs(nums_a - nums_b) / max(abs(nums_a), abs(nums_b))
            return diff_pct <= NUMERICAL_AGREEMENT_PCT

        # Text-based agreement
        words_a = set(answer_a.lower().split())
        words_b = set(answer_b.lower().split())
        stops   = {"the", "a", "an", "is", "was", "were", "in", "of",
                   "and", "for", "to", "at", "by"}
        words_a -= stops
        words_b -= stops

        if not words_a or not words_b:
            return False

        overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
        return overlap >= 0.60

    def _extract_primary_number(self, text: str) -> Optional[float]:
        """Extract the first significant number from text including zero."""
        pattern = r'\$?([\d,]+(?:\.\d+)?)\s*(?:million|billion|M|B|%)?'
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            try:
                val = float(m.replace(",", ""))
                if val >= 0:   # include zero — fixed from > 0
                    return val
            except ValueError:
                pass
        return None

    # ═══════════════════════════════════════════════════════════════════════
    # CONFIDENCE COMPUTATION
    # ═══════════════════════════════════════════════════════════════════════

    def _compute_confidence(
        self, agreement: str, pod_confidence: float
    ) -> float:
        """Compute final confidence score based on agreement type."""
        weights = {
            "unanimous":     CONFIDENCE_UNANIMOUS,
            "majority":      CONFIDENCE_MAJORITY,
            "full_disagree": CONFIDENCE_MEDIATED,
        }
        weight = weights.get(agreement, CONFIDENCE_FALLBACK)
        return round(min(pod_confidence * weight, 1.0), 4)

    # ═══════════════════════════════════════════════════════════════════════
    # LLM MEDIATION
    # ═══════════════════════════════════════════════════════════════════════

    def _mediate_with_llm(
        self,
        state:      BAState,
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Call Llama to mediate. Falls back to highest-confidence."""
        context = self._build_context(state)

        prompt = MEDIATOR_PROMPT.format(
            query               = state.query or "",
            retrieved_context   = context,
            analyst_output      = state.analyst_output  or "No answer",
            analyst_confidence  = state.analyst_confidence,
            quant_result        = state.quant_result    or "No answer",
            quant_confidence    = state.quant_confidence,
            auditor_output      = state.auditor_output  or "No answer",
            auditor_confidence  = state.auditor_confidence,
            contradiction_flags = str(state.contradiction_flags or []),
        )

        raw = self._call_ollama(prompt)

        if raw:
            return self._parse_mediation(raw, candidates)
        else:
            return self._fallback_mediation(candidates)

    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Call Ollama API. Returns response text or None."""
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
            print(f"[N15] Ollama call failed: {e}")
        return None

    def _parse_mediation(
        self,
        raw:        str,
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Parse Ollama mediation response."""
        ans_match = re.search(
            r"FINAL_ANSWER:\s*(.*?)(?=RESOLUTION_REASONING:|CONFIDENCE:|$)",
            raw, re.DOTALL | re.IGNORECASE
        )
        answer = ans_match.group(1).strip() if ans_match else ""

        pod_match = re.search(
            r"WINNING_POD:\s*(LeadAnalyst|QuantAnalyst|BlindAuditor|Mediator_synthesis)",
            raw, re.IGNORECASE
        )
        pod_map = {
            "leadanalyst":        "analyst",
            "quantanalyst":       "quant",
            "blindauditor":       "auditor",
            "mediator_synthesis": "mediator",
        }
        raw_pod     = pod_match.group(1).lower() if pod_match else "mediator"
        winning_pod = pod_map.get(raw_pod, "mediator")

        conf_match = re.search(
            r"CONFIDENCE:\s*([0-9.]+)", raw, re.IGNORECASE
        )
        raw_conf   = float(conf_match.group(1)) if conf_match else 0.7
        confidence = round(
            min(max(raw_conf, 0.0), 1.0) * CONFIDENCE_MEDIATED, 4
        )

        if not answer and candidates:
            answer = self._fallback_mediation(candidates)["answer"]

        return {
            "answer":      answer,
            "confidence":  confidence,
            "winning_pod": winning_pod,
        }

    def _fallback_mediation(
        self, candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Fallback — pick highest-confidence candidate."""
        print("[N15] Using fallback mediation — highest confidence wins")
        if not candidates:
            return {"answer": "", "confidence": 0.0, "winning_pod": "none"}
        best = max(candidates, key=lambda c: c["confidence"])
        return {
            "answer":      best["answer"],
            "confidence":  round(best["confidence"] * CONFIDENCE_FALLBACK, 4),
            "winning_pod": best["pod"],
        }

    def _build_context(self, state: BAState) -> str:
        """Build context string from BAState chunks."""
        chunks = state.retrieval_stage_2 or state.retrieval_stage_1 or []
        if not chunks:
            return "No context available."
        parts = []
        for i, chunk in enumerate(chunks, 1):
            text    = chunk.get("text") or chunk.get("content") or ""
            section = chunk.get("section", "Unknown")
            page    = chunk.get("page", "?")
            parts.append(f"[Source {i}: {section}/P{page}] {text}")
        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK
# run: python src/agents/piv_mediator.py
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        from rich import print as rprint
    except ImportError:
        rprint = print

    rprint("\n[bold cyan]-- PIVMediator (N15) sanity check --[/bold cyan]")

    mediator = PIVMediator()
    rprint("[green]✓[/green] PIVMediator instantiated")

    # Unanimous
    state = BAState(
        session_id         = "sanity-n15-unanimous",
        query              = "What was Apple net income FY2023?",
        company_name       = "Apple Inc",
        analyst_output     = "Net income $96,995M [FS/42].",
        analyst_confidence = 0.92,
        analyst_piv_status = PIVStatus.PASS,
        quant_result       = "Net income $96,995 million [FS/42].",
        quant_confidence   = 0.88,
        quant_piv_status   = PIVStatus.PASS,
        auditor_output     = "Net income was $96,995M [FS/42].",
        auditor_confidence = 0.90,
        auditor_piv_status = PIVStatus.PASS,
        retrieval_stage_2  = [{
            "text": "Net income $96,995M FY2023.",
            "section": "Financial Statements", "page": "42",
        }],
    )
    state = mediator.run(state)
    assert "unanimous" in state.agreement_status
    assert state.final_answer_pre_xgb != ""
    assert state.confidence_score > 0.8
    rprint(f"[green]✓[/green] Unanimous: {state.agreement_status} "
           f"conf={state.confidence_score:.2f}")

    # Majority
    state2 = BAState(
        session_id         = "sanity-n15-majority",
        query              = "What was Apple net income FY2023?",
        analyst_output     = "Net income $96,995M FY2023.",
        analyst_confidence = 0.92,
        analyst_piv_status = PIVStatus.PASS,
        quant_result       = "Net income $96,995 million FY2023.",
        quant_confidence   = 0.88,
        quant_piv_status   = PIVStatus.PASS,
        auditor_output     = "Net income $57,411M FY2022.",
        auditor_confidence = 0.75,
        auditor_piv_status = PIVStatus.PASS,
    )
    state2 = mediator.run(state2)
    assert "majority" in state2.agreement_status
    rprint(f"[green]✓[/green] Majority: {state2.agreement_status} "
           f"conf={state2.confidence_score:.2f}")

    # Agreement detection
    assert mediator._answers_agree("$96,995M", "$96,995 million") is True
    assert mediator._answers_agree("$96,995M", "$57,411M")        is False
    assert mediator._answers_agree("0", "0.0")                    is True
    rprint(f"[green]✓[/green] Agreement detection + zero handling correct")

    assert state.seed == 42
    rprint(f"[green]✓[/green] seed=42 preserved")

    rprint(f"\n[bold green]All checks passed. PIVMediator N15 ready.[/bold green]\n")