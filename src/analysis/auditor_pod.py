"""
N14 Auditor Pod (BLIND) — Independent Blind Analysis
PDR-BAAAI-001 · Rev 1.0 · Node N14

Purpose:
    BLIND analysis pod — never sees N11 or N12 output.
    Re-retrieves independently to prevent anchoring bias.
    Explicitly checks for contradictions between document sections.
    Produces Candidate Answer 3 for N15 PIV Debate Mediator.

    BLINDNESS is architecturally enforced:
        - AuditorPod uses its own separate PIVLoopController instance
        - Never reads state.analyst_output or state.quant_result
        - Uses independent retrieval via state.retrieval_stage_2 only
        - Writes to state.auditor_output (separate field)

Constraints satisfied:
    C1  $0 cost — reuses Ollama client from piv_loop
    C2  100% local — localhost:11434
    C5  seed=42
    C9  No _rlef_ fields in output
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class AuditorPod:
    """
    N14 Blind Auditor Pod.

    Architecturally blind — never reads analyst_output or quant_result.
    Uses its own PIVLoopController instance (separate from N11/N12).
    Focuses on contradiction detection and independent verification.

    Two usage modes:
        1. pod.run_audit(query, chunks) → dict
        2. pod.run(ba_state)            → BAState (LangGraph node)
    """

    # Auditor-specific prompt additions injected into the PIV planner
    AUDITOR_CONTEXT = (
        "You are the BLIND AUDITOR. You have NOT seen any other analysis. "
        "Your job: independently verify the answer and flag contradictions. "
        "Check: (1) Does MD&A narrative match financial statement numbers? "
        "(2) Are there restatements or prior-period adjustments? "
        "(3) Do footnote disclosures contradict face of statements? "
        "(4) Are there going concern or audit qualification signals? "
        "Flag every inconsistency. Your independence is your value."
    )

    def __init__(self, llm_client=None) -> None:
        from src.analysis.piv_loop import PIVLoopController, OllamaClient
        # Separate instance — never shared with N11 or N12
        self._piv = PIVLoopController(
            llm_client = llm_client or OllamaClient(),
            pod_role   = "blind_auditor",
        )

    # ── LangGraph pipeline node entry point ───────────────────────────────────

    def run(self, state) -> object:
        """
        LangGraph N14 node entry point.

        BLIND: reads ONLY state.query and state.retrieval_stage_2.
        Never reads state.analyst_output or state.quant_result.

        Reads:  state.query, state.retrieval_stage_2,
                state.query_type, state.query_difficulty
        Writes: state.auditor_output, state.auditor_confidence,
                state.contradiction_flags

        Args:
            state: BAState object

        Returns:
            BAState with auditor fields populated
        """
        query      = getattr(state, "query",             "") or ""
        # BLIND: use retrieval_stage_2 only — no analyst output
        chunks     = getattr(state, "retrieval_stage_2", []) or []
        query_type = getattr(state, "query_type",        "text") or "text"
        difficulty = getattr(state, "query_difficulty",  "medium") or "medium"

        if not query:
            logger.warning("N14: empty query — skipping auditor pod")
            state.auditor_output      = ""
            state.auditor_confidence  = 0.0
            state.contradiction_flags = []
            return state

        result = self.run_audit(
            query            = query,
            chunks           = chunks,
            query_type       = query_type,
            query_difficulty = difficulty,
        )

        state.auditor_output      = result.get("answer", "")
        state.auditor_confidence  = result.get("confidence", 0.0)
        state.contradiction_flags = result.get("contradiction_flags", [])

        logger.info(
            "N14 Auditor (BLIND): confidence=%.3f | contradictions=%d | low_conf=%s",
            result.get("confidence", 0.0),
            len(result.get("contradiction_flags", [])),
            result.get("low_conf", False),
        )
        return state

    # ── Core audit method ─────────────────────────────────────────────────────

    def run_audit(
        self,
        query:            str,
        chunks:           List[Dict],
        query_type:       str = "text",
        query_difficulty: str = "medium",
    ) -> Dict:
        """
        Run blind audit using independent PIV loop.

        Appends auditor context to query so the Planner focuses on
        contradiction detection rather than just answer extraction.

        Args:
            query            : Analyst question
            chunks           : Retrieved chunks (independent of N11/N12)
            query_type       : From N04 CART Router
            query_difficulty : From N05 LR Difficulty

        Returns:
            Dict with answer, confidence, contradiction_flags
        """
        # Prepend auditor context to guide the PIV loop
        auditor_query = f"{self.AUDITOR_CONTEXT}\n\nQUESTION: {query}"

        piv_result = self._piv.run_piv(
            query            = auditor_query,
            chunks           = chunks,
            query_type       = query_type,
            query_difficulty = query_difficulty,
        )

        # Extract contradiction flags from answer text
        contradiction_flags = self._extract_contradictions(piv_result.answer)

        return {
            "answer":               piv_result.answer,
            "confidence":           piv_result.confidence,
            "citations":            piv_result.citations,
            "computation":          piv_result.computation,
            "retries":              piv_result.retries_used,
            "low_conf":             piv_result.low_confidence,
            "contradiction_flags":  contradiction_flags,
            "pod_role":             "blind_auditor",
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_contradictions(answer: str) -> List[str]:
        """
        Extract contradiction signals from the auditor's answer text.

        Looks for keywords indicating the auditor found inconsistencies.
        Returns list of contradiction description strings.
        """
        if not answer:
            return []

        contradiction_keywords = [
            "contradict", "inconsisten", "discrepan", "mismatch",
            "conflict", "disagree", "differs from", "does not match",
            "restatement", "prior period", "going concern",
            "qualified opinion", "material weakness", "significant deficiency",
        ]

        flags   = []
        lines   = answer.split("\n")
        for line in lines:
            line_lower = line.lower()
            for keyword in contradiction_keywords:
                if keyword in line_lower and len(line.strip()) > 20:
                    flags.append(line.strip()[:300])
                    break  # one flag per line max

        return flags[:10]  # cap at 10 flags


# ── Convenience wrapper for LangGraph N14 node ───────────────────────────────

def run_auditor_pod(state, llm_client=None) -> object:
    """
    Convenience wrapper for the LangGraph N14 Blind Auditor Pod node.
    """
    pod = AuditorPod(llm_client=llm_client)
    return pod.run(state)