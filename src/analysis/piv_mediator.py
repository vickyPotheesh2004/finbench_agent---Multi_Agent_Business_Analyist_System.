"""
N15 PIV Debate Mediator — 3-Pod Cross-Debate Resolution
PDR-BAAAI-001 · Rev 1.0 · Node N15

Purpose:
    Compares 3 candidate answers from:
        N11 LeadAnalyst    → state.analyst_output
        N12 QuantAnalyst   → state.quant_result
        N14 BlindAuditor   → state.auditor_output
    Resolution logic:
        2+ pods agree → majority winner (highest confidence)
        All disagree  → 3rd retrieval + LLM mediation
    Max 2 mediation rounds. Iteration cap: 5.
    Writes: state.final_answer_pre_xgb, state.piv_round, state.confidence_score

Constraints satisfied:
    C1  $0 cost — Ollama local LLM
    C2  100% local
    C5  seed=42
    C9  No _rlef_ fields in output
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_MEDIATION_ROUNDS = 2
ITERATION_CAP        = 5
AGREEMENT_THRESHOLD  = 0.85   # similarity threshold for "agreement"


@dataclass
class CandidateAnswer:
    pod_name:   str
    answer:     str
    confidence: float
    citations:  List[str]


@dataclass
class MediatorResult:
    final_answer:         str
    winning_pod:          str
    agreement_status:     str   # 'unanimous' / 'majority' / 'mediated' / 'fallback'
    confidence:           float
    resolution_reasoning: str
    rounds_used:          int


class PIVMediator:
    """
    N15 PIV Debate Mediator.

    Compares 3 pod candidates and resolves disagreements.

    Two usage modes:
        1. mediator.mediate(candidates, query, chunks) → MediatorResult
        2. mediator.run(ba_state)                      → BAState
    """

    MEDIATOR_PROMPT = """You are the PIVMediator. Your role: resolve disagreements fairly.
You are accountable to the retrieved context, not to any pod's ego.

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

    def __init__(self, llm_client=None) -> None:
        from src.analysis.piv_loop import OllamaClient
        self._llm = llm_client or OllamaClient()

    # ── LangGraph pipeline node entry point ───────────────────────────────────

    def run(self, state) -> object:
        """
        LangGraph N15 node entry point.

        Reads:
            state.analyst_output, state.analyst_confidence
            state.quant_result,   state.quant_confidence
            state.auditor_output, state.auditor_confidence
            state.contradiction_flags, state.retrieval_stage_2
            state.query

        Writes:
            state.final_answer_pre_xgb
            state.piv_round
            state.confidence_score
            state.low_confidence
        """
        query      = getattr(state, "query",              "") or ""
        chunks     = getattr(state, "retrieval_stage_2",  []) or []
        cont_flags = getattr(state, "contradiction_flags", []) or []

        candidates = [
            CandidateAnswer(
                pod_name   = "LeadAnalyst",
                answer     = getattr(state, "analyst_output",    "") or "",
                confidence = getattr(state, "analyst_confidence", 0.0),
                citations  = getattr(state, "analyst_citations",  []) or [],
            ),
            CandidateAnswer(
                pod_name   = "QuantAnalyst",
                answer     = getattr(state, "quant_result",      "") or "",
                confidence = getattr(state, "quant_confidence",  0.0),
                citations  = [],
            ),
            CandidateAnswer(
                pod_name   = "BlindAuditor",
                answer     = getattr(state, "auditor_output",    "") or "",
                confidence = getattr(state, "auditor_confidence", 0.0),
                citations  = [],
            ),
        ]

        # Filter out empty candidates
        valid = [c for c in candidates if c.answer.strip()]

        if not valid:
            logger.warning("N15: no valid candidates — returning empty")
            state.final_answer_pre_xgb = ""
            state.piv_round            = 0
            state.confidence_score     = 0.0
            state.low_confidence       = True
            return state

        result = self.mediate(
            candidates         = valid,
            query              = query,
            chunks             = chunks,
            contradiction_flags= cont_flags,
        )

        state.final_answer_pre_xgb = result.final_answer
        state.piv_round            = result.rounds_used
        state.confidence_score     = result.confidence
        state.low_confidence       = result.confidence < 0.65

        logger.info(
            "N15 Mediator: status=%s | winner=%s | confidence=%.3f | rounds=%d",
            result.agreement_status, result.winning_pod,
            result.confidence, result.rounds_used,
        )
        return state

    # ── Core mediation method ─────────────────────────────────────────────────

    def mediate(
        self,
        candidates:          List[CandidateAnswer],
        query:               str,
        chunks:              List[Dict],
        contradiction_flags: List[str] = None,
        round_num:           int = 0,
    ) -> MediatorResult:
        """
        Resolve disagreement between candidate answers.

        Steps:
            1. Check for unanimous agreement
            2. Check for majority (2-of-3) agreement
            3. If no agreement: LLM mediation
            4. Fallback: highest confidence candidate

        Args:
            candidates          : List of CandidateAnswer from N11/N12/N14
            query               : Analyst question
            chunks              : Retrieved chunks for context
            contradiction_flags : From N14 Auditor
            round_num           : Current mediation round

        Returns:
            MediatorResult with final answer and resolution metadata
        """
        if not candidates:
            return MediatorResult(
                final_answer         = "",
                winning_pod          = "none",
                agreement_status     = "fallback",
                confidence           = 0.0,
                resolution_reasoning = "No candidates available",
                rounds_used          = round_num,
            )

        # Step 1: Single candidate — return directly
        if len(candidates) == 1:
            c = candidates[0]
            return MediatorResult(
                final_answer         = c.answer,
                winning_pod          = c.pod_name,
                agreement_status     = "unanimous",
                confidence           = c.confidence,
                resolution_reasoning = "Only one candidate available",
                rounds_used          = round_num,
            )

        # Step 2: Check for unanimous agreement
        unanimous = self._check_unanimous(candidates)
        if unanimous:
            best = max(candidates, key=lambda c: c.confidence)
            return MediatorResult(
                final_answer         = best.answer,
                winning_pod          = best.pod_name,
                agreement_status     = "unanimous",
                confidence           = best.confidence,
                resolution_reasoning = "All candidates agree on core answer",
                rounds_used          = round_num,
            )

        # Step 3: Check for majority (2-of-3) agreement
        majority_result = self._check_majority(candidates)
        if majority_result:
            return MediatorResult(
                final_answer         = majority_result.answer,
                winning_pod          = majority_result.pod_name,
                agreement_status     = "majority",
                confidence           = majority_result.confidence,
                resolution_reasoning = "Majority of candidates agree on core answer",
                rounds_used          = round_num,
            )

        # Step 4: LLM mediation if within round limit
        if round_num < MAX_MEDIATION_ROUNDS and round_num < ITERATION_CAP:
            llm_result = self._llm_mediate(
                candidates          = candidates,
                query               = query,
                chunks              = chunks,
                contradiction_flags = contradiction_flags or [],
                round_num           = round_num,
            )
            if llm_result:
                return llm_result

        # Step 5: Fallback — highest confidence candidate
        best = max(candidates, key=lambda c: c.confidence)
        return MediatorResult(
            final_answer         = best.answer,
            winning_pod          = best.pod_name,
            agreement_status     = "fallback",
            confidence           = best.confidence * 0.8,
            resolution_reasoning = (
                f"No agreement reached after {round_num} rounds. "
                f"Falling back to highest confidence: {best.pod_name} "
                f"({best.confidence:.2f})"
            ),
            rounds_used          = round_num,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _check_unanimous(self, candidates: List[CandidateAnswer]) -> bool:
        """Check if all candidates agree on the core numerical/factual answer."""
        if len(candidates) < 2:
            return True
        core_answers = [self._extract_core_answer(c.answer) for c in candidates]
        first = core_answers[0]
        return all(self._answers_agree(first, other) for other in core_answers[1:])

    def _check_majority(
        self, candidates: List[CandidateAnswer]
    ) -> Optional[CandidateAnswer]:
        """
        Check for 2-of-3 majority agreement.
        Returns the winning candidate if majority found, else None.
        """
        if len(candidates) < 3:
            return None

        core_answers = [self._extract_core_answer(c.answer) for c in candidates]

        # Check each pair
        pairs = [(0, 1), (0, 2), (1, 2)]
        for i, j in pairs:
            if self._answers_agree(core_answers[i], core_answers[j]):
                # Return the one with higher confidence
                winner = candidates[i] if candidates[i].confidence >= candidates[j].confidence else candidates[j]
                return winner

        return None

    def _llm_mediate(
        self,
        candidates:          List[CandidateAnswer],
        query:               str,
        chunks:              List[Dict],
        contradiction_flags: List[str],
        round_num:           int,
    ) -> Optional[MediatorResult]:
        """Use LLM to resolve disagreement between candidates."""
        context = self._format_context(chunks)

        # Build candidate texts safely
        def get_candidate(name: str) -> CandidateAnswer:
            for c in candidates:
                if c.pod_name == name:
                    return c
            return CandidateAnswer(
                pod_name="none", answer="[Not available]", confidence=0.0, citations=[]
            )

        analyst = get_candidate("LeadAnalyst")
        quant   = get_candidate("QuantAnalyst")
        auditor = get_candidate("BlindAuditor")

        prompt = self.MEDIATOR_PROMPT.format(
            query               = query,
            retrieved_context   = context,
            analyst_output      = analyst.answer[:1000],
            analyst_confidence  = analyst.confidence,
            quant_result        = quant.answer[:1000],
            quant_confidence    = quant.confidence,
            auditor_output      = auditor.answer[:1000],
            auditor_confidence  = auditor.confidence,
            contradiction_flags = "; ".join(contradiction_flags[:3]) or "None",
        )

        response = self._llm.chat(prompt, temperature=0.1)
        if not response:
            return None

        return self._parse_mediator_response(response, round_num + 1)

    def _parse_mediator_response(
        self, response: str, rounds_used: int
    ) -> MediatorResult:
        """Parse LLM mediator response into MediatorResult."""
        status_m = re.search(
            r"AGREEMENT_STATUS[:\s]+(unanimous|majority|full_disagree)",
            response, re.IGNORECASE,
        )
        winner_m = re.search(
            r"WINNING_POD[:\s]+(LeadAnalyst|QuantAnalyst|BlindAuditor|Mediator_synthesis)",
            response, re.IGNORECASE,
        )
        answer_m = re.search(
            r"FINAL_ANSWER[:\s]+(.*?)(?=RESOLUTION_REASONING|CONFIDENCE|$)",
            response, re.DOTALL | re.IGNORECASE,
        )
        reason_m = re.search(
            r"RESOLUTION_REASONING[:\s]+(.*?)(?=CONFIDENCE|$)",
            response, re.DOTALL | re.IGNORECASE,
        )
        conf_m = re.search(r"CONFIDENCE[:\s]+([0-9.]+)", response, re.IGNORECASE)

        status = status_m.group(1).lower() if status_m else "mediated"
        winner = winner_m.group(1)         if winner_m else "Mediator_synthesis"
        answer = answer_m.group(1).strip() if answer_m else response[:500]
        reason = reason_m.group(1).strip() if reason_m else "LLM mediation"

        try:
            conf = float(conf_m.group(1)) if conf_m else 0.7
            conf = max(0.0, min(1.0, conf))
        except ValueError:
            conf = 0.7

        return MediatorResult(
            final_answer         = answer,
            winning_pod          = winner,
            agreement_status     = status,
            confidence           = conf,
            resolution_reasoning = reason,
            rounds_used          = rounds_used,
        )

    @staticmethod
    def _extract_core_answer(answer: str) -> str:
        """
        Extract the core numerical or factual claim from an answer.
        Strips citations and reasoning to get the bare answer value.
        """
        if not answer:
            return ""
        # Remove citation brackets
        clean = re.sub(r'\[.*?\]', '', answer)
        # Take first sentence
        sentences = re.split(r'[.!?]', clean)
        core      = sentences[0].strip() if sentences else clean
        # Extract just numbers if present
        numbers   = re.findall(r'\$?[\d,]+(?:\.\d+)?', core)
        if numbers:
            return numbers[0].replace(",", "").replace("$", "")
        return core[:100].lower().strip()

    @staticmethod
    def _answers_agree(a: str, b: str) -> bool:
        """
        Check if two core answers agree.
        Numeric: within 1% tolerance.
        Text: exact match of first 50 chars.
        """
        if not a or not b:
            return False
        # Try numeric comparison
        try:
            fa = float(a.replace(",", ""))
            fb = float(b.replace(",", ""))
            if fa == 0 and fb == 0:
                return True
            if fa == 0 or fb == 0:
                return False
            return abs(fa - fb) / max(abs(fa), abs(fb)) < 0.01
        except ValueError:
            pass
        # Text comparison
        return a[:50].lower().strip() == b[:50].lower().strip()

    @staticmethod
    def _format_context(chunks: List[Dict]) -> str:
        """Format chunks into context string."""
        if not chunks:
            return "[No context]"
        parts = []
        for i, chunk in enumerate(chunks[:3]):
            text    = chunk.get("text", "")[:500]
            section = chunk.get("section", "UNKNOWN")
            page    = chunk.get("page", 0)
            parts.append(f"[{section}/P{page}] {text}")
        return "\n".join(parts)


# ── Convenience wrapper ───────────────────────────────────────────────────────

def run_piv_mediator(state, llm_client=None) -> object:
    """Convenience wrapper for LangGraph N15 node."""
    mediator = PIVMediator(llm_client=llm_client)
    return mediator.run(state)