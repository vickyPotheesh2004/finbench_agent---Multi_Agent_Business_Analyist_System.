"""
src/pipeline/pipeline.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 · Rev 1.0

CHANGELOG:
    2026-05-10 S27  Bug Fix 2 v2: Refined early-exit logic.
                    - Check BM25 hits (0 = strong "no data" signal) since BGE
                      always returns chunks even for garbage queries.
                    - Check retrieval_stage_2 top score < 0.3 (low confidence).
                    - Detect LLM unavailable BEFORE running PIV pods, not after.
                    - Skip SHAP, N16, N17 when early-exit fires.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ─── Node function imports (tolerant of optional modules) ─────────────────────

from src.state.ba_state                  import BAState

# Ingestion
from src.ingestion.pdf_ingestor          import run_pdf_ingestor
from src.ingestion.section_tree_builder  import run_section_tree_builder
from src.ingestion.chunker               import run_chunker

# Routing
from src.routing.cart_router             import run_cart_router
from src.routing.lr_difficulty           import run_lr_difficulty

# Retrieval
from src.retrieval.sniper_rag            import run_sniper, SniperResult
from src.retrieval.bm25_retriever        import BM25Retriever
from src.retrieval.bge_retriever         import run_bge
from src.retrieval.rrf_reranker          import run_rrf_reranker

# Analysis
from src.analysis.prompt_assembler       import run_prompt_assembler
from src.analysis.piv_loop               import run_analyst_pod
from src.analysis.cfo_quant_pod          import run_cfo_quant_pod
from src.analysis.triguard               import run_triguard
from src.analysis.auditor_pod            import run_auditor_pod
from src.analysis.piv_mediator           import run_piv_mediator

# Explainability + ML
from src.analysis.shap_dag               import run_shap_dag
try:
    from src.ml.xgb_arbiter              import run_xgb_arbiter
except ImportError:
    run_xgb_arbiter = None

# RLEF + Output
from src.rlef.jee_engine                 import run_rlef_engine
from src.output.docx_generator           import run_output_generator
try:
    from src.output.pdf_report_generator import run_pdf_report_generator
except ImportError:
    run_pdf_report_generator = None

# Optional LLM client
try:
    from src.utils.llm_client            import get_llm_client
except ImportError:
    get_llm_client = None


# ─── Retrieval wrappers (some nodes have unique signatures) ───────────────────

def _run_sniper_node(state) -> object:
    """Adapter: run_sniper takes (query, table_cells) not (state)."""
    query = getattr(state, "query",       "") or ""
    cells = getattr(state, "table_cells", []) or []
    if not query or not cells:
        state.sniper_hit        = False
        state.sniper_confidence = 0.0
        state.sniper_answer     = ""
        return state

    try:
        result: SniperResult = run_sniper(query, cells)
        state.sniper_hit        = bool(result.sniper_hit)
        state.sniper_confidence = float(result.confidence)
        state.sniper_answer     = str(result.answer  or "")
        state.sniper_value      = str(result.value   or "")
        state.sniper_unit       = str(result.unit    or "")
        state.sniper_citation   = str(result.citation or "")
        state.sniper_pattern    = str(result.matched_pattern or "")
        state.sniper_result     = result.answer if result.sniper_hit else None
    except Exception as exc:
        logger.warning("[N06] SniperRAG failed: %s", exc)
        state.sniper_hit        = False
        state.sniper_confidence = 0.0
        state.sniper_answer     = ""
    return state


def _run_bm25_node(state) -> object:
    """Adapter: BM25Retriever has no run_bm25 function — use class."""
    try:
        retriever = BM25Retriever()
        return retriever.run(state)
    except Exception as exc:
        logger.warning("[N07] BM25 failed: %s", exc)
        if not hasattr(state, "bm25_results") or state.bm25_results is None:
            state.bm25_results = []
        return state


# ─── Early-exit helpers (Bug Fix 2 v2) ────────────────────────────────────────

def _check_llm_available(llm_client) -> bool:
    """Fast check if LLM is reachable. Uses cached is_available()."""
    if llm_client is None:
        return False
    try:
        return bool(llm_client.is_available())
    except Exception:
        return False


def _check_retrieval_quality(state) -> tuple:
    """Decide if retrieval has enough signal to warrant LLM analysis.

    Returns (has_signal: bool, reason: str).

    Strategy:
    - BM25 returns 0 results for genuine garbage queries (keyword miss).
      BGE returns 10 chunks for everything (semantic always matches).
      So BM25=0 is a reliable garbage signal.
    - retrieval_stage_2 (RRF reranker) top score < 0.3 indicates
      no chunk was strongly relevant.
    """
    bm25_hits = len(getattr(state, "bm25_results", []) or [])
    bge_hits  = len(getattr(state, "bge_results",  []) or [])
    stage2    = getattr(state, "retrieval_stage_2", []) or []

    if bm25_hits == 0 and bge_hits == 0 and len(stage2) == 0:
        return False, "all retrievers returned 0 chunks"

    # BM25 zero is a strong signal of "no real keyword match"
    if bm25_hits == 0 and len(stage2) > 0:
        # Check if stage2 (reranker output) has decent scores
        top_score = 0.0
        for chunk in stage2[:3]:
            if isinstance(chunk, dict):
                score = chunk.get("score") or chunk.get("rerank_score") or 0.0
                top_score = max(top_score, float(score or 0.0))
        if top_score < 0.3:
            return False, f"BM25=0 and stage2 top score {top_score:.2f} < 0.3"

    return True, f"BM25={bm25_hits}, BGE={bge_hits}, stage2={len(stage2)}"


def _build_early_exit_state(state, reason: str, label: str, conf: float,
                             include_preview: bool = False) -> object:
    """Set state fields for an early-exit answer + run only RLEF + Output."""
    if include_preview:
        top_chunk = ""
        for src_list in [
            getattr(state, "retrieval_stage_2", []) or [],
            getattr(state, "bge_results",       []) or [],
            getattr(state, "bm25_results",      []) or [],
        ]:
            if src_list:
                top = src_list[0]
                if isinstance(top, dict):
                    top_chunk = (top.get("text") or top.get("content") or "")[:300]
                    if top_chunk:
                        break
        state.final_answer = f"{label}: {reason}. Preview: {top_chunk}"
    else:
        state.final_answer = f"{label}: {reason}"

    state.final_answer_pre_xgb = state.final_answer
    state.confidence_score = conf
    state.winning_pod = label
    state.low_confidence = True
    # Run RLEF (logging) + Output (DOCX) only — skip SHAP, N17 to save time
    state = _safe_run("N18 RLEF Engine", run_rlef_engine, state)
    state = _safe_run("N19 Output Generator", run_output_generator, state)
    return state


# ─── FinBenchPipeline (external API) ──────────────────────────────────────────

class FinBenchPipeline:
    """Thin orchestrator over run_*(state) node functions."""

    def __init__(self) -> None:
        self.llm_client = None
        if get_llm_client is not None:
            try:
                self.llm_client = get_llm_client()
            except Exception as exc:
                logger.debug("[PIPELINE] LLM client unavailable: %s", exc)
        logger.info("[PIPELINE] FinBenchPipeline ready (llm=%s)",
                    self.llm_client is not None)

    # ── INGEST: N01 → N02 → N03 ───────────────────────────────────────────────

    def ingest(
        self,
        document_path: str,
        session_id:    str = "",
        company_name:  str = "",
        doc_type:      str = "",
        fiscal_year:   str = "",
        enable_images: bool = False,
        **_unused,
    ) -> BAState:
        """Run N01 + N02 + N03 ingestion sequence."""
        state = BAState(
            session_id    = session_id or "session",
            document_path = document_path,
            company_name  = company_name,
            doc_type      = doc_type,
            fiscal_year   = fiscal_year,
        )
        logger.info("[PIPELINE] Ingest: %s", document_path)

        state = run_pdf_ingestor(
            state,
            enable_images = enable_images,
            llm_client    = self.llm_client,
        )
        state = run_section_tree_builder(state, llm_client=self.llm_client)
        state = run_chunker(state)
        logger.info("[PIPELINE] Ingest complete: chunks=%d",
                    getattr(state, "chunk_count", 0))
        return state

    # ── QUERY: N04 → N19 ──────────────────────────────────────────────────────

    def query(self, state: BAState, question: str) -> BAState:
        """Run the full analysis pipeline on an ingested state."""
        if state is None:
            raise ValueError("state is None — run ingest() first")

        state.query = question
        logger.info("[PIPELINE] Query: %s", question[:80])

        # ── Routing (N04, N05) ────────────────────────────────────────────────
        state = _safe_run("N04 CART Router",      run_cart_router,   state)
        state = _safe_run("N05 LR Difficulty",    run_lr_difficulty, state)

        # ── Retrieval (N06 first, early-exit if sniper hits) ──────────────────
        state = _safe_run("N06 SniperRAG",        _run_sniper_node, state)

        sniper_hit = bool(getattr(state, "sniper_hit", False))

        # ── Bug F fix (S19): Sniper short-circuit on TRUE hit ─────────────────
        if sniper_hit:
            sniper_answer     = str(getattr(state, "sniper_answer",     "") or "")
            sniper_confidence = float(getattr(state, "sniper_confidence", 0.0) or 0.0)
            logger.info(
                "[PIPELINE] Sniper short-circuit: answer=%s | conf=%.3f",
                sniper_answer[:100], sniper_confidence,
            )
            state.final_answer        = sniper_answer
            state.final_answer_pre_xgb = sniper_answer
            state.confidence_score    = sniper_confidence
            state.winning_pod         = "SniperRAG"
            state.low_confidence      = sniper_confidence < 0.95
            state = _safe_run("N18 RLEF Engine",      run_rlef_engine,   state)
            state = _safe_run("N19 Output Generator", run_output_generator, state)
            logger.info(
                "[PIPELINE] Query complete (sniper short-circuit): "
                "confidence=%.3f winning_pod=%s",
                state.confidence_score, state.winning_pod,
            )
            return state

        # ── Bug Fix 2 v2: Check LLM availability BEFORE retrieval ─────────────
        # If LLM is dead, we know PIV pods will fail. We still run retrieval
        # so we can return chunk preview, but we skip the analysis pods.
        llm_available = _check_llm_available(self.llm_client)

        # ── Sniper missed → run retrieval ─────────────────────────────────────
        state = _safe_run("N07 BM25",             _run_bm25_node,    state)
        state = _safe_run("N08 BGE-M3",           run_bge,           state)
        state = _safe_run("N09 RRF+Reranker",     run_rrf_reranker,  state)

        # ── Bug Fix 2 v2: Retrieval quality check ────────────────────────────
        has_signal, reason = _check_retrieval_quality(state)

        # Branch A: low-quality retrieval (likely garbage query)
        if not has_signal:
            logger.warning("[PIPELINE] Early exit: %s", reason)
            return _build_early_exit_state(
                state,
                reason=reason,
                label="INSUFFICIENT_DATA",
                conf=0.0,
                include_preview=False,
            )

        # Branch B: chunks exist but LLM is dead — return chunks preview
        if not llm_available:
            logger.warning(
                "[PIPELINE] Early exit: LLM unavailable, %s "
                "but no analyst pod can run", reason,
            )
            return _build_early_exit_state(
                state,
                reason=f"Retrieved chunks ({reason}) but LLM is not running",
                label="LLM_UNAVAILABLE",
                conf=0.30,
                include_preview=True,
            )

        # ── Analysis (N10 assembles prompt, N11/N12/N13/N14 parallel pods) ────
        state = _safe_run("N10 Prompt Assembler", run_prompt_assembler, state)
        state = _safe_run(
            "N11 Analyst Pod",
            lambda s: run_analyst_pod(s, llm_client=self.llm_client),
            state,
        )

        state = _safe_run(
            "N12 CFO/Quant Pod",
            lambda s: run_cfo_quant_pod(s, llm_client=self.llm_client),
            state,
        )
        state = _safe_run("N13 TriGuard",         run_triguard,      state)
        state = _safe_run(
            "N14 Blind Auditor",
            lambda s: run_auditor_pod(s, llm_client=self.llm_client),
            state,
        )

        # ── Mediation (N15) ───────────────────────────────────────────────────
        state = _safe_run(
            "N15 PIV Mediator",
            lambda s: run_piv_mediator(s, llm_client=self.llm_client),
            state,
        )

        # ── Explainability + ML (N16, N17) ────────────────────────────────────
        state = _safe_run("N16 SHAP+DAG",         run_shap_dag,      state)
        if run_xgb_arbiter is not None:
            state = _safe_run("N17 XGB Arbiter",  run_xgb_arbiter,   state)

        # ── RLEF + Output (N18, N19) ──────────────────────────────────────────
        state = _safe_run("N18 RLEF Engine",      run_rlef_engine,   state)
        state = _safe_run("N19 Output Generator", run_output_generator, state)

        logger.info(
            "[PIPELINE] Query complete: confidence=%.3f winning_pod=%s",
            float(getattr(state, "confidence_score", 0.0) or 0.0),
            getattr(state, "winning_pod", "—"),
        )
        return state

    # ── RUN: ingest + query combined ──────────────────────────────────────────

    def run(self, document_path: str, question: str, **kwargs) -> BAState:
        """End-to-end: ingest then query."""
        state = self.ingest(document_path=document_path, **kwargs)
        return self.query(state, question)


# ─── Safety harness ───────────────────────────────────────────────────────────

def _safe_run(label: str, fn, state) -> object:
    """Run a node function with exception isolation."""
    try:
        result = fn(state)
        return result if result is not None else state
    except Exception as exc:
        logger.error("[%s] Failed: %s", label, exc)
        return state


# ─── Convenience top-level function ───────────────────────────────────────────

def run_pipeline(document_path: str, question: str, **kwargs) -> Any:
    """One-shot ingest + query."""
    return FinBenchPipeline().run(document_path, question, **kwargs)