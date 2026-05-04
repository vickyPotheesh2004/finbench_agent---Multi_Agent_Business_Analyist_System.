"""
src/pipeline/pipeline.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 · Rev 1.0

FinBench Pipeline — Thin Wrapper Over run_*(state) Functions

Design principle: this module does NOT duplicate node logic. Each node
has its own run_*(state) function in its own module. This wrapper simply
sequences those calls in the correct order and preserves the external API
(ingest / query / run) that app.py and run_eval.py depend on.

Pipeline flow:
    INGEST    N01 -> N02 -> N03
    QUERY     N04 -> N05 -> (N06 short-circuit? -> N10 -> ... )
                          -> N07 -> N08 -> N09
                          -> N10 -> N11 -> N12 -> N13 -> N14
                          -> N15 -> N16 -> N17 -> N18 -> N19

Constraints:
    C1  $0 cost — reuses existing node functions
    C2  100% local
    C5  seed=42 applied by SeedManager at BAState init
    C9  No _rlef_ fields in outputs
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
    """Adapter: run_sniper takes (query, table_cells) not (state).

    Bug A fix (S17): unpack SniperResult dataclass into individual
    BAState fields. Previously wrote the whole object into a str field
    causing Pydantic validation error and silently dropping all hits.
    """
    query = getattr(state, "query",       "") or ""
    cells = getattr(state, "table_cells", []) or []
    if not query or not cells:
        state.sniper_hit        = False
        state.sniper_confidence = 0.0
        state.sniper_answer     = ""
        return state

    try:
        result: SniperResult = run_sniper(query, cells)
        # Unpack into individual string/float/bool fields
        state.sniper_hit        = bool(result.sniper_hit)
        state.sniper_confidence = float(result.confidence)
        state.sniper_answer     = str(result.answer  or "")
        state.sniper_value      = str(result.value   or "")
        state.sniper_unit       = str(result.unit    or "")
        state.sniper_citation   = str(result.citation or "")
        state.sniper_pattern    = str(result.matched_pattern or "")
        # sniper_result stays as a human-readable answer string (or None)
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


# ─── FinBenchPipeline (external API) ──────────────────────────────────────────

class FinBenchPipeline:
    """
    Thin orchestrator over run_*(state) node functions.

    Preserves the original external API so app.py and run_eval.py keep
    working unchanged:
        pipeline = FinBenchPipeline()
        state    = pipeline.ingest(document_path=..., session_id=..., ...)
        state    = pipeline.query(state, question)
        state    = pipeline.run(document_path, question)
    """

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

        # ── Bug F fix (S19): Sniper short-circuit ─────────────────────────────
        # When SniperRAG hits with high confidence, its answer is authoritative
        # — a direct table cell lookup with full citation. Skip the LLM pipeline
        # entirely; LLM rephrasing only adds noise and hallucination risk.
        if sniper_hit:
            sniper_answer     = str(getattr(state, "sniper_answer",     "") or "")
            sniper_confidence = float(getattr(state, "sniper_confidence", 0.0) or 0.0)
            sniper_citation   = str(getattr(state, "sniper_citation",   "") or "")
            logger.info(
                "[PIPELINE] Sniper short-circuit: answer=%s | conf=%.3f",
                sniper_answer[:100], sniper_confidence,
            )
            state.final_answer        = sniper_answer
            state.final_answer_pre_xgb = sniper_answer
            state.confidence_score    = sniper_confidence
            state.winning_pod         = "SniperRAG"
            state.low_confidence      = sniper_confidence < 0.95
            # Run only N18 RLEF (for self-improvement) and N19 Output (for DOCX)
            state = _safe_run("N18 RLEF Engine",      run_rlef_engine,   state)
            state = _safe_run("N19 Output Generator", run_output_generator, state)
            logger.info(
                "[PIPELINE] Query complete (sniper short-circuit): "
                "confidence=%.3f winning_pod=%s",
                state.confidence_score, state.winning_pod,
            )
            return state

        # ── Sniper missed → run full retrieval + analysis ─────────────────────
        state = _safe_run("N07 BM25",             _run_bm25_node,    state)
        state = _safe_run("N08 BGE-M3",           run_bge,           state)
        state = _safe_run("N09 RRF+Reranker",     run_rrf_reranker,  state)

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

        # Optional: PDF report (N19b) runs on demand from UI, not here
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
    """
    Run a node function with exception isolation.
    If one node fails, log and continue — never crash the pipeline.
    """
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