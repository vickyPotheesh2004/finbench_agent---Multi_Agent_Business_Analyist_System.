"""
N09 RRF + Reranker — Tier 4 Reciprocal Rank Fusion + Cross-Encoder Reranking
PDR-BAAAI-001 · Rev 1.0 · Node N09

Purpose:
    1. RRF merges BM25 top-10 (from N07) + BGE-M3 top-10 (from N08)
       into a single ranked list using Reciprocal Rank Fusion (k=60).
    2. BGE cross-encoder reranker scores each (query, chunk) pair more
       accurately than bi-encoder similarity.
    3. Final top-3 stored in ba_state.retrieval_stage_2.

Pipeline position:
    N07 BM25 results  ──┐
                        ├──► N09 RRF ──► Reranker ──► top-3 → retrieval_stage_2
    N08 BGE-M3 results ─┘

Constraints satisfied:
    C1  $0 cost — FlagEmbedding is free, no paid APIs
    C2  100% local — zero network calls
    C5  seed=42 — no randomness in RRF or reranker
    C7  N/A — no LLM prompt at this node
    C8  All results carry 5-field metadata prefix
    C9  No _rlef_ fields

PDR Reference:
    Section 7.5 — RRF formula: score = Σ 1/(k + rank_i), k=60
    Section 7.5 — bge-reranker-base cross-encoder
    Algorithm:   8-line pure Python RRF (no library needed)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# RRF constant — PDR Section 7.5, Cormack & Lynam 2009
_RRF_K = 60

# Final output size after reranking
_DEFAULT_FINAL_TOP_K = 3

# Reranker model — PDR Section 7.5
_RERANKER_MODEL = "BAAI/bge-reranker-base"

# Label for results from this node
_RETRIEVER_LABEL = "rrf_reranker"

# Lazy import holder
_flag_reranker = None


def _get_flag_reranker():
    """Lazy load FlagEmbedding reranker."""
    global _flag_reranker
    if _flag_reranker is None:
        try:
            from FlagEmbedding import FlagReranker
            _flag_reranker = FlagReranker
        except ImportError:
            logger.warning(
                "FlagEmbedding not installed. "
                "Reranker disabled — RRF only. "
                "Install with: pip install FlagEmbedding"
            )
            _flag_reranker = None
    return _flag_reranker


# ── Pure Python RRF — 8 lines (PDR Section 7.5) ───────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: List[List[str]],
    k:            int = _RRF_K,
) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion (Cormack & Lynam, SIGIR 2009).

    Args:
        ranked_lists : List of ranked lists of chunk_ids.
                       Each inner list is ordered best-first.
        k            : RRF constant (default 60, per PDR).

    Returns:
        List of (chunk_id, rrf_score) sorted by score descending.

    Formula: score(d) = Σ 1 / (k + rank(d))
    This is exactly 8 lines as specified in PDR Section 7.5.
    """
    scores: Dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── RRFReranker ───────────────────────────────────────────────────────────────

class RRFReranker:
    """
    N09 RRF + Cross-Encoder Reranker.

    Two usage modes:
        1. reranker.rerank(query, bm25_results, bge_results) → List[Dict]
        2. reranker.run(ba_state)                            → BAState

    The reranker first merges BM25 and BGE-M3 results using RRF,
    then applies a cross-encoder to rerank the merged list.
    The final top-3 chunks are stored in retrieval_stage_2.

    Graceful degradation:
        - If FlagEmbedding is not installed: RRF only (no cross-encoder)
        - If either BM25 or BGE results are empty: uses whichever is available
        - If both are empty: returns empty list
    """

    def __init__(
        self,
        reranker_model: str = _RERANKER_MODEL,
        final_top_k:    int = _DEFAULT_FINAL_TOP_K,
        rrf_k:          int = _RRF_K,
        use_reranker:   bool = True,
    ) -> None:
        self.reranker_model = reranker_model
        self.final_top_k    = final_top_k
        self.rrf_k          = rrf_k
        self.use_reranker   = use_reranker
        self._reranker      = None   # lazy loaded

    # ── LangGraph pipeline node entry point ───────────────────────────────────

    def run(self, state) -> object:
        """
        LangGraph N09 node entry point.

        Reads:
            state.query             — analyst question
            state.bm25_results      — N07 BM25 top-10 results
            state.retrieval_stage_1 — N08 BGE-M3 top-10 results

        Writes:
            state.retrieval_stage_2 — final top-3 reranked chunks

        Args:
            state: BAState object

        Returns:
            BAState with retrieval_stage_2 populated
        """
        query       = getattr(state, "query",             "") or ""
        bm25_results = getattr(state, "bm25_results",     []) or []
        bge_results  = getattr(state, "retrieval_stage_1", []) or []

        if not query:
            logger.warning("N09 RRF: empty query — returning empty results")
            state.retrieval_stage_2 = []
            return state

        final_results = self.rerank(
            query       = query,
            bm25_results = bm25_results,
            bge_results  = bge_results,
        )

        state.retrieval_stage_2 = final_results

        logger.info(
            "N09 RRF+Reranker: bm25=%d bge=%d → merged → top%d",
            len(bm25_results), len(bge_results), len(final_results),
        )
        return state

    # ── Core reranking method ─────────────────────────────────────────────────

    def rerank(
        self,
        query:        str,
        bm25_results: List[Dict],
        bge_results:  List[Dict],
    ) -> List[Dict]:
        """
        Merge BM25 and BGE-M3 results using RRF then apply cross-encoder.

        Steps:
            1. Build chunk lookup dict (chunk_id → chunk data)
            2. Extract ranked lists of chunk_ids from each retriever
            3. Apply RRF to get merged ranked list
            4. Look up full chunk data for top candidates
            5. Apply cross-encoder reranker if available
            6. Return final top-k results

        Args:
            query        : Analyst question string
            bm25_results : List of result dicts from N07 BM25
            bge_results  : List of result dicts from N08 BGE-M3

        Returns:
            List of reranked result dicts (max final_top_k items)
        """
        if not bm25_results and not bge_results:
            logger.warning("N09: Both BM25 and BGE results empty")
            return []

        # Step 1: Build chunk lookup by chunk_id
        chunk_lookup: Dict[str, Dict] = {}
        for r in bm25_results + bge_results:
            cid = r.get("chunk_id", "") or r.get("id", "")
            if cid and cid not in chunk_lookup:
                chunk_lookup[cid] = r

        # Step 2: Extract ordered chunk_id lists
        bm25_ids = [
            r.get("chunk_id", "") or r.get("id", "")
            for r in bm25_results
        ]
        bge_ids = [
            r.get("chunk_id", "") or r.get("id", "")
            for r in bge_results
        ]

        # Filter out empty ids
        bm25_ids = [i for i in bm25_ids if i]
        bge_ids  = [i for i in bge_ids  if i]

        # Step 3: RRF merge
        ranked_lists = [l for l in [bm25_ids, bge_ids] if l]
        if not ranked_lists:
            return []

        rrf_ranked = reciprocal_rank_fusion(ranked_lists, k=self.rrf_k)

        # Step 4: Collect top candidates for reranking
        # Take up to 10 candidates (more than final_top_k for reranker to work with)
        rerank_pool_size = max(self.final_top_k * 3, 10)
        candidates: List[Dict] = []
        for chunk_id, rrf_score in rrf_ranked[:rerank_pool_size]:
            if chunk_id in chunk_lookup:
                chunk = dict(chunk_lookup[chunk_id])
                chunk["rrf_score"]  = round(rrf_score, 6)
                chunk["retriever"]  = _RETRIEVER_LABEL
                candidates.append(chunk)

        if not candidates:
            return []

        # Step 5: Apply cross-encoder reranker if available
        if self.use_reranker and len(candidates) > 1:
            candidates = self._apply_reranker(query, candidates)

        # Step 6: Return final top-k with updated ranks
        final = candidates[:self.final_top_k]
        for i, chunk in enumerate(final):
            chunk["rank"]      = i + 1
            chunk["retriever"] = _RETRIEVER_LABEL
        return final

    # ── Private helpers ───────────────────────────────────────────────────────

    def _apply_reranker(
        self,
        query:      str,
        candidates: List[Dict],
    ) -> List[Dict]:
        """
        Apply BGE cross-encoder reranker to (query, chunk) pairs.

        The cross-encoder sees both the query and the chunk text together,
        giving more accurate relevance scores than bi-encoder similarity.

        Falls back to RRF ordering if reranker fails or is unavailable.

        Args:
            query      : Analyst question
            candidates : List of candidate chunks from RRF merge

        Returns:
            Candidates sorted by cross-encoder score descending
        """
        try:
            reranker = self._load_reranker()
            if reranker is None:
                logger.debug("N09: Reranker unavailable — using RRF order")
                return candidates

            # Build (query, passage) pairs for cross-encoder
            pairs = [
                [query, c.get("text", "") or c.get("page_content", "")]
                for c in candidates
            ]

            # Score all pairs — cross-encoder returns relevance scores
            scores = reranker.compute_score(pairs, normalize=True)

            # Handle both single score and list of scores
            if isinstance(scores, float):
                scores = [scores]

            # Attach reranker scores and sort
            for chunk, score in zip(candidates, scores):
                chunk["reranker_score"] = round(float(score), 6)

            candidates.sort(
                key=lambda x: x.get("reranker_score", 0.0),
                reverse=True,
            )
            logger.debug(
                "N09: Reranker scored %d candidates, top score=%.3f",
                len(candidates),
                candidates[0].get("reranker_score", 0.0) if candidates else 0.0,
            )
            return candidates

        except Exception as exc:
            logger.warning(
                "N09: Reranker failed (%s) — falling back to RRF order", exc
            )
            return candidates

    def _load_reranker(self):
        """Lazy load BGE cross-encoder reranker."""
        if self._reranker is None:
            FlagReranker = _get_flag_reranker()
            if FlagReranker is None:
                return None
            try:
                self._reranker = FlagReranker(
                    self.reranker_model,
                    use_fp16=False,   # C4: stable on CPU
                )
                logger.info("N09: Loaded reranker '%s'", self.reranker_model)
            except Exception as exc:
                logger.warning("N09: Failed to load reranker: %s", exc)
                return None
        return self._reranker


# ── LangChain EnsembleRetriever compatibility ─────────────────────────────────

class RRFEnsembleRetriever:
    """
    LangChain EnsembleRetriever-compatible wrapper.

    Combines BM25 and BGE-M3 LangChain retrievers using RRF,
    then applies cross-encoder reranking.

    Used by N09 when LangChain pipeline integration is needed.

    Example:
        bm25_lc = bm25_retriever.as_langchain_retriever(index_path)
        bge_lc  = bge_retriever.as_langchain_retriever(collection, data_dir)
        ensemble = RRFEnsembleRetriever(
            retrievers = [bm25_lc, bge_lc],
            rrf_reranker = RRFReranker(),
            query = "net income FY2023",
        )
        docs = ensemble.invoke("net income FY2023")
    """

    def __init__(
        self,
        retrievers:   List[Any],
        rrf_reranker: RRFReranker,
    ) -> None:
        self.retrievers   = retrievers
        self.rrf_reranker = rrf_reranker

    def invoke(self, query: str) -> List[Any]:
        """
        Invoke all retrievers, merge with RRF, rerank, return Documents.
        """
        from langchain_core.documents import Document

        all_results: List[List[Dict]] = []

        for retriever in self.retrievers:
            try:
                docs = retriever.invoke(query)
                # Convert LangChain Documents to result dicts
                result_dicts = []
                for rank, doc in enumerate(docs, start=1):
                    meta = doc.metadata if hasattr(doc, "metadata") else {}
                    result_dicts.append({
                        "chunk_id":    meta.get("chunk_id", f"doc_{rank}"),
                        "text":        doc.page_content,
                        "rank":        rank,
                        "retriever":   meta.get("retriever", "unknown"),
                        "company":     meta.get("company",     "UNKNOWN"),
                        "doc_type":    meta.get("doc_type",    "UNKNOWN"),
                        "fiscal_year": meta.get("fiscal_year", "UNKNOWN"),
                        "section":     meta.get("section",     "UNKNOWN"),
                        "page":        meta.get("page",         0),
                        "prefix":      meta.get("prefix",      ""),
                        "bm25_score":  meta.get("bm25_score",  0.0),
                        "bge_score":   meta.get("bge_score",   0.0),
                    })
                all_results.append(result_dicts)
            except Exception as exc:
                logger.warning("N09 EnsembleRetriever error: %s", exc)
                all_results.append([])

        # Separate BM25 and BGE results for RRF
        bm25_results = all_results[0] if len(all_results) > 0 else []
        bge_results  = all_results[1] if len(all_results) > 1 else []

        final_results = self.rrf_reranker.rerank(
            query        = query,
            bm25_results = bm25_results,
            bge_results  = bge_results,
        )

        # Convert back to LangChain Documents
        return [
            Document(
                page_content = r.get("text", ""),
                metadata     = {
                    k: v for k, v in r.items() if k != "text"
                },
            )
            for r in final_results
        ]

    def get_relevant_documents(self, query: str) -> List[Any]:
        """Alias for older LangChain versions."""
        return self.invoke(query)


# ── Convenience wrapper for LangGraph N09 node ───────────────────────────────

def run_rrf_reranker(state, use_reranker: bool = True) -> object:
    """
    Convenience wrapper used by the LangGraph pipeline node N09.

    Args:
        state        : BAState object
        use_reranker : If False, skip cross-encoder (RRF only, faster)

    Returns:
        BAState with retrieval_stage_2 populated
    """
    node = RRFReranker(
        final_top_k  = 3,
        rrf_k        = _RRF_K,
        use_reranker = use_reranker,
    )
    return node.run(state)