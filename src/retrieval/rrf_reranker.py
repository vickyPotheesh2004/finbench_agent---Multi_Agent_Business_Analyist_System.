"""
src/retrieval/rrf_reranker.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 Rev1.0 FINAL

N09 — RRF Merge + Cross-Encoder Reranker Tier 4
Final retrieval node. Runs after N07 BM25 and N08 BGE-M3.

Responsibilities:
  1. Take BM25 top-10 (from state.bm25_results)
  2. Take BGE-M3 top-10 (from state.retrieval_stage_1)
  3. Merge using Reciprocal Rank Fusion (RRF) — 8 lines of math
  4. Cross-encoder reranker scores merged top-10
  5. Return final top-3 → state.retrieval_stage_2

RRF formula (Cormack & Lynam 2009):
  score(chunk) = Σ 1/(k + rank_i)  where k=60
  Rewards chunks appearing HIGH in BOTH BM25 and BGE lists.

Cross-encoder:
  sentence-transformers CrossEncoder
  Scores (query, chunk) pairs more accurately than bi-encoder
  Runs on merged top-10 only — not full corpus

Speed: ~800ms total
Writes to BAState: retrieval_stage_2 (final top-3 chunks)
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.state.ba_state import BAState
from src.utils.resource_governor import ResourceGovernor
from src.utils.seed_manager import SeedManager

SeedManager.set_all()

# ── Config ────────────────────────────────────────────────────────────────────
RRF_K          = 60    # RRF constant — standard value from original paper
TOP_K_MERGE    = 10    # how many to merge before reranking
FINAL_TOP_K    = 3     # final chunks passed to PIV pods

# Cross-encoder model — lightweight, no fine-tuning needed
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RRFReranker:
    """
    N09: RRF Merge + Cross-Encoder Reranker.

    Merges BM25 + BGE results using Reciprocal Rank Fusion.
    Cross-encoder reranks merged top-10 → final top-3.
    Writes retrieval_stage_2 to BAState.
    """

    def __init__(
        self,
        rrf_k:       int = RRF_K,
        top_k_merge: int = TOP_K_MERGE,
        final_top_k: int = FINAL_TOP_K,
    ):
        SeedManager.set_all()
        self.rrf_k       = rrf_k
        self.top_k_merge = top_k_merge
        self.final_top_k = final_top_k
        self._cross_encoder = None  # lazy load

    def run(self, state: BAState) -> BAState:
        """
        Main entry point.
        Reads state.bm25_results + state.retrieval_stage_1.
        Writes state.retrieval_stage_2 (final top-3).
        """
        ResourceGovernor.check("N09 RRF Reranker")

        bm25_results = state.bm25_results        or []
        bge_results  = state.retrieval_stage_1   or []

        if not bm25_results and not bge_results:
            print("[N09] No results from N07 or N08 — skipping")
            state.retrieval_stage_2 = []
            return state

        if not state.query:
            print("[N09] No query — skipping")
            state.retrieval_stage_2 = []
            return state

        # Step 1: RRF merge
        merged = self._reciprocal_rank_fusion(bm25_results, bge_results)
        print(f"[N09] RRF merged: {len(merged)} unique chunks")

        # Step 2: Cross-encoder rerank
        reranked = self._cross_encode_rerank(state.query, merged)
        print(f"[N09] Reranked → top-{self.final_top_k} selected")

        # Step 3: Final top-3
        final = reranked[: self.final_top_k]
        state.retrieval_stage_2 = final

        if final:
            print(f"[N09] Top chunk score: "
                  f"{final[0].get('rerank_score', 0):.4f} | "
                  f"section: {final[0].get('section', 'unknown')}")

        return state

    # ═══════════════════════════════════════════════════════════════════════
    # RRF — RECIPROCAL RANK FUSION
    # ═══════════════════════════════════════════════════════════════════════

    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[Dict[str, Any]],
        bge_results:  List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion — Cormack & Lynam 2009.
        score(chunk) = Σ 1/(k + rank_i)

        k=60 is the standard value from the original paper.
        Rewards chunks that appear high in BOTH lists.
        Pure Python — 8 lines of logic, no library needed.
        """
        rrf_scores: Dict[str, float]        = {}
        chunk_map:  Dict[str, Dict[str, Any]] = {}

        # Process BM25 list
        for rank, chunk in enumerate(bm25_results, start=1):
            cid = chunk.get("chunk_id", f"bm25_{rank}")
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
            if cid not in chunk_map:
                chunk_map[cid] = {**chunk, "sources": ["bm25"]}
            else:
                chunk_map[cid]["sources"].append("bm25")

        # Process BGE list
        for rank, chunk in enumerate(bge_results, start=1):
            cid = chunk.get("chunk_id", f"bge_{rank}")
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
            if cid not in chunk_map:
                chunk_map[cid] = {**chunk, "sources": ["bge_m3"]}
            else:
                if "bge_m3" not in chunk_map[cid].get("sources", []):
                    chunk_map[cid]["sources"].append("bge_m3")

        # Sort by RRF score descending
        sorted_ids = sorted(
            rrf_scores.keys(),
            key=lambda cid: rrf_scores[cid],
            reverse=True
        )

        # Build merged list with RRF scores
        merged = []
        for i, cid in enumerate(sorted_ids[: self.top_k_merge]):
            chunk = chunk_map[cid].copy()
            chunk["rrf_score"]    = round(rrf_scores[cid], 8)
            chunk["rrf_rank"]     = i + 1
            chunk["in_both"]      = len(chunk.get("sources", [])) > 1
            chunk["retriever"]    = "rrf"
            merged.append(chunk)

        return merged

    # ═══════════════════════════════════════════════════════════════════════
    # CROSS-ENCODER RERANKER
    # ═══════════════════════════════════════════════════════════════════════

    def _load_cross_encoder(self) -> None:
        """Lazy-load cross-encoder model."""
        if self._cross_encoder is not None:
            return
        ResourceGovernor.check("N09 cross-encoder load")
        print(f"[N09] Loading cross-encoder: {CROSS_ENCODER_MODEL}")
        from sentence_transformers import CrossEncoder
        self._cross_encoder = CrossEncoder(
            CROSS_ENCODER_MODEL,
            max_length=512,
        )
        print(f"[N09] Cross-encoder loaded")

    def _cross_encode_rerank(
        self,
        query:  str,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Cross-encoder reranks merged chunks.
        Takes (query, chunk_text) pairs → scores each pair.
        Much more accurate than bi-encoder similarity.
        Runs only on top-10 merged chunks — not full corpus.
        """
        if not chunks:
            return []

        # If only 1-2 chunks — skip reranker, return as-is
        if len(chunks) <= 2:
            for i, chunk in enumerate(chunks):
                chunk["rerank_score"] = chunk.get("rrf_score", 0.5)
                chunk["final_rank"]   = i + 1
            return chunks

        self._load_cross_encoder()
        ResourceGovernor.check("N09 cross-encoder scoring")

        # Build (query, text) pairs
        pairs = [
            (query, chunk.get("text", chunk.get("content", "")))
            for chunk in chunks
        ]

        # Score all pairs
        scores = self._cross_encoder.predict(pairs, show_progress_bar=False)

        # Attach scores and sort
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        reranked = sorted(
            chunks,
            key=lambda c: c["rerank_score"],
            reverse=True
        )

        # Assign final ranks
        for i, chunk in enumerate(reranked):
            chunk["final_rank"] = i + 1
            chunk["retriever"]  = "rrf_reranked"

        return reranked

    # ═══════════════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════════════

    def rrf_only(
        self,
        bm25_results: List[Dict[str, Any]],
        bge_results:  List[Dict[str, Any]],
        top_k:        int = 3,
    ) -> List[Dict[str, Any]]:
        """
        RRF merge only — no cross-encoder.
        Used when cross-encoder is not available or for speed.
        """
        merged = self._reciprocal_rank_fusion(bm25_results, bge_results)
        for i, chunk in enumerate(merged[:top_k]):
            chunk["rerank_score"] = chunk["rrf_score"]
            chunk["final_rank"]   = i + 1
        return merged[:top_k]

    def is_cross_encoder_loaded(self) -> bool:
        return self._cross_encoder is not None


# ═══════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK
# run: python src/retrieval/rrf_reranker.py
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        from rich import print as rprint
    except ImportError:
        rprint = print

    rprint("\n[bold cyan]── RRFReranker sanity check ──[/bold cyan]")

    reranker = RRFReranker()
    rprint("[green]✓[/green] RRFReranker instantiated")

    # Mock BM25 results
    bm25_results = [
        {"chunk_id": "chunk_000001", "text": "Apple / 10-K / FY2023 / Financial Statements / 42\nNet income was 96995 million dollars in fiscal 2023.",
         "section": "Financial Statements", "company": "Apple Inc", "doc_type": "10-K",
         "fiscal_year": "FY2023", "page": "42", "bm25_score": 0.91, "bm25_score_norm": 1.0, "rank": 1},
        {"chunk_id": "chunk_000002", "text": "Apple / 10-K / FY2023 / MD&A / 28\nNet sales were 383285 million in fiscal 2023.",
         "section": "MD&A", "company": "Apple Inc", "doc_type": "10-K",
         "fiscal_year": "FY2023", "page": "28", "bm25_score": 0.72, "bm25_score_norm": 0.79, "rank": 2},
        {"chunk_id": "chunk_000003", "text": "Apple / 10-K / FY2023 / Business Overview / 3\nApple designs smartphones and computers.",
         "section": "Business Overview", "company": "Apple Inc", "doc_type": "10-K",
         "fiscal_year": "FY2023", "page": "3", "bm25_score": 0.45, "bm25_score_norm": 0.49, "rank": 3},
    ]

    # Mock BGE results — different order
    bge_results = [
        {"chunk_id": "chunk_000002", "text": "Apple / 10-K / FY2023 / MD&A / 28\nNet sales were 383285 million in fiscal 2023.",
         "section": "MD&A", "company": "Apple Inc", "doc_type": "10-K",
         "fiscal_year": "FY2023", "page": "28", "semantic_score": 0.88, "distance": 0.24, "rank": 1},
        {"chunk_id": "chunk_000001", "text": "Apple / 10-K / FY2023 / Financial Statements / 42\nNet income was 96995 million dollars in fiscal 2023.",
         "section": "Financial Statements", "company": "Apple Inc", "doc_type": "10-K",
         "fiscal_year": "FY2023", "page": "42", "semantic_score": 0.85, "distance": 0.30, "rank": 2},
        {"chunk_id": "chunk_000004", "text": "Apple / 10-K / FY2023 / Risk Factors / 8\nCompetition in markets is intense.",
         "section": "Risk Factors", "company": "Apple Inc", "doc_type": "10-K",
         "fiscal_year": "FY2023", "page": "8", "semantic_score": 0.62, "distance": 0.76, "rank": 3},
    ]

    # Test 1: RRF only (no cross-encoder)
    merged = reranker.rrf_only(bm25_results, bge_results, top_k=3)
    assert len(merged) > 0
    assert "rrf_score" in merged[0]
    rprint(f"[green]✓[/green] RRF merge: {len(merged)} chunks")
    rprint(f"[green]✓[/green] Top chunk: '{merged[0]['section']}' "
           f"rrf_score={merged[0]['rrf_score']:.6f} "
           f"in_both={merged[0].get('in_both')}")

    # Verify chunk appearing in both lists gets highest RRF score
    top_id = merged[0]["chunk_id"]
    assert merged[0].get("in_both") is True, \
        "Chunk in both lists should have highest RRF score"
    rprint(f"[green]✓[/green] Chunk in both lists ranked highest (RRF working)")

    # Test 2: Full pipeline with cross-encoder
    state = BAState(
        session_id        = "sanity-n09",
        query             = "What was Apple net income in 2023?",
        bm25_results      = bm25_results,
        retrieval_stage_1 = bge_results,
    )
    state = reranker.run(state)
    assert isinstance(state.retrieval_stage_2, list)
    assert len(state.retrieval_stage_2) > 0
    assert len(state.retrieval_stage_2) <= 3
    rprint(f"[green]✓[/green] run(): {len(state.retrieval_stage_2)} final chunks")

    # Check required fields
    for chunk in state.retrieval_stage_2:
        assert "rerank_score" in chunk
        assert "final_rank"   in chunk
        assert "retriever"    in chunk
    rprint(f"[green]✓[/green] All required fields present")

    # Check ranked correctly
    scores = [c["rerank_score"] for c in state.retrieval_stage_2]
    assert scores == sorted(scores, reverse=True)
    rprint(f"[green]✓[/green] Results ranked by rerank_score descending")

    # Test 3: Empty inputs
    empty_state = BAState(
        session_id        = "empty-n09",
        query             = "net income",
        bm25_results      = [],
        retrieval_stage_1 = [],
    )
    empty_state = reranker.run(empty_state)
    assert empty_state.retrieval_stage_2 == []
    rprint(f"[green]✓[/green] Empty inputs handled correctly")

    # Test 4: RRF score formula
    scores_map = {c["chunk_id"]: c["rrf_score"] for c in merged}
    # chunk_000001 is rank 1 in BM25 and rank 2 in BGE
    # chunk_000002 is rank 2 in BM25 and rank 1 in BGE
    # Both appear in both lists — should have higher scores than chunk_000003
    assert scores_map.get("chunk_000001", 0) > scores_map.get("chunk_000003", 0)
    assert scores_map.get("chunk_000002", 0) > scores_map.get("chunk_000003", 0)
    rprint(f"[green]✓[/green] RRF formula verified — dual-list chunks score higher")

    rprint(f"\n[bold green]All checks passed. RRFReranker ready.[/bold green]\n")