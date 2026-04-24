"""
src/retrieval/bm25_retriever.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 Rev1.0 FINAL

N07 — BM25 Retriever Tier 2
Activates when SniperRAG confidence < 0.95.

Responsibilities:
  1. Load BM25 index built by N03 Chunker
  2. Tokenise the analyst query
  3. Search index — return top-10 chunks with BM25 scores
  4. Normalise scores to 0.0-1.0 range
  5. Write results to BAState.bm25_results

Also wraps as LangChain BM25Retriever for N09 EnsembleRetriever.

Speed: <100ms · Zero GPU · Pure keyword matching
Writes to BAState: bm25_results
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.state.ba_state import BAState
from src.utils.resource_governor import ResourceGovernor
from src.utils.seed_manager import SeedManager

SeedManager.set_all()

TOP_K = 10


class BM25Retriever:
    """
    N07: BM25 Keyword Retriever.
    Loads BM25 index from N03. Searches with analyst query.
    Returns top-10 chunks with normalised scores.
    LangChain integration for N09 EnsembleRetriever.
    """

    def __init__(self, top_k: int = TOP_K):
        SeedManager.set_all()
        self.top_k         = top_k
        self._retriever    = None
        self._chunks       = []
        self._lc_retriever = None

    def run(self, state: BAState) -> BAState:
        """Main entry point."""
        ResourceGovernor.check("N07 BM25 Retriever")

        if not state.bm25_index_path:
            print("[N07] No BM25 index path — skipping")
            state.bm25_results = []
            return state

        if not state.query:
            print("[N07] No query — skipping")
            state.bm25_results = []
            return state

        self._load_index(state.bm25_index_path)

        if not self._chunks:
            print("[N07] Empty index — skipping")
            state.bm25_results = []
            return state

        results = self._search(state.query)
        state.bm25_results = results

        print(f"[N07] BM25 search: '{state.query[:50]}' "
              f"→ {len(results)} results")
        if results:
            print(f"[N07] Top score: {results[0]['bm25_score']:.4f} | "
                  f"section: {results[0].get('section', 'unknown')}")

        return state

    # ═══════════════════════════════════════════════════════════════════════
    # INDEX LOADING
    # ═══════════════════════════════════════════════════════════════════════

    def _load_index(self, index_path: str) -> None:
        """Load BM25 index and chunk metadata from disk."""
        import bm25s

        index_dir = Path(index_path)
        meta_path = index_dir / "chunks_meta.json"

        if not index_dir.exists():
            print(f"[N07] Index not found: {index_dir}")
            return
        if not meta_path.exists():
            print(f"[N07] chunks_meta.json not found")
            return

        with open(meta_path, "r", encoding="utf-8") as f:
            self._chunks = json.load(f)

        self._retriever = bm25s.BM25.load(
            str(index_dir), load_corpus=True
        )
        print(f"[N07] Loaded BM25 index: {len(self._chunks)} chunks")

    # ═══════════════════════════════════════════════════════════════════════
    # SEARCH
    # ═══════════════════════════════════════════════════════════════════════

    def _search(self, query: str) -> List[Dict[str, Any]]:
        """Search BM25 index. Returns top-k chunks with scores."""
        import bm25s

        if self._retriever is None or not self._chunks:
            return []

        query_tokens = bm25s.tokenize([query], stopwords="en")

                # bm25s crashes when k == len(corpus) on small corpora
        # Safe k = min(top_k, corpus_size - 1) but at least 1
        safe_k = max(1, min(self.top_k, len(self._chunks) - 1)) \
                 if len(self._chunks) > 1 else 1
        safe_k = min(safe_k, len(self._chunks))

        try:
            results, scores = self._retriever.retrieve(
                query_tokens, k=safe_k
            )
        except Exception as e:
            print(f"[N07] BM25 search error: {e}")
            # Retry with k=1 as last resort
            try:
                results, scores = self._retriever.retrieve(
                    query_tokens, k=1
                )
            except Exception:
                return []

        # bm25s version compatibility: handle both 1D and 2D return shapes
        import numpy as _np
        _r = _np.asarray(results, dtype=object)
        _s = _np.asarray(scores)
        if _r.ndim >= 2:
            top_results = list(_r[0]) if len(_r) > 0 else []
            top_scores  = list(_s[0]) if len(_s) > 0 else []
        else:
            top_results = list(_r)
            top_scores  = list(_s)
        if len(top_results) == 0:
            print("[N07] BM25: empty result set")
            return []
        max_score = float(max(top_scores)) if len(top_scores) > 0 else 1.0
        if max_score == 0:
            max_score = 1.0

        output = []
        for i, (result, score) in enumerate(zip(top_results, top_scores)):
            chunk_id   = (result.get("id", f"chunk_{i:06d}")
                          if isinstance(result, dict) else str(result))
            chunk_meta = self._find_chunk_meta(chunk_id)
            if chunk_meta is None:
                idx        = min(i, len(self._chunks) - 1)
                chunk_meta = self._chunks[idx]

            norm_score = float(score) / max_score
            output.append({
                **chunk_meta,
                "bm25_score":      round(float(score), 6),
                "bm25_score_norm": round(norm_score, 6),
                "rank":            i + 1,
                "retriever":       "bm25",
            })

        output.sort(key=lambda x: x["bm25_score"], reverse=True)
        return output

    def _find_chunk_meta(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Find chunk metadata by chunk_id."""
        for chunk in self._chunks:
            if chunk.get("chunk_id") == chunk_id:
                return chunk
        return None

    # ═══════════════════════════════════════════════════════════════════════
    # LANGCHAIN INTEGRATION
    # ═══════════════════════════════════════════════════════════════════════

    def as_langchain_retriever(self, index_path: str):
        """
        Return a LangChain-compatible BM25Retriever.
        Used by N09 EnsembleRetriever.
        Requires: pip install rank-bm25
        """
        from langchain_community.retrievers import BM25Retriever as LCBm25
        from langchain_core.documents import Document

        if not self._chunks:
            self._load_index(index_path)

        if not self._chunks:
            return None

        docs = [
            Document(
                page_content=chunk["text"],
                metadata={
                    "chunk_id":    chunk.get("chunk_id", ""),
                    "company":     chunk.get("company", ""),
                    "doc_type":    chunk.get("doc_type", ""),
                    "fiscal_year": chunk.get("fiscal_year", ""),
                    "section":     chunk.get("section", ""),
                    "page":        str(chunk.get("page", "")),
                }
            )
            for chunk in self._chunks
        ]

        lc_retriever       = LCBm25.from_documents(docs, k=self.top_k)
        self._lc_retriever = lc_retriever
        return lc_retriever

    # ═══════════════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════════════

    def get_chunk_count(self) -> int:
        return len(self._chunks)

    def is_loaded(self) -> bool:
        return self._retriever is not None and len(self._chunks) > 0

    def search_direct(
        self, query: str, index_path: str, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Convenience: load index and search in one call."""
        self._load_index(index_path)
        self.top_k = top_k
        return self._search(query)


# ═══════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK
# run: python src/retrieval/bm25_retriever.py
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        from rich import print as rprint
    except ImportError:
        rprint = print

    import os
    import shutil

    rprint("\n[bold cyan]── BM25Retriever sanity check ──[/bold cyan]")

    retriever = BM25Retriever(top_k=5)
    rprint("[green]✓[/green] BM25Retriever instantiated")

    from src.ingestion.chunker import Chunker
    from src.state.ba_state import BAState

    # Use a fixed temp dir to avoid Windows ChromaDB lock issues
    tmp_dir = Path("data") / "tmp_sanity_n07"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        chunker = Chunker(data_dir=str(tmp_dir))

        sample_text = """
Financial Statements

Net sales were 383285 million dollars in fiscal year 2023.
Net income was 96995 million dollars for the year ended September 2023.
Diluted earnings per share were 6.13 dollars in fiscal 2023.
Total assets were 352583 million dollars as of September 2023.
Gross margin was 169148 million representing 44.1 percent of net sales.
Operating income was 114301 million dollars in fiscal year 2023.

Risk Factors

Competition in each of the Company markets is intense.
The Company faces competition from companies with greater resources.
Changes in global economic conditions could affect demand for products.
        """

        state = BAState(
            session_id   = "sanity-n07",
            company_name = "Apple Inc",
            doc_type     = "10-K",
            fiscal_year  = "FY2023",
            raw_text     = sample_text,
            section_tree = {}
        )
        state = chunker.run(state)
        rprint(f"[green]✓[/green] Test index: {state.chunk_count} chunks")

        # Test search_direct
        results = retriever.search_direct(
            "What was net income in 2023?",
            state.bm25_index_path,
            top_k=5
        )
        assert len(results) > 0
        assert "bm25_score" in results[0]
        assert "rank"       in results[0]
        rprint(f"[green]✓[/green] search_direct: "
               f"{len(results)} results, top score={results[0]['bm25_score']:.4f}")

        # Test score normalisation
        for r in results:
            assert 0.0 <= r["bm25_score_norm"] <= 1.0
        rprint("[green]✓[/green] Scores normalised 0-1")

        # Test sorted descending
        for i in range(len(results) - 1):
            assert results[i]["bm25_score"] >= results[i+1]["bm25_score"]
        rprint("[green]✓[/green] Results ranked by score")

        # Test run() via BAState
        state.query = "What was diluted EPS in fiscal 2023?"
        state       = retriever.run(state)
        assert len(state.bm25_results) > 0
        rprint(f"[green]✓[/green] run(): {len(state.bm25_results)} results in BAState")

        # Test LangChain retriever
        lc_ret = retriever.as_langchain_retriever(state.bm25_index_path)
        assert lc_ret is not None
        lc_docs = lc_ret.invoke("net income 2023")
        assert len(lc_docs) > 0
        rprint(f"[green]✓[/green] LangChain retriever: {len(lc_docs)} docs")

    finally:
        # Safe cleanup — close ChromaDB before deleting
        try:
            import chromadb
            client = chromadb.PersistentClient(
                path=str(tmp_dir / "chromadb" / "sanity-n07")
            )
            client.reset()
        except Exception:
            pass
        # Give Windows a moment to release file handles
        import time
        time.sleep(0.5)
        shutil.rmtree(str(tmp_dir), ignore_errors=True)

    rprint(f"\n[bold green]All checks passed. BM25Retriever ready.[/bold green]\n")