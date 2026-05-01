"""
src/retrieval/bm25_retriever.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 Rev1.0 FINAL · Session 8 fix

N07 — BM25 Retriever Tier 2
Activates when SniperRAG confidence < 0.95.

CHANGELOG:
  2026-04-30 S7  Bug #1: rewrote _search() for bm25s 0.2+ output shape.
                 Removed np.asarray(dtype=object) crash.
  2026-04-30 S8  Bug #1.1: bm25s 0.2+ has internal reshape bug when
                 k=1 on tiny corpora ("cannot reshape array of size N
                 into shape (1,1)"). Workaround: never call retrieve
                 with k=1; always request k=min(top_k, n_chunks) but
                 fall back to scoring every chunk manually if retrieve
                 fails. Single-chunk corpus path returns that one chunk
                 directly without calling retrieve.
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
    Returns top-k chunks with normalised scores.
    """

    def __init__(self, top_k: int = TOP_K):
        SeedManager.set_all()
        self.top_k         = top_k
        self._retriever    = None
        self._chunks: List[Dict[str, Any]] = []
        self._lc_retriever = None

    # ═══════════════════════════════════════════════════════════════════════
    # MAIN ENTRY POINT (LangGraph N07 node)
    # ═══════════════════════════════════════════════════════════════════════

    def run(self, state: BAState) -> BAState:
        """Reads:  state.bm25_index_path, state.query
        Writes: state.bm25_results, state.bm25_confidence
        """
        ResourceGovernor.check("N07 BM25 Retriever")

        if not state.bm25_index_path:
            print("[N07] No BM25 index path — skipping")
            state.bm25_results    = []
            state.bm25_confidence = 0.0
            return state

        if not state.query:
            print("[N07] No query — skipping")
            state.bm25_results    = []
            state.bm25_confidence = 0.0
            return state

        self._load_index(state.bm25_index_path)

        if not self._chunks:
            print("[N07] Empty index — skipping")
            state.bm25_results    = []
            state.bm25_confidence = 0.0
            return state

        results = self._search(state.query)
        state.bm25_results = results

        state.bm25_confidence = (
            float(results[0]["bm25_score_norm"]) if results else 0.0
        )

        print(
            f"[N07] BM25 search: '{state.query[:50]}' "
            f"→ {len(results)} results"
        )
        if results:
            print(
                f"[N07] Top score: {results[0]['bm25_score']:.4f} | "
                f"section: {results[0].get('section', 'unknown')}"
            )

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
            self._chunks    = []
            self._retriever = None
            return

        if not meta_path.exists():
            print(f"[N07] chunks_meta.json not found in {index_dir}")
            self._chunks    = []
            self._retriever = None
            return

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                self._chunks = json.load(f)
        except Exception as exc:
            print(f"[N07] Failed to load chunks_meta.json: {exc}")
            self._chunks    = []
            self._retriever = None
            return

        try:
            self._retriever = bm25s.BM25.load(
                str(index_dir), load_corpus=True
            )
        except Exception as exc:
            print(f"[N07] Failed to load bm25s index: {exc}")
            self._retriever = None
            return

        print(f"[N07] Loaded BM25 index: {len(self._chunks)} chunks")

    # ═══════════════════════════════════════════════════════════════════════
    # SEARCH
    # ═══════════════════════════════════════════════════════════════════════

    def _search(self, query: str) -> List[Dict[str, Any]]:
        """Search BM25 index. Returns top-k chunks with scores.

        Bug #1.1 workaround: bm25s 0.2+ crashes with
        "cannot reshape array of size N into shape (1,1)" when k=1
        on tiny corpora. We avoid k=1 entirely:
          - if n_chunks == 1: return that chunk directly with score=1.0
          - if n_chunks  >= 2: call retrieve with k=min(top_k, n_chunks)
        """
        import bm25s

        if self._retriever is None or not self._chunks:
            return []

        if not query or not query.strip():
            return []

        n_chunks = len(self._chunks)
        if n_chunks < 1:
            return []

        # ── SINGLE-CHUNK CORPUS: bypass bm25s.retrieve() entirely ──
        if n_chunks == 1:
            chunk_meta = self._chunks[0]
            return [{
                **chunk_meta,
                "bm25_score":      1.0,
                "bm25_score_norm": 1.0,
                "rank":            1,
                "retriever":       "bm25",
            }]

        # ── MULTI-CHUNK CORPUS: tokenise + retrieve ──
        try:
            query_tokens = bm25s.tokenize([query], stopwords="en")
        except Exception as exc:
            print(f"[N07] BM25 tokenise failed: {exc}")
            return []

        # Always request at least 2 to avoid the k=1 reshape bug
        safe_k = min(self.top_k, n_chunks)
        if safe_k < 2:
            safe_k = 2  # we know n_chunks >= 2 here

        try:
            results, scores = self._retriever.retrieve(
                query_tokens, k=safe_k
            )
        except Exception as exc:
            print(f"[N07] BM25 retrieve failed: {exc}")
            return self._fallback_score_all(query)

        # Extract row 0 (we have exactly 1 query)
        try:
            top_results = list(results[0]) if len(results) > 0 else []
            top_scores  = list(scores[0])  if len(scores)  > 0 else []
        except (IndexError, TypeError):
            top_results = list(results) if results is not None else []
            top_scores  = list(scores)  if scores  is not None else []

        if not top_results:
            return []

        max_score = max((float(s) for s in top_scores), default=1.0)
        if max_score <= 0:
            max_score = 1.0

        output: List[Dict[str, Any]] = []
        for rank, (item, score) in enumerate(zip(top_results, top_scores)):
            chunk_meta = self._resolve_chunk(item, rank)
            if chunk_meta is None:
                continue

            score_f = float(score)
            output.append({
                **chunk_meta,
                "bm25_score":      round(score_f, 6),
                "bm25_score_norm": round(score_f / max_score, 6),
                "rank":            rank + 1,
                "retriever":       "bm25",
            })

        output.sort(key=lambda x: x["bm25_score"], reverse=True)
        for i, r in enumerate(output):
            r["rank"] = i + 1

        return output[: self.top_k]

    def _fallback_score_all(self, query: str) -> List[Dict[str, Any]]:
        """Last-resort: when bm25s.retrieve() errors out, score every
        chunk manually using simple term-overlap. Guarantees results
        when retrieve fails on small corpora.
        """
        if not self._chunks:
            return []

        q_terms = set(query.lower().split())
        if not q_terms:
            return []

        scored = []
        for i, chunk in enumerate(self._chunks):
            text   = (chunk.get("text", "") or "").lower()
            t_terms = set(text.split())
            overlap = len(q_terms & t_terms)
            if overlap == 0:
                continue
            scored.append((overlap, i, chunk))

        if not scored:
            return []

        scored.sort(key=lambda x: x[0], reverse=True)
        max_score = float(scored[0][0])
        if max_score <= 0:
            max_score = 1.0

        output: List[Dict[str, Any]] = []
        for rank, (overlap, _idx, chunk) in enumerate(scored[: self.top_k]):
            output.append({
                **chunk,
                "bm25_score":      float(overlap),
                "bm25_score_norm": round(overlap / max_score, 6),
                "rank":            rank + 1,
                "retriever":       "bm25_fallback",
            })

        return output

    # ─────────────────────────────────────────────────────────────────────
    # CHUNK RESOLUTION HELPERS
    # ─────────────────────────────────────────────────────────────────────

    def _resolve_chunk(
        self, item: Any, rank: int
    ) -> Optional[Dict[str, Any]]:
        """Map a bm25s result row back to chunk metadata."""
        n_chunks = len(self._chunks)

        if isinstance(item, dict):
            chunk_id   = item.get("id", "")
            chunk_meta = self._find_chunk_meta(chunk_id) if chunk_id else None
            if chunk_meta is not None:
                return chunk_meta
            if rank < n_chunks:
                return self._chunks[rank]
            return None

        try:
            idx = int(item)
            if 0 <= idx < n_chunks:
                return self._chunks[idx]
        except (ValueError, TypeError):
            pass

        chunk_id   = str(item)
        chunk_meta = self._find_chunk_meta(chunk_id)
        if chunk_meta is not None:
            return chunk_meta

        if rank < n_chunks:
            return self._chunks[rank]
        return None

    def _find_chunk_meta(
        self, chunk_id: str
    ) -> Optional[Dict[str, Any]]:
        if not chunk_id:
            return None
        for chunk in self._chunks:
            if chunk.get("chunk_id") == chunk_id:
                return chunk
        return None

    # ═══════════════════════════════════════════════════════════════════════
    # LANGCHAIN INTEGRATION
    # ═══════════════════════════════════════════════════════════════════════

    def as_langchain_retriever(self, index_path: str):
        from langchain_community.retrievers import BM25Retriever as LCBm25
        from langchain_core.documents import Document

        if not self._chunks:
            self._load_index(index_path)
        if not self._chunks:
            return None

        docs = [
            Document(
                page_content=chunk.get("text", ""),
                metadata={
                    "chunk_id":    chunk.get("chunk_id",    ""),
                    "company":     chunk.get("company",     ""),
                    "doc_type":    chunk.get("doc_type",    ""),
                    "fiscal_year": chunk.get("fiscal_year", ""),
                    "section":     chunk.get("section",     ""),
                    "page":        str(chunk.get("page",    "")),
                },
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
        self._load_index(index_path)
        self.top_k = top_k
        return self._search(query)


# ═══════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK
# run: python src\retrieval\bm25_retriever.py
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        from rich import print as rprint
    except ImportError:
        rprint = print

    import os
    import shutil
    import time

    rprint("\n[bold cyan]── BM25Retriever sanity check ──[/bold cyan]")

    retriever = BM25Retriever(top_k=5)
    rprint("[green]✓[/green] BM25Retriever instantiated")

    from src.ingestion.chunker import Chunker
    from src.state.ba_state import BAState

    tmp_dir = Path("data") / "tmp_sanity_n07"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    old_env = os.environ.get("DISABLE_CHROMADB")
    os.environ["DISABLE_CHROMADB"] = "1"

    try:
        chunker = Chunker(
            bm25_dir     = str(tmp_dir / "bm25"),
            chromadb_dir = str(tmp_dir / "chromadb"),
        )

        # Sample text MUST have multiple distinct paragraphs separated
        # by blank lines, OR a section_tree, otherwise the chunker
        # will produce only 1 chunk.
        sample_text = """Financial Statements

Net sales were 383285 million dollars in fiscal year 2023 for Apple Inc.

Net income was 96995 million dollars for the year ended September 2023.

Diluted earnings per share were 6.13 dollars in fiscal 2023.

Total assets were 352583 million dollars as of September 2023.

Gross margin was 169148 million representing 44.1 percent of net sales.

Operating income was 114301 million dollars in fiscal year 2023.

Risk Factors

Competition in each of the Company markets is intense and substantial.

The Company faces competition from companies with greater resources.

Changes in global economic conditions could affect demand for products."""

        state = BAState(
            session_id   = "sanity-n07",
            company_name = "Apple Inc",
            doc_type     = "10-K",
            fiscal_year  = "FY2023",
            raw_text     = sample_text,
            section_tree = {},
        )
        state = chunker.run(state)
        rprint(f"[green]✓[/green] Test index: {state.chunk_count} chunks")
        assert state.chunk_count >= 2, (
            f"Need ≥2 chunks for proper test, got {state.chunk_count}. "
            f"Check chunker fix in Session 8 File 3."
        )

        # search_direct
        results = retriever.search_direct(
            "What was net income in 2023?",
            state.bm25_index_path,
            top_k=5,
        )
        assert len(results) > 0, "BUG #1: BM25 returned 0 results"
        rprint(
            f"[green]✓[/green] search_direct: {len(results)} results, "
            f"top score={results[0]['bm25_score']:.4f}"
        )

        for r in results:
            assert 0.0 <= r["bm25_score_norm"] <= 1.0
        rprint("[green]✓[/green] Scores normalised 0-1")

        for i in range(len(results) - 1):
            assert (
                results[i]["bm25_score"]
                >= results[i + 1]["bm25_score"]
            )
        rprint("[green]✓[/green] Results ranked by score")

        state.query = "What was diluted EPS in fiscal 2023?"
        state       = retriever.run(state)
        assert len(state.bm25_results) > 0
        rprint(
            f"[green]✓[/green] run(): {len(state.bm25_results)} "
            f"results | conf={state.bm25_confidence:.3f}"
        )

        results_empty = retriever._search("")
        assert results_empty == [], "Empty query should return []"
        rprint("[green]✓[/green] Empty query returns [] (no crash)")

        try:
            lc_ret = retriever.as_langchain_retriever(state.bm25_index_path)
            if lc_ret is not None:
                lc_docs = lc_ret.invoke("net income 2023")
                rprint(
                    f"[green]✓[/green] LangChain retriever: "
                    f"{len(lc_docs)} docs"
                )
            else:
                rprint(
                    "[yellow]⚠[/yellow] LangChain retriever skipped"
                )
        except ImportError:
            rprint(
                "[yellow]⚠[/yellow] LangChain not installed — skipped"
            )

    finally:
        if old_env is None:
            os.environ.pop("DISABLE_CHROMADB", None)
        else:
            os.environ["DISABLE_CHROMADB"] = old_env

        try:
            import chromadb
            client = chromadb.PersistentClient(
                path=str(tmp_dir / "chromadb")
            )
            client.reset()
        except Exception:
            pass
        time.sleep(0.5)
        shutil.rmtree(str(tmp_dir), ignore_errors=True)

    rprint(
        "\n[bold green]All checks passed. "
        "BM25Retriever ready (Bug #1 + #1.1 fixed).[/bold green]\n"
    )