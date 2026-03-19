"""
src/retrieval/bge_retriever.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 Rev1.0 FINAL

N08 — BGE-M3 Semantic Retriever Tier 3
Runs in PARALLEL with N07 BM25.

Responsibilities:
  1. Load ChromaDB collection built by N03 Chunker
  2. Embed the analyst query using BGE sentence-transformers
  3. Search by cosine similarity — returns top-10 semantic matches
  4. Write results to BAState.retrieval_stage_1

Model used now : BAAI/bge-small-en-v1.5 (fast, ~130MB)
Week 6 upgrade : BAAI/bge-m3 (full model, fine-tuned on FinanceBench pairs)

Writes to BAState:
  retrieval_stage_1 (List of top-10 chunk dicts with semantic scores)
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

BGE_MODEL_NAME   = "BAAI/bge-small-en-v1.5"
TOP_K            = 10
DEFAULT_DATA_DIR = "data"


class BGERetriever:
    """
    N08: BGE Semantic Retriever.
    Embeds queries and searches ChromaDB by cosine similarity.
    Phase 2: base bge-small-en-v1.5.
    Week 6: upgraded to fine-tuned bge-m3.
    """

    def __init__(
        self,
        model_name:   str = BGE_MODEL_NAME,
        top_k:        int = TOP_K,
        data_dir:     str = DEFAULT_DATA_DIR,
    ):
        SeedManager.set_all()
        self.model_name   = model_name
        self.top_k        = top_k
        self.data_dir     = Path(data_dir)
        self._model       = None
        self._client      = None
        self._collection  = None

    def run(self, state: BAState) -> BAState:
        """Main entry point."""
        ResourceGovernor.check("N08 BGE Retriever")

        if not state.chromadb_collection:
            print("[N08] No ChromaDB collection — skipping")
            state.retrieval_stage_1 = []
            return state

        if not state.query:
            print("[N08] No query — skipping")
            state.retrieval_stage_1 = []
            return state

        self._load_model()

        # Load collection if not already loaded
        if self._collection is None:
            chromadb_dir = self.data_dir / "chromadb" / state.session_id
            self._load_collection(
                collection_name = state.chromadb_collection,
                chromadb_dir    = str(chromadb_dir),
            )

        if self._collection is None:
            print("[N08] Collection not found — skipping")
            state.retrieval_stage_1 = []
            return state

        results = self._search(state.query)
        state.retrieval_stage_1 = results

        print(f"[N08] BGE search: '{state.query[:50]}' "
              f"→ {len(results)} results")
        if results:
            print(f"[N08] Top score: {results[0]['semantic_score']:.4f} | "
                  f"section: {results[0].get('section', 'unknown')}")

        return state

    # ═══════════════════════════════════════════════════════════════════════
    # MODEL + COLLECTION LOADING
    # ═══════════════════════════════════════════════════════════════════════

    def _load_model(self) -> None:
        """Lazy-load BGE sentence-transformers model."""
        if self._model is not None:
            return
        ResourceGovernor.check("N08 model load")
        print(f"[N08] Loading model: {self.model_name}")
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name)
        print(f"[N08] Model loaded")

    def _load_collection(
        self,
        collection_name: str,
        chromadb_dir:    str,
    ) -> None:
        """Load ChromaDB collection by name."""
        import chromadb

        db_path = Path(chromadb_dir)
        if not db_path.exists():
            print(f"[N08] ChromaDB dir not found: {db_path}")
            return

        try:
            self._client     = chromadb.PersistentClient(path=str(db_path))
            self._collection = self._client.get_collection(collection_name)
            print(f"[N08] Loaded collection: {collection_name} "
                  f"({self._collection.count()} docs)")
        except Exception as e:
            print(f"[N08] Could not load collection '{collection_name}': {e}")
            self._collection = None

    def load_collection_direct(
        self,
        collection_name: str,
        chromadb_dir:    str,
    ) -> None:
        """Public: load collection from a custom path. Used by tests."""
        self._load_collection(collection_name, chromadb_dir)

    # ═══════════════════════════════════════════════════════════════════════
    # SEARCH
    # ═══════════════════════════════════════════════════════════════════════

    def _search(self, query: str) -> List[Dict[str, Any]]:
        """Embed query and search ChromaDB by cosine similarity."""
        if self._model is None or self._collection is None:
            return []

        ResourceGovernor.check("N08 embedding query")

        query_with_instruction = (
            f"Represent this sentence for searching: {query}"
        )
        query_embedding = self._model.encode(
            [query_with_instruction],
            normalize_embeddings = True,
            show_progress_bar    = False,
        ).tolist()

        try:
            chroma_results = self._collection.query(
                query_embeddings = query_embedding,
                n_results        = min(self.top_k, self._collection.count()),
                include          = ["documents", "metadatas", "distances"],
            )
        except Exception as e:
            print(f"[N08] ChromaDB query error: {e}")
            return []

        documents = chroma_results.get("documents", [[]])[0]
        metadatas = chroma_results.get("metadatas", [[]])[0]
        distances = chroma_results.get("distances", [[]])[0]

        output = []
        for i, (doc, meta, dist) in enumerate(
            zip(documents, metadatas, distances)
        ):
            semantic_score = max(0.0, 1.0 - (dist / 2.0))
            output.append({
                "chunk_id":       meta.get("chunk_id", f"chunk_{i:06d}"),
                "text":           doc,
                "company":        meta.get("company", ""),
                "doc_type":       meta.get("doc_type", ""),
                "fiscal_year":    meta.get("fiscal_year", ""),
                "section":        meta.get("section", ""),
                "page":           meta.get("page", ""),
                "semantic_score": round(semantic_score, 6),
                "distance":       round(float(dist), 6),
                "rank":           i + 1,
                "retriever":      "bge_m3",
            })

        output.sort(key=lambda x: x["semantic_score"], reverse=True)
        return output

    # ═══════════════════════════════════════════════════════════════════════
    # PUBLIC HELPERS
    # ═══════════════════════════════════════════════════════════════════════

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query. Used by N09 cross-encoder."""
        self._load_model()
        vec = self._model.encode(
            [f"Represent this sentence for searching: {query}"],
            normalize_embeddings = True,
            show_progress_bar    = False,
        )
        return vec[0].tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts. Used for Week 6 fine-tuning pairs."""
        self._load_model()
        vecs = self._model.encode(
            texts,
            normalize_embeddings = True,
            show_progress_bar    = False,
            batch_size           = 32,
        )
        return vecs.tolist()

    def search_direct(
        self,
        query:           str,
        collection_name: str,
        chromadb_dir:    str,
        top_k:           int = 10,
    ) -> List[Dict[str, Any]]:
        """Convenience: load model + collection and search in one call."""
        self.top_k = top_k
        self._load_model()
        self._load_collection(collection_name, chromadb_dir)
        return self._search(query)

    def is_loaded(self) -> bool:
        return self._model is not None

    def collection_count(self) -> int:
        if self._collection is None:
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0

    @property
    def model_name_active(self) -> str:
        return self.model_name


# ═══════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK
# run: python src/retrieval/bge_retriever.py
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        from rich import print as rprint
    except ImportError:
        rprint = print

    import shutil
    import time

    rprint("\n[bold cyan]── BGERetriever sanity check ──[/bold cyan]")

    from src.ingestion.chunker import Chunker
    from src.state.ba_state import BAState

    tmp_dir = Path("data") / "tmp_sanity_n08"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Pass tmp_dir as data_dir so run() finds the collection
    retriever = BGERetriever(top_k=5, data_dir=str(tmp_dir))
    rprint(f"[green]✓[/green] BGERetriever instantiated")
    rprint(f"[green]✓[/green] Model: {retriever.model_name}")

    try:
        chunker = Chunker(data_dir=str(tmp_dir))

        sample_text = """
Financial Statements

Net sales were 383285 million dollars in fiscal year 2023.
Net income was 96995 million dollars for the year ended September 2023.
Diluted earnings per share were 6.13 dollars in fiscal 2023.
Total assets were 352583 million dollars as of September 2023.
Gross margin was 169148 million representing 44.1 percent.
Operating income was 114301 million dollars in 2023.

Business Overview

Apple designs smartphones personal computers tablets and wearables.
Services revenue grew to 85200 million dollars in fiscal 2023.
iPhone represented 52 percent of total net revenue in 2023.
International sales accounted for 58 percent of total net sales.
        """

        state = BAState(
            session_id   = "sanity-n08",
            company_name = "Apple Inc",
            doc_type     = "10-K",
            fiscal_year  = "FY2023",
            raw_text     = sample_text,
            section_tree = {}
        )
        state = chunker.run(state)
        rprint(f"[green]✓[/green] Test index: {state.chunk_count} chunks")

        chromadb_dir = str(tmp_dir / "chromadb" / "sanity-n08")

        # Test 1: search_direct
        results = retriever.search_direct(
            query           = "What was net income in 2023?",
            collection_name = state.chromadb_collection,
            chromadb_dir    = chromadb_dir,
            top_k           = 5,
        )
        assert len(results) > 0, "No results returned"
        assert "semantic_score" in results[0]
        assert "rank"           in results[0]
        rprint(f"[green]✓[/green] search_direct: {len(results)} results, "
               f"top score={results[0]['semantic_score']:.4f}")

        # Test 2: scores in range
        for r in results:
            assert 0.0 <= r["semantic_score"] <= 1.0
        rprint("[green]✓[/green] Scores in valid 0-1 range")

        # Test 3: retriever field
        for r in results:
            assert r["retriever"] == "bge_m3"
        rprint("[green]✓[/green] retriever field = 'bge_m3'")

        # Test 4: embed_query
        vec = retriever.embed_query("What was diluted EPS?")
        assert len(vec) > 0 and isinstance(vec[0], float)
        rprint(f"[green]✓[/green] embed_query: vector dim={len(vec)}")

        # Test 5: embed_texts
        vecs = retriever.embed_texts(["net income 2023", "diluted EPS"])
        assert len(vecs) == 2
        rprint(f"[green]✓[/green] embed_texts: 2 vectors, dim={len(vecs[0])}")

        # Test 6: collection_count
        assert retriever.collection_count() > 0
        rprint(f"[green]✓[/green] collection_count()={retriever.collection_count()}")

        # Test 7: run() via BAState — collection already loaded
        state.query = "What was diluted earnings per share?"
        state = retriever.run(state)
        assert isinstance(state.retrieval_stage_1, list)
        assert len(state.retrieval_stage_1) > 0
        rprint(f"[green]✓[/green] run(): "
               f"{len(state.retrieval_stage_1)} results in BAState")

        # Test 8: is_loaded
        assert retriever.is_loaded() is True
        rprint("[green]✓[/green] is_loaded() = True")

    finally:
        try:
            import chromadb
            client = chromadb.PersistentClient(
                path=str(tmp_dir / "chromadb" / "sanity-n08")
            )
            client.reset()
        except Exception:
            pass
        time.sleep(0.5)
        shutil.rmtree(str(tmp_dir), ignore_errors=True)

    rprint(f"\n[bold green]All checks passed. "
           f"BGERetriever ready.[/bold green]\n")