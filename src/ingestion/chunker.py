"""
src/ingestion/chunker.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 Rev1.0 FINAL

N03 — Chunker + Indexer
Runs ONCE per document, after N01 and N02.

Responsibilities:
  1. Split document at section boundaries (NEVER arbitrary word counts)
  2. Add mandatory C8 metadata prefix to every chunk:
       COMPANY / DOCTYPE / FISCAL_YEAR / SECTION / PAGE
  3. Build BM25 sparse index (bm25s) — for keyword search N07
  4. Build ChromaDB dense index — for semantic search N08
  5. Write chunk_count, bm25_index_path, chromadb_collection to BAState

PDR spec: Chunker.py asserts on every chunk —
rejects any missing the 5 mandatory metadata fields.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.state.ba_state import BAState
from src.utils.resource_governor import ResourceGovernor
from src.utils.seed_manager import SeedManager

SeedManager.set_all()

# ── Chunk size parameters ─────────────────────────────────────────────────────
CHUNK_SIZE_CHARS    = 1000
CHUNK_OVERLAP_CHARS = 200
MIN_CHUNK_CHARS     = 30   # lowered to handle short tabular content


class Chunker:
    """
    N03: Chunker + Indexer.
    Splits at section boundaries → paragraphs → sentences.
    Single-newline fallback for CSV/tabular content.
    Enforces C8 on every chunk.
    Builds BM25 + ChromaDB indexes.
    """

    def __init__(self, data_dir: str = "data"):
        SeedManager.set_all()
        self.data_dir     = Path(data_dir)
        self.bm25_dir     = self.data_dir / "bm25_index"
        self.chromadb_dir = self.data_dir / "chromadb"
        self.bm25_dir.mkdir(parents=True, exist_ok=True)
        self.chromadb_dir.mkdir(parents=True, exist_ok=True)
        self._global_chunk_counter = 0

    def run(self, state: BAState) -> BAState:
        """Main entry point."""
        ResourceGovernor.check("N03 Chunker start")
        self._global_chunk_counter = 0

        if not state.raw_text:
            print("[N03] No raw text — skipping chunking")
            state.chunk_count = 0
            return state

        chunks = self._split_into_chunks(state)
        print(f"[N03] Created {len(chunks)} chunks")

        if not chunks:
            state.chunk_count = 0
            return state

        self._validate_chunks(chunks)

        bm25_path       = self._build_bm25_index(chunks, state)
        print(f"[N03] BM25 index saved: {bm25_path}")

        collection_name = self._build_chromadb(chunks, state)
        print(f"[N03] ChromaDB collection: {collection_name}")

        state.chunk_count         = len(chunks)
        state.bm25_index_path     = str(bm25_path)
        state.chromadb_collection = collection_name

        ResourceGovernor.check("N03 Chunker complete")
        return state

    # ═══════════════════════════════════════════════════════════════════════
    # SPLITTING
    # ═══════════════════════════════════════════════════════════════════════

    def _split_into_chunks(self, state: BAState) -> List[Dict[str, Any]]:
        company     = state.company_name or "UNKNOWN"
        doc_type    = state.doc_type     or "UNKNOWN"
        fiscal_year = state.fiscal_year  or "UNKNOWN"
        raw_text    = state.raw_text
        section_tree = state.section_tree

        if section_tree and section_tree.get("sections"):
            return self._split_by_sections(
                raw_text, section_tree, company, doc_type, fiscal_year
            )
        return self._split_by_paragraphs(
            raw_text, company, doc_type, fiscal_year,
            section="DOCUMENT", page=0
        )

    def _split_by_sections(
        self,
        raw_text:    str,
        section_tree: Dict[str, Any],
        company:     str,
        doc_type:    str,
        fiscal_year: str,
    ) -> List[Dict[str, Any]]:
        chunks   = []
        sections = self._flatten_sections(section_tree.get("sections", []))

        if not sections:
            return self._split_by_paragraphs(
                raw_text, company, doc_type, fiscal_year,
                section="DOCUMENT", page=0
            )

        for i, section in enumerate(sections):
            section_name = section["title"][:60]
            page_start   = section.get("page_start", 0)
            section_text = self._extract_section_text(
                raw_text, section,
                sections[i + 1] if i + 1 < len(sections) else None
            )
            if not section_text.strip():
                continue
            section_chunks = self._split_by_paragraphs(
                section_text, company, doc_type, fiscal_year,
                section=section_name, page=page_start
            )
            chunks.extend(section_chunks)

        return chunks

    def _extract_section_text(
        self,
        raw_text:     str,
        section:      Dict[str, Any],
        next_section: Optional[Dict[str, Any]],
    ) -> str:
        title     = section["title"].strip()
        start_idx = raw_text.lower().find(title.lower())
        if start_idx == -1:
            return ""
        if next_section:
            next_title = next_section["title"].strip()
            end_idx    = raw_text.lower().find(
                next_title.lower(), start_idx + len(title)
            )
            if end_idx == -1:
                end_idx = len(raw_text)
        else:
            end_idx = len(raw_text)
        return raw_text[start_idx:end_idx]

    def _split_by_paragraphs(
        self,
        text:        str,
        company:     str,
        doc_type:    str,
        fiscal_year: str,
        section:     str,
        page:        int,
    ) -> List[Dict[str, Any]]:
        """
        Split text into chunks at paragraph boundaries.
        Falls back to single-newline splitting for tabular/CSV content.
        Falls back to treating entire text as one chunk for very short text.
        """
        chunks = []

        # Try double-newline paragraph split first
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text)
                      if p.strip() and len(p.strip()) >= MIN_CHUNK_CHARS]

        # Fallback 1: single-newline split (CSV, tables, short docs)
        if not paragraphs:
            paragraphs = [p.strip() for p in text.split('\n')
                          if p.strip() and len(p.strip()) >= MIN_CHUNK_CHARS]

        # Fallback 2: treat entire text as one chunk
        if not paragraphs and text.strip() and len(text.strip()) >= MIN_CHUNK_CHARS:
            paragraphs = [text.strip()]

        if not paragraphs:
            return []

        current_chunk = []
        current_size  = 0

        for para in paragraphs:
            # Large paragraph → split by sentence
            if len(para) > CHUNK_SIZE_CHARS * 2:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    if current_size + len(sent) > CHUNK_SIZE_CHARS and current_chunk:
                        chunk_text = " ".join(current_chunk)
                        if len(chunk_text) >= MIN_CHUNK_CHARS:
                            chunks.append(self._make_chunk(
                                chunk_text, company, doc_type,
                                fiscal_year, section, page
                            ))
                        overlap_text  = chunk_text[-CHUNK_OVERLAP_CHARS:]
                        current_chunk = [overlap_text]
                        current_size  = len(overlap_text)
                    current_chunk.append(sent)
                    current_size += len(sent)
            else:
                if current_size + len(para) > CHUNK_SIZE_CHARS and current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    if len(chunk_text) >= MIN_CHUNK_CHARS:
                        chunks.append(self._make_chunk(
                            chunk_text, company, doc_type,
                            fiscal_year, section, page
                        ))
                    overlap_text  = chunk_text[-CHUNK_OVERLAP_CHARS:]
                    current_chunk = [overlap_text]
                    current_size  = len(overlap_text)
                current_chunk.append(para)
                current_size += len(para)

        # Final chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            if len(chunk_text) >= MIN_CHUNK_CHARS:
                chunks.append(self._make_chunk(
                    chunk_text, company, doc_type,
                    fiscal_year, section, page
                ))

        return chunks

    def _make_chunk(
        self,
        text:        str,
        company:     str,
        doc_type:    str,
        fiscal_year: str,
        section:     str,
        page:        int,
    ) -> Dict[str, Any]:
        """Create chunk with C8 mandatory metadata prefix."""
        chunk_id_int = self._global_chunk_counter
        self._global_chunk_counter += 1

        prefix    = f"{company} / {doc_type} / {fiscal_year} / {section} / {page}"
        full_text = f"{prefix}\n{text}"

        return {
            "chunk_id":    f"chunk_{chunk_id_int:06d}",
            "text":        full_text,
            "content":     text,
            "prefix":      prefix,
            "company":     company,
            "doc_type":    doc_type,
            "fiscal_year": fiscal_year,
            "section":     section,
            "page":        page,
            "char_count":  len(full_text),
        }

    def _flatten_sections(
        self, sections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        flat = []
        for section in sections:
            flat.append(section)
            if section.get("children"):
                flat.extend(self._flatten_sections(section["children"]))
        return flat

    # ═══════════════════════════════════════════════════════════════════════
    # C8 VALIDATION
    # ═══════════════════════════════════════════════════════════════════════

    def _validate_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """C8: Assert every chunk has all 5 mandatory metadata fields."""
        required = ["company", "doc_type", "fiscal_year", "section", "page"]
        seen_ids = set()

        for i, chunk in enumerate(chunks):
            for field in required:
                if field not in chunk:
                    raise ValueError(
                        f"[C8 VIOLATION] Chunk {i} missing field '{field}'"
                    )
            if chunk.get("prefix") and chunk["prefix"] not in chunk.get("text", ""):
                raise ValueError(
                    f"[C8 VIOLATION] Chunk {i} prefix not in text"
                )
            cid = chunk["chunk_id"]
            if cid in seen_ids:
                raise ValueError(
                    f"[C8 VIOLATION] Duplicate chunk_id: {cid}"
                )
            seen_ids.add(cid)

    # ═══════════════════════════════════════════════════════════════════════
    # BM25 INDEX
    # ═══════════════════════════════════════════════════════════════════════

    def _build_bm25_index(
        self, chunks: List[Dict[str, Any]], state: BAState
    ) -> Path:
        import bm25s

        session_id = state.session_id or "default"
        index_dir  = self.bm25_dir / session_id
        index_dir.mkdir(parents=True, exist_ok=True)

        corpus        = [chunk["text"] for chunk in chunks]
        corpus_tokens = bm25s.tokenize(corpus, stopwords="en")
        retriever     = bm25s.BM25()
        retriever.index(corpus_tokens)
        retriever.save(str(index_dir))

        meta_path = index_dir / "chunks_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        return index_dir

    # ═══════════════════════════════════════════════════════════════════════
    # CHROMADB INDEX
    # ═══════════════════════════════════════════════════════════════════════

    def _build_chromadb(
        self, chunks: List[Dict[str, Any]], state: BAState
    ) -> str:
        import chromadb

        ResourceGovernor.check("N03 ChromaDB build")

        session_id      = state.session_id or "default"
        collection_name = f"finbench_{session_id}"

        client = chromadb.PersistentClient(
            path=str(self.chromadb_dir / session_id)
        )

        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]
            collection.add(
                ids      = [c["chunk_id"] for c in batch],
                documents= [c["text"]     for c in batch],
                metadatas= [
                    {
                        "company":     c["company"],
                        "doc_type":    c["doc_type"],
                        "fiscal_year": c["fiscal_year"],
                        "section":     str(c["section"]),
                        "page":        str(c["page"]),
                        "char_count":  c["char_count"],
                    }
                    for c in batch
                ]
            )
            ResourceGovernor.check(f"N03 ChromaDB batch {i // batch_size + 1}")

        return collection_name

    # ═══════════════════════════════════════════════════════════════════════
    # QUERY HELPERS
    # ═══════════════════════════════════════════════════════════════════════

    def load_bm25(self, bm25_index_path: str):
        import bm25s
        return bm25s.BM25.load(bm25_index_path, load_corpus=True)

    def load_chunks_meta(self, bm25_index_path: str) -> List[Dict[str, Any]]:
        meta_path = Path(bm25_index_path) / "chunks_meta.json"
        if not meta_path.exists():
            return []
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_chromadb_collection(self, collection_name: str, session_id: str):
        import chromadb
        client = chromadb.PersistentClient(
            path=str(self.chromadb_dir / session_id)
        )
        return client.get_collection(collection_name)


# ═══════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK
# run: python src/ingestion/chunker.py
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        from rich import print as rprint
    except ImportError:
        rprint = print

    rprint("\n[bold cyan]── Chunker sanity check ──[/bold cyan]")

    chunker = Chunker(data_dir="data")
    rprint("[green]✓[/green] Chunker instantiated")

    sample_text = """
Business Overview

Apple Inc designs, manufactures, and markets smartphones, personal computers,
tablets, wearables, and accessories worldwide. The Company also sells various
related services. iPhone is the primary product and represented 52 percent of
total net revenue in fiscal 2023.

Risk Factors

The Company operates in highly competitive markets. Competition in each of
the Company markets is intense and the Company expects it to remain so.
A failure to obtain the content or services our customers desire could impact
our business and financial performance significantly.

Management Discussion and Analysis

The Company total net revenue decreased 3 percent during 2023 compared to 2022.
Products segment net sales were 298.1 billion in 2023. Services net sales were
85.2 billion in fiscal 2023. International net sales accounted for 58 percent.

Financial Statements

Net sales: 383285 million
Cost of sales: 214137 million
Gross margin: 169148 million
Net income: 96995 million
Earnings per share diluted: 6.13

Notes to Financial Statements

Note 1 Summary of Significant Accounting Policies.
The consolidated financial statements include accounts of Apple Inc
and its wholly owned subsidiaries across all regions worldwide.
    """

    state = BAState(
        session_id   = "sanity-n03",
        company_name = "Apple Inc",
        doc_type     = "10-K",
        fiscal_year  = "FY2023",
        raw_text     = sample_text,
        section_tree = {
            "sections": [
                {"id": "sec_0000", "title": "Business Overview",
                 "level": 1, "page_start": 3, "page_end": 7,
                 "is_bold": True, "font_size": 16.0,
                 "summary": "", "children": [], "key_section": None},
                {"id": "sec_0001", "title": "Risk Factors",
                 "level": 1, "page_start": 8, "page_end": 23,
                 "is_bold": True, "font_size": 16.0,
                 "summary": "", "children": [], "key_section": None},
                {"id": "sec_0002",
                 "title": "Management Discussion and Analysis",
                 "level": 1, "page_start": 24, "page_end": 41,
                 "is_bold": True, "font_size": 16.0,
                 "summary": "", "children": [], "key_section": None},
                {"id": "sec_0003", "title": "Financial Statements",
                 "level": 1, "page_start": 42, "page_end": 45,
                 "is_bold": True, "font_size": 16.0,
                 "summary": "", "children": [], "key_section": None},
                {"id": "sec_0004",
                 "title": "Notes to Financial Statements",
                 "level": 1, "page_start": 46, "page_end": 60,
                 "is_bold": True, "font_size": 16.0,
                 "summary": "", "children": [], "key_section": None},
            ],
            "key_sections": {},
            "total_sections": 5,
            "page_count": 60,
        }
    )

    state = chunker.run(state)

    assert state.chunk_count > 0
    rprint(f"[green]✓[/green] Created {state.chunk_count} chunks")

    chunks_meta = chunker.load_chunks_meta(state.bm25_index_path)
    first = chunks_meta[0]
    assert "Apple Inc" in first["prefix"]
    assert "10-K"      in first["prefix"]
    assert "FY2023"    in first["prefix"]
    rprint(f"[green]✓[/green] C8 prefix: {first['prefix']}")

    all_ids = [c["chunk_id"] for c in chunks_meta]
    assert len(all_ids) == len(set(all_ids))
    rprint(f"[green]✓[/green] All chunk IDs unique")

    retriever = chunker.load_bm25(state.bm25_index_path)
    rprint(f"[green]✓[/green] BM25 loads correctly")

    # Test CSV fallback (single newlines)
    csv_state = BAState(
        session_id   = "sanity-n03-csv",
        company_name = "Goldman Sachs",
        doc_type     = "10-K",
        fiscal_year  = "FY2022",
        raw_text     = "Metric\tFY2022\tFY2023\nRevenue\t47.4B\t44.3B\nNet Income\t10.8B\t8.5B\nEPS\t30.06\t22.87",
        section_tree = {}
    )
    csv_state = chunker.run(csv_state)
    assert csv_state.chunk_count > 0
    rprint(f"[green]✓[/green] CSV single-newline fallback works: {csv_state.chunk_count} chunks")

    rprint(f"\n[bold green]All checks passed. Chunker ready.[/bold green]\n")