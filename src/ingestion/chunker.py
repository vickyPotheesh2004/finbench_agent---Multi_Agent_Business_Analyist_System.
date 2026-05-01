"""
N03 Chunker + Index Builder — Section-Boundary Chunking
PDR-BAAAI-001 · Rev 1.0 · Node N03

Purpose:
    Splits document text at section boundaries (NEVER at arbitrary
    512-token word boundaries). Adds mandatory 5-field metadata prefix
    to every chunk (C8). Builds bm25s sparse index + ChromaDB collection.

CHANGELOG:
  2026-04-30 S8  Bug #1.5: _chunk_by_paragraphs() was consolidating
                 multiple short paragraphs into a single chunk if total
                 text was small. This produced 1-chunk corpora that
                 broke BM25 retrieval. NEW behaviour: emit one chunk
                 per non-empty paragraph (above MIN_CHUNK_CHARS), only
                 merging when a single paragraph exceeds max_chars.
                 Also: DISABLE_CHROMADB env var honoured (Session 6 patch
                 made permanent).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SEED             = 42
MAX_CHUNK_TOKENS = 800
MIN_CHUNK_CHARS  = 50
MAX_CHUNKS_CAP   = 5000
CHUNK_OVERLAP    = 100


class DocumentChunk:
    """A single document chunk with mandatory 5-field metadata (C8)."""

    __slots__ = (
        "chunk_id", "text", "prefix",
        "company", "doc_type", "fiscal_year",
        "section", "page",
        "char_count", "token_estimate",
    )

    def __init__(
        self,
        chunk_id:    str,
        text:        str,
        company:     str,
        doc_type:    str,
        fiscal_year: str,
        section:     str,
        page:        int,
    ) -> None:
        self.chunk_id      = chunk_id
        self.text          = text
        self.company       = company
        self.doc_type      = doc_type
        self.fiscal_year   = fiscal_year
        self.section       = section
        self.page          = page
        self.char_count    = len(text)
        self.token_estimate= len(text) // 4

        # C8: mandatory 5-field prefix
        self.prefix = f"{company} / {doc_type} / {fiscal_year} / {section} / {page}"

    @property
    def prefixed_text(self) -> str:
        return f"{self.prefix}\n{self.text}"

    def to_dict(self) -> Dict:
        return {
            "chunk_id":       self.chunk_id,
            "text":           self.text,
            "prefix":         self.prefix,
            "company":        self.company,
            "doc_type":       self.doc_type,
            "fiscal_year":    self.fiscal_year,
            "section":        self.section,
            "page":           self.page,
            "char_count":     self.char_count,
            "token_estimate": self.token_estimate,
        }


class Chunker:
    """N03 Chunker + Index Builder."""

    def __init__(
        self,
        bm25_dir:     str = "data/bm25_index",
        chromadb_dir: str = "data/chromadb",
        max_tokens:   int = MAX_CHUNK_TOKENS,
        seed:         int = SEED,
    ) -> None:
        self.bm25_dir     = bm25_dir
        self.chromadb_dir = chromadb_dir
        self.max_tokens   = max_tokens
        self.seed         = seed

    # ── LangGraph pipeline node ───────────────────────────────────────────────

    def run(self, state) -> object:
        raw_text     = getattr(state, "raw_text",      "") or ""
        section_tree = getattr(state, "section_tree",  {}) or {}
        company      = getattr(state, "company_name",  "UNKNOWN") or "UNKNOWN"
        doc_type     = getattr(state, "doc_type",      "UNKNOWN") or "UNKNOWN"
        fiscal_year  = getattr(state, "fiscal_year",   "UNKNOWN") or "UNKNOWN"
        session_id   = getattr(state, "session_id",    "default") or "default"

        if not raw_text:
            logger.warning("N03: empty raw_text — skipping chunking")
            return state

        chunks = self.chunk(
            raw_text     = raw_text,
            section_tree = section_tree,
            company      = company,
            doc_type     = doc_type,
            fiscal_year  = fiscal_year,
        )

        collection_name = f"doc-{session_id[:16]}"
        bm25_path       = os.path.join(self.bm25_dir, session_id[:16])

        self._build_bm25_index(chunks, bm25_path)
        self._build_chromadb_index(chunks, collection_name)

        state.chunk_count          = len(chunks)
        state.bm25_index_path      = bm25_path
        state.chromadb_collection  = collection_name

        logger.info(
            "N03 Chunker: %d chunks | bm25=%s | chromadb=%s",
            len(chunks), bm25_path, collection_name,
        )
        return state

    # ── Core chunk method ─────────────────────────────────────────────────────

    def chunk(
        self,
        raw_text:     str,
        section_tree: Dict,
        company:      str = "UNKNOWN",
        doc_type:     str = "UNKNOWN",
        fiscal_year:  str = "UNKNOWN",
    ) -> List[DocumentChunk]:
        sections = section_tree.get("children", []) if section_tree else []
        chunks: List[DocumentChunk] = []

        if sections:
            chunks = self._chunk_by_sections(
                raw_text, sections, company, doc_type, fiscal_year
            )

        # If sections produced nothing, OR no sections at all,
        # fall back to paragraph chunking
        if not chunks:
            chunks = self._chunk_by_paragraphs(
                raw_text, company, doc_type, fiscal_year
            )

        # Last-ditch: if still nothing (text too short), emit single chunk
        if not chunks and raw_text.strip():
            chunks = [DocumentChunk(
                chunk_id    = "chunk_0000",
                text        = raw_text.strip(),
                company     = company,
                doc_type    = doc_type,
                fiscal_year = fiscal_year,
                section     = "DOCUMENT",
                page        = 0,
            )]

        chunks = chunks[:MAX_CHUNKS_CAP]
        self._assert_metadata_prefixes(chunks)
        return chunks

    # ── Private chunking strategies ───────────────────────────────────────────

    def _chunk_by_sections(
        self,
        raw_text:    str,
        sections:    List[Dict],
        company:     str,
        doc_type:    str,
        fiscal_year: str,
    ) -> List[DocumentChunk]:
        """Split text at section boundaries using the section tree."""
        chunks   = []
        chunk_id = 0

        for section in sections:
            section_name = section.get("name",       "UNKNOWN")
            start_page   = section.get("start_page", 0)

            section_text = self._extract_section_text(
                raw_text, section_name, start_page
            )

            if not section_text or len(section_text) < MIN_CHUNK_CHARS:
                for child in section.get("children", []):
                    child_text = self._extract_section_text(
                        raw_text, child.get("name", ""), child.get("start_page", 0)
                    )
                    if child_text and len(child_text) >= MIN_CHUNK_CHARS:
                        sub_chunks = self._split_large_text(
                            child_text,
                            company, doc_type, fiscal_year,
                            child.get("name", section_name),
                            child.get("start_page", start_page),
                            chunk_id,
                        )
                        chunks.extend(sub_chunks)
                        chunk_id += len(sub_chunks)
                continue

            sub_chunks = self._split_large_text(
                section_text,
                company, doc_type, fiscal_year,
                section_name, start_page, chunk_id,
            )
            chunks.extend(sub_chunks)
            chunk_id += len(sub_chunks)

            for child in section.get("children", []):
                child_text = self._extract_section_text(
                    raw_text, child.get("name", ""), child.get("start_page", 0)
                )
                if child_text and len(child_text) >= MIN_CHUNK_CHARS:
                    child_chunks = self._split_large_text(
                        child_text,
                        company, doc_type, fiscal_year,
                        child.get("name", section_name),
                        child.get("start_page", start_page),
                        chunk_id,
                    )
                    chunks.extend(child_chunks)
                    chunk_id += len(child_chunks)

        return chunks

    def _chunk_by_paragraphs(
        self,
        raw_text:    str,
        company:     str,
        doc_type:    str,
        fiscal_year: str,
    ) -> List[DocumentChunk]:
        """Bug #1.5 fix: emit ONE chunk per non-empty paragraph.
        Only split a single paragraph further if it exceeds max_chars.
        Never consolidate multiple paragraphs into one chunk.
        """
        max_chars  = self.max_tokens * 4
        paragraphs = re.split(r'\n\s*\n', raw_text)
        chunks: List[DocumentChunk] = []
        chunk_id = 0

        # Detect rough section header from first non-empty paragraph
        # (a 1-3 word line is likely a section header)
        current_section = "DOCUMENT"

        for para in paragraphs:
            para = para.strip()
            if not para or len(para) < MIN_CHUNK_CHARS:
                # Possibly a section header — capture if 1-5 words
                wc = len(para.split())
                if 1 <= wc <= 5 and len(para) <= 60:
                    current_section = para[:50]
                continue

            # Long paragraph: split into multiple chunks
            if len(para) > max_chars:
                sub_chunks = self._split_large_text(
                    para,
                    company, doc_type, fiscal_year,
                    current_section, 0, chunk_id,
                )
                chunks.extend(sub_chunks)
                chunk_id += len(sub_chunks)
            else:
                chunks.append(DocumentChunk(
                    chunk_id    = f"chunk_{chunk_id:04d}",
                    text        = para,
                    company     = company,
                    doc_type    = doc_type,
                    fiscal_year = fiscal_year,
                    section     = current_section,
                    page        = 0,
                ))
                chunk_id += 1

        return chunks

    def _split_large_text(
        self,
        text:        str,
        company:     str,
        doc_type:    str,
        fiscal_year: str,
        section:     str,
        page:        int,
        base_id:     int = 0,
    ) -> List[DocumentChunk]:
        """Split a large text block into smaller chunks with overlap."""
        max_chars = self.max_tokens * 4
        chunks    = []

        if len(text) <= max_chars:
            if len(text) >= MIN_CHUNK_CHARS:
                chunks.append(DocumentChunk(
                    chunk_id    = f"chunk_{base_id:04d}",
                    text        = text.strip(),
                    company     = company,
                    doc_type    = doc_type,
                    fiscal_year = fiscal_year,
                    section     = section,
                    page        = page,
                ))
            return chunks

        sentences = re.split(r'(?<=[.!?])\s+', text)
        current   = ""
        idx       = 0

        for sent in sentences:
            if len(current) + len(sent) > max_chars and current:
                if len(current) >= MIN_CHUNK_CHARS:
                    chunks.append(DocumentChunk(
                        chunk_id    = f"chunk_{base_id + idx:04d}",
                        text        = current.strip(),
                        company     = company,
                        doc_type    = doc_type,
                        fiscal_year = fiscal_year,
                        section     = section,
                        page        = page,
                    ))
                    idx += 1
                current = current[-CHUNK_OVERLAP:] + " " + sent
            else:
                current = (current + " " + sent).strip()

        if current and len(current) >= MIN_CHUNK_CHARS:
            chunks.append(DocumentChunk(
                chunk_id    = f"chunk_{base_id + idx:04d}",
                text        = current.strip(),
                company     = company,
                doc_type    = doc_type,
                fiscal_year = fiscal_year,
                section     = section,
                page        = page,
            ))

        return chunks

    @staticmethod
    def _extract_section_text(
        raw_text: str, section_name: str, page: int
    ) -> str:
        if not section_name or not raw_text:
            return ""

        pattern = re.compile(
            re.escape(section_name[:30]),
            re.IGNORECASE,
        )
        m = pattern.search(raw_text)
        if not m:
            return ""

        start = m.start()
        end   = min(start + 3000, len(raw_text))
        return raw_text[start:end].strip()

    @staticmethod
    def _assert_metadata_prefixes(chunks: List[DocumentChunk]) -> None:
        for chunk in chunks:
            if not all([
                chunk.company,
                chunk.doc_type,
                chunk.fiscal_year,
                chunk.section,
                chunk.page is not None,
            ]):
                raise ValueError(
                    f"C8 violation: chunk {chunk.chunk_id} missing metadata. "
                    f"company={chunk.company!r} doc_type={chunk.doc_type!r} "
                    f"fiscal_year={chunk.fiscal_year!r} section={chunk.section!r} "
                    f"page={chunk.page!r}"
                )

    # ── Index builders ────────────────────────────────────────────────────────

    def _build_bm25_index(
        self, chunks: List[DocumentChunk], index_path: str
    ) -> None:
        """Build and save bm25s sparse index from chunks."""
        if not chunks:
            return
        try:
            import bm25s
            os.makedirs(index_path, exist_ok=True)
            corpus    = [c.prefixed_text for c in chunks]
            tokenised = bm25s.tokenize(corpus, stopwords="en")
            retriever = bm25s.BM25(corpus=tokenised)
            retriever.index(tokenised)
            retriever.save(index_path)
            import json as _json
            meta_path = os.path.join(index_path, "chunks_meta.json")
            with open(meta_path, "w", encoding="utf-8") as _f:
                _json.dump([c.to_dict() for c in chunks], _f)
            logger.debug("BM25 index saved: %s (%d docs)", index_path, len(chunks))
        except ImportError:
            logger.warning("bm25s not installed — BM25 index skipped")
        except Exception as exc:
            logger.warning("BM25 index build failed: %s", exc)

    def _build_chromadb_index(
        self, chunks: List[DocumentChunk], collection_name: str
    ) -> None:
        """Build ChromaDB vector collection from chunks.

        Honours DISABLE_CHROMADB=1 env var (Session 6 patch).
        """
        if os.environ.get("DISABLE_CHROMADB"):
            logger.info(
                "ChromaDB index build disabled by DISABLE_CHROMADB env var"
            )
            return
        if not chunks:
            return
        try:
            import chromadb
            from chromadb.utils import embedding_functions
            os.makedirs(self.chromadb_dir, exist_ok=True)
            client = chromadb.PersistentClient(path=self.chromadb_dir)
            bge_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="BAAI/bge-m3",
            )
            collection = client.get_or_create_collection(
                name               = collection_name,
                metadata           = {"hnsw:space": "cosine"},
                embedding_function = bge_ef,
            )
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                collection.add(
                    ids       = [c.chunk_id for c in batch],
                    documents = [c.prefixed_text for c in batch],
                    metadatas = [
                        {
                            "section":     c.section,
                            "page":        c.page,
                            "company":     c.company,
                            "doc_type":    c.doc_type,
                            "fiscal_year": c.fiscal_year,
                        }
                        for c in batch
                    ],
                )
            logger.debug(
                "ChromaDB collection '%s': %d documents",
                collection_name, len(chunks),
            )
        except ImportError:
            logger.warning("chromadb not installed — ChromaDB index skipped")
        except Exception as exc:
            logger.warning("ChromaDB index build failed: %s", exc)


# ── Convenience wrapper ───────────────────────────────────────────────────────

def run_chunker(
    state,
    bm25_dir:     str = "data/bm25_index",
    chromadb_dir: str = "data/chromadb",
) -> object:
    """Convenience wrapper for LangGraph N03 node."""
    return Chunker(bm25_dir=bm25_dir, chromadb_dir=chromadb_dir).run(state)