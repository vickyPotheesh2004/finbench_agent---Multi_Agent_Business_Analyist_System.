"""
src/ingestion/image_processor.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 · Rev 1.0

N01b — Image Processor (sub-module of N01 PDF Ingestor)

Extracts data from images inside financial documents:
    1. Embedded images (charts, infographics) -> Gemma4 multimodal
    2. Scanned pages without text layer       -> pytesseract OCR
    3. Chart -> {label: value} via Gemma4 vision
    4. Auto-detects which pages need OCR

Outputs merge into state.raw_text and state.table_cells.

Constraints:
    C1  $0 cost — pytesseract + Gemma4 local
    C2  100% local — Gemma4 at localhost:11434
    C3  Gemma4 multimodal (gemma4:e4b) — image input supported
    C5  seed=42
    C8  Image-extracted data includes source page (for metadata prefix)
    C9  No _rlef_ in any output
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

SEED                   = 42
MIN_OCR_TEXT_CHARS     = 100      # pages with fewer text chars get OCR'd
MIN_IMAGE_WIDTH        = 200      # ignore tiny images (logos, icons, bullets)
MIN_IMAGE_HEIGHT       = 100
MAX_IMAGES_PER_PAGE    = 10       # safety cap
MAX_IMAGES_PER_DOC     = 100      # C4 memory safety
OCR_DPI                = 200      # DPI for pdf2image rendering
VISION_PROMPT_TEMPLATE = (
    "This is an image from a financial document (page {page}). "
    "If this is a chart, table, or infographic with NUMERICAL data, "
    "extract ALL visible numbers with their labels as JSON: "
    '{{"type": "chart|table|infographic|other", '
    '"data": [{{"label": "<text>", "value": "<number>", "unit": "<unit>"}}], '
    '"title": "<chart title if visible>"}}. '
    "If it's a logo, signature, or decorative image, return "
    '{{"type": "decorative", "data": []}}. '
    "Return ONLY valid JSON, nothing else."
)


class ExtractedImage:
    """A single image extracted from a document."""

    __slots__ = (
        "page_number", "image_index", "width", "height",
        "format", "extraction_method", "ocr_text",
        "vision_data", "image_bytes_b64",
    )

    def __init__(
        self,
        page_number:       int,
        image_index:       int,
        width:             int  = 0,
        height:            int  = 0,
        format:            str  = "png",
        extraction_method: str  = "pymupdf",
        ocr_text:          str  = "",
        vision_data:       Optional[Dict] = None,
        image_bytes_b64:   str  = "",
    ) -> None:
        self.page_number       = page_number
        self.image_index       = image_index
        self.width             = width
        self.height            = height
        self.format            = format
        self.extraction_method = extraction_method
        self.ocr_text          = ocr_text
        self.vision_data       = vision_data or {}
        self.image_bytes_b64   = image_bytes_b64

    @property
    def has_data(self) -> bool:
        """True if this image yielded useful data."""
        return bool(self.ocr_text.strip()) or bool(
            self.vision_data.get("data", [])
        )

    @property
    def is_decorative(self) -> bool:
        return self.vision_data.get("type") == "decorative"

    def to_dict(self) -> Dict:
        return {
            "page":              self.page_number,
            "index":             self.image_index,
            "width":             self.width,
            "height":            self.height,
            "format":            self.format,
            "extraction_method": self.extraction_method,
            "ocr_text":          self.ocr_text,
            "vision_data":       self.vision_data,
            "has_data":          self.has_data,
            "is_decorative":     self.is_decorative,
        }


class ImageProcessor:
    """
    N01b Image Processor.

    Two modes:
        1. extract_images(pdf_path) -> list of ExtractedImage
        2. run(state)               -> augments state.raw_text + table_cells

    Gracefully degrades:
        - No PyMuPDF    -> skips embedded image extraction
        - No Tesseract  -> skips OCR
        - No Gemma4     -> skips vision analysis
        - All-off mode  -> returns empty list (no crash)
    """

    def __init__(
        self,
        enable_ocr:    bool = True,
        enable_vision: bool = True,
        llm_client             = None,
        min_ocr_chars: int  = MIN_OCR_TEXT_CHARS,
        max_images:    int  = MAX_IMAGES_PER_DOC,
    ) -> None:
        self.enable_ocr    = enable_ocr
        self.enable_vision = enable_vision
        self._llm          = llm_client
        self.min_ocr_chars = min_ocr_chars
        self.max_images    = max_images

    # ── LangGraph entry point ─────────────────────────────────────────────────

    def run(self, state) -> object:
        """
        Augment BAState with image-extracted data.
        Merges OCR text into state.raw_text.
        Merges chart/table data into state.table_cells.
        """
        pdf_path = getattr(state, "document_path", "")
        if not pdf_path or not os.path.exists(pdf_path):
            logger.debug("[N01b] No PDF path — skipping image extraction")
            return state

        if not pdf_path.lower().endswith(".pdf"):
            logger.debug("[N01b] Not a PDF (%s) — skipping", pdf_path)
            return state

        images = self.extract_images(pdf_path)

        # Merge OCR text into raw_text
        ocr_additions = self._collect_ocr_text(images)
        if ocr_additions:
            existing          = getattr(state, "raw_text", "") or ""
            state.raw_text    = existing + "\n\n[OCR-EXTRACTED]\n" + ocr_additions
            logger.info(
                "[N01b] Added %d chars of OCR text to raw_text",
                len(ocr_additions),
            )

        # Merge vision data into table_cells
        vision_cells = self._collect_vision_cells(images)
        if vision_cells:
            existing = getattr(state, "table_cells", []) or []
            state.table_cells = existing + vision_cells
            logger.info(
                "[N01b] Added %d cells from chart/image vision",
                len(vision_cells),
            )

        logger.info(
            "[N01b] Processed %d images | OCR=%d | Vision=%d | decorative=%d",
            len(images),
            sum(1 for i in images if i.ocr_text),
            sum(1 for i in images if i.vision_data.get("data")),
            sum(1 for i in images if i.is_decorative),
        )
        return state

    # ── Public image extraction ───────────────────────────────────────────────

    def extract_images(self, pdf_path: str) -> List[ExtractedImage]:
        """
        Extract all images from a PDF.

        Returns:
            List of ExtractedImage. Each may have ocr_text, vision_data,
            or both populated depending on configuration.
        """
        if not os.path.exists(pdf_path):
            logger.warning("[N01b] PDF not found: %s", pdf_path)
            return []

        images: List[ExtractedImage] = []

        # Step 1: Extract embedded images via PyMuPDF
        images.extend(self._extract_embedded_images(pdf_path))

        # Step 2: OCR pages with sparse text layer
        if self.enable_ocr:
            images.extend(self._ocr_sparse_pages(pdf_path))

        # Safety cap (C4)
        images = images[: self.max_images]

        # Step 3: Run Gemma4 vision on extracted embedded images
        if self.enable_vision and self._llm:
            self._apply_vision_analysis(images)

        return images

    # ── Private — embedded image extraction ──────────────────────────────────

    def _extract_embedded_images(
        self, pdf_path: str
    ) -> List[ExtractedImage]:
        """Extract embedded images using PyMuPDF (fitz)."""
        extracted: List[ExtractedImage] = []
        try:
            import fitz     # PyMuPDF
        except ImportError:
            logger.warning("[N01b] PyMuPDF not installed — skipping embedded")
            return extracted

        try:
            doc = fitz.open(pdf_path)
        except Exception as exc:
            logger.error("[N01b] Failed to open PDF: %s", exc)
            return extracted

        try:
            for page_num in range(len(doc)):
                if len(extracted) >= self.max_images:
                    break

                page        = doc[page_num]
                image_list  = page.get_images(full=True)[: MAX_IMAGES_PER_PAGE]

                for idx, img_info in enumerate(image_list):
                    xref = img_info[0]
                    try:
                        pix   = fitz.Pixmap(doc, xref)
                        width, height = pix.width, pix.height

                        if width  < MIN_IMAGE_WIDTH:  continue
                        if height < MIN_IMAGE_HEIGHT: continue

                        # Convert CMYK to RGB if needed
                        if pix.n - pix.alpha >= 4:
                            pix = fitz.Pixmap(fitz.csRGB, pix)

                        img_bytes = pix.tobytes("png")
                        b64       = base64.b64encode(img_bytes).decode("ascii")

                        extracted.append(ExtractedImage(
                            page_number       = page_num + 1,
                            image_index       = idx,
                            width             = width,
                            height            = height,
                            format            = "png",
                            extraction_method = "pymupdf_embedded",
                            image_bytes_b64   = b64,
                        ))
                        pix = None
                    except Exception as exc:
                        logger.debug(
                            "[N01b] Page %d img %d failed: %s",
                            page_num + 1, idx, exc,
                        )
        finally:
            doc.close()

        logger.debug(
            "[N01b] Extracted %d embedded images", len(extracted)
        )
        return extracted

    # ── Private — OCR sparse pages ────────────────────────────────────────────

    def _ocr_sparse_pages(self, pdf_path: str) -> List[ExtractedImage]:
        """
        Render pages with sparse text as images and OCR them.
        Catches scanned PDFs where text layer is missing.
        """
        extracted: List[ExtractedImage] = []
        try:
            import fitz
            import pytesseract
            from PIL import Image
        except ImportError as exc:
            logger.warning("[N01b] OCR dependencies missing: %s", exc)
            return extracted

        try:
            doc = fitz.open(pdf_path)
        except Exception:
            return extracted

        try:
            for page_num in range(len(doc)):
                if len(extracted) >= self.max_images:
                    break

                page       = doc[page_num]
                text       = page.get_text("text") or ""
                text_chars = len(text.strip())

                # Only OCR pages with little text
                if text_chars >= self.min_ocr_chars:
                    continue

                try:
                    mat   = fitz.Matrix(OCR_DPI / 72, OCR_DPI / 72)
                    pix   = page.get_pixmap(matrix=mat)
                    img   = Image.open(io.BytesIO(pix.tobytes("png")))

                    ocr_text = pytesseract.image_to_string(img) or ""
                    ocr_text = ocr_text.strip()

                    if len(ocr_text) < 20:
                        continue

                    extracted.append(ExtractedImage(
                        page_number       = page_num + 1,
                        image_index       = 0,
                        width             = pix.width,
                        height            = pix.height,
                        format            = "png",
                        extraction_method = "tesseract_ocr",
                        ocr_text          = ocr_text,
                    ))
                except (pytesseract.TesseractNotFoundError, FileNotFoundError):
                    logger.warning(
                        "[N01b] Tesseract binary not found — skipping OCR"
                    )
                    break
                except Exception as exc:
                    logger.debug("[N01b] OCR page %d failed: %s", page_num + 1, exc)
        finally:
            doc.close()

        logger.debug("[N01b] OCR'd %d sparse-text pages", len(extracted))
        return extracted

    # ── Private — Gemma4 vision analysis ──────────────────────────────────────

    def _apply_vision_analysis(self, images: List[ExtractedImage]) -> None:
        """
        Run Gemma4 multimodal on embedded images to extract chart/table data.
        Only runs on images that don't already have OCR text.
        """
        if not self._llm:
            return

        for img in images:
            if img.ocr_text:
                continue     # OCR already handled text
            if not img.image_bytes_b64:
                continue     # No image data

            prompt = VISION_PROMPT_TEMPLATE.format(page=img.page_number)

            try:
                response = self._call_vision_llm(prompt, img.image_bytes_b64)
                parsed   = self._parse_vision_json(response)
                if parsed:
                    img.vision_data = parsed
            except Exception as exc:
                logger.debug(
                    "[N01b] Vision failed on page %d: %s",
                    img.page_number, exc,
                )

    def _call_vision_llm(self, prompt: str, image_b64: str) -> str:
        """
        Send multimodal prompt to Gemma4.
        Supports Ollama multimodal API ({"prompt": ..., "images": [b64]}).
        """
        # Try the LLM client's multimodal interface first
        if hasattr(self._llm, "chat_with_image"):
            return self._llm.chat_with_image(prompt, image_b64)

        # Fallback: direct Ollama API call
        try:
            import urllib.request
            payload = json.dumps({
                "model":  getattr(self._llm, "model", "gemma4:e4b"),
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "seed":        SEED,
                },
            }).encode("utf-8")

            base = getattr(self._llm, "base_url", "http://localhost:11434")
            req  = urllib.request.Request(
                f"{base}/api/generate",
                data    = payload,
                headers = {"Content-Type": "application/json"},
                method  = "POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "")
        except Exception as exc:
            logger.debug("[N01b] Vision API call failed: %s", exc)
            return ""

    @staticmethod
    def _parse_vision_json(response: str) -> Dict:
        """Parse Gemma4's JSON response, tolerant of markdown fences."""
        if not response:
            return {}
        cleaned = response.strip()
        # Strip markdown fences
        if cleaned.startswith("```"):
            lines   = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned
        # Find first { ... last }
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            cleaned = m.group(0)
        try:
            return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            return {}

    # ── Private — state merging helpers ───────────────────────────────────────

    @staticmethod
    def _collect_ocr_text(images: List[ExtractedImage]) -> str:
        """Concatenate OCR text with page markers."""
        parts = []
        for img in images:
            if img.ocr_text:
                parts.append(
                    f"[Page {img.page_number} OCR]\n{img.ocr_text}"
                )
        return "\n\n".join(parts)

    @staticmethod
    def _collect_vision_cells(
        images: List[ExtractedImage]
    ) -> List[Dict[str, Any]]:
        """Convert vision_data into table_cells format."""
        cells = []
        for img in images:
            if img.is_decorative:
                continue
            data = img.vision_data.get("data", [])
            for item in data:
                label = str(item.get("label", "")).strip()
                value = str(item.get("value", "")).strip()
                unit  = str(item.get("unit",  "")).strip()
                if not label or not value:
                    continue
                cells.append({
                    "source":   "chart_vision",
                    "page":     img.page_number,
                    "label":    label,
                    "value":    value,
                    "unit":     unit,
                    "chart_type": img.vision_data.get("type", "chart"),
                })
        return cells


# ── Convenience wrappers ──────────────────────────────────────────────────────

def extract_images_from_pdf(
    pdf_path: str,
    enable_ocr:    bool = True,
    enable_vision: bool = False,
    llm_client                 = None,
) -> List[ExtractedImage]:
    """One-liner to extract images from a PDF."""
    processor = ImageProcessor(
        enable_ocr    = enable_ocr,
        enable_vision = enable_vision,
        llm_client    = llm_client,
    )
    return processor.extract_images(pdf_path)


def run_image_processor(state, llm_client=None) -> object:
    """LangGraph N01b entry point."""
    return ImageProcessor(
        enable_ocr    = True,
        enable_vision = llm_client is not None,
        llm_client    = llm_client,
    ).run(state)