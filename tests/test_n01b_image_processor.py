"""
tests/test_n01b_image_processor.py
Tests for N01b Image Processor
PDR-BAAAI-001 Rev 1.0
40 tests - no network, no real PDFs required
"""
import base64
import json
import os
import pytest
from unittest.mock import patch, MagicMock
from src.ingestion.image_processor import (
    ImageProcessor,
    ExtractedImage,
    extract_images_from_pdf,
    run_image_processor,
    SEED,
    MIN_OCR_TEXT_CHARS,
    MIN_IMAGE_WIDTH,
    MIN_IMAGE_HEIGHT,
    MAX_IMAGES_PER_DOC,
    VISION_PROMPT_TEMPLATE,
)
from src.state.ba_state import BAState


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def processor():
    return ImageProcessor(enable_ocr=False, enable_vision=False)


@pytest.fixture
def mock_llm():
    m = MagicMock()
    m.model    = "gemma4:e4b"
    m.base_url = "http://localhost:11434"
    return m


def _make_image(**kwargs):
    defaults = {
        "page_number":       1,
        "image_index":       0,
        "width":             800,
        "height":            600,
        "format":            "png",
        "extraction_method": "pymupdf_embedded",
    }
    defaults.update(kwargs)
    return ExtractedImage(**defaults)


# ── Group 1: Constants ────────────────────────────────────────────────────────

class TestConstants:

    def test_01_seed_is_42(self):
        assert SEED == 42

    def test_02_min_ocr_chars_defined(self):
        assert MIN_OCR_TEXT_CHARS > 0

    def test_03_min_image_width_defined(self):
        assert MIN_IMAGE_WIDTH > 0

    def test_04_min_image_height_defined(self):
        assert MIN_IMAGE_HEIGHT > 0

    def test_05_max_images_safety_cap(self):
        assert MAX_IMAGES_PER_DOC > 0
        assert MAX_IMAGES_PER_DOC <= 1000

    def test_06_vision_prompt_mentions_json(self):
        assert "JSON" in VISION_PROMPT_TEMPLATE or "json" in VISION_PROMPT_TEMPLATE

    def test_07_vision_prompt_has_page_placeholder(self):
        assert "{page}" in VISION_PROMPT_TEMPLATE


# ── Group 2: ExtractedImage class ────────────────────────────────────────────

class TestExtractedImage:

    def test_08_creates_with_all_fields(self):
        img = _make_image(page_number=5, image_index=2)
        assert img.page_number == 5
        assert img.image_index == 2

    def test_09_has_data_false_when_empty(self):
        img = _make_image()
        assert img.has_data is False

    def test_10_has_data_true_with_ocr(self):
        img = _make_image(ocr_text="some extracted text here")
        assert img.has_data is True

    def test_11_has_data_true_with_vision(self):
        img = _make_image(vision_data={
            "type": "chart",
            "data": [{"label": "Revenue", "value": "383"}],
        })
        assert img.has_data is True

    def test_12_is_decorative_true(self):
        img = _make_image(vision_data={"type": "decorative"})
        assert img.is_decorative is True

    def test_13_is_decorative_false_by_default(self):
        img = _make_image()
        assert img.is_decorative is False

    def test_14_to_dict_has_required_keys(self):
        d = _make_image().to_dict()
        for k in ["page", "index", "width", "height", "format",
                  "extraction_method", "has_data", "is_decorative"]:
            assert k in d


# ── Group 3: Instantiation ────────────────────────────────────────────────────

class TestInstantiation:

    def test_15_default_init(self, processor):
        assert processor.enable_ocr is False

    def test_16_custom_flags(self):
        p = ImageProcessor(enable_ocr=True, enable_vision=True)
        assert p.enable_ocr    is True
        assert p.enable_vision is True

    def test_17_no_llm_by_default(self, processor):
        assert processor._llm is None

    def test_18_llm_wired_when_passed(self, mock_llm):
        p = ImageProcessor(llm_client=mock_llm)
        assert p._llm is mock_llm

    def test_19_min_ocr_chars_custom(self):
        p = ImageProcessor(min_ocr_chars=50)
        assert p.min_ocr_chars == 50


# ── Group 4: Extract images (no PDF) ──────────────────────────────────────────

class TestExtractImages:

    def test_20_nonexistent_pdf_returns_empty(self, processor):
        result = processor.extract_images("does_not_exist.pdf")
        assert result == []

    def test_21_extract_returns_list(self, processor, tmp_path):
        # Create a tiny non-PDF file
        fake = tmp_path / "fake.pdf"
        fake.write_text("not a pdf")
        result = processor.extract_images(str(fake))
        assert isinstance(result, list)

    def test_22_safety_cap_respected(self, processor):
        # Synthetic test — ensure the cap is enforced
        assert processor.max_images == MAX_IMAGES_PER_DOC


# ── Group 5: JSON parsing ─────────────────────────────────────────────────────

class TestVisionJsonParsing:

    def test_23_parse_clean_json(self):
        result = ImageProcessor._parse_vision_json(
            '{"type": "chart", "data": []}'
        )
        assert result["type"] == "chart"

    def test_24_parse_markdown_fenced_json(self):
        raw = '```json\n{"type": "chart", "data": [{"label": "x", "value": "1"}]}\n```'
        result = ImageProcessor._parse_vision_json(raw)
        assert result["type"] == "chart"
        assert len(result["data"]) == 1

    def test_25_parse_json_in_extra_text(self):
        raw = 'Here is the data: {"type": "table", "data": []} end.'
        result = ImageProcessor._parse_vision_json(raw)
        assert result["type"] == "table"

    def test_26_parse_invalid_json_returns_empty(self):
        result = ImageProcessor._parse_vision_json("this is not json")
        assert result == {}

    def test_27_parse_empty_returns_empty(self):
        result = ImageProcessor._parse_vision_json("")
        assert result == {}


# ── Group 6: OCR text collection ─────────────────────────────────────────────

class TestCollectOCRText:

    def test_28_empty_list_returns_empty_string(self):
        assert ImageProcessor._collect_ocr_text([]) == ""

    def test_29_collects_ocr_with_page_markers(self):
        imgs = [
            _make_image(page_number=1, ocr_text="first page text"),
            _make_image(page_number=2, ocr_text="second page text"),
        ]
        result = ImageProcessor._collect_ocr_text(imgs)
        assert "Page 1" in result
        assert "Page 2" in result
        assert "first page text"  in result
        assert "second page text" in result

    def test_30_skips_images_without_ocr(self):
        imgs = [
            _make_image(page_number=1, ocr_text="has text"),
            _make_image(page_number=2, ocr_text=""),
        ]
        result = ImageProcessor._collect_ocr_text(imgs)
        assert "has text" in result
        assert "Page 2"   not in result


# ── Group 7: Vision cell extraction ──────────────────────────────────────────

class TestCollectVisionCells:

    def test_31_empty_returns_empty(self):
        assert ImageProcessor._collect_vision_cells([]) == []

    def test_32_converts_vision_data_to_cells(self):
        imgs = [_make_image(
            page_number = 5,
            vision_data = {
                "type": "chart",
                "data": [
                    {"label": "Revenue", "value": "383285", "unit": "millions"},
                    {"label": "Net Income", "value": "96995", "unit": "millions"},
                ],
            },
        )]
        cells = ImageProcessor._collect_vision_cells(imgs)
        assert len(cells) == 2
        assert cells[0]["label"] == "Revenue"
        assert cells[0]["value"] == "383285"
        assert cells[0]["page"]  == 5

    def test_33_skips_decorative_images(self):
        imgs = [_make_image(vision_data={"type": "decorative", "data": []})]
        assert ImageProcessor._collect_vision_cells(imgs) == []

    def test_34_skips_cells_without_label_or_value(self):
        imgs = [_make_image(vision_data={
            "type": "chart",
            "data": [
                {"label": "", "value": "100"},
                {"label": "Valid", "value": ""},
                {"label": "Good", "value": "200"},
            ],
        })]
        cells = ImageProcessor._collect_vision_cells(imgs)
        assert len(cells) == 1
        assert cells[0]["label"] == "Good"

    def test_35_cells_include_source_marker(self):
        imgs = [_make_image(vision_data={
            "type": "chart",
            "data": [{"label": "X", "value": "1"}],
        })]
        cells = ImageProcessor._collect_vision_cells(imgs)
        assert cells[0]["source"] == "chart_vision"


# ── Group 8: BAState integration ─────────────────────────────────────────────

class TestBAStateIntegration:

    def test_36_no_document_path_returns_state(self, processor):
        state  = BAState(session_id="t36")
        result = processor.run(state)
        assert result is state

    def test_37_nonexistent_file_returns_state(self, processor):
        state = BAState(session_id="t37", document_path="nonexistent.pdf")
        result = processor.run(state)
        assert result is state

    def test_38_non_pdf_returns_state(self, processor, tmp_path):
        docx = tmp_path / "doc.docx"
        docx.write_text("fake")
        state = BAState(session_id="t38", document_path=str(docx))
        result = processor.run(state)
        assert result is state

    def test_39_seed_unchanged(self, processor):
        state  = BAState(session_id="t39")
        result = processor.run(state)
        assert result.seed == 42


# ── Group 9: C9 privacy + convenience ─────────────────────────────────────────

class TestC9AndWrapper:

    def test_40_no_rlef_in_extracted_image_dict(self):
        img = _make_image(
            ocr_text    = "test text",
            vision_data = {"type": "chart", "data": []},
        )
        assert "_rlef_" not in str(img.to_dict())