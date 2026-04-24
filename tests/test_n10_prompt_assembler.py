"""
tests/test_n10_prompt_assembler.py
Tests for N10 Prompt Assembler
PDR-BAAAI-001 · Rev 1.0

C7 constraint: retrieved_context MUST appear before question in 100% of prompts.
"""

import pytest
from src.analysis.prompt_assembler import (
    PromptAssembler,
    run_prompt_assembler,
    assemble_prompt,
    QUERY_TYPES,
    TEMPLATES,
    MAX_CHUNKS,
    MAX_CHUNK_CHARS,
    _CONTEXT_MARKER,
    _QUESTION_MARKER,
)
from src.state.ba_state import BAState


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_chunk(
    text,
    chunk_id    = "chunk_1",
    company     = "Apple Inc",
    doc_type    = "10-K",
    fiscal_year = "FY2023",
    section     = "INCOME_STATEMENT",
    page        = 94,
):
    return {
        "chunk_id":    chunk_id,
        "text":        text,
        "company":     company,
        "doc_type":    doc_type,
        "fiscal_year": fiscal_year,
        "section":     section,
        "page":        page,
        "rank":        1,
        "retriever":   "rrf_reranker",
    }


SAMPLE_CHUNKS = [
    _make_chunk(
        "Total net sales were 383285 million dollars in fiscal year 2023.",
        chunk_id="chunk_1", section="INCOME_STATEMENT", page=94,
    ),
    _make_chunk(
        "Net income was 96995 million dollars for the year ended September 2023.",
        chunk_id="chunk_2", section="INCOME_STATEMENT", page=94,
    ),
    _make_chunk(
        "Total assets were 352583 million dollars as of September 2023.",
        chunk_id="chunk_3", section="BALANCE_SHEET", page=96,
    ),
]


@pytest.fixture
def assembler():
    return PromptAssembler()


# ── Group 1: Constants ────────────────────────────────────────────────────────

class TestConstants:

    def test_01_five_query_types(self):
        assert len(QUERY_TYPES) == 5

    def test_02_query_types_correct(self):
        assert set(QUERY_TYPES) == {
            "numerical", "ratio", "multi_doc", "text", "forensic"
        }

    def test_03_all_templates_present(self):
        for qt in QUERY_TYPES:
            assert qt in TEMPLATES, f"Missing template for '{qt}'"

    def test_04_max_chunks_is_5(self):
        assert MAX_CHUNKS == 5

    def test_05_context_marker_defined(self):
        assert _CONTEXT_MARKER == "RETRIEVED CONTEXT"

    def test_06_question_marker_defined(self):
        assert _QUESTION_MARKER == "QUESTION"


# ── Group 2: Template C7 compliance ──────────────────────────────────────────

class TestTemplateC7Compliance:
    """
    C7: retrieved_context MUST appear before question in ALL templates.
    This is the most critical test group — CI/CD gate enforces this.
    """

    @pytest.mark.parametrize("query_type", QUERY_TYPES)
    def test_07_context_before_question_all_templates(
        self, assembler, query_type
    ):
        """C7: RETRIEVED CONTEXT must appear before QUESTION in every template."""
        prompt = assembler.assemble(
            query        = "What was net income FY2023?",
            chunks       = SAMPLE_CHUNKS,
            query_type   = query_type,
            company_name = "Apple Inc",
            fiscal_year  = "FY2023",
        )
        context_pos  = prompt.find(_CONTEXT_MARKER)
        question_pos = prompt.find(_QUESTION_MARKER)

        assert context_pos != -1, (
            f"Template '{query_type}': RETRIEVED CONTEXT marker missing"
        )
        assert question_pos != -1, (
            f"Template '{query_type}': QUESTION marker missing"
        )
        assert context_pos < question_pos, (
            f"C7 VIOLATION in template '{query_type}': "
            f"QUESTION (pos={question_pos}) before "
            f"RETRIEVED CONTEXT (pos={context_pos})"
        )

    @pytest.mark.parametrize("query_type", QUERY_TYPES)
    def test_08_prompt_contains_query(self, assembler, query_type):
        """Every prompt must contain the actual question text."""
        query  = "What was total net sales FY2023?"
        prompt = assembler.assemble(
            query        = query,
            chunks       = SAMPLE_CHUNKS,
            query_type   = query_type,
            company_name = "Apple Inc",
            fiscal_year  = "FY2023",
        )
        assert query in prompt, (
            f"Template '{query_type}': query text not found in prompt"
        )

    @pytest.mark.parametrize("query_type", QUERY_TYPES)
    def test_09_prompt_contains_chunk_text(self, assembler, query_type):
        """Retrieved chunk text must appear in the assembled prompt."""
        prompt = assembler.assemble(
            query        = "net income",
            chunks       = SAMPLE_CHUNKS,
            query_type   = query_type,
            company_name = "Apple Inc",
            fiscal_year  = "FY2023",
        )
        assert "383285" in prompt or "96995" in prompt or "352583" in prompt, (
            f"Template '{query_type}': chunk text not found in prompt"
        )

    @pytest.mark.parametrize("query_type", QUERY_TYPES)
    def test_10_prompt_contains_company_name(self, assembler, query_type):
        """Company name must appear in the assembled prompt."""
        prompt = assembler.assemble(
            query        = "net income",
            chunks       = SAMPLE_CHUNKS,
            query_type   = query_type,
            company_name = "Apple Inc",
            fiscal_year  = "FY2023",
        )
        assert "Apple Inc" in prompt, (
            f"Template '{query_type}': company name not found in prompt"
        )

    @pytest.mark.parametrize("query_type", QUERY_TYPES)
    def test_11_prompt_contains_fiscal_year(self, assembler, query_type):
        """Fiscal year must appear in the assembled prompt."""
        prompt = assembler.assemble(
            query        = "net income",
            chunks       = SAMPLE_CHUNKS,
            query_type   = query_type,
            company_name = "Apple Inc",
            fiscal_year  = "FY2023",
        )
        assert "FY2023" in prompt, (
            f"Template '{query_type}': fiscal year not found in prompt"
        )


# ── Group 3: C7 violation detection ──────────────────────────────────────────

class TestC7ViolationDetection:

    def test_12_c7_assert_passes_for_valid_prompt(self, assembler):
        """Valid prompt should not raise."""
        prompt = assembler.assemble(
            query        = "net income",
            chunks       = SAMPLE_CHUNKS,
            query_type   = "numerical",
            company_name = "Apple Inc",
            fiscal_year  = "FY2023",
        )
        assert _CONTEXT_MARKER  in prompt
        assert _QUESTION_MARKER in prompt

    def test_13_c7_assert_raises_for_inverted_prompt(self, assembler):
        """If QUESTION appears before RETRIEVED CONTEXT, must raise."""
        with pytest.raises(ValueError, match="C7 VIOLATION"):
            assembler._assert_context_first(
                prompt = "QUESTION: what was revenue?\nRETRIEVED CONTEXT: text",
                query  = "what was revenue?",
            )

    def test_14_c7_assert_raises_if_no_context_marker(self, assembler):
        """Missing RETRIEVED CONTEXT marker must raise."""
        with pytest.raises(ValueError, match="C7 VIOLATION"):
            assembler._assert_context_first(
                prompt = "Some prompt without the context marker\nQUESTION: x",
                query  = "x",
            )

    def test_15_c7_assert_raises_if_no_question_marker(self, assembler):
        """Missing QUESTION marker must raise."""
        with pytest.raises(ValueError, match="C7 VIOLATION"):
            assembler._assert_context_first(
                prompt = "RETRIEVED CONTEXT: text\nSome prompt without question marker",
                query  = "x",
            )


# ── Group 4: Chunk formatting ─────────────────────────────────────────────────

class TestChunkFormatting:

    def test_16_empty_chunks_returns_no_context_message(self, assembler):
        context = assembler._format_chunks([])
        assert "No relevant context" in context

    def test_17_chunks_include_c8_prefix(self, assembler):
        """C8: formatted chunks must include COMPANY/DOCTYPE/FY/SECTION/PAGE."""
        context = assembler._format_chunks(SAMPLE_CHUNKS)
        assert "Apple Inc" in context
        assert "10-K"      in context
        assert "FY2023"    in context

    def test_18_max_chunks_respected(self, assembler):
        """Must not include more than MAX_CHUNKS chunks."""
        many_chunks = [
            _make_chunk(f"chunk text {i}", chunk_id=f"chunk_{i}")
            for i in range(20)
        ]
        context = assembler._format_chunks(many_chunks)
        assert context.count("[CHUNK") <= MAX_CHUNKS

    def test_19_long_chunk_truncated(self, assembler):
        """Chunks exceeding MAX_CHUNK_CHARS must be truncated."""
        long_text = "x" * (MAX_CHUNK_CHARS + 500)
        chunk     = _make_chunk(long_text)
        context   = assembler._format_chunks([chunk])
        assert "truncated" in context

    def test_20_chunk_rank_shown(self, assembler):
        """CHUNK 1, CHUNK 2 etc must appear in formatted context."""
        context = assembler._format_chunks(SAMPLE_CHUNKS)
        assert "CHUNK 1" in context
        assert "CHUNK 2" in context


# ── Group 5: Assemble method ──────────────────────────────────────────────────

class TestAssembleMethod:

    def test_21_returns_non_empty_string(self, assembler):
        prompt = assembler.assemble(
            query  = "What was net income?",
            chunks = SAMPLE_CHUNKS,
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    def test_22_unknown_query_type_falls_back_to_text(self, assembler):
        prompt = assembler.assemble(
            query      = "What was net income?",
            chunks     = SAMPLE_CHUNKS,
            query_type = "unknown_type",
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_23_empty_chunks_still_produces_prompt(self, assembler):
        prompt = assembler.assemble(
            query      = "What was net income?",
            chunks     = [],
            query_type = "numerical",
        )
        assert "No relevant context" in prompt
        assert "RETRIEVED CONTEXT" in prompt

    def test_24_numerical_prompt_mentions_units_rule(self, assembler):
        """Numerical template must instruct about units."""
        prompt = assembler.assemble(
            query      = "What was net income?",
            chunks     = SAMPLE_CHUNKS,
            query_type = "numerical",
        )
        assert "units" in prompt.lower() or "millions" in prompt.lower()

    def test_25_ratio_prompt_mentions_formula(self, assembler):
        """Ratio template must instruct about showing formula."""
        prompt = assembler.assemble(
            query      = "Calculate gross margin",
            chunks     = SAMPLE_CHUNKS,
            query_type = "ratio",
        )
        assert "formula" in prompt.lower()

    def test_26_forensic_prompt_mentions_risk(self, assembler):
        """Forensic template must mention risk signals."""
        prompt = assembler.assemble(
            query      = "Are there anomalies?",
            chunks     = SAMPLE_CHUNKS,
            query_type = "forensic",
        )
        assert "risk" in prompt.lower() or "anomal" in prompt.lower()

    def test_27_no_rlef_fields_in_prompt(self, assembler):
        """C9: No _rlef_ fields must appear in any prompt."""
        for qt in QUERY_TYPES:
            prompt = assembler.assemble(
                query      = "net income",
                chunks     = SAMPLE_CHUNKS,
                query_type = qt,
            )
            assert "_rlef_" not in prompt, (
                f"C9 VIOLATION: _rlef_ found in {qt} template prompt"
            )


# ── Group 6: BAState integration ─────────────────────────────────────────────

class TestBAStateIntegration:

    def test_28_run_writes_assembled_prompt(self, assembler):
        state = BAState(
            session_id        = "t28",
            query             = "What was net income FY2023?",
            query_type        = "numerical",
            retrieval_stage_2 = SAMPLE_CHUNKS,
            company_name      = "Apple Inc",
            fiscal_year       = "FY2023",
        )
        state = assembler.run(state)
        assert isinstance(state.assembled_prompt, str)
        assert len(state.assembled_prompt) > 50

    def test_29_run_sets_prompt_template_to_context_first(self, assembler):
        """C7: prompt_template field must always be 'context_first'."""
        state = BAState(
            session_id        = "t29",
            query             = "net income",
            query_type        = "text",
            retrieval_stage_2 = SAMPLE_CHUNKS,
        )
        state = assembler.run(state)
        assert state.prompt_template == "context_first"

    def test_30_seed_unchanged_after_run(self, assembler):
        """C5: seed must remain 42."""
        state = BAState(
            session_id        = "t30",
            query             = "net income",
            retrieval_stage_2 = SAMPLE_CHUNKS,
        )
        state = assembler.run(state)
        assert state.seed == 42

    def test_31_empty_query_sets_empty_prompt(self, assembler):
        state = BAState(
            session_id        = "t31",
            query             = "",
            retrieval_stage_2 = SAMPLE_CHUNKS,
        )
        state = assembler.run(state)
        assert state.assembled_prompt == ""

    def test_32_context_before_question_in_state_prompt(self, assembler):
        """C7: assembled_prompt in state must have context before question."""
        state = BAState(
            session_id        = "t32",
            query             = "What was net income?",
            query_type        = "numerical",
            retrieval_stage_2 = SAMPLE_CHUNKS,
            company_name      = "Apple Inc",
            fiscal_year       = "FY2023",
        )
        state        = assembler.run(state)
        prompt       = state.assembled_prompt
        context_pos  = prompt.find(_CONTEXT_MARKER)
        question_pos = prompt.find(_QUESTION_MARKER)
        assert context_pos < question_pos, "C7 VIOLATION in assembled state prompt"

    def test_33_all_query_types_produce_valid_prompt(self, assembler):
        """All 5 query types must produce C7-compliant prompts via run()."""
        for qt in QUERY_TYPES:
            state = BAState(
                session_id        = f"t33_{qt}",
                query             = "What was net income?",
                query_type        = qt,
                retrieval_stage_2 = SAMPLE_CHUNKS,
                company_name      = "Apple Inc",
                fiscal_year       = "FY2023",
            )
            state = assembler.run(state)
            assert len(state.assembled_prompt) > 50
            assert _CONTEXT_MARKER in state.assembled_prompt


# ── Group 7: Convenience wrappers ─────────────────────────────────────────────

class TestConvenienceWrappers:

    def test_34_run_prompt_assembler_returns_state(self):
        state = BAState(
            session_id        = "t34",
            query             = "net income",
            query_type        = "numerical",
            retrieval_stage_2 = SAMPLE_CHUNKS,
        )
        result = run_prompt_assembler(state)
        assert hasattr(result, "assembled_prompt")

    def test_35_assemble_prompt_returns_string(self):
        result = assemble_prompt(
            query      = "What was net income?",
            chunks     = SAMPLE_CHUNKS,
            query_type = "numerical",
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_36_assemble_prompt_c7_compliant(self):
        """Functional interface must also produce C7-compliant prompts."""
        for qt in QUERY_TYPES:
            prompt = assemble_prompt(
                query        = "net income",
                chunks       = SAMPLE_CHUNKS,
                query_type   = qt,
                company_name = "Apple Inc",
                fiscal_year  = "FY2023",
            )
            context_pos  = prompt.find(_CONTEXT_MARKER)
            question_pos = prompt.find(_QUESTION_MARKER)
            assert context_pos < question_pos, (
                f"C7 VIOLATION in assemble_prompt for query_type='{qt}'"
            )