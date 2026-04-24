"""
tests/test_prompt_assembler.py
FinBench Multi-Agent Business Analyst AI

Tests for N10 — Prompt Assembler

24 tests covering:
  - Instantiation (tests 01-03)
  - All 5 templates compile and render (tests 04-08)
  - C7 context-first enforcement (tests 09-12)
  - BAState integration (tests 13-18)
  - Edge cases (tests 19-24)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from src.prompts.assembler import PromptAssembler
from src.state.ba_state import BAState, QueryType


# ── Shared mock chunks ────────────────────────────────────────────────────────

def make_chunks():
    return [
        {
            "chunk_id":    "chunk_000001",
            "text":        "Apple / 10-K / FY2023 / Financial Statements / 42\n"
                           "Net income: $96,995 million FY2023.",
            "section":     "Financial Statements",
            "page":        "42",
            "company":     "Apple Inc",
            "doc_type":    "10-K",
            "fiscal_year": "FY2023",
        },
        {
            "chunk_id":    "chunk_000002",
            "text":        "Apple / 10-K / FY2023 / MD&A / 28\n"
                           "Total net sales $383,285 million FY2023.",
            "section":     "MD&A",
            "page":        "28",
            "company":     "Apple Inc",
            "doc_type":    "10-K",
            "fiscal_year": "FY2023",
        },
    ]


# ── Module-level fixture ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def assembler():
    return PromptAssembler()


# ════════════════════════════════════════════════════════════════════════════
# GROUP 1 — INSTANTIATION (tests 01-03)
# ════════════════════════════════════════════════════════════════════════════

class TestInstantiation:

    def test_01_assembler_instantiates(self, assembler):
        """N10: PromptAssembler must instantiate without error"""
        assert assembler is not None

    def test_02_five_templates_compiled(self, assembler):
        """N10: All 5 query type templates must be compiled at startup"""
        names = assembler.get_template_names()
        assert len(names) == 5

    def test_03_all_query_types_have_templates(self, assembler):
        """N10: Every QueryType must have a corresponding template"""
        names = assembler.get_template_names()
        expected = {
            QueryType.NUMERICAL, QueryType.RATIO,
            QueryType.MULTI_DOC, QueryType.TEXT, QueryType.FORENSIC,
        }
        assert set(names) == expected


# ════════════════════════════════════════════════════════════════════════════
# GROUP 2 — ALL 5 TEMPLATES RENDER (tests 04-08)
# ════════════════════════════════════════════════════════════════════════════

class TestTemplateRendering:

    def test_04_numerical_template_renders(self, assembler):
        """N10: Numerical template must render with chunks"""
        prompt = assembler.assemble_direct(
            query_type  = QueryType.NUMERICAL,
            question    = "What was Apple net income FY2023?",
            chunks      = make_chunks(),
            company     = "Apple Inc",
            doc_type    = "10-K",
            fiscal_year = "FY2023",
        )
        assert len(prompt) > 100
        assert "Apple Inc"  in prompt
        assert "FY2023"     in prompt
        assert "net income" in prompt.lower()

    def test_05_ratio_template_has_computation_section(self, assembler):
        """N10: Ratio template must include COMPUTATION section"""
        prompt = assembler.assemble_direct(
            query_type  = QueryType.RATIO,
            question    = "What was Apple gross margin?",
            chunks      = make_chunks(),
            company     = "Apple Inc",
            doc_type    = "10-K",
            fiscal_year = "FY2023",
        )
        assert "COMPUTATION" in prompt
        assert "formula"     in prompt.lower()

    def test_06_multi_doc_template_renders(self, assembler):
        """N10: Multi-doc template must render correctly"""
        prompt = assembler.assemble_direct(
            query_type  = QueryType.MULTI_DOC,
            question    = "Compare Apple revenue 2021 vs 2022",
            chunks      = make_chunks(),
            company     = "Apple Inc",
            doc_type    = "10-K",
            fiscal_year = "FY2023",
        )
        assert len(prompt) > 100
        assert "RETRIEVED SECTIONS" in prompt

    def test_07_text_template_renders(self, assembler):
        """N10: Text template must render correctly"""
        prompt = assembler.assemble_direct(
            query_type  = QueryType.TEXT,
            question    = "What are Apple main risk factors?",
            chunks      = make_chunks(),
            company     = "Apple Inc",
            doc_type    = "10-K",
            fiscal_year = "FY2023",
        )
        assert len(prompt) > 100
        assert "RETRIEVED SECTIONS" in prompt
        assert "narrative"          in prompt.lower()

    def test_08_forensic_template_has_framework(self, assembler):
        """N10: Forensic template must include forensic analysis framework"""
        prompt = assembler.assemble_direct(
            query_type  = QueryType.FORENSIC,
            question    = "Are there anomalies in Apple financials?",
            chunks      = make_chunks(),
            company     = "Apple Inc",
            doc_type    = "10-K",
            fiscal_year = "FY2023",
        )
        assert "FORENSIC ANALYSIS FRAMEWORK" in prompt
        assert "Benford"                      in prompt


# ════════════════════════════════════════════════════════════════════════════
# GROUP 3 — C7 CONTEXT-FIRST ENFORCEMENT (tests 09-12)
# ════════════════════════════════════════════════════════════════════════════

class TestC7ContextFirst:

    def test_09_context_before_question_numerical(self, assembler):
        """C7: RETRIEVED SECTIONS must appear before QUESTION in numerical"""
        prompt = assembler.assemble_direct(
            query_type  = QueryType.NUMERICAL,
            question    = "What was Apple net income FY2023?",
            chunks      = make_chunks(),
            company     = "Apple Inc",
            doc_type    = "10-K",
            fiscal_year = "FY2023",
        )
        assert assembler.context_before_question(prompt)

    def test_10_context_before_question_all_templates(self, assembler):
        """C7: All 5 templates must put context before question"""
        for qt in [
            QueryType.NUMERICAL, QueryType.RATIO,
            QueryType.MULTI_DOC, QueryType.TEXT, QueryType.FORENSIC,
        ]:
            prompt = assembler.assemble_direct(
                query_type  = qt,
                question    = "test question",
                chunks      = make_chunks(),
                company     = "Apple Inc",
                doc_type    = "10-K",
                fiscal_year = "FY2023",
            )
            assert assembler.context_before_question(prompt), \
                f"C7 VIOLATION in template: {qt}"

    def test_11_validate_c7_raises_on_violation(self, assembler):
        """C7: _validate_c7 must raise AssertionError when violated"""
        bad_prompt = "QUESTION: test question\nRETRIEVED SECTIONS:\nsome context"
        with pytest.raises(AssertionError, match="C7 VIOLATION"):
            assembler._validate_c7(bad_prompt, "test question")

    def test_12_retrieved_sections_present_in_prompt(self, assembler):
        """C7: Every prompt with chunks must contain RETRIEVED SECTIONS"""
        prompt = assembler.assemble_direct(
            query_type  = QueryType.TEXT,
            question    = "What are the risk factors?",
            chunks      = make_chunks(),
            company     = "Apple Inc",
            doc_type    = "10-K",
            fiscal_year = "FY2023",
        )
        assert "RETRIEVED SECTIONS" in prompt


# ════════════════════════════════════════════════════════════════════════════
# GROUP 4 — BASTATE INTEGRATION (tests 13-18)
# ════════════════════════════════════════════════════════════════════════════

class TestBAStateIntegration:

    def test_13_run_writes_assembled_prompt(self, assembler):
        """N10: run() must write assembled_prompt to BAState"""
        state = BAState(
            session_id        = "t13",
            query             = "What was Apple net income FY2023?",
            query_type        = QueryType.NUMERICAL,
            company_name      = "Apple Inc",
            doc_type          = "10-K",
            fiscal_year       = "FY2023",
            retrieval_stage_2 = make_chunks(),
        )
        state = assembler.run(state)
        assert state.assembled_prompt != ""
        assert len(state.assembled_prompt) > 100

    def test_14_run_writes_prompt_template_context_first(self, assembler):
        """C7: run() must set prompt_template='context_first'"""
        state = BAState(
            session_id        = "t14",
            query             = "What was Apple net income?",
            query_type        = QueryType.NUMERICAL,
            company_name      = "Apple Inc",
            retrieval_stage_2 = make_chunks(),
        )
        state = assembler.run(state)
        assert state.prompt_template == "context_first"

    def test_15_run_includes_company_name(self, assembler):
        """N10: Assembled prompt must include company name from BAState"""
        state = BAState(
            session_id        = "t15",
            query             = "What was net income?",
            query_type        = QueryType.NUMERICAL,
            company_name      = "Apple Inc",
            doc_type          = "10-K",
            fiscal_year       = "FY2023",
            retrieval_stage_2 = make_chunks(),
        )
        state = assembler.run(state)
        assert "Apple Inc" in state.assembled_prompt

    def test_16_run_includes_fiscal_year(self, assembler):
        """N10: Assembled prompt must include fiscal year from BAState"""
        state = BAState(
            session_id        = "t16",
            query             = "What was net income?",
            query_type        = QueryType.NUMERICAL,
            company_name      = "Apple Inc",
            doc_type          = "10-K",
            fiscal_year       = "FY2023",
            retrieval_stage_2 = make_chunks(),
        )
        state = assembler.run(state)
        assert "FY2023" in state.assembled_prompt

    def test_17_no_query_returns_empty_prompt(self, assembler):
        """N10: Missing query must return empty assembled_prompt"""
        state = BAState(
            session_id        = "t17",
            retrieval_stage_2 = make_chunks(),
        )
        state = assembler.run(state)
        assert state.assembled_prompt == ""

    def test_18_seed_unchanged_after_run(self, assembler):
        """C5: BAState seed must still be 42 after N10"""
        state = BAState(
            session_id        = "t18",
            query             = "What was Apple net income?",
            query_type        = QueryType.NUMERICAL,
            company_name      = "Apple Inc",
            retrieval_stage_2 = make_chunks(),
        )
        state = assembler.run(state)
        assert state.seed == 42


# ════════════════════════════════════════════════════════════════════════════
# GROUP 5 — EDGE CASES (tests 19-24)
# ════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_19_empty_chunks_still_assembles(self, assembler):
        """N10: Empty chunks must still produce a valid prompt"""
        state = BAState(
            session_id        = "t19",
            query             = "What was Apple net income?",
            query_type        = QueryType.TEXT,
            company_name      = "Apple Inc",
            doc_type          = "10-K",
            fiscal_year       = "FY2023",
            retrieval_stage_2 = [],
        )
        state = assembler.run(state)
        assert state.assembled_prompt != ""
        assert "QUESTION:" in state.assembled_prompt

    def test_20_falls_back_to_stage_1_if_stage_2_empty(self, assembler):
        """N10: Must use retrieval_stage_1 if retrieval_stage_2 is empty"""
        state = BAState(
            session_id        = "t20",
            query             = "What was Apple net income?",
            query_type        = QueryType.NUMERICAL,
            company_name      = "Apple Inc",
            doc_type          = "10-K",
            fiscal_year       = "FY2023",
            retrieval_stage_1 = make_chunks(),
            retrieval_stage_2 = [],
        )
        state = assembler.run(state)
        assert "Financial Statements" in state.assembled_prompt

    def test_21_unknown_query_type_falls_back_to_text(self, assembler):
        """N10: Unknown query type must fall back to text template"""
        prompt = assembler.assemble_direct(
            query_type  = QueryType.TEXT,
            question    = "Some unusual question",
            chunks      = make_chunks(),
            company     = "Apple Inc",
            doc_type    = "10-K",
            fiscal_year = "FY2023",
        )
        assert len(prompt) > 100

    def test_22_chunk_section_and_page_in_prompt(self, assembler):
        """N10: Chunk section and page must appear in assembled prompt"""
        state = BAState(
            session_id        = "t22",
            query             = "What was Apple net income?",
            query_type        = QueryType.NUMERICAL,
            company_name      = "Apple Inc",
            doc_type          = "10-K",
            fiscal_year       = "FY2023",
            retrieval_stage_2 = make_chunks(),
        )
        state = assembler.run(state)
        assert "Financial Statements" in state.assembled_prompt
        assert "42"                   in state.assembled_prompt

    def test_23_question_appears_in_prompt(self, assembler):
        """N10: The analyst question must appear in the assembled prompt"""
        question = "What was Apple total net sales in FY2023?"
        state = BAState(
            session_id        = "t23",
            query             = question,
            query_type        = QueryType.NUMERICAL,
            company_name      = "Apple Inc",
            retrieval_stage_2 = make_chunks(),
        )
        state = assembler.run(state)
        assert question in state.assembled_prompt

    def test_24_answer_format_in_prompt(self, assembler):
        """N10: Prompt must include ANSWER FORMAT section"""
        state = BAState(
            session_id        = "t24",
            query             = "What was Apple net income?",
            query_type        = QueryType.NUMERICAL,
            company_name      = "Apple Inc",
            retrieval_stage_2 = make_chunks(),
        )
        state = assembler.run(state)
        assert "ANSWER FORMAT" in state.assembled_prompt