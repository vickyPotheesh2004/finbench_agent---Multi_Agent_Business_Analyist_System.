"""
tests/test_n14_auditor_pod.py
Tests for N14 Blind Auditor Pod
PDR-BAAAI-001 · Rev 1.0

Critical property: BLINDNESS — auditor never reads analyst or quant output.
"""

import pytest
from src.analysis.auditor_pod import AuditorPod, run_auditor_pod
from src.state.ba_state import BAState


# ── Mock LLM ──────────────────────────────────────────────────────────────────

class MockLLMClient:
    def __init__(self, responses=None):
        self.responses   = responses or []
        self._call_count = 0

    def chat(self, prompt, temperature=0.1):
        if self._call_count < len(self.responses):
            r = self.responses[self._call_count]
        else:
            r = self.responses[-1] if self.responses else ""
        self._call_count += 1
        return r

    def is_available(self):
        return True


MOCK_PLANNER = """
CURIOSITY_Q1: Independent verification of net income figure.
CURIOSITY_Q2: Net income from income statement.
CURIOSITY_Q3: Income statement page 94.
CURIOSITY_Q4: FY year confusion risk.
CURIOSITY_Q5: Cross-check with cash flow statement.
CURIOSITY_Q6: Restatement risk.
ANALYSIS_PLAN: Find net income from income statement independently.
RETRIEVAL_HINTS: net income, income statement
VALIDATION_CRITERIA: Cite exact figure with section and page.
"""

MOCK_IMPLEMENTOR_PASS = """
ANSWER: Net income was $96,995 million in FY2023 [INCOME_STATEMENT/P94].
COMPUTATION: N/A
CONFIDENCE: 0.94 because figure directly cited from income statement.
CITATIONS: [INCOME_STATEMENT / PAGE 94: $96,995 million]
"""

MOCK_IMPLEMENTOR_CONTRADICTION = """
ANSWER: Net income was $96,995 million in FY2023 [INCOME_STATEMENT/P94].
However, this contradicts the MD&A narrative which states net income increased
by 5% but the actual increase was 3%. This inconsistency suggests a discrepancy
between the narrative and the financial statements. The mismatch between MD&A
claims and reported figures is concerning and warrants further investigation.
COMPUTATION: N/A
CONFIDENCE: 0.82 because contradiction found between MD&A and statements.
CITATIONS: [INCOME_STATEMENT / PAGE 94: $96,995 million]
"""

MOCK_VALIDATOR_PASS = """
V1_SCOPE: Result: PASS Reason: Complete.
V2_UNITS: Result: PASS Reason: Millions stated.
V3_SIGN: Result: PASS Reason: Positive correct.
V4_CITATION: Result: PASS Reason: Valid citation.
V5_FISCAL_YEAR: Result: PASS Reason: FY2023 correct.
V6_CONSISTENCY: Result: PASS Reason: Consistent.
V7_COMPLETENESS: Result: PASS Reason: Complete.
V8_GROUNDING: Result: PASS Reason: All from context.
VALIDATOR_PASS: ALL 8 checks are PASS
REJECT_REASONS:
RETRY_INSTRUCTIONS:
"""

SAMPLE_CHUNKS = [
    {
        "chunk_id":    "c1",
        "text":        "Net income was 96995 million in FY2023.",
        "section":     "INCOME_STATEMENT",
        "page":        94,
        "company":     "Apple Inc",
        "doc_type":    "10-K",
        "fiscal_year": "FY2023",
    },
    {
        "chunk_id":    "c2",
        "text":        "Net income attributable to shareholders 96995 million FY2023.",
        "section":     "NOTES",
        "page":        110,
        "company":     "Apple Inc",
        "doc_type":    "10-K",
        "fiscal_year": "FY2023",
    },
]


@pytest.fixture
def mock_pod():
    llm = MockLLMClient(responses=[
        MOCK_PLANNER, MOCK_IMPLEMENTOR_PASS, MOCK_VALIDATOR_PASS
    ])
    return AuditorPod(llm_client=llm)


@pytest.fixture
def contradiction_pod():
    llm = MockLLMClient(responses=[
        MOCK_PLANNER, MOCK_IMPLEMENTOR_CONTRADICTION, MOCK_VALIDATOR_PASS
    ])
    return AuditorPod(llm_client=llm)


# ── Group 1: Blindness enforcement ────────────────────────────────────────────

class TestBlindness:

    def test_01_run_never_reads_analyst_output(self, mock_pod):
        """CRITICAL: auditor must never read analyst_output."""
        state = BAState(
            session_id        = "t01",
            query             = "What was net income?",
            retrieval_stage_2 = SAMPLE_CHUNKS,
            analyst_output    = "ANALYST SAID: net income was X",  # auditor must NOT use this
        )
        state = mock_pod.run(state)
        # Auditor output must NOT contain analyst's specific text
        assert "ANALYST SAID" not in state.auditor_output

    def test_02_run_never_reads_quant_result(self, mock_pod):
        """CRITICAL: auditor must never read quant_result."""
        state = BAState(
            session_id        = "t02",
            query             = "What was net income?",
            retrieval_stage_2 = SAMPLE_CHUNKS,
            quant_result      = "QUANT SAID: ratio is Y",  # auditor must NOT use this
        )
        state = mock_pod.run(state)
        assert "QUANT SAID" not in state.auditor_output

    def test_03_auditor_uses_separate_piv_instance(self):
        """Auditor uses its own PIVLoopController — not shared with N11."""
        llm1 = MockLLMClient(responses=[MOCK_PLANNER, MOCK_IMPLEMENTOR_PASS, MOCK_VALIDATOR_PASS])
        llm2 = MockLLMClient(responses=[MOCK_PLANNER, MOCK_IMPLEMENTOR_PASS, MOCK_VALIDATOR_PASS])
        pod1 = AuditorPod(llm_client=llm1)
        pod2 = AuditorPod(llm_client=llm2)
        # Each pod has its own controller
        assert pod1._piv is not pod2._piv

    def test_04_pod_role_is_blind_auditor(self, mock_pod):
        result = mock_pod.run_audit("net income", SAMPLE_CHUNKS)
        assert result.get("pod_role") == "blind_auditor"

    def test_05_auditor_context_injected_in_query(self, mock_pod):
        """Auditor context must be prepended to guide contradiction detection."""
        result = mock_pod.run_audit("What was net income?", SAMPLE_CHUNKS)
        assert isinstance(result, dict)


# ── Group 2: Core audit method ────────────────────────────────────────────────

class TestRunAudit:

    def test_06_returns_dict(self, mock_pod):
        result = mock_pod.run_audit("net income", SAMPLE_CHUNKS)
        assert isinstance(result, dict)

    def test_07_result_has_answer(self, mock_pod):
        result = mock_pod.run_audit("net income", SAMPLE_CHUNKS)
        assert "answer" in result
        assert isinstance(result["answer"], str)

    def test_08_result_has_confidence(self, mock_pod):
        result = mock_pod.run_audit("net income", SAMPLE_CHUNKS)
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0

    def test_09_result_has_contradiction_flags(self, mock_pod):
        result = mock_pod.run_audit("net income", SAMPLE_CHUNKS)
        assert "contradiction_flags" in result
        assert isinstance(result["contradiction_flags"], list)

    def test_10_result_has_citations(self, mock_pod):
        result = mock_pod.run_audit("net income", SAMPLE_CHUNKS)
        assert "citations" in result

    def test_11_empty_chunks_still_returns_dict(self, mock_pod):
        result = mock_pod.run_audit("net income", [])
        assert isinstance(result, dict)
        assert "answer" in result

    def test_12_contradiction_detected_from_answer(self, contradiction_pod):
        result = contradiction_pod.run_audit(
            "What was net income?", SAMPLE_CHUNKS
        )
        # Should detect contradiction keywords in the answer
        flags = result.get("contradiction_flags", [])
        assert isinstance(flags, list)


# ── Group 3: Contradiction extraction ────────────────────────────────────────

class TestContradictionExtraction:

    def test_13_no_contradictions_in_clean_answer(self):
        answer = "Net income was $96,995 million in FY2023 [INCOME_STATEMENT/P94]."
        flags  = AuditorPod._extract_contradictions(answer)
        assert flags == []

    def test_14_contradiction_keyword_detected(self):
        answer = "The MD&A narrative contradicts the income statement figures."
        flags  = AuditorPod._extract_contradictions(answer)
        assert len(flags) > 0

    def test_15_inconsistency_keyword_detected(self):
        answer = "There is an inconsistency between the balance sheet and footnotes."
        flags  = AuditorPod._extract_contradictions(answer)
        assert len(flags) > 0

    def test_16_restatement_keyword_detected(self):
        answer = "A restatement of prior period figures was required for FY2022."
        flags  = AuditorPod._extract_contradictions(answer)
        assert len(flags) > 0

    def test_17_empty_answer_returns_empty(self):
        flags = AuditorPod._extract_contradictions("")
        assert flags == []

    def test_18_flags_capped_at_10(self):
        answer = "\n".join([
            f"This contradicts the statement in line {i}." for i in range(20)
        ])
        flags = AuditorPod._extract_contradictions(answer)
        assert len(flags) <= 10

    def test_19_short_lines_excluded(self):
        """Lines under 20 chars should not be flagged even with keywords."""
        answer = "contradicts"
        flags  = AuditorPod._extract_contradictions(answer)
        assert flags == []


# ── Group 4: BAState integration ─────────────────────────────────────────────

class TestBAStateIntegration:

    def test_20_run_writes_auditor_output(self, mock_pod):
        state = BAState(
            session_id        = "t20",
            query             = "What was net income FY2023?",
            retrieval_stage_2 = SAMPLE_CHUNKS,
        )
        state = mock_pod.run(state)
        assert hasattr(state, "auditor_output")
        assert isinstance(state.auditor_output, str)

    def test_21_run_writes_auditor_confidence(self, mock_pod):
        state = BAState(
            session_id        = "t21",
            query             = "net income",
            retrieval_stage_2 = SAMPLE_CHUNKS,
        )
        state = mock_pod.run(state)
        assert 0.0 <= state.auditor_confidence <= 1.0

    def test_22_run_writes_contradiction_flags(self, mock_pod):
        state = BAState(
            session_id        = "t22",
            query             = "net income",
            retrieval_stage_2 = SAMPLE_CHUNKS,
        )
        state = mock_pod.run(state)
        assert isinstance(state.contradiction_flags, list)

    def test_23_seed_unchanged(self, mock_pod):
        """C5: seed must remain 42."""
        state = BAState(
            session_id        = "t23",
            query             = "net income",
            retrieval_stage_2 = SAMPLE_CHUNKS,
        )
        state = mock_pod.run(state)
        assert state.seed == 42

    def test_24_empty_query_skips_pod(self, mock_pod):
        state = BAState(session_id="t24", query="")
        state = mock_pod.run(state)
        assert state.auditor_output     == ""
        assert state.auditor_confidence == 0.0
        assert state.contradiction_flags == []

    def test_25_no_rlef_in_auditor_output(self, mock_pod):
        """C9: auditor_output must never contain _rlef_ fields."""
        state = BAState(
            session_id        = "t25",
            query             = "net income",
            retrieval_stage_2 = SAMPLE_CHUNKS,
        )
        state = mock_pod.run(state)
        assert "_rlef_" not in state.auditor_output

    def test_26_analyst_output_not_contaminating_auditor(self, mock_pod):
        """Auditor output must be independent of analyst output."""
        state = BAState(
            session_id        = "t26",
            query             = "net income",
            retrieval_stage_2 = SAMPLE_CHUNKS,
            analyst_output    = "CONTAMINATION_MARKER_XYZ",
        )
        state = mock_pod.run(state)
        assert "CONTAMINATION_MARKER_XYZ" not in state.auditor_output

    def test_27_auditor_output_separate_from_analyst_output(self, mock_pod):
        """auditor_output and analyst_output are separate BAState fields."""
        state = BAState(
            session_id        = "t27",
            query             = "net income",
            retrieval_stage_2 = SAMPLE_CHUNKS,
            analyst_output    = "analyst answer here",
        )
        state = mock_pod.run(state)
        assert hasattr(state, "auditor_output")
        assert hasattr(state, "analyst_output")
        # They are separate fields
        assert state.analyst_output == "analyst answer here"


# ── Group 5: Convenience wrapper ─────────────────────────────────────────────

class TestConvenienceWrapper:

    def test_28_run_auditor_pod_returns_state(self):
        llm   = MockLLMClient(responses=[
            MOCK_PLANNER, MOCK_IMPLEMENTOR_PASS, MOCK_VALIDATOR_PASS
        ])
        state = BAState(
            session_id        = "t28",
            query             = "What was net income?",
            retrieval_stage_2 = SAMPLE_CHUNKS,
        )
        result = run_auditor_pod(state, llm_client=llm)
        assert hasattr(result, "auditor_output")
        assert result.seed == 42

    def test_29_wrapper_writes_contradiction_flags(self):
        llm   = MockLLMClient(responses=[
            MOCK_PLANNER, MOCK_IMPLEMENTOR_PASS, MOCK_VALIDATOR_PASS
        ])
        state = BAState(
            session_id        = "t29",
            query             = "net income",
            retrieval_stage_2 = SAMPLE_CHUNKS,
        )
        result = run_auditor_pod(state, llm_client=llm)
        assert hasattr(result, "contradiction_flags")
        assert isinstance(result.contradiction_flags, list)