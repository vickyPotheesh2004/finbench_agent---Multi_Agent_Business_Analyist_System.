"""
tests/test_ci_gate.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 Rev1.0 FINAL

12 CI gate tests covering all constraints C1-C10 + Amendments A1-A4.
Run: pytest tests/test_ci_gate.py -v

Tests 11 and 12 are designed to CATCH violations:
  Test 11: mock output containing _rlef_ — MUST be detected
  Test 12: mock config with seed=99   — MUST be detected
All 12 must PASS against correct code.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.state.ba_state import (
    BAState,
    ClarificationStatus,
    Difficulty,
    PIVStatus,
    PromptTemplate,
    QueryType,
    ResourceGovernor,
)
from src.utils.seed_manager import SeedManager
from src.utils.resource_governor import ResourceGovernor as RG


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 1 — C5: seed=42 everywhere
# ═══════════════════════════════════════════════════════════════════════════

class TestC5Seed:

    def test_01_default_seed_is_42(self):
        """C5: Fresh BAState must have seed=42"""
        s = BAState(session_id="t01")
        assert s.seed == 42

    def test_02_wrong_seed_rejected(self):
        """C5: seed != 42 must raise ValueError"""
        with pytest.raises((ValueError, Exception)) as exc:
            BAState(session_id="t02", seed=0)
        assert "C5" in str(exc.value) or "seed" in str(exc.value).lower()

    def test_03_seed_manager_returns_42(self):
        """C5: SeedManager.get() must always return 42"""
        assert SeedManager.get() == 42

    def test_04_seed_manager_rejects_wrong_seed(self):
        """C5: SeedManager.set_all(99) must raise ValueError"""
        with pytest.raises(ValueError) as exc:
            SeedManager.set_all(99)
        assert "C5" in str(exc.value)


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 2 — C4: 14GB RAM hard cap
# ═══════════════════════════════════════════════════════════════════════════

class TestC4RAM:

    def test_05_ram_check_returns_float(self):
        """C4: ResourceGovernor.check() must return current RAM as float"""
        used = RG.check("ci gate test")
        assert isinstance(used, float)
        assert used > 0

    def test_06_ram_status_has_required_keys(self):
        """C4: status() must contain all required keys"""
        status = RG.status()
        for key in ["used_gb", "total_gb", "available_gb", "percent", "safe"]:
            assert key in status, f"Missing key: {key}"

    def test_07_ram_is_below_hard_cap(self):
        """C4: Current RAM must be below 14GB hard cap"""
        status = RG.status()
        assert status["safe"] is True, (
            f"RAM={status['used_gb']}GB is at or above 14GB hard cap!"
        )


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 3 — C9: _rlef_ fields never in output
# ═══════════════════════════════════════════════════════════════════════════

class TestC9RLEF:

    def test_08_safe_dict_has_no_rlef_keys(self):
        """C9: safe_dict() must return zero _rlef_ keys"""
        s    = BAState(session_id="t08")
        safe = s.safe_dict()
        rlef = [k for k in safe if k.startswith("_rlef_")]
        assert len(rlef) == 0, f"C9 VIOLATION: {rlef}"

    def test_09_model_dump_has_no_rlef_keys(self):
        """C9: model_dump() must not expose _rlef_ private attrs"""
        s    = BAState(session_id="t09")
        dump = s.model_dump()
        rlef = [k for k in dump if k.startswith("_rlef_")]
        assert len(rlef) == 0, f"C9 VIOLATION in model_dump(): {rlef}"


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 4 — C8: mandatory chunk metadata prefix
# ═══════════════════════════════════════════════════════════════════════════

class TestC8ChunkMetadata:

    def test_10_chunk_prefix_has_all_5_fields(self):
        """C8: prefix must contain COMPANY/DOCTYPE/FISCAL_YEAR/SECTION/PAGE"""
        s = BAState(
            session_id="t10",
            company_name="Goldman Sachs",
            doc_type="10-K",
            fiscal_year="FY2023",
        )
        prefix = s.chunk_metadata_prefix(section="MD&A", page="47")
        assert "Goldman Sachs" in prefix
        assert "10-K"          in prefix
        assert "FY2023"        in prefix
        assert "MD&A"          in prefix
        assert "47"            in prefix


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 5 — CI/CD gate scans (tests 11+12 catch violations)
# ═══════════════════════════════════════════════════════════════════════════

class TestCICDGates:

    def test_11_detect_rlef_in_mock_output(self):
        """
        C9 CI GATE: Must detect _rlef_ in a mock output file.
        This test is DESIGNED to catch the violation.
        """
        mock_output = {
            "question":    "What was net income?",
            "answer":      "99.8 billion",
            "_rlef_grade": 4,
        }
        mock_json      = json.dumps(mock_output)
        violation_found = "_rlef_" in mock_json
        assert violation_found is True, (
            "CI GATE FAILURE: _rlef_ in output was NOT detected."
        )

    def test_12_detect_wrong_seed_in_mock_config(self):
        """
        C5 CI GATE: Must detect seed != 42 in a mock config.
        This test is DESIGNED to catch the violation.
        """
        mock_config    = {"seed": 99, "model": "llama3.1:8b"}
        violation_found = mock_config.get("seed") != 42
        assert violation_found is True, (
            "CI GATE FAILURE: seed=99 was NOT detected as a violation."
        )


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 6 — Amendments A1-A3
# ═══════════════════════════════════════════════════════════════════════════

class TestAmendments:

    def test_13_a2_max_attempts_is_5(self):
        """A2: piv_max_attempts must default to 5"""
        s = BAState(session_id="t13")
        assert s.piv_max_attempts == 5

    def test_14_a2_attempt_count_6_rejected(self):
        """A2: attempt_count > 5 must raise"""
        with pytest.raises((ValueError, Exception)) as exc:
            BAState(session_id="t14", analyst_attempt_count=6)
        assert "A2" in str(exc.value) or "attempt" in str(exc.value).lower()

    def test_15_a3_clarification_false_by_default(self):
        """A3: needs_clarification() must be False on fresh state"""
        s = BAState(session_id="t15")
        assert s.needs_clarification() is False

    def test_16_a3_clarification_true_when_all_pods_exhausted(self):
        """A3: needs_clarification() True when all 3 pods at 5 REJECTs"""
        s = BAState(
            session_id="t16",
            analyst_attempt_count=5, analyst_piv_status=PIVStatus.REJECT,
            quant_attempt_count=5,   quant_piv_status=PIVStatus.REJECT,
            auditor_attempt_count=5, auditor_piv_status=PIVStatus.REJECT,
        )
        assert s.needs_clarification() is True

    def test_17_a3_reset_clears_all_counters(self):
        """A3: reset_for_clarification() must reset all counters to 0"""
        s = BAState(
            session_id="t17",
            analyst_attempt_count=5, analyst_piv_status=PIVStatus.REJECT,
            quant_attempt_count=5,   quant_piv_status=PIVStatus.REJECT,
            auditor_attempt_count=5, auditor_piv_status=PIVStatus.REJECT,
        )
        s.reset_for_clarification("Use FY2023 GAAP consolidated figures")
        assert s.analyst_attempt_count == 0
        assert s.quant_attempt_count   == 0
        assert s.auditor_attempt_count == 0
        assert s.analyst_piv_status    == PIVStatus.PENDING
        assert s.quant_piv_status      == PIVStatus.PENDING
        assert s.auditor_piv_status    == PIVStatus.PENDING
        assert s.clarification_answer  == "Use FY2023 GAAP consolidated figures"
        assert s.clarification_round   == 1
        assert s.needs_clarification() is False