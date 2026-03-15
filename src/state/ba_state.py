"""
src/state/ba_state.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 Rev1.0 FINAL + Amendments A1-A4

Constraints enforced:
  C4 — 14GB RAM hard cap
  C5 — seed=42 always
  C8 — chunk metadata prefix format
  C9 — _rlef_ fields private, never in outputs
  A1 — PIV REJECT goes back to Planner
  A2 — max_retries = 5 per pod
  A3 — Clarification Engine after 5 exhausted attempts
"""

from __future__ import annotations

import random
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import psutil
from pydantic import BaseModel, Field, field_validator, model_validator

# ── C5: Seed at import time ─────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)


# ═══════════════════════════════════════════════════════════════════════════
# ENUMERATIONS
# ═══════════════════════════════════════════════════════════════════════════

class QueryType(str, Enum):
    NUMERICAL = "numerical"
    RATIO     = "ratio"
    MULTI_DOC = "multi_doc"
    TEXT      = "text"
    FORENSIC  = "forensic"


class Difficulty(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


class PIVStatus(str, Enum):
    PENDING = "PENDING"
    PASS    = "PASS"
    REJECT  = "REJECT"


class ClarificationStatus(str, Enum):
    NOT_NEEDED = "not_needed"
    PENDING    = "pending"
    ANSWERED   = "answered"


class PromptTemplate(str, Enum):
    NUMERICAL = "numerical"
    RATIO     = "ratio"
    MULTI_DOC = "multi_doc"
    TEXT      = "text"
    FORENSIC  = "forensic"


# ═══════════════════════════════════════════════════════════════════════════
# C4 — RESOURCE GOVERNOR
# ═══════════════════════════════════════════════════════════════════════════

class ResourceGovernor:
    """
    C4: RAM monitoring.
    warn  @ 12GB
    alert @ 13GB
    halt  @ 14GB — raises MemoryError
    """
    WARN_GB  = 12.0
    ALERT_GB = 13.0
    HALT_GB  = 14.0

    @staticmethod
    def check() -> float:
        used_gb = psutil.virtual_memory().used / (1024 ** 3)
        if used_gb >= ResourceGovernor.HALT_GB:
            raise MemoryError(
                f"[C4 HALT] RAM={used_gb:.2f}GB >= 14GB hard cap. "
                "Stopping to protect system."
            )
        if used_gb >= ResourceGovernor.ALERT_GB:
            print(f"[C4 ALERT] RAM={used_gb:.2f}GB — approaching 14GB")
        elif used_gb >= ResourceGovernor.WARN_GB:
            print(f"[C4 WARN]  RAM={used_gb:.2f}GB — monitor usage")
        return used_gb

    @staticmethod
    def used_gb() -> float:
        return psutil.virtual_memory().used / (1024 ** 3)

    @staticmethod
    def total_gb() -> float:
        return psutil.virtual_memory().total / (1024 ** 3)


# ═══════════════════════════════════════════════════════════════════════════
# BASTATE
# ═══════════════════════════════════════════════════════════════════════════

class BAState(BaseModel):
    """
    Shared state object flowing through all 19 nodes.
    PDR-BAAAI-001 Rev1.0 FINAL + Amendments A1-A4.
    """

    model_config = {
        "validate_assignment": True,
        "arbitrary_types_allowed": True,
        "protected_namespaces": (),
    }

    # ── Session ──────────────────────────────────────────────────────────
    session_id:    str = Field(default="")
    model_version: str = Field(default="financebench-expert-v1")
    seed:          int = Field(default=42)

    # ── Document — N01 ───────────────────────────────────────────────────
    document_path:     str  = Field(default="")
    company_name:      str  = Field(default="")
    doc_type:          str  = Field(default="")
    fiscal_year:       str  = Field(default="")
    raw_text:          str  = Field(default="")
    table_cells:       List[Dict[str, Any]] = Field(default_factory=list)
    heading_positions: List[Dict[str, Any]] = Field(default_factory=list)

    # ── Section tree — N02 ───────────────────────────────────────────────
    section_tree: Dict[str, Any] = Field(default_factory=dict)

    # ── Index — N03 ──────────────────────────────────────────────────────
    chunk_count:         int = Field(default=0)
    bm25_index_path:     str = Field(default="")
    chromadb_collection: str = Field(default="")

    # ── Query ────────────────────────────────────────────────────────────
    query:               str            = Field(default="")
    query_type:          QueryType      = Field(default=QueryType.TEXT)
    query_difficulty:    Difficulty     = Field(default=Difficulty.MEDIUM)
    routing_path:        str            = Field(default="")
    context_window_size: int            = Field(default=3, ge=1, le=10)
    prompt_template:     PromptTemplate = Field(default=PromptTemplate.TEXT)

    # ── Retrieval ────────────────────────────────────────────────────────
    sniper_hit:        bool  = Field(default=False)
    sniper_result:     str   = Field(default="")
    sniper_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    bm25_results:      List[Dict[str, Any]] = Field(default_factory=list)
    retrieval_stage_1: List[Dict[str, Any]] = Field(default_factory=list)
    retrieval_stage_2: List[Dict[str, Any]] = Field(default_factory=list)

    # ── Prompt — N10 ─────────────────────────────────────────────────────
    assembled_prompt: str = Field(default="")

    # ── PIV Loop — A1 + A2 ───────────────────────────────────────────────
    piv_max_attempts: int = Field(default=5)
    piv_round:        int = Field(default=0)
    iteration_count:  int = Field(default=0)

    # N11 Analyst Pod
    analyst_attempt_count:    int       = Field(default=0)
    analyst_piv_status:       PIVStatus = Field(default=PIVStatus.PENDING)
    analyst_rejection_reason: str       = Field(default="")
    analyst_output:           str       = Field(default="")
    analyst_confidence:       float     = Field(default=0.0, ge=0.0, le=1.0)
    analyst_citations:        List[str] = Field(default_factory=list)

    # N12 CFO/Quant Pod
    quant_attempt_count:    int       = Field(default=0)
    quant_piv_status:       PIVStatus = Field(default=PIVStatus.PENDING)
    quant_rejection_reason: str       = Field(default="")
    quant_result:           str       = Field(default="")
    quant_confidence:       float     = Field(default=0.0, ge=0.0, le=1.0)
    quant_citations:        List[str] = Field(default_factory=list)

    # N14 Auditor Pod
    auditor_attempt_count:    int       = Field(default=0)
    auditor_piv_status:       PIVStatus = Field(default=PIVStatus.PENDING)
    auditor_rejection_reason: str       = Field(default="")
    auditor_output:           str       = Field(default="")
    auditor_confidence:       float     = Field(default=0.0, ge=0.0, le=1.0)
    auditor_citations:        List[str] = Field(default_factory=list)

    # ── Clarification Engine — A3 ─────────────────────────────────────────
    clarification_status:    ClarificationStatus = Field(default=ClarificationStatus.NOT_NEEDED)
    clarification_questions: List[str]           = Field(default_factory=list)
    clarification_answer:    str                 = Field(default="")
    clarification_round:     int                 = Field(default=0)

    # ── Forensics — N13 ──────────────────────────────────────────────────
    forensic_flags:   List[str] = Field(default_factory=list)
    risk_score:       float     = Field(default=0.0, ge=0.0, le=100.0)
    anomaly_detected: bool      = Field(default=False)
    benford_chi2:     float     = Field(default=0.0)
    benford_p_value:  float     = Field(default=1.0, ge=0.0, le=1.0)

    # ── Output ───────────────────────────────────────────────────────────
    final_answer_pre_xgb:    str   = Field(default="")
    agreement_status:        str   = Field(default="")
    confidence_score:        float = Field(default=0.0, ge=0.0, le=1.0)
    low_confidence:          bool  = Field(default=False)
    shap_values:             List[Dict[str, Any]] = Field(default_factory=list)
    causal_dag_path:         str   = Field(default="")
    xgb_ranked_answer:       str   = Field(default="")
    final_answer:            str   = Field(default="")
    analyst_citations_final: List[str] = Field(default_factory=list)
    output_docx_path:        str   = Field(default="")

    # ── C9: RLEF — PRIVATE FOREVER ───────────────────────────────────────
    _rlef_grade:        Optional[int] = None
    _rlef_chosen:       Optional[str] = None
    _rlef_rejected:     Optional[str] = None
    _rlef_session_hash: Optional[str] = None

    # ═══════════════════════════════════════════════════════════════════════
    # VALIDATORS
    # ═══════════════════════════════════════════════════════════════════════

    @field_validator("seed")
    @classmethod
    def enforce_seed_42(cls, v: int) -> int:
        if v != 42:
            raise ValueError(f"[C5 VIOLATION] seed must be 42, got {v}")
        return v

    @field_validator("analyst_attempt_count", "quant_attempt_count", "auditor_attempt_count")
    @classmethod
    def enforce_max_attempts(cls, v: int) -> int:
        if v > 5:
            raise ValueError(
                f"[A2 VIOLATION] attempt_count={v} exceeds max_retries=5"
            )
        return v

    @field_validator("iteration_count")
    @classmethod
    def enforce_iteration_cap(cls, v: int) -> int:
        if v > 5:
            raise ValueError(
                f"[PDR VIOLATION] iteration_count={v} exceeds cap of 5"
            )
        return v

    # ═══════════════════════════════════════════════════════════════════════
    # HELPER METHODS
    # ═══════════════════════════════════════════════════════════════════════

    def check_ram(self) -> float:
        """C4: Check RAM usage."""
        return ResourceGovernor.check()

    def needs_clarification(self) -> bool:
        """A3: True when all 3 pods exhausted 5 attempts."""
        return (
            self.analyst_attempt_count  >= self.piv_max_attempts
            and self.analyst_piv_status == PIVStatus.REJECT
            and self.quant_attempt_count   >= self.piv_max_attempts
            and self.quant_piv_status   == PIVStatus.REJECT
            and self.auditor_attempt_count >= self.piv_max_attempts
            and self.auditor_piv_status == PIVStatus.REJECT
        )

    def reset_for_clarification(self, user_answer: str) -> "BAState":
        """A3: Reset all pods after user provides clarification."""
        self.clarification_answer     = user_answer
        self.clarification_status     = ClarificationStatus.ANSWERED
        self.clarification_round     += 1
        self.analyst_attempt_count    = 0
        self.quant_attempt_count      = 0
        self.auditor_attempt_count    = 0
        self.analyst_piv_status       = PIVStatus.PENDING
        self.quant_piv_status         = PIVStatus.PENDING
        self.auditor_piv_status       = PIVStatus.PENDING
        self.analyst_rejection_reason = ""
        self.quant_rejection_reason   = ""
        self.auditor_rejection_reason = ""
        return self

    def chunk_metadata_prefix(self, section: str = "{section}", page: str = "{page}") -> str:
        """C8: COMPANY / DOCTYPE / FISCAL_YEAR / SECTION / PAGE"""
        return (
            f"{self.company_name} / {self.doc_type} / "
            f"{self.fiscal_year} / {section} / {page}"
        )

    def safe_dict(self) -> Dict[str, Any]:
        """C9: Remove all _rlef_ fields before any output."""
        return {
            k: v
            for k, v in self.model_dump().items()
            if not k.startswith("_rlef_")
        }

    def pod_summary(self) -> str:
        """Human readable pod status summary."""
        return (
            f"Analyst: {self.analyst_piv_status.value} "
            f"({self.analyst_attempt_count}/{self.piv_max_attempts}) | "
            f"Quant: {self.quant_piv_status.value} "
            f"({self.quant_attempt_count}/{self.piv_max_attempts}) | "
            f"Auditor: {self.auditor_piv_status.value} "
            f"({self.auditor_attempt_count}/{self.piv_max_attempts})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK
# run: python src/state/ba_state.py
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        from rich import print as rprint
    except ImportError:
        rprint = print

    rprint("\n[bold cyan]── BAState sanity check ──[/bold cyan]")

    s = BAState(session_id="sanity-001")
    rprint(f"[green]✓[/green] Created | seed={s.seed} | max_attempts={s.piv_max_attempts}")

    ram = s.check_ram()
    rprint(f"[green]✓[/green] C4 RAM={ram:.2f}GB")

    try:
        BAState(session_id="bad", seed=99)
    except Exception as e:
        rprint(f"[green]✓[/green] C5 enforced: {e}")

    try:
        BAState(session_id="bad2", analyst_attempt_count=6)
    except Exception as e:
        rprint(f"[green]✓[/green] A2 enforced: {e}")

    safe = s.safe_dict()
    assert not any(k.startswith("_rlef_") for k in safe)
    rprint(f"[green]✓[/green] C9 enforced: no _rlef_ in safe_dict()")

    s.company_name = "Apple Inc"
    s.doc_type     = "10-K"
    s.fiscal_year  = "FY2023"
    prefix = s.chunk_metadata_prefix("MD&A", "42")
    assert "Apple Inc" in prefix and "10-K" in prefix and "FY2023" in prefix
    rprint(f"[green]✓[/green] C8 prefix: {prefix}")

    s2 = BAState(
        session_id="clarify-test",
        analyst_attempt_count=5,  analyst_piv_status=PIVStatus.REJECT,
        quant_attempt_count=5,    quant_piv_status=PIVStatus.REJECT,
        auditor_attempt_count=5,  auditor_piv_status=PIVStatus.REJECT,
    )
    assert s2.needs_clarification() is True
    s2.reset_for_clarification("Use FY2023 GAAP figures")
    assert s2.analyst_attempt_count == 0
    assert s2.clarification_round   == 1
    rprint(f"[green]✓[/green] A3 clarification engine works")

    rprint("\n[bold green]All checks passed. BAState ready.[/bold green]\n")