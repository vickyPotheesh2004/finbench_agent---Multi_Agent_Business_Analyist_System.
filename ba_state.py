"""
ba_state.py — BAState Pydantic v2 Schema
PDR-BAAAI-001 Rev1.0 FINAL + Amendments A1-A4
Constraint C5: seed=42 | C9: _rlef_ fields private
"""

import random
import numpy as np
from typing import Optional, Any
from pydantic import BaseModel, Field, model_validator
from enum import Enum
import psutil
import os

# ── C5: Deterministic seed everywhere ──────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── Enumerations ────────────────────────────────────────────────────────────

class QueryType(str, Enum):
    NUMERICAL   = "numerical"
    RATIO       = "ratio"
    MULTI_DOC   = "multi_doc"
    TEXT        = "text"
    FORENSIC    = "forensic"

class Difficulty(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"

class PIVStatus(str, Enum):
    PASS   = "PASS"
    REJECT = "REJECT"
    PENDING = "PENDING"

class ClarificationStatus(str, Enum):
    NOT_NEEDED = "not_needed"
    PENDING    = "pending"
    ANSWERED   = "answered"

# ── Resource Governor (C4: 14GB RAM hard cap) ───────────────────────────────

class ResourceGovernor:
    WARN_GB  = 12
    ALERT_GB = 13
    HALT_GB  = 14

    @staticmethod
    def check():
        ram_gb = psutil.virtual_memory().used / (1024 ** 3)
        if ram_gb >= ResourceGovernor.HALT_GB:
            raise MemoryError(
                f"[C4 VIOLATION] RAM usage {ram_gb:.1f}GB >= 14GB hard cap. "
                f"Halting to protect system."
            )
        if ram_gb >= ResourceGovernor.ALERT_GB:
            print(f"[ALERT] RAM at {ram_gb:.1f}GB — approaching 14GB limit")
        elif ram_gb >= ResourceGovernor.WARN_GB:
            print(f"[WARN] RAM at {ram_gb:.1f}GB — monitor usage")
        return ram_gb

# ── BAState — shared object flowing through all 19 nodes ────────────────────

class BAState(BaseModel):
    """
    Shared state object for the FinBench Multi-Agent Business Analyst AI.
    Flows through all 19 nodes. Every node reads and writes to this object.
    Amendments A1-A4 fields included.
    """

    model_config = {"arbitrary_types_allowed": True}

    # ── Session identity ──────────────────────────────────────────────────
    session_id:         str  = Field(default="", description="Unique session UUID")
    seed:               int  = Field(default=42,  description="C5: always 42")

    # ── Document fields (written by N01) ─────────────────────────────────
    document_path:      str  = Field(default="", description="Path to input PDF/DOCX/XLSX")
    company_name:       str  = Field(default="", description="e.g. Apple Inc.")
    doc_type:           str  = Field(default="", description="10-K / 10-Q / 8-K / Earnings")
    fiscal_year:        str  = Field(default="", description="e.g. FY2023")
    raw_text:           str  = Field(default="", description="Full extracted text from document")
    table_cells:        list = Field(default_factory=list, description="List of table cell dicts")
    heading_positions:  list = Field(default_factory=list, description="Font-detected headings")

    # ── Section tree (written by N02) ────────────────────────────────────
    section_tree:       dict = Field(default_factory=dict, description="Hierarchical section JSON")

    # ── Index paths (written by N03) ─────────────────────────────────────
    chunk_count:            int = Field(default=0,  description="Total chunks indexed")
    bm25_index_path:        str = Field(default="", description="Path to bm25s index file")
    chromadb_collection:    str = Field(default="", description="ChromaDB collection name")

    # ── Query fields (written at query time) ─────────────────────────────
    query:              str         = Field(default="", description="Analyst's question")
    query_type:         QueryType   = Field(default=QueryType.TEXT)
    query_difficulty:   Difficulty  = Field(default=Difficulty.MEDIUM)
    routing_path:       str         = Field(default="", description="Which retrieval tiers fired")
    context_window_size: int        = Field(default=3,  description="Top-k chunks to retrieve")

    # ── Retrieval results (written by N06-N09) ───────────────────────────
    sniper_hit:         bool  = Field(default=False, description="N06: regex match found")
    sniper_result:      str   = Field(default="",    description="N06: extracted value")
    sniper_confidence:  float = Field(default=0.0,   description="N06: confidence 0-1")
    bm25_results:       list  = Field(default_factory=list, description="N07: top-10 chunks")
    semantic_results:   list  = Field(default_factory=list, description="N08: top-10 chunks")
    retrieval_stage_2:  list  = Field(default_factory=list, description="N09: final top-3 chunks")

    # ── Prompt (written by N10) ───────────────────────────────────────────
    assembled_prompt:   str = Field(default="", description="C7: context always before question")

    # ── PIV Loop fields — Amendment A1 (back to Planner) + A2 (max 5) ────
    piv_max_attempts:       int        = Field(default=5, description="A2: max 5 attempts per pod")
    analyst_attempt_count:  int        = Field(default=0, description="N11 attempt counter")
    analyst_piv_status:     PIVStatus  = Field(default=PIVStatus.PENDING)
    analyst_rejection_reason: str      = Field(default="", description="Why Validator rejected")
    analyst_output:         str        = Field(default="", description="N11 candidate answer")
    analyst_confidence:     float      = Field(default=0.0)
    analyst_citations:      list       = Field(default_factory=list)

    quant_attempt_count:    int        = Field(default=0, description="N12 attempt counter")
    quant_piv_status:       PIVStatus  = Field(default=PIVStatus.PENDING)
    quant_rejection_reason: str        = Field(default="")
    quant_result:           str        = Field(default="", description="N12 candidate answer")
    quant_confidence:       float      = Field(default=0.0)
    quant_citations:        list       = Field(default_factory=list)

    auditor_attempt_count:  int        = Field(default=0, description="N14 attempt counter")
    auditor_piv_status:     PIVStatus  = Field(default=PIVStatus.PENDING)
    auditor_rejection_reason: str      = Field(default="")
    auditor_output:         str        = Field(default="", description="N14 candidate answer")
    auditor_confidence:     float      = Field(default=0.0)
    auditor_citations:      list       = Field(default_factory=list)

    # ── Clarification Engine — Amendment A3 ──────────────────────────────
    clarification_status:   ClarificationStatus = Field(default=ClarificationStatus.NOT_NEEDED)
    clarification_questions: list = Field(
        default_factory=list,
        description="5 targeted questions generated after 5 failed attempts"
    )
    clarification_answer:   str  = Field(default="", description="User's clarification response")
    clarification_round:    int  = Field(default=0,  description="How many times clarified")

    # ── Forensics (written by N13) ────────────────────────────────────────
    forensic_flags:     list  = Field(default_factory=list, description="N13 anomaly flags")
    risk_score:         float = Field(default=0.0,  description="0-100 forensic risk score")
    benford_chi2:       float = Field(default=0.0)
    benford_p_value:    float = Field(default=1.0)

    # ── Debate + output (written by N15-N19) ──────────────────────────────
    piv_round:              int   = Field(default=0)
    final_answer_pre_xgb:   str   = Field(default="", description="N15 mediator output")
    confidence_score:       float = Field(default=0.0, description="Final confidence 0-1")
    agreement_status:       str   = Field(default="", description="unanimous/majority/full_disagree")
    shap_values:            list  = Field(default_factory=list, description="N16 feature attribution")
    causal_dag_path:        str   = Field(default="", description="N16 DAG PNG path")
    xgb_ranked_answer:      str   = Field(default="", description="N17 arbiter output")
    final_answer:           str   = Field(default="", description="N19 final verified answer")
    output_docx_path:       str   = Field(default="", description="N19 report path")

    # ── C9: RLEF fields — PRIVATE — never in outputs ──────────────────────
    _rlef_grade:    Optional[float] = None
    _rlef_chosen:   Optional[str]   = None
    _rlef_rejected: Optional[str]   = None
    _rlef_session_hash: Optional[str] = None

    # ── Validators ────────────────────────────────────────────────────────

    @model_validator(mode="after")
    def enforce_seed(self):
        """C5: seed must always be 42"""
        if self.seed != 42:
            raise ValueError(f"[C5 VIOLATION] seed must be 42, got {self.seed}")
        return self

    @model_validator(mode="after")
    def enforce_max_attempts(self):
        """A2: attempt counters must never exceed 5"""
        for field, val in [
            ("analyst_attempt_count",  self.analyst_attempt_count),
            ("quant_attempt_count",    self.quant_attempt_count),
            ("auditor_attempt_count",  self.auditor_attempt_count),
        ]:
            if val > self.piv_max_attempts:
                raise ValueError(
                    f"[A2 VIOLATION] {field}={val} exceeds max_attempts={self.piv_max_attempts}"
                )
        return self

    # ── Helper methods ────────────────────────────────────────────────────

    def check_ram(self) -> float:
        """C4: Check RAM — warn at 12GB, alert at 13GB, halt at 14GB"""
        return ResourceGovernor.check()

    def needs_clarification(self) -> bool:
        """A3: True when all 3 pods have exhausted 5 attempts"""
        return (
            self.analyst_attempt_count  >= self.piv_max_attempts and
            self.analyst_piv_status     == PIVStatus.REJECT and
            self.quant_attempt_count    >= self.piv_max_attempts and
            self.quant_piv_status       == PIVStatus.REJECT and
            self.auditor_attempt_count  >= self.piv_max_attempts and
            self.auditor_piv_status     == PIVStatus.REJECT
        )

    def reset_for_clarification(self, user_answer: str):
        """A3: Reset attempt counters after user provides clarification"""
        self.clarification_answer  = user_answer
        self.clarification_status  = ClarificationStatus.ANSWERED
        self.clarification_round  += 1
        self.analyst_attempt_count  = 0
        self.quant_attempt_count    = 0
        self.auditor_attempt_count  = 0
        self.analyst_piv_status     = PIVStatus.PENDING
        self.quant_piv_status       = PIVStatus.PENDING
        self.auditor_piv_status     = PIVStatus.PENDING
        self.analyst_rejection_reason  = ""
        self.quant_rejection_reason    = ""
        self.auditor_rejection_reason  = ""
        return self

    def safe_dict(self) -> dict:
        """C9: Return state dict with all _rlef_ fields removed"""
        d = self.model_dump()
        return {k: v for k, v in d.items() if not k.startswith("_rlef_")}

    def chunk_metadata_prefix(self) -> str:
        """C8: Build the mandatory chunk metadata prefix"""
        return (
            f"{self.company_name} / {self.doc_type} / "
            f"{self.fiscal_year} / {{section}} / {{page}}"
        )


# ── Quick sanity test when run directly ──────────────────────────────────────
if __name__ == "__main__":
    from rich import print as rprint

    rprint("[bold green]Creating BAState...[/bold green]")
    state = BAState(session_id="test-001")

    rprint(f"[cyan]seed:[/cyan] {state.seed}")
    rprint(f"[cyan]piv_max_attempts:[/cyan] {state.piv_max_attempts}")
    rprint(f"[cyan]RAM usage:[/cyan] {state.check_ram():.2f} GB")
    rprint(f"[cyan]chunk prefix template:[/cyan] {state.chunk_metadata_prefix()}")

    # Test C9 — _rlef_ fields must not appear in safe output
    safe = state.safe_dict()
    rlef_keys = [k for k in safe if k.startswith("_rlef_")]
    assert len(rlef_keys) == 0, f"C9 VIOLATION: {rlef_keys}"
    rprint("[bold green]C9 check passed — no _rlef_ fields in output[/bold green]")

    # Test A2 — seed violation caught
    try:
        bad = BAState(session_id="bad", seed=99)
        assert False, "Should have raised"
    except ValueError as e:
        rprint(f"[bold green]C5 check passed — caught: {e}[/bold green]")

    rprint("[bold green]BAState v2 ready.[/bold green]")