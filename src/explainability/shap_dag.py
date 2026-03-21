"""
src/explainability/shap_dag.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 Rev1.0 FINAL

N16 — SHAP + Causal DAG Explainability Node
Pure sklearn + shap + networkx + matplotlib. No LLM.

Part 1 — SHAP Feature Attribution
  Trains a lightweight XGBoost surrogate on retrieval features.
  Uses SHAP TreeExplainer to compute feature importance.
  500-row hard cap enforced (C4 RAM protection).
  Shows which retrieved chunks most influenced the final answer.

  Features per chunk:
    bm25_score       — BM25 keyword match score
    cosine_sim       — BGE-M3 semantic similarity
    section_weight   — section relevance weight
    page_position    — normalised page position 0-1
    chunk_length     — token count normalised
    has_table        — 1 if chunk contains table data
    fiscal_year_match— 1 if chunk FY matches query FY
    company_match    — 1 if chunk company matches query company

Part 2 — Causal DAG
  Standard financial causal chain using networkx DiGraph.
  Revenue → Gross Profit → Operating Income → Net Income → EPS
  Nodes populated with values extracted from BAState.
  Exported as PNG to state.causal_dag_path.

Writes to BAState:
  shap_values        — dict feature → shap value
  feature_importance — dict feature → importance score
  causal_dag_path    — path string to PNG (or None if matplotlib unavailable)
"""

import sys
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from src.state.ba_state          import BAState
from src.utils.seed_manager      import SeedManager
from src.utils.resource_governor import ResourceGovernor

SeedManager.set_all()

# ── Config ────────────────────────────────────────────────────────────────────
SHAP_ROW_CAP    = 500    # C4 hard cap
OUTPUT_DIR      = ROOT / "outputs"
DAG_DPI         = 150    # 300 DPI for production — 150 for speed in tests
SHAP_N_TREES    = 50     # small surrogate — fast + RAM-safe

# ── Financial causal chain ────────────────────────────────────────────────────
CAUSAL_EDGES = [
    ("Revenue",          "Gross Profit"),
    ("Gross Profit",     "Operating Income"),
    ("Cost of Revenue",  "Gross Profit"),
    ("Operating Expense","Operating Income"),
    ("Operating Income", "Net Income"),
    ("Interest Expense", "Net Income"),
    ("Tax Expense",      "Net Income"),
    ("Net Income",       "EPS"),
]

CAUSAL_NODES = [
    "Revenue", "Cost of Revenue", "Gross Profit",
    "Operating Expense", "Operating Income",
    "Interest Expense", "Tax Expense", "Net Income", "EPS",
]

# ── Feature names ─────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "bm25_score",
    "cosine_sim",
    "section_weight",
    "page_position",
    "chunk_length",
    "has_table",
    "fiscal_year_match",
    "company_match",
]


class SHAPDag:
    """
    N16 — SHAP + Causal DAG Explainability.

    Part 1: SHAP TreeExplainer on retrieval features.
    Part 2: Causal DAG with financial chain.
    No LLM. Pure statistical explainability.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        SeedManager.set_all()
        self.output_dir = output_dir or OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, state: BAState) -> BAState:
        """
        Main entry point — N16 node.

        Reads:  state.retrieval_stage_2, state.final_answer_pre_xgb,
                state.company_name, state.fiscal_year
        Writes: state.shap_values, state.feature_importance,
                state.causal_dag_path
        """
        ResourceGovernor.check("N16 SHAP+DAG")

        # ── Part 1: SHAP ───────────────────────────────────────────────────
        chunks = (
            state.retrieval_stage_2 or
            state.retrieval_stage_1 or
            []
        )[:SHAP_ROW_CAP]

        if chunks:
            shap_vals, feat_imp = self._compute_shap(
                chunks,
                state.query        or "",
                state.company_name or "",
                state.fiscal_year  or "",
            )
        else:
            shap_vals = {f: 0.0 for f in FEATURE_NAMES}
            feat_imp  = {f: 0.0 for f in FEATURE_NAMES}

        state.shap_values        = shap_vals
        state.feature_importance = feat_imp

        # ── Part 2: Causal DAG ─────────────────────────────────────────────
        node_values = self._extract_node_values(state)
        dag_path    = self._build_causal_dag(
            state.session_id or "session",
            node_values,
        )
        state.causal_dag_path = str(dag_path) if dag_path else None

        print(f"[N16] Complete — "
              f"shap_features={len(shap_vals)} "
              f"dag={'saved' if dag_path else 'skipped'}")

        return state

    # ═══════════════════════════════════════════════════════════════════════
    # PART 1 — SHAP
    # ═══════════════════════════════════════════════════════════════════════

    def _compute_shap(
        self,
        chunks:      List[Dict[str, Any]],
        query:       str,
        company:     str,
        fiscal_year: str,
    ) -> tuple:
        """
        Compute SHAP values using XGBoost surrogate.
        Returns (shap_values dict, feature_importance dict).
        """
        try:
            import shap
            import xgboost as xgb
        except ImportError:
            print("[N16] shap/xgboost not available — skipping SHAP")
            empty = {f: 0.0 for f in FEATURE_NAMES}
            return empty, empty

        SeedManager.set_all()

        # Build feature matrix
        X = self._build_feature_matrix(chunks, query, company, fiscal_year)

        if X.shape[0] < 2:
            empty = {f: 0.0 for f in FEATURE_NAMES}
            return empty, empty

        # Synthetic relevance labels — use cosine_sim as proxy
        # In production these come from RLEF grades
        y = X[:, 1].copy()   # cosine_sim column as relevance proxy

        # Train lightweight surrogate
        model = xgb.XGBRegressor(
            n_estimators      = SHAP_N_TREES,
            max_depth         = 3,
            learning_rate     = 0.1,
            random_state      = 42,       # C5
            verbosity         = 0,
            tree_method       = "hist",
        )
        model.fit(X, y)

        # SHAP TreeExplainer — 500-row cap already applied
        explainer   = shap.TreeExplainer(model)
        shap_matrix = explainer.shap_values(X)

        # Mean absolute SHAP per feature
        mean_shap = np.abs(shap_matrix).mean(axis=0)

        shap_vals = {
            FEATURE_NAMES[i]: round(float(mean_shap[i]), 6)
            for i in range(len(FEATURE_NAMES))
        }

        # Feature importance from XGBoost
        raw_imp = model.feature_importances_
        feat_imp = {
            FEATURE_NAMES[i]: round(float(raw_imp[i]), 6)
            for i in range(len(FEATURE_NAMES))
        }

        return shap_vals, feat_imp

    def _build_feature_matrix(
        self,
        chunks:      List[Dict[str, Any]],
        query:       str,
        company:     str,
        fiscal_year: str,
    ) -> np.ndarray:
        """Build feature matrix from retrieval chunks."""
        rows = []
        for chunk in chunks:
            text     = chunk.get("text") or chunk.get("content") or ""
            section  = chunk.get("section", "")
            page     = chunk.get("page", "0")
            chunk_fy = chunk.get("fiscal_year", "")
            chunk_co = chunk.get("company", "")

            # bm25_score — use stored score or proxy
            bm25 = float(chunk.get("bm25_score", 0.5))

            # cosine_sim — use stored score or proxy
            cos = float(chunk.get("cosine_sim", 0.5))

            # section_weight — financial sections weighted higher
            sec_lower = section.lower()
            if any(k in sec_lower for k in
                   ["income", "financial", "balance", "cash"]):
                sec_w = 1.0
            elif any(k in sec_lower for k in
                     ["md&a", "management", "operations"]):
                sec_w = 0.8
            else:
                sec_w = 0.5

            # page_position — normalised 0-1 (assume max 300 pages)
            try:
                page_pos = min(float(str(page).strip()) / 300.0, 1.0)
            except (ValueError, TypeError):
                page_pos = 0.5

            # chunk_length — normalised by 1000 chars
            chunk_len = min(len(text) / 1000.0, 1.0)

            # has_table — simple heuristic
            has_table = 1.0 if any(
                c in text for c in ["$", "%", "|", "\t"]
            ) else 0.0

            # fiscal_year_match
            fy_match = 1.0 if (
                fiscal_year and fiscal_year in chunk_fy
            ) else 0.0

            # company_match
            co_match = 1.0 if (
                company and company.lower() in chunk_co.lower()
            ) else 0.0

            rows.append([
                bm25, cos, sec_w, page_pos,
                chunk_len, has_table, fy_match, co_match,
            ])

        if not rows:
            return np.zeros((1, len(FEATURE_NAMES)))

        return np.array(rows, dtype=np.float32)

    # ═══════════════════════════════════════════════════════════════════════
    # PART 2 — CAUSAL DAG
    # ═══════════════════════════════════════════════════════════════════════

    def _extract_node_values(
        self, state: BAState
    ) -> Dict[str, Optional[float]]:
        """
        Extract financial values for DAG nodes from BAState.
        Sources: final_answer_pre_xgb, retrieval chunks, table_cells.
        """
        text_sources = []

        if state.final_answer_pre_xgb:
            text_sources.append(state.final_answer_pre_xgb)

        for chunk in (state.retrieval_stage_2 or [])[:3]:
            t = chunk.get("text") or chunk.get("content") or ""
            if t:
                text_sources.append(t)

        combined = " ".join(text_sources)

        # Pattern matching for financial line items
        patterns = {
            "Revenue":          r'(?:total\s+)?(?:net\s+)?(?:revenue|sales)[^\d]*\$?([\d,]+)',
            "Gross Profit":     r'gross\s+profit[^\d]*\$?([\d,]+)',
            "Operating Income": r'operating\s+(?:income|profit)[^\d]*\$?([\d,]+)',
            "Net Income":       r'net\s+income[^\d]*\$?([\d,]+)',
            "EPS":              r'(?:diluted\s+)?(?:earnings|eps)[^\d]*\$?([\d.]+)',
            "Cost of Revenue":  r'cost\s+of\s+(?:revenue|goods)[^\d]*\$?([\d,]+)',
            "Interest Expense": r'interest\s+expense[^\d]*\$?([\d,]+)',
            "Tax Expense":      r'(?:income\s+)?tax[^\d]*\$?([\d,]+)',
            "Operating Expense":r'operating\s+expense[^\d]*\$?([\d,]+)',
        }

        node_values = {}
        for node, pattern in patterns.items():
            match = re.search(pattern, combined, re.IGNORECASE)
            if match:
                try:
                    val = float(match.group(1).replace(",", ""))
                    node_values[node] = val
                except (ValueError, IndexError):
                    node_values[node] = None
            else:
                node_values[node] = None

        return node_values

    def _build_causal_dag(
        self,
        session_id:  str,
        node_values: Dict[str, Optional[float]],
    ) -> Optional[Path]:
        """
        Build financial causal DAG using networkx.
        Saves PNG to output_dir.
        Returns path or None if matplotlib unavailable.
        """
        try:
            import networkx as nx
            import matplotlib
            matplotlib.use("Agg")   # non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            print("[N16] networkx/matplotlib not available — skipping DAG")
            return None

        SeedManager.set_all()

        G = nx.DiGraph()

        # Add nodes
        for node in CAUSAL_NODES:
            G.add_node(node)

        # Add edges
        for src, dst in CAUSAL_EDGES:
            G.add_edge(src, dst)

        # Node colours — green if value present, grey if missing
        node_colors = []
        for node in G.nodes():
            val = node_values.get(node)
            if val is not None and val > 0:
                node_colors.append("#2ecc71")   # green — value found
            elif val is not None and val < 0:
                node_colors.append("#e74c3c")   # red — negative value
            else:
                node_colors.append("#95a5a6")   # grey — no data

        # Node labels with values
        labels = {}
        for node in G.nodes():
            val = node_values.get(node)
            if val is not None:
                labels[node] = f"{node}\n${val:,.0f}M"
            else:
                labels[node] = node

        # Layout
        pos = {
            "Revenue":           (0, 4),
            "Cost of Revenue":   (2, 4),
            "Gross Profit":      (1, 3),
            "Operating Expense": (2, 2),
            "Operating Income":  (1, 2),
            "Interest Expense":  (2, 1),
            "Tax Expense":       (0, 1),
            "Net Income":        (1, 1),
            "EPS":               (1, 0),
        }

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_facecolor("#f8f9fa")
        fig.patch.set_facecolor("#f8f9fa")

        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color = node_colors,
            node_size  = 2000,
            alpha      = 0.9,
        )
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color = "#2c3e50",
            arrows     = True,
            arrowsize  = 20,
            width      = 2,
        )
        nx.draw_networkx_labels(
            G, pos, labels=labels, ax=ax,
            font_size   = 7,
            font_weight = "bold",
            font_color  = "#2c3e50",
        )

        ax.set_title(
            "Financial Causal Chain — FinBench AI",
            fontsize=12, fontweight="bold", pad=15,
        )
        ax.axis("off")
        plt.tight_layout()

        # Save
        safe_id  = re.sub(r'[^\w-]', '_', session_id)[:40]
        out_path = self.output_dir / f"causal_dag_{safe_id}.png"
        plt.savefig(out_path, dpi=DAG_DPI, bbox_inches="tight")
        plt.close(fig)

        return out_path

    # ═══════════════════════════════════════════════════════════════════════
    # PUBLIC HELPERS
    # ═══════════════════════════════════════════════════════════════════════

    def compute_shap_for_chunks(
        self,
        chunks:      List[Dict[str, Any]],
        query:       str       = "",
        company:     str       = "",
        fiscal_year: str       = "",
    ) -> Dict[str, float]:
        """Public interface — compute SHAP values for given chunks."""
        shap_vals, _ = self._compute_shap(chunks, query, company, fiscal_year)
        return shap_vals

    def build_dag_for_state(self, state: BAState) -> Optional[str]:
        """Public interface — build DAG and return path string."""
        node_values = self._extract_node_values(state)
        path        = self._build_causal_dag(
            state.session_id or "session", node_values
        )
        return str(path) if path else None


# ═══════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK
# run: python src/explainability/shap_dag.py
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        from rich import print as rprint
    except ImportError:
        rprint = print

    rprint("\n[bold cyan]-- SHAPDag (N16) sanity check --[/bold cyan]")

    explainer = SHAPDag()
    rprint("[green]✓[/green] SHAPDag instantiated")

    # Test with Apple 10-K data
    state = BAState(
        session_id          = "sanity-n16",
        query               = "What was Apple net income FY2023?",
        company_name        = "Apple Inc",
        fiscal_year         = "FY2023",
        final_answer_pre_xgb= "Net income was $96,995 million FY2023 "
                              "[Financial Statements/P42].",
        retrieval_stage_2   = [
            {
                "text":        "Net income: $96,995 million. "
                               "Revenue: $383,285 million. "
                               "Gross profit: $169,148 million.",
                "section":     "Financial Statements",
                "page":        "42",
                "company":     "Apple Inc",
                "fiscal_year": "FY2023",
                "bm25_score":  0.85,
                "cosine_sim":  0.92,
            },
            {
                "text":        "Operating income: $114,301 million. "
                               "EPS diluted: $6.13.",
                "section":     "Financial Statements",
                "page":        "43",
                "company":     "Apple Inc",
                "fiscal_year": "FY2023",
                "bm25_score":  0.72,
                "cosine_sim":  0.88,
            },
        ],
    )

    state = explainer.run(state)

    rprint(f"[green]✓[/green] shap_values: {len(state.shap_values)} features")
    rprint(f"[green]✓[/green] feature_importance: "
           f"{len(state.feature_importance)} features")
    rprint(f"[green]✓[/green] causal_dag_path: {state.causal_dag_path}")
    rprint(f"[green]✓[/green] seed: {state.seed}")

    assert len(state.shap_values)        == 8
    assert len(state.feature_importance) == 8
    assert state.seed                    == 42
    assert all(isinstance(v, float)
               for v in state.shap_values.values())

    # Test node value extraction
    node_vals = explainer._extract_node_values(state)
    rprint(f"[green]✓[/green] Node values extracted: "
           f"{[k for k,v in node_vals.items() if v is not None]}")

    # Test empty state
    state2 = BAState(session_id="sanity-n16-empty")
    state2 = explainer.run(state2)
    assert state2.shap_values        == {f: 0.0 for f in
                                         ["bm25_score","cosine_sim",
                                          "section_weight","page_position",
                                          "chunk_length","has_table",
                                          "fiscal_year_match","company_match"]}
    rprint(f"[green]✓[/green] Empty state handled — zero SHAP values")

    rprint(f"\n[bold green]All checks passed. SHAPDag N16 ready.[/bold green]\n")