"""
N16 SHAP + Causal DAG — Explainability and Causal Reasoning
PDR-BAAAI-001 · Rev 1.0 · Node N16
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

SEED         = 42
SHAP_ROW_CAP = 500
DAG_DPI      = 300

CAUSAL_EDGES = [
    ("Revenue",            "Gross Profit"),
    ("Cost of Revenue",    "Gross Profit"),
    ("Gross Profit",       "Operating Income"),
    ("Operating Expenses", "Operating Income"),
    ("Operating Income",   "Net Income"),
    ("Interest Expense",   "Net Income"),
    ("Tax Expense",        "Net Income"),
    ("Net Income",         "EPS"),
    ("Share Count",        "EPS"),
]


def compute_shap_importance(
    chunks: List[Dict],
    answer: str,
    seed:   int = SEED,
) -> Optional[Dict]:
    if not chunks or not answer:
        return None

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.ensemble import RandomForestClassifier
        import shap

        capped = chunks[:SHAP_ROW_CAP]
        texts  = [
            c.get("text", "") or c.get("page_content", "")
            for c in capped
        ]
        texts = [t for t in texts if t.strip()]

        if len(texts) < 2:
            return None

        answer_words = set(answer.lower().split())
        labels = []
        for text in texts:
            overlap = len(answer_words & set(text.lower().split()))
            labels.append(1 if overlap > 2 else 0)

        if len(set(labels)) < 2:
            labels = [i % 2 for i in range(len(labels))]

        vectorizer = TfidfVectorizer(max_features=50, ngram_range=(1, 2))
        X          = vectorizer.fit_transform(texts).toarray().astype(float)

        model = RandomForestClassifier(
            n_estimators=10, random_state=seed, max_depth=3
        )
        model.fit(X, labels)

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # For binary classification take class-1 values
        if isinstance(shap_values, list) and len(shap_values) > 1:
            sv = np.array(shap_values[1])
        elif isinstance(shap_values, list):
            sv = np.array(shap_values[0])
        else:
            sv = np.array(shap_values)

        # Ensure 2D: shape (n_samples, n_features)
        if sv.ndim == 1:
            sv = sv.reshape(1, -1)

        mean_shap     = np.abs(sv).mean(axis=0)
        feature_names = vectorizer.get_feature_names_out().tolist()
        top_indices   = np.argsort(mean_shap)[::-1][:10]

        top_features = [
            {
                "feature":    feature_names[int(i)],
                "importance": round(float(mean_shap[int(i)]), 6),
            }
            for i in top_indices
        ]

        top_chunks = []
        for idx in range(min(len(texts), 10)):
            chunk_shap = float(np.abs(sv[idx]).sum()) if idx < sv.shape[0] else 0.0
            top_chunks.append({
                "chunk_id": capped[idx].get("chunk_id", f"chunk_{idx}"),
                "section":  capped[idx].get("section",  "UNKNOWN"),
                "page":     capped[idx].get("page",     0),
                "shap_sum": round(chunk_shap, 6),
            })

        top_chunks.sort(key=lambda x: x["shap_sum"], reverse=True)

        return {
            "top_features": top_features,
            "top_chunks":   top_chunks[:5],
            "n_chunks":     len(capped),
            "n_features":   len(feature_names),
        }

    except Exception as exc:
        logger.warning("SHAP computation failed: %s", exc)
        return None


def build_causal_dag(
    output_path:     Optional[str] = None,
    highlight_nodes: List[str]     = None,
) -> Optional[str]:
    try:
        import networkx as nx
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        G = nx.DiGraph()
        G.add_edges_from(CAUSAL_EDGES)

        pos = {
            "Revenue":            (0,  2),
            "Cost of Revenue":    (0,  1),
            "Operating Expenses": (2,  1),
            "Gross Profit":       (1,  2),
            "Operating Income":   (1,  1),
            "Interest Expense":   (0,  0),
            "Tax Expense":        (2,  0),
            "Net Income":         (1,  0),
            "Share Count":        (2, -1),
            "EPS":                (1, -1),
        }

        highlight    = set(highlight_nodes or [])
        node_colors  = [
            "#e74c3c" if n in highlight else "#3498db"
            for n in G.nodes()
        ]

        fig, ax = plt.subplots(figsize=(10, 8))
        nx.draw(
            G, pos, ax=ax,
            with_labels     = True,
            node_color      = node_colors,
            node_size       = 2000,
            font_size       = 9,
            font_color      = "white",
            font_weight     = "bold",
            arrows          = True,
            arrowsize       = 20,
            edge_color      = "#7f8c8d",
            connectionstyle = "arc3,rad=0.1",
        )
        ax.set_title(
            "Financial Causal DAG — Revenue to EPS",
            fontsize=12, fontweight="bold", pad=15,
        )

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            plt.savefig(output_path, dpi=DAG_DPI, bbox_inches="tight")
            plt.close(fig)
            logger.info("N16 Causal DAG saved: %s", output_path)
            return output_path
        else:
            plt.close(fig)
            return "dag_built_no_path"

    except Exception as exc:
        logger.warning("Causal DAG build failed: %s", exc)
        return None


class SHAPDAGNode:
    """N16 SHAP + Causal DAG Node."""

    def __init__(self, output_dir: str = "outputs") -> None:
        self.output_dir = output_dir

    def run(self, state) -> object:
        """
        LangGraph N16 node.
        Reads:  state.retrieval_stage_2, state.final_answer_pre_xgb
        Writes: state.shap_values, state.feature_importance,
                state.causal_dag_path   ← matches BAState field name
        """
        chunks = getattr(state, "retrieval_stage_2",    []) or []
        answer = getattr(state, "final_answer_pre_xgb", "") or ""

        result = self.explain(chunks=chunks, answer=answer,
                              output_dir=self.output_dir)

        # Use correct BAState field names
        state.shap_values        = result.get("shap")
        state.feature_importance = result.get("feature_importance")
        state.causal_dag_path    = result.get("dag_path")   # ← correct name

        logger.info(
            "N16 SHAP+DAG: shap=%s | dag=%s",
            result.get("shap") is not None,
            result.get("dag_path"),
        )
        return state

    def explain(
        self,
        chunks:     List[Dict],
        answer:     str,
        output_dir: str = "outputs",
    ) -> Dict:
        shap_result = compute_shap_importance(chunks=chunks, answer=answer,
                                              seed=SEED)

        feature_importance = None
        if shap_result:
            feature_importance = {
                f["feature"]: f["importance"]
                for f in shap_result.get("top_features", [])
            }

        highlight = self._detect_relevant_nodes(answer)
        dag_path  = os.path.join(output_dir, "causal_dag.png")
        dag_saved = build_causal_dag(output_path=dag_path,
                                     highlight_nodes=highlight)

        return {
            "shap":               shap_result,
            "feature_importance": feature_importance,
            "dag_path":           dag_saved,
            "highlight_nodes":    highlight,
        }

    @staticmethod
    def _detect_relevant_nodes(answer: str) -> List[str]:
        if not answer:
            return []

        node_keywords = {
            "Revenue":            ["revenue", "net sales", "total sales"],
            "Gross Profit":       ["gross profit", "gross margin"],
            "Operating Income":   ["operating income", "ebit", "operating profit"],
            "Net Income":         ["net income", "net earnings", "profit"],
            "EPS":                ["eps", "earnings per share", "diluted"],
            "Cost of Revenue":    ["cost of revenue", "cogs", "cost of sales"],
            "Operating Expenses": ["operating expense", "sg&a", "r&d"],
            "Tax Expense":        ["tax", "income tax"],
            "Interest Expense":   ["interest expense", "interest"],
        }

        answer_lower = answer.lower()
        return [
            node for node, kws in node_keywords.items()
            if any(kw in answer_lower for kw in kws)
        ]


def run_shap_dag(state, output_dir: str = "outputs") -> object:
    """Convenience wrapper for LangGraph N16 node."""
    return SHAPDAGNode(output_dir=output_dir).run(state)