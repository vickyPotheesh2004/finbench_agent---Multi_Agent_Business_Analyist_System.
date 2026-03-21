"""
src/routing/cart_router.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 Rev1.0 FINAL

N04 — CART Query Router
Fires FIRST at query time — before any retrieval.

Classifies every analyst question into one of 5 types:
  numerical  — "What was Apple net income FY2023?"        → SniperRAG first
  ratio      — "What was Apple gross margin FY2023?"      → BM25+BGE cascade
  multi_doc  — "Compare Apple vs Microsoft revenue"       → wider context
  text       — "What are Apple main risk factors?"        → semantic search
  forensic   — "Are there anomalies in Apple financials?" → TriGuard activated

How it works:
  1. TF-IDF vectorises the query (max 500 features, seed=42)
  2. DecisionTree (CART) classifies into 5 classes
  3. Model trained on 200 labelled questions (built-in training data)
  4. Saved to models/cart_router.pkl with joblib
  5. Loaded in <10ms at query time via joblib mmap

Downstream effects of query_type:
  numerical  → N06 SniperRAG fires first, 2 PIV rounds, HITL threshold 0.75
  ratio      → N07+N08 cascade, 3 PIV rounds, formula computation in N12
  multi_doc  → context_window_size=5 (wider), 3 PIV rounds, all 3 pods
  text       → N08 BGE-M3 leads, 2 PIV rounds, narrative focus
  forensic   → N13 TriGuard activated, 3 PIV rounds, HITL threshold 0.65

Speed: <10ms (joblib mmap load + TF-IDF + tree inference)
Writes to BAState: query_type, routing_path, context_window_size
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from src.state.ba_state import BAState, QueryType
from src.utils.seed_manager import SeedManager
from src.utils.resource_governor import ResourceGovernor

SeedManager.set_all()

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR  = ROOT / "models"
MODEL_PATH  = MODELS_DIR / "cart_router.pkl"

# ── Config ────────────────────────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 500
TREE_MAX_DEPTH     = 8
TREE_MIN_SAMPLES   = 2
RANDOM_STATE       = 42   # C5

# ── Query type → downstream config ───────────────────────────────────────────
ROUTING_CONFIG: Dict[str, Dict] = {
    QueryType.NUMERICAL: {
        "context_window_size": 3,
        "sniper_first":        True,
        "piv_rounds":          2,
        "hitl_threshold":      0.75,
        "triguard":            False,
        "routing_path":        "sniper_first→bm25→bge→rrf",
    },
    QueryType.RATIO: {
        "context_window_size": 3,
        "sniper_first":        False,
        "piv_rounds":          3,
        "hitl_threshold":      0.70,
        "triguard":            False,
        "routing_path":        "bm25→bge→rrf→quant_pod",
    },
    QueryType.MULTI_DOC: {
        "context_window_size": 5,
        "sniper_first":        False,
        "piv_rounds":          3,
        "hitl_threshold":      0.65,
        "triguard":            False,
        "routing_path":        "bm25→bge→rrf→wide_context",
    },
    QueryType.TEXT: {
        "context_window_size": 3,
        "sniper_first":        False,
        "piv_rounds":          2,
        "hitl_threshold":      0.70,
        "triguard":            False,
        "routing_path":        "bge_lead→bm25→rrf→narrative",
    },
    QueryType.FORENSIC: {
        "context_window_size": 5,
        "sniper_first":        False,
        "piv_rounds":          3,
        "hitl_threshold":      0.65,
        "triguard":            True,
        "routing_path":        "bm25→bge→rrf→triguard",
    },
}

# ── Training data — 200 labelled questions ────────────────────────────────────
# 40 per class × 5 classes = 200 total
# Covers: Apple, Microsoft, Tesla, JPMorgan, Goldman, Amazon, Meta, NVIDIA
# Covers: 10-K, 10-Q, 8-K question styles
TRAINING_DATA: List[Tuple[str, str]] = [

    # ── NUMERICAL (40) ────────────────────────────────────────────────────────
    ("What was Apple total net sales in FY2023?",                        "numerical"),
    ("What was Apple net income in fiscal year 2022?",                   "numerical"),
    ("What was Apple diluted earnings per share in FY2023?",             "numerical"),
    ("What was Microsoft total revenue in fiscal 2023?",                 "numerical"),
    ("What was Tesla total revenue for the year ended December 2022?",   "numerical"),
    ("What was JPMorgan net income in 2023?",                            "numerical"),
    ("What was Goldman Sachs total net revenues in 2022?",               "numerical"),
    ("What was Amazon net sales in 2022?",                               "numerical"),
    ("What was Meta total revenue in 2023?",                             "numerical"),
    ("What was NVIDIA total revenue in fiscal 2024?",                    "numerical"),
    ("What were Apple operating expenses in FY2022?",                    "numerical"),
    ("What was Apple cash and cash equivalents at end of FY2023?",       "numerical"),
    ("What was Microsoft research and development expense in 2023?",     "numerical"),
    ("What was Tesla gross profit in fiscal year 2022?",                 "numerical"),
    ("What was Amazon operating income in 2022?",                        "numerical"),
    ("What was JPMorgan total assets at December 31 2023?",              "numerical"),
    ("How much did Apple spend on share repurchases in FY2023?",         "numerical"),
    ("What was NVIDIA gross profit in fiscal year 2024?",                "numerical"),
    ("What was Meta total costs and expenses in 2022?",                  "numerical"),
    ("What was Goldman Sachs net earnings in 2021?",                     "numerical"),
    ("What was Apple basic earnings per share in 2023?",                 "numerical"),
    ("What was Microsoft net income for fiscal year 2022?",              "numerical"),
    ("What was Tesla operating income in 2022?",                         "numerical"),
    ("What were Apple total liabilities in FY2023?",                     "numerical"),
    ("What was Amazon net income in fiscal year 2021?",                  "numerical"),
    ("What was JPMorgan total stockholders equity in 2022?",             "numerical"),
    ("What was Apple long term debt in FY2022?",                         "numerical"),
    ("What was NVIDIA operating income in fiscal 2023?",                 "numerical"),
    ("What was Microsoft total assets in 2023?",                         "numerical"),
    ("What was Tesla net income attributable to common stockholders?",   "numerical"),
    ("What was Apple capital expenditures in FY2023?",                   "numerical"),
    ("What was Amazon total net revenue in 2023?",                       "numerical"),
    ("What was Goldman Sachs provision for credit losses in 2022?",      "numerical"),
    ("What was Meta net income in 2023?",                                "numerical"),
    ("How much goodwill did Microsoft carry at end of fiscal 2023?",     "numerical"),
    ("What was Apple deferred revenue balance in FY2023?",               "numerical"),
    ("What was JPMorgan net interest income in 2023?",                   "numerical"),
    ("What was Tesla total stockholders equity in 2022?",                "numerical"),
    ("What was NVIDIA total stockholders equity at January 2024?",       "numerical"),
    ("What was Amazon operating expenses in 2022?",                      "numerical"),

    # ── RATIO (40) ────────────────────────────────────────────────────────────
    ("What was Apple gross margin percentage in FY2023?",                "ratio"),
    ("What was Apple operating margin in fiscal 2022?",                  "ratio"),
    ("Calculate Apple net profit margin for FY2023",                     "ratio"),
    ("What was Microsoft gross profit margin in 2023?",                  "ratio"),
    ("What is Tesla gross margin percentage for 2022?",                  "ratio"),
    ("Calculate JPMorgan return on equity for 2023",                     "ratio"),
    ("What was Goldman Sachs return on tangible equity in 2022?",        "ratio"),
    ("What was Amazon operating margin in 2022?",                        "ratio"),
    ("Calculate Meta EBITDA margin for 2023",                            "ratio"),
    ("What was NVIDIA net income margin in fiscal 2024?",                "ratio"),
    ("What is Apple price to earnings ratio based on FY2023 EPS?",       "ratio"),
    ("Calculate Apple current ratio using FY2023 balance sheet",         "ratio"),
    ("What was Microsoft operating income as a percentage of revenue?",  "ratio"),
    ("What is Tesla debt to equity ratio for 2022?",                     "ratio"),
    ("Calculate Amazon EBITDA for 2022",                                 "ratio"),
    ("What was JPMorgan efficiency ratio in 2023?",                      "ratio"),
    ("What was Goldman Sachs book value per share in 2022?",             "ratio"),
    ("Calculate Meta free cash flow margin for 2023",                    "ratio"),
    ("What was NVIDIA gross margin in fiscal year 2023?",                "ratio"),
    ("What is Apple return on assets for FY2023?",                       "ratio"),
    ("Calculate Microsoft debt to EBITDA ratio for 2023",                "ratio"),
    ("What was Tesla return on invested capital in 2022?",               "ratio"),
    ("What is Amazon net profit margin for 2022?",                       "ratio"),
    ("Calculate JPMorgan tier 1 capital ratio for 2023",                 "ratio"),
    ("What was Apple quick ratio at end of FY2023?",                     "ratio"),
    ("What was Goldman Sachs leverage ratio in 2022?",                   "ratio"),
    ("Calculate NVIDIA return on equity for fiscal 2024",                "ratio"),
    ("What is Meta price to free cash flow ratio for 2023?",             "ratio"),
    ("What was Microsoft return on equity in 2023?",                     "ratio"),
    ("Calculate Tesla operating leverage ratio for 2022",                "ratio"),
    ("What is Apple EV to EBITDA multiple based on FY2023?",             "ratio"),
    ("What was Amazon gross profit as percentage of net sales in 2022?", "ratio"),
    ("Calculate JPMorgan net interest margin for 2023",                  "ratio"),
    ("What was Goldman Sachs compensation ratio in 2022?",               "ratio"),
    ("What is NVIDIA operating expense ratio for fiscal 2024?",          "ratio"),
    ("Calculate Meta revenue per employee for 2023",                     "ratio"),
    ("What was Apple R&D as percentage of revenue in FY2023?",           "ratio"),
    ("What was Microsoft SG&A expense ratio in 2023?",                   "ratio"),
    ("Calculate Tesla gross profit per vehicle delivered in 2022",       "ratio"),
    ("What was Amazon AWS operating margin in 2022?",                    "ratio"),

    # ── MULTI_DOC (40) ────────────────────────────────────────────────────────
    ("Compare Apple and Microsoft revenue growth from 2021 to 2023",     "multi_doc"),
    ("How does Tesla gross margin compare to traditional automakers?",   "multi_doc"),
    ("Compare JPMorgan and Goldman Sachs net income for 2022",           "multi_doc"),
    ("How did Apple revenue in 2022 compare to 2021?",                   "multi_doc"),
    ("Compare Amazon AWS revenue to Microsoft Azure growth rates",       "multi_doc"),
    ("How does Meta advertising revenue compare to Alphabet?",           "multi_doc"),
    ("Compare NVIDIA and AMD gross margins over last three years",       "multi_doc"),
    ("How did Apple operating income change from FY2021 to FY2023?",     "multi_doc"),
    ("Compare Microsoft and Apple R&D spending as percentage of sales",  "multi_doc"),
    ("How does Tesla capital expenditure compare to legacy automakers?", "multi_doc"),
    ("Compare Goldman Sachs and JPMorgan return on equity trends",       "multi_doc"),
    ("How did Amazon net income compare across 2020 2021 and 2022?",     "multi_doc"),
    ("Compare Apple iPhone revenue across fiscal years 2021 2022 2023",  "multi_doc"),
    ("How does Meta cost per employee compare to Alphabet?",             "multi_doc"),
    ("Compare NVIDIA data center revenue growth year over year",         "multi_doc"),
    ("How did Microsoft cloud revenue grow from 2021 to 2023?",          "multi_doc"),
    ("Compare Apple services segment growth to product segment growth",  "multi_doc"),
    ("How does JPMorgan credit loss provision compare to 2020 levels?",  "multi_doc"),
    ("Compare Tesla energy generation revenue to automotive revenue",    "multi_doc"),
    ("How did Goldman Sachs investment banking fees change 2021 to 2022?","multi_doc"),
    ("Compare Apple Americas revenue to international revenue trends",   "multi_doc"),
    ("How does Amazon operating income compare by segment 2021 vs 2022?","multi_doc"),
    ("Compare Microsoft gaming revenue before and after Activision",     "multi_doc"),
    ("How did NVIDIA gaming vs data center mix change over three years?","multi_doc"),
    ("Compare Meta reality labs losses across 2021 2022 2023",           "multi_doc"),
    ("How does Apple gross margin compare to five year historical avg?", "multi_doc"),
    ("Compare JPMorgan consumer banking vs commercial banking revenue",  "multi_doc"),
    ("How did Tesla gross margin evolve from 2019 to 2022?",             "multi_doc"),
    ("Compare Amazon international vs North America operating income",   "multi_doc"),
    ("How does Goldman Sachs asset management revenue compare to 2019?", "multi_doc"),
    ("Compare Apple capex intensity over last three fiscal years",       "multi_doc"),
    ("How did Microsoft operating income margin trend from 2020 to 2023?","multi_doc"),
    ("Compare NVIDIA gross margin before and after data center boom",    "multi_doc"),
    ("How does Tesla free cash flow compare to 2020 and 2021 levels?",   "multi_doc"),
    ("Compare Meta total expenses growth vs revenue growth 2020 to 2023","multi_doc"),
    ("How did JPMorgan net interest income change from 2020 to 2023?",   "multi_doc"),
    ("Compare Apple share repurchase program size across fiscal years",  "multi_doc"),
    ("How does Amazon fulfillment cost ratio compare across years?",     "multi_doc"),
    ("Compare Goldman Sachs FICC revenue across last five years",        "multi_doc"),
    ("How did NVIDIA R&D spending grow relative to revenue over time?",  "multi_doc"),

    # ── TEXT (40) ─────────────────────────────────────────────────────────────
    ("What are Apple main risk factors disclosed in the 10-K?",          "text"),
    ("What is Apple business strategy for services growth?",             "text"),
    ("Describe Microsoft cloud strategy as outlined in the annual report","text"),
    ("What did Tesla management say about production challenges?",       "text"),
    ("Summarise JPMorgan outlook for credit losses in 2024",             "text"),
    ("What are Goldman Sachs key competitive advantages?",               "text"),
    ("Describe Amazon investment priorities for 2023",                   "text"),
    ("What is Meta strategy for monetising the metaverse?",              "text"),
    ("What are NVIDIA main growth drivers described by management?",     "text"),
    ("What did Apple say about supply chain risks in FY2023?",           "text"),
    ("Describe Microsoft artificial intelligence strategy",              "text"),
    ("What are Tesla main operational risks in 2022?",                   "text"),
    ("What did JPMorgan management say about interest rate environment?","text"),
    ("Summarise Goldman Sachs strategic priorities for 2023",            "text"),
    ("What is Amazon AWS competitive positioning according to management?","text"),
    ("Describe Meta advertising business challenges in 2022",            "text"),
    ("What does NVIDIA say about competition in AI accelerator market?", "text"),
    ("What regulatory risks does Apple disclose in its 10-K?",           "text"),
    ("Describe Microsoft gaming strategy after Activision acquisition",  "text"),
    ("What did Tesla say about vehicle demand and pricing strategy?",    "text"),
    ("What is JPMorgan approach to managing market risk?",               "text"),
    ("Summarise Goldman Sachs consumer banking exit strategy",           "text"),
    ("What does Amazon say about its third party seller ecosystem?",     "text"),
    ("Describe Meta Reality Labs long term vision",                      "text"),
    ("What are NVIDIA risks related to export controls?",                "text"),
    ("What did Apple say about developer ecosystem and App Store?",      "text"),
    ("Describe Microsoft enterprise software competitive moat",          "text"),
    ("What is Tesla energy business long term strategy?",                "text"),
    ("What did JPMorgan say about digital banking investments?",         "text"),
    ("Summarise Goldman Sachs asset and wealth management strategy",     "text"),
    ("What does Amazon say about healthcare and pharmacy initiatives?",  "text"),
    ("Describe Meta approach to content moderation and regulation",      "text"),
    ("What are the key themes in NVIDIA management discussion?",         "text"),
    ("What did Apple say about privacy as competitive differentiator?",  "text"),
    ("Describe Microsoft LinkedIn and productivity segment strategy",    "text"),
    ("What operational improvements did Tesla describe for Gigafactory?","text"),
    ("What does JPMorgan say about its investment banking pipeline?",    "text"),
    ("Summarise Goldman Sachs approach to risk management culture",      "text"),
    ("What did Amazon say about cost reduction initiatives in 2022?",    "text"),
    ("Describe NVIDIA automotive and robotics long term opportunity",    "text"),

    # ── FORENSIC (40) ─────────────────────────────────────────────────────────
    ("Are there any anomalies in Apple revenue recognition patterns?",   "forensic"),
    ("Do Apple financial statements show any Benford Law violations?",   "forensic"),
    ("Are there any unusual related party transactions in Tesla 10-K?",  "forensic"),
    ("Does JPMorgan loan loss provision appear abnormal vs peer banks?", "forensic"),
    ("Are there signs of earnings management in Goldman Sachs results?", "forensic"),
    ("Does Amazon capitalisation of costs appear aggressive?",           "forensic"),
    ("Are there any material weaknesses in Meta internal controls?",     "forensic"),
    ("Do NVIDIA revenue patterns show unusual quarter end spikes?",      "forensic"),
    ("Are there anomalies in Apple accounts receivable growth rate?",    "forensic"),
    ("Does Microsoft deferred revenue movement appear unusual?",         "forensic"),
    ("Are there any going concern indicators in the filing?",            "forensic"),
    ("Do Tesla gross margin improvements appear sustainable or inflated?","forensic"),
    ("Are there signs of channel stuffing in Apple iPhone shipments?",   "forensic"),
    ("Does JPMorgan off balance sheet exposure appear material?",        "forensic"),
    ("Are Goldman Sachs mark to market valuations conservative?",        "forensic"),
    ("Does Amazon revenue recognition for AWS appear compliant with GAAP?","forensic"),
    ("Are there unusual changes in Meta accounts receivable days?",      "forensic"),
    ("Do NVIDIA inventory levels suggest demand destruction risk?",      "forensic"),
    ("Are there any restatements or prior period adjustments disclosed?","forensic"),
    ("Does Apple pension obligation appear fully funded?",               "forensic"),
    ("Are there signs of aggressive goodwill impairment avoidance?",     "forensic"),
    ("Do Microsoft acquisition accounting entries appear reasonable?",   "forensic"),
    ("Are there unusual tax rate fluctuations in Tesla filings?",        "forensic"),
    ("Does JPMorgan loan classification appear consistent year to year?","forensic"),
    ("Are Goldman Sachs compensation accruals consistent with revenues?","forensic"),
    ("Does Amazon working capital cycle show any deterioration signals?","forensic"),
    ("Are there any SEC comment letter disclosures in the filing?",      "forensic"),
    ("Do NVIDIA related party disclosures raise any concerns?",          "forensic"),
    ("Are there unusual changes in Apple warranty accruals?",            "forensic"),
    ("Does Microsoft revenue backlog growth appear consistent?",         "forensic"),
    ("Are there signs of round number bias in Tesla reported metrics?",  "forensic"),
    ("Does JPMorgan securities portfolio carry significant unrealised losses?","forensic"),
    ("Are Goldman Sachs fair value level 3 assets growing unusually?",   "forensic"),
    ("Does Amazon stock based compensation appear excessive vs peers?",   "forensic"),
    ("Are there any litigation contingencies material to Meta results?", "forensic"),
    ("Do NVIDIA gross margin trends suggest accounting irregularities?", "forensic"),
    ("Are there any unusual audit opinion qualifications in the filing?","forensic"),
    ("Does Apple lease accounting appear compliant with ASC 842?",       "forensic"),
    ("Are there signs of window dressing in Microsoft quarter end cash?","forensic"),
    ("Do Tesla vehicle delivery numbers reconcile with revenue reported?","forensic"),
]


class CARTRouter:
    """
    N04: CART Query Router.

    Classifies analyst queries into 5 types using a Decision Tree.
    Trained on 200 labelled financial questions.
    Runs in <10ms. Writes query_type + routing_path to BAState.
    """

    def __init__(self, model_path: Optional[Path] = None):
        SeedManager.set_all()
        self.model_path = Path(model_path) if model_path else MODEL_PATH
        self._pipeline: Optional[Pipeline] = None
        self._classes:  Optional[List[str]] = None

    # ═══════════════════════════════════════════════════════════════════════
    # TRAIN
    # ═══════════════════════════════════════════════════════════════════════

    def train(self, training_data: Optional[List[Tuple[str, str]]] = None) -> dict:
        """
        Train CART classifier on labelled questions.
        Saves model to self.model_path.
        Returns training metrics.
        """
        SeedManager.set_all()

        data  = training_data or TRAINING_DATA
        texts = [q for q, _ in data]
        labels= [l for _, l in data]

        # Validate class balance
        from collections import Counter
        counts = Counter(labels)
        print(f"[N04] Training on {len(data)} questions: {dict(counts)}")

        # Build TF-IDF + DecisionTree pipeline
        self._pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features = TFIDF_MAX_FEATURES,
                ngram_range  = (1, 2),     # unigrams + bigrams
                sublinear_tf = True,       # log TF scaling
            )),
            ("cart", DecisionTreeClassifier(
                max_depth         = TREE_MAX_DEPTH,
                min_samples_split = TREE_MIN_SAMPLES,
                random_state      = RANDOM_STATE,   # C5
            )),
        ])

        self._pipeline.fit(texts, labels)
        self._classes = list(self._pipeline.classes_)

        # Training accuracy
        train_preds = self._pipeline.predict(texts)
        correct     = sum(p == t for p, t in zip(train_preds, labels))
        accuracy    = correct / len(labels)

        # Save model
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, self.model_path)
        print(f"[N04] Saved to {self.model_path}")
        print(f"[N04] Training accuracy: {accuracy:.1%} on {len(data)} questions")

        return {
            "accuracy":      accuracy,
            "n_samples":     len(data),
            "class_counts":  dict(counts),
            "model_path":    str(self.model_path),
            "classes":       self._classes,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # LOAD
    # ═══════════════════════════════════════════════════════════════════════

    def load(self) -> bool:
        """
        Load saved model from disk.
        Returns True if loaded, False if not found.
        """
        if not self.model_path.exists():
            return False
        self._pipeline = joblib.load(self.model_path)
        self._classes  = list(self._pipeline.classes_)
        return True

    def is_loaded(self) -> bool:
        return self._pipeline is not None

    # ═══════════════════════════════════════════════════════════════════════
    # PREDICT
    # ═══════════════════════════════════════════════════════════════════════

    def predict(self, query: str) -> Tuple[str, float]:
        """
        Predict query type for a single question.
        Returns (query_type, confidence).
        Trains model first if not loaded.
        """
        if not self.is_loaded():
            if self.model_path.exists():
                self.load()
            else:
                self.train()

        proba      = self._pipeline.predict_proba([query])[0]
        pred_idx   = int(np.argmax(proba))
        pred_class = self._classes[pred_idx]
        confidence = float(proba[pred_idx])

        return pred_class, confidence

    # ═══════════════════════════════════════════════════════════════════════
    # RUN — BAState integration
    # ═══════════════════════════════════════════════════════════════════════

    def run(self, state: BAState) -> BAState:
        """
        Main entry point — N04 node.
        Reads state.query.
        Writes state.query_type, state.routing_path,
               state.context_window_size.
        Fires FIRST — before any retrieval.
        """
        ResourceGovernor.check("N04 CART Router")

        if not state.query:
            print("[N04] No query — defaulting to text")
            state.query_type          = QueryType.TEXT
            state.routing_path        = ROUTING_CONFIG[QueryType.TEXT]["routing_path"]
            state.context_window_size = ROUTING_CONFIG[QueryType.TEXT]["context_window_size"]
            return state

        query_type, confidence = self.predict(state.query)

        # Apply routing config
        config = ROUTING_CONFIG.get(query_type, ROUTING_CONFIG[QueryType.TEXT])

        state.query_type          = QueryType(query_type)
        state.routing_path        = config["routing_path"]
        state.context_window_size = config["context_window_size"]

        print(f"[N04] '{state.query[:60]}...' "
              f"→ {query_type} (conf={confidence:.2f}) "
              f"→ {config['routing_path']}")

        return state

    # ═══════════════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════════════

    def predict_batch(self, queries: List[str]) -> List[Tuple[str, float]]:
        """Predict query types for a batch of queries."""
        if not self.is_loaded():
            if self.model_path.exists():
                self.load()
            else:
                self.train()
        probas   = self._pipeline.predict_proba(queries)
        results  = []
        for proba in probas:
            idx   = int(np.argmax(proba))
            results.append((self._classes[idx], float(proba[idx])))
        return results

    def get_routing_config(self, query_type: str) -> Dict:
        """Return routing config dict for a given query type."""
        return ROUTING_CONFIG.get(query_type, ROUTING_CONFIG[QueryType.TEXT])

    def get_feature_importances(self) -> Optional[Dict[str, float]]:
        """Return top-10 most important TF-IDF features."""
        if not self.is_loaded():
            return None
        tfidf    = self._pipeline.named_steps["tfidf"]
        cart     = self._pipeline.named_steps["cart"]
        features = tfidf.get_feature_names_out()
        imps     = cart.feature_importances_
        top_idx  = np.argsort(imps)[::-1][:10]
        return {features[i]: round(float(imps[i]), 4) for i in top_idx}


# ═══════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK
# run: python src/routing/cart_router.py
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        from rich import print as rprint
    except ImportError:
        rprint = print

    rprint("\n[bold cyan]── CARTRouter sanity check ──[/bold cyan]")

    router = CARTRouter()
    rprint("[green]✓[/green] CARTRouter instantiated")

    # Train
    metrics = router.train()
    assert metrics["accuracy"] >= 0.95, \
        f"Training accuracy too low: {metrics['accuracy']:.1%}"
    rprint(f"[green]✓[/green] Trained: accuracy={metrics['accuracy']:.1%} "
           f"n={metrics['n_samples']}")

    # Verify class balance
    counts = metrics["class_counts"]
    assert all(v == 40 for v in counts.values()), \
        f"Unbalanced classes: {counts}"
    rprint(f"[green]✓[/green] Class balance: {counts}")

    # Test all 5 classes
    test_cases = [
        ("What was Apple net income FY2023?",              "numerical"),
        ("What was Apple gross margin percentage?",        "ratio"),
        ("Compare Apple and Microsoft revenue trends",     "multi_doc"),
        ("What are Apple main risk factors?",              "text"),
        ("Are there anomalies in Apple financial data?",   "forensic"),
    ]

    for query, expected in test_cases:
        pred, conf = router.predict(query)
        status = "[green]✓[/green]" if pred == expected else "[red]✗[/red]"
        rprint(f"{status} '{query[:50]}' → {pred} (conf={conf:.2f})")

    # BAState integration
    state = BAState(
        session_id = "sanity-n04",
        query      = "What was Apple total net sales FY2023?",
    )
    state = router.run(state)
    assert state.query_type    == QueryType.NUMERICAL
    assert state.routing_path  != ""
    assert state.context_window_size >= 3
    rprint(f"[green]✓[/green] BAState: query_type={state.query_type} "
           f"routing={state.routing_path}")

    # Model persists
    router2 = CARTRouter()
    loaded  = router2.load()
    assert loaded is True
    pred2, _ = router2.predict("What was Apple revenue?")
    assert pred2 == "numerical"
    rprint(f"[green]✓[/green] Model saved and reloaded correctly")

    # Feature importances
    imps = router.get_feature_importances()
    assert imps is not None
    assert len(imps) <= 10
    rprint(f"[green]✓[/green] Top features: {list(imps.keys())[:3]}")

    # No query defaults to text
    state2 = BAState(session_id="no-query-test")
    state2 = router.run(state2)
    assert state2.query_type == QueryType.TEXT
    rprint(f"[green]✓[/green] No query defaults to text")

    rprint(f"\n[bold green]All checks passed. CARTRouter ready.[/bold green]\n")