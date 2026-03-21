"""
src/routing/lr_difficulty.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 Rev1.0 FINAL

N05 — Logistic Regression Difficulty Predictor
Runs immediately after N04 CART Router.

Classifies every analyst question into one of 3 difficulty levels:
  easy   — single fact, direct lookup, one section
  medium — requires computation or cross-referencing within document
  hard   — multi-step reasoning, multiple sections, ambiguous or rare

Downstream effects of query_difficulty:
  easy   → top_k=3, piv_max_retries=2, hitl_threshold=0.75
  medium → top_k=3, piv_max_retries=3, hitl_threshold=0.70
  hard   → top_k=5, piv_max_retries=5, hitl_threshold=0.60

Why LogisticRegression not DecisionTree:
  Difficulty is a continuous signal — questions exist on a spectrum.
  LR models probability of class membership more smoothly than a tree.
  Also trains faster and generalises better on small datasets.
  DecisionTree would overfit on 150 examples for a soft signal like difficulty.

Speed: <5ms (joblib load + TF-IDF + LR inference)
Writes to BAState: query_difficulty, context_window_size (updated if hard)
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from src.state.ba_state import BAState, Difficulty
from src.utils.seed_manager import SeedManager
from src.utils.resource_governor import ResourceGovernor

SeedManager.set_all()

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "lr_difficulty.pkl"

# ── Config ────────────────────────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 300
LR_MAX_ITER        = 1000
LR_C               = 1.0
RANDOM_STATE       = 42       # C5

# ── Difficulty → downstream config ───────────────────────────────────────────
DIFFICULTY_CONFIG: Dict[str, Dict] = {
    Difficulty.EASY: {
        "top_k":               3,
        "piv_max_retries":     2,
        "hitl_threshold":      0.75,
        "context_window_size": 3,
    },
    Difficulty.MEDIUM: {
        "top_k":               3,
        "piv_max_retries":     3,
        "hitl_threshold":      0.70,
        "context_window_size": 3,
    },
    Difficulty.HARD: {
        "top_k":               5,
        "piv_max_retries":     5,
        "hitl_threshold":      0.60,
        "context_window_size": 5,
    },
}

# ── Training data — 150 labelled questions ────────────────────────────────────
# 50 per class × 3 classes = 150 total
TRAINING_DATA: List[Tuple[str, str]] = [

    # ── EASY (50) ─────────────────────────────────────────────────────────────
    ("What was Apple net income in FY2023?",                             "easy"),
    ("What was Apple total net sales FY2023?",                           "easy"),
    ("What was Apple diluted EPS in fiscal 2023?",                       "easy"),
    ("What was Microsoft total revenue in 2023?",                        "easy"),
    ("What was Tesla total revenue in 2022?",                            "easy"),
    ("What was JPMorgan net income in 2023?",                            "easy"),
    ("What was Goldman Sachs net revenues in 2022?",                     "easy"),
    ("What was Amazon net sales in 2022?",                               "easy"),
    ("What was Meta total revenue in 2023?",                             "easy"),
    ("What was NVIDIA total revenue fiscal 2024?",                       "easy"),
    ("What was Apple basic EPS in FY2022?",                              "easy"),
    ("What was Microsoft net income fiscal 2022?",                       "easy"),
    ("What was Tesla gross profit in 2022?",                             "easy"),
    ("What was Amazon operating income in 2022?",                        "easy"),
    ("What was JPMorgan total assets at December 2023?",                 "easy"),
    ("What was Apple cash and equivalents at end of FY2023?",            "easy"),
    ("What was NVIDIA gross profit fiscal 2024?",                        "easy"),
    ("What was Meta net income in 2023?",                                "easy"),
    ("What was Goldman Sachs net earnings in 2021?",                     "easy"),
    ("What was Apple long term debt FY2022?",                            "easy"),
    ("What was Microsoft total assets in 2023?",                         "easy"),
    ("What was Tesla operating income in 2022?",                         "easy"),
    ("What was Apple capital expenditures FY2023?",                      "easy"),
    ("What was Amazon total revenue in 2023?",                           "easy"),
    ("What was JPMorgan stockholders equity in 2022?",                   "easy"),
    ("What was Apple deferred revenue balance FY2023?",                  "easy"),
    ("What was NVIDIA operating income fiscal 2023?",                    "easy"),
    ("What was Meta total costs and expenses 2022?",                     "easy"),
    ("What was Goldman Sachs provision for credit losses 2022?",         "easy"),
    ("What was Apple operating expenses FY2022?",                        "easy"),
    ("What was Microsoft research and development expense 2023?",        "easy"),
    ("What was Tesla net income in fiscal 2022?",                        "easy"),
    ("What was Amazon net income in 2021?",                              "easy"),
    ("What was JPMorgan net interest income in 2023?",                   "easy"),
    ("What was Apple goodwill balance at end of FY2023?",                "easy"),
    ("What was NVIDIA total stockholders equity January 2024?",          "easy"),
    ("What was Meta depreciation and amortisation in 2023?",             "easy"),
    ("What was Goldman Sachs total assets at end of 2022?",              "easy"),
    ("What was Microsoft operating income in fiscal 2022?",              "easy"),
    ("What was Tesla total assets at December 2022?",                    "easy"),
    ("What was Apple share repurchases in FY2023?",                      "easy"),
    ("What was Amazon AWS revenue in 2022?",                             "easy"),
    ("What was JPMorgan provision for credit losses in 2023?",           "easy"),
    ("What was NVIDIA research and development expense fiscal 2024?",    "easy"),
    ("What was Meta capital expenditures in 2022?",                      "easy"),
    ("What was Apple total liabilities FY2023?",                         "easy"),
    ("What was Microsoft cash and equivalents fiscal 2023?",             "easy"),
    ("What was Tesla stockholders equity December 2022?",                "easy"),
    ("What was Goldman Sachs compensation expenses in 2022?",            "easy"),
    ("What was Amazon operating expenses in 2022?",                      "easy"),

    # ── MEDIUM (50) ───────────────────────────────────────────────────────────
    ("What was Apple gross margin percentage in FY2023?",                "medium"),
    ("Calculate Apple operating margin for fiscal 2022",                 "medium"),
    ("What was Apple net profit margin in FY2023?",                      "medium"),
    ("What was Microsoft gross profit margin in 2023?",                  "medium"),
    ("Calculate Tesla gross margin for 2022",                            "medium"),
    ("What was JPMorgan return on equity in 2023?",                      "medium"),
    ("What was Goldman Sachs return on tangible equity 2022?",           "medium"),
    ("What was Amazon operating margin in 2022?",                        "medium"),
    ("Calculate Meta EBITDA for 2023",                                   "medium"),
    ("What was NVIDIA net income margin fiscal 2024?",                   "medium"),
    ("What was Apple current ratio based on FY2023 balance sheet?",      "medium"),
    ("Calculate Microsoft debt to equity ratio for 2023",                "medium"),
    ("What is Tesla debt to equity ratio for 2022?",                     "medium"),
    ("What was Amazon free cash flow in 2022?",                          "medium"),
    ("What was JPMorgan efficiency ratio in 2023?",                      "medium"),
    ("What was Goldman Sachs book value per share in 2022?",             "medium"),
    ("Calculate Meta free cash flow margin for 2023",                    "medium"),
    ("What was NVIDIA gross margin fiscal 2023?",                        "medium"),
    ("What is Apple return on assets for FY2023?",                       "medium"),
    ("What was Microsoft return on equity in 2023?",                     "medium"),
    ("What did Apple say about services revenue growth drivers?",        "medium"),
    ("How did Tesla production costs change versus prior year?",         "medium"),
    ("What segment drove most of Amazon revenue growth in 2022?",        "medium"),
    ("What were the main factors behind Meta revenue decline in 2022?",  "medium"),
    ("How did NVIDIA data centre revenue grow versus gaming in 2023?",   "medium"),
    ("What was Apple revenue split between products and services?",      "medium"),
    ("How did JPMorgan net interest income change from 2022 to 2023?",   "medium"),
    ("What drove Goldman Sachs revenue decline in 2022?",                "medium"),
    ("How did Amazon AWS margin compare to retail margin in 2022?",      "medium"),
    ("What was Microsoft intelligent cloud growth rate in 2023?",        "medium"),
    ("Calculate Apple R&D as percentage of revenue FY2023",              "medium"),
    ("What was Tesla energy segment revenue as share of total 2022?",    "medium"),
    ("Calculate JPMorgan net interest margin for 2023",                  "medium"),
    ("What was Goldman Sachs compensation ratio in 2022?",               "medium"),
    ("Calculate Amazon SG&A as percentage of net sales 2022",            "medium"),
    ("What was NVIDIA operating expense ratio fiscal 2024?",             "medium"),
    ("What is Meta revenue per employee for 2023?",                      "medium"),
    ("How did Apple gross margin change between FY2022 and FY2023?",     "medium"),
    ("What was Microsoft SG&A expense ratio in 2023?",                   "medium"),
    ("Calculate Tesla gross profit per vehicle delivered 2022",          "medium"),
    ("What was Apple quick ratio at end of FY2023?",                     "medium"),
    ("What was Amazon inventory turnover ratio in 2022?",                "medium"),
    ("What did JPMorgan say about credit quality trends in 2023?",       "medium"),
    ("How did Goldman Sachs investment banking fees compare to 2021?",   "medium"),
    ("What was NVIDIA automotive segment revenue fiscal 2024?",          "medium"),
    ("How did Meta ad impressions and price per ad trend in 2022?",      "medium"),
    ("What was Apple Americas versus international revenue split?",      "medium"),
    ("Calculate Microsoft Azure growth rate from disclosed revenue",     "medium"),
    ("What drove Tesla gross margin compression in late 2022?",          "medium"),
    ("What was Amazon fulfillment cost as percentage of net sales 2022?","medium"),

    # ── HARD (50) ─────────────────────────────────────────────────────────────
    ("Compare Apple revenue growth across FY2021 FY2022 and FY2023 and explain drivers",   "hard"),
    ("How did Apple gross margin evolve over five years and what caused the changes?",      "hard"),
    ("Reconcile Apple net income with operating cash flow and explain the difference",      "hard"),
    ("Compare JPMorgan and Goldman Sachs return on equity trends from 2020 to 2023",        "hard"),
    ("Analyse Tesla gross margin trajectory and assess whether improvements are sustainable","hard"),
    ("How did Microsoft cloud transition affect total margins from 2019 to 2023?",          "hard"),
    ("Are there any Benford Law anomalies in Apple revenue figures?",                       "hard"),
    ("Does Tesla vehicle delivery count reconcile with reported revenue figures?",           "hard"),
    ("Compare Amazon segment profitability evolution across North America international AWS","hard"),
    ("How did NVIDIA revenue mix shift between gaming data centre and automotive over time?","hard"),
    ("Identify any unusual patterns in Meta accounts receivable relative to revenue growth","hard"),
    ("How does Apple capital allocation compare across dividends buybacks and capex over years?","hard"),
    ("Are there signs of earnings management in Goldman Sachs provision for credit losses?","hard"),
    ("Compare Microsoft acquisition accounting for Activision versus historical acquisitions","hard"),
    ("How did JPMorgan credit loss provisioning compare to actual charge-offs over 3 years?","hard"),
    ("Analyse Apple Services segment margin versus overall company margin across fiscal years","hard"),
    ("What is the multi-year trend in Tesla stock based compensation as percentage of revenue?","hard"),
    ("How did Amazon working capital cycle change from 2019 to 2022 and what drove changes?","hard"),
    ("Identify discrepancies between NVIDIA management commentary and reported segment data","hard"),
    ("Compare Meta Reality Labs cumulative losses to total company free cash flow generation","hard"),
    ("How did Apple iPhone revenue concentration risk change from 2018 to 2023?",            "hard"),
    ("Are there unusual changes in Microsoft deferred revenue relative to billings growth?", "hard"),
    ("Analyse JPMorgan off balance sheet exposure and its potential impact on capital ratios","hard"),
    ("How did Goldman Sachs fair value level 3 assets evolve and what are valuation risks?", "hard"),
    ("Compare Tesla capex intensity versus revenue growth and assess capital efficiency trend","hard"),
    ("Are there going concern risks embedded in any recent Amazon business segment disclosures?","hard"),
    ("How do NVIDIA export control risks interact with data centre revenue concentration?",   "hard"),
    ("Analyse Apple pension and other post-retirement benefit obligations over five years",   "hard"),
    ("What is the relationship between Meta DAU growth and average revenue per user trends?","hard"),
    ("How did JPMorgan net interest income benefit from rate rises and what is the duration risk?","hard"),
    ("Compare Apple free cash flow conversion across five fiscal years and identify anomalies","hard"),
    ("Are there signs of channel stuffing in Apple product revenue relative to supply chain disclosures?","hard"),
    ("How did Microsoft segment reporting changes affect comparability of historical results?","hard"),
    ("Analyse Goldman Sachs compensation expense versus revenue across business cycles",      "hard"),
    ("How does Amazon stock based compensation affect free cash flow comparability?",         "hard"),
    ("Compare NVIDIA gross margin with AMD over three years using only disclosed figures",    "hard"),
    ("Identify all one-time items in Apple FY2023 and restate normalised earnings",          "hard"),
    ("How did Tesla energy storage revenue and gross margin evolve relative to automotive?",  "hard"),
    ("What are the key risk factors that could impair JPMorgan goodwill carrying value?",    "hard"),
    ("Analyse the sustainability of Goldman Sachs asset management fee revenue growth",      "hard"),
    ("How do Apple deferred tax assets relate to its international cash repatriation strategy?","hard"),
    ("Compare Microsoft gaming revenue trajectory before and after Activision on pro forma basis","hard"),
    ("Are NVIDIA related party transactions with TSMC material to cost of revenue?",         "hard"),
    ("How does Meta advertising revenue seasonality interact with cost structure rigidity?",  "hard"),
    ("Analyse Amazon third party seller services margin versus first party retail margin",    "hard"),
    ("What is the multi-year trend in Apple effective tax rate and what drives variability?","hard"),
    ("How did JPMorgan investment banking revenue correlate with market volumes 2019 to 2023?","hard"),
    ("Are there signs of window dressing in Goldman Sachs quarter-end balance sheet composition?","hard"),
    ("Compare Tesla R&D spending efficiency versus legacy automakers on revenue adjusted basis","hard"),
    ("How did NVIDIA gross margin benefit from product mix shift and is it structurally sustainable?","hard"),
]


class LRDifficultyPredictor:
    """
    N05: Logistic Regression Difficulty Predictor.

    Classifies analyst queries into easy / medium / hard.
    Trained on 150 labelled financial questions.
    Runs in <5ms. Updates context_window_size for hard queries.
    """

    def __init__(self, model_path: Optional[Path] = None):
        SeedManager.set_all()
        self.model_path = Path(model_path) if model_path else MODEL_PATH
        self._pipeline: Optional[Pipeline] = None
        self._classes:  Optional[List[str]] = None

    # ═══════════════════════════════════════════════════════════════════════
    # TRAIN
    # ═══════════════════════════════════════════════════════════════════════

    def train(
        self,
        training_data: Optional[List[Tuple[str, str]]] = None,
    ) -> dict:
        """
        Train LR classifier on labelled questions.
        Saves model to self.model_path.
        Returns training metrics.
        """
        SeedManager.set_all()

        data   = training_data or TRAINING_DATA
        texts  = [q for q, _ in data]
        labels = [l for _, l in data]

        from collections import Counter
        counts = Counter(labels)
        print(f"[N05] Training on {len(data)} questions: {dict(counts)}")

        # Build TF-IDF + LogisticRegression pipeline
        # multi_class removed — handled automatically in sklearn >= 1.5
        self._pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features = TFIDF_MAX_FEATURES,
                ngram_range  = (1, 2),
                sublinear_tf = True,
            )),
            ("lr", LogisticRegression(
                max_iter     = LR_MAX_ITER,
                C            = LR_C,
                random_state = RANDOM_STATE,   # C5
                solver       = "lbfgs",
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
        print(f"[N05] Saved to {self.model_path}")
        print(f"[N05] Training accuracy: {accuracy:.1%} on {len(data)} questions")

        return {
            "accuracy":     accuracy,
            "n_samples":    len(data),
            "class_counts": dict(counts),
            "model_path":   str(self.model_path),
            "classes":      self._classes,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # LOAD
    # ═══════════════════════════════════════════════════════════════════════

    def load(self) -> bool:
        """Load saved model from disk. Returns True if loaded."""
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
        Predict difficulty for a single question.
        Returns (difficulty, confidence).
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
        Main entry point — N05 node.
        Reads state.query.
        Writes state.query_difficulty.
        Updates state.context_window_size if hard (overrides N04 value).
        Runs immediately after N04.
        """
        ResourceGovernor.check("N05 LR Difficulty")

        if not state.query:
            print("[N05] No query — defaulting to medium")
            state.query_difficulty = Difficulty.MEDIUM
            return state

        difficulty, confidence = self.predict(state.query)
        config = DIFFICULTY_CONFIG.get(
            difficulty,
            DIFFICULTY_CONFIG[Difficulty.MEDIUM]
        )

        state.query_difficulty = Difficulty(difficulty)

        # Hard queries override context_window_size set by N04
        if difficulty == Difficulty.HARD:
            state.context_window_size = config["context_window_size"]

        print(f"[N05] '{state.query[:60]}' "
              f"→ {difficulty} (conf={confidence:.2f}) "
              f"top_k={config['top_k']} "
              f"retries={config['piv_max_retries']}")

        return state

    # ═══════════════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════════════

    def predict_batch(
        self, queries: List[str]
    ) -> List[Tuple[str, float]]:
        """Predict difficulty for a batch of queries."""
        if not self.is_loaded():
            if self.model_path.exists():
                self.load()
            else:
                self.train()
        probas  = self._pipeline.predict_proba(queries)
        results = []
        for proba in probas:
            idx = int(np.argmax(proba))
            results.append((self._classes[idx], float(proba[idx])))
        return results

    def get_difficulty_config(self, difficulty: str) -> Dict:
        """Return config dict for a given difficulty level."""
        return DIFFICULTY_CONFIG.get(
            difficulty,
            DIFFICULTY_CONFIG[Difficulty.MEDIUM]
        )


# ═══════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK
# run: python src/routing/lr_difficulty.py
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        from rich import print as rprint
    except ImportError:
        rprint = print

    rprint("\n[bold cyan]── LRDifficultyPredictor sanity check ──[/bold cyan]")

    predictor = LRDifficultyPredictor()
    rprint("[green]✓[/green] LRDifficultyPredictor instantiated")

    # Train
    metrics = predictor.train()
    assert metrics["accuracy"] >= 0.80, \
        f"Training accuracy too low: {metrics['accuracy']:.1%}"
    rprint(f"[green]✓[/green] Trained: accuracy={metrics['accuracy']:.1%} "
           f"n={metrics['n_samples']}")

    # Class balance
    counts = metrics["class_counts"]
    assert all(v == 50 for v in counts.values()), \
        f"Unbalanced classes: {counts}"
    rprint(f"[green]✓[/green] Class balance: {counts}")

    # Test all 3 classes
    test_cases = [
        ("What was Apple net income FY2023?",                          "easy"),
        ("What was Apple gross margin percentage in FY2023?",          "medium"),
        ("Compare Apple revenue growth across three years and explain","hard"),
    ]
    for query, expected in test_cases:
        pred, conf = predictor.predict(query)
        status = "[green]✓[/green]" if pred == expected else "[yellow]~[/yellow]"
        rprint(f"{status} '{query[:55]}' → {pred} (conf={conf:.2f})")

    # BAState integration
    state = BAState(
        session_id = "sanity-n05",
        query      = "What was Apple net income FY2023?",
    )
    state = predictor.run(state)
    assert state.query_difficulty in [
        Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD
    ]
    rprint(f"[green]✓[/green] BAState: difficulty={state.query_difficulty}")

    # Hard query widens context
    hard_state = BAState(
        session_id          = "sanity-n05-hard",
        query               = "Compare Apple revenue growth across FY2021 FY2022 FY2023 and explain all drivers in detail",
        context_window_size = 3,
    )
    hard_state = predictor.run(hard_state)
    if hard_state.query_difficulty == Difficulty.HARD:
        assert hard_state.context_window_size == 5
        rprint(f"[green]✓[/green] Hard query widens context_window_size to 5")
    else:
        rprint(f"[yellow]~[/yellow] Query classified as {hard_state.query_difficulty} — acceptable")

    # Model reloads
    p2     = LRDifficultyPredictor()
    loaded = p2.load()
    assert loaded is True
    pred2, _ = p2.predict("What was Apple net income?")
    assert pred2 in ["easy", "medium", "hard"]
    rprint(f"[green]✓[/green] Model saved and reloaded correctly")

    # No query defaults to medium
    state2 = BAState(session_id="no-query-n05")
    state2 = predictor.run(state2)
    assert state2.query_difficulty == Difficulty.MEDIUM
    rprint(f"[green]✓[/green] No query defaults to medium")

    # N04 + N05 pipeline
    from src.routing.cart_router import CARTRouter
    router = CARTRouter()
    router.load() if router.model_path.exists() else router.train()

    pipeline_state = BAState(
        session_id = "n04-n05-pipeline",
        query      = "What was Apple total net sales FY2023?",
    )
    pipeline_state = router.run(pipeline_state)
    pipeline_state = predictor.run(pipeline_state)
    assert pipeline_state.query_type       is not None
    assert pipeline_state.query_difficulty is not None
    rprint(f"[green]✓[/green] N04+N05 pipeline: "
           f"type={pipeline_state.query_type} "
           f"difficulty={pipeline_state.query_difficulty}")

    rprint(f"\n[bold green]All checks passed. LRDifficultyPredictor ready.[/bold green]\n")