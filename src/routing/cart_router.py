"""
N04 CART Query Router — 5-Class Query Type Classification
PDR-BAAAI-001 · Rev 1.0 · Node N04

Purpose:
    Classify every analyst query into one of 5 types BEFORE any retrieval.
    Controls downstream pipeline behaviour:
        numerical  → activate SniperRAG (N06) first
        ratio      → activate CFO/Quant Pod (N12) formula mode
        multi_doc  → widen context window, activate BlindAuditor (N14)
        text       → standard PIV loop, narrative retrieval
        forensic   → activate TriGuard (N13), extra Validator scrutiny

Architecture:
    Trained on 200 labelled questions (40 per class) using
    sklearn DecisionTreeClassifier with TF-IDF features.
    Model saved with joblib. Loads in <10ms at query time.

Constraints satisfied:
    C1  $0 cost — sklearn is free
    C2  100% local — zero network calls
    C5  seed=42 — random_state=42 on all sklearn objects
    C7  N/A — no LLM prompt at this node
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

QUERY_CLASSES = ["numerical", "ratio", "multi_doc", "text", "forensic"]

_MODEL_DIR          = os.path.join("models")
_MODEL_FILENAME     = "cart_router.pkl"
_VECTORIZER_FILENAME= "cart_vectorizer.pkl"
_SEED               = 42
_MIN_CONFIDENCE     = 0.0   # always return a class — never abstain

# Routing configuration per query type
ROUTING_CONFIG: Dict[str, Dict] = {
    "numerical": {
        "sniper_first":      True,
        "piv_rounds":        1,
        "context_window":    3,
        "triguard_active":   False,
        "description":       "Direct numerical extraction — SniperRAG first",
    },
    "ratio": {
        "sniper_first":      False,
        "piv_rounds":        2,
        "context_window":    3,
        "triguard_active":   False,
        "description":       "Formula computation — CFO/Quant pod activated",
    },
    "multi_doc": {
        "sniper_first":      False,
        "piv_rounds":        2,
        "context_window":    5,
        "triguard_active":   False,
        "description":       "Multi-document — wider context window",
    },
    "text": {
        "sniper_first":      False,
        "piv_rounds":        1,
        "context_window":    3,
        "triguard_active":   False,
        "description":       "Narrative — standard PIV loop",
    },
    "forensic": {
        "sniper_first":      False,
        "piv_rounds":        2,
        "context_window":    5,
        "triguard_active":   True,
        "description":       "Forensic — TriGuard activated",
    },
}


# ── Training data — 200 labelled questions (40 per class) ─────────────────────

TRAINING_DATA: List[Tuple[str, str]] = [
    # ── numerical (40) ────────────────────────────────────────────────────────
    ("What was total net sales in fiscal year 2023?",                      "numerical"),
    ("What was net income for FY2022?",                                    "numerical"),
    ("What was diluted earnings per share in 2023?",                       "numerical"),
    ("What were total assets as of September 2023?",                       "numerical"),
    ("What was gross profit in fiscal 2022?",                              "numerical"),
    ("What was operating income for the year ended 2023?",                 "numerical"),
    ("What was cash and cash equivalents at year end?",                    "numerical"),
    ("What was long-term debt as of the balance sheet date?",              "numerical"),
    ("What were capital expenditures in fiscal 2023?",                     "numerical"),
    ("What was research and development expense?",                         "numerical"),
    ("What was total revenue for the quarter?",                            "numerical"),
    ("How much was shareholders equity?",                                  "numerical"),
    ("What were total liabilities at year end?",                           "numerical"),
    ("What was operating cash flow in FY2022?",                            "numerical"),
    ("What was basic earnings per share?",                                 "numerical"),
    ("What was cost of revenue for fiscal 2023?",                         "numerical"),
    ("How much was SG&A expense?",                                         "numerical"),
    ("What was interest expense for the year?",                            "numerical"),
    ("What was the income tax provision?",                                 "numerical"),
    ("What was goodwill on the balance sheet?",                            "numerical"),
    ("What was deferred revenue at year end?",                             "numerical"),
    ("What was accounts receivable net?",                                  "numerical"),
    ("How much inventory did the company hold?",                           "numerical"),
    ("What were total current assets?",                                    "numerical"),
    ("What were total current liabilities?",                               "numerical"),
    ("What was net cash from operating activities?",                       "numerical"),
    ("What was free cash flow in fiscal 2023?",                            "numerical"),
    ("What were dividends paid per share?",                                "numerical"),
    ("How much did the company spend on share repurchases?",               "numerical"),
    ("What was the effective tax rate?",                                   "numerical"),
    ("What was EBITDA for the fiscal year?",                               "numerical"),
    ("What was net revenue in the most recent quarter?",                   "numerical"),
    ("What were total operating expenses?",                                "numerical"),
    ("How much cash was generated from operations?",                       "numerical"),
    ("What was the book value per share?",                                 "numerical"),
    ("What was amortization expense for the year?",                        "numerical"),
    ("What were stock-based compensation expenses?",                       "numerical"),
    ("What was working capital at year end?",                              "numerical"),
    ("How much was the net pension liability?",                            "numerical"),
    ("What were property plant and equipment net?",                        "numerical"),

    # ── ratio (40) ────────────────────────────────────────────────────────────
    ("What was the gross margin percentage?",                              "ratio"),
    ("Calculate the current ratio for FY2023",                             "ratio"),
    ("What was the debt to equity ratio?",                                 "ratio"),
    ("What was the operating margin?",                                     "ratio"),
    ("Calculate return on equity for the year",                            "ratio"),
    ("What was the net profit margin?",                                    "ratio"),
    ("What was the price to earnings ratio?",                              "ratio"),
    ("Calculate the quick ratio",                                          "ratio"),
    ("What was return on assets?",                                         "ratio"),
    ("What was the asset turnover ratio?",                                 "ratio"),
    ("Calculate the interest coverage ratio",                              "ratio"),
    ("What was the EV to EBITDA multiple?",                                "ratio"),
    ("What was the debt to EBITDA ratio?",                                 "ratio"),
    ("Calculate return on invested capital",                               "ratio"),
    ("What was the inventory turnover ratio?",                             "ratio"),
    ("What was days sales outstanding?",                                   "ratio"),
    ("Calculate the cash conversion cycle",                                "ratio"),
    ("What was the dividend payout ratio?",                                "ratio"),
    ("What was the price to book ratio?",                                  "ratio"),
    ("Calculate the free cash flow yield",                                 "ratio"),
    ("What was EBITDA margin for fiscal 2023?",                            "ratio"),
    ("What was the revenue growth rate year over year?",                   "ratio"),
    ("Calculate the earnings per share growth rate",                       "ratio"),
    ("What was the return on capital employed?",                           "ratio"),
    ("What was the net debt to EBITDA ratio?",                             "ratio"),
    ("Calculate operating leverage for the year",                          "ratio"),
    ("What was the accounts receivable turnover?",                         "ratio"),
    ("What was the gross profit as a percentage of revenue?",              "ratio"),
    ("Calculate the capex to sales ratio",                                 "ratio"),
    ("What was the SG&A as a percentage of revenue?",                      "ratio"),
    ("What was the R&D intensity ratio?",                                   "ratio"),
    ("Calculate the altman z score",                                       "ratio"),
    ("What was the financial leverage ratio?",                             "ratio"),
    ("Calculate the equity multiplier",                                    "ratio"),
    ("What was the fixed asset turnover?",                                 "ratio"),
    ("What was the cash ratio?",                                           "ratio"),
    ("Calculate the net working capital ratio",                            "ratio"),
    ("What was the effective interest rate on debt?",                      "ratio"),
    ("What was the tax burden ratio?",                                     "ratio"),
    ("Calculate the DuPont decomposition of ROE",                          "ratio"),

    # ── multi_doc (40) ────────────────────────────────────────────────────────
    ("How did revenue compare between FY2022 and FY2023?",                 "multi_doc"),
    ("Compare gross margins across the last three years",                  "multi_doc"),
    ("How has net income trended over the past five years?",               "multi_doc"),
    ("Compare Apple revenue to Microsoft revenue",                         "multi_doc"),
    ("How did operating margins change from 2021 to 2023?",                "multi_doc"),
    ("Compare the balance sheet in FY2022 versus FY2023",                  "multi_doc"),
    ("How has EPS grown compared to the prior year?",                      "multi_doc"),
    ("Contrast revenue growth between segments over time",                 "multi_doc"),
    ("Compare cash flow from operations year over year",                   "multi_doc"),
    ("How have total assets changed over three fiscal years?",             "multi_doc"),
    ("Compare capital expenditure trends across periods",                  "multi_doc"),
    ("How did the debt level change between FY2021 and FY2023?",           "multi_doc"),
    ("Compare R&D spending as a percentage of revenue across years",       "multi_doc"),
    ("How has working capital evolved over the last two years?",           "multi_doc"),
    ("Compare iPhone revenue to Services revenue growth",                  "multi_doc"),
    ("How did gross profit trend from 2019 to 2023?",                      "multi_doc"),
    ("Compare the current ratio across multiple years",                    "multi_doc"),
    ("How has the share count changed year over year?",                    "multi_doc"),
    ("Compare dividend payments across fiscal years",                      "multi_doc"),
    ("How has the tax rate changed over time?",                            "multi_doc"),
    ("Compare operating expenses between Q1 and Q4",                       "multi_doc"),
    ("How did inventory levels change versus prior year?",                 "multi_doc"),
    ("Compare accounts receivable turnover across years",                  "multi_doc"),
    ("How has free cash flow generation trended?",                         "multi_doc"),
    ("Compare the segment profitability across fiscal years",              "multi_doc"),
    ("How did the Americas revenue compare to prior year?",                "multi_doc"),
    ("Compare net income margins between 2020 and 2023",                   "multi_doc"),
    ("How has goodwill changed since the last acquisition?",               "multi_doc"),
    ("Compare deferred revenue balances across periods",                   "multi_doc"),
    ("How did the effective tax rate change year over year?",              "multi_doc"),
    ("Compare iPhone units sold between fiscal years",                     "multi_doc"),
    ("How did services gross margin evolve over three years?",             "multi_doc"),
    ("Compare the interest coverage ratio across years",                   "multi_doc"),
    ("How has the debt maturity profile changed since FY2021?",            "multi_doc"),
    ("Compare research spending between Apple and the prior period",       "multi_doc"),
    ("How did operating cash flow compare to prior fiscal year?",          "multi_doc"),
    ("Compare the pension obligations across years",                       "multi_doc"),
    ("How has stock compensation expense trended?",                        "multi_doc"),
    ("Compare the book value per share across three years",                "multi_doc"),
    ("How did total revenue change between the quarterly reports?",        "multi_doc"),

    # ── text (40) ─────────────────────────────────────────────────────────────
    ("What are the main risk factors disclosed in Item 1A?",               "text"),
    ("Describe the company's business model and revenue streams",          "text"),
    ("What did management say about future guidance?",                     "text"),
    ("Summarise the MD&A section key points",                              "text"),
    ("What are the company's competitive advantages?",                     "text"),
    ("Describe the company's geographic revenue breakdown",                "text"),
    ("What is the company's strategy for the next fiscal year?",           "text"),
    ("What did the CEO say about product innovation?",                     "text"),
    ("Describe the company's supply chain risks",                          "text"),
    ("What regulatory risks are disclosed in the filing?",                 "text"),
    ("Summarise the auditor's report findings",                            "text"),
    ("What are the key accounting policies?",                              "text"),
    ("Describe the company's segment reporting structure",                 "text"),
    ("What did management highlight as the key growth drivers?",           "text"),
    ("What is the company's dividend policy?",                             "text"),
    ("Describe the employee headcount and workforce trends",               "text"),
    ("What litigation is disclosed in the legal proceedings section?",     "text"),
    ("Summarise the company's sustainability and ESG initiatives",         "text"),
    ("What are the key assumptions in the goodwill impairment test?",      "text"),
    ("Describe the related party transactions disclosed",                  "text"),
    ("What did management say about macroeconomic headwinds?",             "text"),
    ("Describe the company's approach to capital allocation",              "text"),
    ("What are the covenant terms on the revolving credit facility?",      "text"),
    ("Summarise the pension and post-retirement benefit disclosures",      "text"),
    ("What cybersecurity risks are disclosed?",                            "text"),
    ("Describe the company's product roadmap as disclosed",                "text"),
    ("What is the company's policy on share buybacks?",                    "text"),
    ("Describe the merger integration progress mentioned in MD&A",         "text"),
    ("What did management say about pricing power?",                       "text"),
    ("Summarise the going concern disclosures if any",                     "text"),
    ("What are the key judgements in lease accounting?",                   "text"),
    ("Describe the company's exposure to interest rate risk",              "text"),
    ("What did management say about customer concentration risk?",         "text"),
    ("Summarise the commitments and contingencies note",                   "text"),
    ("What is the company's revenue recognition policy?",                  "text"),
    ("Describe the executive compensation structure",                      "text"),
    ("What did management say about market share trends?",                 "text"),
    ("Summarise the subsequent events disclosed",                          "text"),
    ("What are the key assumptions in the stock option valuation?",        "text"),
    ("Describe the company's approach to foreign currency hedging",        "text"),

    # ── forensic (40) ─────────────────────────────────────────────────────────
    ("Are there any unusual patterns in the revenue recognition?",         "forensic"),
    ("Does the Benford law test flag anomalies in the financial data?",    "forensic"),
    ("Are accounts receivable growing faster than revenue?",               "forensic"),
    ("Are there signs of earnings manipulation in the reported figures?",  "forensic"),
    ("Does inventory growth outpace cost of goods sold?",                  "forensic"),
    ("Are there any restatements or prior period adjustments disclosed?",  "forensic"),
    ("Do the financial statements show signs of channel stuffing?",        "forensic"),
    ("Are there unusual related party transactions that could signal fraud?","forensic"),
    ("Does the cash flow statement contradict the income statement?",      "forensic"),
    ("Are there round number anomalies in the reported financials?",       "forensic"),
    ("Do the footnotes reveal any off-balance-sheet arrangements?",        "forensic"),
    ("Are there signs of aggressive revenue recognition?",                 "forensic"),
    ("Does the audit opinion contain any qualifications or concerns?",     "forensic"),
    ("Are there material weaknesses in internal controls?",                "forensic"),
    ("Do accruals appear unusually large relative to revenue?",            "forensic"),
    ("Are there signs of cookie jar accounting reserves?",                 "forensic"),
    ("Does the effective tax rate show unusual fluctuations?",             "forensic"),
    ("Are there unexplained changes in accounting policies?",              "forensic"),
    ("Do the segment disclosures show any anomalies?",                     "forensic"),
    ("Are there signs of premature revenue recognition in the quarter?",   "forensic"),
    ("Does the Beneish M-score indicate earnings manipulation?",           "forensic"),
    ("Are there unusual spikes in accounts payable days?",                 "forensic"),
    ("Do the pension assumptions appear overly optimistic?",               "forensic"),
    ("Are there signs of big bath accounting in the write-downs?",         "forensic"),
    ("Does the deferred revenue balance show unusual movements?",          "forensic"),
    ("Are there signs of improper capitalisation of expenses?",            "forensic"),
    ("Do the cash flow patterns suggest financial irregularities?",        "forensic"),
    ("Are there discrepancies between reported and pro forma earnings?",   "forensic"),
    ("Does the auditor flag going concern issues or emphasis of matter?",  "forensic"),
    ("Are there unusual year-end transactions that inflate revenue?",      "forensic"),
    ("Do the financial ratios show sudden unexplained improvements?",      "forensic"),
    ("Are there signs of window dressing in the balance sheet?",          "forensic"),
    ("Does the management tone in MD&A contradict the numbers?",           "forensic"),
    ("Are there signs of fictitious revenue in the receivables?",          "forensic"),
    ("Do the gross margins show an unusual step change?",                  "forensic"),
    ("Are there unexplained decreases in days payable outstanding?",       "forensic"),
    ("Does the cash conversion cycle show suspicious improvements?",       "forensic"),
    ("Are there signs of channel stuffing in the inventory data?",         "forensic"),
    ("Do the expense trends show signs of undisclosed cost cuts?",         "forensic"),
    ("Are there anomalies in the geographic revenue breakdown?",           "forensic"),
]


# ── CARTRouter ────────────────────────────────────────────────────────────────

class CARTRouter:
    """
    N04 CART Query Router.

    Classifies analyst queries into 5 types using a trained
    DecisionTreeClassifier on TF-IDF features.

    Two usage modes:
        1. router.classify(query) → (query_type, confidence, routing_config)
        2. router.run(ba_state)   → BAState (LangGraph pipeline node)

    Training:
        router.train(training_data)  — trains on 200 labelled questions
        router.save(model_dir)       — saves model + vectorizer with joblib
        router.load(model_dir)       — loads from disk

    The router trains automatically on first use if no saved model exists.
    """

    def __init__(self, model_dir: str = _MODEL_DIR) -> None:
        self.model_dir   = model_dir
        self._model      = None
        self._vectorizer = None
        self._is_trained = False

    # ── LangGraph pipeline node entry point ───────────────────────────────────

    def run(self, state) -> object:
        """
        LangGraph N04 node entry point. Fires FIRST at query time.

        Reads:  state.query
        Writes: state.query_type, state.routing_path,
                state.context_window_size

        Args:
            state: BAState object

        Returns:
            BAState with routing fields populated
        """
        query = getattr(state, "query", "") or ""

        if not query:
            logger.warning("N04 CART: empty query — defaulting to 'text'")
            state.query_type         = "text"
            state.routing_path       = "default_text"
            state.context_window_size = 3
            return state

        query_type, confidence, config = self.classify(query)

        state.query_type          = query_type
        state.routing_path        = (
            f"cart_{query_type}_conf{confidence:.2f}"
        )
        state.context_window_size = config["context_window"]

        logger.info(
            "N04 CART: query_type=%s | confidence=%.3f | "
            "sniper=%s | triguard=%s",
            query_type, confidence,
            config["sniper_first"],
            config["triguard_active"],
        )
        return state

    # ── Core classification method ────────────────────────────────────────────

    def classify(
        self, query: str
    ) -> Tuple[str, float, Dict]:
        """
        Classify a query into one of 5 query types.

        Ensures model is trained before classifying.
        Auto-trains on TRAINING_DATA if no saved model found.

        Args:
            query: Analyst question string

        Returns:
            Tuple of:
                query_type  : one of numerical/ratio/multi_doc/text/forensic
                confidence  : float 0.0–1.0 from predict_proba
                config      : routing config dict for this query type
        """
        self._ensure_trained()

        features   = self._vectorizer.transform([query])
        prediction = self._model.predict(features)[0]
        proba      = self._model.predict_proba(features)[0]
        confidence = float(max(proba))

        return prediction, confidence, ROUTING_CONFIG[prediction]

    def classify_batch(
        self, queries: List[str]
    ) -> List[Tuple[str, float, Dict]]:
        """Classify multiple queries efficiently."""
        self._ensure_trained()
        features    = self._vectorizer.transform(queries)
        predictions = self._model.predict(features)
        probas      = self._model.predict_proba(features)
        return [
            (pred, float(max(prob)), ROUTING_CONFIG[pred])
            for pred, prob in zip(predictions, probas)
        ]

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        training_data: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict:
        """
        Train the CART DecisionTree classifier.

        Args:
            training_data: List of (query, label) tuples.
                           Defaults to built-in 200-question set.

        Returns:
            Dict with train_accuracy and n_samples.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.tree import DecisionTreeClassifier

        data = training_data or TRAINING_DATA

        texts  = [q for q, _ in data]
        labels = [l for _, l in data]

        # TF-IDF vectoriser — captures financial terminology well
        self._vectorizer = TfidfVectorizer(
            ngram_range  = (1, 3),   # unigrams + bigrams + trigrams
            max_features = 2000,
            sublinear_tf = True,
        )
        features = self._vectorizer.fit_transform(texts)

        # Decision Tree — interpretable, fast, C5 seed=42
        self._model = DecisionTreeClassifier(
            max_depth    = 10,
            min_samples_leaf = 2,
            random_state = _SEED,
        )
        self._model.fit(features, labels)

        # Compute training accuracy
        preds         = self._model.predict(features)
        train_accuracy = sum(p == l for p, l in zip(preds, labels)) / len(labels)

        self._is_trained = True

        logger.info(
            "N04 CART trained: %d samples | train_accuracy=%.3f | "
            "classes=%s",
            len(data), train_accuracy, QUERY_CLASSES,
        )
        return {
            "train_accuracy": train_accuracy,
            "n_samples":      len(data),
            "n_classes":      len(QUERY_CLASSES),
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, model_dir: Optional[str] = None) -> str:
        """
        Save trained model and vectorizer to disk using joblib.

        Returns:
            Path to saved model file.
        """
        import joblib

        save_dir = model_dir or self.model_dir
        os.makedirs(save_dir, exist_ok=True)

        model_path      = os.path.join(save_dir, _MODEL_FILENAME)
        vectorizer_path = os.path.join(save_dir, _VECTORIZER_FILENAME)

        joblib.dump(self._model,      model_path)
        joblib.dump(self._vectorizer, vectorizer_path)

        logger.info("N04 CART saved: %s", model_path)
        return model_path

    def load(self, model_dir: Optional[str] = None) -> bool:
        """
        Load model and vectorizer from disk.

        Returns:
            True if loaded successfully, False otherwise.
        """
        import joblib

        load_dir        = model_dir or self.model_dir
        model_path      = os.path.join(load_dir, _MODEL_FILENAME)
        vectorizer_path = os.path.join(load_dir, _VECTORIZER_FILENAME)

        if not os.path.exists(model_path):
            return False
        if not os.path.exists(vectorizer_path):
            return False

        try:
            self._model      = joblib.load(model_path)
            self._vectorizer = joblib.load(vectorizer_path)
            self._is_trained = True
            logger.info("N04 CART loaded from: %s", model_path)
            return True
        except Exception as exc:
            logger.warning("N04 CART load failed: %s", exc)
            return False

    def is_trained(self) -> bool:
        return self._is_trained

    # ── Private helpers ───────────────────────────────────────────────────────

    def _ensure_trained(self) -> None:
        """Auto-train if model not yet trained or loaded."""
        if not self._is_trained:
            loaded = self.load()
            if not loaded:
                logger.info("N04 CART: no saved model — training now")
                self.train()


# ── Convenience wrapper for LangGraph N04 node ───────────────────────────────

def run_cart_router(state, model_dir: str = _MODEL_DIR) -> object:
    """
    Convenience wrapper used by the LangGraph pipeline node N04.

    Args:
        state     : BAState object
        model_dir : Directory containing saved model files

    Returns:
        BAState with query_type and routing_path populated
    """
    router = CARTRouter(model_dir=model_dir)
    return router.run(state)