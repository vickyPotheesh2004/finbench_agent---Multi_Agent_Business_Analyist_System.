"""
src/ml/xgb_arbiter.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 Rev 1.0 Node N17

XGB Arbiter - ranks candidate answers from N11/N12/N14 using
gradient boosting on quality features. Expected +3-5% on FinanceBench.

GATE M6 PROTECTION:
    XGB only activates when >=300 DPO pairs exist in data/rlef_training.db.
    Below threshold it returns state unchanged (no-op).

Features for each candidate:
    - Confidence score (from PIV loop)
    - Citation count
    - Answer length
    - Numeric density (fraction of tokens that are numbers)
    - Word overlap with retrieved context
    - PIV attempt count (fewer = better)
    - Low-confidence flag

Constraints:
    C1  $0 cost - local XGBoost
    C2  100% local - no network calls
    C5  seed=42
    C9  no _rlef_ in any output
"""

from __future__ import annotations

import logging
import os
import pickle
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Gate M6 threshold - XGB only trains/activates above this
GATE_M6_MIN_DPO_PAIRS = 300
SEED                  = 42
MODEL_PATH            = "models/xgb_arbiter.pkl"
RLEF_DB_PATH          = "data/rlef_training.db"
N_FEATURES            = 7


class XGBArbiter:
    """
    N17 XGB Arbiter - ranks candidate answers post-mediation.

    Usage:
        arb = XGBArbiter()
        if arb.is_ready():
            state = arb.run(state)
    """

    def __init__(
        self,
        model_path:    str = MODEL_PATH,
        db_path:       str = RLEF_DB_PATH,
        min_dpo_pairs: int = GATE_M6_MIN_DPO_PAIRS,
    ) -> None:
        self.model_path    = model_path
        self.db_path       = db_path
        self.min_dpo_pairs = min_dpo_pairs
        self._model        = None

    # Gate M6 check
    def is_ready(self) -> bool:
        """
        Gate M6 check - returns True only if:
            1. A trained XGB model exists on disk
            2. RLEF DB has >=300 DPO pairs (quality data for training)
        """
        if not os.path.exists(self.model_path):
            logger.debug("[N17] No trained model at %s", self.model_path)
            return False

        dpo_count = self._count_dpo_pairs()
        if dpo_count < self.min_dpo_pairs:
            logger.debug(
                "[N17] Gate M6 failed: %d/%d DPO pairs",
                dpo_count, self.min_dpo_pairs,
            )
            return False

        return True

    def gate_m6_status(self) -> Dict[str, Any]:
        """Return detailed Gate M6 status for diagnostics."""
        dpo_count    = self._count_dpo_pairs()
        model_exists = os.path.exists(self.model_path)
        return {
            "gate_name":        "M6",
            "dpo_pairs":        dpo_count,
            "dpo_required":     self.min_dpo_pairs,
            "dpo_met":          dpo_count >= self.min_dpo_pairs,
            "model_exists":     model_exists,
            "overall_passed":   model_exists and dpo_count >= self.min_dpo_pairs,
            "blocker":          self._get_blocker(dpo_count, model_exists),
        }

    # LangGraph entry
    def run(self, state) -> object:
        """
        N17 entry - ranks candidate answers if Gate M6 passed.
        Writes to state.xgb_ranked_answer and state.xgb_score.
        """
        if not self.is_ready():
            logger.info("[N17] Gate M6 not passed - skipping (no-op)")
            return state

        candidates = self._collect_candidates(state)
        if not candidates:
            logger.debug("[N17] No candidates to rank")
            return state

        # Extract features and predict
        features    = [self._extract_features(c, state) for c in candidates]
        scores      = self._predict(features)
        best_idx    = int(max(range(len(scores)), key=lambda i: scores[i]))
        best_answer = candidates[best_idx]

        state.xgb_ranked_answer = best_answer["text"]
        state.xgb_score         = float(scores[best_idx])

        # Override final_answer if xgb chose differently with higher confidence
        if (
            state.xgb_score > getattr(state, "confidence_score", 0.0)
            and best_answer["text"] != getattr(state, "final_answer", "")
        ):
            state.final_answer     = best_answer["text"]
            state.confidence_score = state.xgb_score

        logger.info(
            "[N17] XGB ranked: pod=%s score=%.3f",
            best_answer.get("pod", "?"),
            state.xgb_score,
        )
        return state

    # Feature extraction
    def _extract_features(
        self, candidate: Dict[str, Any], state: Any
    ) -> List[float]:
        """
        Extract 7 features for a candidate answer.
        Returns list of floats in fixed order (critical for model consistency).
        """
        text  = candidate.get("text", "") or ""
        conf  = float(candidate.get("confidence", 0.0))
        cites = candidate.get("citations", []) or []

        # 1. Confidence
        f1_confidence = conf

        # 2. Citation count
        f2_citation_count = min(len(cites), 10) / 10.0

        # 3. Answer length (normalised)
        f3_length = min(len(text), 2000) / 2000.0

        # 4. Numeric density
        tokens       = text.split()
        numeric_toks = sum(1 for t in tokens if self._is_numeric(t))
        f4_numeric_density = (numeric_toks / len(tokens)) if tokens else 0.0

        # 5. Context overlap (word overlap with retrieved chunks)
        context_text = self._get_context_text(state)
        f5_overlap   = self._word_overlap(text, context_text)

        # 6. PIV attempt count (fewer retries = higher quality)
        attempts             = int(candidate.get("attempt_count", 1))
        f6_attempt_penalty   = max(0.0, 1.0 - (attempts - 1) * 0.25)

        # 7. Low confidence flag
        f7_low_conf_flag = 0.0 if candidate.get("low_confidence", False) else 1.0

        return [
            f1_confidence,
            f2_citation_count,
            f3_length,
            f4_numeric_density,
            f5_overlap,
            f6_attempt_penalty,
            f7_low_conf_flag,
        ]

    @staticmethod
    def _is_numeric(token: str) -> bool:
        """Returns True if token looks numeric (incl $, %, commas)."""
        stripped = token.strip("$%,.()").replace(",", "")
        try:
            float(stripped)
            return True
        except (ValueError, AttributeError):
            return False

    @staticmethod
    def _word_overlap(answer: str, context: str) -> float:
        """Jaccard overlap of non-trivial words."""
        if not answer or not context:
            return 0.0
        ans_words = {
            w.lower()
            for w in re.findall(r"\w+", answer) if len(w) > 3
        }
        ctx_words = {
            w.lower()
            for w in re.findall(r"\w+", context) if len(w) > 3
        }
        if not ans_words:
            return 0.0
        return len(ans_words & ctx_words) / len(ans_words)

    @staticmethod
    def _get_context_text(state: Any) -> str:
        """Extract retrieved context text from state."""
        chunks = (
            getattr(state, "retrieval_stage_2", [])
            or getattr(state, "retrieval_stage_1", [])
            or []
        )
        return "\n".join(
            c.get("text", "") if isinstance(c, dict) else str(c)
            for c in chunks
        )

    @staticmethod
    def _collect_candidates(state: Any) -> List[Dict[str, Any]]:
        """Gather candidates from the three analysis pods."""
        candidates = []

        if getattr(state, "analyst_output", ""):
            candidates.append({
                "pod":            "analyst",
                "text":           state.analyst_output,
                "confidence":     getattr(state, "analyst_confidence", 0.0),
                "citations":      getattr(state, "analyst_citations",  []),
                "attempt_count":  getattr(state, "analyst_attempt_count", 1),
                "low_confidence": getattr(state, "analyst_low_conf", False),
            })

        if getattr(state, "quant_result", ""):
            candidates.append({
                "pod":            "quant",
                "text":           state.quant_result,
                "confidence":     getattr(state, "quant_confidence", 0.0),
                "citations":      getattr(state, "quant_citations",  []),
                "attempt_count":  getattr(state, "quant_attempt_count", 1),
                "low_confidence": False,
            })

        if getattr(state, "auditor_output", ""):
            candidates.append({
                "pod":            "auditor",
                "text":           state.auditor_output,
                "confidence":     getattr(state, "auditor_confidence", 0.0),
                "citations":      getattr(state, "auditor_citations",  []),
                "attempt_count":  getattr(state, "auditor_attempt_count", 1),
                "low_confidence": False,
            })

        return candidates

    # Model IO
    def _load_model(self):
        """Load trained XGB model from pickle."""
        if self._model is not None:
            return self._model
        try:
            with open(self.model_path, "rb") as f:
                self._model = pickle.load(f)
            logger.info("[N17] Model loaded: %s", self.model_path)
            return self._model
        except Exception as exc:
            logger.error("[N17] Failed to load model: %s", exc)
            return None

    def _predict(self, features_list: List[List[float]]) -> List[float]:
        """Run XGB prediction on feature vectors."""
        model = self._load_model()
        if model is None:
            # Fallback: return confidence as score
            return [feats[0] for feats in features_list]

        try:
            import numpy as np
            X          = np.array(features_list, dtype=float)
            raw_scores = model.predict_proba(X)[:, 1] if hasattr(
                model, "predict_proba"
            ) else model.predict(X)
            return [float(s) for s in raw_scores]
        except Exception as exc:
            logger.warning("[N17] Predict failed: %s - using fallback", exc)
            return [feats[0] for feats in features_list]

    # RLEF DB query
    def _count_dpo_pairs(self) -> int:
        """Count DPO pairs in RLEF SQLite DB."""
        if not os.path.exists(self.db_path):
            return 0
        try:
            conn = sqlite3.connect(self.db_path)
            cur  = conn.cursor()
            # Check which table exists
            cur.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name IN ('dpo_pairs', 'rlef_grades')"
            )
            tables = [r[0] for r in cur.fetchall()]

            if "dpo_pairs" in tables:
                cur.execute("SELECT COUNT(*) FROM dpo_pairs")
            elif "rlef_grades" in tables:
                cur.execute("SELECT COUNT(*) FROM rlef_grades")
            else:
                conn.close()
                return 0

            count = cur.fetchone()[0]
            conn.close()
            return int(count)
        except Exception as exc:
            logger.debug("[N17] DB count failed: %s", exc)
            return 0

    @staticmethod
    def _get_blocker(dpo_count: int, model_exists: bool) -> str:
        """Return human-readable blocker for Gate M6."""
        if not model_exists and dpo_count < GATE_M6_MIN_DPO_PAIRS:
            return (
                f"Need {GATE_M6_MIN_DPO_PAIRS - dpo_count} more DPO pairs "
                f"AND model training"
            )
        if not model_exists:
            return "DPO threshold met - run training script"
        if dpo_count < GATE_M6_MIN_DPO_PAIRS:
            return f"Need {GATE_M6_MIN_DPO_PAIRS - dpo_count} more DPO pairs"
        return "Ready"


def run_xgb_arbiter(state) -> object:
    """LangGraph N17 convenience wrapper."""
    return XGBArbiter().run(state)