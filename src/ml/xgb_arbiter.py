"""
src/ml/xgb_arbiter.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 Rev1.0 FINAL

N17 — XGBoost Arbiter
ML ranking of 3 pod candidates on 8 objective features.

GATE M6 HARD STOP:
  XGB Arbiter only activates after >=300 quality DPO pairs.
  Before Gate M6: runs in STUB mode — passes final_answer_pre_xgb through.
  After Gate M6: re-ranks candidates, may override PIV Mediator choice.

Why XGBoost not LLM self-evaluation:
  LLMs favour confident-sounding wrong answers over correctly-cited
  right answers. XGBoost uses objective features — citations present,
  numbers match context, length appropriate — not prose style.

8 Features per candidate:
  F1 numerical_match    — primary number matches context (0/1)
  F2 citation_present   — at least 1 citation in answer (0/1)
  F3 section_relevance  — cited section matches query type (0-1)
  F4 length_appropriate — answer length in expected range (0-1)
  F5 unit_consistent    — units stated explicitly (0/1)
  F6 sign_correct       — negative values use correct sign (0/1)
  F7 confidence_score   — pod confidence score (0-1)
  F8 retry_penalty      — penalty for high retry count (0-1, lower=better)

Writes to BAState:
  xgb_ranked_answer — answer selected by XGB (or pass-through in stub)
  xgb_score         — XGB confidence score for selected answer
  final_answer      — same as xgb_ranked_answer (feeds N18+N19)
"""

import sys
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from src.state.ba_state          import BAState, PIVStatus
from src.utils.seed_manager      import SeedManager
from src.utils.resource_governor import ResourceGovernor

SeedManager.set_all()

# ── Config ────────────────────────────────────────────────────────────────────
GATE_M6_MIN_PAIRS   = 300     # minimum DPO pairs before XGB activates
MODEL_PATH          = ROOT / "models" / "xgb_arbiter.pkl"
FEATURE_NAMES = [
    "numerical_match",
    "citation_present",
    "section_relevance",
    "length_appropriate",
    "unit_consistent",
    "sign_correct",
    "confidence_score",
    "retry_penalty",
]

# ── Answer length thresholds ──────────────────────────────────────────────────
MIN_ANSWER_LEN = 20
MAX_ANSWER_LEN = 800


class XGBArbiter:
    """
    N17 — XGBoost Arbiter.

    Stub mode (Gate M6 not passed):
      - Passes final_answer_pre_xgb directly to final_answer
      - xgb_score = confidence_score from N15
      - No model loaded or trained

    Active mode (Gate M6 passed, model available):
      - Extracts 8 features from each candidate
      - XGBClassifier ranks candidates
      - Best-ranked candidate becomes final_answer
    """

    def __init__(
        self,
        model_path:    Optional[Path] = None,
        dpo_pair_count: int           = 0,
    ):
        SeedManager.set_all()
        self.model_path     = model_path or MODEL_PATH
        self.dpo_pair_count = dpo_pair_count
        self.model          = None
        self.gate_m6_passed = dpo_pair_count >= GATE_M6_MIN_PAIRS

        # Try to load existing model
        if self.gate_m6_passed:
            self._try_load_model()

    def run(self, state: BAState) -> BAState:
        """
        Main entry point — N17 node.

        Reads:  state.final_answer_pre_xgb, state.confidence_score,
                state.analyst_output, state.quant_result,
                state.auditor_output + their confidences
        Writes: state.xgb_ranked_answer, state.xgb_score,
                state.final_answer
        """
        ResourceGovernor.check("N17 XGB Arbiter")

        if not state.final_answer_pre_xgb and not state.analyst_output:
            print("[N17] No answer to rank — passing through empty")
            state.xgb_ranked_answer = ""
            state.xgb_score         = 0.0
            state.final_answer      = ""
            return state

        # ── STUB MODE (Gate M6 not passed) ────────────────────────────────
        if not self.gate_m6_passed or self.model is None:
            return self._stub_mode(state)

        # ── ACTIVE MODE (Gate M6 passed) ──────────────────────────────────
        return self._active_mode(state)

    # ═══════════════════════════════════════════════════════════════════════
    # STUB MODE
    # ═══════════════════════════════════════════════════════════════════════

    def _stub_mode(self, state: BAState) -> BAState:
        """
        Stub mode — pass final_answer_pre_xgb through unchanged.
        XGB score = N15 confidence score.
        """
        answer = state.final_answer_pre_xgb or state.analyst_output or ""
        score  = state.confidence_score or state.analyst_confidence or 0.0

        state.xgb_ranked_answer = answer
        state.xgb_score         = round(score, 4)
        state.final_answer      = answer

        print(f"[N17] STUB mode — pass-through "
              f"(Gate M6 needs {GATE_M6_MIN_PAIRS} DPO pairs, "
              f"have {self.dpo_pair_count}) "
              f"score={score:.2f}")
        return state

    # ═══════════════════════════════════════════════════════════════════════
    # ACTIVE MODE
    # ═══════════════════════════════════════════════════════════════════════

    def _active_mode(self, state: BAState) -> BAState:
        """
        Active mode — extract features and rank candidates with XGBoost.
        """
        candidates = self._collect_candidates(state)

        if not candidates:
            return self._stub_mode(state)

        # Extract features
        feature_matrix = np.array([
            self._extract_features(c, state)
            for c in candidates
        ], dtype=np.float32)

        # XGB predict — returns probability of being the best answer
        try:
            scores = self.model.predict_proba(feature_matrix)[:, 1]
        except Exception as e:
            print(f"[N17] XGB predict failed: {e} — falling back to stub")
            return self._stub_mode(state)

        best_idx = int(np.argmax(scores))
        best     = candidates[best_idx]

        state.xgb_ranked_answer = best["answer"]
        state.xgb_score         = round(float(scores[best_idx]), 4)
        state.final_answer      = best["answer"]

        print(f"[N17] ACTIVE mode — "
              f"winner={best['pod']} "
              f"xgb_score={state.xgb_score:.2f}")
        return state

    # ═══════════════════════════════════════════════════════════════════════
    # FEATURE EXTRACTION
    # ═══════════════════════════════════════════════════════════════════════

    def _collect_candidates(
        self, state: BAState
    ) -> List[Dict[str, Any]]:
        """Collect all available candidates."""
        candidates = []

        if state.analyst_output:
            candidates.append({
                "pod":        "analyst",
                "answer":     state.analyst_output,
                "confidence": state.analyst_confidence,
                "retries":    state.analyst_attempt_count,
                "citations":  state.analyst_citations,
            })
        if state.quant_result:
            candidates.append({
                "pod":        "quant",
                "answer":     state.quant_result,
                "confidence": state.quant_confidence,
                "retries":    state.quant_attempt_count,
                "citations":  state.quant_citations,
            })
        if state.auditor_output:
            candidates.append({
                "pod":        "auditor",
                "answer":     state.auditor_output,
                "confidence": state.auditor_confidence,
                "retries":    state.auditor_attempt_count,
                "citations":  state.auditor_citations,
            })
        return candidates

    def _extract_features(
        self,
        candidate: Dict[str, Any],
        state:     BAState,
    ) -> List[float]:
        """
        Extract 8 objective features from a candidate answer.
        Returns list of 8 floats.
        """
        answer    = candidate.get("answer", "")
        conf      = float(candidate.get("confidence", 0.0))
        retries   = int(candidate.get("retries", 0))
        citations = candidate.get("citations", [])

        # F1: numerical_match — does answer number appear in context?
        f1 = self._check_numerical_match(answer, state)

        # F2: citation_present — at least 1 citation
        f2 = 1.0 if citations and len(citations) > 0 else 0.0

        # F3: section_relevance — cited section matches query type
        f3 = self._check_section_relevance(citations, state.query_type)

        # F4: length_appropriate — answer length in expected range
        alen = len(answer)
        if MIN_ANSWER_LEN <= alen <= MAX_ANSWER_LEN:
            f4 = 1.0
        elif alen < MIN_ANSWER_LEN:
            f4 = alen / MIN_ANSWER_LEN
        else:
            f4 = MAX_ANSWER_LEN / alen

        # F5: unit_consistent — units stated explicitly
        f5 = 1.0 if re.search(
            r'\b(million|billion|percent|%|M\b|B\b|\$)',
            answer, re.IGNORECASE
        ) else 0.0

        # F6: sign_correct — check for parenthetical negatives
        has_parens = bool(re.search(r'\(\s*[\d,]+\s*\)', answer))
        f6         = 0.5 if has_parens else 1.0   # penalise ambiguous sign

        # F7: confidence_score — pod confidence
        f7 = min(max(conf, 0.0), 1.0)

        # F8: retry_penalty — fewer retries = better
        f8 = max(0.0, 1.0 - (retries * 0.15))

        return [f1, f2, f3, f4, f5, f6, f7, f8]

    def _check_numerical_match(
        self, answer: str, state: BAState
    ) -> float:
        """Check if primary number in answer appears in retrieved context."""
        nums_answer = self._extract_numbers(answer)
        if not nums_answer:
            return 0.5   # neutral — no number to check

        context_text = " ".join(
            chunk.get("text", "") or chunk.get("content", "")
            for chunk in (state.retrieval_stage_2 or [])
        )
        nums_context = self._extract_numbers(context_text)

        if not nums_context:
            return 0.5

        primary = nums_answer[0]
        for ctx_num in nums_context:
            if ctx_num > 0:
                diff = abs(primary - ctx_num) / max(primary, ctx_num)
                if diff <= 0.05:   # within 5%
                    return 1.0

        return 0.0

    def _check_section_relevance(
        self,
        citations:  List[str],
        query_type: Any,
    ) -> float:
        """Check if cited sections are relevant to query type."""
        if not citations:
            return 0.0

        qt = str(query_type).lower() if query_type else ""
        financial_sections = {
            "income", "financial", "balance", "cash", "earnings",
            "revenue", "statement", "notes",
        }

        score = 0.0
        for cit in citations:
            cit_lower = cit.lower()
            if any(s in cit_lower for s in financial_sections):
                score += 1.0

        return min(score / len(citations), 1.0)

    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numerical values from text."""
        pattern = r'\$?([\d,]+(?:\.\d+)?)'
        matches = re.findall(pattern, text)
        numbers = []
        for m in matches:
            try:
                val = float(m.replace(",", ""))
                if val > 0:
                    numbers.append(val)
            except ValueError:
                pass
        return numbers[:5]

    # ═══════════════════════════════════════════════════════════════════════
    # MODEL TRAINING
    # ═══════════════════════════════════════════════════════════════════════

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """
        Train XGBClassifier on labelled feature matrix.
        Returns validation accuracy.
        Gate M6: must achieve >=2% improvement over PIV-only baseline.
        """
        try:
            import xgboost as xgb
            from sklearn.model_selection import train_test_split
        except ImportError:
            print("[N17] xgboost/sklearn not available")
            return 0.0

        SeedManager.set_all()

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = xgb.XGBClassifier(
            n_estimators  = 100,
            max_depth      = 4,
            learning_rate  = 0.1,
            random_state   = 42,     # C5
            verbosity      = 0,
            tree_method    = "hist",
            eval_metric    = "logloss",
        )
        self.model.fit(X_train, y_train)

        val_acc = float(np.mean(
            self.model.predict(X_val) == y_val
        ))

        # Save model
        self._save_model()
        print(f"[N17] Model trained — val_acc={val_acc:.3f}")
        return val_acc

    def _save_model(self) -> None:
        """Save trained model to disk."""
        try:
            import joblib
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, self.model_path)
            print(f"[N17] Model saved to {self.model_path}")
        except Exception as e:
            print(f"[N17] Model save failed: {e}")

    def _try_load_model(self) -> None:
        """Try to load existing model from disk."""
        try:
            import joblib
            if self.model_path.exists():
                self.model = joblib.load(self.model_path)
                print(f"[N17] Model loaded from {self.model_path}")
        except Exception as e:
            print(f"[N17] Model load failed: {e}")
            self.model = None

    def build_synthetic_training_data(
        self, n_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build synthetic training data for testing Gate M6 logic.
        In production, training data comes from RLEF DPO pairs.
        Returns (X, y) where y=1 means this candidate was chosen.
        """
        SeedManager.set_all()
        rng = np.random.RandomState(42)

        # Positive examples — high quality answers
        X_pos = rng.uniform(0.6, 1.0, size=(n_samples // 2, 8))
        y_pos = np.ones(n_samples // 2, dtype=int)

        # Negative examples — low quality answers
        X_neg = rng.uniform(0.0, 0.5, size=(n_samples // 2, 8))
        y_neg = np.zeros(n_samples // 2, dtype=int)

        X = np.vstack([X_pos, X_neg]).astype(np.float32)
        y = np.concatenate([y_pos, y_neg])

        # Shuffle
        idx = rng.permutation(len(X))
        return X[idx], y[idx]


# ═══════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK
# run: python src/ml/xgb_arbiter.py
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        from rich import print as rprint
    except ImportError:
        rprint = print

    rprint("\n[bold cyan]-- XGBArbiter (N17) sanity check --[/bold cyan]")

    # Test 1: Stub mode (Gate M6 not passed)
    arbiter = XGBArbiter(dpo_pair_count=0)
    rprint(f"[green]✓[/green] XGBArbiter instantiated "
           f"(gate_m6_passed={arbiter.gate_m6_passed})")

    state = BAState(
        session_id          = "sanity-n17",
        query               = "What was Apple net income FY2023?",
        company_name        = "Apple Inc",
        final_answer_pre_xgb= "Net income $96,995M [FS/42].",
        confidence_score    = 0.88,
        analyst_output      = "Net income $96,995M [FS/42].",
        analyst_confidence  = 0.92,
        analyst_piv_status  = PIVStatus.PASS,
        retrieval_stage_2   = [{
            "text":    "Net income $96,995M FY2023.",
            "section": "Financial Statements",
            "page":    "42",
        }],
    )
    state = arbiter.run(state)
    rprint(f"[green]✓[/green] Stub mode: "
           f"final_answer='{state.final_answer[:40]}...' "
           f"xgb_score={state.xgb_score:.2f}")
    assert state.final_answer      != ""
    assert state.xgb_ranked_answer != ""
    assert 0.0 <= state.xgb_score  <= 1.0
    assert state.seed              == 42

    # Test 2: Feature extraction
    candidate = {
        "pod":        "analyst",
        "answer":     "Net income was $96,995 million [FS/42].",
        "confidence": 0.92,
        "retries":    0,
        "citations":  ["Financial Statements / Page 42"],
    }
    features = arbiter._extract_features(candidate, state)
    rprint(f"[green]✓[/green] Features extracted: {len(features)} "
           f"= {[round(f,2) for f in features]}")
    assert len(features) == 8
    assert all(0.0 <= f <= 1.0 for f in features)

    # Test 3: Synthetic training data
    X, y = arbiter.build_synthetic_training_data(100)
    rprint(f"[green]✓[/green] Synthetic training data: "
           f"X={X.shape} y={y.shape}")
    assert X.shape == (100, 8)
    assert len(y)  == 100

    # Test 4: Train model + active mode
    val_acc = arbiter.train(X, y)
    rprint(f"[green]✓[/green] Model trained: val_acc={val_acc:.3f}")
    assert val_acc > 0.0

    # Test 5: Active mode after training
    arbiter.gate_m6_passed = True
    state2 = BAState(
        session_id          = "sanity-n17-active",
        query               = "What was Apple net income FY2023?",
        final_answer_pre_xgb= "Net income $96,995M [FS/42].",
        confidence_score    = 0.88,
        analyst_output      = "Net income $96,995M [FS/42].",
        analyst_confidence  = 0.92,
        analyst_piv_status  = PIVStatus.PASS,
        quant_result        = "Gross margin 44.1% [FS/42].",
        quant_confidence    = 0.85,
        quant_piv_status    = PIVStatus.PASS,
        retrieval_stage_2   = [{
            "text":    "Net income $96,995M. Revenue $383,285M.",
            "section": "Financial Statements",
            "page":    "42",
        }],
    )
    state2 = arbiter.run(state2)
    rprint(f"[green]✓[/green] Active mode: "
           f"final_answer='{state2.final_answer[:40]}' "
           f"xgb_score={state2.xgb_score:.2f}")
    assert state2.final_answer != ""

    # Test 6: Empty state
    state3 = BAState(session_id="sanity-n17-empty")
    state3 = arbiter.run(state3)
    assert state3.final_answer == ""
    rprint(f"[green]✓[/green] Empty state handled")

    rprint(f"\n[bold green]All checks passed. XGBArbiter N17 ready.[/bold green]\n")