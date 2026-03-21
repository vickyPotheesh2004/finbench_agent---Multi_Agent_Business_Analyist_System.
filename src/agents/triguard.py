"""
src/agents/triguard.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 Rev1.0 FINAL

N13 — TriGuard Forensics
3-layer anomaly detection pipeline. No LLM — pure statistical ML.
Runs in PARALLEL with N11, N12, N14.

Layer 1 — Benford Law Test (scipy.stats.chi2_contingency)
  Checks if first-digit distribution of financial numbers follows
  Benford's Law. Fraud and errors often produce non-Benford distributions.
  Output: chi2 statistic, p-value, flag if p < 0.05

Layer 2 — Isolation Forest (sklearn, 500-row hard cap)
  Unsupervised anomaly detection on extracted financial ratios.
  Flags statistical outliers vs the document's own internal consistency.
  Output: anomaly_score, outlier_indices

Layer 3 — Risk Score Aggregation
  Combines Layer 1 + Layer 2 signals into a 0-100 risk score.
  Low: 0-30 | Medium: 31-60 | High: 61-100

Writes to BAState:
  forensic_flags     — list of specific anomaly descriptions
  risk_score         — float 0.0-100.0
  anomaly_detected   — bool
  anomaly_severity   — low / medium / high
  benford_chi2       — chi2 statistic
  benford_p_value    — p-value (< 0.05 = suspicious)
"""

import sys
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.stats import chi2_contingency
from sklearn.ensemble import IsolationForest

from src.state.ba_state     import BAState
from src.utils.seed_manager import SeedManager
from src.utils.resource_governor import ResourceGovernor

SeedManager.set_all()

# ── Config ────────────────────────────────────────────────────────────────────
ISOLATION_FOREST_CAP    = 500     # hard cap on rows — C4 protection
ISOLATION_CONTAMINATION = 0.1     # expect ~10% anomalies
BENFORD_P_THRESHOLD     = 0.05    # p < 0.05 → suspicious
RISK_SCORE_CAP          = 100.0

# ── Benford's Law expected first-digit probabilities ──────────────────────────
BENFORD_EXPECTED = np.array([
    np.log10(1 + 1/d) for d in range(1, 10)
])  # digits 1-9

# ── Risk score weights ────────────────────────────────────────────────────────
WEIGHT_BENFORD   = 0.40
WEIGHT_ISOLATION = 0.40
WEIGHT_RATIO     = 0.20

# ── Severity thresholds ───────────────────────────────────────────────────────
SEVERITY_HIGH   = 60.0
SEVERITY_MEDIUM = 30.0


class TriGuard:
    """
    N13 — TriGuard Forensics.

    3-layer forensic anomaly detection:
      Layer 1: Benford Law digit frequency test
      Layer 2: Isolation Forest outlier detection
      Layer 3: Risk score aggregation → severity

    No LLM calls. Pure statistical ML.
    Runs in parallel with N11, N12, N14.
    seed=42 enforced on all sklearn operations (C5).
    """

    def __init__(self):
        SeedManager.set_all()

    def run(self, state: BAState) -> BAState:
        """
        Main entry point — N13 node.

        Reads:  state.raw_text, state.table_cells,
                state.retrieval_stage_2
        Writes: state.forensic_flags, state.risk_score,
                state.anomaly_detected, state.anomaly_severity,
                state.benford_chi2, state.benford_p_value
        """
        ResourceGovernor.check("N13 TriGuard")

        # Extract numbers from all available sources
        numbers = self._extract_all_numbers(state)

        if len(numbers) < 3:
            print(f"[N13] Insufficient data ({len(numbers)} numbers) "
                  f"— minimal forensic analysis")
            state.forensic_flags   = ["INSUFFICIENT_DATA: fewer than 3 numbers extracted"]
            state.risk_score       = 0.0
            state.anomaly_detected = False
            state.anomaly_severity = "low"
            state.benford_chi2     = 0.0
            state.benford_p_value  = 1.0
            return state

        flags:      List[str] = []
        risk_parts: List[float] = []

        # ── LAYER 1: Benford Law ───────────────────────────────────────────
        benford_score, chi2_stat, p_value, benford_flags = \
            self._benford_test(numbers)
        flags.extend(benford_flags)
        risk_parts.append(benford_score * WEIGHT_BENFORD)
        state.benford_chi2   = round(chi2_stat, 4)
        state.benford_p_value = round(p_value,  4)

        # ── LAYER 2: Isolation Forest ──────────────────────────────────────
        iso_score, iso_flags = self._isolation_forest_test(numbers)
        flags.extend(iso_flags)
        risk_parts.append(iso_score * WEIGHT_ISOLATION)

        # ── LAYER 3: Ratio consistency check ──────────────────────────────
        ratio_score, ratio_flags = self._ratio_consistency_check(
            state.table_cells, numbers
        )
        flags.extend(ratio_flags)
        risk_parts.append(ratio_score * WEIGHT_RATIO)

        # ── Aggregate risk score ───────────────────────────────────────────
        risk_score = min(sum(risk_parts), RISK_SCORE_CAP)
        severity   = self._classify_severity(risk_score)

        # Write to BAState
        state.forensic_flags   = flags
        state.risk_score       = round(risk_score, 2)
        state.anomaly_detected = risk_score >= SEVERITY_MEDIUM
        state.anomaly_severity = severity

        print(f"[N13] Complete — "
              f"risk_score={state.risk_score:.1f} "
              f"severity={severity} "
              f"flags={len(flags)} "
              f"benford_p={p_value:.4f}")

        return state

    # ═══════════════════════════════════════════════════════════════════════
    # LAYER 1 — BENFORD LAW TEST
    # ═══════════════════════════════════════════════════════════════════════

    def _benford_test(
        self, numbers: List[float]
    ) -> Tuple[float, float, float, List[str]]:
        """
        Benford Law first-digit frequency test.
        Returns: (risk_score 0-100, chi2_stat, p_value, flags)

        Benford's Law: In naturally occurring datasets, the first digit
        follows a logarithmic distribution. Digit 1 appears ~30% of the time,
        digit 9 appears ~4.6% of the time.
        """
        flags = []

        # Extract first digits
        first_digits = self._extract_first_digits(numbers)
        if len(first_digits) < 5:
            return 0.0, 0.0, 1.0, []

        # Count observed frequencies
        observed = np.zeros(9)
        for d in first_digits:
            if 1 <= d <= 9:
                observed[d - 1] += 1

        if observed.sum() == 0:
            return 0.0, 0.0, 1.0, []

        # Expected counts based on Benford's Law
        n        = observed.sum()
        expected = BENFORD_EXPECTED * n

        # Chi-square test
        # Combine cells with low expected count to avoid chi2 issues
        obs_safe = np.maximum(observed, 0.5)
        exp_safe = np.maximum(expected, 0.5)

        try:
            chi2_stat, p_value, _, _ = chi2_contingency(
                np.array([obs_safe, exp_safe])
            )
        except Exception:
            chi2_stat, p_value = 0.0, 1.0

        # Risk scoring
        if p_value < 0.01:
            risk_score = 80.0
            flags.append(
                f"BENFORD_VIOLATION_HIGH: p={p_value:.4f} chi2={chi2_stat:.2f} "
                f"— digit distribution highly inconsistent with Benford Law"
            )
        elif p_value < BENFORD_P_THRESHOLD:
            risk_score = 50.0
            flags.append(
                f"BENFORD_VIOLATION: p={p_value:.4f} chi2={chi2_stat:.2f} "
                f"— digit distribution inconsistent with Benford Law"
            )
        elif p_value < 0.10:
            risk_score = 20.0
            flags.append(
                f"BENFORD_MARGINAL: p={p_value:.4f} — minor deviation "
                f"from expected digit distribution"
            )
        else:
            risk_score = 0.0   # No Benford anomaly

        return risk_score, chi2_stat, p_value, flags

    def run_benford_test(
        self, numbers: List[float]
    ) -> Tuple[float, float, List[str]]:
        """
        Public interface for Benford test.
        Returns (chi2_stat, p_value, flags).
        """
        _, chi2_stat, p_value, flags = self._benford_test(numbers)
        return chi2_stat, p_value, flags

    # ═══════════════════════════════════════════════════════════════════════
    # LAYER 2 — ISOLATION FOREST
    # ═══════════════════════════════════════════════════════════════════════

    def _isolation_forest_test(
        self, numbers: List[float]
    ) -> Tuple[float, List[str]]:
        """
        Isolation Forest outlier detection.
        Flags values that are statistical outliers vs the dataset.
        500-row hard cap enforced.
        Returns: (risk_score 0-100, flags)
        """
        flags = []

        if len(numbers) < 4:
            return 0.0, []

        # Hard cap — C4 RAM protection
        capped = numbers[:ISOLATION_FOREST_CAP]
        arr    = np.array(capped).reshape(-1, 1)

        try:
            SeedManager.set_all()
            iso = IsolationForest(
                contamination = ISOLATION_CONTAMINATION,
                random_state  = 42,   # C5
                n_estimators  = 100,
            )
            predictions = iso.fit_predict(arr)
            scores      = iso.score_samples(arr)

            # Count anomalies (-1 = anomaly, 1 = normal)
            n_anomalies    = int(np.sum(predictions == -1))
            anomaly_pct    = n_anomalies / len(capped)
            anomaly_values = [
                capped[i] for i, p in enumerate(predictions) if p == -1
            ]

            if anomaly_pct > 0.20:
                risk_score = 70.0
                flags.append(
                    f"ISOLATION_FOREST_HIGH: {n_anomalies}/{len(capped)} "
                    f"({anomaly_pct:.1%}) values flagged as outliers"
                )
            elif anomaly_pct > 0.10:
                risk_score = 40.0
                flags.append(
                    f"ISOLATION_FOREST_MEDIUM: {n_anomalies}/{len(capped)} "
                    f"({anomaly_pct:.1%}) values flagged as outliers"
                )
            else:
                risk_score = anomaly_pct * 200   # proportional, max ~20

            if anomaly_values and risk_score >= 40.0:
                top_anomalies = sorted(anomaly_values,
                                       key=lambda x: abs(x),
                                       reverse=True)[:3]
                flags.append(
                    f"OUTLIER_VALUES: {[round(v, 2) for v in top_anomalies]}"
                )

        except Exception as e:
            print(f"[N13] Isolation Forest error: {e}")
            risk_score = 0.0

        return min(risk_score, 100.0), flags

    def run_isolation_forest(
        self, numbers: List[float]
    ) -> Tuple[float, List[str]]:
        """Public interface for Isolation Forest test."""
        return self._isolation_forest_test(numbers)

    # ═══════════════════════════════════════════════════════════════════════
    # LAYER 3 — RATIO CONSISTENCY CHECK
    # ═══════════════════════════════════════════════════════════════════════

    def _ratio_consistency_check(
        self,
        table_cells: List[Dict[str, Any]],
        numbers:     List[float],
    ) -> Tuple[float, List[str]]:
        """
        Basic ratio consistency check.
        Looks for round number bias and extreme values.
        Returns: (risk_score 0-100, flags)
        """
        flags      = []
        risk_score = 0.0

        if not numbers:
            return 0.0, []

        # Round number bias check
        round_count = sum(
            1 for n in numbers
            if n > 100 and n % 100 == 0
        )
        round_pct = round_count / len(numbers) if numbers else 0

        if round_pct > 0.50:
            risk_score += 30.0
            flags.append(
                f"ROUND_NUMBER_BIAS: {round_count}/{len(numbers)} "
                f"({round_pct:.1%}) values are suspiciously round"
            )

        # Extreme value check — values > 10x median
        if len(numbers) >= 3:
            median    = float(np.median(numbers))
            if median > 0:
                extremes  = [
                    n for n in numbers
                    if n > median * 10 or (n < median / 10 and n > 0)
                ]
                if extremes:
                    risk_score += 15.0
                    flags.append(
                        f"EXTREME_VALUES: {len(extremes)} values are "
                        f">10x or <0.1x the median ({median:.0f})"
                    )

        return min(risk_score, 100.0), flags

    # ═══════════════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════════════

    def _extract_all_numbers(self, state: BAState) -> List[float]:
        """Extract all numerical values from BAState data sources."""
        numbers = []

        # From table cells
        for cell in state.table_cells[:ISOLATION_FOREST_CAP]:
            val = cell.get("value", "") or cell.get("cell_value", "")
            n   = self._parse_number(str(val))
            if n is not None:
                numbers.append(n)

        # From retrieval chunks
        for chunk in (state.retrieval_stage_2 or state.retrieval_stage_1 or []):
            text = chunk.get("text") or chunk.get("content") or ""
            numbers.extend(self._extract_numbers_from_text(text))

        # From raw text (limited)
        if state.raw_text and len(numbers) < 10:
            numbers.extend(
                self._extract_numbers_from_text(state.raw_text[:2000])
            )

        # Deduplicate and cap
        seen  = set()
        dedup = []
        for n in numbers:
            if n not in seen:
                seen.add(n)
                dedup.append(n)

        return dedup[:ISOLATION_FOREST_CAP]

    def _extract_numbers_from_text(self, text: str) -> List[float]:
        """Extract numerical values from text string."""
        pattern = r'\$?([\d,]+(?:\.\d+)?)'
        matches = re.findall(pattern, text)
        numbers = []
        for m in matches:
            n = self._parse_number(m)
            if n is not None and n > 0:
                numbers.append(n)
        return numbers

    def _parse_number(self, s: str) -> Optional[float]:
        """Parse a string to float. Returns None if not parseable."""
        try:
            cleaned = re.sub(r'[,$%\s]', '', str(s))
            if cleaned and cleaned not in ('', '-', '.'):
                val = float(cleaned)
                if 0 < val < 1e15:   # reasonable financial range
                    return val
        except (ValueError, TypeError):
            pass
        return None

    def _extract_first_digits(self, numbers: List[float]) -> List[int]:
        """Extract first significant digit from each number."""
        digits = []
        for n in numbers:
            if n <= 0:
                continue
            s = str(abs(n)).replace('.', '').lstrip('0')
            if s:
                try:
                    digits.append(int(s[0]))
                except (ValueError, IndexError):
                    pass
        return digits

    def _classify_severity(self, risk_score: float) -> str:
        """Classify risk score into low/medium/high severity."""
        if risk_score >= SEVERITY_HIGH:
            return "high"
        elif risk_score >= SEVERITY_MEDIUM:
            return "medium"
        return "low"

    def compute_risk_score(
        self,
        numbers:     List[float],
        table_cells: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
        """
        Public convenience method — compute risk score directly.
        Used in tests and standalone analysis.
        """
        if len(numbers) < 3:
            return 0.0

        benford_score, _, _, _ = self._benford_test(numbers)
        iso_score, _           = self._isolation_forest_test(numbers)
        ratio_score, _         = self._ratio_consistency_check(
            table_cells or [], numbers
        )

        risk = (
            benford_score * WEIGHT_BENFORD +
            iso_score     * WEIGHT_ISOLATION +
            ratio_score   * WEIGHT_RATIO
        )
        return min(round(risk, 2), RISK_SCORE_CAP)


# ═══════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK
# run: python src/agents/triguard.py
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        from rich import print as rprint
    except ImportError:
        rprint = print

    rprint("\n[bold cyan]-- TriGuard (N13) sanity check --[/bold cyan]")

    tg = TriGuard()
    rprint("[green]✓[/green] TriGuard instantiated")

    # Test Benford Law — normal financial data
    normal_numbers = [
        96995, 383285, 169148, 57411, 99803,
        29998, 50672, 18575, 394328, 365817,
        12345, 23456, 34567, 45678, 56789,
    ]
    chi2, p_val, flags = tg.run_benford_test(normal_numbers)
    rprint(f"[green]✓[/green] Benford test: chi2={chi2:.2f} p={p_val:.4f} "
           f"flags={len(flags)}")
    assert isinstance(chi2,  float)
    assert 0.0 <= p_val <= 1.0

    # Test Isolation Forest
    iso_score, iso_flags = tg.run_isolation_forest(normal_numbers)
    rprint(f"[green]✓[/green] Isolation Forest: score={iso_score:.1f} "
           f"flags={len(iso_flags)}")
    assert 0.0 <= iso_score <= 100.0

    # Test risk score
    score = tg.compute_risk_score(normal_numbers)
    rprint(f"[green]✓[/green] Risk score: {score:.1f} / 100")
    assert 0.0 <= score <= 100.0

    # Test severity classification
    assert tg._classify_severity(70.0) == "high"
    assert tg._classify_severity(40.0) == "medium"
    assert tg._classify_severity(10.0) == "low"
    rprint(f"[green]✓[/green] Severity classification correct")

    # BAState integration — normal data
    state = BAState(
        session_id        = "sanity-n13",
        query             = "What was Apple net income FY2023?",
        company_name      = "Apple Inc",
        doc_type          = "10-K",
        fiscal_year       = "FY2023",
        retrieval_stage_2 = [{
            "text":    "Net income: $96,995 million. Revenue: $383,285 million. "
                       "Gross profit: $169,148 million. Operating income: $114,301 million.",
            "section": "Financial Statements",
            "page":    "42",
        }],
    )
    state = tg.run(state)
    rprint(f"[green]✓[/green] BAState: risk_score={state.risk_score} "
           f"severity={state.anomaly_severity} "
           f"detected={state.anomaly_detected} "
           f"flags={len(state.forensic_flags)}")

    assert 0.0 <= state.risk_score <= 100.0
    assert state.anomaly_severity in ["low", "medium", "high"]
    assert isinstance(state.forensic_flags, list)
    assert state.seed == 42

    # Test with suspicious data (round numbers)
    suspicious = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    score2 = tg.compute_risk_score(suspicious)
    rprint(f"[green]✓[/green] Suspicious data score: {score2:.1f}")

    # Test insufficient data
    state2 = BAState(session_id="sanity-n13-empty")
    state2 = tg.run(state2)
    assert state2.risk_score       == 0.0
    assert state2.anomaly_detected is False
    rprint(f"[green]✓[/green] Insufficient data handled correctly")

    rprint(f"\n[bold green]All checks passed. TriGuard N13 ready.[/bold green]\n")