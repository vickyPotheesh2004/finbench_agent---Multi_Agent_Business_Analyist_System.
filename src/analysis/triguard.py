"""
N13 TriGuard Forensics — Three-Dimensional Anomaly Detection
PDR-BAAAI-001 · Rev 1.0 · Node N13
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

SEED                    = 42
BENFORD_PVALUE_THRESH   = 0.05
ISOLATION_CONTAMINATION = 0.1
MAX_ROWS_ISOLATION      = 500
MIN_VALUES_BENFORD      = 10

BENFORD_EXPECTED = np.array([np.log10(1 + 1/d) for d in range(1, 10)])

RISK_WEIGHTS = {"benford": 0.40, "isolation": 0.35, "volatility": 0.25}


@dataclass
class BenfordResult:
    chi2_statistic: float
    p_value:        float
    is_anomaly:     bool
    observed_freq:  List[float]
    expected_freq:  List[float]
    n_values:       int
    flag_message:   str = ""


@dataclass
class IsolationResult:
    outlier_indices:  List[int]
    outlier_count:    int
    outlier_fraction: float
    anomaly_scores:   List[float]
    is_anomaly:       bool
    n_samples:        int
    flag_message:     str = ""


@dataclass
class TriGuardResult:
    risk_score:       float
    anomaly_detected: bool
    anomaly_severity: str
    forensic_flags:   List[str]
    benford:          Optional[BenfordResult]
    isolation:        Optional[IsolationResult]
    details:          Dict


def extract_first_digits(values: List[float]) -> List[int]:
    digits = []
    for v in values:
        v = abs(v)
        if v <= 0:
            continue
        s = f"{v:.10g}".lstrip("0").replace(".", "").lstrip("0")
        if s and s[0].isdigit():
            d = int(s[0])
            if 1 <= d <= 9:
                digits.append(d)
    return digits


def run_benford_test(values: List[float]) -> Optional[BenfordResult]:
    if len(values) < MIN_VALUES_BENFORD:
        return None

    digits = extract_first_digits(values)
    if len(digits) < MIN_VALUES_BENFORD:
        return None

    n = len(digits)

    observed_counts = np.zeros(9)
    for d in digits:
        if 1 <= d <= 9:
            observed_counts[d - 1] += 1

    # Key fix: scale expected proportions so sum(f_exp) == sum(f_obs) == n
    # Renormalise first to remove floating point drift, then scale to n
    normed   = BENFORD_EXPECTED / BENFORD_EXPECTED.sum()
    expected_counts = normed * observed_counts.sum()

    try:
        chi2, p_value = stats.chisquare(
            f_obs=observed_counts,
            f_exp=expected_counts,
        )
    except ValueError as exc:
        logger.warning("Benford chi2 failed: %s", exc)
        return None

    is_anomaly   = bool(p_value < BENFORD_PVALUE_THRESH)
    flag_message = ""
    if is_anomaly:
        flag_message = (
            f"Benford Law anomaly detected: chi2={chi2:.2f}, "
            f"p={p_value:.4f} < {BENFORD_PVALUE_THRESH}. "
            f"Digit distribution deviates from expected natural distribution."
        )

    return BenfordResult(
        chi2_statistic = float(chi2),
        p_value        = float(p_value),
        is_anomaly     = is_anomaly,
        observed_freq  = (observed_counts / n).tolist(),
        expected_freq  = normed.tolist(),
        n_values       = n,
        flag_message   = flag_message,
    )


def run_isolation_forest(
    values:        List[float],
    contamination: float = ISOLATION_CONTAMINATION,
    seed:          int   = SEED,
) -> Optional[IsolationResult]:
    if not values or len(values) < 3:
        return None

    from sklearn.ensemble import IsolationForest

    capped = values[:MAX_ROWS_ISOLATION]
    arr    = np.array(capped).reshape(-1, 1)

    model          = IsolationForest(contamination=contamination,
                                     random_state=seed, n_estimators=100)
    predictions    = model.fit_predict(arr)
    anomaly_scores = model.score_samples(arr).tolist()

    outlier_indices  = [i for i, p in enumerate(predictions) if p == -1]
    outlier_count    = len(outlier_indices)
    outlier_fraction = outlier_count / len(capped)
    is_anomaly       = outlier_count > 0

    flag_message = ""
    if is_anomaly:
        flag_message = (
            f"Isolation Forest: {outlier_count} outlier(s) detected "
            f"({outlier_fraction:.1%} of {len(capped)} values). "
            f"Outlier positions: {outlier_indices[:5]}"
        )

    return IsolationResult(
        outlier_indices  = outlier_indices,
        outlier_count    = outlier_count,
        outlier_fraction = outlier_fraction,
        anomaly_scores   = anomaly_scores,
        is_anomaly       = is_anomaly,
        n_samples        = len(capped),
        flag_message     = flag_message,
    )


def classify_severity(risk_score: float) -> str:
    if risk_score >= 70:
        return "high"
    elif risk_score >= 40:
        return "medium"
    return "low"


def compute_risk_score(
    benford_result:   Optional[BenfordResult],
    isolation_result: Optional[IsolationResult],
    volatility:       Optional[float] = None,
) -> float:
    score = 0.0

    if benford_result and benford_result.is_anomaly:
        score += min(benford_result.chi2_statistic / 50.0, 1.0) * 40

    if isolation_result and isolation_result.is_anomaly:
        score += min(isolation_result.outlier_fraction * 5.0, 1.0) * 35

    if volatility is not None and volatility > 0:
        score += min(volatility / 0.30, 1.0) * 25

    return round(min(score, 100.0), 2)


class TriGuard:
    """
    N13 TriGuard Forensics — Three-Dimensional Anomaly Detection.
    Runs parallel to analysis pods (N11, N12, N14).
    """

    def __init__(self, seed: int = SEED) -> None:
        self.seed = seed

    def run(self, state) -> object:
        raw_text    = getattr(state, "raw_text",    "") or ""
        table_cells = getattr(state, "table_cells", []) or []
        garch_res   = getattr(state, "garch_result", None)

        volatility = None
        if isinstance(garch_res, dict):
            volatility = garch_res.get("conditional_volatility")

        values = self._extract_values(table_cells, raw_text)
        result = self.analyze(values, volatility=volatility)

        state.forensic_flags   = result.forensic_flags
        state.risk_score       = result.risk_score
        state.anomaly_detected = result.anomaly_detected
        state.anomaly_severity = result.anomaly_severity

        logger.info(
            "N13 TriGuard: risk_score=%.1f | severity=%s | flags=%d | anomaly=%s",
            result.risk_score, result.anomaly_severity,
            len(result.forensic_flags), result.anomaly_detected,
        )
        return state

    def analyze(
        self,
        values:     List[float],
        volatility: Optional[float] = None,
    ) -> TriGuardResult:
        forensic_flags = []

        benford_result = run_benford_test(values)
        if benford_result and benford_result.is_anomaly:
            forensic_flags.append(benford_result.flag_message)

        isolation_result = run_isolation_forest(
            values=values, contamination=ISOLATION_CONTAMINATION, seed=self.seed
        )
        if isolation_result and isolation_result.is_anomaly:
            forensic_flags.append(isolation_result.flag_message)

        if volatility is not None and volatility > 0.30:
            forensic_flags.append(
                f"High conditional volatility detected: {volatility:.3f} "
                f"(threshold: 0.30). Elevated temporal risk."
            )

        risk_score       = compute_risk_score(benford_result, isolation_result, volatility)
        severity         = classify_severity(risk_score)
        anomaly_detected = risk_score > 0 or len(forensic_flags) > 0

        return TriGuardResult(
            risk_score       = risk_score,
            anomaly_detected = anomaly_detected,
            anomaly_severity = severity,
            forensic_flags   = forensic_flags,
            benford          = benford_result,
            isolation        = isolation_result,
            details          = {"n_values": len(values), "volatility": volatility},
        )

    @staticmethod
    def _extract_values(table_cells: List[Dict], raw_text: str) -> List[float]:
        values = []

        for cell in table_cells:
            raw_val = cell.get("value", "") or cell.get("numeric_value", "")
            if isinstance(raw_val, (int, float)):
                v = float(raw_val)
                if v != 0:
                    values.append(v)
            elif isinstance(raw_val, str):
                s        = raw_val.strip()
                negative = s.startswith("(") and s.endswith(")")
                s        = s.strip("()").replace(",", "").replace("$", "")
                try:
                    v = float(s)
                    values.append(-v if negative else v)
                except ValueError:
                    pass

        if len(values) < MIN_VALUES_BENFORD and raw_text:
            for m in re.compile(r'\b([0-9,]+(?:\.[0-9]+)?)\b').findall(raw_text):
                try:
                    v = float(m.replace(",", ""))
                    if v > 0:
                        values.append(v)
                except ValueError:
                    continue

        return values[:MAX_ROWS_ISOLATION]


def run_triguard(state) -> object:
    return TriGuard(seed=SEED).run(state)