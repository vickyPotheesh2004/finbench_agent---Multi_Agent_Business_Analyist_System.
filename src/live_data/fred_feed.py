"""
src/live_data/fred_feed.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 · Rev 1.0

Phase 7D — FRED Macro Data Feed

Federal Reserve Economic Data (FRED) — provides macroeconomic context
for the CRO risk agent and CFO/Quant pod:
    - Federal Funds Rate (FEDFUNDS)
    - CPI / Inflation (CPIAUCSL)
    - GDP Growth (GDP, GDPC1)
    - Unemployment Rate (UNRATE)
    - 10-Year Treasury Yield (DGS10)
    - VIX Volatility Index

C2 NOTE: Network only when FRED_ENABLED=True.
         Analysis runs 100% local.

Constraints:
    C1  $0 cost — fredapi is free (requires free API key)
    C2  Network ONLY for fetch — inference stays local
    C5  seed=42
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

FRED_ENABLED      = os.getenv("FRED_ENABLED", "false").lower() == "true"
FRED_API_KEY      = os.getenv("FRED_API_KEY", "")
DEFAULT_CACHE_DIR = "data/fred_cache"
CACHE_TTL_HOURS   = 24        # macro data updates daily at most
SEED              = 42

# Standard FRED series we track
FRED_SERIES = {
    "fed_funds_rate":    "FEDFUNDS",      # Effective Federal Funds Rate
    "cpi":               "CPIAUCSL",      # Consumer Price Index
    "gdp_real":          "GDPC1",         # Real GDP
    "gdp_nominal":       "GDP",           # Nominal GDP
    "unemployment":      "UNRATE",        # Unemployment Rate
    "treasury_10y":      "DGS10",         # 10-Year Treasury Yield
    "treasury_2y":       "DGS2",          # 2-Year Treasury Yield
    "vix":               "VIXCLS",        # VIX Volatility
    "dollar_index":      "DTWEXBGS",      # Trade-Weighted Dollar Index
    "oil_wti":           "DCOILWTICO",    # WTI Oil Price
    "m2_money_supply":   "M2SL",          # M2 Money Stock
    "housing_starts":    "HOUST",         # Housing Starts
    "industrial_prod":   "INDPRO",        # Industrial Production
    "retail_sales":      "RSAFS",         # Retail Sales
    "consumer_sentiment":"UMCSENT",       # UMich Consumer Sentiment
}


class MacroObservation:
    """Single macro data observation for a date."""

    __slots__ = ("series_id", "date", "value", "series_name")

    def __init__(
        self,
        series_id:   str,
        date:        str,
        value:       Optional[float],
        series_name: str = "",
    ) -> None:
        self.series_id   = series_id
        self.date        = date
        self.value       = value
        self.series_name = series_name or series_id

    def to_dict(self) -> Dict:
        return {
            "series_id":   self.series_id,
            "series_name": self.series_name,
            "date":        self.date,
            "value":       self.value,
        }

    def __repr__(self) -> str:
        return f"MacroObservation({self.series_id} {self.date}={self.value})"


class MacroSnapshot:
    """Collection of current macro indicators."""

    def __init__(
        self,
        observations: Dict[str, MacroObservation],
        fetched_at:   Optional[float] = None,
    ) -> None:
        self.observations = observations
        self.fetched_at   = fetched_at or time.time()

    @property
    def fed_funds_rate(self) -> Optional[float]:
        obs = self.observations.get("fed_funds_rate")
        return obs.value if obs else None

    @property
    def cpi(self) -> Optional[float]:
        obs = self.observations.get("cpi")
        return obs.value if obs else None

    @property
    def unemployment(self) -> Optional[float]:
        obs = self.observations.get("unemployment")
        return obs.value if obs else None

    @property
    def treasury_10y(self) -> Optional[float]:
        obs = self.observations.get("treasury_10y")
        return obs.value if obs else None

    @property
    def vix(self) -> Optional[float]:
        obs = self.observations.get("vix")
        return obs.value if obs else None

    @property
    def is_stale(self) -> bool:
        age_hours = (time.time() - self.fetched_at) / 3600
        return age_hours > CACHE_TTL_HOURS

    @property
    def inversion_signal(self) -> bool:
        """
        Yield curve inversion: 2Y > 10Y is a recession signal.
        Returns True if inverted.
        """
        t2  = self.observations.get("treasury_2y")
        t10 = self.observations.get("treasury_10y")
        if not t2 or not t10 or t2.value is None or t10.value is None:
            return False
        return t2.value > t10.value

    def to_dict(self) -> Dict:
        return {
            "fetched_at":       self.fetched_at,
            "fed_funds_rate":   self.fed_funds_rate,
            "cpi":              self.cpi,
            "unemployment":     self.unemployment,
            "treasury_10y":     self.treasury_10y,
            "vix":              self.vix,
            "inversion_signal": self.inversion_signal,
            "observations":     {
                k: v.to_dict() for k, v in self.observations.items()
            },
        }


class FREDFeed:
    """
    Phase 7D — FRED macro data feed.

    Features:
        - Fetch current macro indicators
        - Historical time series retrieval
        - Local JSON cache with TTL
        - Yield curve inversion detection
        - C2 gate — network disabled unless FRED_ENABLED=True
    """

    def __init__(
        self,
        api_key:   str  = FRED_API_KEY,
        cache_dir: str  = DEFAULT_CACHE_DIR,
        enabled:   bool = FRED_ENABLED,
    ) -> None:
        self.api_key   = api_key
        self.cache_dir = cache_dir
        self.enabled   = enabled
        os.makedirs(cache_dir, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_snapshot(self) -> Optional[MacroSnapshot]:
        """
        Get current macro snapshot.
        Checks cache first, then fetches from FRED if enabled.
        """
        cached = self._load_snapshot_cache()
        if cached and not cached.is_stale:
            logger.info("[7D FRED] Snapshot cache hit")
            return cached

        if not self.enabled:
            logger.info("[7D FRED] Network disabled — returning cached snapshot")
            return cached

        return self._fetch_snapshot()

    def get_series(
        self,
        series_key:       str,
        observation_start: Optional[str] = None,
        limit:            int = 100,
    ) -> List[MacroObservation]:
        """
        Get historical time series for a FRED series.
        series_key can be either a friendly name (from FRED_SERIES)
        or a raw FRED series ID.
        """
        series_id = FRED_SERIES.get(series_key, series_key)

        cached = self._load_series_cache(series_id)
        if cached and len(cached) > 0:
            return cached[:limit]

        if not self.enabled:
            return []

        return self._fetch_series(series_id, observation_start, limit)

    def get_current_value(self, series_key: str) -> Optional[float]:
        """Get single most recent value for a series."""
        obs = self.get_series(series_key, limit=1)
        if obs and obs[0].value is not None:
            return obs[0].value
        return None

    def save_snapshot(self, snapshot: MacroSnapshot) -> None:
        """Save snapshot to cache."""
        path = os.path.join(self.cache_dir, "snapshot.json")
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(snapshot.to_dict(), fp)

    def save_series(
        self, series_id: str, observations: List[MacroObservation]
    ) -> None:
        """Save series observations to cache."""
        path = os.path.join(self.cache_dir, f"{series_id}.json")
        data = [obs.to_dict() for obs in observations]
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(data, fp)

    def clear_cache(self) -> int:
        """Clear all cached data. Returns files deleted."""
        deleted = 0
        for f in Path(self.cache_dir).glob("*.json"):
            f.unlink()
            deleted += 1
        return deleted

    def is_cached(self, series_key: str) -> bool:
        """Check if a series has cached data."""
        series_id = FRED_SERIES.get(series_key, series_key)
        return os.path.exists(
            os.path.join(self.cache_dir, f"{series_id}.json")
        )

    # ── Private ───────────────────────────────────────────────────────────────

    def _fetch_snapshot(self) -> Optional[MacroSnapshot]:
        """Fetch current snapshot from FRED (requires fredapi)."""
        try:
            from fredapi import Fred
        except ImportError:
            logger.error("[7D FRED] fredapi not installed: pip install fredapi")
            return None

        if not self.api_key:
            logger.warning("[7D FRED] FRED_API_KEY not set — fetch skipped")
            return None

        try:
            fred = Fred(api_key=self.api_key)
            observations = {}
            for key, sid in FRED_SERIES.items():
                try:
                    series = fred.get_series_latest_release(sid)
                    if len(series) == 0:
                        continue
                    latest = series.iloc[-1]
                    observations[key] = MacroObservation(
                        series_id   = sid,
                        date        = str(series.index[-1])[:10],
                        value       = float(latest) if latest is not None else None,
                        series_name = key,
                    )
                except Exception as exc:
                    logger.debug("[7D FRED] %s fetch failed: %s", sid, exc)

            snap = MacroSnapshot(observations=observations)
            self.save_snapshot(snap)
            logger.info(
                "[7D FRED] Snapshot fetched: %d indicators", len(observations)
            )
            return snap

        except Exception as exc:
            logger.error("[7D FRED] Snapshot fetch failed: %s", exc)
            return None

    def _fetch_series(
        self,
        series_id:         str,
        observation_start: Optional[str],
        limit:             int,
    ) -> List[MacroObservation]:
        """Fetch time series from FRED."""
        try:
            from fredapi import Fred
        except ImportError:
            return []

        if not self.api_key:
            return []

        try:
            fred   = Fred(api_key=self.api_key)
            series = fred.get_series(
                series_id,
                observation_start=observation_start,
            )
            obs = []
            for date, value in series.items():
                obs.append(MacroObservation(
                    series_id = series_id,
                    date      = str(date)[:10],
                    value     = float(value) if value is not None else None,
                ))
            # Most recent first
            obs = sorted(obs, key=lambda o: o.date, reverse=True)[:limit]
            self.save_series(series_id, obs)
            return obs
        except Exception as exc:
            logger.error("[7D FRED] Series %s fetch failed: %s", series_id, exc)
            return []

    def _load_snapshot_cache(self) -> Optional[MacroSnapshot]:
        """Load snapshot from cache."""
        path = os.path.join(self.cache_dir, "snapshot.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            observations = {}
            for k, v in data.get("observations", {}).items():
                observations[k] = MacroObservation(
                    series_id   = v.get("series_id",   ""),
                    date        = v.get("date",        ""),
                    value       = v.get("value"),
                    series_name = v.get("series_name", ""),
                )
            return MacroSnapshot(
                observations = observations,
                fetched_at   = data.get("fetched_at", 0.0),
            )
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("[7D FRED] Snapshot cache read failed: %s", exc)
            return None

    def _load_series_cache(
        self, series_id: str
    ) -> List[MacroObservation]:
        """Load series from cache."""
        path = os.path.join(self.cache_dir, f"{series_id}.json")
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            return [
                MacroObservation(
                    series_id   = d.get("series_id",   series_id),
                    date        = d.get("date",        ""),
                    value       = d.get("value"),
                    series_name = d.get("series_name", ""),
                )
                for d in data
            ]
        except (json.JSONDecodeError, KeyError):
            return []