"""
src/live_data/yfinance_feed.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 · Rev 1.0

Phase 7B — Yahoo Finance Live Market Data

Fetches real-time and historical market data for use in
CFO/Quant Pod (N12) context enrichment:
    - Current price, P/E ratio, market cap
    - 52-week high/low
    - Historical price series for VaR calculation
    - Analyst estimates and EPS history

C2 NOTE: Network calls only when YFINANCE_ENABLED=True.
         All inference remains 100% local.

Constraints:
    C1  $0 cost — yfinance is free
    C2  Network ONLY for data fetch — inference stays local
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

YFINANCE_ENABLED  = os.getenv("YFINANCE_ENABLED", "false").lower() == "true"
DEFAULT_CACHE_DIR = "data/yfinance_cache"
CACHE_TTL_HOURS   = 4       # refresh market data every 4 hours
SEED              = 42

# Fields we extract from yfinance info dict
INFO_FIELDS = {
    "shortName", "longName", "sector", "industry",
    "marketCap", "trailingPE", "forwardPE",
    "trailingEps", "forwardEps",
    "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
    "currentPrice", "previousClose",
    "dividendYield", "payoutRatio",
    "returnOnEquity", "returnOnAssets",
    "debtToEquity", "currentRatio",
    "revenueGrowth", "earningsGrowth",
    "totalRevenue", "netIncomeToCommon",
    "freeCashflow", "operatingCashflow",
    "currency", "exchange",
}


class MarketSnapshot:
    """
    Point-in-time market data snapshot for a ticker.
    Passed to CFO/Quant Pod for context enrichment.
    """

    def __init__(
        self,
        ticker:         str,
        info:           Dict[str, Any],
        price_history:  Optional[List[Dict]] = None,
        fetched_at:     Optional[float]      = None,
    ) -> None:
        self.ticker       = ticker.upper()
        self.info         = info
        self.price_history = price_history or []
        self.fetched_at   = fetched_at or time.time()

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def current_price(self) -> Optional[float]:
        return self.info.get("currentPrice") or self.info.get("previousClose")

    @property
    def market_cap(self) -> Optional[float]:
        return self.info.get("marketCap")

    @property
    def market_cap_billions(self) -> Optional[float]:
        mc = self.market_cap
        return round(mc / 1e9, 2) if mc else None

    @property
    def pe_ratio(self) -> Optional[float]:
        return self.info.get("trailingPE") or self.info.get("forwardPE")

    @property
    def eps(self) -> Optional[float]:
        return self.info.get("trailingEps")

    @property
    def revenue_billions(self) -> Optional[float]:
        rev = self.info.get("totalRevenue")
        return round(rev / 1e9, 2) if rev else None

    @property
    def company_name(self) -> str:
        return (
            self.info.get("shortName")
            or self.info.get("longName")
            or self.ticker
        )

    @property
    def is_stale(self) -> bool:
        """Returns True if data is older than CACHE_TTL_HOURS."""
        age_hours = (time.time() - self.fetched_at) / 3600
        return age_hours > CACHE_TTL_HOURS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker":           self.ticker,
            "company_name":     self.company_name,
            "current_price":    self.current_price,
            "market_cap_B":     self.market_cap_billions,
            "pe_ratio":         self.pe_ratio,
            "eps":              self.eps,
            "revenue_B":        self.revenue_billions,
            "52w_high":         self.info.get("fiftyTwoWeekHigh"),
            "52w_low":          self.info.get("fiftyTwoWeekLow"),
            "sector":           self.info.get("sector"),
            "currency":         self.info.get("currency", "USD"),
            "fetched_at":       self.fetched_at,
            "price_history_len": len(self.price_history),
        }

    def __repr__(self) -> str:
        return (
            f"MarketSnapshot({self.ticker} "
            f"price={self.current_price} "
            f"mcap={self.market_cap_billions}B)"
        )


class YFinanceFeed:
    """
    Phase 7B — Yahoo Finance market data feed.

    Features:
        - Fetch current price, ratios, market cap
        - Historical price series (1Y default) for VaR
        - Local JSON cache with TTL
        - C2 gate — network disabled unless YFINANCE_ENABLED=True
    """

    def __init__(
        self,
        cache_dir: str  = DEFAULT_CACHE_DIR,
        enabled:   bool = YFINANCE_ENABLED,
        cache_ttl: int  = CACHE_TTL_HOURS,
    ) -> None:
        self.cache_dir = cache_dir
        self.enabled   = enabled
        self.cache_ttl = cache_ttl
        os.makedirs(cache_dir, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_snapshot(
        self,
        ticker:       str,
        history_days: int = 365,
    ) -> Optional[MarketSnapshot]:
        """
        Get market snapshot for a ticker.

        Checks cache first. If enabled and cache stale/missing,
        fetches fresh data from Yahoo Finance.

        Args:
            ticker       : Stock ticker (e.g. "AAPL")
            history_days : Days of price history for VaR

        Returns:
            MarketSnapshot or None if unavailable
        """
        ticker = ticker.upper()

        # Cache first
        cached = self._load_cache(ticker)
        if cached and not cached.is_stale:
            logger.info("[7B YFinance] Cache hit: %s", ticker)
            return cached

        if not self.enabled:
            logger.info(
                "[7B YFinance] Network disabled — no data for %s", ticker
            )
            return cached  # return stale cache if available, else None

        return self._fetch(ticker, history_days)

    def get_price_history(
        self, ticker: str, days: int = 365
    ) -> List[Dict[str, Any]]:
        """
        Get historical daily closing prices for VaR calculation.
        Returns list of {"date": str, "close": float} dicts.
        """
        snap = self.get_snapshot(ticker, history_days=days)
        if snap:
            return snap.price_history
        return []

    def get_pe_ratio(self, ticker: str) -> Optional[float]:
        """Quick accessor for P/E ratio."""
        snap = self.get_snapshot(ticker)
        return snap.pe_ratio if snap else None

    def get_market_cap(self, ticker: str) -> Optional[float]:
        """Quick accessor for market cap in dollars."""
        snap = self.get_snapshot(ticker)
        return snap.market_cap if snap else None

    def is_cached(self, ticker: str) -> bool:
        """Check if a ticker has any cached data."""
        return self._load_cache(ticker.upper()) is not None

    def save_snapshot(self, snapshot: MarketSnapshot) -> None:
        """Manually save a snapshot to cache (used in tests)."""
        path = self._cache_path(snapshot.ticker)
        data = {
            "ticker":        snapshot.ticker,
            "info":          snapshot.info,
            "price_history": snapshot.price_history,
            "fetched_at":    snapshot.fetched_at,
        }
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(data, fp)
        logger.debug("[7B YFinance] Cached: %s", snapshot.ticker)

    def clear_cache(self, ticker: Optional[str] = None) -> int:
        """Clear cache. Returns files deleted."""
        deleted = 0
        for f in Path(self.cache_dir).glob("*.json"):
            if ticker is None or f.stem == ticker.upper():
                f.unlink()
                deleted += 1
        return deleted

    # ── Private ───────────────────────────────────────────────────────────────

    def _fetch(self, ticker: str, history_days: int) -> Optional[MarketSnapshot]:
        """Fetch data from Yahoo Finance (requires yfinance package)."""
        try:
            import yfinance as yf
        except ImportError:
            logger.error("[7B YFinance] yfinance not installed: pip install yfinance")
            return None

        try:
            t    = yf.Ticker(ticker)
            info = {k: t.info.get(k) for k in INFO_FIELDS if k in t.info}

            # Price history
            end   = datetime.now()
            start = end - timedelta(days=history_days)
            hist  = t.history(start=start.strftime("%Y-%m-%d"),
                               end=end.strftime("%Y-%m-%d"))

            price_history = []
            for date, row in hist.iterrows():
                price_history.append({
                    "date":   str(date)[:10],
                    "close":  round(float(row["Close"]), 4),
                    "volume": int(row.get("Volume", 0)),
                })

            snap = MarketSnapshot(
                ticker        = ticker,
                info          = info,
                price_history = price_history,
            )
            self.save_snapshot(snap)
            logger.info(
                "[7B YFinance] Fetched %s: price=%.2f mcap=%.1fB",
                ticker,
                snap.current_price or 0,
                snap.market_cap_billions or 0,
            )
            return snap

        except Exception as exc:
            logger.error("[7B YFinance] Fetch failed for %s: %s", ticker, exc)
            return None

    def _load_cache(self, ticker: str) -> Optional[MarketSnapshot]:
        """Load snapshot from local JSON cache."""
        path = self._cache_path(ticker)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            return MarketSnapshot(
                ticker        = data["ticker"],
                info          = data.get("info",          {}),
                price_history = data.get("price_history", []),
                fetched_at    = data.get("fetched_at",    0.0),
            )
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("[7B YFinance] Cache read failed: %s", exc)
            return None

    def _cache_path(self, ticker: str) -> str:
        return os.path.join(self.cache_dir, f"{ticker.upper()}.json")