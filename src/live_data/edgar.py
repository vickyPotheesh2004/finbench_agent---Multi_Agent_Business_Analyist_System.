"""
src/live_data/edgar.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 · Rev 1.0

Phase 7A — SEC EDGAR Live Filing Ingest

Downloads 10-K, 10-Q, 8-K filings from SEC EDGAR.
Saves locally then triggers N01-N03 ingestion pipeline.

C2 NOTE: Network calls only happen when EDGAR_ENABLED=True.
         Inference (N04-N19) remains 100% local always.
         Set EDGAR_ENABLED=False for air-gapped environments.

Constraints:
    C1  $0 cost — SEC EDGAR is free
    C2  Network ONLY for download — inference stays local
    C5  seed=42
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

EDGAR_BASE_URL    = "https://efts.sec.gov/LATEST/search-index"
EDGAR_SEARCH_URL  = "https://efts.sec.gov/LATEST/search-index"
EDGAR_ENABLED     = os.getenv("EDGAR_ENABLED", "false").lower() == "true"

SUPPORTED_FORMS   = {"10-K", "10-Q", "8-K", "DEF 14A", "S-1"}
DEFAULT_CACHE_DIR = "data/edgar_cache"
RATE_LIMIT_SEC    = 0.5      # SEC fair-use: max 10 req/sec, use 2/sec
SEED              = 42

# Known ticker → CIK mappings (cached to avoid network lookup in tests)
KNOWN_CIKS = {
    "AAPL":  "0000320193",
    "MSFT":  "0000789019",
    "GOOGL": "0001652044",
    "AMZN":  "0001018724",
    "TSLA":  "0001318605",
    "META":  "0001326801",
    "NVDA":  "0001045810",
    "JPM":   "0000019617",
    "GS":    "0000886982",
    "BAC":   "0000070858",
}


class EDGARFiling:
    """Represents a single SEC EDGAR filing."""

    __slots__ = (
        "ticker", "cik", "form_type", "fiscal_year",
        "filed_date", "accession_number", "document_url",
        "local_path", "cached",
    )

    def __init__(
        self,
        ticker:           str,
        cik:              str,
        form_type:        str,
        filed_date:       str,
        accession_number: str,
        document_url:     str   = "",
        fiscal_year:      str   = "",
        local_path:       str   = "",
    ) -> None:
        self.ticker           = ticker.upper()
        self.cik              = cik
        self.form_type        = form_type
        self.filed_date       = filed_date
        self.accession_number = accession_number
        self.document_url     = document_url
        self.fiscal_year      = fiscal_year or self._infer_fy(filed_date)
        self.local_path       = local_path
        self.cached           = bool(local_path and Path(local_path).exists())

    def _infer_fy(self, filed_date: str) -> str:
        """Infer fiscal year from filing date."""
        try:
            year = int(filed_date[:4])
            return f"FY{year}"
        except (ValueError, IndexError):
            return "UNKNOWN"

    def to_dict(self) -> Dict:
        return {
            "ticker":           self.ticker,
            "cik":              self.cik,
            "form_type":        self.form_type,
            "fiscal_year":      self.fiscal_year,
            "filed_date":       self.filed_date,
            "accession_number": self.accession_number,
            "document_url":     self.document_url,
            "local_path":       self.local_path,
            "cached":           self.cached,
        }

    def __repr__(self) -> str:
        return (
            f"EDGARFiling({self.ticker} {self.form_type} "
            f"{self.fiscal_year} filed={self.filed_date})"
        )


class EDGARClient:
    """
    Phase 7A — SEC EDGAR filing downloader.

    Features:
        - Ticker → CIK resolution (local cache first)
        - Filing search by form type + date range
        - Local caching — never re-downloads same filing
        - Rate limiting — respects SEC fair-use policy
        - C2 gate — network disabled unless EDGAR_ENABLED=True
    """

    def __init__(
        self,
        cache_dir:   str  = DEFAULT_CACHE_DIR,
        enabled:     bool = EDGAR_ENABLED,
        rate_limit:  float = RATE_LIMIT_SEC,
    ) -> None:
        self.cache_dir  = cache_dir
        self.enabled    = enabled
        self.rate_limit = rate_limit
        self._last_req  = 0.0
        self._cik_cache: Dict[str, str] = dict(KNOWN_CIKS)

        os.makedirs(cache_dir, exist_ok=True)
        self._load_cik_cache()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_filings(
        self,
        ticker:     str,
        form_type:  str = "10-K",
        limit:      int = 5,
    ) -> List[EDGARFiling]:
        """
        Get recent filings for a ticker.

        If EDGAR_ENABLED=False: returns cached filings only.
        If EDGAR_ENABLED=True:  fetches from SEC EDGAR.

        Args:
            ticker    : Stock ticker (e.g. "AAPL")
            form_type : Filing type ("10-K", "10-Q", "8-K")
            limit     : Max filings to return

        Returns:
            List of EDGARFiling objects
        """
        if form_type not in SUPPORTED_FORMS:
            raise ValueError(
                f"Unsupported form type: {form_type}. "
                f"Supported: {SUPPORTED_FORMS}"
            )

        ticker = ticker.upper()

        # Always check cache first
        cached = self._load_from_cache(ticker, form_type, limit)
        if cached:
            logger.info(
                "[7A EDGAR] Cache hit: %d %s filings for %s",
                len(cached), form_type, ticker,
            )
            return cached

        if not self.enabled:
            logger.info(
                "[7A EDGAR] Network disabled (EDGAR_ENABLED=False) — "
                "no cached filings for %s %s", ticker, form_type,
            )
            return []

        # Network fetch
        return self._fetch_filings(ticker, form_type, limit)

    def get_cik(self, ticker: str) -> Optional[str]:
        """
        Resolve ticker to CIK number.
        Checks local cache first, then EDGAR lookup.
        """
        ticker = ticker.upper()
        if ticker in self._cik_cache:
            return self._cik_cache[ticker]

        if not self.enabled:
            return None

        return self._lookup_cik(ticker)

    def is_cached(self, ticker: str, form_type: str) -> bool:
        """Check if any filings are cached for this ticker/form."""
        cached = self._load_from_cache(ticker.upper(), form_type, 1)
        return len(cached) > 0

    def get_cache_path(self, ticker: str, form_type: str) -> str:
        """Return the cache file path for a ticker/form."""
        return os.path.join(
            self.cache_dir,
            f"{ticker.upper()}_{form_type.replace(' ', '_')}.json",
        )

    def save_to_cache(
        self, ticker: str, form_type: str, filings: List[EDGARFiling]
    ) -> None:
        """Manually save filings to cache."""
        path = self.get_cache_path(ticker, form_type)
        data = [f.to_dict() for f in filings]
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2)
        logger.debug(
            "[7A EDGAR] Cached %d %s filings for %s",
            len(filings), form_type, ticker,
        )

    def clear_cache(self, ticker: Optional[str] = None) -> int:
        """
        Clear local filing cache.
        If ticker given: clear only that ticker's cache.
        Otherwise: clear all cached filings.
        Returns number of files deleted.
        """
        deleted = 0
        for f in Path(self.cache_dir).glob("*.json"):
            if f.name == "cik_cache.json":
                continue
            if ticker is None or f.name.startswith(ticker.upper()):
                f.unlink()
                deleted += 1
        return deleted

    # ── Private ───────────────────────────────────────────────────────────────

    def _fetch_filings(
        self, ticker: str, form_type: str, limit: int
    ) -> List[EDGARFiling]:
        """Fetch filings from SEC EDGAR (network required)."""
        import urllib.request
        import urllib.parse

        cik = self.get_cik(ticker)
        if not cik:
            logger.warning("[7A EDGAR] CIK not found for %s", ticker)
            return []

        self._rate_limit()

        params = urllib.parse.urlencode({
            "q":          f'"{form_type}"',
            "dateRange":  "custom",
            "entity":     ticker,
            "forms":      form_type,
            "_source":    "hits.hits._source",
            "hits.hits.total.value": 1,
        })

        url = f"https://efts.sec.gov/LATEST/search-index?{params}"

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "FinBenchAgent/1.0 analyst@finbench.ai"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw  = resp.read().decode("utf-8")
                data = json.loads(raw)

            filings = self._parse_edgar_response(
                data, ticker, cik, form_type
            )[:limit]

            if filings:
                self.save_to_cache(ticker, form_type, filings)

            return filings

        except Exception as exc:
            logger.error("[7A EDGAR] Fetch failed for %s: %s", ticker, exc)
            return []

    def _lookup_cik(self, ticker: str) -> Optional[str]:
        """Look up CIK from SEC EDGAR company search."""
        import urllib.request

        self._rate_limit()
        url = (
            f"https://www.sec.gov/cgi-bin/browse-edgar"
            f"?company={ticker}&CIK=&type=10-K&dateb=&owner=include"
            f"&count=1&search_text=&action=getcompany&output=atom"
        )
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "FinBenchAgent/1.0 analyst@finbench.ai"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                content = resp.read().decode("utf-8")
            # Simple extraction — CIK is in the atom feed
            import re
            match = re.search(r"CIK=(\d+)", content)
            if match:
                cik = match.group(1).zfill(10)
                self._cik_cache[ticker] = cik
                self._save_cik_cache()
                return cik
        except Exception as exc:
            logger.debug("[7A EDGAR] CIK lookup failed: %s", exc)
        return None

    @staticmethod
    def _parse_edgar_response(
        data: Dict, ticker: str, cik: str, form_type: str
    ) -> List[EDGARFiling]:
        """Parse SEC EDGAR search response into EDGARFiling objects."""
        filings = []
        hits    = data.get("hits", {}).get("hits", [])
        for hit in hits:
            src = hit.get("_source", {})
            filings.append(EDGARFiling(
                ticker           = ticker,
                cik              = cik,
                form_type        = form_type,
                filed_date       = src.get("file_date", ""),
                accession_number = src.get("accession_no", ""),
                document_url     = src.get("file_date", ""),
                fiscal_year      = "",
            ))
        return filings

    def _rate_limit(self) -> None:
        """Enforce SEC rate limit between requests."""
        elapsed = time.time() - self._last_req
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_req = time.time()

    def _load_from_cache(
        self, ticker: str, form_type: str, limit: int
    ) -> List[EDGARFiling]:
        """Load filings from local JSON cache."""
        path = self.get_cache_path(ticker, form_type)
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            filings = []
            for d in data[:limit]:
                filings.append(EDGARFiling(
                    ticker           = d.get("ticker",           ticker),
                    cik              = d.get("cik",              ""),
                    form_type        = d.get("form_type",        form_type),
                    filed_date       = d.get("filed_date",       ""),
                    accession_number = d.get("accession_number", ""),
                    document_url     = d.get("document_url",     ""),
                    fiscal_year      = d.get("fiscal_year",      ""),
                    local_path       = d.get("local_path",       ""),
                ))
            return filings
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("[7A EDGAR] Cache read failed: %s", exc)
            return []

    def _load_cik_cache(self) -> None:
        """Load CIK cache from disk."""
        path = os.path.join(self.cache_dir, "cik_cache.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fp:
                    self._cik_cache.update(json.load(fp))
            except Exception:
                pass

    def _save_cik_cache(self) -> None:
        """Persist CIK cache to disk."""
        path = os.path.join(self.cache_dir, "cik_cache.json")
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(self._cik_cache, fp, indent=2)