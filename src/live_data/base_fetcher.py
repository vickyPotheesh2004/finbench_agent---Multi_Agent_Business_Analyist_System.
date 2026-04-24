"""
src/live_data/base_fetcher.py
FinBench Multi-Agent Business Analyst AI
Phase 7A — Live Data Infrastructure

Base class for all API fetchers.
Every fetcher inherits from BaseAPIFetcher.

Features:
  - Automatic caching via CacheManager
  - Retry logic with exponential backoff
  - Timeout enforcement
  - DataShield freshness tagging
  - C2 compliant: fetch happens pre-inference only

Usage:
    class YFinanceFetcher(BaseAPIFetcher):
        API_NAME  = "yfinance"
        DATA_TYPE = "stock_price"

        def fetch_live(self, ticker: str) -> Dict:
            ...implement...

        def to_chunks(self, data: Dict) -> List[Dict]:
            ...convert to BAState chunks...
"""

import sys
import time
import hashlib
import datetime
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.live_data.cache_manager import CacheManager
from src.utils.seed_manager      import SeedManager

SeedManager.set_all()

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_TIMEOUT    = 10       # seconds per request
MAX_RETRIES        = 3
RETRY_BACKOFF      = [1, 2, 4]  # seconds between retries


class FetchResult:
    """Result from a fetcher call."""

    def __init__(
        self,
        data:        Any,
        source:      str,
        fetched_at:  str,
        freshness:   str,
        cache_hit:   bool  = False,
        error:       str   = "",
    ):
        self.data        = data
        self.source      = source
        self.fetched_at  = fetched_at
        self.freshness   = freshness
        self.cache_hit   = cache_hit
        self.error       = error
        self.success     = bool(data) and not error

    def __repr__(self):
        return (f"FetchResult(source={self.source} "
                f"freshness={self.freshness} "
                f"cache_hit={self.cache_hit} "
                f"success={self.success})")


class BaseAPIFetcher(ABC):
    """
    Base class for all live data API fetchers.

    Subclasses must implement:
        fetch_live(**kwargs) -> Dict
        to_chunks(data: Dict) -> List[Dict]

    Subclasses should set class attributes:
        API_NAME  = "api_name"
        DATA_TYPE = "stock_price" (matches CacheManager TTL keys)
    """

    API_NAME:  str = "base"
    DATA_TYPE: str = "default"

    def __init__(
        self,
        cache:   Optional[CacheManager] = None,
        timeout: int                    = DEFAULT_TIMEOUT,
    ):
        SeedManager.set_all()
        self.cache   = cache or CacheManager()
        self.timeout = timeout
        self.enabled = True   # can disable per-instance

    # ═══════════════════════════════════════════════════════════════════════
    # ABSTRACT INTERFACE
    # ═══════════════════════════════════════════════════════════════════════

    @abstractmethod
    def fetch_live(self, **kwargs) -> Dict[str, Any]:
        """
        Fetch data from live API.
        Must be implemented by every subclass.
        Should raise on network error — base class handles retries.
        """
        raise NotImplementedError

    @abstractmethod
    def to_chunks(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert fetched data into BAState context chunks.
        Each chunk: {text, section, page, source, freshness, ...}
        """
        raise NotImplementedError

    # ═══════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════════════════

    def fetch(self, **kwargs) -> FetchResult:
        """
        Main fetch entry point.
        1. Check cache first
        2. If miss: fetch live with retry
        3. Cache result
        4. Tag with freshness
        Returns FetchResult always — never raises.
        """
        if not self.enabled:
            return FetchResult(
                data=None, source=self.API_NAME,
                fetched_at=self._now(), freshness="DISABLED",
                error="Fetcher disabled",
            )

        cache_key = CacheManager.make_key(self.API_NAME, kwargs)

        # Cache check
        cached = self.cache.get(cache_key, self.DATA_TYPE)
        if cached is not None:
            freshness = self._compute_freshness(
                self.cache.get_ttl(self.DATA_TYPE)
            )
            return FetchResult(
                data       = cached,
                source     = self.API_NAME,
                fetched_at = self._now(),
                freshness  = freshness,
                cache_hit  = True,
            )

        # Live fetch with retry
        data, error = self._fetch_with_retry(**kwargs)

        if data:
            self.cache.set(cache_key, data, self.DATA_TYPE)
            freshness = "LIVE_VERIFIED"
        else:
            freshness = "UNVERIFIED"

        return FetchResult(
            data       = data,
            source     = self.API_NAME,
            fetched_at = self._now(),
            freshness  = freshness,
            cache_hit  = False,
            error      = error,
        )

    def fetch_chunks(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Convenience: fetch + convert to BAState chunks.
        Returns empty list on failure.
        """
        result = self.fetch(**kwargs)
        if not result.success:
            return []
        try:
            chunks = self.to_chunks(result.data)
            # Tag each chunk with freshness + source
            for chunk in chunks:
                chunk["live_source"]    = self.API_NAME
                chunk["live_freshness"] = result.freshness
                chunk["live_fetched"]   = result.fetched_at
            return chunks
        except Exception as e:
            print(f"[{self.API_NAME}] to_chunks error: {e}")
            return []

    def health_check(self) -> bool:
        """
        Quick health check — can this fetcher reach its API?
        Override in subclass for API-specific check.
        """
        return self.enabled

    # ═══════════════════════════════════════════════════════════════════════
    # RETRY LOGIC
    # ═══════════════════════════════════════════════════════════════════════

    def _fetch_with_retry(
        self, **kwargs
    ) -> tuple:
        """
        Fetch with exponential backoff retry.
        Returns (data, error_string).
        """
        last_error = ""
        for attempt in range(MAX_RETRIES):
            try:
                data = self.fetch_live(**kwargs)
                return data, ""
            except Exception as e:
                last_error = str(e)
                if attempt < MAX_RETRIES - 1:
                    wait = RETRY_BACKOFF[attempt]
                    print(f"[{self.API_NAME}] Retry {attempt+1}/{MAX_RETRIES} "
                          f"in {wait}s: {e}")
                    time.sleep(wait)

        print(f"[{self.API_NAME}] All retries failed: {last_error}")
        return None, last_error

    # ═══════════════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════════════

    def _now(self) -> str:
        """Return current UTC time as ISO string."""
        return datetime.datetime.utcnow().isoformat()

    def _compute_freshness(self, ttl_seconds: int) -> str:
        """
        Compute freshness tag based on TTL.
        Used for cache hits to show how fresh the data is.
        """
        if ttl_seconds <= 15 * 60:
            return "LIVE_VERIFIED"
        elif ttl_seconds <= 24 * 60 * 60:
            return "RECENT_24H"
        elif ttl_seconds <= 7 * 24 * 60 * 60:
            return "STALE_7D"
        return "OLD"

    def _make_chunk(
        self,
        text:     str,
        section:  str = "Live Market Data",
        source:   str = "",
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Helper to build a standard context chunk."""
        chunk = {
            "text":        text,
            "section":     section,
            "page":        "0",
            "source":      source or self.API_NAME,
            "live_source": self.API_NAME,
        }
        if metadata:
            chunk.update(metadata)
        return chunk


# ── Concrete stub for testing ─────────────────────────────────────────────────

class EchoFetcher(BaseAPIFetcher):
    """
    Test fetcher — echoes params back as data.
    Used to verify base class logic without network calls.
    """
    API_NAME  = "echo"
    DATA_TYPE = "default"

    def fetch_live(self, **kwargs) -> Dict[str, Any]:
        return {"echo": kwargs, "timestamp": self._now()}

    def to_chunks(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [self._make_chunk(
            text    = f"Echo data: {data}",
            section = "Test Data",
        )]


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile
    try:
        from rich import print as rprint
    except ImportError:
        rprint = print

    rprint("\n[bold cyan]-- BaseAPIFetcher sanity check --[/bold cyan]")

    with tempfile.TemporaryDirectory() as tmp:
        cache   = CacheManager(db_path=Path(tmp) / "test.db")
        fetcher = EchoFetcher(cache=cache)

        rprint("[green]✓[/green] EchoFetcher instantiated")

        # First fetch — live
        result = fetcher.fetch(ticker="AAPL", field="price")
        rprint(f"[green]✓[/green] Live fetch: {result}")
        assert result.success
        assert result.cache_hit is False
        assert result.freshness == "LIVE_VERIFIED"

        # Second fetch — cache hit
        result2 = fetcher.fetch(ticker="AAPL", field="price")
        rprint(f"[green]✓[/green] Cache hit: {result2}")
        assert result2.cache_hit is True

        # fetch_chunks
        chunks = fetcher.fetch_chunks(ticker="AAPL")
        rprint(f"[green]✓[/green] Chunks: {len(chunks)}")
        assert len(chunks) > 0
        assert "live_source"    in chunks[0]
        assert "live_freshness" in chunks[0]

        # health check
        assert fetcher.health_check() is True
        rprint("[green]✓[/green] Health check OK")

        # disabled fetcher
        fetcher.enabled = False
        result3 = fetcher.fetch(ticker="AAPL")
        assert not result3.success
        rprint("[green]✓[/green] Disabled fetcher handled")

    rprint(f"\n[bold green]BaseAPIFetcher ready.[/bold green]\n")