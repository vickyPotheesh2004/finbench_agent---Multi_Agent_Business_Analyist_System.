"""
src/live_data/fetch_queue.py
FinBench Multi-Agent Business Analyst AI
Phase 7A — Live Data Infrastructure

Async-style fetch queue for pre-inference data gathering.
Runs ALL fetchers in parallel before inference starts.
Results cached locally — zero network during inference (C2).

Usage:
    queue = FetchQueue(shield=shield)
    queue.register(yfinance_fetcher)
    queue.register(fred_fetcher)

    # Pre-inference: fetch all in parallel
    results = queue.run_all(ticker="AAPL", company="Apple Inc")

    # Get chunks for BAState injection
    chunks = queue.get_all_chunks()
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.live_data.base_fetcher import BaseAPIFetcher, FetchResult
from src.live_data.data_shield  import DataShield
from src.utils.seed_manager     import SeedManager

SeedManager.set_all()

# ── Config ────────────────────────────────────────────────────────────────────
MAX_WORKERS         = 8      # parallel fetchers
QUEUE_TIMEOUT_SEC   = 15     # max seconds for entire queue
PER_FETCHER_TIMEOUT = 10     # max seconds per fetcher


class FetchQueue:
    """
    Parallel fetch queue — runs all registered fetchers simultaneously.
    Results validated by DataShield before returning.
    C2 compliant: all fetches happen pre-inference.
    """

    def __init__(
        self,
        shield:      Optional[DataShield] = None,
        max_workers: int                  = MAX_WORKERS,
        timeout:     int                  = QUEUE_TIMEOUT_SEC,
    ):
        SeedManager.set_all()
        self.shield      = shield or DataShield()
        self.max_workers = max_workers
        self.timeout     = timeout
        self._fetchers:  List[BaseAPIFetcher] = []
        self._results:   Dict[str, FetchResult] = {}
        self._chunks:    List[Dict[str, Any]]    = []

    # ═══════════════════════════════════════════════════════════════════════
    # REGISTRATION
    # ═══════════════════════════════════════════════════════════════════════

    def register(self, fetcher: BaseAPIFetcher) -> None:
        """Register a fetcher with the queue."""
        self._fetchers.append(fetcher)

    def register_many(self, fetchers: List[BaseAPIFetcher]) -> None:
        """Register multiple fetchers at once."""
        for f in fetchers:
            self.register(f)

    def unregister(self, api_name: str) -> None:
        """Remove a fetcher by API name."""
        self._fetchers = [
            f for f in self._fetchers
            if f.API_NAME != api_name
        ]

    def get_registered(self) -> List[str]:
        """Return list of registered fetcher names."""
        return [f.API_NAME for f in self._fetchers]

    # ═══════════════════════════════════════════════════════════════════════
    # QUEUE EXECUTION
    # ═══════════════════════════════════════════════════════════════════════

    def run_all(self, **kwargs) -> Dict[str, FetchResult]:
        """
        Run all registered fetchers in parallel.
        Returns dict: {api_name: FetchResult}

        Blocks until all complete or timeout reached.
        Failed fetchers return FetchResult with success=False.
        MNPI guard checked before any fetches.
        """
        # MNPI guard — check before any network calls
        allowed, reason = self.shield.check_fetch_allowed()
        if not allowed:
            print(f"[FetchQueue] {reason}")
            return {}

        if not self._fetchers:
            return {}

        self._results = {}
        self._chunks  = []

        start_time = time.time()

        def run_one(fetcher: BaseAPIFetcher):
            try:
                result = fetcher.fetch(**kwargs)
                return fetcher.API_NAME, result
            except Exception as e:
                from src.live_data.base_fetcher import FetchResult
                return fetcher.API_NAME, FetchResult(
                    data       = None,
                    source     = fetcher.API_NAME,
                    fetched_at = "",
                    freshness  = "UNVERIFIED",
                    error      = str(e),
                )

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {
                ex.submit(run_one, f): f.API_NAME
                for f in self._fetchers
                if f.enabled
            }
            for future in as_completed(
                futures, timeout=self.timeout
            ):
                try:
                    api_name, result = future.result(
                        timeout=PER_FETCHER_TIMEOUT
                    )
                    self._results[api_name] = result
                except Exception as e:
                    api_name = futures[future]
                    print(f"[FetchQueue] {api_name} timeout/error: {e}")

        elapsed = time.time() - start_time
        success = sum(1 for r in self._results.values() if r.success)
        print(f"[FetchQueue] Complete — "
              f"{success}/{len(self._fetchers)} success "
              f"in {elapsed:.1f}s")

        # Build chunks from all successful results
        self._build_chunks(**kwargs)

        return self._results

    def _build_chunks(self, **kwargs) -> None:
        """Convert all successful results to BAState chunks."""
        all_chunks = []
        for api_name, result in self._results.items():
            if not result.success:
                continue
            fetcher = self._get_fetcher(api_name)
            if not fetcher:
                continue
            try:
                chunks = fetcher.to_chunks(result.data)
                # Shield each chunk
                shielded = self.shield.shield_chunks(
                    chunks, result.fetched_at
                )
                all_chunks.extend(shielded)
            except Exception as e:
                print(f"[FetchQueue] {api_name} chunk build error: {e}")

        self._chunks = all_chunks

    # ═══════════════════════════════════════════════════════════════════════
    # RESULTS ACCESS
    # ═══════════════════════════════════════════════════════════════════════

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Get all live data chunks ready for BAState injection."""
        return self._chunks.copy()

    def get_result(self, api_name: str) -> Optional[FetchResult]:
        """Get result for specific API."""
        return self._results.get(api_name)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of last queue run."""
        total   = len(self._results)
        success = sum(1 for r in self._results.values() if r.success)
        cached  = sum(1 for r in self._results.values() if r.cache_hit)
        return {
            "total":        total,
            "success":      success,
            "failed":       total - success,
            "cached":       cached,
            "live":         success - cached,
            "total_chunks": len(self._chunks),
            "freshness":    {
                api: r.freshness
                for api, r in self._results.items()
            },
        }

    def get_live_summary_text(self) -> str:
        """
        Build a human-readable summary of live data.
        Used for BAState.live_data_summary field.
        """
        if not self._results:
            return ""

        lines = ["LIVE MARKET DATA SUMMARY"]
        lines.append(f"Fetched: {len(self._results)} sources")

        for api_name, result in self._results.items():
            if result.success:
                lines.append(
                    f"  [{result.freshness}] {api_name}: "
                    f"{len(self._chunks)} data points available"
                )
            else:
                lines.append(f"  [FAILED] {api_name}: {result.error[:50]}")

        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════════════

    def _get_fetcher(self, api_name: str) -> Optional[BaseAPIFetcher]:
        """Get fetcher by API name."""
        for f in self._fetchers:
            if f.API_NAME == api_name:
                return f
        return None

    def clear(self) -> None:
        """Clear results and chunks."""
        self._results = {}
        self._chunks  = []

    def disable_all(self) -> None:
        """Disable all registered fetchers."""
        for f in self._fetchers:
            f.enabled = False

    def enable_all(self) -> None:
        """Enable all registered fetchers."""
        for f in self._fetchers:
            f.enabled = True


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile
    try:
        from rich import print as rprint
    except ImportError:
        rprint = print

    from src.live_data.cache_manager import CacheManager
    from src.live_data.base_fetcher  import EchoFetcher

    rprint("\n[bold cyan]-- FetchQueue sanity check --[/bold cyan]")

    with tempfile.TemporaryDirectory() as tmp:
        cache   = CacheManager(db_path=Path(tmp) / "test.db")
        shield  = DataShield()
        queue   = FetchQueue(shield=shield)

        rprint("[green]✓[/green] FetchQueue instantiated")

        # Register fetchers
        f1 = EchoFetcher(cache=cache)
        f1.API_NAME = "echo_1"
        f2 = EchoFetcher(cache=cache)
        f2.API_NAME = "echo_2"
        queue.register(f1)
        queue.register(f2)

        registered = queue.get_registered()
        rprint(f"[green]✓[/green] Registered: {registered}")
        assert len(registered) == 2

        # Run all
        results = queue.run_all(ticker="AAPL", company="Apple Inc")
        rprint(f"[green]✓[/green] Results: {list(results.keys())}")
        assert len(results) == 2
        assert all(r.success for r in results.values())

        # Chunks
        chunks = queue.get_all_chunks()
        rprint(f"[green]✓[/green] Chunks: {len(chunks)}")
        assert len(chunks) >= 2

        # Summary
        summary = queue.get_summary()
        rprint(f"[green]✓[/green] Summary: {summary}")
        assert summary["success"] == 2

        # Summary text
        text = queue.get_live_summary_text()
        assert "LIVE MARKET DATA SUMMARY" in text
        rprint(f"[green]✓[/green] Summary text generated")

        # MNPI guard
        shield.set_mnpi_mode(True)
        results2 = queue.run_all(ticker="AAPL")
        assert results2 == {}
        shield.set_mnpi_mode(False)
        rprint("[green]✓[/green] MNPI guard blocks queue")

        # Disable all
        queue.disable_all()
        results3 = queue.run_all(ticker="AAPL")
        assert all(not r.success for r in results3.values()) \
               or len(results3) == 0
        queue.enable_all()
        rprint("[green]✓[/green] Disable/enable all works")

        assert f1.enabled is True
        rprint("[green]✓[/green] Re-enabled after enable_all")

    rprint(f"\n[bold green]FetchQueue ready.[/bold green]\n")