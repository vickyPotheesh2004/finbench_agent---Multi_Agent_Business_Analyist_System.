"""
tests/test_live_data_infra.py
FinBench Multi-Agent Business Analyst AI

Tests for Phase 7A infrastructure:
  CacheManager, BaseAPIFetcher, DataShield, FetchQueue

24 tests — no network calls needed.
"""

import sys
import datetime
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from src.live_data.cache_manager import CacheManager, TTL_MAP
from src.live_data.base_fetcher  import BaseAPIFetcher, EchoFetcher, FetchResult
from src.live_data.data_shield   import DataShield, FRESHNESS_LIVE
from src.live_data.fetch_queue   import FetchQueue


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_cache(tmp_path):
    return CacheManager(db_path=tmp_path / "test.db")

@pytest.fixture
def echo_fetcher(tmp_path):
    cache = CacheManager(db_path=tmp_path / "echo.db")
    return EchoFetcher(cache=cache)

@pytest.fixture
def shield():
    return DataShield()

@pytest.fixture
def queue(tmp_path, shield):
    cache = CacheManager(db_path=tmp_path / "q.db")
    q     = FetchQueue(shield=shield)
    f1    = EchoFetcher(cache=cache); f1.API_NAME = "echo_a"
    f2    = EchoFetcher(cache=cache); f2.API_NAME = "echo_b"
    q.register(f1)
    q.register(f2)
    return q


# ════════════════════════════════════════════════════════════════════════════
# GROUP 1 -- CacheManager (tests 01-06)
# ════════════════════════════════════════════════════════════════════════════

class TestCacheManager:

    def test_01_instantiates(self, tmp_cache):
        """CacheManager must instantiate and create DB"""
        assert tmp_cache is not None

    def test_02_set_and_get(self, tmp_cache):
        """Set then Get must return same value"""
        tmp_cache.set("k1", {"price": 189.3}, "stock_price")
        val = tmp_cache.get("k1", "stock_price")
        assert val == {"price": 189.3}

    def test_03_miss_returns_none(self, tmp_cache):
        """Missing key must return None"""
        assert tmp_cache.get("nonexistent") is None

    def test_04_ttl_stock_price_15min(self, tmp_cache):
        """stock_price TTL must be 15 minutes"""
        assert tmp_cache.get_ttl("stock_price") == 15 * 60

    def test_05_make_key_deterministic(self, tmp_cache):
        """Same params must always produce same key"""
        k1 = CacheManager.make_key("yfinance", {"ticker": "AAPL"})
        k2 = CacheManager.make_key("yfinance", {"ticker": "AAPL"})
        assert k1 == k2
        assert len(k1) == 32

    def test_06_clear_all(self, tmp_cache):
        """clear_all must remove all entries"""
        tmp_cache.set("k1", {"v": 1})
        tmp_cache.set("k2", {"v": 2})
        tmp_cache.clear_all()
        assert tmp_cache.get("k1") is None
        assert tmp_cache.get("k2") is None


# ════════════════════════════════════════════════════════════════════════════
# GROUP 2 -- BaseAPIFetcher (tests 07-12)
# ════════════════════════════════════════════════════════════════════════════

class TestBaseAPIFetcher:

    def test_07_echo_fetcher_instantiates(self, echo_fetcher):
        """EchoFetcher must instantiate"""
        assert echo_fetcher is not None

    def test_08_fetch_returns_fetch_result(self, echo_fetcher):
        """fetch() must return FetchResult"""
        result = echo_fetcher.fetch(ticker="AAPL")
        assert isinstance(result, FetchResult)

    def test_09_first_fetch_is_live(self, echo_fetcher):
        """First fetch must be live (cache_hit=False)"""
        result = echo_fetcher.fetch(ticker="TEST_LIVE")
        assert result.cache_hit is False
        assert result.freshness == "LIVE_VERIFIED"

    def test_10_second_fetch_is_cached(self, echo_fetcher):
        """Second identical fetch must be cache hit"""
        echo_fetcher.fetch(ticker="TEST_CACHE")
        result2 = echo_fetcher.fetch(ticker="TEST_CACHE")
        assert result2.cache_hit is True

    def test_11_disabled_fetcher_returns_failure(self, echo_fetcher):
        """Disabled fetcher must return success=False"""
        echo_fetcher.enabled = False
        result = echo_fetcher.fetch(ticker="AAPL")
        assert not result.success
        echo_fetcher.enabled = True

    def test_12_fetch_chunks_returns_tagged_list(self, echo_fetcher):
        """fetch_chunks must return list with live_source tag"""
        chunks = echo_fetcher.fetch_chunks(ticker="AAPL")
        assert len(chunks) > 0
        assert "live_source"    in chunks[0]
        assert "live_freshness" in chunks[0]


# ════════════════════════════════════════════════════════════════════════════
# GROUP 3 -- DataShield (tests 13-18)
# ════════════════════════════════════════════════════════════════════════════

class TestDataShield:

    def test_13_instantiates(self, shield):
        """DataShield must instantiate"""
        assert shield is not None

    def test_14_live_tag_for_recent(self, shield):
        """Data fetched now must be LIVE_VERIFIED"""
        now = datetime.datetime.utcnow().isoformat()
        assert shield.tag_freshness(now) == "LIVE_VERIFIED"

    def test_15_old_tag_for_stale(self, shield):
        """Data from 10 days ago must be OLD"""
        old = (datetime.datetime.utcnow() -
               datetime.timedelta(days=10)).isoformat()
        assert shield.tag_freshness(old) == "OLD"

    def test_16_validation_passes_complete_data(self, shield):
        """Valid stock data must pass validation"""
        ok, errs = shield.validate(
            {"ticker": "AAPL", "price": 189.3}, "stock_price"
        )
        assert ok
        assert errs == []

    def test_17_mnpi_mode_blocks_fetch(self, shield):
        """MNPI mode must block fetches"""
        shield.set_mnpi_mode(True)
        allowed, _ = shield.check_fetch_allowed()
        assert not allowed
        shield.set_mnpi_mode(False)

    def test_18_shield_chunks_filters_empty(self, shield):
        """shield_chunks must filter empty text chunks"""
        now    = datetime.datetime.utcnow().isoformat()
        chunks = [
            {"text": "Good data $189.30"},
            {"text": ""},
            {"text": "x"},   # too short
        ]
        result = shield.shield_chunks(chunks, now)
        assert len(result) == 1
        assert result[0]["text"] == "Good data $189.30"


# ════════════════════════════════════════════════════════════════════════════
# GROUP 4 -- FetchQueue (tests 19-24)
# ════════════════════════════════════════════════════════════════════════════

class TestFetchQueue:

    def test_19_instantiates(self, queue):
        """FetchQueue must instantiate with registered fetchers"""
        assert len(queue.get_registered()) == 2

    def test_20_run_all_returns_dict(self, queue):
        """run_all must return dict of results"""
        results = queue.run_all(ticker="AAPL")
        assert isinstance(results, dict)
        assert len(results) == 2

    def test_21_all_fetchers_succeed(self, queue):
        """All echo fetchers must succeed"""
        results = queue.run_all(ticker="AAPL")
        assert all(r.success for r in results.values())

    def test_22_chunks_available_after_run(self, queue):
        """Chunks must be available after run_all"""
        queue.run_all(ticker="AAPL")
        chunks = queue.get_all_chunks()
        assert len(chunks) >= 2

    def test_23_mnpi_blocks_entire_queue(self, queue, shield):
        """MNPI mode must block all fetches in queue"""
        shield.set_mnpi_mode(True)
        results = queue.run_all(ticker="AAPL")
        assert results == {}
        shield.set_mnpi_mode(False)

    def test_24_summary_correct(self, queue):
        """get_summary must return correct counts"""
        queue.run_all(ticker="AAPL")
        summary = queue.get_summary()
        assert summary["total"]   == 2
        assert summary["success"] == 2
        assert summary["failed"]  == 0