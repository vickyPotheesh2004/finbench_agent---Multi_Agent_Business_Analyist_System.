"""
tests/test_7b_yfinance.py
Tests for Phase 7B Yahoo Finance Feed
PDR-BAAAI-001 Rev 1.0
25 tests - no network needed (cache + local data)
"""
import json
import os
import time
import pytest
from pathlib import Path
from src.live_data.yfinance_feed import (
    YFinanceFeed,
    MarketSnapshot,
    YFINANCE_ENABLED,
    DEFAULT_CACHE_DIR,
    CACHE_TTL_HOURS,
    INFO_FIELDS,
    SEED,
)


@pytest.fixture
def cache_dir(tmp_path):
    return str(tmp_path / "yfinance_cache")


@pytest.fixture
def feed(cache_dir):
    return YFinanceFeed(cache_dir=cache_dir, enabled=False)


def _make_snapshot(ticker="AAPL") -> MarketSnapshot:
    return MarketSnapshot(
        ticker = ticker,
        info   = {
            "shortName":         "Apple Inc.",
            "currentPrice":      189.30,
            "marketCap":         2_950_000_000_000,
            "trailingPE":        31.2,
            "trailingEps":       6.13,
            "totalRevenue":      383_285_000_000,
            "fiftyTwoWeekHigh":  199.62,
            "fiftyTwoWeekLow":   124.17,
            "sector":            "Technology",
            "currency":          "USD",
        },
        price_history = [
            {"date": "2023-09-01", "close": 189.46, "volume": 50_000_000},
            {"date": "2023-09-05", "close": 186.33, "volume": 48_000_000},
            {"date": "2023-09-06", "close": 177.79, "volume": 52_000_000},
        ],
        fetched_at = time.time(),
    )


class TestConstants:

    def test_01_cache_ttl_defined(self):
        assert CACHE_TTL_HOURS > 0

    def test_02_info_fields_defined(self):
        assert "currentPrice" in INFO_FIELDS
        assert "marketCap"    in INFO_FIELDS
        assert "trailingPE"   in INFO_FIELDS

    def test_03_seed_is_42(self):
        assert SEED == 42


class TestMarketSnapshot:

    def test_04_creates_correctly(self):
        s = _make_snapshot()
        assert s.ticker == "AAPL"

    def test_05_ticker_uppercased(self):
        s = _make_snapshot("aapl")
        assert s.ticker == "AAPL"

    def test_06_current_price_property(self):
        s = _make_snapshot()
        assert s.current_price == 189.30

    def test_07_market_cap_billions(self):
        s = _make_snapshot()
        assert s.market_cap_billions == pytest.approx(2950.0, rel=0.01)

    def test_08_pe_ratio_property(self):
        s = _make_snapshot()
        assert s.pe_ratio == 31.2

    def test_09_eps_property(self):
        s = _make_snapshot()
        assert s.eps == 6.13

    def test_10_revenue_billions(self):
        s = _make_snapshot()
        assert s.revenue_billions == pytest.approx(383.29, rel=0.01)

    def test_11_company_name_property(self):
        s = _make_snapshot()
        assert s.company_name == "Apple Inc."

    def test_12_not_stale_when_fresh(self):
        s = _make_snapshot()
        assert s.is_stale is False

    def test_13_stale_when_old(self):
        s = _make_snapshot()
        s.fetched_at = time.time() - (CACHE_TTL_HOURS + 1) * 3600
        assert s.is_stale is True

    def test_14_to_dict_has_keys(self):
        d = _make_snapshot().to_dict()
        assert "ticker"       in d
        assert "current_price" in d
        assert "market_cap_B" in d
        assert "pe_ratio"     in d
        assert "fetched_at"   in d

    def test_15_price_history_in_dict(self):
        d = _make_snapshot().to_dict()
        assert d["price_history_len"] == 3

    def test_16_repr_contains_ticker(self):
        assert "AAPL" in repr(_make_snapshot())


class TestYFinanceFeed:

    def test_17_instantiates(self, feed):
        assert feed is not None

    def test_18_disabled_by_default_in_tests(self, feed):
        assert feed.enabled is False

    def test_19_cache_dir_created(self, cache_dir):
        YFinanceFeed(cache_dir=cache_dir, enabled=False)
        assert os.path.isdir(cache_dir)

    def test_20_save_and_load_cache(self, feed):
        snap = _make_snapshot("AAPL")
        feed.save_snapshot(snap)
        loaded = feed._load_cache("AAPL")
        assert loaded is not None
        assert loaded.ticker == "AAPL"

    def test_21_is_cached_false_initially(self, feed):
        assert feed.is_cached("AAPL") is False

    def test_22_is_cached_true_after_save(self, feed):
        feed.save_snapshot(_make_snapshot("AAPL"))
        assert feed.is_cached("AAPL") is True

    def test_23_get_snapshot_from_cache(self, feed):
        feed.save_snapshot(_make_snapshot("AAPL"))
        snap = feed.get_snapshot("AAPL")
        assert snap is not None
        assert snap.ticker == "AAPL"

    def test_24_clear_cache(self, feed):
        feed.save_snapshot(_make_snapshot("AAPL"))
        feed.save_snapshot(_make_snapshot("MSFT"))
        deleted = feed.clear_cache()
        assert deleted >= 2

    def test_25_get_price_history_returns_list(self, feed):
        feed.save_snapshot(_make_snapshot("AAPL"))
        history = feed.get_price_history("AAPL")
        assert isinstance(history, list)
        assert len(history) == 3