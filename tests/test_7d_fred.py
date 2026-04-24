"""
tests/test_7d_fred.py
Tests for Phase 7D FRED Macro Data Feed
PDR-BAAAI-001 Rev 1.0
20 tests - no network needed
"""
import os
import time
import pytest
from pathlib import Path
from src.live_data.fred_feed import (
    FREDFeed,
    MacroObservation,
    MacroSnapshot,
    FRED_SERIES,
    CACHE_TTL_HOURS,
    SEED,
)


@pytest.fixture
def cache_dir(tmp_path):
    return str(tmp_path / "fred_cache")


@pytest.fixture
def feed(cache_dir):
    return FREDFeed(cache_dir=cache_dir, enabled=False)


def _make_snapshot(fed_rate=5.25, cpi=310.3, unemp=3.9, t10=4.45, t2=4.85):
    obs = {
        "fed_funds_rate": MacroObservation("FEDFUNDS", "2024-04-01", fed_rate),
        "cpi":            MacroObservation("CPIAUCSL", "2024-03-01", cpi),
        "unemployment":   MacroObservation("UNRATE",   "2024-03-01", unemp),
        "treasury_10y":   MacroObservation("DGS10",    "2024-04-12", t10),
        "treasury_2y":    MacroObservation("DGS2",     "2024-04-12", t2),
        "vix":            MacroObservation("VIXCLS",   "2024-04-12", 16.2),
    }
    return MacroSnapshot(observations=obs, fetched_at=time.time())


class TestConstants:

    def test_01_fred_series_defined(self):
        assert "fed_funds_rate" in FRED_SERIES
        assert "cpi"             in FRED_SERIES
        assert "unemployment"    in FRED_SERIES

    def test_02_fred_series_ids_correct(self):
        assert FRED_SERIES["fed_funds_rate"] == "FEDFUNDS"
        assert FRED_SERIES["cpi"]            == "CPIAUCSL"

    def test_03_cache_ttl_defined(self):
        assert CACHE_TTL_HOURS > 0

    def test_04_seed_is_42(self):
        assert SEED == 42


class TestMacroObservation:

    def test_05_creates_correctly(self):
        o = MacroObservation("FEDFUNDS", "2024-04-01", 5.25)
        assert o.series_id == "FEDFUNDS"
        assert o.date      == "2024-04-01"
        assert o.value     == 5.25

    def test_06_to_dict(self):
        d = MacroObservation("UNRATE", "2024-03-01", 3.9).to_dict()
        assert d["series_id"] == "UNRATE"
        assert d["value"]     == 3.9

    def test_07_repr_contains_series_id(self):
        o = MacroObservation("CPIAUCSL", "2024-03-01", 310.3)
        assert "CPIAUCSL" in repr(o)


class TestMacroSnapshot:

    def test_08_fed_funds_rate_property(self):
        s = _make_snapshot(fed_rate=5.50)
        assert s.fed_funds_rate == 5.50

    def test_09_cpi_property(self):
        s = _make_snapshot(cpi=315.7)
        assert s.cpi == 315.7

    def test_10_unemployment_property(self):
        s = _make_snapshot(unemp=4.1)
        assert s.unemployment == 4.1

    def test_11_treasury_10y_property(self):
        s = _make_snapshot(t10=4.50)
        assert s.treasury_10y == 4.50

    def test_12_not_stale_when_fresh(self):
        s = _make_snapshot()
        assert s.is_stale is False

    def test_13_stale_when_old(self):
        s = _make_snapshot()
        s.fetched_at = time.time() - (CACHE_TTL_HOURS + 1) * 3600
        assert s.is_stale is True

    def test_14_inversion_detected(self):
        """2Y > 10Y = inverted curve = recession signal."""
        s = _make_snapshot(t2=5.00, t10=4.45)
        assert s.inversion_signal is True

    def test_15_no_inversion_when_normal(self):
        s = _make_snapshot(t2=4.00, t10=4.45)
        assert s.inversion_signal is False

    def test_16_to_dict_has_keys(self):
        d = _make_snapshot().to_dict()
        assert "fed_funds_rate"   in d
        assert "cpi"              in d
        assert "inversion_signal" in d


class TestFREDFeed:

    def test_17_instantiates(self, feed):
        assert feed is not None
        assert feed.enabled is False

    def test_18_cache_dir_created(self, cache_dir):
        FREDFeed(cache_dir=cache_dir, enabled=False)
        assert os.path.isdir(cache_dir)

    def test_19_save_and_load_snapshot(self, feed):
        snap = _make_snapshot()
        feed.save_snapshot(snap)
        loaded = feed._load_snapshot_cache()
        assert loaded is not None
        assert loaded.fed_funds_rate == 5.25

    def test_20_clear_cache(self, feed):
        feed.save_snapshot(_make_snapshot())
        feed.save_series("TEST_SERIES", [
            MacroObservation("TEST_SERIES", "2024-01-01", 100.0),
        ])
        deleted = feed.clear_cache()
        assert deleted >= 2