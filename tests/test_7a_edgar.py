"""
tests/test_7a_edgar.py
Tests for Phase 7A SEC EDGAR Client
PDR-BAAAI-001 Rev 1.0
30 tests - no network needed (cache + mocks)
"""
import json
import os
import pytest
from pathlib import Path
from src.live_data.edgar import (
    EDGARClient,
    EDGARFiling,
    SUPPORTED_FORMS,
    KNOWN_CIKS,
    DEFAULT_CACHE_DIR,
    SEED,
    RATE_LIMIT_SEC,
)


@pytest.fixture
def cache_dir(tmp_path):
    return str(tmp_path / "edgar_cache")


@pytest.fixture
def client(cache_dir):
    return EDGARClient(cache_dir=cache_dir, enabled=False)


def _make_filing(ticker="AAPL", form_type="10-K") -> EDGARFiling:
    return EDGARFiling(
        ticker           = ticker,
        cik              = "0000320193",
        form_type        = form_type,
        filed_date       = "2023-11-03",
        accession_number = "0000320193-23-000106",
        document_url     = "https://example.com/filing.htm",
        fiscal_year      = "FY2023",
    )


class TestConstants:

    def test_01_supported_forms_defined(self):
        assert "10-K" in SUPPORTED_FORMS
        assert "10-Q" in SUPPORTED_FORMS
        assert "8-K"  in SUPPORTED_FORMS

    def test_02_known_ciks_has_major_tickers(self):
        assert "AAPL" in KNOWN_CIKS
        assert "MSFT" in KNOWN_CIKS
        assert "GOOGL" in KNOWN_CIKS

    def test_03_seed_is_42(self):
        assert SEED == 42

    def test_04_rate_limit_defined(self):
        assert RATE_LIMIT_SEC > 0


class TestEDGARFiling:

    def test_05_creates_correctly(self):
        f = _make_filing()
        assert f.ticker    == "AAPL"
        assert f.form_type == "10-K"
        assert f.cik       == "0000320193"

    def test_06_ticker_uppercased(self):
        f = _make_filing(ticker="aapl")
        assert f.ticker == "AAPL"

    def test_07_fiscal_year_inferred(self):
        f = EDGARFiling(
            ticker="AAPL", cik="123", form_type="10-K",
            filed_date="2023-11-03", accession_number="123",
        )
        assert f.fiscal_year == "FY2023"

    def test_08_to_dict_has_required_keys(self):
        d = _make_filing().to_dict()
        assert "ticker"           in d
        assert "cik"              in d
        assert "form_type"        in d
        assert "fiscal_year"      in d
        assert "filed_date"       in d
        assert "accession_number" in d
        assert "cached"           in d

    def test_09_repr_contains_ticker(self):
        assert "AAPL" in repr(_make_filing())

    def test_10_cached_false_when_no_local_path(self):
        f = _make_filing()
        assert f.cached is False


class TestInstantiation:

    def test_11_instantiates_with_defaults(self, client):
        assert client is not None

    def test_12_disabled_by_default_in_tests(self, client):
        assert client.enabled is False

    def test_13_cache_dir_created(self, cache_dir):
        EDGARClient(cache_dir=cache_dir, enabled=False)
        assert os.path.isdir(cache_dir)

    def test_14_known_ciks_preloaded(self, client):
        assert client.get_cik("AAPL") == KNOWN_CIKS["AAPL"]
        assert client.get_cik("MSFT") == KNOWN_CIKS["MSFT"]


class TestCacheOperations:

    def test_15_save_and_load_cache(self, client):
        filing  = _make_filing("AAPL", "10-K")
        client.save_to_cache("AAPL", "10-K", [filing])
        loaded  = client._load_from_cache("AAPL", "10-K", 5)
        assert len(loaded) == 1
        assert loaded[0].ticker == "AAPL"

    def test_16_cache_path_format(self, client):
        path = client.get_cache_path("AAPL", "10-K")
        assert "AAPL" in path
        assert "10-K" in path
        assert path.endswith(".json")

    def test_17_is_cached_false_when_empty(self, client):
        assert client.is_cached("AAPL", "10-K") is False

    def test_18_is_cached_true_after_save(self, client):
        client.save_to_cache("AAPL", "10-K", [_make_filing()])
        assert client.is_cached("AAPL", "10-K") is True

    def test_19_clear_cache_deletes_files(self, client):
        client.save_to_cache("AAPL", "10-K", [_make_filing()])
        deleted = client.clear_cache("AAPL")
        assert deleted >= 1

    def test_20_clear_all_cache(self, client):
        client.save_to_cache("AAPL", "10-K",  [_make_filing("AAPL", "10-K")])
        client.save_to_cache("MSFT", "10-K",  [_make_filing("MSFT", "10-K")])
        deleted = client.clear_cache()
        assert deleted >= 2

    def test_21_cache_limit_respected(self, client):
        filings = [_make_filing() for _ in range(5)]
        client.save_to_cache("AAPL", "10-K", filings)
        loaded = client._load_from_cache("AAPL", "10-K", 2)
        assert len(loaded) == 2

    def test_22_corrupt_cache_returns_empty(self, client, cache_dir):
        path = client.get_cache_path("AAPL", "10-K")
        Path(path).write_text("NOT JSON", encoding="utf-8")
        result = client._load_from_cache("AAPL", "10-K", 5)
        assert result == []


class TestGetFilings:

    def test_23_disabled_returns_empty_when_no_cache(self, client):
        results = client.get_filings("AAPL", "10-K")
        assert results == []

    def test_24_returns_cached_when_available(self, client):
        client.save_to_cache("AAPL", "10-K", [_make_filing()])
        results = client.get_filings("AAPL", "10-K")
        assert len(results) == 1

    def test_25_invalid_form_type_raises(self, client):
        with pytest.raises(ValueError):
            client.get_filings("AAPL", "INVALID_FORM")

    def test_26_ticker_case_insensitive(self, client):
        client.save_to_cache("AAPL", "10-K", [_make_filing()])
        results = client.get_filings("aapl", "10-K")
        assert len(results) == 1

    def test_27_multiple_form_types_cached_separately(self, client):
        client.save_to_cache("AAPL", "10-K", [_make_filing("AAPL", "10-K")])
        client.save_to_cache("AAPL", "10-Q", [_make_filing("AAPL", "10-Q")])
        r1 = client.get_filings("AAPL", "10-K")
        r2 = client.get_filings("AAPL", "10-Q")
        assert r1[0].form_type == "10-K"
        assert r2[0].form_type == "10-Q"


class TestCIKResolution:

    def test_28_known_cik_returned_without_network(self, client):
        cik = client.get_cik("AAPL")
        assert cik == "0000320193"

    def test_29_unknown_ticker_returns_none_when_disabled(self, client):
        cik = client.get_cik("UNKNOWN_TICKER_XYZ")
        assert cik is None

    def test_30_seed_unchanged(self):
        assert SEED == 42