"""
tools/download_sample_pdfs.py
FinBench — SEC EDGAR Sample PDF Downloader

Downloads 10 real 10-K / 10-Q PDFs from SEC EDGAR for eval and training.
Free, legal, public domain. Follows SEC fair-use rate limits.

Usage:
    python tools/download_sample_pdfs.py
    python tools/download_sample_pdfs.py --output-dir documents/sec_filings
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_pdfs")

# SEC EDGAR fair-use: max 10 req/sec. We use 2/sec to be safe.
RATE_LIMIT_SEC = 0.5
USER_AGENT     = "FinBenchAgent/1.0 research@example.com"

# 10 curated 10-K filings — all major US corporations, FY2023-2024
# Format: (ticker, fiscal_year, doc_type, accession_number, primary_document_name)
FILINGS = [
    # Big Tech
    ("AAPL",  "FY2023", "10-K", "0000320193-23-000106",  "aapl-20230930.htm"),
    ("MSFT",  "FY2023", "10-K", "0000950170-23-035122",  "msft-20230630.htm"),
    ("GOOGL", "FY2023", "10-K", "0001652044-24-000022",  "goog-20231231.htm"),
    ("AMZN",  "FY2023", "10-K", "0001018724-24-000008",  "amzn-20231231.htm"),
    ("META",  "FY2023", "10-K", "0001326801-24-000012",  "meta-20231231.htm"),
    ("NVDA",  "FY2024", "10-K", "0001045810-24-000029",  "nvda-20240128.htm"),
    ("TSLA",  "FY2023", "10-K", "0001628280-24-002390",  "tsla-20231231.htm"),
    # Finance
    ("JPM",   "FY2023", "10-K", "0000019617-24-000244",  "jpm-20231231.htm"),
    ("GS",    "FY2023", "10-K", "0000886982-24-000008",  "gs-20231231.htm"),
    ("BAC",   "FY2023", "10-K", "0000070858-24-000140",  "bac-20231231.htm"),
]

# CIK lookup (cached — matches KNOWN_CIKS in edgar.py)
CIKS = {
    "AAPL":  "0000320193",
    "MSFT":  "0000789019",
    "GOOGL": "0001652044",
    "AMZN":  "0001018724",
    "META":  "0001326801",
    "NVDA":  "0001045810",
    "TSLA":  "0001318605",
    "JPM":   "0000019617",
    "GS":    "0000886982",
    "BAC":   "0000070858",
}

SEC_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}"


def download_one(
    ticker:        str,
    fiscal_year:   str,
    doc_type:      str,
    accession:     str,
    primary_doc:   str,
    output_dir:    Path,
) -> bool:
    """
    Download a single filing. Returns True on success.

    Saves as HTML (SEC's primary format). HTMLs work fine with your
    N01 PDFIngestor via the HTML path. We don't need actual PDFs —
    your system handles HTML equally well.
    """
    cik = CIKS.get(ticker)
    if not cik:
        logger.error("No CIK for %s", ticker)
        return False

    accession_nodash = accession.replace("-", "")
    cik_int          = int(cik)   # SEC uses integer CIK in archive URLs

    url = (
        f"https://www.sec.gov/Archives/edgar/data/{cik_int}/"
        f"{accession_nodash}/{primary_doc}"
    )

    filename = f"{ticker}_{fiscal_year}_{doc_type}.html"
    out_path = output_dir / filename

    if out_path.exists() and out_path.stat().st_size > 10_000:
        logger.info("  [CACHED] %s", filename)
        return True

    logger.info("  Downloading %s ...", filename)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as resp:
            content = resp.read()

        if len(content) < 10_000:
            logger.warning("  Suspiciously small file (%d bytes) — skipping save", len(content))
            return False

        out_path.write_bytes(content)
        size_mb = len(content) / 1_000_000
        logger.info("  [DONE]   %s (%.1f MB)", filename, size_mb)
        return True

    except Exception as exc:
        logger.error("  [FAIL]   %s: %s", filename, exc)
        logger.error("           URL: %s", url)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Download SEC EDGAR sample filings")
    parser.add_argument(
        "--output-dir",
        default="documents/sec_filings",
        help="Output directory (default: documents/sec_filings)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("SEC EDGAR Sample Downloader")
    logger.info("  Target: %d filings", len(FILINGS))
    logger.info("  Output: %s", output_dir.resolve())
    logger.info("  Rate:   %.1f requests/sec (SEC fair-use)", 1.0 / RATE_LIMIT_SEC)
    logger.info("=" * 70)

    success = 0
    failed  = []
    for i, (ticker, fy, doc_type, accession, primary_doc) in enumerate(FILINGS, start=1):
        logger.info("[%d/%d] %s %s %s", i, len(FILINGS), ticker, fy, doc_type)
        if download_one(ticker, fy, doc_type, accession, primary_doc, output_dir):
            success += 1
        else:
            failed.append(f"{ticker}_{fy}_{doc_type}")
        time.sleep(RATE_LIMIT_SEC)

    logger.info("=" * 70)
    logger.info("Complete: %d/%d succeeded", success, len(FILINGS))
    if failed:
        logger.warning("Failed:")
        for f in failed:
            logger.warning("  - %s", f)
        logger.warning("")
        logger.warning("Note: SEC occasionally restructures accession URLs.")
        logger.warning("If a specific filing fails, manually download from:")
        logger.warning("  https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany")
    logger.info("=" * 70)

    return 0 if success >= 7 else 1   # Need >= 7 of 10 for meaningful eval


if __name__ == "__main__":
    sys.exit(main())