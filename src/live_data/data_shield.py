"""
src/live_data/data_shield.py
FinBench Multi-Agent Business Analyst AI
Phase 7C — DataShield Full Implementation

Quality + freshness validation layer for all live data.

Freshness tags:
  LIVE_VERIFIED   — fetched < 15 min ago
  RECENT_24H      — fetched < 24 hours ago
  STALE_7D        — fetched < 7 days ago
  OLD             — older than 7 days
  UNVERIFIED      — source quality unknown
  DISABLED        — fetcher disabled

Quality checks:
  Schema validation    — required fields present
  Outlier detection    — price/rate anomalies flagged
  Source conflict      — 2 APIs disagree > threshold
  MNPI guard           — MNPI doc disables all outbound fetches

C2 compliant: shield runs locally, zero network calls.
"""

import sys
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.seed_manager import SeedManager

SeedManager.set_all()

# ── Freshness thresholds (seconds) ────────────────────────────────────────────
FRESHNESS_LIVE     = 15 * 60          # 15 min
FRESHNESS_RECENT   = 24 * 60 * 60     # 24 hours
FRESHNESS_STALE    = 7  * 24 * 60 * 60# 7 days

# ── Conflict threshold ────────────────────────────────────────────────────────
CONFLICT_PCT       = 0.05             # 5% disagreement = conflict

# ── Required fields per data type ─────────────────────────────────────────────
REQUIRED_FIELDS: Dict[str, List[str]] = {
    "stock_price": ["ticker", "price"],
    "fx_rate":     ["base", "target", "rate"],
    "macro_data":  ["series_id", "value"],
    "news":        ["title", "url"],
    "sec_filing":  ["cik", "form_type"],
    "world_bank":  ["indicator", "value"],
    "default":     [],
}


class DataShield:
    """
    Quality + freshness validation for all live data.

    Usage:
        shield = DataShield()
        tag    = shield.tag_freshness(fetched_at_iso)
        ok, errors = shield.validate(data, data_type)
        shield.set_mnpi_mode(True)  # disable all fetches
    """

    def __init__(self):
        SeedManager.set_all()
        self._mnpi_mode    = False
        self._source_cache: Dict[str, Any] = {}

    # ═══════════════════════════════════════════════════════════════════════
    # FRESHNESS TAGGING
    # ═══════════════════════════════════════════════════════════════════════

    def tag_freshness(self, fetched_at_iso: str) -> str:
        """
        Tag data with freshness label based on fetch time.

        Args:
            fetched_at_iso: ISO format UTC timestamp

        Returns:
            LIVE_VERIFIED | RECENT_24H | STALE_7D | OLD | UNVERIFIED
        """
        try:
            fetched_at = datetime.datetime.fromisoformat(fetched_at_iso)
            age_secs   = (
                datetime.datetime.utcnow() - fetched_at
            ).total_seconds()

            if age_secs   <= FRESHNESS_LIVE:
                return "LIVE_VERIFIED"
            elif age_secs <= FRESHNESS_RECENT:
                return "RECENT_24H"
            elif age_secs <= FRESHNESS_STALE:
                return "STALE_7D"
            else:
                return "OLD"

        except Exception:
            return "UNVERIFIED"

    def tag_chunks(
        self,
        chunks:     List[Dict[str, Any]],
        fetched_at: str,
    ) -> List[Dict[str, Any]]:
        """Tag a list of chunks with freshness label."""
        tag = self.tag_freshness(fetched_at)
        for chunk in chunks:
            chunk["live_freshness"] = tag
        return chunks

    # ═══════════════════════════════════════════════════════════════════════
    # SCHEMA VALIDATION
    # ═══════════════════════════════════════════════════════════════════════

    def validate(
        self,
        data:      Dict[str, Any],
        data_type: str = "default",
    ) -> Tuple[bool, List[str]]:
        """
        Validate data against schema for its type.

        Returns:
            (is_valid, list_of_errors)
        """
        errors   = []
        required = REQUIRED_FIELDS.get(data_type, REQUIRED_FIELDS["default"])

        if not data:
            return False, ["Empty data"]

        for field in required:
            if field not in data or data[field] is None:
                errors.append(f"Missing required field: {field}")

        return len(errors) == 0, errors

    # ═══════════════════════════════════════════════════════════════════════
    # OUTLIER DETECTION
    # ═══════════════════════════════════════════════════════════════════════

    def check_outlier(
        self,
        value:     float,
        reference: float,
        threshold: float = 0.20,
    ) -> Tuple[bool, str]:
        """
        Check if a value is an outlier vs reference.
        Threshold: 20% deviation = outlier.

        Returns:
            (is_outlier, reason_string)
        """
        if reference == 0:
            return False, ""

        deviation = abs(value - reference) / abs(reference)

        if deviation > threshold:
            return True, (
                f"Outlier detected: {value:.2f} vs reference {reference:.2f} "
                f"({deviation:.1%} deviation)"
            )
        return False, ""

    # ═══════════════════════════════════════════════════════════════════════
    # SOURCE CONFLICT DETECTION
    # ═══════════════════════════════════════════════════════════════════════

    def check_conflict(
        self,
        value_a:  float,
        source_a: str,
        value_b:  float,
        source_b: str,
    ) -> Tuple[bool, str]:
        """
        Detect if two sources disagree beyond threshold.

        Returns:
            (conflict_detected, description)
        """
        if value_a == 0 or value_b == 0:
            return False, ""

        diff_pct = abs(value_a - value_b) / max(abs(value_a), abs(value_b))

        if diff_pct > CONFLICT_PCT:
            return True, (
                f"SOURCE CONFLICT: {source_a}={value_a:.4f} vs "
                f"{source_b}={value_b:.4f} ({diff_pct:.1%} difference)"
            )
        return False, ""

    # ═══════════════════════════════════════════════════════════════════════
    # MNPI GUARD
    # ═══════════════════════════════════════════════════════════════════════

    def set_mnpi_mode(self, enabled: bool) -> None:
        """
        Enable/disable MNPI mode.
        When enabled: all live data fetches are blocked.
        Use when document contains Material Non-Public Information.
        """
        self._mnpi_mode = enabled
        if enabled:
            print("[DataShield] MNPI MODE ACTIVE — all live fetches disabled")
        else:
            print("[DataShield] MNPI mode deactivated")

    @property
    def mnpi_mode(self) -> bool:
        """Return current MNPI mode status."""
        return self._mnpi_mode

    def check_fetch_allowed(self) -> Tuple[bool, str]:
        """
        Check if live data fetching is allowed.
        Returns (allowed, reason).
        """
        if self._mnpi_mode:
            return False, "MNPI_BLOCKED: document contains MNPI — live fetches disabled"
        return True, ""

    # ═══════════════════════════════════════════════════════════════════════
    # CHUNK QUALITY FILTER
    # ═══════════════════════════════════════════════════════════════════════

    def filter_chunks(
        self,
        chunks:          List[Dict[str, Any]],
        min_text_length: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Filter out low-quality chunks.
        Removes empty, too-short, or error chunks.
        """
        filtered = []
        for chunk in chunks:
            text = chunk.get("text", "")
            if not text or len(text.strip()) < min_text_length:
                continue
            if any(err in text.lower() for err in
                   ["error:", "failed:", "exception:"]):
                continue
            filtered.append(chunk)
        return filtered

    def shield_chunks(
        self,
        chunks:     List[Dict[str, Any]],
        fetched_at: str,
    ) -> List[Dict[str, Any]]:
        """
        Full shield pipeline:
        1. Tag freshness
        2. Filter low quality
        Returns shielded chunks ready for BAState injection.
        """
        if self._mnpi_mode:
            return []

        chunks = self.tag_chunks(chunks, fetched_at)
        chunks = self.filter_chunks(chunks)
        return chunks


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        from rich import print as rprint
    except ImportError:
        rprint = print

    rprint("\n[bold cyan]-- DataShield sanity check --[/bold cyan]")

    shield = DataShield()
    rprint("[green]✓[/green] DataShield instantiated")

    # Freshness tagging
    now       = datetime.datetime.utcnow().isoformat()
    old_time  = (
        datetime.datetime.utcnow() - datetime.timedelta(days=10)
    ).isoformat()
    hour_ago  = (
        datetime.datetime.utcnow() - datetime.timedelta(hours=2)
    ).isoformat()

    assert shield.tag_freshness(now)      == "LIVE_VERIFIED"
    assert shield.tag_freshness(hour_ago) == "RECENT_24H"
    assert shield.tag_freshness(old_time) == "OLD"
    rprint("[green]✓[/green] Freshness tags correct")

    # Schema validation
    ok, errs = shield.validate(
        {"ticker": "AAPL", "price": 189.30}, "stock_price"
    )
    assert ok
    rprint(f"[green]✓[/green] Validation pass: {ok}")

    ok2, errs2 = shield.validate({"ticker": "AAPL"}, "stock_price")
    assert not ok2
    rprint(f"[green]✓[/green] Validation fail: {errs2}")

    # Outlier detection
    is_out, reason = shield.check_outlier(250.0, 189.0, threshold=0.20)
    assert is_out
    rprint(f"[green]✓[/green] Outlier detected: {reason[:50]}")

    is_out2, _ = shield.check_outlier(190.0, 189.0, threshold=0.20)
    assert not is_out2
    rprint("[green]✓[/green] Normal value not flagged")

    # Source conflict
    conflict, desc = shield.check_conflict(1.085, "ECB", 1.200, "Frankfurter")
    assert conflict
    rprint(f"[green]✓[/green] Conflict detected: {desc[:50]}")

    # MNPI guard
    shield.set_mnpi_mode(True)
    allowed, reason = shield.check_fetch_allowed()
    assert not allowed
    rprint(f"[green]✓[/green] MNPI guard active: {reason[:50]}")

    shield.set_mnpi_mode(False)
    allowed2, _ = shield.check_fetch_allowed()
    assert allowed2
    rprint("[green]✓[/green] MNPI guard deactivated")

    # Chunk shielding
    chunks = [
        {"text": "Apple stock price $189.30", "section": "Live"},
        {"text": "", "section": "Empty"},
        {"text": "error: API failed", "section": "Bad"},
    ]
    shielded = shield.shield_chunks(chunks, now)
    assert len(shielded) == 1
    assert shielded[0]["live_freshness"] == "LIVE_VERIFIED"
    rprint(f"[green]✓[/green] Shield filtered: {len(chunks)} → {len(shielded)}")

    rprint(f"\n[bold green]DataShield ready.[/bold green]\n")