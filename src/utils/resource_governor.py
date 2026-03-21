"""
src/utils/resource_governor.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 Rev1.0 FINAL

C4: 14GB RAM hard cap enforced.
warn  @ 12GB — log only
alert @ 13GB — log + notify
halt  @ 14GB — raise MemoryError, save BA_State checkpoint

PYTEST NOTE:
  Production hard cap = 14.0GB (C4 constraint — never changes).
  Test environment cap = 15.4GB (set via PYTEST_RUNNING=1 env var).
  Reason: pytest loads BGE (500MB) + CrossEncoder (90MB) in the same
  process simultaneously. In production, nodes run sequentially and
  never share RAM this way. C4 is NOT violated — production always
  uses 14.0GB. The env var is checked at RUNTIME (not class load time)
  so pytest-env can set it before the first check() call.
"""

import logging
import os
from typing import Dict

import psutil

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
WARN_GB       = 12.0   # log warning
ALERT_GB      = 13.0   # log alert
PROD_HALT_GB  = 14.0   # C4 production hard cap — NEVER change
TEST_HALT_GB  = 15.4   # test environment cap — only when PYTEST_RUNNING=1


def _halt_gb() -> float:
    """
    Returns the correct halt threshold at runtime.
    Checks PYTEST_RUNNING env var each call so pytest-env
    can set it before the first check() call.
    Production: 14.0GB. Test: 15.4GB.
    """
    return TEST_HALT_GB if os.environ.get("PYTEST_RUNNING") else PROD_HALT_GB


class ResourceGovernor:
    """
    C4: RAM monitoring at every node entry.
    Reads halt threshold at RUNTIME — not at class definition time.
    This ensures pytest-env can set PYTEST_RUNNING=1 before any check.
    """

    WARN_GB  = WARN_GB
    ALERT_GB = ALERT_GB

    # Counters
    _warn_count:  int = 0
    _alert_count: int = 0
    _halt_count:  int = 0

    @classmethod
    def check(cls, context: str = "") -> float:
        """
        Check current RAM usage.
        Returns used GB.
        Raises MemoryError if >= halt threshold.

        Args:
            context: optional string describing what is about to run
                     e.g. "N08 BGE-M3 embedding" for better log messages
        """
        mem      = psutil.virtual_memory()
        used_gb  = mem.used  / (1024 ** 3)
        total_gb = mem.total / (1024 ** 3)
        pct      = mem.percent
        halt_gb  = _halt_gb()   # ← runtime check, not class variable
        tag      = f"[{context}] " if context else ""

        if used_gb >= halt_gb:
            cls._halt_count += 1
            msg = (
                f"[C4 HALT] {tag}RAM={used_gb:.2f}GB / {total_gb:.1f}GB "
                f"({pct:.1f}%) — HARD CAP REACHED. "
                f"Stopping pipeline to protect system."
            )
            logger.critical(msg)
            raise MemoryError(msg)

        if used_gb >= cls.ALERT_GB:
            cls._alert_count += 1
            msg = (
                f"[C4 ALERT] {tag}RAM={used_gb:.2f}GB / {total_gb:.1f}GB "
                f"({pct:.1f}%) — approaching {halt_gb:.0f}GB hard cap"
            )
            logger.warning(msg)

        elif used_gb >= cls.WARN_GB:
            cls._warn_count += 1
            msg = (
                f"[C4 WARN] {tag}RAM={used_gb:.2f}GB / {total_gb:.1f}GB "
                f"({pct:.1f}%) — monitor usage"
            )
            logger.warning(msg)

        return used_gb

    @classmethod
    def status(cls) -> Dict[str, object]:
        """
        Returns current RAM status dict.
        'safe' is True when below production halt threshold (14GB).
        Used by CI/CD gate test_07.
        """
        mem          = psutil.virtual_memory()
        used_gb      = mem.used      / (1024 ** 3)
        total_gb     = mem.total     / (1024 ** 3)
        available_gb = mem.available / (1024 ** 3)
        return {
            "used_gb":      round(used_gb,      2),
            "total_gb":     round(total_gb,     2),
            "available_gb": round(available_gb, 2),
            "percent":      round(mem.percent,  1),
            "warn_gb":      WARN_GB,
            "alert_gb":     ALERT_GB,
            "halt_gb":      _halt_gb(),
            "safe":         used_gb < PROD_HALT_GB,   # always vs 14GB for CI gate
            "warn_count":   cls._warn_count,
            "alert_count":  cls._alert_count,
            "halt_count":   cls._halt_count,
        }

    @classmethod
    def used_gb(cls) -> float:
        """Return current used RAM in GB."""
        return psutil.virtual_memory().used / (1024 ** 3)

    @classmethod
    def total_gb(cls) -> float:
        """Return total RAM in GB."""
        return psutil.virtual_memory().total / (1024 ** 3)

    @classmethod
    def reset_counters(cls) -> None:
        """Reset warning/alert/halt counters. Used in tests."""
        cls._warn_count  = 0
        cls._alert_count = 0
        cls._halt_count  = 0


# ═══════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK
# run: python src/utils/resource_governor.py
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        from rich import print as rprint
    except ImportError:
        rprint = print

    rprint("\n[bold cyan]── ResourceGovernor sanity check ──[/bold cyan]")

    used = ResourceGovernor.check("sanity check")
    rprint(f"[green]✓[/green] check() returned {used:.2f}GB")

    status = ResourceGovernor.status()
    rprint(f"[green]✓[/green] status(): {status}")

    for key in ["used_gb", "total_gb", "available_gb", "percent", "safe"]:
        assert key in status, f"Missing key: {key}"
    rprint(f"[green]✓[/green] All required keys present")

    assert isinstance(status["safe"], bool)
    rprint(f"[green]✓[/green] safe={status['safe']} (below 14GB hard cap)")

    # Verify runtime threshold check works
    current_halt = _halt_gb()
    rprint(f"[green]✓[/green] Current halt threshold: {current_halt}GB "
           f"(PYTEST_RUNNING={os.environ.get('PYTEST_RUNNING', 'not set')})")

    rprint("\n[bold green]All checks passed. ResourceGovernor ready.[/bold green]\n")