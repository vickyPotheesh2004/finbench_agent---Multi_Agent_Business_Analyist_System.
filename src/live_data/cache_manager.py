"""
src/live_data/cache_manager.py
FinBench Multi-Agent Business Analyst AI
Phase 7A — Live Data Infrastructure

SQLite-backed cache with TTL expiry.
Speed-first: aggressive caching, minimal network calls.

Cache TTL by data type:
  stock_price      15 min
  fx_rate          1 hour
  news             1 hour
  macro_data       24 hours
  sec_filing       7 days
  world_bank       30 days
  xbrl_tags        7 days
  default          1 hour

C2 compliant: all data stored locally, zero network during inference.
"""

import sys
import json
import sqlite3
import hashlib
import datetime
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.seed_manager import SeedManager

SeedManager.set_all()

# ── Cache DB path ─────────────────────────────────────────────────────────────
CACHE_DB = ROOT / "data" / "live_data_cache.db"

# ── TTL in seconds ────────────────────────────────────────────────────────────
TTL_MAP: Dict[str, int] = {
    "stock_price":   15 * 60,          # 15 minutes
    "fx_rate":       60 * 60,          # 1 hour
    "news":          60 * 60,          # 1 hour
    "macro_data":    24 * 60 * 60,     # 24 hours
    "sec_filing":    7  * 24 * 60 * 60,# 7 days
    "world_bank":    30 * 24 * 60 * 60,# 30 days
    "xbrl_tags":     7  * 24 * 60 * 60,# 7 days
    "earnings":      60 * 60,          # 1 hour
    "treasury":      60 * 60,          # 1 hour
    "employment":    24 * 60 * 60,     # 24 hours
    "default":       60 * 60,          # 1 hour
}


class CacheManager:
    """
    SQLite-backed cache with TTL expiry.
    Speed-first: returns cached data if within TTL.
    Thread-safe: each call opens/closes connection.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or CACHE_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def get(
        self,
        key:       str,
        data_type: str = "default",
    ) -> Optional[Any]:
        """
        Get cached value if within TTL.
        Returns None if expired or not found.
        """
        ttl = TTL_MAP.get(data_type, TTL_MAP["default"])
        try:
            conn = sqlite3.connect(str(self.db_path))
            cur  = conn.cursor()
            cur.execute("""
                SELECT value, fetched_at FROM live_cache
                WHERE cache_key = ?
            """, (key,))
            row = cur.fetchone()
            conn.close()

            if not row:
                return None

            value_json, fetched_at_str = row
            fetched_at = datetime.datetime.fromisoformat(fetched_at_str)
            age_secs   = (
                datetime.datetime.utcnow() - fetched_at
            ).total_seconds()

            if age_secs > ttl:
                return None   # expired

            return json.loads(value_json)

        except Exception as e:
            print(f"[Cache] GET error: {e}")
            return None

    def set(
        self,
        key:       str,
        value:     Any,
        data_type: str = "default",
    ) -> bool:
        """
        Store value in cache.
        Returns True on success.
        """
        try:
            value_json = json.dumps(value, default=str)
            now        = datetime.datetime.utcnow().isoformat()

            conn = sqlite3.connect(str(self.db_path))
            conn.execute("""
                INSERT OR REPLACE INTO live_cache
                (cache_key, data_type, value, fetched_at)
                VALUES (?, ?, ?, ?)
            """, (key, data_type, value_json, now))
            conn.commit()
            conn.close()
            return True

        except Exception as e:
            print(f"[Cache] SET error: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete a cache entry."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("DELETE FROM live_cache WHERE cache_key = ?", (key,))
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False

    def clear_expired(self) -> int:
        """Remove all expired entries. Returns count deleted."""
        deleted = 0
        try:
            conn = sqlite3.connect(str(self.db_path))
            cur  = conn.cursor()
            cur.execute("SELECT cache_key, data_type, fetched_at FROM live_cache")
            rows = cur.fetchall()
            for key, data_type, fetched_at_str in rows:
                ttl        = TTL_MAP.get(data_type, TTL_MAP["default"])
                fetched_at = datetime.datetime.fromisoformat(fetched_at_str)
                age        = (
                    datetime.datetime.utcnow() - fetched_at
                ).total_seconds()
                if age > ttl:
                    conn.execute(
                        "DELETE FROM live_cache WHERE cache_key = ?", (key,)
                    )
                    deleted += 1
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[Cache] clear_expired error: {e}")
        return deleted

    def clear_all(self) -> bool:
        """Clear entire cache. Used in tests."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("DELETE FROM live_cache")
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False

    def get_stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        try:
            conn  = sqlite3.connect(str(self.db_path))
            cur   = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM live_cache")
            total = cur.fetchone()[0]
            cur.execute(
                "SELECT data_type, COUNT(*) FROM live_cache GROUP BY data_type"
            )
            by_type = dict(cur.fetchall())
            conn.close()
            return {"total": total, **by_type}
        except Exception:
            return {"total": 0}

    @staticmethod
    def make_key(api_name: str, params: Dict[str, Any]) -> str:
        """Generate deterministic cache key from API name + params."""
        param_str = json.dumps(params, sort_keys=True, default=str)
        raw       = f"{api_name}:{param_str}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def get_ttl(self, data_type: str) -> int:
        """Return TTL seconds for a data type."""
        return TTL_MAP.get(data_type, TTL_MAP["default"])

    def _init_db(self) -> None:
        """Initialise SQLite cache table."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("""
                CREATE TABLE IF NOT EXISTS live_cache (
                    cache_key  TEXT PRIMARY KEY,
                    data_type  TEXT NOT NULL,
                    value      TEXT NOT NULL,
                    fetched_at TEXT NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_fetched ON "
                "live_cache(fetched_at)"
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[Cache] DB init error: {e}")


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile
    try:
        from rich import print as rprint
    except ImportError:
        rprint = print

    rprint("\n[bold cyan]-- CacheManager sanity check --[/bold cyan]")

    with tempfile.TemporaryDirectory() as tmp:
        cache = CacheManager(db_path=Path(tmp) / "test_cache.db")
        rprint("[green]✓[/green] CacheManager instantiated")

        # Set + Get
        cache.set("test_key", {"price": 189.30}, "stock_price")
        val = cache.get("test_key", "stock_price")
        assert val == {"price": 189.30}
        rprint(f"[green]✓[/green] Set + Get: {val}")

        # Miss
        assert cache.get("missing_key") is None
        rprint("[green]✓[/green] Miss returns None")

        # Make key
        key = CacheManager.make_key("yfinance", {"ticker": "AAPL"})
        assert len(key) == 32
        rprint(f"[green]✓[/green] make_key: {key}")

        # Stats
        stats = cache.get_stats()
        assert stats["total"] >= 1
        rprint(f"[green]✓[/green] Stats: {stats}")

        # TTL values
        assert cache.get_ttl("stock_price") == 15 * 60
        assert cache.get_ttl("sec_filing")  == 7 * 24 * 60 * 60
        rprint("[green]✓[/green] TTL values correct")

        # Clear
        cache.clear_all()
        assert cache.get("test_key") is None
        rprint("[green]✓[/green] Clear all works")

    rprint(f"\n[bold green]CacheManager ready.[/bold green]\n")