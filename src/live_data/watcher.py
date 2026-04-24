"""
src/live_data/watcher.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 · Rev 1.0

Phase 7E — Local File Watcher

Watches a folder for new financial documents and auto-triggers
N01-N03 ingestion pipeline. 100% local — C2 compliant.

Supported file types: PDF, DOCX, XLSX, CSV, PPTX, TXT

Constraints:
    C1  $0 cost — watchdog is free
    C2  100% local — zero network calls
    C5  seed=42
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Callable, List, Optional, Set

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".xlsx", ".csv",
    ".pptx", ".txt", ".json", ".eml",
}

DEFAULT_WATCH_DIR  = "documents"
DEFAULT_POLL_SEC   = 5.0        # seconds between scans
STABLE_WAIT_SEC    = 2.0        # wait for file to finish writing
SEED               = 42


class FileWatcherEvent:
    """Represents a file detected by the watcher."""

    __slots__ = ("path", "filename", "extension", "size_bytes", "detected_at")

    def __init__(self, path: str) -> None:
        p                = Path(path)
        self.path        = str(p.resolve())
        self.filename    = p.name
        self.extension   = p.suffix.lower()
        self.size_bytes  = p.stat().st_size if p.exists() else 0
        self.detected_at = time.time()

    def to_dict(self) -> dict:
        return {
            "path":        self.path,
            "filename":    self.filename,
            "extension":   self.extension,
            "size_bytes":  self.size_bytes,
            "detected_at": self.detected_at,
        }

    def __repr__(self) -> str:
        return f"FileWatcherEvent({self.filename})"


class FileWatcher:
    """
    Phase 7E — Local file watcher.

    Polls a directory for new documents and calls the callback
    function for each new file detected.

    C2 compliant — no network calls, fully local.

    Usage:
        def on_new_doc(event):
            pipeline.ingest(event.path)

        watcher = FileWatcher("documents/", callback=on_new_doc)
        watcher.start()
        # ... runs in background thread ...
        watcher.stop()
    """

    def __init__(
        self,
        watch_dir:  str                           = DEFAULT_WATCH_DIR,
        callback:   Optional[Callable]            = None,
        extensions: Optional[Set[str]]            = None,
        poll_sec:   float                         = DEFAULT_POLL_SEC,
        recursive:  bool                          = False,
    ) -> None:
        self.watch_dir  = str(Path(watch_dir).resolve())
        self.callback   = callback
        self.extensions = extensions or SUPPORTED_EXTENSIONS
        self.poll_sec   = poll_sec
        self.recursive  = recursive

        self._seen:     Set[str] = set()
        self._running:  bool     = False
        self._thread            = None
        self._events_queue: List[FileWatcherEvent] = []

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start watching in a background thread."""
        import threading

        if self._running:
            logger.warning("[7E Watcher] Already running")
            return

        os.makedirs(self.watch_dir, exist_ok=True)
        self._running = True

        # Seed the seen set with existing files (don't reprocess on start)
        self._seed_existing()

        self._thread = threading.Thread(
            target = self._poll_loop,
            name   = "FileWatcher",
            daemon = True,
        )
        self._thread.start()
        logger.info(
            "[7E Watcher] Started | dir=%s | extensions=%s | poll=%.1fs",
            self.watch_dir, self.extensions, self.poll_sec,
        )

    def stop(self) -> None:
        """Stop the background watcher thread."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self.poll_sec + 1)
        logger.info("[7E Watcher] Stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Manual scan ───────────────────────────────────────────────────────────

    def scan_once(self) -> List[FileWatcherEvent]:
        """
        Perform a single scan and return new files found.
        Does NOT require start() — useful for testing and
        one-shot batch processing.
        """
        os.makedirs(self.watch_dir, exist_ok=True)
        return self._detect_new_files()

    def scan_and_process(self) -> int:
        """
        Scan once and call callback for each new file.
        Returns count of files processed.
        """
        events = self.scan_once()
        for event in events:
            self._handle_event(event)
        return len(events)

    # ── State ─────────────────────────────────────────────────────────────────

    def get_seen_count(self) -> int:
        """Return number of files already seen/processed."""
        return len(self._seen)

    def reset_seen(self) -> None:
        """Clear the seen set — next scan will reprocess all files."""
        self._seen.clear()

    def add_to_seen(self, path: str) -> None:
        """Manually mark a file as already processed."""
        self._seen.add(str(Path(path).resolve()))

    def get_queued_events(self) -> List[FileWatcherEvent]:
        """Return events queued when no callback is set."""
        return list(self._events_queue)

    def clear_queue(self) -> None:
        """Clear the events queue."""
        self._events_queue.clear()

    # ── Private ───────────────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        """Background polling loop."""
        while self._running:
            try:
                new_files = self._detect_new_files()
                for event in new_files:
                    self._handle_event(event)
            except Exception as exc:
                logger.error("[7E Watcher] Poll error: %s", exc)
            time.sleep(self.poll_sec)

    def _detect_new_files(self) -> List[FileWatcherEvent]:
        """Scan directory for new files not yet in _seen."""
        new_events = []

        try:
            if self.recursive:
                paths = Path(self.watch_dir).rglob("*")
            else:
                paths = Path(self.watch_dir).iterdir()

            for p in paths:
                if not p.is_file():
                    continue
                if p.suffix.lower() not in self.extensions:
                    continue

                resolved = str(p.resolve())
                if resolved in self._seen:
                    continue

                # Wait for file to stabilise (finish writing)
                if not self._is_stable(p):
                    continue

                self._seen.add(resolved)
                event = FileWatcherEvent(str(p))
                new_events.append(event)
                logger.info(
                    "[7E Watcher] New file: %s (%.1f KB)",
                    p.name, event.size_bytes / 1024,
                )

        except PermissionError as exc:
            logger.warning("[7E Watcher] Permission error: %s", exc)

        return new_events

    def _is_stable(self, path: Path) -> bool:
        """
        Check if file has finished writing by comparing size twice.
        Returns True if size is stable over STABLE_WAIT_SEC.
        """
        try:
            size1 = path.stat().st_size
            time.sleep(0.1)
            size2 = path.stat().st_size
            return size1 == size2 and size1 > 0
        except (OSError, FileNotFoundError):
            return False

    def _handle_event(self, event: FileWatcherEvent) -> None:
        """Process a detected file event."""
        if self.callback:
            try:
                self.callback(event)
            except Exception as exc:
                logger.error(
                    "[7E Watcher] Callback error for %s: %s",
                    event.filename, exc,
                )
        else:
            # Queue for later retrieval if no callback set
            self._events_queue.append(event)

    def _seed_existing(self) -> None:
        """
        Mark all currently existing files as already seen.
        Prevents reprocessing files that existed before watcher started.
        """
        try:
            if self.recursive:
                paths = Path(self.watch_dir).rglob("*")
            else:
                paths = Path(self.watch_dir).iterdir()
            for p in paths:
                if p.is_file() and p.suffix.lower() in self.extensions:
                    self._seen.add(str(p.resolve()))
        except Exception:
            pass


# ── Convenience functions ─────────────────────────────────────────────────────

def watch_folder(
    folder:    str,
    callback:  Callable,
    recursive: bool = False,
) -> FileWatcher:
    """
    Start watching a folder and call callback for each new document.

    Args:
        folder   : Directory path to watch
        callback : Function(FileWatcherEvent) called per new file
        recursive: Also watch subdirectories

    Returns:
        Running FileWatcher instance. Call .stop() to stop.
    """
    watcher = FileWatcher(
        watch_dir = folder,
        callback  = callback,
        recursive = recursive,
    )
    watcher.start()
    return watcher