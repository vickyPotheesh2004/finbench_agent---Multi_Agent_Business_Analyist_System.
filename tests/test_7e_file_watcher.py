"""
tests/test_7e_file_watcher.py
Tests for Phase 7E File Watcher
PDR-BAAAI-001 Rev 1.0
25 tests - no network needed (fully local)
"""
import os
import time
import pytest
from pathlib import Path
from src.live_data.watcher import (
    FileWatcher,
    FileWatcherEvent,
    watch_folder,
    SUPPORTED_EXTENSIONS,
    DEFAULT_WATCH_DIR,
    DEFAULT_POLL_SEC,
    SEED,
)


@pytest.fixture
def watch_dir(tmp_path):
    d = tmp_path / "watch"
    d.mkdir()
    return str(d)


@pytest.fixture
def watcher(watch_dir):
    w = FileWatcher(watch_dir=watch_dir)
    yield w
    if w.is_running:
        w.stop()


def _make_file(directory: str, name: str, content: str = "test content") -> str:
    path = Path(directory) / name
    path.write_text(content, encoding="utf-8")
    return str(path)


class TestConstants:

    def test_01_supported_extensions_defined(self):
        assert ".pdf"  in SUPPORTED_EXTENSIONS
        assert ".docx" in SUPPORTED_EXTENSIONS
        assert ".xlsx" in SUPPORTED_EXTENSIONS

    def test_02_default_poll_sec_defined(self):
        assert DEFAULT_POLL_SEC > 0

    def test_03_seed_is_42(self):
        assert SEED == 42


class TestFileWatcherEvent:

    def test_04_event_creates_from_path(self, tmp_path):
        f = tmp_path / "test.pdf"
        f.write_text("content")
        e = FileWatcherEvent(str(f))
        assert e.filename  == "test.pdf"
        assert e.extension == ".pdf"

    def test_05_event_to_dict_has_keys(self, tmp_path):
        f = tmp_path / "report.docx"
        f.write_text("content")
        d = FileWatcherEvent(str(f)).to_dict()
        assert "path"        in d
        assert "filename"    in d
        assert "extension"   in d
        assert "size_bytes"  in d
        assert "detected_at" in d

    def test_06_event_size_bytes_correct(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b,c")
        e = FileWatcherEvent(str(f))
        assert e.size_bytes == f.stat().st_size

    def test_07_event_repr_contains_filename(self, tmp_path):
        f = tmp_path / "apple.pdf"
        f.write_text("content")
        assert "apple.pdf" in repr(FileWatcherEvent(str(f)))


class TestInstantiation:

    def test_08_instantiates_with_defaults(self, watcher):
        assert watcher is not None

    def test_09_not_running_on_init(self, watcher):
        assert watcher.is_running is False

    def test_10_seen_count_starts_zero(self, watcher):
        assert watcher.get_seen_count() == 0

    def test_11_custom_extensions(self, watch_dir):
        w = FileWatcher(watch_dir=watch_dir, extensions={".pdf"})
        assert ".pdf" in w.extensions
        assert ".xlsx" not in w.extensions


class TestScanOnce:

    def test_12_scan_empty_dir_returns_empty(self, watcher):
        results = watcher.scan_once()
        assert results == []

    def test_13_scan_detects_pdf(self, watcher, watch_dir):
        _make_file(watch_dir, "report.pdf", "PDF content here")
        results = watcher.scan_once()
        assert len(results) == 1
        assert results[0].filename == "report.pdf"

    def test_14_scan_detects_multiple_files(self, watcher, watch_dir):
        _make_file(watch_dir, "a.pdf",  "pdf content")
        _make_file(watch_dir, "b.docx", "docx content")
        _make_file(watch_dir, "c.xlsx", "xlsx content")
        results = watcher.scan_once()
        assert len(results) == 3

    def test_15_scan_ignores_unsupported(self, watcher, watch_dir):
        _make_file(watch_dir, "ignore.exe", "binary")
        _make_file(watch_dir, "keep.pdf",   "pdf content")
        results = watcher.scan_once()
        names = [r.filename for r in results]
        assert "keep.pdf"   in names
        assert "ignore.exe" not in names

    def test_16_scan_twice_no_duplicates(self, watcher, watch_dir):
        _make_file(watch_dir, "doc.pdf", "content")
        first  = watcher.scan_once()
        second = watcher.scan_once()
        assert len(first)  == 1
        assert len(second) == 0  # already seen

    def test_17_seen_count_increments(self, watcher, watch_dir):
        _make_file(watch_dir, "doc1.pdf", "content1")
        _make_file(watch_dir, "doc2.pdf", "content2")
        watcher.scan_once()
        assert watcher.get_seen_count() == 2

    def test_18_reset_seen_allows_rescan(self, watcher, watch_dir):
        _make_file(watch_dir, "doc.pdf", "content")
        watcher.scan_once()
        assert watcher.get_seen_count() == 1
        watcher.reset_seen()
        assert watcher.get_seen_count() == 0
        results = watcher.scan_once()
        assert len(results) == 1


class TestCallback:

    def test_19_callback_called_on_new_file(self, watch_dir):
        received = []
        def cb(event): received.append(event)
        w = FileWatcher(watch_dir=watch_dir, callback=cb)
        _make_file(watch_dir, "new.pdf", "content")
        w.scan_and_process()
        assert len(received) == 1
        assert received[0].filename == "new.pdf"

    def test_20_no_callback_queues_events(self, watcher, watch_dir):
        _make_file(watch_dir, "queued.pdf", "content")
        watcher.scan_and_process()
        events = watcher.get_queued_events()
        assert len(events) == 1
        assert events[0].filename == "queued.pdf"

    def test_21_clear_queue_empties_events(self, watcher, watch_dir):
        _make_file(watch_dir, "q.pdf", "content")
        watcher.scan_and_process()
        watcher.clear_queue()
        assert watcher.get_queued_events() == []

    def test_22_callback_error_does_not_crash(self, watch_dir):
        def bad_cb(event): raise RuntimeError("callback error")
        w = FileWatcher(watch_dir=watch_dir, callback=bad_cb)
        _make_file(watch_dir, "err.pdf", "content")
        w.scan_and_process()  # should not raise


class TestStartStop:

    def test_23_start_sets_running(self, watcher):
        watcher.start()
        assert watcher.is_running is True
        watcher.stop()

    def test_24_stop_sets_not_running(self, watcher):
        watcher.start()
        watcher.stop()
        assert watcher.is_running is False


class TestConvenienceWrapper:

    def test_25_watch_folder_returns_watcher(self, watch_dir):
        w = watch_folder(watch_dir, callback=lambda e: None)
        assert isinstance(w, FileWatcher)
        assert w.is_running is True
        w.stop()