"""Tests for measured progress and its interaction with existing spinners."""

import io
import logging
from contextlib import nullcontext
from typing import ClassVar

import pooch
import pytest
import requests
from rich.console import Console

import irdl.logging as irdl_logging


@pytest.fixture
def displays(monkeypatch):
    """Capture real Rich tasks without writing to the terminal."""
    captured = []
    progress_class = irdl_logging.Progress

    def create(*args, **kwargs):
        display = progress_class(*args, **kwargs)
        captured.append(display)
        return display

    monkeypatch.setattr(irdl_logging, "console", Console(file=io.StringIO()))
    monkeypatch.setattr(irdl_logging, "Progress", create)
    return captured


@pytest.mark.parametrize("fail", [False, True])
def test_progress_reports_actual_work_and_stops_on_failure(displays, fail):
    """Errors close the display without falsely filling it to 100 percent."""
    expected = 5
    handling = pytest.raises(ValueError, match="failed") if fail else nullcontext()
    with handling, irdl_logging.logger.progress("Checking IRs", total=expected) as advance:
        advance(2)
        assert displays[0].tasks[0].completed == 2  # noqa: PLR2004
        if fail:
            msg = "failed"
            raise ValueError(msg)
        advance(3)

    assert displays[0].tasks[0].completed == (2 if fail else expected)
    assert not displays[0].live.is_started


@pytest.mark.parametrize("fail", [False, True])
def test_progress_pauses_and_restores_outer_spinner(monkeypatch, displays, fail):
    """A progress bar never competes with the outer Ingest spinner's live display."""
    events = []

    class Status:
        def start(self):
            events.append("start")

        def stop(self):
            events.append("stop")

        def __enter__(self):
            self.start()
            return self

        def __exit__(self, *_args):
            self.stop()

    monkeypatch.setattr(irdl_logging.console, "status", lambda _message: Status())
    with irdl_logging.logger.spin("Writing SOFA"):
        handling = pytest.raises(ValueError, match="failed") if fail else nullcontext()
        with handling, irdl_logging.logger.progress("Writing IRs", total=1) as advance:
            assert events == ["start", "stop"]
            if fail:
                msg = "failed"
                raise ValueError(msg)
            advance(1)
        assert events == ["start", "stop", "start"]

    assert events == ["start", "stop", "start", "stop"]
    with irdl_logging.logger.progress("Checking IRs", total=1) as advance:
        advance(1)
    assert events == ["start", "stop", "start", "stop"]
    assert all(not display.live.is_started for display in displays)


def test_error_level_silences_forwarded_info_and_progress(caplog, capsys):
    """One logger level controls IRDL, forwarded records, and progress displays."""
    with caplog.at_level(logging.ERROR, logger="irdl"):
        pooch.get_logger().info("hidden upstream message")
        with irdl_logging.logger.progress("hidden progress", total=1) as advance:
            advance(1)
        with irdl_logging.logger.spin("hidden spinner"):
            pass
        download = irdl_logging.RichProgressBar("hidden download", preset_total=1)
        download.update(1)
        download.close()
    assert not caplog.records
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_download_progress_close_preserves_partial_completion(displays):
    """Stopping a failed transfer must not invent successful work."""
    bar = irdl_logging.RichProgressBar("partial.sofa", preset_total=100)
    bar.update(25)
    bar.close()
    assert displays[0].tasks[0].completed == 25  # noqa: PLR2004
    assert not displays[0].live.is_started


def test_download_retry_starts_with_zero_completed_bytes(displays):
    """A new response header must not retain bytes from an interrupted attempt."""
    bar = irdl_logging.RichProgressBar("retry.sofa", preset_total=100)
    bar.update(25)
    bar.total = 100
    assert displays[0].tasks[0].completed == 0
    bar.update(10)
    bar.close()
    assert displays[0].tasks[0].completed == 10  # noqa: PLR2004


def test_failed_fetch_stops_download_progress(monkeypatch, displays, tmp_path):
    """Exercise cleanup through the real Pooch HTTP failure path."""
    from irdl.downloader import _fetch  # noqa: PLC0415

    class Response:
        headers: ClassVar[dict[str, str]] = {"content-length": "4096"}

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b"x" * chunk_size
            msg = "interrupted transfer"
            raise requests.ConnectionError(msg)

        def close(self):
            pass

    monkeypatch.setattr(requests, "get", lambda *_args, **_kwargs: Response())
    pup = pooch.create(
        path=tmp_path,
        base_url="https://example.invalid/",
        registry={"partial.sofa": None},
        retry_if_failed=0,
    )
    try:
        with pytest.raises(requests.ConnectionError, match="interrupted"):
            _fetch(pup, "partial.sofa")
        assert not displays[0].live.is_started
        assert displays[0].tasks[0].completed < displays[0].tasks[0].total
    finally:
        for display in displays:
            display.stop()


def test_forwarding_preserves_upstream_record_and_honors_levels(caplog):
    """Attribution must not mutate a record owned by another package."""
    target = irdl_logging.get_logger("pooch")
    record = logging.LogRecord("upstream", logging.INFO, __file__, 1, "message %s", ("body",), None)
    forwarder = irdl_logging.LoggerForwarder(target)
    with caplog.at_level(logging.INFO, logger="irdl"):
        forwarder.emit(record)
    assert record.name == "upstream"
    assert caplog.records[-1].name == "irdl.pooch"
    assert caplog.messages[-1] == "message body"


def test_context_labels_and_captured_output_are_source_aware(caplog):
    """Dataset selection follows IRDL and third-party records and is then reset."""
    with caplog.at_level(logging.INFO, logger="irdl"):
        with irdl_logging.log_context("miracle", "A1-C2"):
            irdl_logging.logger.info("Ingest started")
            with irdl_logging.capture_stdout(irdl_logging.get_logger("sofar")):
                print("first line\nsecond line")  # noqa: T201
        irdl_logging.logger.info("Outside Get")
    records = caplog.records[-4:]
    assert [record.source for record in records] == ["MIRACLE", "SOFAR", "SOFAR", "IRDL"]
    assert [record.selection for record in records] == ["A1-C2", "A1-C2", "A1-C2", ""]
    assert records[1].dataset == "MIRACLE"
    assert records[-1].dataset == ""


def test_logger_lookup_and_cli_setup_preserve_application_configuration():
    """Obtaining a logger must not reset caller levels or discard handlers."""
    target = irdl_logging.get_logger("sofar")
    custom = logging.NullHandler()
    target.addHandler(custom)
    old_target_level, old_level = target.level, irdl_logging.logger.level
    target.setLevel(logging.ERROR)
    irdl_logging.logger.setLevel(logging.WARNING)
    try:
        assert irdl_logging.get_logger("sofar") is target
        irdl_logging.configure_cli_logging()
        assert custom in target.handlers
        assert target.level == logging.ERROR
        assert irdl_logging.logger.level == logging.WARNING
    finally:
        target.removeHandler(custom)
        target.setLevel(old_target_level)
        irdl_logging.logger.setLevel(old_level)
