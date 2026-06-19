"""Logging configuration for IRDL."""

import io
import logging
import sys

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.text import Text

console = Console()

_LOGGER_STYLES = {
    "IRDL": "bold blue",
    "POOCH": "bold yellow",
    "SOFAR": "bold red",
}
_DEFAULT_SOURCE_STYLE = "bold white"
_LEVEL_WIDTH = 8


def _source_width() -> int:
    """Return the current source-column width."""
    return max(len(name) for name in _LOGGER_STYLES | {"DEFAULT": _DEFAULT_SOURCE_STYLE})


def _log_message_indent() -> str:
    """Return the left padding needed to align with log message text."""
    return " " * (_LEVEL_WIDTH + _source_width() + 1)


class IrdlRichHandler(RichHandler):
    """Rich handler with a colorized source prefix."""

    def render_message(self, record: logging.LogRecord, message: str) -> Text:
        """Render one log message with a styled source name."""
        rendered = super().render_message(record, message)
        style = _LOGGER_STYLES.get(record.name, _DEFAULT_SOURCE_STYLE)
        source = f"{record.name:<{_source_width()}}"
        return Text.assemble((source, style), " ", rendered)


_rich_handler = IrdlRichHandler(
    console=console,
    show_time=False,
    show_level=True,
    show_path=False,
)
_rich_handler.setFormatter(logging.Formatter("%(message)s"))


class StdoutCapture:
    """Context manager to capture stdout and forward it to a logger."""

    def __init__(self, logger_instance: logging.Logger):
        self.logger = logger_instance

    def __enter__(self) -> io.StringIO:
        """Enter the context manager and start capturing stdout."""
        self.old_stdout = sys.stdout
        self.capture_buffer = io.StringIO()
        sys.stdout = self.capture_buffer
        return self.capture_buffer

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager and restore stdout."""
        sys.stdout = self.old_stdout
        output = self.capture_buffer.getvalue().strip()
        if output:
            for line in output.splitlines():
                self.logger.info(line)


def get_logger(source: str, style: str | None = None) -> logging.Logger:
    """Return a configured logger for one visible log source."""
    source = source.upper()
    if style is not None:
        _LOGGER_STYLES[source] = style
    named_logger = logging.getLogger(source)
    if not any(handler is _rich_handler for handler in named_logger.handlers):
        named_logger.handlers.clear()
        named_logger.addHandler(_rich_handler)
    named_logger.propagate = False
    named_logger.setLevel(logging.INFO)
    named_logger.as_stdout = StdoutCapture(named_logger)
    return named_logger


logger = get_logger("IRDL")
pooch_logger = get_logger("POOCH")
sofar_logger = get_logger("SOFAR")


def configure_cli_logging() -> logging.Logger:
    """Configure logging for CLI usage with Rich handler."""
    logger.setLevel(logging.DEBUG)
    pooch_logger.setLevel(logging.DEBUG)
    sofar_logger.setLevel(logging.DEBUG)
    return logger


try:
    import pooch as po

    upstream_pooch_logger = po.get_logger()

    for handler in upstream_pooch_logger.handlers[:]:
        upstream_pooch_logger.removeHandler(handler)

    class LoggerForwarder(logging.Handler):
        """Forward log records to a target logger."""

        def __init__(self, target_logger: logging.Logger) -> None:
            super().__init__()
            self.target_logger = target_logger

        def emit(self, record: logging.LogRecord) -> None:
            """Forward log records to the target logger."""
            forwarded = logging.LogRecord(
                name=self.target_logger.name,
                level=record.levelno,
                pathname=record.pathname,
                lineno=record.lineno,
                msg=record.getMessage(),
                args=(),
                exc_info=record.exc_info,
            )
            self.target_logger.handle(forwarded)

    upstream_pooch_logger.addHandler(LoggerForwarder(pooch_logger))
    upstream_pooch_logger.propagate = False
    upstream_pooch_logger.setLevel(logging.DEBUG)
except ImportError:
    pass


class RichProgressBar:
    """Wrap rich.progress.Progress to satisfy the pooch progress bar interface.

    Pooch expects an object with a ``total`` attribute and ``update``, ``reset``, and
    ``close`` methods. This class provides that interface backed by a Rich progress bar.
    """

    def __init__(self, description: str, preset_total: int = 0):
        source = f"{'IRDL':<{_source_width()}}"
        irld_style = _LOGGER_STYLES["IRDL"]
        self._progress = Progress(
            TextColumn(f" {' ' * _LEVEL_WIDTH}[{irld_style}]{source}[/] [progress.description]{{task.description}}"),
            BarColumn(complete_style=irld_style, finished_style=irld_style),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
        )
        self._description = description
        self._task_id = None
        self._preset_total = preset_total
        self.total = 0

    @property
    def total(self) -> int:
        """Total download size in bytes."""
        return self._total

    @total.setter
    def total(self, value: int) -> None:
        self._total = value or self._preset_total
        if self._task_id is not None:
            self._progress.update(self._task_id, total=self._total or None)

    def update(self, n: int) -> None:
        """Advance the progress bar by n bytes."""
        if self._task_id is None:
            self._progress.start()
            self._task_id = self._progress.add_task(self._description, total=self.total or None)
        self._progress.advance(self._task_id, n)

    def reset(self) -> None:
        """Reset the completed byte count to zero."""
        if self._task_id is not None:
            self._progress.reset(self._task_id, total=self.total or None)

    def close(self) -> None:
        """Fill to 100% and stop the progress display."""
        if self._task_id is not None:
            if self.total:
                self._progress.update(self._task_id, completed=self.total)
            self._progress.stop()
            self._task_id = None
