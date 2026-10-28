"""Consistent, contextual diagnostics and progress displays for IRDL."""

import io
import logging
from contextlib import contextmanager, redirect_stdout
from contextvars import ContextVar
from copy import copy

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.text import Text

console = Console(stderr=True)
_active_spinner = ContextVar("irdl_active_spinner", default=None)
_context: ContextVar[tuple[str, str] | None] = ContextVar("irdl_log_context", default=None)
_SOURCE_WIDTH = 10
_SOURCE_STYLES = {"IRDL": "bold blue", "POOCH": "bold yellow", "SOFAR": "bold magenta"}


class _ContextFilter(logging.Filter):
    """Attach the active Dataset and selection to ordinary logging records."""

    def filter(self, record: logging.LogRecord) -> bool:
        context = _context.get()
        record.dataset, record.selection = context if context is not None else ("", "")
        record.source = (record.dataset or "IRDL") if record.name == "irdl" else record.name.rsplit(".", 1)[-1].upper()
        return True


_context_filter = _ContextFilter()


def _selection_label(source: str) -> str:
    context = _context.get()
    if context is None:
        return ""
    dataset, selection = context
    return selection if source == dataset else f"{dataset} {selection}".strip()


class IrdlRichHandler(RichHandler):
    """Render a readable source column without changing logger names or messages."""

    def render_message(self, record: logging.LogRecord, message: str) -> Text:
        """Keep Dataset and third-party output visibly distinct."""
        source = record.source
        style = _SOURCE_STYLES.get(source, "bold cyan")
        label = record.selection if source == record.dataset else f"{record.dataset} {record.selection}".strip()
        return Text.assemble(
            (f"{source:<{_SOURCE_WIDTH}}", style),
            " ",
            (f"[{label}] " if label else "", "dim"),
            super().render_message(record, message),
        )


logger = logging.getLogger("irdl")
rich_handler = IrdlRichHandler(console=console, show_time=False, show_path=False)
rich_handler.setFormatter(logging.Formatter("%(message)s"))
rich_handler.addFilter(_context_filter)
logger.addHandler(rich_handler)
logger.addFilter(_context_filter)
if logger.level == logging.NOTSET:
    logger.setLevel(logging.INFO)


def get_logger(source: str) -> logging.Logger:
    """Get an IRDL child logger without resetting application levels or handlers."""
    child = logger.getChild(source.lower())
    if _context_filter not in child.filters:
        child.addFilter(_context_filter)
    return child


@contextmanager
def log_context(dataset: str, selection: str):
    """Scope Dataset identity to one Get, including its third-party diagnostics."""
    token = _context.set((dataset.upper(), selection))
    try:
        yield
    finally:
        _context.reset(token)


@contextmanager
def capture_stdout(target: logging.Logger, level: int = logging.INFO):
    """Attribute printed library output to its source, one line per record."""
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer):
            yield
    finally:
        for line in buffer.getvalue().splitlines():
            if line.strip():
                target.log(level, "%s", line)


@contextmanager
def spin(message: str):
    """Show a spinner for unmeasurable work when INFO output is enabled."""
    if not logger.isEnabledFor(logging.INFO):
        yield
        return
    source = (_context.get() or ("IRDL", ""))[0]
    description = Text.assemble((source, _SOURCE_STYLES.get(source, "bold cyan")), " ", _description(message, source))
    with console.status(description) as status:
        token = _active_spinner.set(status)
        try:
            yield
        finally:
            _active_spinner.reset(token)


def _progress_columns(source: str):
    """Align progress with the log source and message columns."""
    style = _SOURCE_STYLES.get(source, "bold cyan")
    return (
        TextColumn(f"{' ' * 9}{source:<{_SOURCE_WIDTH}}", style=style),
        TextColumn("{task.description}", markup=False),
        BarColumn(complete_style=style, finished_style=style),
    )


def _description(message: str, source: str) -> str:
    label = _selection_label(source)
    return f"[{label}] {message}" if label else message


@contextmanager
def progress(message: str, total: int):
    """Show measured work, pausing any surrounding spinner until the bar closes."""
    if not logger.isEnabledFor(logging.INFO):
        yield lambda _amount: None
        return
    status = _active_spinner.get()
    if status is not None:
        status.stop()
    source = (_context.get() or ("IRDL", ""))[0]
    try:
        with Progress(
            *_progress_columns(source), TaskProgressColumn(), TimeRemainingColumn(), console=console
        ) as display:
            task = display.add_task(_description(message, source), total=total)
            yield lambda amount: display.advance(task, amount)
    finally:
        if status is not None:
            status.start()


logger.spin = spin
logger.progress = progress


def configure_cli_logging() -> logging.Logger:
    """Use the shared CLI display without overriding caller verbosity settings."""
    return logger


class LoggerForwarder(logging.Handler):
    """Forward upstream records without mutating them or bypassing target levels."""

    def __init__(self, target_logger: logging.Logger) -> None:
        super().__init__()
        self.target_logger = target_logger

    def emit(self, record: logging.LogRecord) -> None:
        """Apply the IRDL logger hierarchy's verbosity before forwarding a copy."""
        if self.target_logger.isEnabledFor(record.levelno):
            forwarded = copy(record)
            forwarded.name = self.target_logger.name
            self.target_logger.handle(forwarded)


# Pooch installs a plain, unformatted StreamHandler itself. Replace that default,
# not additional/customized application handlers.
try:
    import pooch as po

    upstream_pooch_logger = po.get_logger()
    if upstream_pooch_logger.handlers:
        default = upstream_pooch_logger.handlers[0]
        if (
            type(default) is logging.StreamHandler
            and default.formatter is None
            and default.level == logging.NOTSET
            and not default.filters
            and default.name is None
        ):
            upstream_pooch_logger.removeHandler(default)
    upstream_pooch_logger.addHandler(LoggerForwarder(get_logger("pooch")))
    upstream_pooch_logger.propagate = False
    if upstream_pooch_logger.level == logging.NOTSET:
        upstream_pooch_logger.setLevel(logging.DEBUG)
except ImportError:
    pass


class RichProgressBar:
    """Implement Pooch's progress interface without inventing completion on close."""

    def __init__(self, description: str, preset_total: int = 0):
        self._progress = Progress(
            *_progress_columns("POOCH"),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
            disable=not logger.isEnabledFor(logging.INFO),
        )
        self._description = _description(description, "POOCH")
        self._task_id = None
        self._preset_total = preset_total
        self.total = 0

    @property
    def total(self) -> int:
        """Total bytes, falling back to the repository's size if HTTP omits it."""
        return self._total

    @total.setter
    def total(self, value: int) -> None:
        self._total = value or self._preset_total
        if self._task_id is not None:
            # A repeated HTTP header starts a retry, not additional bytes of the old transfer.
            self._progress.reset(self._task_id, total=self._total or None)

    def update(self, n: int) -> None:
        """Advance by the bytes reported by Pooch."""
        if self._task_id is None:
            self._progress.start()
            self._task_id = self._progress.add_task(self._description, total=self.total or None)
        self._progress.advance(self._task_id, n)

    def reset(self) -> None:
        """Reset completion before Pooch supplies the final byte count."""
        if self._task_id is not None:
            self._progress.reset(self._task_id, total=self.total or None)

    def close(self) -> None:
        """Stop the display, retaining actual completion even after a failure."""
        if self._task_id is not None:
            self._progress.stop()
            self._task_id = None
