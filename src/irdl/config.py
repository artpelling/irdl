"""Package-wide settings for streamed Ingest, payload checks, and HDF5 export.

Set ``irdl.config.batch_bytes`` before starting a Get operation. The default is
64 MiB, initialized from ``IRDL_BATCH_MIB`` when the package is imported.
This controls array batches, not total process memory or in-memory Output Formats.
"""

import os
from contextlib import contextmanager
from contextvars import ContextVar

try:
    batch_bytes = int(os.environ.get("IRDL_BATCH_MIB", "64")) * 1024**2
except ValueError as error:
    msg = "IRDL_BATCH_MIB must be a positive integer number of MiB"
    raise ValueError(msg) from error
if batch_bytes <= 0:
    msg = "IRDL_BATCH_MIB must be a positive integer number of MiB"
    raise ValueError(msg)
_operation_bytes: ContextVar[int | None] = ContextVar("irdl_batch_bytes", default=None)


def _get_batch_bytes() -> int:
    """Read and validate the shared budget without copying a mutable setting."""
    size = _operation_bytes.get()
    if size is None:
        size = batch_bytes
    if isinstance(size, bool) or not isinstance(size, int) or size < 8:  # noqa: PLR2004
        msg = "config.batch_bytes must be an integer of at least 8 bytes (one float64 value)"
        raise ValueError(msg)
    return size


@contextmanager
def _batch_budget():
    """Keep the starting budget for this Get and restore it even after a failure."""
    token = _operation_bytes.set(_get_batch_bytes())
    try:
        yield
    finally:
        _operation_bytes.reset(token)
