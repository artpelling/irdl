"""Shared IR storage helpers; Dataset metadata and source reading stay with Ingest."""

import os
import zlib
from collections import deque
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py as h5
import numpy as np

from irdl import config
from irdl.logging import logger


def choose_ir_chunk_shape(
    shape: tuple[int, ...], dtype: np.dtype, *, max_bytes: int | None = None
) -> tuple[int, int, int, int]:
    """Choose byte-bounded FIR-E chunks for measurement- or receiver-heavy arrays.

    Chunks contain one measurement and one emitter. Multiple measurements retain
    the draft's receiver grouping; a single measurement prioritizes adjacent samples.
    Both receivers and samples can be divided to keep oversized measurements bounded.
    Storage chunks are capped at 1 MiB and the shared batch budget. ``max_bytes``
    may impose a tighter bound when a check combines multiple input tiles.
    """
    if len(shape) != 4 or any(not isinstance(size, (int, np.integer)) or size <= 0 for size in shape):  # noqa: PLR2004
        msg = "IR shape must contain four positive integer dimensions"
        raise ValueError(msg)
    dtype = np.dtype(dtype)
    if dtype.kind != "f" or dtype.itemsize != np.dtype("f8").itemsize:
        msg = "IR chunks require float64 storage"
        raise ValueError(msg)
    itemsize = dtype.itemsize
    if max_bytes is not None and (not isinstance(max_bytes, int) or max_bytes < itemsize):
        msg = "IR chunk bytes must accommodate at least one stored value"
        raise ValueError(msg)
    batch_bytes = config._get_batch_bytes()
    max_bytes = min(1024**2, batch_bytes, batch_bytes if max_bytes is None else max_bytes)
    if shape[0] > 1:
        receivers = min(shape[1], max_bytes // itemsize)
        samples = min(shape[2], max_bytes // (receivers * itemsize))
    else:
        samples = min(shape[2], max_bytes // itemsize)
        receivers = min(shape[1], max_bytes // (samples * itemsize))
    return 1, receivers, samples, 1


def write_ir_chunks(
    sofa_path: Path, chunks: Iterable[tuple[tuple[int, ...], np.ndarray]], *, workers: int | None = None
) -> None:
    """Fill an existing SOFA Data.IR variable from complete storage chunks.

    Parameters
    ----------
    sofa_path : Path
        Existing, closed SOFA skeleton. Its temporary path is managed by the caller.
    chunks : iterable of (tuple of int, numpy.ndarray)
        Destination offsets and IR arrays in stored dimension order. Arrays cover
        one storage chunk, clipped to the logical array shape at its edges. Supply
        each chunk exactly once in array order, advancing the last dimension first.
    workers : int, optional
        Compression workers, defaulting to at most six CPUs. One encodes inline.
        Only compression runs in workers; source iteration and HDF5 stay on the caller.
    """
    batch_bytes = config._get_batch_bytes()
    workers = min(6, os.cpu_count() or 1) if workers is None else workers
    if not isinstance(workers, int) or workers <= 0:
        msg = "Compression workers must be a positive integer"
        raise ValueError(msg)
    with h5.File(sofa_path, "r+") as sofa:
        ir = _ir_variable(sofa)
        encoding = (ir.chunks, ir.dtype, ir.fillvalue, ir.compression_opts)
        chunk_bytes = int(np.prod(ir.chunks)) * ir.dtype.itemsize
        if chunk_bytes > batch_bytes:
            msg = "Data.IR storage chunks exceed config.batch_bytes; create smaller storage chunks"
            raise ValueError(msg)
        logger.info("IRs: shape=%s, dtype=%s", ir.shape, ir.dtype)
        logger.debug(
            "IR storage: chunks=%s, gzip=%s, workers=%s, batch_bytes=%s",
            ir.chunks,
            ir.compression_opts,
            workers,
            batch_bytes,
        )
        max_pending = min(2 * workers, batch_bytes // chunk_bytes)
        pending = deque()
        with (
            logger.progress("Writing SOFA IRs", total=ir.size) as advance,
            ThreadPoolExecutor(max_workers=workers) as executor,
        ):
            try:
                for offset, values in _checked_chunks(ir, chunks):
                    # Own buffers before the source iterator can reuse them or release a batch.
                    data = np.array(values, dtype=ir.dtype, copy=True, order="C")
                    if workers == 1:
                        ir.id.write_direct_chunk(offset, _encode_ir_chunk(data, *encoding), filter_mask=0)
                        advance(data.size)
                    else:
                        pending.append((offset, data.size, executor.submit(_encode_ir_chunk, data, *encoding)))
                        if len(pending) >= max_pending:
                            destination, size, future = pending.popleft()
                            ir.id.write_direct_chunk(destination, future.result(), filter_mask=0)
                            advance(size)
                while pending:
                    destination, size, future = pending.popleft()
                    ir.id.write_direct_chunk(destination, future.result(), filter_mask=0)
                    advance(size)
            finally:
                for _, _, future in pending:
                    future.cancel()


def _checked_chunks(ir: h5.Dataset, chunks: Iterable[tuple[tuple[int, ...], np.ndarray]]):
    """Check coverage in array order without retaining a list of visited offsets."""
    iterator = iter(chunks)
    for selection in ir.iter_chunks():
        expected_offset = tuple(axis.start for axis in selection)
        try:
            offset, data = next(iterator)
        except StopIteration as error:
            msg = f"Missing IR chunk at {expected_offset}"
            raise ValueError(msg) from error
        if (
            not isinstance(offset, tuple)
            or any(not isinstance(value, (int, np.integer)) for value in offset)
            or offset != expected_offset
        ):
            msg = f"Expected IR chunk at {expected_offset}, got {offset}"
            raise ValueError(msg)
        expected_shape = tuple(axis.stop - axis.start for axis in selection)
        if not isinstance(data, np.ndarray) or data.shape != expected_shape:
            msg = f"IR chunk at {offset} must have shape {expected_shape}"
            raise ValueError(msg)
        if np.ma.isMaskedArray(data) or data.dtype.kind != "f" or not np.can_cast(data.dtype, ir.dtype, casting="safe"):
            msg = f"IR chunk at {offset} cannot be stored losslessly as {ir.dtype}"
            raise ValueError(msg)
        yield offset, data
    sentinel = object()
    if next(iterator, sentinel) is not sentinel:
        msg = "Extra IR chunk after the complete Data.IR array"
        raise ValueError(msg)


def _ir_variable(sofa: h5.File) -> h5.Dataset:
    """Reject unsupported storage before any Provider data is consumed."""
    ir = sofa.get("Data.IR")
    if ir is None:
        msg = "SOFA skeleton must contain a Data.IR variable"
        raise ValueError(msg)
    if not isinstance(ir, h5.Dataset):
        msg = "Data.IR must be an HDF5 dataset"
        raise TypeError(msg)
    creation = ir.id.get_create_plist()
    filters = tuple(creation.get_filter(index)[0] for index in range(creation.get_nfilters()))
    if (
        ir.ndim != 4  # noqa: PLR2004 - FIR-E: M, R, N, E.
        or not all(ir.shape)
        or ir.dtype.kind != "f"
        or ir.dtype.itemsize != np.dtype(np.float64).itemsize
        or ir.chunks is None
        or ir.chunks[0] != 1
        or ir.chunks[-1] != 1
        or filters != (h5.h5z.FILTER_SHUFFLE, h5.h5z.FILTER_DEFLATE)
    ):
        msg = (
            "Data.IR requires nonempty float64 FIR-E storage, single-measurement/emitter chunks, shuffle then gzip; "
            f"actual shape={ir.shape}, dtype={ir.dtype}, chunks={ir.chunks}, filters={filters}"
        )
        raise ValueError(msg)
    return ir


def _encode_ir_chunk(data: np.ndarray, shape: tuple[int, ...], dtype: np.dtype, fill_value: float, level: int) -> bytes:
    """Apply HDF5 byte shuffle and gzip, padding incomplete storage chunks."""
    if data.shape == shape:
        chunk = np.asarray(data, dtype=dtype, order="C")
    else:
        chunk = np.full(shape, fill_value, dtype=dtype)
        chunk[tuple(slice(0, size) for size in data.shape)] = data
    shuffled = chunk.view(np.uint8).reshape(-1, dtype.itemsize).T.copy().tobytes()
    return zlib.compress(shuffled, level)
