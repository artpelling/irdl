"""Tests for writing IR chunks into an existing SOFA skeleton."""

import weakref
import zlib
from itertools import product
from threading import get_ident

import h5py as h5
import netCDF4
import numpy as np
import pytest

from irdl import _sofa, config
from irdl._sofa import choose_ir_chunk_shape, write_ir_chunks

_SAMPLING_RATE = 48000


def _skeleton(path, shape, chunks, *, dtype="f8", endian="native", **kwargs):
    """Create a small NetCDF4 skeleton with metadata the writer must leave alone."""
    with netCDF4.Dataset(path, "w", format="NETCDF4") as sofa:
        for name, size in zip(("M", "R", "N", "E"), shape, strict=True):
            sofa.createDimension(name, size)
        sofa.createDimension("I", 1)
        sofa.Conventions = "SOFA"
        sofa.Title = "Unchanged Dataset metadata"
        sampling_rate = sofa.createVariable("Data.SamplingRate", "f8", ("I",))
        sampling_rate.Units = "hertz"
        sampling_rate[:] = _SAMPLING_RATE
        sofa.createVariable(
            "Data.IR",
            dtype,
            ("M", "R", "N", "E"),
            chunksizes=chunks,
            endian=endian,
            fill_value=-12345.0,
            zlib=True,
            complevel=4,
            **kwargs,
        )


def _chunks(data, chunk_shape):
    """Produce complete chunks, including their clipped logical edges."""
    for offset in product(*(range(0, size, step) for size, step in zip(data.shape, chunk_shape, strict=True))):
        selection = tuple(slice(start, start + step) for start, step in zip(offset, chunk_shape, strict=True))
        yield offset, data[selection]


def test_writer_preserves_ir_values_and_existing_metadata(tmp_path):
    """Write IR values exactly without recreating the skeleton or its metadata."""
    shape = (2, 3, 7, 1)
    chunks = (1, 2, 4, 1)
    expected = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    expected.flat[1] = np.nextafter(1.0, 2.0)
    path = tmp_path / "streamed.sofa"
    _skeleton(path, shape, chunks)

    write_ir_chunks(path, _chunks(expected, chunks))

    with h5.File(path) as sofa:
        np.testing.assert_array_equal(sofa["Data.IR"][:], expected)
        assert sofa["Data.IR"].chunks == chunks
    with netCDF4.Dataset(path) as sofa:
        np.testing.assert_array_equal(sofa.variables["Data.IR"][:], expected)
        assert sofa.Title == "Unchanged Dataset metadata"
        assert sofa.variables["Data.SamplingRate"][0] == _SAMPLING_RATE
        assert sofa.variables["Data.SamplingRate"].Units == "hertz"


@pytest.mark.parametrize(
    "options",
    [
        {"name": "not-ir"},
        {"dtype": "f4"},
        {"dtype": "i8"},
        {"shape": (2, 3, 7), "chunks": (1, 2, 4)},
        {"shape": (0, 3, 7, 1), "maxshape": (None, 3, 7, 1)},
        {"chunks": (2, 2, 4, 1)},
        {"shape": (2, 3, 7, 2), "chunks": (1, 2, 4, 2)},
        {"shuffle": False},
        {"compression": None},
        {"chunks": None, "compression": None, "shuffle": False},
        {"compression": "lzf"},
        {"fletcher32": True},
    ],
)
def test_writer_rejects_unsupported_storage_before_reading_source(tmp_path, options):
    """Unsupported layouts fail without consuming Provider data."""
    settings = {
        "name": "Data.IR",
        "shape": (2, 3, 7, 1),
        "chunks": (1, 2, 4, 1),
        "dtype": "f8",
        "compression": "gzip",
        "shuffle": True,
    }
    settings.update(options)
    name = settings.pop("name")
    path = tmp_path / "unsupported.sofa"
    with h5.File(path, "w") as sofa:
        sofa.create_dataset(name, **settings)

    def unread_source():
        pytest.fail("Source was consumed before storage validation")
        yield  # pragma: no cover

    with pytest.raises(ValueError, match=r"Data\.IR"):
        write_ir_chunks(path, unread_source())
    with h5.File(path, "a") as sofa:
        sofa.attrs["closed_after_failure"] = True


def test_writer_rejects_storage_chunks_larger_than_the_global_budget_before_reading(tmp_path, monkeypatch):
    """Raw-chunk writing cannot split an existing oversized storage chunk safely."""
    path = tmp_path / "oversized.sofa"
    _skeleton(path, (2, 1, 16, 1), (1, 1, 16, 1))
    monkeypatch.setattr(config, "batch_bytes", 64)

    def unread_source():
        pytest.fail("source consumed before checking the budget")
        yield  # pragma: no cover

    with pytest.raises(ValueError, match="batch_bytes"):
        write_ir_chunks(path, unread_source())


def test_writer_rejects_a_group_in_place_of_the_ir_variable(tmp_path):
    """A named group is not a writable numeric IR variable."""
    path = tmp_path / "not-a-variable.sofa"
    with h5.File(path, "w") as sofa:
        sofa.create_group("Data.IR")

    with pytest.raises(TypeError, match="HDF5 dataset"):
        write_ir_chunks(path, ())


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "duplicate",
        "reordered",
        "extra",
        "negative",
        "unaligned",
        "outside",
        "fractional",
        "wrong_dimensions",
        "too_small",
        "edge_too_large",
        "complex",
        "masked",
    ],
)
def test_writer_rejects_invalid_or_incomplete_chunks(tmp_path, fault):
    """Every logical storage chunk must be supplied exactly once, in order."""
    shape = (2, 3, 7, 1)
    chunk_shape = (1, 2, 4, 1)
    expected = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    chunks = list(_chunks(expected, chunk_shape))
    data = chunks[0][1]
    if fault == "missing":
        chunks.pop()
    elif fault == "duplicate":
        chunks.insert(1, chunks[0])
    elif fault == "reordered":
        chunks[0], chunks[1] = chunks[1], chunks[0]
    elif fault == "extra":
        chunks.append(chunks[-1])
    elif fault in ("negative", "unaligned", "outside", "fractional"):
        offsets = {
            "negative": (-1, 0, 0, 0),
            "unaligned": (0, 1, 0, 0),
            "outside": (2, 0, 0, 0),
            "fractional": (0.0, 0, 0, 0),
        }
        chunks[0] = offsets[fault], data
    else:
        invalid_data = {
            "wrong_dimensions": data[0],
            "too_small": data[:, :, :1, :],
            "edge_too_large": np.zeros(chunk_shape),
            "complex": data.astype(complex),
            "masked": np.ma.array(data, mask=True),
        }
        index = -1 if fault == "edge_too_large" else 0
        chunks[index] = chunks[index][0], invalid_data[fault]
    path = tmp_path / "invalid-chunks.sofa"
    _skeleton(path, shape, chunk_shape)

    with pytest.raises(ValueError, match="IR chunk"):
        write_ir_chunks(path, iter(chunks))


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("dtype", ["<f8", ">f8"])
def test_writer_preserves_storage_bytes_and_edge_padding(tmp_path, progress_tasks, workers, dtype):
    """Serial and parallel encoders preserve precision, byte order, and fill bytes."""
    shape = (1, 5, 9, 1)
    chunk_shape = (1, 2, 4, 1)
    expected = np.linspace(-1, 1, np.prod(shape)).reshape(shape)
    expected.flat[:6] = [np.nan, np.inf, -np.inf, -0.0, np.nextafter(1.0, 2.0), np.nextafter(0.0, 1.0)]
    path = tmp_path / "edge.sofa"
    _skeleton(path, shape, chunk_shape, dtype=dtype, endian="little" if dtype == "<f8" else "big")

    write_ir_chunks(path, _chunks(expected, chunk_shape), workers=workers)

    assert len(progress_tasks) == 1
    assert progress_tasks[0]["total"] == expected.size
    assert progress_tasks[0]["completed"] == expected.size
    with netCDF4.Dataset(path) as sofa:
        sofa.set_auto_mask(False)
        actual = sofa.variables["Data.IR"][:].astype("f8")
        assert actual.tobytes() == expected.tobytes()
    with h5.File(path) as sofa:
        ir = sofa["Data.IR"]
        mask, encoded = ir.id.read_direct_chunk((0, 4, 8, 0))
        assert mask == 0
        shuffled = np.frombuffer(zlib.decompress(encoded), dtype=np.uint8)
        decoded = shuffled.reshape(8, -1).T.copy().view(dtype).reshape(chunk_shape)
        padded = np.full(chunk_shape, -12345.0, dtype=dtype)
        padded[:, :1, :1, :] = expected[:, 4:5, 8:9, :]
        assert decoded.tobytes() == padded.tobytes()


@pytest.mark.parametrize("shape", [(144, 8, 16384, 1), (1, 8592, 48000, 1), (1, 2, 200000, 1), (2, 1, 1, 1)])
def test_chunk_shape_bounds_bytes_even_for_large_measurements(shape):
    """A single measurement may need receiver or sample chunks, not a whole row."""
    chunks = choose_ir_chunk_shape(shape, np.dtype("f8"))

    assert chunks[0] == chunks[-1] == 1
    assert all(0 < step <= size for step, size in zip(chunks, shape, strict=True))
    assert np.prod(chunks) * 8 <= 1024**2
    if shape[0] == 1 and shape[1] > 1:
        assert chunks[1] < shape[1]


@pytest.mark.parametrize("budget", [8, 64, 2 * 1024**2])
def test_storage_chunks_follow_the_global_budget_without_becoming_huge(monkeypatch, budget):
    """One package setting caps storage chunks; larger budgets retain the automatic 1 MiB cap."""
    monkeypatch.setattr(config, "batch_bytes", budget)
    chunks = choose_ir_chunk_shape((1, 3072, 4096, 1), np.dtype("f8"))
    assert np.prod(chunks) * 8 <= min(budget, 1024**2)


def test_measurement_chunks_keep_receivers_together_like_the_optimized_draft():
    """Retain the draft's smaller compressed output for measurement-heavy IR arrays."""
    shape = (24, 64, 32768, 1)
    expected_chunks = (1, 64, 2048, 1)
    assert choose_ir_chunk_shape(shape, np.dtype("f8")) == expected_chunks


@pytest.mark.parametrize("shape", [(0, 2, 3, 1), (1, 0, 3, 1), (1, 2, 0, 1), (1, 2, 3), (1, -2, 3, 1)])
def test_chunk_shape_rejects_empty_or_invalid_dimensions(shape):
    """Invalid shapes are rejected before creating a partially defined SOFA file."""
    with pytest.raises(ValueError, match="shape"):
        choose_ir_chunk_shape(shape, np.dtype("f8"))


@pytest.mark.parametrize("max_bytes", [0, 7, -64, 1.5])
def test_chunk_shape_rejects_a_budget_smaller_than_one_value(max_bytes):
    """The memory limit must accommodate at least one stored value."""
    with pytest.raises(ValueError, match="bytes"):
        choose_ir_chunk_shape((2, 3, 4, 1), np.dtype("f8"), max_bytes=max_bytes)


@pytest.mark.parametrize("dtype", ["f4", "i8", "O", "V0"])
def test_chunk_shape_rejects_unsupported_storage_types(dtype):
    """Chunk planning must not accept storage that the writer cannot encode."""
    with pytest.raises(ValueError, match="float64"):
        choose_ir_chunk_shape((2, 3, 4, 1), np.dtype(dtype))


@pytest.mark.parametrize("workers", [0, -1, 1.5])
def test_writer_rejects_invalid_worker_counts(tmp_path, workers):
    """Invalid worker counts fail before opening or consuming anything."""
    with pytest.raises(ValueError, match="workers"):
        write_ir_chunks(tmp_path / "missing.sofa", (), workers=workers)


def test_source_iteration_stays_on_caller_while_compression_runs_in_workers(tmp_path, monkeypatch):
    """Only the encoding work moves to worker threads."""
    caller = get_ident()
    compression_threads = []
    encode = _sofa._encode_ir_chunk
    shape = (12, 1, 8, 1)
    path = tmp_path / "threaded.sofa"
    _skeleton(path, shape, (1, 1, 8, 1))

    def checked_encode(*args):
        compression_threads.append(get_ident())
        assert get_ident() != caller
        return encode(*args)

    def source():
        buffer = np.empty((1, 1, 8, 1), dtype=np.float32)
        for row in range(shape[0]):
            assert get_ident() == caller
            buffer.fill(row)
            yield (row, 0, 0, 0), buffer

    monkeypatch.setattr(_sofa, "_encode_ir_chunk", checked_encode)
    write_ir_chunks(path, source(), workers=2)

    assert compression_threads
    with h5.File(path) as sofa:
        expected = np.broadcast_to(np.arange(shape[0])[:, None, None, None], shape)
        np.testing.assert_array_equal(sofa["Data.IR"][:], expected)


@pytest.mark.parametrize("budget", [64, 128, 64 * 1024**2])
def test_writer_bounds_pending_work_and_does_not_retain_source_batches(tmp_path, monkeypatch, budget):
    """Delayed compression cannot pin a large source batch in every queued item."""
    source_batches = []
    pending = []
    peak_pending = []
    shape = (12, 1, 8, 1)
    path = tmp_path / "bounded.sofa"
    _skeleton(path, shape, (1, 1, 8, 1))

    class DeferredFuture:
        def __init__(self, function, data, args):
            self.function, self.data, self.args = function, data, args

        def result(self):
            assert all(reference() is None for reference in source_batches[:-1])
            pending.remove(self)
            return self.function(self.data, *self.args)

        def cancel(self):
            pending.remove(self)

    class DeferredExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            assert not pending

        def submit(self, function, data, *args):
            assert data.base is None
            assert data.flags.c_contiguous
            future = DeferredFuture(function, data, args)
            pending.append(future)
            peak_pending.append(len(pending))
            assert len(pending) <= 2 * self.max_workers
            return future

    def source():
        for row in range(shape[0]):
            batch = np.full((1, 32, 128, 1), row, dtype=np.float32)
            source_batches.append(weakref.ref(batch))
            yield (row, 0, 0, 0), batch[:, :1, :8, :]
            del batch

    monkeypatch.setattr(config, "batch_bytes", budget)
    monkeypatch.setattr(_sofa, "ThreadPoolExecutor", DeferredExecutor)
    workers = 2
    write_ir_chunks(path, source(), workers=workers)

    assert max(peak_pending) == min(2 * workers, budget // 64)
    assert all(reference() is None for reference in source_batches)
    with h5.File(path) as sofa:
        expected = np.broadcast_to(np.arange(shape[0])[:, None, None, None], shape)
        np.testing.assert_array_equal(sofa["Data.IR"][:], expected)


@pytest.mark.parametrize("failure", ["source", "compression", "write"])
def test_writer_closes_handles_after_failures(tmp_path, monkeypatch, progress_tasks, failure):
    """Read, encoding, and storage errors propagate after output handles close."""
    path = tmp_path / "failure.sofa"
    shape = (3, 1, 8, 1)
    data = np.zeros(shape)
    _skeleton(path, shape, (1, 1, 8, 1))
    opened = []
    open_file = h5.File

    def tracked_file(*args, **kwargs):
        handle = open_file(*args, **kwargs)
        opened.append(handle)
        return handle

    def fail(*_args, **_kwargs):
        msg = "injected failure"
        raise OSError(msg)

    def source():
        yield from _chunks(data[:1], (1, 1, 8, 1))
        fail()

    if failure == "compression":
        monkeypatch.setattr(_sofa, "_encode_ir_chunk", fail)
    elif failure == "write":
        get_ir = _sofa._ir_variable

        class FailedWrites:
            def __init__(self, ir):
                self.ir = ir

            def __getattr__(self, name):
                return getattr(self.ir, name)

            @property
            def id(self):
                return self

            write_direct_chunk = staticmethod(fail)

        monkeypatch.setattr(_sofa, "_ir_variable", lambda sofa: FailedWrites(get_ir(sofa)))
    monkeypatch.setattr(h5, "File", tracked_file)

    with pytest.raises(OSError, match="injected failure"):
        write_ir_chunks(path, source() if failure == "source" else _chunks(data, (1, 1, 8, 1)), workers=2)

    assert opened
    assert all(not handle.id.valid for handle in opened)
    assert progress_tasks[0]["completed"] == 0
    assert progress_tasks[0]["total"] == data.size
