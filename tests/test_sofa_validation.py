"""Tests for ISTA SOFA stream writing and payload checks."""

import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from threading import get_ident
from unittest.mock import MagicMock, Mock

import h5py
import netCDF4
import numpy as np
import pytest
import sofar as sf

from irdl import ista
from irdl.ista import SrirachaDataset


class TinyChunkSrirachaDataset(SrirachaDataset):
    """SRIRACHA test double with tiny streaming chunks."""

    _chunk_size = 1


def test_ista_streaming_sofa_writer_produces_valid_checked_sofa(tmp_path):
    """ISTA streaming writer produces a SofaStream-verifiable checked SOFA file."""
    hdf5_path, _ = _write_matching_ista_files(tmp_path)
    sofa_path = tmp_path / "streamed.sofa"

    TinyChunkSrirachaDataset()._ingest(hdf5_path, sofa_path, scenario="SR1D", dataset_split=None)

    with sf.SofaStream(sofa_path) as sofa:
        assert sofa.verify(issue_handling="return", mode="read") is None
    with netCDF4.Dataset(sofa_path) as sofa:
        assert sofa.variables["Data.SamplingRate"].shape == (1,)
        assert sofa.variables["RoomVolume"].shape == (1,)
        assert sofa.RoomLocation == "TU Berlin, Einsteinufer 25, 10587 Berlin"
        assert sofa.ListenerShortName == "Custom planar microphone array"
        assert sofa.ReceiverShortName == "GRAS 40PL-1 Short CCP"
        assert sofa.SourceShortName == "Loudspeaker"
        assert sofa.ReceiverDescription == "GRAS 40PL-1 Short CCP"
        assert "ReceiverDescriptions" not in sofa.variables
        assert sofa.variables["ReceiverView"].shape == (3, 3, 1)
        assert sofa.variables["ReceiverUp"].shape == (3, 3, 1)
        assert sofa.DateCreated == sofa.DateModified
        assert sofa.DateCreated != "2026-01-01 00:00:00"
        TinyChunkSrirachaDataset()._verify_payload(sofa_path, hdf5_path, scenario="SR1D", dataset_split=None)
    with h5py.File(sofa_path) as sofa:
        assert sofa.attrs["SourceDescription"].dtype.kind == "S"


@pytest.mark.parametrize("shape", [(2, 3, 4), (4096, 64, 32_000)])
def test_ista_sofa_chunks_fit_cache_and_do_not_span_measurements(tmp_path, shape):
    """IR chunks support streaming without rewriting adjacent measurements."""
    with netCDF4.Dataset(tmp_path / "chunks.sofa", "w") as sofa:
        m, r, n = shape
        SrirachaDataset()._create_default_variables(
            sofa,
            m=m,
            r=r,
            n=n,
            has_humidity=False,
            receiver_position=np.zeros((r, 3)),
            sampling_rate=48_000,
        )
        ir = sofa.variables["Data.IR"]
        assert ir.chunking()[0] == 1
        assert np.prod(ir.chunking()) * ir.dtype.itemsize <= 1024**2
        assert ir.filters()["zlib"]
        assert ir.dtype == np.dtype(np.float64)


def test_sriracha_split_writer_streams_provider_files_to_sofa(tmp_path):
    """SRIRACHA split provider files stream directly to a checked SOFA file."""
    provider_dir = tmp_path / "provider"
    provider_dir.mkdir()
    _write_sriracha_split_files(provider_dir)
    sofa_path = tmp_path / "SR1.sofa"

    SrirachaDataset()._ingest(provider_dir, sofa_path, scenario="SR1", dataset_split=None)

    with sf.SofaStream(sofa_path) as sofa:
        assert sofa.verify(issue_handling="return", mode="read") is None
    SrirachaDataset()._verify_payload(sofa_path, provider_dir, scenario="SR1", dataset_split=None)


@pytest.mark.parametrize("chunk_size", [1, 4, 256])
@pytest.mark.parametrize("has_humidity", [False, True])
def test_sriracha_full_plane_preserves_grid_order_and_metadata(tmp_path, chunk_size, has_humidity):
    """Interleaved Split batches preserve every full-plane measurement."""
    provider_dir = tmp_path / "provider"
    provider_dir.mkdir()
    expected = _write_sriracha_split_files(provider_dir, split_grid_size=3, has_humidity=has_humidity)
    sofa_path = tmp_path / "SR1.sofa"
    dataset = SrirachaDataset()
    dataset._chunk_size = chunk_size

    dataset._ingest(provider_dir, sofa_path, scenario="SR1", dataset_split=None)

    with netCDF4.Dataset(sofa_path) as sofa:
        np.testing.assert_array_equal(sofa.variables["Data.IR"][:, :, :, 0], expected["ir"])
        np.testing.assert_array_equal(sofa.variables["SourcePosition"][:], expected["source"])
        np.testing.assert_array_equal(
            sofa.variables["RoomTemperature"][:], expected["temperature"].astype(np.float64) + 273.15
        )
        np.testing.assert_array_equal(sofa.variables["SpeedOfSound"][:, 0], expected["c0"])
        np.testing.assert_array_equal(sofa.variables["ReceiverPosition"][:, :, 0], expected["receiver"])
        np.testing.assert_array_equal(sofa.variables["Data.SamplingRate"][:], [48_000])
        assert ("Humidity" in sofa.variables) == has_humidity
        if has_humidity:
            np.testing.assert_array_equal(sofa.variables["Humidity"][:, 0], expected["humidity"])
    with sf.SofaStream(sofa_path) as sofa:
        assert sofa.verify(issue_handling="return", mode="read") is None


@pytest.mark.parametrize("variable_name", ["Data.IR", "SourcePosition"])
def test_sriracha_payload_verification_rejects_changed_full_plane(tmp_path, variable_name):
    """Contiguous validation detects corruption in the last Split row batch."""
    provider_dir = tmp_path / "provider"
    provider_dir.mkdir()
    _write_sriracha_split_files(provider_dir, split_grid_size=3)
    sofa_path = tmp_path / "SR1.sofa"
    dataset = SrirachaDataset()
    dataset._chunk_size = 4
    dataset._ingest(provider_dir, sofa_path, scenario="SR1", dataset_split=None)
    with netCDF4.Dataset(sofa_path, "a") as sofa:
        index = (-1, 0, 0, 0) if variable_name == "Data.IR" else (-1, 0)
        sofa.variables[variable_name][index] = -99.0

    with pytest.raises(ValueError, match=f"checksum differs from SRIRACHA split data: {variable_name}"):
        dataset._verify_payload(sofa_path, provider_dir, scenario="SR1", dataset_split=None)


@pytest.mark.parametrize(
    ("workers", "parallel_default", "available_gib"),
    [(None, False, 8), (None, True, 8), (None, True, 0), (1, False, 8), (2, False, 8), (4, False, 8)],
)
def test_sriracha_checksum_uses_independent_spawned_readers(
    tmp_path, monkeypatch, workers, parallel_default, available_gib
):
    """Parallel checksum readers use fresh processes and validate the entire plane."""
    provider_dir = tmp_path / "provider"
    provider_dir.mkdir()
    _write_sriracha_split_files(provider_dir, split_grid_size=4)
    sofa_path = tmp_path / "SR1.sofa"
    dataset = SrirachaDataset()
    dataset._ingest(provider_dir, sofa_path, scenario="SR1", dataset_split=None)
    pools = []

    def pool(**kwargs):
        pools.append(kwargs)
        return ProcessPoolExecutor(**kwargs)

    monkeypatch.setattr(ista, "ProcessPoolExecutor", pool)
    monkeypatch.setattr(ista.os, "cpu_count", lambda: 40)
    monkeypatch.setattr(ista.psutil, "virtual_memory", lambda: Mock(available=available_gib * 1024**3))
    if parallel_default:
        monkeypatch.setattr(ista, "_PARALLEL_CHECKSUM_MIN_BYTES", 0)
    dataset._verify_payload(sofa_path, provider_dir, scenario="SR1", workers=workers)

    default_workers = min(6, max(1, available_gib)) if parallel_default else 1
    expected_workers = workers if workers is not None else default_workers
    assert len(pools) == (expected_workers > 1)
    if pools:
        assert pools[0]["max_workers"] == expected_workers
        assert pools[0]["mp_context"].get_start_method() == "spawn"


@pytest.mark.parametrize("variable_name", ["Data.IR", "SourcePosition"])
def test_parallel_sriracha_checksum_failure_prevents_promotion(tmp_path, monkeypatch, variable_name):
    """Corruption in the final C4 batch cannot replace an existing artifact."""
    provider_dir = tmp_path / "provider"
    provider_dir.mkdir()
    _write_sriracha_split_files(provider_dir, split_grid_size=3)
    sofa_path = tmp_path / "retained.sofa"
    original = b"existing Canonical SOFA Artifact"
    sofa_path.write_bytes(original)
    dataset = SrirachaDataset()
    dataset._chunk_size = 1
    verify = dataset._verify_payload

    def corrupt_and_verify(staged, ingest, **kwargs):
        with netCDF4.Dataset(staged, "a") as sofa:
            index = (-1, 0, 0, 0) if variable_name == "Data.IR" else (-1, 0)
            sofa.variables[variable_name][index] = -99.0
        verify(staged, ingest, workers=2, **kwargs)

    monkeypatch.setattr(dataset, "_verify_payload", corrupt_and_verify)
    with pytest.raises(ValueError, match=f"checksum differs from SRIRACHA split data: {variable_name}"):
        dataset._ingest(provider_dir, sofa_path, scenario="SR1", dataset_split=None)

    assert sofa_path.read_bytes() == original
    assert not list(tmp_path.glob(".retained-*"))


@pytest.mark.parametrize("workers", [0, -1])
def test_sriracha_checksum_rejects_invalid_workers(tmp_path, workers):
    """An invalid worker count must not silently disable verification."""
    provider_dir = tmp_path / "provider"
    provider_dir.mkdir()
    _write_sriracha_split_files(provider_dir)
    sofa_path = tmp_path / "SR1.sofa"
    dataset = SrirachaDataset()
    dataset._ingest(provider_dir, sofa_path, scenario="SR1", dataset_split=None)
    with pytest.raises(ValueError, match="Checksum workers must be > 0"):
        dataset._verify_payload(sofa_path, provider_dir, scenario="SR1", workers=workers)


def test_ista_ir_batch_size_limits_memory_for_long_impulse_responses():
    """Dense SRIRACHA batches stay bounded instead of allocating gigabytes."""
    dataset = SrirachaDataset()
    rows = dataset._ir_batch_size(64, 32_000)
    assert 0 < rows < dataset._chunk_size
    assert rows * 64 * 32_000 * np.dtype(np.float64).itemsize <= 64 * 1024**2
    assert TinyChunkSrirachaDataset()._ir_batch_size(64, 32_000) == 1
    dataset._chunk_size = 0
    with pytest.raises(ValueError, match="_chunk_size must be > 0"):
        dataset._ir_batch_size(64, 32_000)


def test_sriracha_full_plane_process_keeps_provider_artifact_set(tmp_path):
    """SRIRACHA full-plane processing no longer materializes merged HDF5."""
    provider_dir = tmp_path / "provider"
    ingest_path = tmp_path / "ingest" / "SR1.h5"

    result = SrirachaDataset()._process(provider_dir, ingest_path, scenario="SR1", dataset_split=None)

    assert result == provider_dir
    assert not ingest_path.exists()


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_raw_ir_chunks_preserve_float64_values_and_partial_chunks(tmp_path, workers):
    """Externally compressed chunks decode through both netCDF4 and sofar."""
    hdf5_path, native_path = _write_matching_ista_files(tmp_path, n_samples=50_000)
    sofa_path = tmp_path / "raw-chunks.sofa"
    shutil.copyfile(native_path, sofa_path)
    with h5py.File(hdf5_path, "r+") as hdf5:
        expected = hdf5["data/impulse_response"][:] * np.float32(-0.0007)
        expected[0, 0, 0] = -0.0
        hdf5["data/impulse_response"][:] = expected
        SrirachaDataset()._write_ir_chunks(sofa_path, [(0, expected)], workers=workers)

    with netCDF4.Dataset(sofa_path) as sofa:
        actual = np.asarray(sofa.variables["Data.IR"][:, :, :, 0])
        np.testing.assert_array_equal(actual.view(np.uint64), expected.astype(np.float64).view(np.uint64))
        assert sofa.variables["Data.IR"].filters()["zlib"]
    with sf.SofaStream(sofa_path) as sofa:
        assert sofa.verify(issue_handling="return", mode="read") is None
    SrirachaDataset()._verify_payload(sofa_path, hdf5_path)


@pytest.mark.parametrize("full_plane", [False, True])
@pytest.mark.parametrize("failure", ["compress", "_verify_payload", "_verify_sofa_convention"])
def test_failed_threaded_ingest_preserves_existing_artifact(tmp_path, monkeypatch, full_plane, failure):
    """Worker and validation errors cannot promote a partial Canonical SOFA Artifact."""
    if full_plane:
        ingest_path = tmp_path / "provider"
        ingest_path.mkdir()
        _write_sriracha_split_files(ingest_path)
        scenario = "SR1"
    else:
        ingest_path, _ = _write_matching_ista_files(tmp_path)
        scenario = "SR1D"
    sofa_path = tmp_path / "retained.sofa"
    original = b"existing artifact must survive failure"
    sofa_path.write_bytes(original)
    dataset = SrirachaDataset()

    def fail(*_args, **_kwargs):
        msg = "Injected failure"
        raise RuntimeError(msg)

    monkeypatch.setattr(ista.zlib if failure == "compress" else dataset, failure, fail)
    with pytest.raises(RuntimeError, match="Injected failure"):
        dataset._ingest(ingest_path, sofa_path, scenario=scenario, dataset_split=None)

    assert sofa_path.read_bytes() == original
    assert not list(tmp_path.glob(".retained-*"))


@pytest.mark.parametrize("full_plane", [False, True])
def test_threaded_ingest_keeps_hdf5_reads_on_main_thread(tmp_path, monkeypatch, full_plane):
    """Compression uses workers, while Provider reads remain on the caller thread."""
    if full_plane:
        ingest_path = tmp_path / "provider"
        ingest_path.mkdir()
        _write_sriracha_split_files(ingest_path, split_grid_size=3)
        scenario = "SR1"
    else:
        ingest_path, _ = _write_matching_ista_files(tmp_path)
        scenario = "SR1D"
    main_thread = get_ident()
    encoder_threads = set()
    original_read = h5py.Dataset.__getitem__
    original_compress = ista.zlib.compress

    def read(dataset, key):
        assert get_ident() == main_thread
        return original_read(dataset, key)

    def compress(*args):
        encoder_threads.add(get_ident())
        return original_compress(*args)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", read)
    monkeypatch.setattr(ista.zlib, "compress", compress)
    monkeypatch.setattr(ista.os, "cpu_count", lambda: 4)
    SrirachaDataset()._ingest(ingest_path, tmp_path / "threaded.sofa", scenario=scenario, dataset_split=None)

    assert encoder_threads
    assert main_thread not in encoder_threads


def test_threaded_chunk_queue_is_bounded_and_owns_submitted_buffers(tmp_path, monkeypatch):
    """Backpressure limits queued work even when the producer reuses its input."""
    sofa_path = tmp_path / "bounded.sofa"
    with netCDF4.Dataset(sofa_path, "w") as sofa:
        SrirachaDataset()._create_default_variables(
            sofa, m=10, r=2, n=3, has_humidity=False, receiver_position=np.zeros((2, 3)), sampling_rate=48_000
        )
    pending = maximum = 0
    workers = 2
    executor = MagicMock()
    executor.__enter__.return_value = executor

    def submit(function, *args):
        nonlocal pending, maximum
        pending += 1
        maximum = max(maximum, pending)
        future = Mock()

        def result():
            nonlocal pending
            pending -= 1
            return function(*args)

        future.result.side_effect = result
        return future

    def batches():
        buffer = np.empty((1, 2, 3), dtype=np.float32)
        for row in range(10):
            buffer.fill(row)
            yield row, buffer

    executor.submit.side_effect = submit
    monkeypatch.setattr(ista, "ThreadPoolExecutor", lambda **_kwargs: executor)
    SrirachaDataset()._write_ir_chunks(sofa_path, batches(), workers=workers)

    assert maximum <= 2 * workers
    assert pending == 0
    with netCDF4.Dataset(sofa_path) as sofa:
        expected = np.broadcast_to(np.arange(10)[:, None, None], (10, 2, 3))
        np.testing.assert_array_equal(sofa.variables["Data.IR"][:, :, :, 0], expected)


@pytest.mark.parametrize("n_sources", [0, 2])
def test_sriracha_rejects_incomplete_split_grids(tmp_path, n_sources):
    """Every Split must describe a nonempty square grid, not a partial plane."""
    provider_dir = tmp_path / "provider"
    provider_dir.mkdir()
    _write_sriracha_split_files(provider_dir)
    with h5py.File(provider_dir / "SR1-C1.h5", "r+") as hdf5:
        del hdf5["data/impulse_response"]
        hdf5["data/impulse_response"] = np.zeros((n_sources, 2, 3), dtype=np.float32)
    sofa_path = tmp_path / "SR1.sofa"
    with pytest.raises(ValueError, match="nonempty square grid"):
        SrirachaDataset()._ingest(provider_dir, sofa_path, scenario="SR1", dataset_split=None)
    assert not sofa_path.exists()


def test_sriracha_rejects_mismatched_split_shapes(tmp_path):
    """Extra rows in another Split cannot silently disappear from the SOFA."""
    provider_dir = tmp_path / "provider"
    provider_dir.mkdir()
    _write_sriracha_split_files(provider_dir)
    with h5py.File(provider_dir / "SR1-C4.h5", "r+") as hdf5:
        del hdf5["data/impulse_response"]
        hdf5["data/impulse_response"] = np.zeros((2, 2, 3), dtype=np.float32)
    sofa_path = tmp_path / "SR1.sofa"
    with pytest.raises(ValueError, match="Split IR shapes must match"):
        SrirachaDataset()._ingest(provider_dir, sofa_path, scenario="SR1", dataset_split=None)
    assert not sofa_path.exists()


@pytest.mark.parametrize(("dtype", "zlib", "shuffle"), [("f4", True, True), ("f8", False, False), ("f8", True, False)])
def test_raw_chunk_writer_rejects_incompatible_storage(tmp_path, dtype, zlib, shuffle):
    """A mismatched filter pipeline must never receive externally encoded bytes."""
    sofa_path = tmp_path / "unsupported.sofa"
    with netCDF4.Dataset(sofa_path, "w") as sofa:
        for name, size in (("M", 1), ("R", 2), ("N", 3), ("E", 1)):
            sofa.createDimension(name, size)
        sofa.createVariable("Data.IR", dtype, ("M", "R", "N", "E"), zlib=zlib, shuffle=shuffle)
    with pytest.raises(ValueError, match="requires float64"):
        SrirachaDataset()._write_ir_chunks(sofa_path, [(0, np.ones((1, 2, 3), dtype=np.float32))])


@pytest.mark.parametrize("dtype", ["<f8", ">f8"])
@pytest.mark.parametrize("receivers", [2, 3])
def test_chunk_encoder_preserves_storage_byte_order_and_edge_fill(dtype, receivers):
    """Raw encoding preserves signed zero/nonfinite values and pads the sample tail."""
    data = np.array([[-0.0, np.inf, np.nan], [1.5, -np.inf, 0.0]], dtype=np.float32)
    storage = np.dtype(dtype)
    encoded = ista._encode_ir_chunk(data, (1, receivers, 4, 1), storage, 99.0, 4)
    shuffled = np.frombuffer(ista.zlib.decompress(encoded), dtype=np.uint8)
    decoded = shuffled.reshape(storage.itemsize, -1).T.copy().reshape(-1)
    expected = np.full((receivers, 4), 99.0, dtype=storage)
    expected[:2, :3] = data
    np.testing.assert_array_equal(decoded, expected.view(np.uint8).reshape(-1))


def test_ista_payload_verification_accepts_matching_hdf5_and_sofa(tmp_path):
    """ISTA payload verification accepts matching HDF5 and SOFA data."""
    hdf5_path, sofa_path = _write_matching_ista_files(tmp_path)

    TinyChunkSrirachaDataset()._verify_payload(sofa_path, hdf5_path, scenario="SR1D", dataset_split=None)


def test_ista_payload_verification_rejects_changed_coordinates(tmp_path):
    """ISTA payload verification rejects changed Coordinates."""
    hdf5_path, sofa_path = _write_matching_ista_files(tmp_path)
    with netCDF4.Dataset(sofa_path, "a") as dataset:
        dataset.variables["SourcePosition"][0, 0] = 99.0

    with pytest.raises(ValueError, match="checksum differs from ISTA HDF5 ingest data: SourcePosition"):
        TinyChunkSrirachaDataset()._verify_payload(sofa_path, hdf5_path, scenario="SR1D", dataset_split=None)


def _write_sriracha_split_files(
    provider_dir: Path, *, split_grid_size: int = 1, has_humidity: bool = False
) -> dict[str, np.ndarray]:
    n = 2 * split_grid_size
    m = n * n
    expected = {
        "ir": np.arange(m * 2 * 3, dtype=np.float32).reshape(m, 2, 3),
        "source": np.column_stack((np.arange(m) // n, np.arange(m) % n, np.ones(m))).astype(np.float64),
        "receiver": np.array([[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]], dtype=np.float64),
        "temperature": np.arange(m, dtype=np.float32) / 10 + 20,
        "c0": np.arange(m, dtype=np.float32) / 10 + 330,
        "humidity": np.arange(m, dtype=np.float32) / 10 + 40,
    }
    for split, row, col in (("C1", 0, 0), ("C2", 0, 1), ("C3", 1, 0), ("C4", 1, 1)):
        selection = np.arange(m).reshape(n, n)[row::2, col::2].reshape(-1)
        with h5py.File(provider_dir / f"SR1-{split}.h5", "w") as hdf5:
            hdf5["data/impulse_response"] = expected["ir"][selection]
            hdf5["data/location/source"] = expected["source"][selection]
            hdf5["data/location/receiver"] = expected["receiver"]
            hdf5["metadata/sampling_rate"] = 48_000
            for name in ("temperature", "c0", "humidity") if has_humidity else ("temperature", "c0"):
                hdf5[f"metadata/{name}"] = expected[name][selection]
    return expected


def _write_matching_ista_files(tmp_path: Path, *, n_samples: int = 4) -> tuple[Path, Path]:
    ir = np.arange(2 * 3 * n_samples, dtype=np.float32).reshape(2, 3, n_samples)
    source = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float64)
    receiver = np.array([[0.0, 0.5, 0.0], [0.0, -0.5, 0.0], [0.5, 0.0, 0.0]], dtype=np.float64)

    hdf5_path = tmp_path / "ista.h5"
    with h5py.File(hdf5_path, "w") as hdf5:
        data = hdf5.create_group("data")
        data.create_dataset("impulse_response", data=ir)
        location = data.create_group("location")
        location.create_dataset("source", data=source)
        location.create_dataset("receiver", data=receiver)
        metadata = hdf5.create_group("metadata")
        metadata.create_dataset("sampling_rate", data=48_000)
        metadata.create_dataset("temperature", data=np.array([20.0, 20.0], dtype=np.float32))
        metadata.create_dataset("c0", data=np.array([343.0, 343.0], dtype=np.float32))

    sofa_path = tmp_path / "ista.sofa"
    TinyChunkSrirachaDataset()._ingest(hdf5_path, sofa_path, scenario="SR1D", dataset_split=None)
    return hdf5_path, sofa_path
