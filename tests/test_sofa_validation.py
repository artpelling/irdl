"""Tests for ISTA SOFA stream writing and payload checks."""

from pathlib import Path

import h5py
import netCDF4
import numpy as np
import pytest
import sofar as sf

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


def _write_matching_ista_files(tmp_path: Path) -> tuple[Path, Path]:
    ir = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
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
