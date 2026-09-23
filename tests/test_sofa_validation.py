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


def _write_sriracha_split_files(provider_dir: Path) -> None:
    split_values = {"C1": 1.0, "C2": 2.0, "C3": 3.0, "C4": 4.0}
    split_positions = {
        "C1": [[0.0, 0.0, 0.0]],
        "C2": [[1.0, 0.0, 0.0]],
        "C3": [[0.0, 1.0, 0.0]],
        "C4": [[1.0, 1.0, 0.0]],
    }
    receiver = np.array([[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]], dtype=np.float64)
    for split, value in split_values.items():
        with h5py.File(provider_dir / f"SR1-{split}.h5", "w") as hdf5:
            data = hdf5.create_group("data")
            data.create_dataset("impulse_response", data=np.full((1, 2, 3), value, dtype=np.float32))
            location = data.create_group("location")
            location.create_dataset("source", data=np.asarray(split_positions[split], dtype=np.float64))
            location.create_dataset("receiver", data=receiver)
            metadata = hdf5.create_group("metadata")
            metadata.create_dataset("sampling_rate", data=48_000)
            metadata.create_dataset("temperature", data=np.array([20.0], dtype=np.float32))
            metadata.create_dataset("c0", data=np.array([343.0], dtype=np.float32))


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
