"""Tests for ISTA SOFA stream writing and payload checks."""

from pathlib import Path
from threading import get_ident

import h5py
import netCDF4
import numpy as np
import pytest
import sofar as sf

from irdl import _sofa, config
from irdl.datasets import ista
from irdl.datasets.ista import SrirachaDataset
from irdl.utils import _atomic_write


@pytest.fixture(autouse=True)
def tiny_batches(monkeypatch):
    """Exercise small batches through the same setting used by all Datasets."""
    monkeypatch.setattr(config, "batch_bytes", 96)


def test_ista_streaming_sofa_writer_produces_valid_checked_sofa(tmp_path):
    """ISTA streaming writer produces a SofaStream-verifiable checked SOFA file."""
    hdf5_path, _ = _write_matching_ista_files(tmp_path)
    sofa_path = tmp_path / "streamed.sofa"

    SrirachaDataset()._ingest(hdf5_path, sofa_path, scenario="SR1-D", dataset_split=None)

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
        SrirachaDataset()._verify_payload(sofa_path, hdf5_path, scenario="SR1-D", dataset_split=None)
    with h5py.File(sofa_path) as sofa:
        assert sofa.attrs["SourceDescription"].dtype.kind == "S"


def test_ista_single_file_uses_byte_bounded_chunks_and_preserves_permissions(tmp_path):
    """Single-file Ingest retains values and permissions with measurement-local chunks."""
    hdf5_path, _ = _write_matching_ista_files(tmp_path)
    hdf5_path.chmod(0o640)
    sofa_path = tmp_path / "single-file.sofa"

    SrirachaDataset()._ingest(hdf5_path, sofa_path, scenario="SR1-D", dataset_split=None)

    with h5py.File(hdf5_path) as source, h5py.File(sofa_path) as sofa:
        ir = sofa["Data.IR"]
        assert ir.chunks[0] == ir.chunks[-1] == 1
        assert np.prod(ir.chunks) * ir.dtype.itemsize <= 1024**2
        np.testing.assert_array_equal(ir[:, :, :, 0], source["data/impulse_response"][:])
    assert sofa_path.stat().st_mode & 0o777 == hdf5_path.stat().st_mode & 0o777


@pytest.mark.parametrize("budget", [80, 4096])
def test_ista_writer_bounds_source_reads_including_oversized_rows(tmp_path, monkeypatch, budget):
    """The writer never requires a whole measurement that exceeds its source budget."""
    expected = np.arange(7 * 3 * 17, dtype=np.float64).reshape(7, 3, 17)
    expected.flat[1] = np.nextafter(1.0, 2.0)
    source_path, _ = _write_matching_ista_files(tmp_path, expected)
    output = tmp_path / "bounded.sofa"
    caller = get_ident()
    reads = []
    writing = False
    original_getitem = h5py.Dataset.__getitem__
    original_write = ista.write_ir_chunks

    def checked_read(dataset, selection):
        values = original_getitem(dataset, selection)
        if writing and dataset.name == "/data/impulse_response":
            assert get_ident() == caller
            assert values.size * np.dtype("f8").itemsize <= budget
            reads.append(values.shape)
        return values

    def checked_write(*args, **kwargs):
        nonlocal writing
        writing = True
        try:
            return original_write(*args, **kwargs)
        finally:
            writing = False

    monkeypatch.setattr(config, "batch_bytes", budget)
    monkeypatch.setattr(ista, "write_ir_chunks", checked_write)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", checked_read)
    SrirachaDataset()._ingest(source_path, output, scenario="SR1-D", dataset_split=None)

    assert reads
    with h5py.File(output) as sofa:
        assert sofa["Data.IR"][:, :, :, 0].tobytes() == expected.tobytes()
    with sf.SofaStream(output) as sofa:
        assert sofa.verify(issue_handling="return", mode="read") is None


@pytest.mark.parametrize("budget", [80, 4096])
def test_ista_payload_checks_bound_source_reads_without_changing_comparison(tmp_path, monkeypatch, budget):
    """Existing checks retain their precision policy without reading oversized rows."""
    expected = np.arange(7 * 3 * 17, dtype=np.float64).reshape(7, 3, 17)
    source_path, sofa_path = _write_matching_ista_files(tmp_path, expected)
    original_getitem = h5py.Dataset.__getitem__
    reads = []

    def checked_read(dataset, selection):
        values = original_getitem(dataset, selection)
        if dataset.name == "/data/impulse_response":
            assert values.size * np.dtype("f8").itemsize <= budget
            reads.append(values.shape)
        return values

    monkeypatch.setattr(config, "batch_bytes", budget)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", checked_read)
    SrirachaDataset()._verify_payload(sofa_path, source_path)
    assert reads

    with netCDF4.Dataset(sofa_path, "a") as sofa:
        sofa.variables["Data.IR"][0, 0, 0, 0] = 999.0
    with pytest.raises(ValueError, match="checksum differs"):
        SrirachaDataset()._verify_payload(sofa_path, source_path)


@pytest.mark.parametrize("failure", ["source", "compression", "payload", "convention"])
def test_single_file_get_does_not_promote_failed_sofa(tmp_path, monkeypatch, failure):
    """Actual ISTA failures are discarded by the existing Get staging mechanism."""
    source_path, _ = _write_matching_ista_files(tmp_path)

    class LocalSrirachaDataset(SrirachaDataset):
        name = "local-sriracha"

        def _download(self, _provider_dir, **_kwargs):
            return source_path

    def fail(*_args, **_kwargs):
        msg = "injected failure"
        raise OSError(msg)

    def failed_source(*args):
        yield next(ista_ir_chunks(*args))
        fail()

    dataset = LocalSrirachaDataset()
    ista_ir_chunks = ista._ista_ir_chunks
    if failure == "source":
        monkeypatch.setattr(ista, "_ista_ir_chunks", failed_source)
    elif failure == "compression":
        monkeypatch.setattr(_sofa, "_encode_ir_chunk", fail)
    elif failure == "payload":
        monkeypatch.setattr(dataset, "_verify_payload", fail)
    elif failure == "convention":
        monkeypatch.setattr(dataset, "_verify_sofa_convention", fail)

    with pytest.raises(OSError, match="injected failure"):
        dataset._get(cache_dir=tmp_path, export_dir=None, output_format="sofa", scenario="SR1-D", dataset_split=None)

    output_dir = tmp_path / dataset.name.upper() / "output"
    assert list(output_dir.iterdir()) == []


def test_single_file_failed_write_keeps_existing_artifact(tmp_path, monkeypatch):
    """A failed staged rewrite cannot replace an existing Canonical SOFA Artifact."""
    source_path, sofa_path = _write_matching_ista_files(tmp_path)
    original = sofa_path.read_bytes()

    def fail(*_args, **_kwargs):
        msg = "injected compression failure"
        raise OSError(msg)

    monkeypatch.setattr(_sofa, "_encode_ir_chunk", fail)
    with pytest.raises(OSError, match="injected compression failure"), _atomic_write(sofa_path) as staged:
        SrirachaDataset()._ingest(source_path, staged, scenario="SR1-D", dataset_split=None)

    assert sofa_path.read_bytes() == original
    assert not sofa_path.with_name(f".tmp-{sofa_path.name}").exists()


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

    SrirachaDataset()._verify_payload(sofa_path, hdf5_path, scenario="SR1-D", dataset_split=None)


def test_ista_payload_verification_rejects_changed_coordinates(tmp_path):
    """ISTA payload verification rejects changed Coordinates."""
    hdf5_path, sofa_path = _write_matching_ista_files(tmp_path)
    with netCDF4.Dataset(sofa_path, "a") as dataset:
        dataset.variables["SourcePosition"][0, 0] = 99.0

    with pytest.raises(ValueError, match="checksum differs from ISTA HDF5 ingest data: SourcePosition"):
        SrirachaDataset()._verify_payload(sofa_path, hdf5_path, scenario="SR1-D", dataset_split=None)


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


def _write_matching_ista_files(tmp_path: Path, ir: np.ndarray | None = None) -> tuple[Path, Path]:
    if ir is None:
        ir = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    measurements = ir.shape[0]
    source = np.column_stack((np.arange(measurements), np.zeros(measurements), np.ones(measurements)))
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
        metadata.create_dataset("temperature", data=np.full(measurements, 20.0, dtype=np.float32))
        metadata.create_dataset("c0", data=np.full(measurements, 343.0, dtype=np.float32))

    sofa_path = tmp_path / "ista.sofa"
    SrirachaDataset()._ingest(hdf5_path, sofa_path, scenario="SR1-D", dataset_split=None)
    return hdf5_path, sofa_path
