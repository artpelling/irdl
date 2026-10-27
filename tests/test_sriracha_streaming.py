"""Tests for full-plane SRIRACHA Ingest through the shared IR writer."""

from pathlib import Path
from threading import get_ident

import h5py as h5
import netCDF4
import numpy as np
import pytest
import sofar as sf

from irdl import _sofa, config
from irdl.datasets import ista
from irdl.datasets.ista import SrirachaDataset
from irdl.utils import _atomic_write

_SAMPLING_RATE = 48000


def _provider_splits(directory: Path, *, side=3, receivers=3, samples=17, humidity=True):
    """Select each Provider Split from an independently constructed full-plane grid."""
    directory.mkdir()
    full_side = 2 * side
    measurements = full_side**2
    ir = np.arange(measurements * receivers * samples, dtype=np.float64).reshape(measurements, receivers, samples)
    ir.flat[1] = np.nextafter(1.0, 2.0)
    source = np.column_stack(
        (np.repeat(np.arange(full_side), full_side), np.tile(np.arange(full_side), full_side), np.ones(measurements))
    ).astype(np.float64)
    expected = {
        "ir": ir,
        "source": source,
        "temperature": 20.0 + np.arange(measurements) / 100,
        "c0": 343.0 + np.arange(measurements) / 1000,
        "humidity": 40.0 + np.arange(measurements) / 100,
        "receiver": np.column_stack((np.arange(receivers), np.zeros((receivers, 2)))).astype(np.float64),
    }
    for name, row, column in (("C1", 0, 0), ("C2", 0, 1), ("C3", 1, 0), ("C4", 1, 1)):

        def select(array, row=row, column=column):
            grid = array.reshape(full_side, full_side, *array.shape[1:])
            return grid[row::2, column::2].reshape(side**2, *array.shape[1:])

        with h5.File(directory / f"SR1-{name}.h5", "w") as provider:
            provider.create_dataset("data/impulse_response", data=select(ir), compression="gzip")
            provider.create_dataset("data/location/source", data=select(source))
            provider.create_dataset("data/location/receiver", data=expected["receiver"])
            provider.create_dataset("metadata/sampling_rate", data=_SAMPLING_RATE)
            for field in ("temperature", "c0", "humidity") if humidity else ("temperature", "c0"):
                provider.create_dataset(f"metadata/{field}", data=select(expected[field]))
    return expected


@pytest.mark.parametrize("budget", [64, 1024, 64 * 1024**2])
@pytest.mark.parametrize("humidity", [False, True])
def test_full_plane_uses_shared_writer_and_preserves_order_and_metadata(tmp_path, monkeypatch, budget, humidity):
    """Combining four Splits preserves independently defined measurements and metadata."""
    provider_dir = tmp_path / "provider"
    expected = _provider_splits(provider_dir, humidity=humidity)
    path = tmp_path / "SR1.sofa"
    writes = []
    original_write = ista.write_ir_chunks

    def observed_write(sofa_path, chunks):
        writes.append(sofa_path)
        return original_write(sofa_path, chunks)

    monkeypatch.setattr(ista, "write_ir_chunks", observed_write)
    dataset = SrirachaDataset()
    monkeypatch.setattr(config, "batch_bytes", budget)
    dataset._ingest(provider_dir, path, scenario="SR1", dataset_split=None)

    assert writes == [path]
    with netCDF4.Dataset(path) as sofa:
        np.testing.assert_array_equal(sofa.variables["Data.IR"][:, :, :, 0], expected["ir"])
        np.testing.assert_array_equal(sofa.variables["SourcePosition"][:], expected["source"])
        np.testing.assert_array_equal(sofa.variables["ReceiverPosition"][:, :, 0], expected["receiver"])
        np.testing.assert_array_equal(sofa.variables["RoomTemperature"][:], expected["temperature"] + 273.15)
        np.testing.assert_array_equal(sofa.variables["SpeedOfSound"][:, 0], expected["c0"])
        assert ("Humidity" in sofa.variables) == humidity
        if humidity:
            np.testing.assert_array_equal(sofa.variables["Humidity"][:, 0], expected["humidity"])
        assert sofa.ListenerShortName == "Custom planar microphone array"
        assert sofa.variables["Data.SamplingRate"][0] == _SAMPLING_RATE
    with h5.File(path) as sofa:
        assert sofa["Data.IR"].chunks[0] == 1
        assert np.prod(sofa["Data.IR"].chunks) * 8 <= 1024**2
    with sf.SofaStream(path) as sofa:
        assert sofa.verify(issue_handling="return", mode="read") is None
    assert path.stat().st_mode & 0o777 == provider_dir.stat().st_mode & 0o777
    assert sorted(file.name for file in provider_dir.iterdir()) == [f"SR1-C{number}.h5" for number in range(1, 5)]


@pytest.mark.parametrize("budget", [8, 64, 80, 4096, 64 * 1024**2])
def test_full_plane_bounds_source_reads_for_writing_and_verification(tmp_path, monkeypatch, budget):
    """Both writing and checking tile measurements that exceed the memory budget."""
    provider_dir = tmp_path / "provider"
    expected = _provider_splits(provider_dir)
    path = tmp_path / "bounded.sofa"
    reads = []
    caller = get_ident()
    getitem = h5.Dataset.__getitem__
    check_values = ista._check_sriracha_values

    def bounded_check(sofa_path, name, expected_values, actual, dtype):
        if name == "Data.IR":
            assert expected_values.nbytes <= budget
            assert actual.nbytes <= budget
        return check_values(sofa_path, name, expected_values, actual, dtype)

    def bounded_read(dataset, selection):
        values = getitem(dataset, selection)
        if dataset.name == "/data/impulse_response":
            assert get_ident() == caller
            assert values.size * np.dtype("f8").itemsize <= budget
            reads.append(values.shape)
        return values

    monkeypatch.setattr(config, "batch_bytes", budget)
    monkeypatch.setattr(ista, "_check_sriracha_values", bounded_check)
    monkeypatch.setattr(h5.Dataset, "__getitem__", bounded_read)
    dataset = SrirachaDataset()
    dataset._ingest(provider_dir, path, scenario="SR1", dataset_split=None)

    assert reads
    with netCDF4.Dataset(path) as sofa:
        assert sofa.variables["Data.IR"][:, :, :, 0].tobytes() == expected["ir"].tobytes()


@pytest.mark.parametrize("variable", ["Data.IR", "SourcePosition"])
def test_full_plane_verification_rejects_changed_values_in_last_split(tmp_path, variable):
    """Payload checks still cover the end of the combined grid and Source Coordinates."""
    provider_dir = tmp_path / "provider"
    _provider_splits(provider_dir)
    path = tmp_path / "changed.sofa"
    dataset = SrirachaDataset()
    dataset._ingest(provider_dir, path, scenario="SR1", dataset_split=None)

    with netCDF4.Dataset(path, "a") as sofa:
        index = (-1, 0, 0, 0) if variable == "Data.IR" else (-1, 0)
        sofa.variables[variable][index] = 999.0

    with pytest.raises(ValueError, match="checksum differs from SRIRACHA split data"):
        dataset._verify_payload(path, provider_dir, scenario="SR1", dataset_split=None)


@pytest.mark.parametrize("shape", [(0, 3, 17), (2, 3, 17), (9, 0, 17), (9, 3, 0), (9, 3), ()])
def test_full_plane_rejects_empty_nonsquare_or_invalid_split_grids(tmp_path, shape):
    """An incomplete Split grid must fail before creating a SOFA skeleton."""
    provider_dir = tmp_path / "provider"
    _provider_splits(provider_dir)
    with h5.File(provider_dir / "SR1-C1.h5", "a") as split:
        del split["data/impulse_response"]
        split.create_dataset("data/impulse_response", shape=shape, dtype="f8")
    path = tmp_path / "invalid.sofa"

    with pytest.raises(ValueError, match="nonempty square grid"):
        SrirachaDataset()._ingest(provider_dir, path, scenario="SR1", dataset_split=None)

    assert not path.exists()
    with h5.File(provider_dir / "SR1-C2.h5", "a") as split:
        split.attrs["source_handle_closed"] = True


@pytest.mark.parametrize("shape", [(4, 3, 17), (9, 2, 17), (9, 3, 13)])
def test_full_plane_rejects_mismatched_ir_shapes(tmp_path, shape):
    """All four Splits must describe the same grid size, receivers, and samples."""
    provider_dir = tmp_path / "provider"
    _provider_splits(provider_dir)
    with h5.File(provider_dir / "SR1-C4.h5", "a") as split:
        del split["data/impulse_response"]
        split.create_dataset("data/impulse_response", shape=shape, dtype="f8")
    path = tmp_path / "mismatched.sofa"

    with pytest.raises(ValueError, match="IR shapes must match"):
        SrirachaDataset()._ingest(provider_dir, path, scenario="SR1", dataset_split=None)
    assert not path.exists()


@pytest.mark.parametrize("split_name", ["C2", "C4"])
def test_full_plane_rejects_truncated_source_coordinates_instead_of_broadcasting(tmp_path, split_name):
    """A short final Coordinate batch must not become repeated valid-looking positions."""
    provider_dir = tmp_path / "provider"
    _provider_splits(provider_dir)
    with h5.File(provider_dir / f"SR1-{split_name}.h5", "a") as split:
        positions = split["data/location/source"][:7]
        del split["data/location/source"]
        split.create_dataset("data/location/source", data=positions)
    path = tmp_path / "short-coordinates.sofa"

    with pytest.raises(ValueError, match="Source Coordinates"):
        SrirachaDataset()._ingest(provider_dir, path, scenario="SR1", dataset_split=None)
    assert not path.exists()


@pytest.mark.parametrize("budget", [0, -1])
def test_full_plane_rejects_invalid_batch_budgets(tmp_path, monkeypatch, budget):
    """Full-plane Ingest and its payload check validate the shared byte budget."""
    provider_dir = tmp_path / "provider"
    _provider_splits(provider_dir)
    path = tmp_path / "invalid.sofa"
    dataset = SrirachaDataset()
    monkeypatch.setattr(config, "batch_bytes", budget)

    with pytest.raises(ValueError, match="batch_bytes"):
        dataset._ingest(provider_dir, path, scenario="SR1", dataset_split=None)
    with pytest.raises(ValueError, match="batch_bytes"):
        dataset._verify_payload(path, provider_dir, scenario="SR1", dataset_split=None)


def test_full_plane_preserves_the_existing_float32_checksum_policy(tmp_path):
    """Changing read sizes does not silently tighten the existing comparison precision."""
    provider_dir = tmp_path / "provider"
    _provider_splits(provider_dir)
    path = tmp_path / "precision.sofa"
    dataset = SrirachaDataset()
    dataset._ingest(provider_dir, path, scenario="SR1", dataset_split=None)
    with netCDF4.Dataset(path, "a") as sofa:
        ir = sofa.variables["Data.IR"]
        value = float(ir[0, 0, 1, 0])
        ir[0, 0, 1, 0] = np.nextafter(value, np.inf)
    dataset._verify_payload(path, provider_dir, scenario="SR1", dataset_split=None)


def _local_dataset(provider_dir):
    """Use real local Provider Splits through Get without network acquisition."""

    class LocalSrirachaDataset(SrirachaDataset):
        name = "local-sriracha"

        def _download(self, _provider_dir, **_kwargs):
            return provider_dir

    return LocalSrirachaDataset()


def _failed_writes(ir, fail):
    """Replace only the storage write with an injected failure."""

    class FailedWrites:
        def __init__(self, ir):
            self.ir = ir

        def __getattr__(self, name):
            return getattr(self.ir, name)

        @property
        def id(self):
            return self

        write_direct_chunk = staticmethod(fail)

    return FailedWrites(ir)


@pytest.mark.parametrize("failure", ["source", "compression", "write", "payload", "convention"])
def test_full_plane_get_discards_failures_and_closes_source_handles(tmp_path, monkeypatch, failure):
    """No failure between source reading and promotion leaves trusted partial output."""
    provider_dir = tmp_path / "provider"
    _provider_splits(provider_dir)
    dataset = _local_dataset(provider_dir)
    opened = []
    open_file = h5.File
    source_chunks = ista._sriracha_ir_chunks

    def tracked_file(*args, **kwargs):
        handle = open_file(*args, **kwargs)
        opened.append(handle)
        return handle

    def fail(*_args, **_kwargs):
        msg = "injected failure"
        raise OSError(msg)

    def failed_source(*args):
        yield next(source_chunks(*args))
        fail()

    if failure == "source":
        monkeypatch.setattr(ista, "_sriracha_ir_chunks", failed_source)
    elif failure == "compression":
        monkeypatch.setattr(_sofa, "_encode_ir_chunk", fail)
    elif failure == "write":
        get_ir = _sofa._ir_variable

        monkeypatch.setattr(_sofa, "_ir_variable", lambda sofa: _failed_writes(get_ir(sofa), fail))
    elif failure == "payload":
        monkeypatch.setattr(dataset, "_verify_payload", fail)
    elif failure == "convention":
        monkeypatch.setattr(dataset, "_verify_sofa_convention", fail)
    monkeypatch.setattr(h5, "File", tracked_file)

    with pytest.raises(OSError, match="injected failure"):
        dataset._get(cache_dir=tmp_path, export_dir=None, output_format="sofa", scenario="SR1", dataset_split=None)

    assert opened
    assert all(not handle.id.valid for handle in opened)
    cache = tmp_path / dataset.name.upper()
    assert list((cache / "output").iterdir()) == []
    assert not (cache / "ingest" / "SR1.h5").exists()


def test_full_plane_get_rejects_a_missing_split_without_partial_output(tmp_path):
    """A missing fourth Split closes already opened files and cannot produce a SOFA."""
    provider_dir = tmp_path / "provider"
    _provider_splits(provider_dir)
    (provider_dir / "SR1-C4.h5").unlink()
    dataset = _local_dataset(provider_dir)

    with pytest.raises(FileNotFoundError):
        dataset._get(cache_dir=tmp_path, export_dir=None, output_format="sofa", scenario="SR1", dataset_split=None)

    assert list((tmp_path / dataset.name.upper() / "output").iterdir()) == []
    with h5.File(provider_dir / "SR1-C1.h5", "a") as source:
        source.attrs["closed_after_missing_split"] = True


def test_full_plane_failed_rewrite_preserves_existing_artifact(tmp_path, monkeypatch):
    """The same staging mechanism keeps an existing artifact intact if a rewrite fails."""
    provider_dir = tmp_path / "provider"
    _provider_splits(provider_dir)
    path = tmp_path / "SR1.sofa"
    dataset = SrirachaDataset()
    dataset._ingest(provider_dir, path, scenario="SR1", dataset_split=None)
    original = path.read_bytes()

    def fail(*_args, **_kwargs):
        msg = "injected compression failure"
        raise OSError(msg)

    monkeypatch.setattr(_sofa, "_encode_ir_chunk", fail)
    with pytest.raises(OSError, match="injected compression failure"), _atomic_write(path) as temporary:
        dataset._ingest(provider_dir, temporary, scenario="SR1", dataset_split=None)

    assert path.read_bytes() == original
    assert not path.with_name(f".tmp-{path.name}").exists()
