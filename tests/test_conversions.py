"""Tests for Dataset conversion methods."""

import os
import tempfile
from pathlib import Path

import h5py
import netCDF4
import numpy as np
import pyfar as pf
import pytest
import sofar as sf

from irdl import config
from irdl.base import BaseDataset


# Create a concrete Dataset subclass for testing conversion methods
# (BaseDataset is abstract and cannot be instantiated directly)
class TestDataset(BaseDataset):
    """Concrete Dataset subclass for testing conversion methods."""

    name = "test"
    doi = "10.0000/test"

    def _validate_params(self, dataset_kwargs):
        """No-op validation for test dataset."""

    def _download(self, **kwargs):
        """No-op download for test dataset."""

    def _ingest(self, file_path):
        """No-op ingest for test dataset."""

    def _source_filename(self, **kwargs):  # noqa: ARG002
        """Return test file name."""
        return "test.sofa"


# Create the TestDataset instance to use for all conversion tests
test_dataset = TestDataset()


def _assert_permissions_preserved(source_path: Path, target_path: Path) -> None:
    """Assert permission preservation with Windows-compatible semantics."""
    source_mode = source_path.stat().st_mode & 0o777
    target_mode = target_path.stat().st_mode & 0o777
    if os.name == "nt":
        assert bool(source_mode & 0o200) == bool(target_mode & 0o200)
    else:
        assert target_mode == source_mode


class TestConversionToPyFar:
    """Tests for _to_pyfar conversion method."""

    def test_conversion_to_pyfar_returns_dict(self, sofa_object):
        """Verify _to_pyfar returns a dictionary."""
        result = test_dataset._to_pyfar(sofa_object)
        assert isinstance(result, dict)

    def test_conversion_to_pyfar_structure(self, sofa_object):
        """Verify _to_pyfar returns dict with expected keys."""
        result = test_dataset._to_pyfar(sofa_object)
        expected_keys = {"impulse_response", "source_coordinates", "receiver_coordinates"}
        assert set(result.keys()) == expected_keys

    def test_conversion_to_pyfar_types(self, sofa_object):
        """Verify _to_pyfar returns pyfar objects of correct types."""
        result = test_dataset._to_pyfar(sofa_object)

        assert isinstance(result["impulse_response"], pf.Signal)
        assert isinstance(result["source_coordinates"], pf.Coordinates)
        assert isinstance(result["receiver_coordinates"], pf.Coordinates)

    def test_conversion_to_pyfar_sampling_rate(self, sofa_object):
        """Verify _to_pyfar preserves sampling rate."""
        result = test_dataset._to_pyfar(sofa_object)
        assert result["impulse_response"].sampling_rate == sofa_object.Data_SamplingRate

    def test_conversion_to_pyfar_shape(self, sofa_object):
        """Verify _to_pyfar preserves IR data shape."""
        result = test_dataset._to_pyfar(sofa_object)
        assert result["impulse_response"].time.shape == sofa_object.Data_IR.shape


class TestConversionToNumpy:
    """Tests for _to_numpy conversion method."""

    def test_conversion_to_numpy_returns_dict(self, sofa_object):
        """Verify _to_numpy returns a dictionary."""
        result = test_dataset._to_numpy(sofa_object)
        assert isinstance(result, dict)

    def test_conversion_to_numpy_structure(self, sofa_object):
        """Verify _to_numpy returns dict with expected keys."""
        result = test_dataset._to_numpy(sofa_object)
        expected_keys = {
            "impulse_response",
            "source_coordinates",
            "receiver_coordinates",
            "sampling_rate",
        }
        assert set(result.keys()) == expected_keys

    def test_conversion_to_numpy_types(self, sofa_object):
        """Verify _to_numpy returns numpy arrays."""
        result = test_dataset._to_numpy(sofa_object)

        assert isinstance(result["impulse_response"], np.ndarray)
        assert isinstance(result["source_coordinates"], np.ndarray)
        assert isinstance(result["receiver_coordinates"], np.ndarray)
        assert isinstance(result["sampling_rate"], float)

    def test_conversion_to_numpy_values(self, sofa_object):
        """Verify _to_numpy preserves data values."""
        result = test_dataset._to_numpy(sofa_object)

        # Check IR data matches
        np.testing.assert_array_equal(
            result["impulse_response"],
            sofa_object.Data_IR,
        )
        # Check sampling rate
        assert result["sampling_rate"] == sofa_object.Data_SamplingRate


class TestConversionToSofa:
    """Tests for _to_sofa conversion method."""

    def test_conversion_to_sofa_returns_path(self, sofa_object):
        """Verify _to_sofa returns a Path object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sofa_path = Path(tmpdir) / "cached.sofa"
            sf.write_sofa(sofa_path, sofa_object)
            result = test_dataset._to_sofa(sofa_path, Path(tmpdir) / "test.sofa")
            assert isinstance(result, Path)
            assert result.exists()

    def test_conversion_to_sofa_writable(self, sofa_object, tmpdir):
        """Verify _to_sofa returns a readable SOFA file."""
        sofa_path = Path(tmpdir) / "cached.sofa"
        sf.write_sofa(sofa_path, sofa_object)
        result = test_dataset._to_sofa(sofa_path, Path(tmpdir) / "test.sofa")

        loaded_sofa = sf.read_sofa(str(result))
        assert loaded_sofa.Data_IR.shape == sofa_object.Data_IR.shape
        assert loaded_sofa.Data_SamplingRate == sofa_object.Data_SamplingRate

    def test_conversion_to_sofa_preserves_permissions(self, sofa_object, tmp_path):
        """Verify _to_sofa reuses the source SOFA file's permission bits."""
        sofa_path = tmp_path / "ingest.sofa"
        sf.write_sofa(sofa_path, sofa_object)
        sofa_path.chmod(0o640)

        output_path = tmp_path / "test.sofa"
        result = test_dataset._to_sofa(sofa_path, output_path)

        _assert_permissions_preserved(sofa_path, result)


class TestConversionToHdf5:
    """Tests for _to_hdf5 conversion method."""

    def test_conversion_to_hdf5_returns_path(self, sofa_object):
        """Verify _to_hdf5 returns a Path object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sofa_path = Path(tmpdir) / "test.sofa"
            sf.write_sofa(sofa_path, sofa_object)
            result = test_dataset._to_hdf5(sofa_path, Path(tmpdir) / "test.h5")
            assert isinstance(result, Path)
            assert result.exists()

    def test_conversion_to_hdf5_structure(self, sofa_object):
        """Verify _to_hdf5 writes file with expected HDF5 structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sofa_path = Path(tmpdir) / "test.sofa"
            sf.write_sofa(sofa_path, sofa_object)
            result = test_dataset._to_hdf5(sofa_path, Path(tmpdir) / "test.h5")

            with h5py.File(result, "r") as f:
                # Check data group exists
                assert "data" in f
                assert "impulse_response" in f["data"]
                assert "location" in f["data"]
                assert "source" in f["data/location"]
                assert "receiver" in f["data/location"]

                # Check metadata group exists
                assert "metadata" in f
                assert "sampling_rate" in f["metadata"]

    def test_conversion_to_hdf5_optional_fields(self):
        """Verify _to_hdf5 includes optional temperature/humidity fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sofa_path = Path(tmpdir) / "test.sofa"
            with netCDF4.Dataset(sofa_path, "w") as sofa:
                sofa.createDimension("M", 2)
                sofa.createDimension("R", 2)
                sofa.createDimension("N", 3)
                sofa.createDimension("E", 1)
                sofa.createDimension("C", 3)
                sofa.createDimension("I", 1)
                sofa.createVariable("Data.IR", "f8", ("M", "R", "N", "E"))[:] = 0
                sofa.createVariable("SourcePosition", "f8", ("M", "C"))[:] = 0
                sofa.createVariable("ReceiverPosition", "f8", ("R", "C", "I"))[:] = 0
                sofa.createVariable("Data.SamplingRate", "f8", ("I",))[:] = 48_000
                sofa.createVariable("RoomTemperature", "f8", ("M",))[:] = 293.15
                sofa.createVariable("Humidity", "f8", ("M", "I"))[:] = 50
                sofa.createVariable("SpeedOfSound", "f8", ("M", "I"))[:] = 343

            result = test_dataset._to_hdf5(sofa_path, Path(tmpdir) / "test.h5")

            with h5py.File(result, "r") as f:
                assert "temperature" in f["metadata"]
                assert "humidity" in f["metadata"]
                assert "c0" in f["metadata"]

    @pytest.mark.parametrize("emitters", [1, 2])
    @pytest.mark.parametrize("compressed", [False, True])
    def test_conversion_to_hdf5_bounds_large_rows_and_preserves_emitters(
        self, tmp_path, monkeypatch, progress_tasks, emitters, compressed
    ):
        """Stored chunks and contiguous arrays both obey the byte budget, including E>1."""
        shape = (1, 5, 17, emitters)
        expected = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
        expected.flat[-1] = -0.0
        source = tmp_path / "tiled.sofa"
        with netCDF4.Dataset(source, "w") as sofa:
            for name, size in zip(("M", "R", "N", "E"), shape, strict=True):
                sofa.createDimension(name, size)
            sofa.createDimension("C", 3)
            sofa.createDimension("I", 1)
            ir = sofa.createVariable(
                "Data.IR",
                "f8",
                ("M", "R", "N", "E"),
                zlib=compressed,
                chunksizes=(1, 3, 7, emitters) if compressed else None,
            )
            ir[:] = expected
            sofa.createVariable("SourcePosition", "f8", ("M", "C"))[:] = 0
            sofa.createVariable("ReceiverPosition", "f8", ("R", "C", "I"))[:] = 0
            sofa.createVariable("Data.SamplingRate", "f8", ("I",))[:] = 48000
        budget = 64
        writes = []
        setitem = h5py.Dataset.__setitem__

        def bounded_write(destination, selection, values):
            if destination.name == "/data/impulse_response":
                assert values.nbytes <= budget
                writes.append(values.nbytes)
            return setitem(destination, selection, values)

        monkeypatch.setattr(config, "batch_bytes", budget)
        monkeypatch.setattr(h5py.Dataset, "__setitem__", bounded_write)
        output = test_dataset._to_hdf5(source, tmp_path / "bounded.h5")

        assert len(writes) > 1
        assert len(progress_tasks) == 1
        assert progress_tasks[0]["total"] == expected.size
        assert progress_tasks[0]["completed"] == expected.size
        with h5py.File(output) as result:
            actual = result["data/impulse_response"][:]
            wanted = expected[..., 0] if emitters == 1 else expected
            assert actual.shape == wanted.shape
            assert actual.tobytes() == wanted.tobytes()

    def test_conversion_to_hdf5_preserves_permissions(self, sofa_object, tmp_path):
        """Verify _to_hdf5 reuses the source SOFA file's permission bits."""
        sofa_path = tmp_path / "ingest.sofa"
        sf.write_sofa(sofa_path, sofa_object)
        sofa_path.chmod(0o640)

        output_path = tmp_path / "test.h5"
        result = test_dataset._to_hdf5(sofa_path, output_path)

        _assert_permissions_preserved(sofa_path, result)
