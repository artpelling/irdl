"""Tests for Dataset conversion methods."""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pyfar as pf
import sofar as sf

from irdl.base import BaseDataset


# Create a concrete Dataset subclass for testing conversion methods
# (BaseDataset is abstract and cannot be instantiated directly)
class TestDataset(BaseDataset):
    """Concrete Dataset subclass for testing conversion methods."""

    name = "test"
    doi = "10.0000/test"
    canonical_provider = "depositonce"
    providers = ("depositonce",)

    def _validate_params(self, **dataset_kwargs):
        """No-op validation for test dataset."""

    def _provider_artifact_format(self, provider: str, **_dataset_kwargs):
        """Return provider-side format for test dataset."""
        assert provider == "depositonce"
        return "sofa"

    def _download(self, provider_dir: Path, provider: str, **_dataset_kwargs):
        """No-op download for test dataset."""
        raise NotImplementedError

    def _ingest(self, file_path):
        """No-op ingest for test dataset."""

    def _source_filename(self, **kwargs):  # noqa: ARG002
        """Return test file name."""
        return "test.sofa"


# Create the TestDataset instance to use for all conversion tests
test_dataset = TestDataset()


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
            result = test_dataset._to_sofa(sofa_object, Path(tmpdir), Path(tmpdir) / "test.sofa")
            assert isinstance(result, Path)
            assert result.exists()

    def test_conversion_to_sofa_writable(self, sofa_object, tmpdir):
        """Verify _to_sofa writes a valid SOFA file."""
        result = test_dataset._to_sofa(sofa_object, Path(tmpdir), Path(tmpdir) / "test.sofa")

        # Verify file can be read back
        loaded_sofa = sf.read_sofa(str(result))
        assert loaded_sofa.Data_IR.shape == sofa_object.Data_IR.shape
        assert loaded_sofa.Data_SamplingRate == sofa_object.Data_SamplingRate


class TestConversionToHdf5:
    """Tests for _to_hdf5 conversion method."""

    def test_conversion_to_hdf5_returns_path(self, sofa_object):
        """Verify _to_hdf5 returns a Path object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = test_dataset._to_hdf5(sofa_object, Path(tmpdir), Path(tmpdir) / "test.h5")
            assert isinstance(result, Path)
            assert result.exists()

    def test_conversion_to_hdf5_structure(self, sofa_object):
        """Verify _to_hdf5 writes file with expected HDF5 structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = test_dataset._to_hdf5(sofa_object, Path(tmpdir), Path(tmpdir) / "test.h5")

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

    def test_conversion_to_hdf5_optional_fields(self, sofa_object):
        """Verify _to_hdf5 includes optional temperature/humidity fields when present."""
        # Manually add optional fields to the sofa object
        # Note: This bypasses SOFA's protected mode for testing purposes
        sofa_object._protected = False
        sofa_object.RoomTemperature = np.array([20.0, 20.0])
        sofa_object.Humidity = np.array([50.0, 50.0])
        sofa_object.SpeedOfSound = np.array([[343.0], [343.0]])

        with tempfile.TemporaryDirectory() as tmpdir:
            result = test_dataset._to_hdf5(sofa_object, Path(tmpdir), Path(tmpdir) / "test.h5")

            with h5py.File(result, "r") as f:
                assert "temperature" in f["metadata"]
                assert "humidity" in f["metadata"]
                assert "c0" in f["metadata"]
