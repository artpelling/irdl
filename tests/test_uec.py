"""Tests for the University of Electro-Communications 3D Meshgrid dataset."""

import h5py
import numpy as np
import sofar as sf

from irdl import uec
from irdl.uec import Meshgrid3dDataset

_SAMPLING_RATE = 16000


def _write_meshgrid_hdf5(path):
    """Write a small provider artifact in the MATLAB HDF5 layout."""
    responses = np.arange(20, dtype=np.float32).reshape(4, 5)
    grid = np.array([[0.0, 0.2, 0.4, 0.6], [1.0, 1.2, 1.4, 1.6], [2.0, 2.2, 2.4, 2.6]], dtype=np.float32)
    with h5py.File(path, "w") as hdf5:
        hdf5.create_dataset("rir", data=responses)
        hdf5.create_dataset("grid", data=grid)
        hdf5.attrs["fs"] = np.float32(_SAMPLING_RATE)
        hdf5.attrs["c"] = np.float32(343)
        hdf5.attrs["sp_pos"] = np.array([5.54, 0.94, 1.42], dtype=np.float32)
        hdf5.attrs["room_size"] = np.array([8.4, 6.14, 2.66], dtype=np.float32)
    return responses, grid.T


def test_meshgrid_filename_and_speaker_validation():
    """Map each loudspeaker selector to its provider filename."""
    dataset = Meshgrid3dDataset()
    assert dataset._source_filename(speaker=3) == "A_ir_16k_sp3.h5"
    dataset._validate_params(speaker=4)

    for speaker in (0, 5, "1"):
        try:
            dataset._validate_params(speaker=speaker)
        except (ValueError, TypeError):
            pass
        else:
            msg = "invalid speaker accepted"
            raise AssertionError(msg)


def test_meshgrid_downloads_selected_zenodo_file(monkeypatch, tmp_path):
    """Use the Dataset DOI and selected provider filename for acquisition."""
    calls = []
    fake_pooch = object()

    def fake_pooch_from_doi(doi, path):
        calls.append(("pooch", doi, path))
        return fake_pooch

    def fake_fetch(pooch, filename):
        calls.append(("fetch", pooch, filename))

    monkeypatch.setattr(uec, "_pooch_from_doi", fake_pooch_from_doi)
    monkeypatch.setattr(uec, "_fetch", fake_fetch)

    assert Meshgrid3dDataset()._download(tmp_path, speaker=2) == tmp_path / "A_ir_16k_sp2.h5"
    assert calls == [
        ("pooch", Meshgrid3dDataset.doi, tmp_path),
        ("fetch", fake_pooch, "A_ir_16k_sp2.h5"),
    ]


def test_meshgrid_ingest_streams_loudspeaker_sources_and_microphone_receivers(tmp_path):
    """Preserve the physical loudspeaker and moving microphone-grid geometry."""
    provider = tmp_path / "A_ir_16k_sp1.h5"
    responses, grid = _write_meshgrid_hdf5(provider)
    sofa_path = tmp_path / "meshgrid.sofa"

    Meshgrid3dDataset()._ingest(provider, sofa_path)
    sofa = sf.read_sofa(sofa_path, verify=False)

    assert sofa.Data_IR.shape == (1, 4, 5, 1)
    np.testing.assert_allclose(sofa.Data_IR[0, :, :, 0], responses)
    np.testing.assert_allclose(sofa.SourcePosition, [[5.54, 0.94, 1.42]])
    np.testing.assert_allclose(sofa.ReceiverPosition[:, :, 0], grid)
    assert sofa.Data_SamplingRate == _SAMPLING_RATE
    assert sofa.GLOBAL_License == "CC BY 4.0"
    Meshgrid3dDataset()._verify_sofa_convention(sofa_path)
