"""Tests for the University of Electro-Communications 3D Meshgrid dataset."""

from threading import get_ident

import h5py
import netCDF4
import numpy as np
import pytest
import sofar as sf

from irdl import ista, uec
from irdl.uec import Meshgrid3dDataset

_SAMPLING_RATE = 16000


def _write_meshgrid_hdf5(path, *, receivers=4, samples=5):
    """Write a small provider artifact in the MATLAB HDF5 layout."""
    responses = np.arange(receivers * samples, dtype=np.float32).reshape(receivers, samples)
    grid = np.arange(receivers, dtype=np.float32)[None, :] * 0.2 + np.arange(3, dtype=np.float32)[:, None]
    with h5py.File(path, "w") as hdf5:
        hdf5.create_dataset("rir", data=responses)
        hdf5.create_dataset("grid", data=grid)
        # MATLAB stores scalar attributes as one-element HDF5 arrays.
        hdf5.attrs["fs"] = np.array([_SAMPLING_RATE], dtype=np.float32)
        hdf5.attrs["c"] = np.array([343], dtype=np.float32)
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


@pytest.mark.parametrize("samples", [4000, 200_000])
def test_meshgrid_chunks_follow_receivers_and_preserve_edge_values(tmp_path, samples):
    """Receiver and sample tails round-trip without changing physical dimensions."""
    provider = tmp_path / "provider.h5"
    responses, grid = _write_meshgrid_hdf5(provider, receivers=5, samples=samples)
    responses[-1, -1] = -0.0
    with h5py.File(provider, "r+") as source:
        source["rir"][:] = responses
    sofa_path = tmp_path / "meshgrid.sofa"
    dataset = Meshgrid3dDataset()
    dataset._chunk_size = 2

    dataset._ingest(provider, sofa_path)

    with h5py.File(sofa_path) as sofa:
        ir = sofa["Data.IR"]
        assert ir.shape == (1, 5, samples, 1)
        assert ir.chunks[0] == ir.chunks[3] == 1
        assert ir.chunks[1] <= dataset._chunk_size
        assert np.prod(ir.chunks) * ir.dtype.itemsize <= 1024**2
        np.testing.assert_array_equal(ir[0, :, :, 0].view(np.uint64), responses.astype(np.float64).view(np.uint64))
        np.testing.assert_array_equal(sofa["ReceiverPosition"][:, :, 0], grid)
    dataset._verify_payload(sofa_path, provider)


@pytest.mark.parametrize("failure", ["encode", "Data.IR", "ReceiverPosition"])
def test_failed_meshgrid_ingest_preserves_existing_artifact(tmp_path, monkeypatch, failure):
    """Compression and payload errors cannot promote a partial Meshgrid artifact."""
    provider = tmp_path / "provider.h5"
    _write_meshgrid_hdf5(provider, receivers=5)
    sofa_path = tmp_path / "retained.sofa"
    original = b"existing artifact"
    sofa_path.write_bytes(original)
    dataset = Meshgrid3dDataset()
    verify = dataset._verify_payload

    def fail_compression(*_args):
        msg = "injected compression failure"
        raise RuntimeError(msg)

    def corrupt_and_verify(staged, source, **kwargs):
        with netCDF4.Dataset(staged, "a") as sofa:
            index = (0, -1, -1, 0) if failure == "Data.IR" else (-1, 0, 0)
            sofa.variables[failure][index] = -99.0
        verify(staged, source, **kwargs)

    if failure == "encode":
        monkeypatch.setattr(ista.zlib, "compress", fail_compression)
    else:
        monkeypatch.setattr(dataset, "_verify_payload", corrupt_and_verify)
    with pytest.raises((RuntimeError, ValueError), match=r"injected compression failure|Meshgrid payload mismatch"):
        dataset._ingest(provider, sofa_path)

    assert sofa_path.read_bytes() == original
    assert not list(tmp_path.glob(".retained-*"))


def test_meshgrid_shared_encoder_keeps_hdf5_on_caller_thread(tmp_path, monkeypatch):
    """Only compression runs in workers; HDF5 reads stay on the caller thread."""
    provider = tmp_path / "provider.h5"
    _write_meshgrid_hdf5(provider, receivers=5)
    dataset = Meshgrid3dDataset()
    dataset._chunk_size = 2
    caller = get_ident()
    threads = set()
    original_read = h5py.Dataset.__getitem__
    original_compress = ista.zlib.compress

    def read(source, key):
        assert get_ident() == caller
        return original_read(source, key)

    def compress(*args):
        threads.add(get_ident())
        return original_compress(*args)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", read)
    monkeypatch.setattr(ista.zlib, "compress", compress)
    monkeypatch.setattr(ista.os, "cpu_count", lambda: 2)
    dataset._ingest(provider, tmp_path / "threaded.sofa")

    assert threads
    assert caller not in threads


def test_meshgrid_hdf5_export_uses_stored_ir_tiles(tmp_path, monkeypatch):
    """A single measurement with many receivers must not become one huge read."""
    provider = tmp_path / "provider.h5"
    expected, _ = _write_meshgrid_hdf5(provider, receivers=5)
    dataset = Meshgrid3dDataset()
    dataset._chunk_size = 2
    sofa_path = tmp_path / "meshgrid.sofa"
    dataset._ingest(provider, sofa_path)
    sizes = []
    original_write = h5py.Dataset.__setitem__

    def write(destination, key, data):
        if destination.name == "/data/impulse_response":
            sizes.append(data.size)
        return original_write(destination, key, data)

    monkeypatch.setattr(h5py.Dataset, "__setitem__", write)
    output = tmp_path / "meshgrid.h5"
    dataset._to_hdf5(sofa_path, output)

    assert len(sizes) == (expected.shape[0] + dataset._chunk_size - 1) // dataset._chunk_size
    assert max(sizes) <= dataset._chunk_size * expected.shape[1]
    with h5py.File(output) as result:
        np.testing.assert_array_equal(result["data/impulse_response"][0], expected)
