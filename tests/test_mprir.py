"""Tests for the Fraunhofer IIS MP-RIR dataset."""

import h5py as h5
import numpy as np
import pytest
import sofar as sf

from irdl import _sofa, config
from irdl.datasets import fraunhofer
from irdl.datasets.fraunhofer import MpRirDataset

SELECTORS = {"source": "S5", "remove_delay": True}
T_GUARD, T_SYSTEM = 4, 3
FS = 48000
HEIGHTS = 0.7 + 0.15 * np.arange(8)


@pytest.fixture
def provider_dir(tmp_path):
    """Write small provider files with the MP-RIR layout for loudspeaker S5."""
    source_dir = tmp_path / "provider" / "S5"
    source_dir.mkdir(parents=True)
    xy = np.array([[0.26, 0.32], [0.41, 0.32], [0.26, 0.47], [5.66, 6.17], [3.41, 5.87]])
    mxyz = np.empty((len(xy), 8, 3), dtype=np.float32)
    mxyz[..., :2] = xy[:, np.newaxis]
    mxyz[..., 2] = HEIGHTS
    rirs = np.arange(len(xy) * 8 * 20, dtype=np.float32).reshape(len(xy), 8, 20) / 1000
    np.save(source_dir / "Mxyz.npy", mxyz)
    np.save(source_dir / "S5_Mrir.npy", rirs)
    np.savez(
        source_dir / "Setup.npz",
        angles_speaker={f"S{k}": 90.0 * k for k in range(1, 9)},
        coord_speaker_center={f"S{k}": np.array([k, 7.0, 0.725]) for k in range(1, 9)},
        coord_polygon=np.array([[0.0, 0.0], [0.0, 7.32], [4.93, 7.31], [6.49, 0.0]]),
        fs=np.array(FS, dtype=np.int32),
        T_guard=np.array(T_GUARD, dtype=np.int32),
        T_system=np.array(T_SYSTEM),
    )
    return source_dir


def test_mprir_names_include_source_and_delay():
    """Build one cache name per loudspeaker and delay handling."""
    dataset = MpRirDataset()
    assert dataset._source_filename(**SELECTORS) == "S5.mprir"
    assert dataset._source_filename(**(SELECTORS | {"remove_delay": False})) == "S5-nodelay.mprir"


@pytest.mark.parametrize(("parameter", "value"), [("source", "S9"), ("source", 1), ("remove_delay", "yes")])
def test_mprir_rejects_unknown_selectors(parameter, value):
    """Reject unavailable loudspeakers and non-boolean delay flags."""
    with pytest.raises((ValueError, TypeError), match=parameter):
        MpRirDataset()._validate_params(**(SELECTORS | {parameter: value}))


def test_mprir_downloads_source_with_shared_files(monkeypatch, tmp_path):
    """Fetch the loudspeaker RIRs, geometry, and setup into one folder per loudspeaker."""
    calls = []
    fake_pooch = object()

    def fake_pooch_from_doi(doi, path):
        calls.append(("pooch", doi, path))
        path.mkdir()
        return fake_pooch

    def fake_fetch(pooch, filename):
        calls.append(("fetch", pooch, filename))
        (tmp_path / "S5" / filename).touch()

    monkeypatch.setattr(fraunhofer, "_pooch_from_doi", fake_pooch_from_doi)
    monkeypatch.setattr(fraunhofer, "_fetch", fake_fetch)
    assert MpRirDataset()._download(tmp_path, **SELECTORS) == tmp_path / "S5"
    assert calls == [
        ("pooch", MpRirDataset.doi, tmp_path / "S5"),
        ("fetch", fake_pooch, "S5_Mrir.npy"),
        ("fetch", fake_pooch, "Mxyz.npy"),
        ("fetch", fake_pooch, "Setup.npz"),
    ]

    calls.clear()
    MpRirDataset()._download(tmp_path, **SELECTORS)
    assert calls == []


@pytest.mark.parametrize("remove_delay", [True, False])
def test_mprir_ingest_reconstructs_grid_and_delay(provider_dir, tmp_path, remove_delay):
    """Write a SingleRoomSRIR SOFA whose receiver positions recover Mxyz."""
    dataset = MpRirDataset()
    selectors = SELECTORS | {"remove_delay": remove_delay}
    ingest_path = tmp_path / "ingest" / dataset._source_filename(**selectors)
    assert dataset._process(provider_dir, ingest_path, **selectors) == ingest_path
    sofa_path = dataset._ingest(ingest_path, tmp_path / "mprir.sofa", **selectors)
    dataset._verify_sofa_convention(sofa_path)

    sofa = sf.read_sofa(sofa_path)
    rirs = np.load(provider_dir / "S5_Mrir.npy")
    mxyz = np.load(provider_dir / "Mxyz.npy")
    start = T_GUARD + T_SYSTEM if remove_delay else 0
    assert sofa.GLOBAL_SOFAConventions == "SingleRoomSRIR"
    np.testing.assert_array_equal(sofa.Data_IR, rirs[:, :, start:])
    np.testing.assert_array_equal(sofa.Data_Delay, np.full((1, 8), T_GUARD + T_SYSTEM - start))
    assert sofa.Data_SamplingRate == FS
    np.testing.assert_array_equal(sofa.ListenerPosition, np.zeros((5, 3)))
    np.testing.assert_allclose(sofa.ReceiverPosition.transpose(2, 0, 1), mxyz)
    np.testing.assert_allclose(sofa.SourcePosition, np.tile([5.0, 7.0, 0.725], (5, 1)))
    np.testing.assert_allclose(sofa.SourceView, [[np.cos(np.deg2rad(450)), np.sin(np.deg2rad(450)), 0.0]], atol=1e-12)


@pytest.mark.parametrize("budget", [8, 80, 1024])
@pytest.mark.parametrize("dtype", ["f4", "f8"])
@pytest.mark.parametrize("remove_delay", [True, False])
def test_mprir_uses_shared_writer_with_bounded_float64_chunks(
    provider_dir, tmp_path, monkeypatch, budget, dtype, remove_delay
):
    """FIR chunks obey the byte budget and losslessly widen Provider values to double."""
    rirs = np.load(provider_dir / "S5_Mrir.npy").astype(dtype)
    rirs.flat[-1] = np.nextafter(np.array(1, dtype=dtype), np.array(2, dtype=dtype))
    np.save(provider_dir / "S5_Mrir.npy", rirs)
    monkeypatch.setattr(config, "batch_bytes", budget)
    original_write = fraunhofer.write_ir_chunks
    assert original_write is _sofa.write_ir_chunks
    assert fraunhofer.choose_ir_chunk_shape is _sofa.choose_ir_chunk_shape
    offsets = []

    def observed_write(path, chunks):
        def checked_chunks():
            for offset, values in chunks:
                assert values.ndim == 3  # noqa: PLR2004 - FIR: M, R, N.
                assert isinstance(values, np.memmap)
                assert values.size * np.dtype("f8").itemsize <= budget
                offsets.append(offset)
                yield offset, values

        return original_write(path, checked_chunks())

    monkeypatch.setattr(fraunhofer, "write_ir_chunks", observed_write)
    selectors = SELECTORS | {"remove_delay": remove_delay}
    path = MpRirDataset()._ingest(provider_dir, tmp_path / "bounded.sofa", **selectors)
    start = T_GUARD + T_SYSTEM if remove_delay else 0
    with h5.File(path) as sofa:
        ir = sofa["Data.IR"]
        assert ir.dtype == np.dtype("f8")
        assert ir.shape == rirs[:, :, start:].shape
        assert np.prod(ir.chunks) * ir.dtype.itemsize <= budget
        assert offsets == [tuple(axis.start for axis in selection) for selection in ir.iter_chunks()]
        np.testing.assert_array_equal(ir[:], rirs[:, :, start:].astype("f8"))


@pytest.mark.parametrize("remove_delay", [True, False])
def test_mprir_get_retains_sofa_and_exports_float64_hdf5(provider_dir, tmp_path, monkeypatch, remove_delay):
    """Get validates and retains the FIR artifact before deriving a float64 HDF5 output."""
    monkeypatch.setattr(MpRirDataset, "_download", lambda *_args, **_kwargs: provider_dir)
    path = MpRirDataset.get(source="S5", cache_dir=tmp_path / "cache", output_format="hdf5", remove_delay=remove_delay)
    start = T_GUARD + T_SYSTEM if remove_delay else 0
    expected = np.load(provider_dir / "S5_Mrir.npy")[:, :, start:]
    sofa_path = path.with_suffix(".sofa")
    assert sofa_path.exists()
    with h5.File(sofa_path) as sofa, h5.File(path) as exported:
        assert sofa["Data.IR"].dtype == exported["data/impulse_response"].dtype == np.dtype("f8")
        np.testing.assert_array_equal(exported["data/impulse_response"][:], expected)
        np.testing.assert_array_equal(sofa["Data.Delay"][:], np.full((1, 8), T_GUARD + T_SYSTEM - start))


@pytest.mark.parametrize("failure", ["source", "compression", "write", "convention"])
def test_mprir_get_discards_failed_output(provider_dir, tmp_path, monkeypatch, failure):
    """Failures in the shared writer or convention validation never promote partial artifacts."""
    monkeypatch.setattr(MpRirDataset, "_download", lambda *_args, **_kwargs: provider_dir)

    def fail(*_args, **_kwargs):
        msg = "injected failure"
        raise OSError(msg)

    if failure == "source":
        monkeypatch.setattr(MpRirDataset, "_mprir_ir_chunks", fail)
    elif failure == "compression":
        monkeypatch.setattr(_sofa, "_encode_ir_chunk", fail)
    elif failure == "write":
        monkeypatch.setattr(fraunhofer, "write_ir_chunks", fail)
    else:
        monkeypatch.setattr(MpRirDataset, "_verify_sofa_convention", fail)

    with pytest.raises(OSError, match="injected failure"):
        MpRirDataset.get(source="S5", cache_dir=tmp_path / "cache", output_format="sofa")
    assert list((tmp_path / "cache" / "MPRIR" / "output").iterdir()) == []
