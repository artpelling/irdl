"""Tests for the Fraunhofer IIS MP-RIR dataset."""

import numpy as np
import pytest
import sofar as sf

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
