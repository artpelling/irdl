"""Tests for the RWTH Aachen MIRD dataset."""

import io
import logging
from zipfile import ZipFile

import numpy as np
import pytest
import sofar as sf
from scipy.io import savemat

from irdl import iks
from irdl.iks import MirdDataset
from irdl.utils import load_hash_registry

SELECTORS = {"t60": 0.16, "spacing": 8, "distance": "both", "remove_delay": True}
AZIMUTHS = (*range(0, 91, 15), *range(270, 360, 15))


def test_mird_names_match_the_provider_layout():
    """Build provider archive and MAT names from simplified selectors."""
    dataset = MirdDataset()
    assert dataset._source_filename(**SELECTORS) == "short-8-both.mird"
    assert dataset._source_filename(**(SELECTORS | {"spacing": 3})) == "short-3-both.mird"
    assert dataset._source_filename(**(SELECTORS | {"t60": 0.61, "distance": 1})) == "long-8-1.mird"
    assert dataset._source_filename(**(SELECTORS | {"remove_delay": False})) == "short-8-both-nodelay.mird"


@pytest.mark.parametrize(
    ("parameter", "value"),
    [("t60", 0.2), ("spacing", 5), ("distance", 3), ("remove_delay", "yes")],
)
def test_mird_rejects_unknown_selectors(parameter, value):
    """Reject unavailable reverberation, array, and distance selections."""
    selectors = SELECTORS | {parameter: value}
    with pytest.raises((ValueError, TypeError), match=parameter):
        MirdDataset()._validate_params(**selectors)


def test_mird_has_hashes_for_every_provider_archive():
    """Pin all three T60 and three spacing archives."""
    hashes = load_hash_registry("mird")
    assert len(hashes) == len(MirdDataset._t60s) * len(MirdDataset._spacings)
    assert all(filename.endswith(".zip") for filename in hashes)


def test_mird_downloads_one_hash_verified_archive(monkeypatch, tmp_path):
    """Retrieve the selected archive through its pinned SHA-256 digest."""
    calls = []
    fake_pooch = object()

    def fake_registry(**kwargs):
        calls.append(("registry", kwargs))
        return fake_pooch

    def fake_fetch(pooch, filename):
        calls.append(("fetch", pooch, filename))
        (tmp_path / filename).touch()

    monkeypatch.setattr(iks, "_pooch_from_static_registry", fake_registry)
    monkeypatch.setattr(iks, "_fetch", fake_fetch)
    result = MirdDataset()._download(tmp_path, **SELECTORS)
    expected = "Impulse_response_Acoustic_Lab_Bar-Ilan_University__Reverberation_0.160s__8-8-8-8-8-8-8.zip"
    assert result == tmp_path / expected
    assert calls == [
        (
            "registry",
            {
                "path": tmp_path,
                "registry": {expected: "sha256:eba8d526d09efd03a770f413e1b253b041106cff1aac5aacba17d21ff6f51965"},
                "urls": {expected: f"{MirdDataset._download_root}/{expected}"},
            },
        ),
        ("fetch", fake_pooch, expected),
    ]


def test_mird_extracts_all_microphones_and_selected_sources(tmp_path, caplog):
    """Create a SOFA with eight microphones and all azimuths at both distances."""
    dataset = MirdDataset()
    archive_path = tmp_path / "mird.zip"
    response = np.arange(32, dtype=float).reshape(4, 8)
    with ZipFile(archive_path, "w") as archive:
        for distance in (1, 2):
            for azimuth in AZIMUTHS:
                payload = io.BytesIO()
                savemat(
                    payload,
                    {
                        "impulse_response": response + distance + azimuth,
                        "metapar": {"distance": distance, "azimuth": azimuth},
                        "simpar": {"int_delay": 1},
                    },
                )
                archive.writestr(f"MIRD_{distance}m_{azimuth:03d}.mat", payload.getvalue())

    ingest_path = tmp_path / "ingest"
    assert dataset._process(archive_path, ingest_path, **SELECTORS) == ingest_path
    sofa_path = tmp_path / "mird.sofa"
    caplog.set_level(logging.INFO, logger="irdl")
    dataset._ingest(ingest_path, sofa_path, **SELECTORS)

    sofa = sf.read_sofa(sofa_path)
    assert "Removing excess measurement delay" in caplog.text
    assert sofa.Data_IR.shape == (26, 8, 3, 1)
    np.testing.assert_allclose(np.linalg.norm(sofa.SourcePosition, axis=1), [1] * 13 + [2] * 13)
    np.testing.assert_allclose(sofa.ReceiverPosition[:, 1, 0], np.arange(-0.28, 0.29, 0.08))

    dataset._ingest(ingest_path, tmp_path / "mird-with-delay.sofa", **(SELECTORS | {"remove_delay": False}))
    assert sf.read_sofa(tmp_path / "mird-with-delay.sofa").Data_IR.shape == (26, 8, 4, 1)
