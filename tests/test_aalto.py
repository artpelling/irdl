"""Tests for datasets from the Aalto Acoustics Lab."""

from pathlib import Path

import pytest

from irdl import aalto
from irdl.aalto import MultiRoomTransitionDataset


@pytest.mark.parametrize(
    ("environment", "receiver", "loudspeaker", "expected"),
    [
        ("offices", "kemar", 1, "offices_kemar_ls_1.sofa"),
        ("hallways-lecturehall", "zoom", 4, "hallways-lecturehall_zoom_ls_4.sofa"),
        ("workshops", "kemar", 2, "workshops_kemar_ls_2.sofa"),
    ],
)
def test_mrtd_source_filename(environment, receiver, loudspeaker, expected):
    """Build provider filenames from meaningful public parameters."""
    result = MultiRoomTransitionDataset()._source_filename(
        environment=environment, receiver=receiver, loudspeaker=loudspeaker
    )
    assert result == expected


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"environment": "studio", "receiver": "kemar", "loudspeaker": 1}, "environment must be"),
        ({"environment": "offices", "receiver": "array", "loudspeaker": 1}, "receiver must be"),
        ({"environment": "offices", "receiver": "kemar", "loudspeaker": 0}, "loudspeaker must be"),
        ({"environment": "offices", "receiver": "kemar", "loudspeaker": 1.5}, "loudspeaker must be"),
    ],
)
def test_mrtd_rejects_invalid_parameters(parameters, message):
    """Reject values that cannot identify an MRTD provider file."""
    with pytest.raises(ValueError, match=message):
        MultiRoomTransitionDataset()._validate_params(**parameters)


def test_mrtd_download_fetches_requested_native_sofa(monkeypatch, tmp_path):
    """Fetch only the requested SOFA file through the DOI repository."""
    calls = []
    fake_pooch = object()

    def fake_pooch_from_doi(doi: str, path: Path):
        calls.append(("repository", doi, path))
        return fake_pooch

    def fake_fetch(pooch, filename: str):
        calls.append(("fetch", pooch, filename))
        (tmp_path / filename).touch()
        return str(tmp_path / filename)

    monkeypatch.setattr(aalto, "_pooch_from_doi", fake_pooch_from_doi)
    monkeypatch.setattr(aalto, "_fetch", fake_fetch)
    result = MultiRoomTransitionDataset()._download(tmp_path, environment="workshops", receiver="zoom", loudspeaker=3)
    assert result == tmp_path / "workshops_zoom_ls_3.sofa"
    assert calls == [
        ("repository", "10.5281/zenodo.13341566", tmp_path),
        ("fetch", fake_pooch, "workshops_zoom_ls_3.sofa"),
    ]


def test_mrtd_download_reuses_cached_file(monkeypatch, tmp_path):
    """Avoid repository API calls when the requested SOFA is cached."""
    provider_path = tmp_path / "offices_kemar_ls_1.sofa"
    provider_path.touch()
    monkeypatch.setattr(
        aalto,
        "_pooch_from_doi",
        lambda *_args, **_kwargs: pytest.fail("repository should not be queried"),
    )
    result = MultiRoomTransitionDataset()._download(tmp_path, environment="offices", receiver="kemar", loudspeaker=1)
    assert result == provider_path
