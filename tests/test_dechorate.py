"""Tests for the dEchorate dataset integration."""

import pytest

from irdl import sonicom
from irdl.dechorate import DechorateDataset


def test_dechorate_validates_sonicom_selectors():
    """Accept every selector represented by the direct-SOFA provider."""
    dataset = DechorateDataset()
    dataset._validate_params(room_code="000000", source=1, array=1)
    dataset._validate_params(room_code="020002", source=9, array=6)
    with pytest.raises(ValueError, match="source must be"):
        dataset._validate_params(room_code="000000", source=10, array=1)


def test_dechorate_source_filename_matches_sonicom_manifest():
    """Build the direct-SOFA provider filename deterministically."""
    dataset = DechorateDataset()
    filename = "dEchorate_room011110_src4_arr2_mics6-10.sofa"
    assert dataset._source_filename(room_code="011110", source=4, array=2) == filename
    assert dataset.direct_sofa_hash(filename)


def test_dechorate_resolves_sonicom_for_non_raw(monkeypatch):
    """Use SONICOM's current manifest URL for a registered SOFA file."""
    filename = "dEchorate_room000000_src1_arr1_mics1-5.sofa"
    url = f"https://ecosystem.sonicom.eu/data/74/1/{filename}"
    monkeypatch.setattr(sonicom, "_sonicom_manifest", lambda _database_id: {filename: url})
    assert DechorateDataset().direct_sofa_url(filename) == url


def test_dechorate_raw_downloads_only_zenodo_rirs(monkeypatch, tmp_path):
    """Keep raw retrieval on Zenodo and omit unrelated recordings."""
    dataset = DechorateDataset()
    captured = []

    class DummyPooch:
        pass

    def fake_pooch_from_doi(doi, path):
        assert doi == dataset.doi
        assert path == tmp_path
        return DummyPooch()

    def fake_fetch(pup, filename):
        assert isinstance(pup, DummyPooch)
        captured.append(filename)
        return str(tmp_path / filename)

    monkeypatch.setattr("irdl.dechorate._pooch_from_doi", fake_pooch_from_doi)
    monkeypatch.setattr("irdl.dechorate._fetch", fake_fetch)
    assert dataset._download(tmp_path) == tmp_path / "dEchorate_rirs_gzip7.hdf5"
    assert captured == ["dEchorate_rirs_gzip7.hdf5"]
