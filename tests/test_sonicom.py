"""Tests for shared SONICOM dataset support."""

from pathlib import Path

import pytest

from irdl.akt import HutubsDataset
from irdl.sonicom import SonicomBaseDataset


class TestSonicomBaseDataset:
    """Tests for SONICOM shared dataset behavior."""

    def test_base_class_remains_abstract(self):
        """Verify SonicomBaseDataset cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            SonicomBaseDataset()


class TestHutubsSonicomProvider:
    """Tests for HUTUBS using SONICOM as an additional Provider."""

    def test_checksum_uses_dataset_name_prefix_and_database_id(self, monkeypatch):
        """Verify HUTUBS derives SONICOM metadata from shared base conventions."""
        monkeypatch.setattr(
            "irdl.sonicom.load_hash_registry",
            lambda _provider_name: {"hutubs/pp1_HRIRs_measured.sofa": "sha256:abc"},
        )

        dataset = HutubsDataset()

        assert dataset._sonicom_database_url() == "https://ecosystem.sonicom.eu/databases/76"
        assert dataset._sonicom_checksum(subject=1, kind="measured") == "sha256:abc"

    def test_provider_artifact_format_differs_by_provider(self):
        """Verify HUTUBS distinguishes canonical ZIP and SONICOM SOFA artifacts."""
        dataset = HutubsDataset()

        assert dataset._provider_artifact_format("depositonce") == "zip"
        assert dataset._provider_artifact_format("sonicom") == "sofa"

    def test_download_resolves_sonicom_file_from_database_manifest(self, monkeypatch, tmp_path):
        """Verify HUTUBS SONICOM resolves one SOFA file from the SONICOM database."""
        captured = {}

        class DummyPooch:
            def __init__(self):
                self.path = tmp_path

        def fake_pooch_from_sonicom_database(path, database_url, fname, checksum=None):
            captured["path"] = path
            captured["database_url"] = database_url
            captured["fname"] = fname
            captured["checksum"] = checksum
            return DummyPooch()

        def fake_fetch(pup, fname):
            captured["fetch_fname"] = fname
            target = Path(pup.path) / fname
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("ok")
            return str(target)

        monkeypatch.setattr("irdl.sonicom._pooch_from_sonicom_database", fake_pooch_from_sonicom_database)
        monkeypatch.setattr("irdl.sonicom._fetch", fake_fetch)
        monkeypatch.setattr(
            "irdl.sonicom.load_hash_registry",
            lambda _provider_name: {"hutubs/pp1_HRIRs_measured.sofa": "sha256:abc"},
        )

        result = HutubsDataset()._download(tmp_path, provider="sonicom", subject=1, kind="measured")

        assert result == tmp_path / "pp1_HRIRs_measured.sofa"
        assert captured == {
            "path": tmp_path,
            "database_url": "https://ecosystem.sonicom.eu/databases/76",
            "fname": "pp1_HRIRs_measured.sofa",
            "checksum": "sha256:abc",
            "fetch_fname": "pp1_HRIRs_measured.sofa",
        }
