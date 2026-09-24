"""Tests for ISTA-specific file processing helpers."""

import os
from pathlib import Path

import h5py
import numpy as np

from irdl import sonicom
from irdl.ista import MiracleDataset, SrirachaDataset


def _assert_permissions_preserved(source_mode: int, target_path: Path) -> None:
    """Assert permission preservation with Windows-compatible semantics."""
    target_mode = target_path.stat().st_mode & 0o777
    if os.name == "nt":
        assert bool(source_mode & 0o200) == bool(target_mode & 0o200)
    else:
        assert target_mode == source_mode


def _write_ista_hdf5(path: Path, *, n_sources: int, start: int = 0) -> None:
    """Write a minimal ISTA-style HDF5 file for processing tests."""
    impulse_response = np.arange(start, start + n_sources * 2 * 8, dtype=np.float32).reshape(n_sources, 2, 8)
    source_coordinates = np.arange(start, start + n_sources * 3, dtype=np.float64).reshape(n_sources, 3)
    receiver_coordinates = np.array([[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]], dtype=np.float64)
    c0 = np.full(n_sources, 343.0, dtype=np.float32)
    temperature = np.full(n_sources, 20.0, dtype=np.float32)

    with h5py.File(path, "w") as handle:
        data_group = handle.create_group("data")
        data_group.create_dataset("impulse_response", data=impulse_response)
        location_group = data_group.create_group("location")
        location_group.create_dataset("source", data=source_coordinates)
        location_group.create_dataset("receiver", data=receiver_coordinates)

        metadata_group = handle.create_group("metadata")
        metadata_group.create_dataset("sampling_rate", data=44100)
        metadata_group.create_dataset("c0", data=c0)
        metadata_group.create_dataset("temperature", data=temperature)


class TestMiracleProcessing:
    """Tests for MIRACLE-specific processing helpers."""

    def test_resolves_full_scenarios_from_sonicom(self, monkeypatch):
        """Use SONICOM's native SOFA files except for artificial splits."""
        urls = {
            "A1.sofa": "https://ecosystem.sonicom.eu/data/98/28532/62248/A1.sofa",
            "A2.sofa": "https://ecosystem.sonicom.eu/data/98/28533/62249/A2.sofa",
            "D1.sofa": "https://ecosystem.sonicom.eu/data/98/28534/62250/D1.sofa",
            "R2.sofa": "https://ecosystem.sonicom.eu/data/98/28535/62251/R2.sofa",
        }
        monkeypatch.setattr(sonicom, "_sonicom_manifest", lambda _database_id: urls)
        dataset = MiracleDataset()

        for scenario, url in urls.items():
            direct_sofa = dataset._direct_sofa(f"{scenario.removesuffix('.sofa')}.h5")
            assert direct_sofa[0] == url
            assert direct_sofa[1] is not None
        assert dataset._direct_sofa("A1-C1.h5") == (None, None)

    def test_extract_split_preserves_permissions(self, tmp_path):
        """Verify extracted split files reuse the source file's permission bits."""
        dataset = MiracleDataset()
        provider_artifact = tmp_path / "A1.h5"
        _write_ista_hdf5(provider_artifact, n_sources=4)
        provider_artifact.chmod(0o640)

        output_path = tmp_path / "ingest" / "A1-C1.h5"
        result = dataset._extract_split(provider_artifact, "C1", output_path)

        _assert_permissions_preserved(provider_artifact.stat().st_mode & 0o777, result)

    def test_sofa_room_corners_are_relative_to_each_array(self, tmp_path):
        """Write MIRACLE's chamber bounds in the local SOFA coordinate frame."""
        dataset = MiracleDataset()
        expected = {
            "A1": ((-4.15, -3.0, -7.5355), (4.15, 4.4, 5.9645)),
            "A2": ((-4.15, -3.0, -6.8), (4.15, 4.4, 6.7)),
            "D1": ((-4.15, -3.0, -7.531), (4.15, 4.4, 5.969)),
            "R2": ((-4.15, -3.0, -6.8), (4.15, 4.4, 6.7)),
        }

        for scenario, (corner_a, corner_b) in expected.items():
            hdf5_path = tmp_path / f"{scenario}.h5"
            sofa_path = tmp_path / f"{scenario}.sofa"
            _write_ista_hdf5(hdf5_path, n_sources=1)

            dataset._ingest(hdf5_path, sofa_path, scenario=scenario)

            with h5py.File(sofa_path, "r") as sofa:
                assert sofa.attrs["RoomType"].decode() == "shoebox"
                np.testing.assert_allclose(sofa["RoomCornerA"][:], [corner_a])
                np.testing.assert_allclose(sofa["RoomCornerB"][:], [corner_b])


class TestSrirachaProcessing:
    """Tests for SRIRACHA-specific processing helpers."""

    def test_full_plane_process_keeps_provider_artifact_set(self, tmp_path):
        """Verify full-plane processing keeps split files as the ingest artifact set."""
        dataset = SrirachaDataset()
        provider_dir = tmp_path / "provider"
        provider_dir.mkdir()

        for index, split in enumerate(("C1", "C2", "C3", "C4")):
            split_path = provider_dir / f"SR1-{split}.h5"
            _write_ista_hdf5(split_path, n_sources=1, start=index * 100)
            split_path.chmod(0o640)

        ingest_path = tmp_path / "ingest" / "SR1.h5"
        ingest_path.parent.mkdir()
        result = dataset._process(provider_dir, ingest_path, scenario="SR1", dataset_split=None)

        assert result == provider_dir
        assert not ingest_path.exists()
        assert len(list(provider_dir.glob("SR1-C*.h5"))) == len(("C1", "C2", "C3", "C4"))
