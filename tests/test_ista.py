"""Tests for ISTA-specific file processing helpers."""

import os
from pathlib import Path

import h5py
import numpy as np
import pytest
import sofar as sf

from irdl.datasets import sonicom
from irdl.datasets.ista import MiracleDataset, SrirachaDataset


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

    @pytest.mark.parametrize(
        ("scenario", "dataset_split", "bitstream"),
        [
            ("SR1", None, "230377c8-f7bd-4e70-abad-f824464d2860"),
            ("SR1-D", None, "72d8ea5b-959e-411d-b185-61d16d5611ce"),
            ("SR1", "C1", "3bb17656-a907-4ff0-b682-00e988f3358d"),
            ("SR1", "C2", "d793a95a-cda7-40c2-aa93-1fda9ed01fc7"),
            ("SR1", "C3", "ebaf10d1-8eba-47ae-a024-100e4865ca37"),
            ("SR1", "C4", "cf87ca0d-a605-483d-9b61-cbca9efb3668"),
            ("SR2", None, "16dfc746-d468-42b3-94d7-bb8b2f62c485"),
            ("SR2-D", None, "16928a00-3307-4aa8-9b10-415292f41b8c"),
            ("SR2", "C1", "4a87658c-0631-4630-8e20-ba5d8f95cd4a"),
            ("SR2", "C2", "da80797a-59e6-426e-b307-5ba05f6e298d"),
            ("SR2", "C3", "b71a3b68-2225-4a3d-b990-78a53f032ad0"),
            ("SR2", "C4", "65b3fe7f-7325-4cee-8e22-e227b95edc16"),
            ("SRA1", None, "fd1ab25c-46ad-43f1-a9c2-0abb30a6a8a5"),
            ("SRA1-D", None, "6961647d-b9e9-460f-bcd0-c23610554cad"),
            ("SRA1", "C1", "8bfa27bc-326e-42b5-aff4-9597bb7c6123"),
            ("SRA1", "C2", "ae7c5065-2c46-4ea8-95e4-1ecde6bb7c34"),
            ("SRA1", "C3", "e789ac85-ddb8-4aa6-b133-e9b2b3e00f4f"),
            ("SRA1", "C4", "713e73c9-6017-4492-8b34-83562d34b9f4"),
            ("SRA2", None, "40f7adf8-7ba8-452a-8d24-73085d2cbbfa"),
            ("SRA2-D", None, "0e76d7a6-6f3b-4fea-b9bb-145518be6f5d"),
            ("SRA2", "C1", "4219d3a5-b683-406c-948c-2ac00d0bcc03"),
            ("SRA2", "C2", "5887c8f5-b60c-4fc7-b7da-b2fed88c2223"),
            ("SRA2", "C3", "c741ccee-b2c3-4b91-be7b-6aed048566b2"),
            ("SRA2", "C4", "697cc94a-52d5-4f33-b8e8-f01d69a23882"),
        ],
    )
    def test_sofa_references_selection_geometry_without_changing_payload(
        self, tmp_path, scenario, dataset_split, bitstream
    ):
        """All 24 selections use their assigned DAE, not placeholder shoebox bounds."""
        dataset = SrirachaDataset()
        kwargs = {"scenario": scenario, "dataset_split": dataset_split}
        if dataset_split is None and not scenario.endswith("-D"):
            artifact = tmp_path / "provider"
            artifact.mkdir()
            inputs = [artifact / f"{scenario}-C{i}.h5" for i in range(1, 5)]
        else:
            artifact = tmp_path / dataset._source_filename(**kwargs)
            inputs = [artifact]
        expected_ir, expected_source = [], []
        for index, path in enumerate(inputs):
            _write_ista_hdf5(path, n_sources=1, start=index * 100)
            with h5py.File(path) as source:
                expected_ir.append(source["data/impulse_response"][:])
                expected_source.append(source["data/location/source"][:])
                expected_receiver = source["data/location/receiver"][:]
        output = tmp_path / "selection.sofa"

        dataset._ingest(artifact, output, **kwargs)

        with h5py.File(output) as sofa:
            assert sofa.attrs["RoomType"].decode() == "dae"
            assert sofa.attrs["RoomGeometry"].decode() == (
                f"https://depositonce.tu-berlin.de/bitstreams/{bitstream}/download"
            )
            assert {"RoomCornerA", "RoomCornerB", "RoomCorners"}.isdisjoint(sofa)
            np.testing.assert_array_equal(sofa["RoomVolume"][:], [73.5])
            np.testing.assert_array_equal(sofa["Data.IR"][:, :, :, 0], np.concatenate(expected_ir))
            np.testing.assert_array_equal(sofa["SourcePosition"][:], np.concatenate(expected_source))
            np.testing.assert_array_equal(sofa["ReceiverPosition"][:, :, 0], expected_receiver)
        with sf.SofaStream(output) as sofa:
            assert sofa.verify(issue_handling="return", mode="read") is None

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
