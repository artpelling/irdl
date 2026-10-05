"""Tests for native-SOFA datasets hosted by SONICOM."""

import io
import json
from pathlib import Path

import pytest

from irdl import sonicom
from irdl.akt import AKTZipBaseDataset, FabianDataset, HutubsDataset
from irdl.sonicom import AriDataset, CipicDataset, SadieDataset
from irdl.utils import load_hash_registry


@pytest.mark.parametrize(
    ("dataset", "kwargs", "expected"),
    [
        (AriDataset(), {"subject": "b_nh10", "kind": "hrtf"}, "hrtf b_nh10.sofa"),
        (AriDataset(), {"subject": "d_nh1379", "kind": "dtf"}, "dtf d_nh1379.sofa"),
        (CipicDataset(), {"subject": 3}, "subject_003.sofa"),
        (CipicDataset(), {"subject": 165}, "subject_165.sofa"),
        (SadieDataset(), {"subject": "H3"}, "H3_48K_24bit_256tap_FIR_SOFA.sofa"),
        (SadieDataset(), {"subject": "D2"}, "D2_48K_24bit_256tap_FIR_SOFA.sofa"),
    ],
)
def test_sonicom_source_filenames(dataset, kwargs, expected):
    """Build native SONICOM filenames from public parameters."""
    assert dataset._source_filename(**kwargs) == expected


@pytest.mark.parametrize(
    ("dataset", "kwargs", "message"),
    [
        (AriDataset(), {"subject": "b_nh1", "kind": "hrtf"}, "subject must be one of"),
        (AriDataset(), {"subject": "b_nh10", "kind": "hrirs"}, "kind must be either"),
        (CipicDataset(), {"subject": 1}, r"subject must be one of.*3.*165"),
        (CipicDataset(), {"subject": "003"}, r"subject must be one of.*3.*165"),
        (SadieDataset(), {"subject": "H1"}, "subject must be one of"),
        (SadieDataset(), {"subject": "h3"}, "subject must be one of"),
    ],
)
def test_sonicom_rejects_unknown_subjects(dataset, kwargs, message):
    """Reject selectors without a registered SONICOM SOFA file."""
    with pytest.raises(ValueError, match=message):
        dataset._validate_params(**kwargs)


def test_sonicom_manifest_resolves_current_url(monkeypatch):
    """Resolve provider URLs from SONICOM's JSON manifest, not package data."""
    payload = {
        "data": [
            {
                "Datafile Name": "subject_003.sofa",
                "Datafile URL": "https://ecosystem.sonicom.eu/data/72/25340/48753/subject_003.sofa",
            }
        ]
    }

    monkeypatch.setattr(sonicom, "urlopen", lambda _url: io.BytesIO(json.dumps(payload).encode()))
    sonicom._sonicom_manifest.cache_clear()

    direct_sofa = CipicDataset()._direct_sofa("subject_003.sofa")
    assert direct_sofa[0] == "https://ecosystem.sonicom.eu/data/72/25340/48753/subject_003.sofa"
    assert direct_sofa[1] is not None


def test_sonicom_manifest_warns_once_and_keeps_first_duplicate_filename(monkeypatch, caplog):
    """Warn once for duplicate auxiliary files without replacing their first URL."""
    first_url = "https://ecosystem.sonicom.eu/data/18/1/generic.sofa"
    payload = {
        "data": [
            {"Datafile Name": "generic.sofa", "Datafile URL": first_url},
            {"Datafile Name": "generic.sofa", "Datafile URL": "https://ecosystem.sonicom.eu/data/18/2/generic.sofa"},
            {"Datafile Name": "generic.sofa", "Datafile URL": "https://ecosystem.sonicom.eu/data/18/3/generic.sofa"},
        ]
    }
    monkeypatch.setattr(sonicom, "urlopen", lambda _url: io.BytesIO(json.dumps(payload).encode()))
    sonicom._sonicom_manifest.cache_clear()

    assert sonicom._sonicom_manifest(18) == {"generic.sofa": first_url}
    assert caplog.messages == ['Duplicate key "generic.sofa" in SONICOM database 18; downloading first match.']


def test_ari_resolves_series_specific_manifest(monkeypatch):
    """Resolve ARI files through the manifest for their B/C/D series."""
    urls = {
        14: {"hrtf b_nh10.sofa": "https://example.test/b"},
        18: {"dtf d_nh1379.sofa": "https://example.test/d"},
    }
    monkeypatch.setattr(sonicom, "_sonicom_manifest", lambda database_id: urls[database_id])
    dataset = AriDataset()

    hashes = load_hash_registry("ari")
    assert dataset._direct_sofa("hrtf b_nh10.sofa") == ("https://example.test/b", hashes["hrtf b_nh10.sofa"])
    assert dataset._direct_sofa("dtf d_nh1379.sofa") == ("https://example.test/d", hashes["dtf d_nh1379.sofa"])


@pytest.mark.parametrize(
    ("source_filename", "direct_filename"),
    [
        ("FABIAN_HRIR_measured_HATO_0.sofa", "FABIAN_HRIR_measured_HATO_0.sofa"),
        ("FABIAN_HRIR_simulated_HATO_0.sofa", "FABIAN_HRIR_modeled_HATO_0.sofa"),
    ],
)
def test_fabian_resolves_individual_sonicom_sofa_url(monkeypatch, source_filename, direct_filename):
    """Use SONICOM's individual FABIAN SOFA files."""
    url = f"https://ecosystem.sonicom.eu/data/101/28598/62784/{direct_filename}"
    monkeypatch.setattr(sonicom, "_sonicom_manifest", lambda _database_id: {direct_filename: url})

    assert FabianDataset()._direct_sofa(source_filename) == (
        url,
        load_hash_registry("fabian")[direct_filename],
    )


def test_fabian_hashes_cover_all_selectors():
    """Pin every FABIAN SOFA that can be requested through SONICOM."""
    dataset = FabianDataset()
    filenames = {
        dataset._source_filename(kind=kind, hato=hato)
        for kind in ("measured", "simulated")
        for hato in (0, 10, 20, 30, 40, 50, 310, 320, 330, 340, 350)
    }
    assert {filename.replace("_simulated_", "_modeled_") for filename in filenames} == set(load_hash_registry("fabian"))


def test_hutubs_resolves_individual_sonicom_sofa_url(monkeypatch):
    """Use SONICOM's individual measured and simulated HUTUBS files."""
    urls = {
        "pp1_HRIRs_measured.sofa": "https://ecosystem.sonicom.eu/data/76/25492/49409/pp1_HRIRs_measured.sofa",
        "pp96_HRIRs_simulated.sofa": "https://ecosystem.sonicom.eu/data/76/25587/49657/pp96_HRIRs_simulated.sofa",
    }
    monkeypatch.setattr(sonicom, "_sonicom_manifest", lambda _database_id: urls)
    dataset = HutubsDataset()
    assert dataset._direct_sofa("pp1_HRIRs_measured.sofa")[0] == urls["pp1_HRIRs_measured.sofa"]
    assert dataset._direct_sofa("pp96_HRIRs_simulated.sofa")[0] == urls["pp96_HRIRs_simulated.sofa"]


def test_sonicom_resolves_an_unpinned_file(monkeypatch):
    """Return a URL without a digest when the manifest has no pinned file."""
    url = "https://ecosystem.sonicom.eu/data/72/1/unregistered.sofa"
    monkeypatch.setattr(sonicom, "_sonicom_manifest", lambda _database_id: {"unregistered.sofa": url})
    dataset = CipicDataset()

    assert dataset._direct_sofa("unregistered.sofa") == (url, None)
    assert dataset._direct_sofa("missing.sofa") == (None, None)


def test_cipic_downloads_only_requested_sonicom_file(monkeypatch, tmp_path):
    """Use the shared SONICOM download implementation for native datasets."""
    direct_sofa = (
        "https://ecosystem.sonicom.eu/data/72/25340/48753/subject_003.sofa",
        load_hash_registry("cipic")["subject_003.sofa"],
    )
    dataset = CipicDataset()
    calls = []

    def download_direct(provider_dir: Path, url: str, known_hash: str) -> Path:
        calls.append((url, known_hash))
        return provider_dir / Path(url).name

    monkeypatch.setattr(dataset, "_direct_sofa", lambda _filename: direct_sofa)
    monkeypatch.setattr(dataset, "_download_direct_sofa", download_direct)
    result = dataset._download(tmp_path, subject=3)

    assert result == tmp_path / "subject_003.sofa"
    assert calls == [direct_sofa]


@pytest.mark.parametrize("dataset", [FabianDataset(), HutubsDataset()])
def test_akt_raw_download_uses_canonical_zip(monkeypatch, tmp_path, dataset):
    """Keep AKT raw retrieval on its DepositOnce ZIP."""
    calls = []

    def download_zip(source_dataset, provider_dir: Path, **_dataset_kwargs) -> Path:
        calls.append(provider_dir)
        return provider_dir / source_dataset._zipfile

    monkeypatch.setattr(AKTZipBaseDataset, "_download", download_zip)
    assert dataset._download(tmp_path) == tmp_path / dataset._zipfile
    assert calls == [tmp_path]


@pytest.mark.parametrize("dataset_name", ["ari", "cipic", "sadie", "fabian", "hutubs", "dechorate", "miracle"])
def test_dataset_hashes_are_keyed_by_source_filename(dataset_name):
    """Keep each Dataset's content pins separate from mutable endpoint URLs."""
    hashes = load_hash_registry(dataset_name)
    assert hashes
    assert all(Path(filename).name == filename for filename in hashes)
