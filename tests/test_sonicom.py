"""Tests for native-SOFA datasets hosted by SONICOM."""

import io
import json
from pathlib import Path

import pytest

from irdl import sonicom
from irdl.akt import HutubsDataset
from irdl.sonicom import CipicDataset, SadieDataset
from irdl.utils import load_hash_registry


@pytest.mark.parametrize(
    ("dataset", "kwargs", "expected"),
    [
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

    assert CipicDataset().direct_sofa_url("subject_003.sofa") == (
        "https://ecosystem.sonicom.eu/data/72/25340/48753/subject_003.sofa"
    )


def test_hutubs_resolves_individual_sonicom_sofa_url(monkeypatch):
    """Use SONICOM's individual measured and simulated HUTUBS files."""
    urls = {
        "pp1_HRIRs_measured.sofa": "https://ecosystem.sonicom.eu/data/76/25492/49409/pp1_HRIRs_measured.sofa",
        "pp96_HRIRs_simulated.sofa": "https://ecosystem.sonicom.eu/data/76/25587/49657/pp96_HRIRs_simulated.sofa",
    }
    monkeypatch.setattr(sonicom, "_sonicom_manifest", lambda _database_id: urls)
    dataset = HutubsDataset()
    assert dataset.direct_sofa_url("pp1_HRIRs_measured.sofa") == urls["pp1_HRIRs_measured.sofa"]
    assert dataset.direct_sofa_url("pp96_HRIRs_simulated.sofa") == urls["pp96_HRIRs_simulated.sofa"]


def test_hutubs_non_raw_downloads_only_requested_sonicom_file(monkeypatch, tmp_path):
    """Bypass the HUTUBS ZIP for normal retrieval."""
    url = "https://ecosystem.sonicom.eu/data/76/25558/49581/pp7_HRIRs_simulated.sofa"
    dataset = HutubsDataset()
    calls = []

    def download_direct(provider_dir: Path, direct_url: str, known_hash: str) -> Path:
        calls.append((direct_url, known_hash))
        return provider_dir / Path(direct_url).name

    monkeypatch.setattr(dataset, "_download_direct_sofa", download_direct)
    result = dataset.download(
        tmp_path,
        direct_sofa_url=url,
        direct_sofa_hash=dataset.direct_sofa_hash("pp7_HRIRs_simulated.sofa"),
    )

    assert result == tmp_path / "pp7_HRIRs_simulated.sofa"
    assert calls == [(url, dataset.direct_sofa_hash("pp7_HRIRs_simulated.sofa"))]


def test_sonicom_hashes_are_keyed_by_source_filename():
    """Pin content by scenario filename rather than mutable endpoint URL."""
    hashes = load_hash_registry("sonicom")
    assert hashes
    assert all(Path(filename).name == filename for filename in hashes)
