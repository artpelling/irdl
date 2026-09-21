"""Tests for native-SOFA datasets hosted by SONICOM."""

from pathlib import Path

import pytest

from irdl.akt import HutubsDataset
from irdl.sonicom import CipicDataset, SadieDataset
from irdl.utils import load_hash_registry, load_url_registry


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
        (CipicDataset(), {"subject": 1}, "subject must identify"),
        (CipicDataset(), {"subject": "003"}, "subject must identify"),
        (SadieDataset(), {"subject": "H1"}, "subject must be one of"),
        (SadieDataset(), {"subject": "h3"}, "subject must be one of"),
    ],
)
def test_sonicom_rejects_unknown_subjects(dataset, kwargs, message):
    """Reject selectors without a registered SONICOM SOFA file."""
    with pytest.raises(ValueError, match=message):
        dataset._validate_params(**kwargs)


def test_hutubs_resolves_individual_sonicom_sofa_url():
    """Use SONICOM's individual measured and simulated HUTUBS files."""
    dataset = HutubsDataset()
    assert dataset.direct_sofa_url("pp1_HRIRs_measured.sofa") == (
        "https://ecosystem.sonicom.eu/data/76/25492/49409/pp1_HRIRs_measured.sofa"
    )
    assert dataset.direct_sofa_url("pp96_HRIRs_simulated.sofa") == (
        "https://ecosystem.sonicom.eu/data/76/25587/49657/pp96_HRIRs_simulated.sofa"
    )


def test_hutubs_non_raw_downloads_only_requested_sonicom_file(monkeypatch, tmp_path):
    """Bypass the HUTUBS ZIP for normal retrieval."""
    dataset = HutubsDataset()
    calls = []

    def download_direct(provider_dir: Path, url: str) -> Path:
        calls.append(url)
        return provider_dir / Path(url).name

    monkeypatch.setattr(dataset, "_download_direct_sofa", download_direct)
    result = dataset.download(
        tmp_path,
        direct_sofa_url=dataset.direct_sofa_url("pp7_HRIRs_simulated.sofa"),
    )

    assert result == tmp_path / "pp7_HRIRs_simulated.sofa"
    assert calls == ["https://ecosystem.sonicom.eu/data/76/25558/49581/pp7_HRIRs_simulated.sofa"]


def test_all_sonicom_urls_have_direct_sofa_hashes():
    """Retain verification for every registered SONICOM direct SOFA file."""
    urls = load_url_registry("sonicom")
    hashes = load_hash_registry("direct_sofa")
    assert urls
    assert set(urls.values()) <= hashes.keys()
