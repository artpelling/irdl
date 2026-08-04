"""Tests for provenance-preserving raw provider adapters."""

from pathlib import Path

import pytest

from irdl.aalto import ArniRoomImpulseResponseDataset, MotusDataset
from irdl.brno import ButReverbDbDataset
from irdl.imperial import AceChallengeDataset
from irdl.mit import MitImpulseResponseSurveyDataset
from irdl.york import OpenAirDataset

LOCAL_ONLY_DATASETS = [
    ArniRoomImpulseResponseDataset,
    MitImpulseResponseSurveyDataset,
    ButReverbDbDataset,
]
ALL_RAW_DATASETS = [*LOCAL_ONLY_DATASETS, MotusDataset, AceChallengeDataset, OpenAirDataset]


@pytest.mark.parametrize("dataset_class", ALL_RAW_DATASETS)
def test_raw_provider_rejects_unproven_sofa_conversion(dataset_class):
    """Do not invent coordinates needed for a SOFA conversion."""
    with pytest.raises(ValueError, match="output_format='raw' only"):
        dataset_class()._validate_params(
            output_format="sofa", source_path=Path("provider"), receiver="Single", environment="x"
        )


@pytest.mark.parametrize("dataset_class", LOCAL_ONLY_DATASETS)
def test_local_only_provider_requires_source_path(dataset_class):
    """Give a clear error where automatic acquisition is unavailable."""
    with pytest.raises(ValueError, match="requires source_path"):
        dataset_class()._validate_params(output_format="raw", source_path=None)


@pytest.mark.parametrize("dataset_class", ALL_RAW_DATASETS)
def test_local_source_preserves_original_artifact(dataset_class, tmp_path):
    """Return the exact local artifact without rewriting provider data."""
    artifact = tmp_path / "provider"
    artifact.mkdir()
    assert dataset_class()._local_source(source_path=artifact) == artifact


def test_missing_local_source_is_rejected(tmp_path):
    """Reject misspelled or unavailable local provider paths."""
    with pytest.raises(FileNotFoundError, match="source_path does not exist"):
        ArniRoomImpulseResponseDataset()._local_source(source_path=tmp_path / "missing")


def test_ace_archive_name_and_receiver_validation():
    """Map semantic receiver names to the provider archive names."""
    dataset = AceChallengeDataset()
    assert dataset._source_filename(receiver="Lin8Ch") == "ACE_Corpus_RIRN_Lin8Ch.tbz2"
    with pytest.raises(ValueError, match="receiver must be"):
        dataset._validate_params(output_format="raw", source_path=None, receiver="unknown")


def test_openair_requires_local_source_for_unverified_environment():
    """Do not construct unverified OpenAIR URLs from arbitrary slugs."""
    with pytest.raises(ValueError, match="no verified automatic URL"):
        OpenAirDataset()._validate_params(output_format="raw", source_path=None, environment="unlisted-room")


def test_non_doi_provider_doc_uses_source_url():
    """Describe provider landing pages without fabricating DOI URLs."""
    assert "Source: https://mcdermottlab.mit.edu/Reverb/IR_Survey.html" in MitImpulseResponseSurveyDataset.get.__doc__
    assert "doi.org/None" not in MitImpulseResponseSurveyDataset.get.__doc__


def test_local_provider_public_get_returns_original_directory(tmp_path):
    """Exercise local ingestion through the public retrieval path."""
    source = tmp_path / "arni-provider"
    source.mkdir()
    result = ArniRoomImpulseResponseDataset.get(source_path=source, cache_dir=tmp_path / "cache")
    assert result == source
