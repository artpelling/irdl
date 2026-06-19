"""Tests for repository detection and metadata parsing."""

from types import SimpleNamespace

import pytest
from requests.exceptions import Timeout

import irdl.repositories as repositories_module
from irdl.repositories import DSpaceRepository, RadarRepository, doi_to_repository


def test_dspace_initialize_requires_only_url_match(monkeypatch):
    """Verify DSpace initialization returns directly for matching URLs."""
    monkeypatch.setattr(
        DSpaceRepository,
        "_probe_repository",
        classmethod(lambda _cls, archive_url: pytest.fail(f"unexpected probe for {archive_url}")),
    )

    repo = DSpaceRepository.initialize(
        "10.14279/depositonce-123", "https://depositonce.tu-berlin.de/handle/11303/12345"
    )

    assert isinstance(repo, DSpaceRepository)


def test_dspace_initialize_rejects_non_matching_host(monkeypatch):
    """Verify DSpace initialization rejects unrelated hosts before construction."""
    monkeypatch.setattr(
        DSpaceRepository,
        "_probe_repository",
        classmethod(lambda _cls, archive_url: pytest.fail(f"unexpected probe for {archive_url}")),
    )

    repo = DSpaceRepository.initialize("10.0000/example", "https://example.org/handle/11303/12345")

    assert repo is None


def test_dspace_api_base_url_derives_api_host():
    """Verify DSpace API URLs are derived from the landing host."""
    archive_url = "https://depositonce.tu-berlin.de/handle/11303/12345"
    item_uuid = "1806cd7f-67f3-4355-b5e8-6cab392813bd"

    assert DSpaceRepository._api_base_url(archive_url) == "https://api-depositonce.tu-berlin.de"
    assert (
        DSpaceRepository._bundles_url(archive_url, item_uuid)
        == f"https://api-depositonce.tu-berlin.de/server/api/core/items/{item_uuid}/bundles"
    )


def test_dspace_api_response_follows_paginated_bitstreams(monkeypatch):
    """Verify DSpace repository collects all bitstreams across API pages."""

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class FakeSession:
        def __init__(self):
            self.urls = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, timeout=None):
            del timeout
            self.urls.append(url)
            payload = responses[url]
            return FakeResponse(payload)

    responses = {
        "https://api-depositonce.tu-berlin.de/server/api/core/items/item-uuid/bundles": {
            "_embedded": {
                "bundles": [
                    {
                        "name": "ORIGINAL",
                        "_links": {
                            "bitstreams": {
                                "href": "https://api-depositonce.tu-berlin.de/server/api/core/bundles/original/bitstreams"
                            }
                        },
                    }
                ]
            },
            "_links": {},
        },
        "https://api-depositonce.tu-berlin.de/server/api/core/bundles/original/bitstreams": {
            "_embedded": {
                "bitstreams": [
                    {
                        "name": "SR1-D.h5",
                        "sizeBytes": 123,
                        "checkSum": {"checkSumAlgorithm": "MD5", "value": "aaa"},
                        "_links": {"content": {"href": "https://example.org/SR1-D.h5"}},
                    }
                ]
            },
            "_links": {"next": {"href": "https://api-depositonce.tu-berlin.de/server/api/core/bundles/original/bitstreams?page=1"}},
        },
        "https://api-depositonce.tu-berlin.de/server/api/core/bundles/original/bitstreams?page=1": {
            "_embedded": {
                "bitstreams": [
                    {
                        "name": "SRA2-D.h5",
                        "sizeBytes": 456,
                        "checkSum": {"checkSumAlgorithm": "MD5", "value": "bbb"},
                        "_links": {"content": {"href": "https://example.org/SRA2-D.h5"}},
                    }
                ]
            },
            "_links": {},
        },
    }

    monkeypatch.setattr(repositories_module, "_make_session", FakeSession)
    repo = DSpaceRepository("10.14279/depositonce-23943", "https://depositonce.tu-berlin.de/items/item-uuid")
    repo._item_uuid = "item-uuid"

    assert repo.api_response == {
        "SR1-D.h5": {"url": "https://example.org/SR1-D.h5", "checksum": "MD5:aaa", "size": 123},
        "SRA2-D.h5": {"url": "https://example.org/SRA2-D.h5", "checksum": "MD5:bbb", "size": 456},
    }


def test_radar_initialize_requires_only_url_match(monkeypatch):
    """Verify RADAR initialization returns directly for matching URLs."""
    monkeypatch.setattr(
        RadarRepository,
        "_probe_repository",
        classmethod(lambda _cls, archive_url: pytest.fail(f"unexpected probe for {archive_url}")),
    )

    repo = RadarRepository.initialize("10.60887/example", "https://datathek.oeaw.ac.at/radar/en/dataset/example")

    assert isinstance(repo, RadarRepository)


def test_radar_parse_archive_checksum():
    """Verify RADAR checksum parser extracts algorithm and value."""
    html = """
    <div class="row tmd-archivechecksum">
      <div class="col-md-3">Archive checksum:</div>
      <div class="col-md-9">ee7976db28d6ce0b88cfd3ef11560d7f (MD5)</div>
    </div>
    """

    assert RadarRepository._parse_archive_checksum(html) == ("md5", "ee7976db28d6ce0b88cfd3ef11560d7f")


def test_radar_distribution_metadata_from_html():
    """Verify RADAR JSON-LD parser extracts DataDownload metadata."""
    html = """
    <script type="application/ld+json">
    {
      "@graph": [
        {
          "@id": "_:download",
          "http://schema.org/encodingFormat": "application/x-tar",
          "http://schema.org/contentUrl": "https://datathek.oeaw.ac.at/radar-backend/archives/example/versions/1/content",
          "http://schema.org/contentSize": "46069760 bytes",
          "@type": "http://schema.org/DataDownload"
        }
      ]
    }
    </script>
    """

    assert RadarRepository._distribution_metadata_from_html(html) == {
        "url": "https://datathek.oeaw.ac.at/radar-backend/archives/example/versions/1/content",
        "size": 46069760,
        "encoding_format": "application/x-tar",
    }


def test_radar_populate_registry_from_cached_api_response():
    """Verify RADAR repository publishes checksum under the archive filename."""
    repo = RadarRepository("10.60887/example", "https://datathek.oeaw.ac.at/radar/en/dataset/example")
    repo._api_response = {
        "10.60887-example.tar": {
            "url": "https://datathek.oeaw.ac.at/radar-backend/archives/example/versions/1/content",
            "checksum": "md5:ee7976db28d6ce0b88cfd3ef11560d7f",
            "size": 46069760,
        }
    }
    pup = SimpleNamespace(registry={})

    repo.populate_registry(pup)

    assert pup.registry == {"10.60887-example.tar": "md5:ee7976db28d6ce0b88cfd3ef11560d7f"}


def test_doi_resolution_warning_includes_doi_resolver_url(monkeypatch):
    """Verify DOI retry warning mentions the DOI resolver URL."""
    warnings = []
    debug_messages = []

    def fake_warning(message):
        warnings.append(message)

    def fake_debug(message):
        debug_messages.append(message)

    call_count = {"count": 0}

    def fake_doi_to_url(_doi, **_kwargs):
        call_count["count"] += 1
        if call_count["count"] == 1:
            msg = "timed out"
            raise Timeout(msg)
        return "https://datathek.oeaw.ac.at/radar/en/dataset/example"

    monkeypatch.setattr(repositories_module, "doi_to_url", fake_doi_to_url)
    monkeypatch.setattr(repositories_module, "sleep", lambda _seconds: None)
    monkeypatch.setattr(repositories_module.logger, "warning", fake_warning)
    monkeypatch.setattr(repositories_module.logger, "debug", fake_debug)
    monkeypatch.setattr(
        RadarRepository,
        "api_response",
        property(lambda _self: {"10.60887-example.tar": {"url": "u", "checksum": "md5:x", "size": 1}}),
    )

    repo = doi_to_repository("10.60887/example")

    expected_warning = (
        "Failed to resolve DOI 10.60887/example via https://doi.org/10.60887/example "
        "due to a connection or timeout error, retrying with exponential backoff..."
    )
    assert isinstance(repo, RadarRepository)
    assert warnings == [expected_warning]
    assert any("Attempt 1/5 failed (Timeout)" in message for message in debug_messages)
