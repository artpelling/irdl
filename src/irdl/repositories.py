"""Implements custom data repositories and patches the pooch DOI resolver.

.. admonition:: This module is based on source code from the `pooch <https://www.fatiando.org/pooch/latest/index.html>`_ project!

  Copyright (c) 2018 The Pooch Developers
  All rights reserved.

  Redistribution and use in source and binary forms, with or without modification,
  are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
  * Neither the name of the copyright holders nor the names of any contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
  ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import json
import re
from time import sleep
from urllib.parse import unquote

import pooch as po
import requests
from pooch.downloaders import (
    DataRepository,
    DataverseRepository,
    FigshareRepository,
    ZenodoRepository,
    doi_to_url,
)
from pooch.utils import parse_url
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError, HTTPError, Timeout
from urllib3.util.retry import Retry

from irdl.logging import logger

# Separate connect vs. read timeout: DepositOnce can be slow to accept connections.
DEFAULT_TIMEOUT = (60, 30)  # (connect_timeout_s, read_timeout_s)

MAX_RETRIES = 5
BACKOFF_FACTOR = 2.0  # exponential backoff: waits 2, 4, 8, 16, 32 s between retries


class RepositoryProbeError(RuntimeError):
    """Raised when a repository probe cannot confirm repository capabilities."""


def _make_session() -> requests.Session:
    """Create a requests Session with automatic retry and exponential backoff.

    Returns
    -------
    requests.Session
        Session with automatic retry and exponential backoff configured.
    """
    session = requests.Session()
    retry = Retry(
        total=MAX_RETRIES,
        connect=MAX_RETRIES,
        read=3,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=[408, 429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _extract_filename_from_content_disposition(content_disposition: str | None) -> str | None:
    """Extract a filename from a Content-Disposition header."""
    if not content_disposition:
        return None

    match = re.search(r"filename\*=UTF-8''([^;]+)", content_disposition, flags=re.IGNORECASE)
    if match:
        return unquote(match.group(1).strip('"'))

    match = re.search(r'filename="?([^";]+)"?', content_disposition, flags=re.IGNORECASE)
    if match:
        return match.group(1)

    return None


def _fetch_paginated_embedded(session: requests.Session, url: str, embedded_key: str) -> list[dict]:
    """Fetch a paginated DSpace collection until all pages are exhausted."""
    results = []
    next_url = url
    while next_url is not None:
        response = session.get(next_url, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        embedded = payload.get("_embedded", {}).get(embedded_key)
        if not isinstance(embedded, list):
            msg = f"DSpace response did not return an embedded {embedded_key} list"
            raise RepositoryProbeError(msg)
        results.extend(embedded)
        next_url = payload.get("_links", {}).get("next", {}).get("href")
    return results


class ProbedRepository(DataRepository):
    """Base class for repository types identified by archive URL patterns."""

    def __init__(self, doi: str, archive_url: str) -> None:
        self.archive_url = archive_url
        self.doi = doi
        self._api_response = None

    @classmethod
    def initialize(cls, doi: str, archive_url: str) -> DataRepository | None:
        """Initialize repository directly when the archive URL matches."""
        if not cls._matches_archive_url(archive_url):
            return None
        return cls(doi, archive_url)

    @classmethod
    def _matches_archive_url(cls, archive_url: str) -> bool:
        raise NotImplementedError

    @classmethod
    def _probe_repository(cls, archive_url: str) -> None:
        raise NotImplementedError


class DSpaceRepository(ProbedRepository):
    """DSpace repository implementation for DepositOnce-like instances."""

    def __init__(self, doi: str, archive_url: str) -> None:
        super().__init__(doi, archive_url)
        self._item_uuid = None

    @classmethod
    def _matches_archive_url(cls, archive_url: str) -> bool:
        parsed_archive_url = parse_url(archive_url)
        return parsed_archive_url["netloc"] == "depositonce.tu-berlin.de" and (
            "/handle/" in parsed_archive_url["path"] or "/items/" in parsed_archive_url["path"]
        )

    @classmethod
    def _probe_repository(cls, archive_url: str) -> None:
        item_uuid = cls._resolve_item_uuid(archive_url)
        with _make_session() as session:
            _fetch_paginated_embedded(session, cls._bundles_url(archive_url, item_uuid), "bundles")

    @staticmethod
    def _api_base_url(archive_url: str) -> str:
        parsed_archive_url = parse_url(archive_url)
        return f"https://api-{parsed_archive_url['netloc']}"

    @staticmethod
    def _resolve_item_uuid(archive_url: str) -> str:
        final_url = archive_url
        if "/items/" not in parse_url(archive_url)["path"]:
            with _make_session() as session:
                response = session.get(archive_url, timeout=DEFAULT_TIMEOUT)
                response.raise_for_status()
                final_url = response.url

        match = re.search(r"/items/([0-9a-fA-F-]+)$", final_url)
        if match is None:
            raise RepositoryProbeError(f"Could not resolve DSpace item UUID from landing page URL: {final_url}")
        return match.group(1)

    def _item_uuid_value(self) -> str:
        if self._item_uuid is None:
            self._item_uuid = self._resolve_item_uuid(self.archive_url)
        return self._item_uuid

    @classmethod
    def _bundles_url(cls, archive_url: str, item_uuid: str) -> str:
        return f"{cls._api_base_url(archive_url)}/server/api/core/items/{item_uuid}/bundles"

    @property
    def api_response(self) -> dict:
        """Get the API response, fetching from server if not cached.

        Returns
        -------
        dict
            API response containing file metadata.

        Raises
        ------
        ValueError
            If no 'ORIGINAL' bundle is found for the item.
        """
        if self._api_response is None:
            with _make_session() as session:
                bundles = _fetch_paginated_embedded(
                    session, self._bundles_url(self.archive_url, self._item_uuid_value()), "bundles"
                )

                original = next((b for b in bundles if b["name"] == "ORIGINAL"), None)
                if original is None:
                    raise ValueError(f"No 'ORIGINAL' bundle found for item {self._item_uuid_value()}.")

                bitstreams = _fetch_paginated_embedded(session, original["_links"]["bitstreams"]["href"], "bitstreams")

            self._api_response = {
                bs["name"]: {
                    "url": bs["_links"]["content"]["href"],
                    "checksum": f"{bs['checkSum']['checkSumAlgorithm']}:{bs['checkSum']['value']}",
                    "size": bs.get("sizeBytes"),
                }
                for bs in bitstreams
            }

        return self._api_response

    def download_url(self, file_name: str) -> str:
        """Use the repository API to get the download URL for a file given the archive URL.

        Parameters
        ----------
        file_name : str
            The name of the file in the archive that will be downloaded.

        Returns
        -------
        download_url : str
            The HTTP URL that can be used to download the file.

        """
        return self.api_response[file_name]["url"]

    def file_size(self, file_name: str) -> int | None:
        """Return the size of a file in bytes, or None if unavailable.

        Parameters
        ----------
        file_name : str
            The name of the file in the archive.

        Returns
        -------
        size : int or None
            The file size in bytes.

        """
        return self.api_response[file_name].get("size")

    def populate_registry(self, pooch: po.Pooch) -> None:
        """Populate the registry using the data repository's API.

        Parameters
        ----------
        pooch : pooch.Pooch
            The pooch instance that the registry will be added to.

        """
        for name, info in self.api_response.items():
            pooch.registry[name] = info["checksum"]


class RadarRepository(ProbedRepository):
    """RADAR repository implementation for DOI-backed single-archive records."""

    @classmethod
    def _matches_archive_url(cls, archive_url: str) -> bool:
        return "/radar/" in parse_url(archive_url)["path"]

    @classmethod
    def _probe_repository(cls, archive_url: str) -> None:
        metadata = cls._fetch_record_metadata(archive_url)
        if not metadata.get("url"):
            raise RepositoryProbeError("RADAR probe could not find a content URL")
        if not metadata.get("checksum"):
            raise RepositoryProbeError("RADAR probe could not find an archive checksum")

    @staticmethod
    def _parse_archive_checksum(html: str) -> tuple[str, str] | None:
        match = re.search(
            r"Archive checksum:\s*</div>\s*<div[^>]*>\s*([0-9a-fA-F]+)\s*\(([^)]+)\)",
            html,
            flags=re.IGNORECASE,
        )
        if match is None:
            return None
        checksum_value, checksum_algorithm = match.groups()
        return checksum_algorithm.lower(), checksum_value.lower()

    @staticmethod
    def _extract_json_ld_payloads(html: str) -> list[dict]:
        payloads = []
        for match in re.finditer(
            r'<script[^>]+type="application/ld\+json"[^>]*>(.*?)</script>',
            html,
            flags=re.DOTALL | re.IGNORECASE,
        ):
            payload = match.group(1).strip()
            try:
                payloads.append(json.loads(payload))
            except json.JSONDecodeError as exc:
                logger.debug(f"Skipping invalid JSON-LD payload: {exc}")
        return payloads

    @classmethod
    def _distribution_metadata_from_html(cls, html: str) -> dict[str, str | int | None]:
        for payload in cls._extract_json_ld_payloads(html):
            graph = payload.get("@graph") if isinstance(payload, dict) else None
            nodes = graph if isinstance(graph, list) else [payload]
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                node_type = node.get("@type")
                if node_type not in {"http://schema.org/DataDownload", "DataDownload"}:
                    continue

                content_url = node.get("http://schema.org/contentUrl") or node.get("contentUrl")
                content_size = node.get("http://schema.org/contentSize") or node.get("contentSize")
                encoding_format = node.get("http://schema.org/encodingFormat") or node.get("encodingFormat")
                size = None
                if isinstance(content_size, str):
                    match = re.match(r"^(\d+)", content_size)
                    if match is not None:
                        size = int(match.group(1))
                return {
                    "url": content_url,
                    "size": size,
                    "encoding_format": encoding_format,
                }

        raise RepositoryProbeError("No RADAR DataDownload block found in JSON-LD")

    @classmethod
    def _fetch_record_metadata(cls, archive_url: str) -> dict[str, str | int | None]:
        with _make_session() as session:
            response = session.get(archive_url, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            html = response.text

            distribution = cls._distribution_metadata_from_html(html)
            checksum = cls._parse_archive_checksum(html)
            if checksum is None:
                raise RepositoryProbeError("No RADAR archive checksum found in landing page")

            content_url = distribution["url"]
            if not isinstance(content_url, str):
                raise RepositoryProbeError("RADAR DataDownload block did not contain a content URL")

            head_response = session.head(content_url, allow_redirects=True, timeout=DEFAULT_TIMEOUT)
            head_response.raise_for_status()

        file_name = _extract_filename_from_content_disposition(head_response.headers.get("Content-Disposition"))
        if file_name is None:
            file_name = content_url.rstrip("/").split("/")[-1]

        size = distribution["size"]
        if size is None:
            content_length = head_response.headers.get("Content-Length")
            if content_length is not None:
                size = int(content_length)

        checksum_algorithm, checksum_value = checksum
        return {
            "name": file_name,
            "url": content_url,
            "checksum": f"{checksum_algorithm}:{checksum_value}",
            "size": size,
            "encoding_format": distribution["encoding_format"],
        }

    @property
    def api_response(self) -> dict:
        """Return a one-file registry for a RADAR archive record."""
        if self._api_response is None:
            metadata = self._fetch_record_metadata(self.archive_url)
            self._api_response = {
                metadata["name"]: {
                    "url": metadata["url"],
                    "checksum": metadata["checksum"],
                    "size": metadata["size"],
                }
            }
        return self._api_response

    def download_url(self, file_name: str) -> str:
        """Return the content URL for the RADAR archive artifact."""
        return self.api_response[file_name]["url"]

    def file_size(self, file_name: str) -> int | None:
        """Return the size of the RADAR archive artifact."""
        return self.api_response[file_name].get("size")

    def populate_registry(self, pooch: po.Pooch) -> None:
        """Populate the pooch registry with the RADAR archive checksum."""
        for name, info in self.api_response.items():
            pooch.registry[name] = info["checksum"]


def doi_to_repository(doi: str) -> DataRepository:
    """Instantiate a data repository instance from a given DOI.

    This function implements the chain of responsibility dispatch
    to the correct data repository class.

    Parameters
    ----------
    doi : str
        The DOI of the archive.

    Returns
    -------
    data_repository : DataRepository
        The data repository object.

    Raises
    ------
    ConnectionError
        If the DOI cannot be resolved to a URL.
    ValueError
        If no repository can handle the given URL.
    """
    # This should go away in a separate issue: DOI handling should
    # not rely on the (non-)existence of trailing slashes. The issue
    # is documented in https://github.com/fatiando/pooch/issues/324
    if doi[-1] == "/":
        doi = doi[:-1]

    repositories = [
        FigshareRepository,
        ZenodoRepository,
        DSpaceRepository,
        DataverseRepository,
        RadarRepository,
    ]

    # Extract the DOI and the repository information
    archive_url = None
    for attempt in range(MAX_RETRIES):
        try:
            archive_url = doi_to_url(doi, timeout=DEFAULT_TIMEOUT)
            break
        except (ConnectionError, Timeout) as e:
            wait = BACKOFF_FACTOR * (2**attempt)
            if attempt == 0:
                logger.warning(
                    f"Failed to resolve DOI {doi} via https://doi.org/{doi} due to a connection or timeout error, retrying with exponential backoff..."
                )
            logger.debug(f"Attempt {attempt + 1}/{MAX_RETRIES} failed ({type(e).__name__}), waiting {wait:.0f}s")
            if attempt < MAX_RETRIES - 1:
                sleep(wait)

    if archive_url is None:
        raise ConnectionError(f"Could not resolve DOI {doi} to a URL. Check the DOI or try running the script again.")

    # Try the converters one by one until one of them returned a URL
    data_repository = None
    for repo in repositories:
        if data_repository is None:
            data_repository = repo.initialize(
                archive_url=archive_url,
                doi=doi,
            )

    if data_repository is None:
        repository = parse_url(archive_url)["netloc"]
        raise ValueError(
            f"Invalid data repository '{repository}'. "
            "To request or contribute support for this repository, "
            "please open an issue at https://github.com/fatiando/pooch/issues"
        )

    return data_repository
