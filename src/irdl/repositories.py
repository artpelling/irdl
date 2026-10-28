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

from time import sleep

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
from requests.exceptions import ConnectionError, Timeout
from urllib3.util.retry import Retry

from irdl.logging import logger

# Separate connect vs. read timeout: DepositOnce can be slow to accept connections.
DEFAULT_TIMEOUT = (60, 30)  # (connect_timeout_s, read_timeout_s)
DSPACE_API_ROOT = "https://api-depositonce.tu-berlin.de/server/api"
EXPECTED_DSPACE_VERSION_PREFIX = "DSpace 9"

MAX_RETRIES = 5
BACKOFF_FACTOR = 2.0  # exponential backoff: waits 2, 4, 8, 16, 32 s between retries


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


def _fetch_paginated_embedded(session: requests.Session, url: str, embedded_key: str) -> list[dict]:
    """Fetch every page of a DSpace embedded collection."""
    results = []
    while url:
        response = session.get(url, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        results.extend(payload["_embedded"][embedded_key])
        url = payload.get("_links", {}).get("next", {}).get("href")
    return results


class DSpaceRepository(DataRepository):
    """DSpace repository implementation for DepositOnce.

    Attributes
    ----------
    archive_url : str
        The archive URL for the repository.
    doi : str
        The DOI of the archive.
    _api_response : dict or None
        Cached API response from the repository.
    """

    def __init__(self, doi: str, archive_url: str) -> None:
        self.archive_url = archive_url
        self.doi = doi
        self._api_response = None

    @classmethod
    def initialize(cls, doi: str, archive_url: str) -> DataRepository | None:
        """Initialize the data repository if the given URL points to a corresponding repository.

        Initializes a data repository object. This is done as part of
        a chain of responsibility. If the class cannot handle the given
        repository URL, it returns None. Otherwise a DSpaceRepository
        instance is returned.

        Parameters
        ----------
        doi : str
            The DOI that identifies the repository.
        archive_url : str
            The resolved URL for the DOI.

        Returns
        -------
        DSpaceRepository or None
            DSpaceRepository instance if URL matches, None otherwise.
        """
        # Check whether this is a Figshare URL
        parsed_archive_url = parse_url(archive_url)
        if parsed_archive_url["netloc"] != "depositonce.tu-berlin.de":
            return None

        return cls(doi, archive_url)

    def _handle_pid(self) -> str | None:
        """Return the DepositOnce handle path from the archive URL, if present."""
        parts = self.archive_url.split("/handle/", maxsplit=1)
        return parts[1] if len(parts) == 2 else None

    def _warn_on_version_mismatch(self, session: requests.Session) -> None:
        """Warn when the DepositOnce DSpace major version differs from the tested API family."""
        response = session.get(DSPACE_API_ROOT, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        dspace_version = response.json().get("dspaceVersion")
        if dspace_version is None:
            logger.warning("DepositOnce API did not report dspaceVersion; assuming compatible DSpace API.")
            return
        if not dspace_version.startswith(EXPECTED_DSPACE_VERSION_PREFIX):
            logger.warning(
                f"DepositOnce reports {dspace_version}; expected {EXPECTED_DSPACE_VERSION_PREFIX}.x. "
                "IRDL may need an API compatibility update."
            )

    def _resolve_item(self, session: requests.Session) -> dict:
        """Resolve DOI/handle to the current DSpace item record."""
        identifiers = [self.doi]
        handle_pid = self._handle_pid()
        if handle_pid is not None:
            identifiers.append(handle_pid)

        last_error = None
        for identifier in dict.fromkeys(identifiers):
            response = session.get(
                f"{DSPACE_API_ROOT}/pid/find",
                params={"id": identifier},
                timeout=DEFAULT_TIMEOUT,
            )
            try:
                response.raise_for_status()
            except requests.HTTPError as error:
                last_error = error
                continue
            return response.json()

        if last_error is not None:
            raise last_error
        raise ValueError(f"Could not resolve DepositOnce item for {self.doi}.")

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
                self._warn_on_version_mismatch(session)
                item = self._resolve_item(session)
                bundles_url = item["_links"]["bundles"]["href"]
                bundles = _fetch_paginated_embedded(session, bundles_url, "bundles")

                original = next((b for b in bundles if b["name"] == "ORIGINAL"), None)
                if original is None:
                    item_id = item.get("uuid") or item.get("id") or self.doi
                    raise ValueError(f"No 'ORIGINAL' bundle found for item {item_id}.")

                bitstreams = _fetch_paginated_embedded(
                    session, original["_links"]["bitstreams"]["href"], "bitstreams"
                )

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
    ]

    # Extract the DOI and the repository information
    archive_url = None
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            archive_url = doi_to_url(doi, timeout=DEFAULT_TIMEOUT)
            break
        except (ConnectionError, Timeout) as error:
            last_error = error
            if attempt < MAX_RETRIES - 1:
                wait = BACKOFF_FACTOR * (2 ** attempt)
                logger.warning(
                    "DOI resolution failed for %s (%s: %s); retrying in %.0fs (attempt %s/%s)",
                    doi, type(error).__name__, error, wait, attempt + 2, MAX_RETRIES,
                )
                sleep(wait)
            else:
                logger.debug("DOI resolution exhausted %s attempts for %s: %s", MAX_RETRIES, doi, error)

    if archive_url is None:
        raise ConnectionError(
            f"Could not resolve DOI {doi} after {MAX_RETRIES} attempts: {type(last_error).__name__}: {last_error}. "
            "Check the DOI and network connection before retrying."
        ) from last_error

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
