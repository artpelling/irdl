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

import requests
from pooch import get_logger
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

# Separate connect vs. read timeout: DepositOnce can be slow to accept connections.
DEFAULT_TIMEOUT = (60, 30)  # (connect_timeout_s, read_timeout_s)

MAX_RETRIES = 5
BACKOFF_FACTOR = 0.5  # exponential backoff: waits 0.5, 1, 2, 4, 8 s between retries


def _make_session():
    """Create a requests Session with automatic retry and exponential backoff."""
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


class DSpaceRepository(DataRepository):
    def __init__(self, doi, archive_url):
        self.archive_url = archive_url
        self.doi = doi
        self._api_response = None

    @classmethod
    def initialize(cls, doi, archive_url):
        """Initialize the data repository if the given URL points to a corresponding repository.

        Initializes a data repository object. This is done as part of
        a chain of responsibility. If the class cannot handle the given
        repository URL, it returns `None`. Otherwise a `DSpaceRepository`
        instance is returned.

        Parameters
        ----------
        doi : :class:`str`
            The DOI that identifies the repository.
        archive_url : :class:`str`
            The resolved URL for the DOI.

        """
        # Check whether this is a Figshare URL
        parsed_archive_url = parse_url(archive_url)
        if parsed_archive_url["netloc"] != "depositonce.tu-berlin.de":
            return None

        return cls(doi, archive_url)

    @property
    def api_response(self):
        if self._api_response is None:
            article_id = self.archive_url.split("/")[-1]
            with _make_session() as session:
                response = session.get(
                    f"https://api-depositonce.tu-berlin.de/server/api/core/items/{article_id}/bundles",
                    timeout=DEFAULT_TIMEOUT,
                )
                response.raise_for_status()
                bundles = response.json()["_embedded"]["bundles"]

                original = next((b for b in bundles if b["name"] == "ORIGINAL"), None)
                if original is None:
                    raise ValueError(f"No 'ORIGINAL' bundle found for item {article_id}.")

                response = session.get(
                    original["_links"]["bitstreams"]["href"],
                    timeout=DEFAULT_TIMEOUT,
                )
                response.raise_for_status()
                bitstreams = response.json()["_embedded"]["bitstreams"]

            self._api_response = {
                bs["name"]: {
                    "url": bs["_links"]["content"]["href"],
                    "checksum": f"{bs['checkSum']['checkSumAlgorithm']}:{bs['checkSum']['value']}",
                    "size": bs.get("sizeBytes"),
                }
                for bs in bitstreams
            }

        return self._api_response

    def download_url(self, file_name):
        """Use the repository API to get the download URL for a file given the archive URL.

        Parameters
        ----------
        file_name : :class:`str`
            The name of the file in the archive that will be downloaded.

        Returns
        -------
        download_url : :class:`str`
            The HTTP URL that can be used to download the file.

        """
        return self.api_response[file_name]["url"]

    def file_size(self, file_name):
        """Return the size of a file in bytes, or ``None`` if unavailable.

        Parameters
        ----------
        file_name : :class:`str`
            The name of the file in the archive.

        Returns
        -------
        size : :class:`int` or None
            The file size in bytes.

        """
        return self.api_response[file_name].get("size")

    def populate_registry(self, pooch):
        """Populate the registry using the data repository's API.

        Parameters
        ----------
        pooch : :class:`pooch.Pooch`
            The pooch instance that the registry will be added to.

        """
        for name, info in self.api_response.items():
            pooch.registry[name] = info["checksum"]


def doi_to_repository(doi):
    """Instantiate a data repository instance from a given DOI.

    This function implements the chain of responsibility dispatch
    to the correct data repository class.

    Parameters
    ----------
    doi : :class:`str`
        The DOI of the archive.

    Returns
    -------
    data_repository : :class:`DataRepository`
        The data repository object.

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
    logger = get_logger()
    archive_url = None
    for attempt in range(MAX_RETRIES):
        try:
            archive_url = doi_to_url(doi, timeout=DEFAULT_TIMEOUT)
            break
        except (ConnectionError, Timeout) as e:
            wait = BACKOFF_FACTOR * (2 ** attempt)
            if attempt == 0:
                logger.warning("Server is slow to respond, retrying with exponential backoff...")
            logger.debug(f"  Attempt {attempt + 1}/{MAX_RETRIES} failed ({type(e).__name__}), waiting {wait:.0f}s")
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
