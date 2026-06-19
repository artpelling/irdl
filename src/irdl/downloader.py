"""Implements download and post-processing based on pooch."""

from collections.abc import Mapping
from pathlib import Path

import pooch as po
from requests.exceptions import RequestException

from irdl.cache import IRDL_CACHE_DIR
from irdl.logging import RichProgressBar, logger, pooch_logger
from irdl.repositories import DEFAULT_TIMEOUT, _make_session, doi_to_repository


def _fetch(pup: po.Pooch, fname: str) -> str:
    """Fetch a file from a pooch registry, displaying a Rich progress bar.

    Parameters
    ----------
    pup : pooch.Pooch
        The Pooch instance managing the registry.
    fname : str
        The file name to fetch (must be registered in pup).

    Returns
    -------
    full_path : str
        The absolute path to the fetched file on disk.

    """
    pooch_logger.debug("Fetching %s", fname)
    preset_total = getattr(pup, "file_sizes", {}).get(fname) or 0
    return pup.fetch(fname, progressbar=RichProgressBar(fname, preset_total=preset_total))


def _pooch_from_doi(doi: str, path: str = IRDL_CACHE_DIR) -> po.Pooch:
    """Create a Pooch instance from a DOI.

    Parameters
    ----------
    doi : str
        The DOI of the archive.
    path : str, optional
        Path to the directory where the data should be stored. Default is IRDL_CACHE_DIR.

    Returns
    -------
    pup : pooch.Pooch
        The Pooch instance.

    """
    pup = po.create(path=path, base_url=doi, retry_if_failed=2)
    repository = doi_to_repository(doi)
    repository.populate_registry(pup)
    for file in pup.registry:
        pup.urls[file] = repository.download_url(file_name=file)
    # Attach file sizes from the repository API for use by the progress bar.
    if hasattr(repository, "file_size"):
        pup.file_sizes = {file: repository.file_size(file_name=file) for file in pup.registry}
    else:
        pup.file_sizes = {}
    return pup


def _pooch_from_static_registry(
    path: str | Path,
    registry: Mapping[str, str | None],
    urls: Mapping[str, str],
) -> po.Pooch:
    """Create a Pooch instance for direct static-file downloads.

    Parameters
    ----------
    path : str or :class:`pathlib.Path`
        Directory where downloaded files should be stored.
    registry : mapping
        Mapping of file names to known hashes.
    urls : mapping
        Mapping of file names to direct download URLs.

    Returns
    -------
    pup : pooch.Pooch
        The Pooch instance.
    """
    pup = po.create(path=path, base_url="", registry=dict(registry), urls=dict(urls), retry_if_failed=2)
    pup.file_sizes = {}
    return pup


def _pooch_from_sonicom_database(
    path: str | Path,
    database_url: str,
    fname: str,
    checksum: str | None = None,
) -> po.Pooch:
    """Create a Pooch instance for one SONICOM file resolved from a database manifest.

    Parameters
    ----------
    path : str or :class:`pathlib.Path`
        Directory where downloaded files should be stored.
    database_url : str
        SONICOM database landing page, for example
        ``https://ecosystem.sonicom.eu/databases/76``.
    fname : str
        Datafile name to resolve from the SONICOM download manifest.
    checksum : str or None, optional
        Optional checksum for the resolved file, for example ``sha256:...``.

    Returns
    -------
    pup : pooch.Pooch
        The Pooch instance for the resolved file.
    """
    manifest_url = f"{database_url.rstrip('/')}/download?type=json"
    with _make_session() as session:
        response = session.get(manifest_url, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        payload = response.json()

        entries = payload.get("data")
        if not isinstance(entries, list):
            msg = f"SONICOM database API at {manifest_url!r} did not return a 'data' list"
            raise TypeError(msg)

        entry = next((item for item in entries if item.get("Datafile Name") == fname), None)
        if entry is None:
            msg = f"SONICOM database {database_url!r} does not list file {fname!r}"
            raise FileNotFoundError(msg)

        resolved_url = entry.get("Datafile URL")
        if not isinstance(resolved_url, str):
            msg = f"SONICOM database {database_url!r} did not provide a valid URL for {fname!r}"
            raise TypeError(msg)

        file_size = None
        try:
            head_response = session.head(resolved_url, allow_redirects=True, timeout=DEFAULT_TIMEOUT)
            head_response.raise_for_status()
            resolved_url = head_response.url
            content_length = head_response.headers.get("Content-Length")
            if content_length is not None:
                file_size = int(content_length)
        except (OSError, RequestException, ValueError) as exc:  # pragma: no cover
            logger.debug("HEAD resolution failed for %s: %s", resolved_url, exc)

    pup = po.create(path=path, base_url="", registry={fname: checksum}, urls={fname: resolved_url}, retry_if_failed=2)
    pup.file_sizes = {fname: file_size} if file_size is not None else {}
    return pup
