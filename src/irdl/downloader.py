"""Implements download and post-processing based on pooch."""

from collections.abc import Mapping
from pathlib import Path

import pooch as po

from irdl.cache import IRDL_CACHE_DIR
from irdl.logging import RichProgressBar, logger
from irdl.repositories import doi_to_repository


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
    known_hash = pup.registry.get(fname)
    logger.debug("Provider request: file=%s, url=%s, digest=%s", fname, pup.get_url(fname), known_hash)
    if (Path(pup.path) / fname).exists():
        logger.info("Provider cache: checking %s", fname)
    if known_hash is None:
        logger.warning("No pinned digest for %s; Provider integrity verification is skipped.", fname)
    preset_total = getattr(pup, "file_sizes", {}).get(fname) or 0
    bar = RichProgressBar(fname, preset_total=preset_total)
    try:
        result = pup.fetch(fname, progressbar=bar)
    finally:
        bar.close()
    logger.info("Provider artifact ready: %s (%s)", fname, "digest verified" if known_hash else "unverified")
    return result


def _pooch_from_static_registry(
    path: str | Path,
    registry: Mapping[str, str | None],
    urls: Mapping[str, str],
) -> po.Pooch:
    """Create a Pooch instance for direct downloads with optional digests."""
    pup = po.create(path=path, base_url="", registry=dict(registry), urls=dict(urls), retry_if_failed=2)
    pup.file_sizes = {}
    return pup


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
    logger.info("Resolving canonical Provider DOI: %s", doi)
    with logger.spin(f"Resolving Provider DOI {doi}"):
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
