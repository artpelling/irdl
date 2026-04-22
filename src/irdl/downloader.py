"""Implements download and post-processing based on pooch."""

import pooch as po

from irdl.repositories import doi_to_repository

#: The cache directory for storage of the temporary downloads. Defaults to the user cache directory.
CACHE_DIR = po.os_cache("irdl")


def _pooch_from_doi(doi, path=CACHE_DIR):
    """Create a Pooch instance from a DOI.

    Parameters
    ----------
    doi : :class:`str`
        The DOI of the archive.
    path : :class:`str`
        Path to the directory where the data should be stored.

    Returns
    -------
    pup : :class:`pooch.Pooch`
        The Pooch instance.

    """
    pup = po.create(path=path, base_url=doi, retry_if_failed=2, env="IRDL_DATA_DIR")
    repository = doi_to_repository(doi)
    repository.populate_registry(pup)
    for file in pup.registry.keys():
        pup.urls[file] = repository.download_url(file_name=file)
    return pup
