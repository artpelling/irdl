"""Implements download and post-processing based on pooch."""

import shutil
from pathlib import Path

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


def _move_to_export_dir(cached_path, export_dir):
    """Move a file from the cache directory to a dedicated export directory.

    Parameters
    ----------
    cached_path : :class:`pathlib.Path`
    Path to the file in the cache directory.
    export_dir : :class:`str`, :class:`pathlib.Path`, or None
    Directory to move the file to. If ``None`` or identical to the file's
    parent directory, the file is not moved and ``cached_path`` is returned.

    Returns
    -------
    path : :class:`pathlib.Path`
    Path to the file, either in ``export_dir`` or unchanged if no move was needed.

    """
    # no export_dir specified
    if export_dir is None or Path(export_dir) == cached_path.parent:
        return cached_path
    dest = Path(export_dir) / cached_path.name
    # file exists already in export_dir
    if dest.exists():
        return dest
    # move file from cache to export_dir
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(cached_path, dest)
    return dest
