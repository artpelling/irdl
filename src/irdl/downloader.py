"""Implements download and post-processing based on pooch."""

from pathlib import Path

import pooch as po
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn, TransferSpeedColumn

from irdl.repositories import doi_to_repository

#: The cache directory for storage of the temporary downloads. Defaults to the user cache directory.
CACHE_DIR = po.os_cache("irdl")


class RichProgressBar:
    """Wraps :class:`rich.progress.Progress` to satisfy the pooch progress bar interface.

    Pooch expects an object with a ``total`` attribute and ``update``, ``reset``, and
    ``close`` methods. This class provides that interface backed by a Rich progress bar.
    """

    def __init__(self, description: str, preset_total: int = 0):
        self._progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
        )
        self._description = description
        self._task_id = None
        # Pooch sets self.total from the HTTP Content-Length header. If the server omits
        # that header, pooch sets it to 0. In that case, fall back to the preset value
        # from the repository API so the bar can show real progress.
        self._preset_total = preset_total
        self.total = 0

    @property
    def total(self):
        """Total download size in bytes."""
        return self._total

    @total.setter
    def total(self, value):
        # Use the API-supplied size when the server omits Content-Length (value == 0).
        self._total = value or self._preset_total
        if self._task_id is not None:
            self._progress.update(self._task_id, total=self._total or None)

    def update(self, n: int) -> None:
        """Advance the progress bar by *n* bytes."""
        if self._task_id is None:
            self._progress.start()
            self._task_id = self._progress.add_task(
                self._description, total=self.total if self.total else None
            )
        self._progress.advance(self._task_id, n)

    def reset(self) -> None:
        """Reset the completed byte count to zero (called by pooch before the final fill)."""
        if self._task_id is not None:
            self._progress.reset(self._task_id, total=self.total if self.total else None)

    def close(self) -> None:
        """Fill to 100 % and stop the progress display."""
        if self._task_id is not None:
            if self.total:
                self._progress.update(self._task_id, completed=self.total)
            self._progress.stop()
            self._task_id = None


def fetch(pup: po.Pooch, fname: str) -> str:
    """Fetch a file from a pooch registry, displaying a Rich progress bar.

    Parameters
    ----------
    pup : :class:`pooch.Pooch`
        The Pooch instance managing the registry.
    fname : :class:`str`
        The file name to fetch (must be registered in *pup*).

    Returns
    -------
    full_path : :class:`str`
        The absolute path to the fetched file on disk.

    """
    preset_total = getattr(pup, "file_sizes", {}).get(fname) or 0
    return pup.fetch(fname, progressbar=RichProgressBar(fname, preset_total=preset_total))


def pooch_from_doi(doi, path=CACHE_DIR):
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
    # Attach file sizes from the repository API for use by the progress bar.
    if hasattr(repository, "file_size"):
        pup.file_sizes = {file: repository.file_size(file_name=file) for file in pup.registry}
    else:
        pup.file_sizes = {}
    return pup


def process(func):
    """Decorator to process downloaded files.

    The decorated function should take two arguments: the input file name and the output file name.
    The decorator checks if the output file already exists and is up to date. If so, it returns the
    output file name. Otherwise, it calls the decorated function to process the input file and
    create the output file.
    """

    def check_process(fname, action, pup=None):
        logger = po.get_logger()
        fname = Path(fname)
        if fname.exists() and action == "fetch":
            logger.info(f"The file '{fname}' exists is up to date.")
            return func(fname, process=False)
        else:
            logger.info(f"Processing and writing to '{fname}'.")
            fname.parent.mkdir(parents=True, exist_ok=True)
            return func(fname, process=True)

    return check_process
