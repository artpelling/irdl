from pathlib import Path

import pooch as po

from irdl.repositories import doi_to_repository

CACHE_DIR = po.os_cache("irdl")


def pooch_from_doi(doi, path=CACHE_DIR):
    """Create a Pooch instance from a DOI.

    Parameters
    ----------
    doi : str
        The DOI of the archive.
    path : str
        Path to the directory where the data should be stored.

    Returns
    -------
    pup : Pooch
        The Pooch instance.

    """
    pup = po.create(path=path, base_url=doi, retry_if_failed=2, env="IRDL_DATA_DIR")
    repository = doi_to_repository(doi)
    repository.populate_registry(pup)
    for file in pup.registry.keys():
        pup.urls[file] = repository.download_url(file_name=file)
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
