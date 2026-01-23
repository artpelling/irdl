from pathlib import Path

import h5py as h5
import pyfar as pf

from irdl.downloader import CACHE_DIR, pooch_from_doi, process


def get_miracle(scenario: str = "A1", path: str = CACHE_DIR):
    """Download and extract the MIRACLE database from DepositOnce.

    DOI: 10.14279/depositonce-20837

    Parameters
    ----------
    scenario : str
        Name of the scenario to download. Either 'A1', 'A2', 'D1', or 'R2'.
    path : str or `pathlib.Path`
        Path to the directory where the data should be stored. Will be overwritten, if the
        environment variable `IRDL_DATA_DIR` is set. Default is the user cache directory.

    Returns
    -------
    data : dict
        Dictionary containing the impulse responses and the source and receiver coordinates. The
        impulse responses are stored in the key 'impulse_response' as a :class:`pyfar.Signal`. The
        source and receiver coordinates are stored as :class:`pyfar.Coordinates` in the keys
        'source_coordinates' and 'receiver_coordinates', respectively.

    """
    assert scenario in ["A1", "A2", "D1", "R2"], "scenario must be one of ['A1', 'A2', 'D1', 'R2']"
    scenario += ".h5"

    path = Path(path) / "MIRACLE" / "raw"
    doi = "10.14279/depositonce-20837"

    pup = pooch_from_doi(doi, path=path)
    pup.fetch(scenario, progressbar=True)

    @process
    def process_miracle(file, process=True):
        data = dict()
        with h5.File(file, "r") as f:
            ir = f.get("data")["impulse_response"][()]
            fs = f.get("metadata")["sampling_rate"][()]
            spos = f.get("data")["location"]["source"][()]
            rpos = f.get("data")["location"]["receiver"][()]

        data["impulse_response"] = pf.Signal(ir, sampling_rate=fs)
        data["source_coordinates"] = pf.Coordinates(*spos.T)
        data["receiver_coordinates"] = pf.Coordinates(*rpos.T)
        return data

    return process_miracle(path / scenario, action="fetch", pup=pup)
