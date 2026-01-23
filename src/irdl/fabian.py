from pathlib import Path
from zipfile import ZipFile

import pooch as po
import pyfar as pf

from irdl.downloader import CACHE_DIR, pooch_from_doi, process


def get_fabian(kind: str = "measured", hato: int = 0, path: str = CACHE_DIR):
    """Download and extract the FABIAN HRTF Database v4 from DepositOnce.

    DOI: 10.14279/depositonce-5718.5

    Parameters
    ----------
    kind : str
        Type of HRTF to download. Either 'measured' or 'modeled'.
    hato : int
        Head-above-torso-rotation of HRTFs in degrees.
        Either 0, 10, 20, 30, 40, 50, 310, 320, 330, 340 or 350.
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
    assert kind in ["measured", "modeled"], "kind must be either 'measured' or 'modeled'"
    assert hato in [0, 10, 20, 30, 40, 50, 310, 320, 330, 340, 350], (
        "hato must be one of [0, 10, 20, 30, 40, 50, 310, 320, 330, 340, 350]"
    )

    path = Path(path) / "FABIAN"
    doi = "10.14279/depositonce-5718.5"
    zipfile = "FABIAN_HRTF_DATABASE_v4.zip"

    pup = pooch_from_doi(doi, path=path)
    pup.fetch(zipfile, progressbar=True)

    logger = po.get_logger()

    @process
    def extract(file, process=True):
        if process:
            with ZipFile(Path(path) / zipfile, "r") as zf:
                for name in zf.namelist():
                    if name.endswith(file.name):
                        zf.getinfo(name).filename = Path(name).name
                        logger.info(f"Extracting {name} to {file.parent / Path(name).name}")
                        zf.extract(name, path=file.parent)
        data = dict(
            zip(
                ("impulse_response", "source_coordinates", "receiver_coordinates"),
                pf.io.read_sofa(file),
                strict=True,
            )
        )
        return data

    return extract(path / f"FABIAN_HRIR_{kind}_HATO_{hato}.sofa", action="fetch", pup=pup)
