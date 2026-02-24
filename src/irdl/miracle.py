"""MIRACLE - Microphone Array Impulse Response Dataset for Acoustic Learning."""

from pathlib import Path

import h5py as h5
import pyfar as pf

from irdl.downloader import CACHE_DIR, pooch_from_doi, process

def load_h5(file):
    """Load raw arrays from an HDF5 file.

    Parameters
    ----------
    file : :class:`pathlib.Path` or :class:`str`
        Path to the HDF5 file.

    Returns
    -------
    ir : :class:`numpy.ndarray`
        Impulse response data.
    fs : :class:`float`
        Sampling rate in Hz.
    spos : :class:`numpy.ndarray`
        Source positions as cartesian coordinates.
    rpos : :class:`numpy.ndarray`
        Receiver positions as cartesian coordinates.

    """
    with h5.File(file, "r") as f:
        ir   = f["data"]["impulse_response"][()]
        fs   = f["metadata"]["sampling_rate"][()]
        spos = f["data"]["location"]["source"][()]
        rpos = f["data"]["location"]["receiver"][()]
    return ir, fs, spos, rpos

def get_miracle(scenario: str = "A1", path: str = CACHE_DIR, output_format: str = "pyfar"):
    """Download and extract the MIRACLE database from DepositOnce.

    DOI: `10.14279/depositonce-20837 <https://doi.org/10.14279/depositonce-20837>`_

    Parameters
    ----------
    scenario : :class:`str`
        Name of the scenario to download. Either ``'A1'``, ``'A2'``, ``'D1'`` or ``'R2'``.
    path : :class:`str` or :class:`pathlib.Path`
        Path to the directory where the data should be stored. Will be overwritten, if the
        environment variable `IRDL_DATA_DIR` is set. Default is the user cache directory.
    output_format : :class:`str`
    Output format of the returned data. Either ``'pyfar'`` (default), ``'hdf5'``, or ``'numpy'``.
    
    Returns
    -------
    data : :class:`dict` or :class:`pathlib.Path`
        Returned data depends on ``output_format``:

        - ``'pyfar'``: :class:`dict` with keys ``'impulse_response'`` (:class:`pyfar.Signal`),
          ``'source_coordinates'`` (:class:`pyfar.Coordinates`), and
          ``'receiver_coordinates'`` (:class:`pyfar.Coordinates`).
        - ``'hdf5'``: :class:`pathlib.Path` to the HDF5 file containing the data.
        - ``'numpy'``: :class:`dict` with keys ``'impulse_response'`` (:class:`numpy.ndarray`),
          ``'source_coordinates'`` (:class:`numpy.ndarray`),
          ``'receiver_coordinates'`` (:class:`numpy.ndarray`), and
          ``'sampling_rate'`` (:class:`float`).

    """
    assert output_format in ["pyfar", "hdf5", "numpy"], "unknown output format"
    assert scenario in ["A1", "A2", "D1", "R2"], "scenario must be one of ['A1', 'A2', 'D1', 'R2']"
    scenario += ".h5"

    path = Path(path) / "MIRACLE" / "raw"
    doi = "10.14279/depositonce-20837"

    pup = pooch_from_doi(doi, path=path)
    pup.fetch(scenario, progressbar=True)

    @process #is always true because we dont extract and pup.fetch checks if file exists already => remove?
    def process_miracle(file, process=True):
        
        match output_format:

            case "hdf5":
                return file
            
            case "pyfar":
                ir, fs, spos, rpos = load_h5(file)

                data = dict()
                data["impulse_response"] = pf.Signal(ir, sampling_rate=fs)
                data["source_coordinates"] = pf.Coordinates(*spos.T)
                data["receiver_coordinates"] = pf.Coordinates(*rpos.T)

                return data
        
            case "numpy":
                ir, fs, spos, rpos = load_h5(file)

                data = {
                    "impulse_response" : ir, 
                    "source_coordinates": spos, 
                    "receiver_coordinates" : rpos,
                    "sampling_rate" : fs,
                }

                return data

    return process_miracle(path / scenario, action="fetch", pup=pup)