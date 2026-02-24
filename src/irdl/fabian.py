"""The FABIAN head-related transfer function data base."""

from pathlib import Path
from zipfile import ZipFile

import pooch as po
import pyfar as pf

import h5py as h5
import numpy as np

from irdl.downloader import CACHE_DIR, pooch_from_doi, process

def load_sofa(file):
    """Load raw arrays from a SOFA file.

    Parameters
    ----------
    file : :class:`pathlib.Path` or :class:`str`
        Path to the SOFA file.

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
    pyfar_obj = pf.io.read_sofa(file)
    ir   = pyfar_obj[0].time
    fs   = pyfar_obj[0].sampling_rate
    spos = pyfar_obj[1].cartesian
    rpos = pyfar_obj[2].cartesian

    rpos = np.squeeze(rpos, axis=1)
    return ir, fs, spos, rpos

def get_fabian(kind: str = "measured", hato: int = 0, path: str = CACHE_DIR, output_format: str = "pyfar"):
    """Download and extract the FABIAN HRTF Database v4 from DepositOnce.

    DOI: `10.14279/depositonce-5718.5 <https://doi.org/10.14279/depositonce-5718.5>`_

    Parameters
    ----------
    kind : :class:`str`
        Type of HRTF to download. Either ``'measured'`` or ``'modeled'``.
    hato : :class:`int`
        Head-above-torso-rotation of HRTFs in degrees.
        Either 0, 10, 20, 30, 40, 50, 310, 320, 330, 340 or 350.
    path : :class:`str` or :class:`pathlib.Path`
        Path to the directory where the data should be stored. Will be overwritten, if the
        environment variable ``IRDL_DATA_DIR`` is set. Default is the user cache directory.
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
    assert kind in ["measured", "modeled"], "kind must be either 'measured' or 'modeled'"
    assert hato in [0, 10, 20, 30, 40, 50, 310, 320, 330, 340, 350], (
        "hato must be one of [0, 10, 20, 30, 40, 50, 310, 320, 330, 340, 350]"
    )
    assert output_format in ["pyfar", "hdf5", "numpy"], "unknown output format"

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

        match output_format:
            case "pyfar":
                data = dict(
                    zip(
                        ("impulse_response", "source_coordinates", "receiver_coordinates"),
                        pf.io.read_sofa(file),
                        strict=True,
                    )
                )
                return data
        
            case "hdf5":
                #define h5 file path
                h5_path = file.with_suffix(".h5")
                #if files does not exist already
                if not h5_path.exists():
                    #load data from sofa file
                    ir, fs, spos, rpos = load_sofa(file)
                    #convert pyfar object to h5 file
                    with h5.File(h5_path , "w") as f: 
                        data_group = f.create_group("data")
                        data_group.create_dataset("impulse_response", data=ir)
                        location_group = data_group.create_group("location")
                        location_group.create_dataset("receiver", data=rpos)
                        location_group.create_dataset("source", data=spos)
                    
                        metadata_group = f.create_group("metadata")
                        metadata_group.create_dataset("sampling_rate", data=fs)
                #delete sofa file
                Path(file).unlink(missing_ok=True)

                return h5_path
            
            case "numpy":
                #read sofa and convert pyfar object into numpy arrays and a float
                ir, fs, spos, rpos = load_sofa(file)

                data = {
                    "impulse_response" : ir, 
                    "source_coordinates": spos, 
                    "receiver_coordinates" : rpos,
                    "sampling_rate" : fs,
                }

                return data


    return extract(path / f"FABIAN_HRIR_{kind}_HATO_{hato}.sofa", action="fetch", pup=pup)
