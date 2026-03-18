"""The FABIAN head-related transfer function data base."""

from pathlib import Path
from zipfile import ZipFile

import h5py as h5
import numpy as np
import pooch as po
import pyfar as pf

from irdl.downloader import CACHE_DIR, pooch_from_doi, process


def sofa_to_pyfar(file):
    """Load data from a SOFA file and return as dictionary of pyfar objects.

    Parameters
    ----------
    file : :class:`pathlib.Path` or :class:`str`
        Path to the SOFA file.

    Returns
    -------
    data : :class:`dict`
        Dictionary with the following keys:

        - ``'impulse_response'`` : :class:`pyfar.Signal` — Impulse response data.
        - ``'source_coordinates'`` : :class:`pyfar.Coordinates` — Source positions.
        - ``'receiver_coordinates'`` : :class:`pyfar.Coordinates` — Receiver positions.

    """
    return dict(
        zip(
            ("impulse_response", "source_coordinates", "receiver_coordinates"),
            pf.io.read_sofa(file),
            strict=True,
        )
    )


def load_sofa(file):
    """Load raw arrays from a SOFA file.

    Parameters
    ----------
    file : :class:`pathlib.Path` or :class:`str`
        Path to the SOFA file.

    Returns
    -------
    data : :class:`dict`
        Dictionary with the following keys:

        - ``'impulse_response'`` : :class:`numpy.ndarray` — Impulse response data.
        - ``'source_coordinates'`` : :class:`numpy.ndarray` — Source positions as cartesian
          coordinates.
        - ``'receiver_coordinates'`` : :class:`numpy.ndarray` — Receiver positions as cartesian
          coordinates.
        - ``'sampling_rate'`` : :class:`float` — Sampling rate in Hz.

    """
    pyfar_obj = pf.io.read_sofa(file)

    return {
        "impulse_response": pyfar_obj[0].time,
        "source_coordinates": pyfar_obj[1].cartesian,
        # fabian need the squeeze, maybe other datasets dont?
        "receiver_coordinates": np.squeeze(pyfar_obj[2].cartesian, axis=1),
        "sampling_rate": pyfar_obj[0].sampling_rate,
    }


def sofa_to_h5(file, extracted_already):
    """Convert a SOFA file to HDF5 format and return the path.

    Parameters
    ----------
    file : :class:`pathlib.Path` or :class:`str`
        Path to the SOFA file.
    extracted_already : :class:`bool`
        Check whether the SOFA file existed before extraction. If ``False``, the SOFA
        file is deleted after conversion.

    Returns
    -------
    h5_path : :class:`pathlib.Path`
        Path to the converted HDF5 file.

    """
    # define h5 file path
    h5_path = Path(file).with_suffix(".h5")
    # if files does not exist already
    if not h5_path.exists():
        data = load_sofa(file)
        # parse dictionary to h5 file
        with h5.File(h5_path, "w") as f:
            data_group = f.create_group("data")
            data_group.create_dataset("impulse_response", data=data["impulse_response"])
            location_group = data_group.create_group("location")
            location_group.create_dataset("source", data=data["source_coordinates"])
            location_group.create_dataset("receiver", data=data["receiver_coordinates"])
            metadata_group = f.create_group("metadata")
            metadata_group.create_dataset("sampling_rate", data=data["sampling_rate"])
    # delete sofa file if the file was just extracted for the conversion
    if not extracted_already:
        Path(file).unlink(missing_ok=True)
    return h5_path


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
        Output format of the returned data.
        Either ``'pyfar'`` (default), ``'hdf5'``, or ``'numpy'``.

    Returns
    -------
    data : :class:`dict` or :class:`pathlib.Path`
        Returned data depends on ``output_format``:
        
        - ``'pyfar'`` : :class:`dict` with keys ``'impulse_response'`` (:class:`pyfar.Signal`),
          ``'source_coordinates'`` (:class:`pyfar.Coordinates`), and
          ``'receiver_coordinates'`` (:class:`pyfar.Coordinates`).
        - ``'hdf5'`` : :class:`pathlib.Path` to the HDF5 file containing the data.
        - ``'numpy'`` : :class:`dict` with keys ``'impulse_response'`` (:class:`numpy.ndarray`),
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
        # check if file was extracted already
        extracted_already = file.exists()
        if process:
            with ZipFile(Path(path) / zipfile, "r") as zf:
                for name in zf.namelist():
                    if name.endswith(file.name):
                        zf.getinfo(name).filename = Path(name).name
                        logger.info(f"Extracting {name} to {file.parent / Path(name).name}")
                        zf.extract(name, path=file.parent)

        match output_format:
            case "pyfar":
                return sofa_to_pyfar(file)
            case "hdf5":
                return sofa_to_h5(file, extracted_already=extracted_already)
            case "numpy":
                return load_sofa(file)

    return extract(path / f"FABIAN_HRIR_{kind}_HATO_{hato}.sofa", action="fetch", pup=pup)
