"""Impulse response datasets from the Department of Engineering Acoustics, TU Berlin.

- MIRACLE: Microphone Array Impulse Response Dataset for Acoustic Learning.
- SRIRACHA: Shoebox Room Impulse Response Archive with Varying Absorption.

"""

from pathlib import Path

import h5py as h5
import numpy as np
import pyfar as pf

from irdl.downloader import CACHE_DIR, pooch_from_doi, process


def load_h5(file):
    """Load raw arrays from an HDF5 file into a dictionary.

    Parameters
    ----------
    file : :class:`pathlib.Path` or :class:`str`
        Path to the HDF5 file.

    Returns
    -------
    data : :class:`dict`
        Dictionary with the following keys:

        - ``'impulse_response'`` : :class:`numpy.ndarray` — Impulse response data.
        - ``'receiver_coordinates'`` : :class:`numpy.ndarray` — Receiver positions as cartesian
          coordinates.
        - ``'source_coordinates'`` : :class:`numpy.ndarray` — Corrected source positions as
          cartesian coordinates.
        - ``'speed_of_sound'`` : :class:`numpy.ndarray` — Speed of sound per source position in
          m/s.
        - ``'temperature'`` : :class:`numpy.ndarray` — Ambient temperature per source position
          in °C.
        - ``'sampling_rate'`` : :class:`int` — Sampling rate in Hz.
        - ``'humidity'`` : :class:`numpy.ndarray` *(optional)* — Ambient humidity per source
          position, if present in the file.

    """
    with h5.File(file, "r") as f:
        data = {
            # data
            "impulse_response": f["data"]["impulse_response"][()],
            "receiver_coordinates": f["data"]["location"]["receiver"][()],
            "source_coordinates": f["data"]["location"]["source"][()],
            # metadata
            "speed_of_sound": f["metadata"]["c0"][()],
            "temperature": f["metadata"]["temperature"][()],
            "sampling_rate": f["metadata"]["sampling_rate"][()],
        }

        if "humidity" in f["metadata"]:
            data["humidity"] = f["metadata"]["humidity"][()]

    return data


def h5_to_pyfar(file, dataset_split=None):
    """Load data from an HDF5 file and convert to pyfar objects.

    Loads raw data via :func:`load_h5` and converts impulse responses to
    :class:`pyfar.Signal` and coordinate arrays to :class:`pyfar.Coordinates`.
    Optionally filters source positions and corresponding impulse responses to a
    analogous to the ``dataset_split`` parameter in :func:`get_sriracha`.

    Parameters
    ----------
    file : :class:`pathlib.Path` or :class:`str`
        Path to the HDF5 file.
    dataset_split : :class:`str` or None
        Filter source positions and impulse responses.
        Analogous to ``dataset_split`` in :func:`get_sriracha`.
        One of ``'C1'``, ``'C2'``, ``'C3'``, ``'C4'``, or ``None`` (default).

    Returns
    -------
    data : :class:`dict`
        Dictionary with the following keys:

        - ``'impulse_response'`` : :class:`pyfar.Signal` — Impulse response data.
        - ``'source_coordinates'`` : :class:`pyfar.Coordinates` — Corrected source positions.
        - ``'receiver_coordinates'`` : :class:`pyfar.Coordinates` — Receiver positions.
    """
    data = load_h5(file)

    if dataset_split:
        data = split_data(data, dataset_split)

    data["impulse_response"] = pf.Signal(data["impulse_response"], sampling_rate=data["sampling_rate"])
    data["source_coordinates"] = pf.Coordinates(*data["source_coordinates"].T)
    data["receiver_coordinates"] = pf.Coordinates(*data["receiver_coordinates"].T)

    for key in ["sampling_rate", "speed_of_sound", "temperature", "humidity"]:
        data.pop(key, None)

    return data


def split_data(data, dataset_split):
    """Filter a data dictionary to a subgroup of source positions.

    Splits source positions and corresponding impulse responses into one of four
    dataset splits analogous to the ``dataset_split`` parameter in :func:`get_sriracha`.

    Parameters
    ----------
    data : :class:`dict`
        Dictionary of numpy arrays as returned by :func:`load_h5`.
    dataset_split : :class:`str`
        Spatial quadrant to filter to. One of ``'C1'``, ``'C2'``, ``'C3'``, or ``'C4'``.

    Returns
    -------
    data : :class:`dict`
        Input dictionary with ``'source_coordinates'``, and ``'impulse_response'`` filtered
        to the requested dataset split.

    """
    # look up dictionary for the slicing indices
    offsets = {"C1": (0, 0), "C2": (0, 1), "C3": (1, 0), "C4": (1, 1)}
    row, column = offsets[dataset_split]

    # get array sizes for variable slicing
    n = int(np.sqrt(data["source_coordinates"].shape[0]))
    ir_shape = data["impulse_response"].shape

    # reshaping, slicing and reshape back to original shape
    data["source_coordinates"] = data["source_coordinates"].reshape(n, n, 3)[row::2, column::2, :].reshape(-1, 3)
    data["impulse_response"] = (
        data["impulse_response"].reshape(n, n, *ir_shape[1:])[row::2, column::2, :].reshape(-1, *ir_shape[1:])
    )

    return data


def save_h5(data, path):
    """Save a data dictionary of numpy arrays to an HDF5 file.

    Writes the contents of a data dictionary as returned by :func:`load_h5` or
    :func:`split_data` to an HDF5 file following the same structure as the
    MIRACLE and SRIRACHA datasets.

    Parameters
    ----------
    data : :class:`dict`
        Dictionary of numpy arrays as returned by :func:`load_h5` or :func:`split_data`.
    path : :class:`pathlib.Path` or :class:`str`
        Path to the HDF5 file to write.

    Returns
    -------
    path : :class:`pathlib.Path`
        Path to the written HDF5 file.

    """
    with h5.File(path, "w") as f:
        data_group = f.create_group("data")
        data_group.create_dataset("impulse_response", data=data["impulse_response"])
        location_group = data_group.create_group("location")
        location_group.create_dataset("source", data=data["source_coordinates"])
        location_group.create_dataset("receiver", data=data["receiver_coordinates"])
        metadata_group = f.create_group("metadata")
        metadata_group.create_dataset("c0", data=data["speed_of_sound"])
        metadata_group.create_dataset("temperature", data=data["temperature"])
        metadata_group.create_dataset("sampling_rate", data=data["sampling_rate"])
        if "humidity" in data:
            metadata_group.create_dataset("humidity", data=data["humidity"])
    return path


def get_miracle(scenario: str = "A1", dataset_split: str = None, path: str = CACHE_DIR, output_format: str = "pyfar"):
    """Download and extract the MIRACLE database from DepositOnce.

    DOI: `10.14279/depositonce-20837 <https://doi.org/10.14279/depositonce-20837>`_

    Parameters
    ----------
    scenario : :class:`str`
        Name of the scenario to download. Either ``'A1'``, ``'A2'``, ``'D1'`` or ``'R2'``.
    dataset_split : :class:`str` or None
        Artificial dataset split. Analogous to ``dataset_split`` in :func:`get_sriracha`.
        One of ``'C1'``, ``'C2'``, ``'C3'``, ``'C4'``, or ``None`` (default).
    path : :class:`str` or :class:`pathlib.Path`
        Path to the directory where the data should be stored. Will be overwritten, if the
        environment variable `IRDL_DATA_DIR` is set. Default is the user cache directory.
    output_format : :class:`str`
        Output format of the returned data.
        Either ``'pyfar'`` (default), ``'hdf5'``, or ``'numpy'``.


    Returns
    -------
    data : :class:`dict` or :class:`pathlib.Path`
        Returned data depends on ``output_format``:

        - ``'pyfar'``: :class:`dict` with keys ``'impulse_response'`` (:class:`pyfar.Signal`),
          ``'source_coordinates'`` (:class:`pyfar.Coordinates`), and
          ``'receiver_coordinates'`` (:class:`pyfar.Coordinates`)
        - ``'hdf5'``: :class:`pathlib.Path` to the HDF5 file containing the data.
        - ``'numpy'``: :class:`dict` with keys ``'impulse_response'`` (:class:`numpy.ndarray`),
          ``'source_coordinates'`` (:class:`numpy.ndarray`),
          ``'receiver_coordinates'`` (:class:`numpy.ndarray`), and
          ``'sampling_rate'`` (:class:`float`).

    """
    assert output_format in ["pyfar", "hdf5", "numpy"], "unknown output format"
    assert scenario in ["A1", "A2", "D1", "R2"], "scenario must be one of ['A1', 'A2', 'D1', 'R2']"
    assert dataset_split in [None, "C1", "C2", "C3", "C4"], "dataset_split must be None or in [C1, C2, C3, C4]"

    scenario += ".h5"

    path = Path(path) / "MIRACLE" / "raw"
    doi = "10.14279/depositonce-20837"

    pup = pooch_from_doi(doi, path=path)
    pup.fetch(scenario, progressbar=True)

    @process  # is always true because we dont extract and pup.fetch checks if file exists already => remove?
    def process_miracle(file, process=True):
        match output_format:
            case "hdf5":
                if dataset_split is None:
                    return file
                else:
                    h5_path = file.with_stem(file.stem + f"-{dataset_split}")
                    return save_h5(split_data(load_h5(file), dataset_split), h5_path)
            case "pyfar":
                return h5_to_pyfar(file, dataset_split=dataset_split)
            case "numpy":
                if dataset_split is None:
                    return load_h5(file)
                else:
                    return split_data(load_h5(file), dataset_split)

    return process_miracle(path / scenario, action="fetch", pup=pup)


def get_sriracha(
    scenario: str = "SR1-D", dataset_split: str = None, path: str = CACHE_DIR, output_format: str = "pyfar"
):
    """Download and extract the SRIRACHA database from DepositOnce.

    DOI: `10.14279/depositonce-23943 <https://doi.org/10.14279/depositonce-23943>`_

    Parameters
    ----------
    scenario : :class:`str`
        Name of the scenario to download. One of ``'SR1'``, ``'SRA1'``, ``'SR1-D'``,
        ``'SRA1-D'``, ``'SR2'``, ``'SRA2'``, ``'SR2-D'``, or ``'SRA2-D'``.
    dataset_split : :class:`str` or None
        Optional dataset split for full-plane scenarios.
        One of ``'C1'``, ``'C2'``, ``'C3'``, ``'C4'``, or ``None`` (default).
        Dense scenarios (ending in ``-D``)do not have splits.
    path : :class:`str` or :class:`pathlib.Path`
        Path to the directory where the data should be stored. Will be overwritten, if the
        environment variable `IRDL_DATA_DIR` is set. Default is the user cache directory.
    output_format : :class:`str`
        Output format of the returned data.
        Either ``'pyfar'`` (default), ``'hdf5'``, or ``'numpy'``.

    Returns
    -------
    data : :class:`dict` or :class:`pathlib.Path`
        Returned data depends on ``output_format``:

        - ``'pyfar'``: :class:`dict` with keys ``'impulse_response'`` (:class:`pyfar.Signal`),
          ``'source_coordinates'`` (:class:`pyfar.Coordinates`),
          ``'receiver_coordinates'`` (:class:`pyfar.Coordinates`), and
        - ``'hdf5'``: :class:`pathlib.Path` to the HDF5 file containing the data.
        - ``'numpy'``: :class:`dict` with keys ``'impulse_response'`` (:class:`numpy.ndarray`),
          ``'source_coordinates'`` (:class:`numpy.ndarray`),
          ``'receiver_coordinates'`` (:class:`numpy.ndarray`),
          ``'speed_of_sound'`` (:class:`numpy.ndarray`),
          ``'temperature'`` (:class:`numpy.ndarray`),
          ``'sampling_rate'`` (:class:`int`), and optionally
          ``'humidity'`` (:class:`numpy.ndarray`).

    """
    assert output_format in ["pyfar", "hdf5", "numpy"], "unknown output format"
    assert scenario in ["SR1", "SRA1", "SR1-D", "SRA1-D", "SR2", "SRA2", "SR2-D", "SRA2-D"], (
        "scenario must be one of [SR1, SRA1, SR1-D, SRA1-D, SR2, SRA2, SR2-D, SRA2-D]"
    )

    assert dataset_split in [None, "C1", "C2", "C3", "C4"], "dataset_split must be None or in [C1, C2, C3, C4]"

    assert not (scenario[-1] != "D" and dataset_split is None), "full datasets need a split"
    assert not (scenario[-1] == "D" and dataset_split is not None), "dense datasets do not have splits"

    if dataset_split is None:
        scenario += ".h5"
    else:
        scenario += "-" + dataset_split + ".h5"

    path = Path(path) / "SRIRACHA" / "raw"
    doi = "10.14279/depositonce-23943"

    pup = pooch_from_doi(doi, path=path)
    pup.fetch(scenario, progressbar=True)

    @process  # is always true because we dont extract and pup.fetch checks if file exists already => remove?
    def process_sriracha(file, process=True):
        match output_format:
            case "hdf5":
                return file
            case "pyfar":
                return h5_to_pyfar(file)
            case "numpy":
                return load_h5(file)

    return process_sriracha(path / scenario, action="fetch", pup=pup)
