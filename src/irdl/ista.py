"""Impulse response datasets from the Department of Engineering Acoustics, TU Berlin.

- MIRACLE: Microphone Array Impulse Response Dataset for Acoustic Learning.
- SRIRACHA: Shoebox Room Impulse Response Archive with Varying Absorption.

"""

from pathlib import Path

import h5py as h5
import numpy as np
import pyfar as pf

from irdl.downloader import CACHE_DIR, pooch_from_doi, process

def download_and_merge(scenario, path, pup):
    """Download and merge four quadrant-split HDF5 files into one full-plane dataset.

    Reverses the interleaving performed by :func:`split_data`, writing
    row-by-row to keep memory usage bounded.

    Parameters
    ----------
    scenario : str
        Base scenario name, e.g. ``'SR1'``.
    path : Path
        Directory where HDF5 files are stored.
    pup : pooch.Pooch
        Pooch instance for downloading files.

    Returns
    -------
    output_path : Path
        Path to the merged HDF5 file.

    """
    # check if merged file already exists
    output_path = path / f"{scenario}.h5"
    if output_path.exists():
        return output_path
    
    offsets = {"C1": (0, 0), "C2": (0, 1), "C3": (1, 0), "C4": (1, 1)}

    # download split files
    split_files = {}
    for split in offsets:
        fname = f"{scenario}-{split}.h5"
        pup.fetch(fname, progressbar=True)
        split_files[split] = path / fname

    # read shapes and shared metadata from the first split
    with h5.File(split_files["C1"], "r") as f:
        ir_shape = f["data"]["impulse_response"].shape  
        ir_dtype = f["data"]["impulse_response"].dtype
        n_split = ir_shape[0]
        sampling_rate = f["metadata"]["sampling_rate"][()]
        receiver = f["data"]["location"]["receiver"][()]
        has_humidity = "humidity" in f["metadata"]

    # calculate total number of sources and grid dimension
    n_sources = len(split_files) * n_split
    n_full_grid = int(np.sqrt(n_sources))
    n_split_grid = n_full_grid // 2

    with h5.File(output_path, "w") as out:
        # create groups and datasets
        data_grp = out.create_group("data")
        ir_ds = data_grp.create_dataset(
            "impulse_response", shape=(n_sources, *ir_shape[1:]), dtype=ir_dtype
        )
        loc_grp = data_grp.create_group("location")
        src_ds = loc_grp.create_dataset("source", shape=(n_sources, 3), dtype="float64")
        loc_grp.create_dataset("receiver", data=receiver)

        meta_grp = out.create_group("metadata")
        meta_grp.create_dataset("sampling_rate", data=sampling_rate)
        c0_ds = meta_grp.create_dataset("c0", shape=(n_sources,), dtype="float32")
        temp_ds = meta_grp.create_dataset("temperature", shape=(n_sources,), dtype="float32")
        if has_humidity:
            hum_ds = meta_grp.create_dataset("humidity", shape=(n_sources,), dtype="float32")

        #open all split files
        handles = {s: h5.File(f, "r") for s, f in split_files.items()}
        try:
            # copy data from each split to the correct location in the output datasets
            for split_name, (row, col) in offsets.items():
                f = handles[split_name]
                for r in range(n_split_grid):
                    #index one row of the split grid
                    src = slice(r * n_split_grid, (r + 1) * n_split_grid)
                    # map split-grid-row to full-grid-row
                    grid_row = 2 * r + row
                    # index one row of the full grid, skipping every other entry to interleave splits
                    out = slice(grid_row * n_full_grid + col, grid_row * n_full_grid + n_full_grid, 2)

                    ir_ds[out] = f["data"]["impulse_response"][src]
                    src_ds[out] = f["data"]["location"]["source"][src]
                    c0_ds[out] = f["metadata"]["c0"][src]
                    temp_ds[out] = f["metadata"]["temperature"][src]
                    if has_humidity:
                        hum_ds[out] = f["metadata"]["humidity"][src]
        # close all files
        finally:
            for fh in handles.values():
                fh.close()
        
        # delete split files
        for f in split_files.values():
            f.unlink()

    return output_path


def download_and_merge_vds(scenario, path, pup):
    """Download splits and create a virtual HDF5 file mapping them into one full-plane dataset.

    The resulting file contains no data — only HDF5 virtual-dataset mappings
    that point back into the split files. The split files must remain in place.

    Parameters
    ----------
    scenario : str
        Base scenario name, e.g. ``'SR1'``.
    path : Path
        Directory where HDF5 files are stored.
    pup : pooch.Pooch
        Pooch instance for downloading files.

    Returns
    -------
    output_path : Path
        Path to the virtual HDF5 file.

    """
    # check if merged file already exists
    output_path = path / f"{scenario}.h5"
    if output_path.exists():
        return output_path

    offsets = {"C1": (0, 0), "C2": (0, 1), "C3": (1, 0), "C4": (1, 1)}

    # download split files
    split_files = {}
    for split in offsets:
        fname = f"{scenario}-{split}.h5"
        pup.fetch(fname, progressbar=True)
        split_files[split] = fname  # filename only — keeps VDS relocatable

    # read shapes and shared metadata from the first split
    with h5.File(path / split_files["C1"], "r") as f:
        ir_shape = f["data"]["impulse_response"].shape
        ir_dtype = f["data"]["impulse_response"].dtype
        src_dtype = f["data"]["location"]["source"].dtype
        n_split = ir_shape[0]
        sampling_rate = f["metadata"]["sampling_rate"][()]
        receiver = f["data"]["location"]["receiver"][()]
        has_humidity = "humidity" in f["metadata"]

    # calculate total number of sources and grid dimension
    n_sources = len(split_files) * n_split
    n_full_grid = int(np.sqrt(n_sources))
    n_split_grid = n_full_grid // 2

    # build virtual layouts
    ir_layout = h5.VirtualLayout(shape=(n_sources, *ir_shape[1:]), dtype=ir_dtype)
    src_layout = h5.VirtualLayout(shape=(n_sources, 3), dtype=src_dtype)
    c0_layout = h5.VirtualLayout(shape=(n_sources,), dtype="float32")
    temp_layout = h5.VirtualLayout(shape=(n_sources,), dtype="float32")
    hum_layout = h5.VirtualLayout(shape=(n_sources,), dtype="float32") if has_humidity else None

    # map each split to the correct location in the output layouts
    for split_name, (row, col) in offsets.items():
        fname = split_files[split_name]

        ir_vsrc = h5.VirtualSource(fname, "data/impulse_response", shape=ir_shape)
        src_vsrc = h5.VirtualSource(fname, "data/location/source", shape=(n_split, 3))
        c0_vsrc = h5.VirtualSource(fname, "metadata/c0", shape=(n_split,))
        temp_vsrc = h5.VirtualSource(fname, "metadata/temperature", shape=(n_split,))
        hum_vsrc = (
            h5.VirtualSource(fname, "metadata/humidity", shape=(n_split,))
            if has_humidity
            else None
        )

        for r in range(n_split_grid):
            # index one row of the split grid
            src = slice(r * n_split_grid, (r + 1) * n_split_grid)
            # map split-grid-row to full-grid-row
            grid_row = 2 * r + row
            # index one row of the full grid, skipping every other entry to interleave splits
            out = slice(grid_row * n_full_grid + col, grid_row * n_full_grid + n_full_grid, 2)

            ir_layout[out] = ir_vsrc[src]
            src_layout[out] = src_vsrc[src]
            c0_layout[out] = c0_vsrc[src]
            temp_layout[out] = temp_vsrc[src]
            if has_humidity:
                hum_layout[out] = hum_vsrc[src]

    with h5.File(output_path, "w") as out:
        # create groups and virtual datasets
        data_grp = out.create_group("data")
        data_grp.create_virtual_dataset("impulse_response", ir_layout)
        loc_grp = data_grp.create_group("location")
        loc_grp.create_virtual_dataset("source", src_layout)
        loc_grp.create_dataset("receiver", data=receiver)

        meta_grp = out.create_group("metadata")
        meta_grp.create_dataset("sampling_rate", data=sampling_rate)
        meta_grp.create_virtual_dataset("c0", c0_layout)
        meta_grp.create_virtual_dataset("temperature", temp_layout)
        if has_humidity:
            meta_grp.create_virtual_dataset("humidity", hum_layout)

    return output_path


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
    assert not (scenario[-1] == "D" and dataset_split is not None), "dense datasets do not have splits"

    path = Path(path) / "SRIRACHA" / "raw"
    doi = "10.14279/depositonce-23943"
    pup = pooch_from_doi(doi, path=path)

    # check if scenario is full-plane or dense and handle accordingly
    is_full_plane = scenario[-1] != "D"

    if is_full_plane and dataset_split is None:
        download_and_merge(scenario, path, pup)
        scenario += ".h5"
    else:
        if dataset_split is None:
            scenario += ".h5"
        else:
            scenario += "-" + dataset_split + ".h5"

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
