"""Datasets from the Department of Engineering Acoustics, TU Berlin, Berlin, Germany.

Currently this module hosts:

- MIRACLE: Microphone Array Impulse Response Dataset for Acoustic Learning.
- SRIRACHA: Shoebox Room Impulse Response Archive with Varying Absorption.
"""

import hashlib
from contextlib import ExitStack
from datetime import UTC, datetime
from math import isqrt, prod
from pathlib import Path
from typing import ClassVar

import h5py as h5
import netCDF4
import numpy as np

from irdl import config
from irdl._sofa import choose_ir_chunk_shape, write_ir_chunks
from irdl.base import BaseDataset, DatasetCategory
from irdl.downloader import _fetch, _pooch_from_doi
from irdl.logging import logger
from irdl.utils import _preserve_permissions


class IstaBaseDataset(BaseDataset):
    """Base class for HDF5-based datasets from ISTA (MIRACLE, SRIRACHA)."""

    def _source_filename(self, **dataset_kwargs) -> str:
        """Construct the selection name used for ingest and output paths."""
        scenario = dataset_kwargs["scenario"]
        split = dataset_kwargs.get("dataset_split")
        return f"{scenario}{('-' + split) if split else ''}.h5"

    def _download_direct_sofa(self, provider_dir: Path, url: str, known_hash: str | None) -> Path:
        """Use the DOI's registered filename for a filename-free SOFA URL."""
        provider_dir.mkdir(exist_ok=True, parents=True)
        manifest = _pooch_from_doi(self.doi, path=provider_dir)
        filenames = [name for name, target in manifest.urls.items() if target == url]
        if len(filenames) != 1:
            msg = f"No unique SOFA filename for {url!r} in {self.doi}"
            raise ValueError(msg)
        filename = filenames[0]
        if Path(filename).name != filename or Path(filename).suffix.lower() != ".sofa":
            msg = f"Direct SOFA download requires a .sofa basename: {filename!r}"
            raise ValueError(msg)
        checksum = manifest.registry.get(filename)
        if not known_hash or not checksum or checksum.lower() != known_hash.lower():
            msg = f"Missing or changed SOFA checksum for {filename!r} in {self.doi}"
            raise ValueError(msg)
        _fetch(manifest, filename)
        return provider_dir / filename

    def _ingest(self, ingest_path: Path, sofa_path: Path, **dataset_kwargs) -> Path:
        """Stream an ISTA HDF5 ingest-ready file to SOFA without loading all IRs."""
        if ingest_path.is_dir():
            msg = f"{type(self).__name__} does not support directory ingest artifacts"
            raise NotImplementedError(msg)

        batch_bytes = config._get_batch_bytes()
        sofa_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Streaming ISTA HDF5 {ingest_path} to SOFA {sofa_path}.")
        with h5.File(ingest_path, "r") as hdf5:
            ir = hdf5["data/impulse_response"]
            m, r, n = ir.shape
            if m == 0:
                msg = "Impulse response dataset is empty"
                raise ValueError(msg)
            ir_chunks = choose_ir_chunk_shape((m, r, n, 1), np.dtype("f8"))
            has_humidity = "humidity" in hdf5["metadata"]
            logger.info(f"Writing {m} measurements, {r} receivers, {n} samples in chunks {ir_chunks}.")
            with netCDF4.Dataset(sofa_path, "w", format="NETCDF4") as sofa:
                self._create_default_variables(
                    sofa,
                    m=m,
                    r=r,
                    n=n,
                    has_humidity=has_humidity,
                    receiver_position=np.asarray(hdf5["data/location/receiver"]),
                    sampling_rate=float(hdf5["metadata/sampling_rate"][()]),
                    ir_chunks=ir_chunks,
                )
                self._create_room_variables(sofa, **dataset_kwargs)
                source = sofa.variables["SourcePosition"]
                temperature = sofa.variables["RoomTemperature"]
                speed = sofa.variables["SpeedOfSound"]
                humidity = sofa.variables["Humidity"] if has_humidity else None
                hdf5_source = hdf5["data/location/source"]
                hdf5_temperature = hdf5["metadata/temperature"]
                hdf5_speed = hdf5["metadata/c0"]
                hdf5_humidity = hdf5["metadata/humidity"] if humidity is not None else None
                for row_slice in _chunk_slices(m, max(1, batch_bytes // (3 * 8))):
                    source[row_slice, :] = hdf5_source[row_slice]
                    temperature[row_slice] = hdf5_temperature[row_slice].astype(np.float64) + 273.15
                    speed[row_slice, 0] = hdf5_speed[row_slice]
                    if humidity is not None:
                        humidity[row_slice, 0] = hdf5_humidity[row_slice]
            # NetCDF4 must close before h5py writes raw compressed chunks.
            write_ir_chunks(sofa_path, _ista_ir_chunks(ir, ir_chunks, batch_bytes))

        _preserve_permissions(ingest_path, sofa_path)
        self._verify_payload(sofa_path, ingest_path, **dataset_kwargs)
        logger.info(f"Finished SOFA file {sofa_path}.")
        return sofa_path

    def _verify_payload(self, sofa_path: Path, ingest_artifact: Path, **_dataset_kwargs) -> None:
        """Verify that a streamed ISTA SOFA matches its HDF5 ingest artifact."""
        batch_bytes = config._get_batch_bytes()
        logger.info(f"Validating SOFA file {sofa_path}.")
        with (
            logger.spin(f"Running data checksum on {sofa_path.name}..."),
            netCDF4.Dataset(sofa_path) as sofa_dataset,
            h5.File(ingest_artifact, "r") as hdf5,
        ):
            comparisons = (
                (
                    "Data.IR",
                    hdf5["data/impulse_response"],
                    sofa_dataset.variables["Data.IR"],
                    np.dtype(np.float32),
                ),
                (
                    "SourcePosition",
                    hdf5["data/location/source"],
                    sofa_dataset.variables["SourcePosition"],
                    np.dtype(np.float64),
                ),
                (
                    "ReceiverPosition",
                    hdf5["data/location/receiver"],
                    sofa_dataset.variables["ReceiverPosition"],
                    np.dtype(np.float64),
                ),
            )
            for variable_name, hdf5_variable, sofa_variable, dtype in comparisons:
                hdf5_shape = _canonical_shape(hdf5_variable.shape)
                sofa_shape = _canonical_shape(sofa_variable.shape)
                if hdf5_shape != sofa_shape:
                    msg = (
                        f"SOFA checksum validation failed for {sofa_path}: "
                        f"HDF5 shape {hdf5_shape} differs from SOFA shape {sofa_shape}: {variable_name}"
                    )
                    raise ValueError(msg)

                hdf5_hash = hashlib.sha256()
                sofa_hash = hashlib.sha256()
                selections = (
                    _ir_check_slices(hdf5_variable.shape, batch_bytes)
                    if variable_name == "Data.IR"
                    else (
                        (rows,)
                        for rows in _chunk_slices(
                            hdf5_variable.shape[0], max(1, batch_bytes // (prod(hdf5_variable.shape[1:]) * 8))
                        )
                    )
                )
                for selection in selections:
                    hdf5_data = hdf5_variable[selection]
                    if len(sofa_variable.shape) == 4 and sofa_variable.shape[-1] == 1:  # noqa: PLR2004 - FIR-E.
                        sofa_data = sofa_variable[(*selection, 0)]
                    else:
                        sofa_data = sofa_variable[selection]
                    hdf5_hash.update(_canonical_array(hdf5_data, dtype).view(np.uint8))
                    sofa_hash.update(_canonical_array(sofa_data, dtype).view(np.uint8))
                if hdf5_hash.hexdigest() != sofa_hash.hexdigest():
                    msg = (
                        f"SOFA checksum validation failed for {sofa_path}: "
                        f"checksum differs from ISTA HDF5 ingest data: {variable_name}"
                    )
                    raise ValueError(msg)

    def _create_default_variables(  # noqa: PLR0915
        self,
        sofa: netCDF4.Dataset,
        *,
        m: int,
        r: int,
        n: int,
        has_humidity: bool,
        receiver_position: np.ndarray,
        sampling_rate: float,
        ir_chunks: tuple[int, int, int, int],
    ) -> None:
        """Create the common ISTA SOFA skeleton and fill shared defaults."""
        for name, size in {"M": m, "R": r, "N": n, "E": 1, "C": 3, "I": 1}.items():
            sofa.createDimension(name, size)

        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        sofa.Conventions = "SOFA"
        sofa.Version = "2.1"
        sofa.SOFAConventions = "SingleRoomMIMOSRIR"
        sofa.SOFAConventionsVersion = "1.0"
        sofa.DataType = "FIR-E"
        sofa.Title = self.name.upper()
        sofa.DatabaseName = self.name.upper()
        sofa.References = self.doi
        sofa.License = "CC BY-NC-SA 4.0"
        sofa.DateCreated = now
        sofa.DateModified = now
        sofa.AuthorContact = "a.pelling@tu-berlin.de"
        sofa.Organization = "TU Berlin"
        sofa.APIName = "IRDL"
        sofa.APIVersion = "1.0"
        sofa.RoomLocation = "TU Berlin, Einsteinufer 25, 10587 Berlin"
        sofa.ListenerShortName = "Custom planar microphone array"
        sofa.ListenerDescription = (
            "64-channel planar microphone array "
            "(1.5 m x 1.5 m aluminium plate, Vogel's spiral, max spacing 1.47 m, 51.2 kHz sampling rate)"
        )
        sofa.ReceiverShortName = "GRAS 40PL-1 Short CCP"
        sofa.ReceiverDescription = "GRAS 40PL-1 Short CCP"
        sofa.SourceShortName = "Loudspeaker"
        sofa.SourceDescription = 'Dynamic 2" cone loudspeaker in a cylindrical enclosure (100 Hz-16 kHz)'

        sofa.createVariable("Data.IR", "f8", ("M", "R", "N", "E"), zlib=True, complevel=4, chunksizes=ir_chunks)
        source = sofa.createVariable("SourcePosition", "f8", ("M", "C"))
        source_view = sofa.createVariable("SourceView", "f8", ("M", "C"))
        source_up = sofa.createVariable("SourceUp", "f8", ("M", "C"))
        receiver = sofa.createVariable("ReceiverPosition", "f8", ("R", "C", "I"))
        receiver_view = sofa.createVariable("ReceiverView", "f8", ("R", "C", "I"))
        receiver_up = sofa.createVariable("ReceiverUp", "f8", ("R", "C", "I"))
        temperature = sofa.createVariable("RoomTemperature", "f8", ("M",))
        sampling_rate_var = sofa.createVariable("Data.SamplingRate", "f8", ("I",))
        delay = sofa.createVariable("Data.Delay", "f8", ("M", "R", "I"))
        listener = sofa.createVariable("ListenerPosition", "f8", ("M", "C"))
        listener_view = sofa.createVariable("ListenerView", "f8", ("M", "C"))
        listener_up = sofa.createVariable("ListenerUp", "f8", ("M", "C"))
        emitter = sofa.createVariable("EmitterPosition", "f8", ("E", "C", "I"))
        measurement_date = sofa.createVariable("MeasurementDate", "f8", ("M",))
        speed = sofa.createVariable("SpeedOfSound", "f8", ("M", "I"))
        humidity = sofa.createVariable("Humidity", "f8", ("M", "I")) if has_humidity else None

        for variable in (
            source,
            source_view,
            receiver,
            receiver_view,
            listener,
            listener_view,
            emitter,
        ):
            variable.Type = "cartesian"
            variable.Units = "metre"
        temperature.Units = "kelvin"
        sampling_rate_var.Units = "hertz"

        receiver[:] = np.asarray(receiver_position)[:, :, np.newaxis]
        receiver_view[:] = np.tile((1.0, 0.0, 0.0), (r, 1))[:, :, np.newaxis]
        receiver_up[:] = np.tile((0.0, 0.0, 1.0), (r, 1))[:, :, np.newaxis]
        sampling_rate_var[:] = sampling_rate
        delay[:] = 0.0
        listener[:] = 0.0
        listener_view[:] = (0.0, 0.0, 1.0)
        listener_up[:] = (0.0, 1.0, 0.0)
        source_view[:] = (0.0, 0.0, -1.0)
        source_up[:] = (0.0, 1.0, 0.0)
        emitter[:] = 0.0
        measurement_date[:] = self.measurement_date
        speed[:] = 0.0
        if humidity is not None:
            humidity[:] = 0.0


def _ista_ir_chunks(ir: h5.Dataset, chunk_shape: tuple[int, int, int, int], batch_bytes: int):
    """Read bounded ISTA batches and yield complete SOFA chunks in stored order."""
    m, r, n = ir.shape
    row_bytes = r * n * max(ir.dtype.itemsize, np.dtype("f8").itemsize)
    batch_rows = max(1, batch_bytes // row_bytes)
    for rows in _chunk_slices(m, batch_rows):
        batch = ir[rows] if row_bytes <= batch_bytes else None
        for row in range(rows.start, rows.stop):
            for receivers in _chunk_slices(r, chunk_shape[1]):
                for samples in _chunk_slices(n, chunk_shape[2]):
                    if batch is None:
                        data = ir[row : row + 1, receivers, samples]
                    else:
                        local_row = row - rows.start
                        data = batch[local_row : local_row + 1, receivers, samples]
                    yield (row, receivers.start, samples.start, 0), data[..., np.newaxis]


def _ir_check_slices(shape: tuple[int, int, int], batch_bytes: int):
    """Bound checksum reads while independently mapping stored positions to the source."""
    m, r, n = shape
    row_bytes = r * n * np.dtype("f8").itemsize
    if row_bytes <= batch_bytes:
        batch_rows = max(1, batch_bytes // row_bytes)
        for rows in _chunk_slices(m, batch_rows):
            yield rows, slice(None), slice(None)
    else:
        chunks = choose_ir_chunk_shape((*shape, 1), np.dtype("f8"))
        for row in range(m):
            for receivers in _chunk_slices(r, chunks[1]):
                for samples in _chunk_slices(n, chunks[2]):
                    yield slice(row, row + 1), receivers, samples


def _sriracha_grid_size(handles: dict[str, h5.File]) -> int:
    """Require four matching, nonempty square Split grids before creating output."""
    shape = handles["C1"]["data/impulse_response"].shape
    side = isqrt(shape[0]) if len(shape) == 3 else 0  # noqa: PLR2004 - Provider IR axes: M, R, N.
    if len(shape) != 3 or not all(shape) or side**2 != shape[0]:  # noqa: PLR2004
        msg = "SRIRACHA Splits must contain a nonempty square grid of IRs"
        raise ValueError(msg)
    if any(handle["data/impulse_response"].shape != shape for handle in handles.values()):
        msg = "SRIRACHA Split IR shapes must match"
        raise ValueError(msg)
    if any(handle["data/location/source"].shape != (shape[0], 3) for handle in handles.values()):
        msg = "SRIRACHA Split Source Coordinates must contain one xyz row per IR"
        raise ValueError(msg)
    return side


def _split_row_slices(side: int, pair_rows: int):
    """Map contiguous Split batches to successive full-plane measurement pairs."""
    full_side = 2 * side
    for grid_row in range(full_side):
        left, right = (("C1", "C2"), ("C3", "C4"))[grid_row % 2]
        for columns in _chunk_slices(side, pair_rows):
            start = (grid_row // 2) * side
            src = slice(start + columns.start, start + columns.stop)
            start = grid_row * full_side
            dst = slice(start + 2 * columns.start, start + 2 * columns.stop)
            yield src, dst, left, right


def _interleave_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Combine adjacent Split columns into contiguous full-plane rows."""
    result = np.empty((2 * len(left), *left.shape[1:]), dtype=np.result_type(left, right))
    result[::2], result[1::2] = left, right
    return result


def _sriracha_ir_chunks(
    handles: dict[str, h5.File], side: int, chunk_shape: tuple[int, int, int, int], batch_bytes: int
):
    """Read bounded pairs of Split batches and yield ordered complete SOFA chunks."""
    r, n = handles["C1"]["data/impulse_response"].shape[1:]
    itemsize = max(
        np.dtype("f8").itemsize, *(handle["data/impulse_response"].dtype.itemsize for handle in handles.values())
    )
    pair_bytes = 2 * r * n * itemsize
    pair_rows = max(1, batch_bytes // pair_bytes)
    for src, dst, left, right in _split_row_slices(side, pair_rows):
        inputs = (handles[left]["data/impulse_response"], handles[right]["data/impulse_response"])
        batches = tuple(ir[src] if pair_bytes <= batch_bytes else None for ir in inputs)
        for row in range(src.stop - src.start):
            for column, (ir, batch) in enumerate(zip(inputs, batches, strict=True)):
                for receivers in _chunk_slices(r, chunk_shape[1]):
                    for samples in _chunk_slices(n, chunk_shape[2]):
                        if batch is None:
                            source_row = src.start + row
                            data = ir[source_row : source_row + 1, receivers, samples]
                        else:
                            data = batch[row : row + 1, receivers, samples]
                        offset = (dst.start + 2 * row + column, receivers.start, samples.start, 0)
                        yield offset, data[..., np.newaxis]


def _check_sriracha_values(
    sofa_path: Path, name: str, expected: np.ndarray, actual: np.ndarray, dtype: np.dtype
) -> None:
    """Keep the existing SRIRACHA checksum precision and error reporting."""
    expected_hash = hashlib.sha256(_canonical_array(expected, dtype).view(np.uint8)).digest()
    actual_hash = hashlib.sha256(_canonical_array(actual, dtype).view(np.uint8)).digest()
    if expected_hash != actual_hash:
        msg = f"SOFA checksum validation failed for {sofa_path}: checksum differs from SRIRACHA split data: {name}"
        raise ValueError(msg)


def _canonical_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    squeezed = tuple(dim for dim in shape if dim != 1)
    return squeezed or (1,)


def _chunk_slices(length: int, chunk_size: int):
    for start in range(0, length, chunk_size):
        yield slice(start, min(start + chunk_size, length))


def _canonical_array(data: np.ndarray, dtype: np.dtype) -> np.ndarray:
    array = np.squeeze(np.asarray(data))
    if array.ndim == 0:
        array = array.reshape(1)
    return np.ascontiguousarray(array.astype(dtype, copy=False))


class MiracleDataset(IstaBaseDataset):
    """Download the MIRACLE database and native SOFAs from DepositOnce.

    Attributes
    ----------
    name : str
        Dataset name ("miracle").
    doi : str
        Digital Object Identifier ("10.14279/depositonce-26877").
    room_volume : float
        Room volume in cubic meters (830).
    measurement_date : float
        Release date in POSIX seconds, used as the SOFA MeasurementDate
        (no per-measurement date is available).
    """

    name = "miracle"
    doi = "10.14279/depositonce-26877"
    _category = DatasetCategory.ROOM_IMPULSE_RESPONSES
    # metadata needed for creation of sofa file
    room_volume = 830
    room_type = "shoebox"
    measurement_date = 1697068800.0

    def _direct_sofa(self, source_filename: str) -> tuple[str, str] | tuple[None, None]:
        """Find a matching native SOFA through DepositOnce, or fall back to HDF5."""
        filename = Path(source_filename).with_suffix(".sofa").name
        manifest = _pooch_from_doi(self.doi)
        url = manifest.urls.get(filename)
        if not url:
            return None, None
        checksum = manifest.registry.get(filename)
        if not checksum:
            msg = f"Missing native SOFA checksum for {filename!r} in {self.doi}"
            raise ValueError(msg)
        return url, checksum

    def _create_room_variables(self, sofa: netCDF4.Dataset, **dataset_kwargs) -> None:
        """Create MIRACLE's shoebox geometry in listener coordinates."""
        array_z = {"A1": 7.5355, "A2": 6.8, "D1": 7.531, "R2": 6.8}[dataset_kwargs["scenario"]]
        sofa.RoomType = self.room_type
        room_volume = sofa.createVariable("RoomVolume", "f8", ("I",))
        room_corner_a = sofa.createVariable("RoomCornerA", "f8", ("I", "C"))
        room_corner_b = sofa.createVariable("RoomCornerB", "f8", ("I", "C"))
        room_corners = sofa.createVariable("RoomCorners", "f8", ("I", "I"))
        room_volume.Units = "cubic metre"
        room_corners.Type = "cartesian"
        room_corners.Units = "metre"
        room_volume[:] = self.room_volume
        room_corner_a[:] = (-4.15, -3.0, -array_z)
        room_corner_b[:] = (4.15, 4.4, 13.5 - array_z)
        room_corners[:] = 0.0

    @classmethod
    def get(
        cls,
        scenario: str = "A1",
        dataset_split: str | None = None,
        cache_dir: str | Path | None = None,
        export_dir: str | Path | None = None,
        output_format: str = "pyfar",
    ) -> dict | Path | None:
        """
        scenario : str
            Scenario to download. One of 'A1', 'A2', 'D1', 'R2'.
        dataset_split : str or None, optional
            Artificial dataset split. One of 'C1', 'C2', 'C3', 'C4' or None.
            Dense scenarios (D1) cannot be split.

        Returns
        -------
        dict or Path
            For 'pyfar' / 'numpy': dict of in-memory objects.
            For 'sofa' / 'hdf5' / 'raw': Path to file on disk.
        """  # noqa: D205, D403
        return cls()._get(
            scenario=scenario,
            dataset_split=dataset_split,
            cache_dir=cache_dir,
            export_dir=export_dir,
            output_format=output_format,
        )

    def _validate_params(self, **dataset_kwargs) -> None:
        """Validate MIRACLE-specific parameters.

        Parameters
        ----------
        **dataset_kwargs : dict
            Must contain 'scenario' (one of 'A1', 'A2', 'D1', 'R2'). May
            contain 'dataset_split' (one of 'C1', 'C2', 'C3', 'C4', or None).
            Scenario 'D1' cannot be split. ``output_format`` is also passed
            but unused here.

        Raises
        ------
        ValueError
            If scenario or split is out of range, or 'D1' is combined with a split.
        """
        scenario = dataset_kwargs["scenario"]
        dataset_split = dataset_kwargs.get("dataset_split")

        if scenario not in ["A1", "A2", "D1", "R2"]:
            msg = "scenario must be one of ['A1', 'A2', 'D1', 'R2']"
            raise ValueError(msg)
        if dataset_split not in [None, "C1", "C2", "C3", "C4"]:
            msg = "dataset_split must be None or one ['C1', 'C2', 'C3', 'C4']"
            raise ValueError(msg)
        if scenario == "D1" and dataset_split is not None:
            msg = "scenario D1 cannot be split"
            raise ValueError(msg)

    def _download(self, provider_dir: Path, **dataset_kwargs) -> Path:
        """Download MIRACLE dataset file.

        Downloads the full scenario HDF5 file. If a split is requested,
        the split will be extracted in _process().

        Parameters
        ----------
        provider_dir : :class:`pathlib.Path`
            Provider directory (e.g., ``cache/MIRACLE/provider/``).
        **dataset_kwargs : dict
            Must contain 'scenario'. May contain 'dataset_split'.

        Returns
        -------
        :class:`pathlib.Path`
            Path to the downloaded full scenario HDF5 file inside the
            provider directory.
        """
        full_path = provider_dir / self._source_filename(**{**dataset_kwargs, "dataset_split": None})
        logger.info(f"Downloading MIRACLE scenario {dataset_kwargs['scenario']}")
        pup = _pooch_from_doi(self.doi, path=provider_dir)
        _fetch(pup, full_path.name)
        return full_path

    def _process(self, provider_artifact: Path, ingest_path: Path, **dataset_kwargs) -> Path:
        """Post-process MIRACLE file if needed.

        If a dataset_split is requested and the file is the full scenario file,
        extracts the corresponding quadrant split into the ingest directory.
        Otherwise promotes the provider file to the ingest stage.

        Parameters
        ----------
        provider_artifact : :class:`pathlib.Path`
            Path to the provider file (full scenario HDF5).
        ingest_path : :class:`pathlib.Path`
            Path to the HDF5 file in the ingest directory.
        **dataset_kwargs : dict
            Must contain 'scenario'. May contain 'dataset_split'.

        Returns
        -------
        :class:`pathlib.Path`
            Path to the processed file in the ingest directory.
        """
        split = dataset_kwargs.get("dataset_split")

        # If no split requested, promote to ingest stage
        if not split:
            return super()._process(provider_artifact, ingest_path, **dataset_kwargs)
        return self._extract_split(provider_artifact, split, ingest_path)

    def _extract_split(self, ingest_path: Path, dataset_split: str, output_path: Path) -> Path:
        """Extract a dataset split from a full MIRACLE HDF5 file.

        Reads the full file, indexes the requested quadrant of the source
        grid, and writes the result to a new HDF5 file.

        Parameters
        ----------
        ingest_path : :class:`pathlib.Path`
            Path to the full HDF5 file in the provider directory.
        dataset_split : str
            Split to extract. One of 'C1', 'C2', 'C3', 'C4'.
        output_path : :class:`pathlib.Path`
            Target path in the ingest directory.

        Returns
        -------
        :class:`pathlib.Path`
            Path to the extracted split HDF5 file.
        """
        logger.info(f"Extracting split {dataset_split} from {ingest_path.name}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Load full data from HDF5
        with h5.File(ingest_path, "r") as f:
            data = {
                "impulse_response": f["data"]["impulse_response"][()],
                "receiver_coordinates": f["data"]["location"]["receiver"][()],
                "source_coordinates": f["data"]["location"]["source"][()],
                "speed_of_sound": f["metadata"]["c0"][()],
                "temperature": f["metadata"]["temperature"][()],
                "sampling_rate": f["metadata"]["sampling_rate"][()],
            }
            if "humidity" in f["metadata"]:
                data["humidity"] = f["metadata"]["humidity"][()]

        # Split to the requested quadrant
        offsets = {"C1": (0, 0), "C2": (0, 1), "C3": (1, 0), "C4": (1, 1)}
        row, column = offsets[dataset_split]
        n = int(np.sqrt(data["source_coordinates"].shape[0]))
        ir_shape = data["impulse_response"].shape

        data["source_coordinates"] = data["source_coordinates"].reshape(n, n, 3)[row::2, column::2, :].reshape(-1, 3)
        data["impulse_response"] = (
            data["impulse_response"].reshape(n, n, *ir_shape[1:])[row::2, column::2, :].reshape(-1, *ir_shape[1:])
        )
        data["temperature"] = data["temperature"].reshape(n, n)[row::2, column::2].reshape(-1)
        data["speed_of_sound"] = data["speed_of_sound"].reshape(n, n)[row::2, column::2].reshape(-1)
        if "humidity" in data:
            data["humidity"] = data["humidity"].reshape(n, n)[row::2, column::2].reshape(-1)

        with h5.File(output_path, "w") as f:
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

        _preserve_permissions(ingest_path, output_path)
        return output_path


class SrirachaDataset(IstaBaseDataset):
    """Download and merge the SRIRACHA database from DepositOnce.

    Attributes
    ----------
    name : str
        Dataset name ("sriracha").
    doi : str
        Digital Object Identifier ("10.14279/depositonce-23943").
    room_volume : float
        Room volume in cubic meters (73.5).
    measurement_date : float
        Release date in POSIX seconds, used as the SOFA MeasurementDate
        (no per-measurement date is available).
    """

    name = "sriracha"
    doi = "10.14279/depositonce-23943"
    _category = DatasetCategory.ROOM_IMPULSE_RESPONSES
    _split_offsets = (
        ("C1", (0, 0)),
        ("C2", (0, 1)),
        ("C3", (1, 0)),
        ("C4", (1, 1)),
    )
    _room_geometry_bitstreams: ClassVar[dict[str, str]] = {
        "SR1": "230377c8-f7bd-4e70-abad-f824464d2860",
        "SR1-D": "72d8ea5b-959e-411d-b185-61d16d5611ce",
        "SR1-C1": "3bb17656-a907-4ff0-b682-00e988f3358d",
        "SR1-C2": "d793a95a-cda7-40c2-aa93-1fda9ed01fc7",
        "SR1-C3": "ebaf10d1-8eba-47ae-a024-100e4865ca37",
        "SR1-C4": "cf87ca0d-a605-483d-9b61-cbca9efb3668",
        "SR2": "16dfc746-d468-42b3-94d7-bb8b2f62c485",
        "SR2-D": "16928a00-3307-4aa8-9b10-415292f41b8c",
        "SR2-C1": "4a87658c-0631-4630-8e20-ba5d8f95cd4a",
        "SR2-C2": "da80797a-59e6-426e-b307-5ba05f6e298d",
        "SR2-C3": "b71a3b68-2225-4a3d-b990-78a53f032ad0",
        "SR2-C4": "65b3fe7f-7325-4cee-8e22-e227b95edc16",
        "SRA1": "fd1ab25c-46ad-43f1-a9c2-0abb30a6a8a5",
        "SRA1-D": "6961647d-b9e9-460f-bcd0-c23610554cad",
        "SRA1-C1": "8bfa27bc-326e-42b5-aff4-9597bb7c6123",
        "SRA1-C2": "ae7c5065-2c46-4ea8-95e4-1ecde6bb7c34",
        "SRA1-C3": "e789ac85-ddb8-4aa6-b133-e9b2b3e00f4f",
        "SRA1-C4": "713e73c9-6017-4492-8b34-83562d34b9f4",
        "SRA2": "40f7adf8-7ba8-452a-8d24-73085d2cbbfa",
        "SRA2-D": "0e76d7a6-6f3b-4fea-b9bb-145518be6f5d",
        "SRA2-C1": "4219d3a5-b683-406c-948c-2ac00d0bcc03",
        "SRA2-C2": "5887c8f5-b60c-4fc7-b7da-b2fed88c2223",
        "SRA2-C3": "c741ccee-b2c3-4b91-be7b-6aed048566b2",
        "SRA2-C4": "697cc94a-52d5-4f33-b8e8-f01d69a23882",
    }
    room_volume = 73.5
    room_type = "dae"
    measurement_date = 1755648000.0

    def _create_room_variables(self, sofa: netCDF4.Dataset, **dataset_kwargs) -> None:
        """Set the matching selection's DAE without transforming Coordinates."""
        selection = self._source_filename(**dataset_kwargs).removesuffix(".h5")
        bitstream = self._room_geometry_bitstreams[selection]
        sofa.RoomType = self.room_type
        sofa.RoomGeometry = f"https://depositonce.tu-berlin.de/bitstreams/{bitstream}/download"
        room_volume = sofa.createVariable("RoomVolume", "f8", ("I",))
        room_volume.Units = "cubic metre"
        room_volume[:] = self.room_volume

    @classmethod
    def get(
        cls,
        scenario: str = "SR1-D",
        dataset_split: str | None = None,
        cache_dir: str | Path | None = None,
        export_dir: str | Path | None = None,
        output_format: str = "pyfar",
    ) -> dict | Path | None:
        """
        scenario : str, optional
            Scenario to download. One of 'SR1', 'SRA1', 'SR1-D', 'SRA1-D',
            'SR2', 'SRA2', 'SR2-D', 'SRA2-D'. Default is 'SR1-D'.
        dataset_split : str or None, optional
            Optional dataset split for full-plane scenarios. One of 'C1',
            'C2', 'C3', 'C4' or None. Dense scenarios (ending in '-D') do not
            have splits. Default is None.

        Returns
        -------
        dict or Path
            For 'pyfar' / 'numpy': dict of in-memory objects.
            For 'sofa' / 'hdf5' / 'raw': Path to file on disk.
        """  # noqa: D205, D403
        return cls()._get(
            scenario=scenario,
            dataset_split=dataset_split,
            cache_dir=cache_dir,
            export_dir=export_dir,
            output_format=output_format,
        )

    def _validate_params(self, **dataset_kwargs) -> None:
        """Validate SRIRACHA-specific parameters.

        Parameters
        ----------
        **dataset_kwargs : dict
            Must contain 'scenario' (one of 'SR1', 'SRA1', 'SR1-D', 'SRA1-D',
            'SR2', 'SRA2', 'SR2-D', 'SRA2-D'). May contain 'dataset_split'
            (one of 'C1', 'C2', 'C3', 'C4', or None). Dense scenarios
            (ending in '-D') cannot be split. ``output_format`` is also
            passed and used to forbid 'raw' for non-dense full-plane
            scenarios.

        Raises
        ------
        ValueError
            If scenario or split is invalid, a dense scenario is combined with
            a split, or 'raw' is requested for a non-dense full plane.
        """
        scenario = dataset_kwargs.get("scenario")
        dataset_split = dataset_kwargs.get("dataset_split")
        output_format = dataset_kwargs.get("output_format")

        if scenario not in ["SR1", "SRA1", "SR1-D", "SRA1-D", "SR2", "SRA2", "SR2-D", "SRA2-D"]:
            msg = "scenario must be one of [SR1, SRA1, SR1-D, SRA1-D, SR2, SRA2, SR2-D, SRA2-D]"
            raise ValueError(msg)
        if dataset_split not in [None, "C1", "C2", "C3", "C4"]:
            msg = "dataset_split must be None or in [C1, C2, C3, C4]"
            raise ValueError(msg)
        if scenario[-1] == "D" and dataset_split is not None:
            msg = "dense datasets do not have splits"
            raise ValueError(msg)
        if output_format == "raw" and scenario and scenario[-1] != "D" and dataset_split is None:
            msg = "raw output_format not supported for non-dense SRIRACHA scenarios without split"
            raise ValueError(msg)

    def _download(self, provider_dir: Path, **dataset_kwargs) -> Path:
        """Download SRIRACHA dataset file(s) to the provider directory.

        For dense scenarios or explicit splits, downloads a single file.
        For non-dense full-plane scenarios, downloads all 4 split files
        and returns the provider directory path.

        Parameters
        ----------
        provider_dir : :class:`pathlib.Path`
            Provider directory (e.g., ``cache/SRIRACHA/provider/``).
        **dataset_kwargs : dict
            Must contain 'scenario'. May contain 'dataset_split'.

        Returns
        -------
        :class:`pathlib.Path`
            Path to the downloaded file (inside provider) or the provider
            directory (for non-dense full-plane scenarios).
        """
        scenario = dataset_kwargs["scenario"]
        split = dataset_kwargs.get("dataset_split")

        # Dense scenario or explicit split -> single-file download
        if scenario.endswith("D") or split is not None:
            fname = self._source_filename(**dataset_kwargs)
            target_file = provider_dir / fname
            logger.info(f"Downloading SRIRACHA scenario {scenario}")
            pup = _pooch_from_doi(self.doi, path=provider_dir)
            _fetch(pup, fname)
            return target_file
        # Non-dense full plane -> the four Split files are streamed directly during Ingest.
        logger.info(f"Downloading SRIRACHA scenario {scenario} (4 split files)")
        pup = _pooch_from_doi(self.doi, path=provider_dir)
        for split_file in ["C1", "C2", "C3", "C4"]:
            fname = f"{scenario}-{split_file}.h5"
            _fetch(pup, fname)
        return provider_dir

    def _process(self, provider_artifact: Path, ingest_path: Path, **dataset_kwargs) -> Path:
        """Post-process SRIRACHA file if needed.

        Non-dense full-plane scenarios keep the four Provider Split files as the
        Ingest-Ready Artifact, without materializing a merged HDF5 file.
        Otherwise promotes the single file to the ingest stage.

        Parameters
        ----------
        provider_artifact : Path
            Path to the downloaded file or the provider directory.
        ingest_path : :class:`pathlib.Path`
            Path to the HDF5 file in the ingest directory.
        **dataset_kwargs : dict
            Must contain 'scenario'. May contain 'dataset_split'.

        Returns
        -------
        Path
            Path to the Ingest-Ready Artifact: the Provider directory for a full
            plane, or the promoted file for a dense Scenario or explicit Split.
        """
        scenario = dataset_kwargs["scenario"]
        split = dataset_kwargs.get("dataset_split")

        # Dense scenarios and explicit splits don't need merging
        if scenario.endswith("D") or split is not None:
            return super()._process(provider_artifact, ingest_path, **dataset_kwargs)
        # Non-dense full plane: provider split files are the ingest-ready artifact set.
        return provider_artifact

    def _ingest(self, ingest_path: Path, sofa_path: Path, **dataset_kwargs) -> Path:
        scenario = dataset_kwargs["scenario"]
        split = dataset_kwargs.get("dataset_split")
        if scenario.endswith("D") or split is not None:
            return super()._ingest(ingest_path, sofa_path, **dataset_kwargs)

        batch_bytes = config._get_batch_bytes()
        provider_dir = ingest_path
        sofa_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Streaming SRIRACHA split files for {scenario} to SOFA {sofa_path}.")
        with ExitStack() as stack:
            handles = {
                name: stack.enter_context(h5.File(provider_dir / f"{scenario}-{name}.h5", "r"))
                for name, _ in self._split_offsets
            }
            side = _sriracha_grid_size(handles)
            first = handles["C1"]
            n_split, r, n = first["data/impulse_response"].shape
            m = len(handles) * n_split
            has_humidity = "humidity" in first["metadata"]
            ir_chunks = choose_ir_chunk_shape((m, r, n, 1), np.dtype("f8"))
            logger.info(f"Writing {m} measurements, {r} receivers, {n} samples in chunks {ir_chunks}.")
            with netCDF4.Dataset(sofa_path, "w", format="NETCDF4") as sofa:
                self._create_default_variables(
                    sofa,
                    m=m,
                    r=r,
                    n=n,
                    has_humidity=has_humidity,
                    receiver_position=np.asarray(first["data/location/receiver"]),
                    sampling_rate=float(first["metadata/sampling_rate"][()]),
                    ir_chunks=ir_chunks,
                )
                self._create_room_variables(sofa, **dataset_kwargs)
                source = sofa.variables["SourcePosition"]
                temperature = sofa.variables["RoomTemperature"]
                speed = sofa.variables["SpeedOfSound"]
                humidity = sofa.variables["Humidity"] if has_humidity else None
                for src, dst, left_name, right_name in _split_row_slices(side, max(1, batch_bytes // (2 * 3 * 8))):
                    left, right = handles[left_name], handles[right_name]
                    source[dst, :] = _interleave_rows(
                        left["data/location/source"][src], right["data/location/source"][src]
                    )
                    temperature[dst] = (
                        _interleave_rows(left["metadata/temperature"][src], right["metadata/temperature"][src]).astype(
                            np.float64
                        )
                        + 273.15
                    )
                    speed[dst, 0] = _interleave_rows(left["metadata/c0"][src], right["metadata/c0"][src])
                    if humidity is not None:
                        humidity[dst, 0] = _interleave_rows(
                            left["metadata/humidity"][src], right["metadata/humidity"][src]
                        )
            write_ir_chunks(sofa_path, _sriracha_ir_chunks(handles, side, ir_chunks, batch_bytes))

        _preserve_permissions(ingest_path, sofa_path)
        self._verify_payload(sofa_path, provider_dir, scenario=scenario)
        logger.info(f"Finished SOFA file {sofa_path}.")
        return sofa_path

    def _verify_payload(self, sofa_path: Path, ingest_artifact: Path, **dataset_kwargs) -> None:
        if not ingest_artifact.is_dir():
            super()._verify_payload(sofa_path, ingest_artifact, **dataset_kwargs)
            return

        batch_bytes = config._get_batch_bytes()
        scenario = dataset_kwargs["scenario"]
        logger.info(f"Validating SOFA file {sofa_path}.")
        with (
            logger.spin(f"Running data checksum on {sofa_path.name}..."),
            netCDF4.Dataset(sofa_path) as sofa,
            ExitStack() as stack,
        ):
            handles = {
                name: stack.enter_context(h5.File(ingest_artifact / f"{scenario}-{name}.h5", "r"))
                for name, _ in self._split_offsets
            }
            side = _sriracha_grid_size(handles)
            n_split, r, n = handles["C1"]["data/impulse_response"].shape
            shape = (len(handles) * n_split, r, n, 1)
            if sofa.variables["Data.IR"].shape != shape:
                msg = f"SOFA checksum validation failed for {sofa_path}: Data.IR shape must be {shape}"
                raise ValueError(msg)
            pair_bytes = 2 * r * n * np.dtype("f8").itemsize
            pair_rows = max(1, min(batch_bytes // pair_bytes, batch_bytes // (2 * 3 * 8)))
            chunks = (
                shape
                if pair_bytes <= batch_bytes
                else choose_ir_chunk_shape(shape, np.dtype("f8"), max_bytes=max(8, batch_bytes // 2))
            )
            for src, dst, left_name, right_name in _split_row_slices(side, pair_rows):
                left, right = handles[left_name], handles[right_name]
                for receivers in _chunk_slices(r, chunks[1]):
                    for samples in _chunk_slices(n, chunks[2]):
                        selection = (src, receivers, samples)
                        if batch_bytes < 2 * np.dtype("f8").itemsize:
                            # A minimal budget cannot hold a pair; compare each Split separately.
                            for column, handle in enumerate((left, right)):
                                _check_sriracha_values(
                                    sofa_path,
                                    "Data.IR",
                                    handle["data/impulse_response"][selection],
                                    sofa.variables["Data.IR"][dst.start + column : dst.stop : 2, receivers, samples, 0],
                                    np.dtype("f4"),
                                )
                            continue
                        expected = _interleave_rows(
                            left["data/impulse_response"][selection], right["data/impulse_response"][selection]
                        )
                        actual = sofa.variables["Data.IR"][dst, receivers, samples, 0]
                        _check_sriracha_values(sofa_path, "Data.IR", expected, actual, np.dtype("f4"))
                expected = _interleave_rows(left["data/location/source"][src], right["data/location/source"][src])
                _check_sriracha_values(
                    sofa_path, "SourcePosition", expected, sofa.variables["SourcePosition"][dst], np.dtype("f8")
                )
