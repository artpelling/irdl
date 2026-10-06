"""Datasets from the Department of Engineering Acoustics, TU Berlin, Berlin, Germany.

Currently this module hosts:

- MIRACLE: Microphone Array Impulse Response Dataset for Acoustic Learning.
- SRIRACHA: Shoebox Room Impulse Response Archive with Varying Absorption.
"""

import hashlib
import os
import zlib
from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py as h5
import netCDF4
import numpy as np
import psutil

from irdl.base import BaseDataset, DatasetCategory
from irdl.downloader import _fetch, _pooch_from_doi
from irdl.logging import logger
from irdl.sonicom import SonicomBaseDataset
from irdl.utils import _preserve_permissions

_SOFA_FIR_E_DIMS = 4
_PARALLEL_CHECKSUM_MIN_BYTES = 1024**3


class IstaBaseDataset(BaseDataset):
    """Base class for HDF5-based datasets from ISTA (MIRACLE, SRIRACHA)."""

    def _source_filename(self, **dataset_kwargs) -> str:
        """Construct the raw input filename with extension."""
        scenario = dataset_kwargs["scenario"]
        split = dataset_kwargs.get("dataset_split")
        return f"{scenario}{('-' + split) if split else ''}.h5"

    def _ir_batch_size(self, r: int, n: int) -> int:
        """Limit streamed IR batches by their float64 size as well as row count."""
        chunk_size = int(self._chunk_size)
        if chunk_size <= 0:
            msg = "_chunk_size must be > 0"
            raise ValueError(msg)
        bytes_per_row = r * n * np.dtype(np.float64).itemsize
        return min(chunk_size, max(1, (64 * 1024**2) // bytes_per_row))

    def _ingest(self, ingest_path: Path, sofa_path: Path, **dataset_kwargs) -> Path:
        """Stream an ISTA HDF5 ingest-ready file to SOFA without loading all IRs."""
        if ingest_path.is_dir():
            msg = f"{type(self).__name__} does not support directory ingest artifacts"
            raise NotImplementedError(msg)

        chunk_size = int(self._chunk_size)
        if chunk_size <= 0:
            msg = "_chunk_size must be > 0"
            raise ValueError(msg)

        logger.info(f"Streaming ISTA HDF5 {ingest_path} to SOFA {sofa_path}.")
        with self._staged_sofa(ingest_path, sofa_path, **dataset_kwargs) as staged, h5.File(ingest_path, "r") as hdf5:
            ir = hdf5["data/impulse_response"]
            m, r, n = ir.shape
            if m == 0:
                msg = "Impulse response dataset is empty"
                raise ValueError(msg)
            chunk_size = self._ir_batch_size(r, n)
            has_humidity = "humidity" in hdf5["metadata"]
            logger.info(f"Writing {m} measurements, {r} receivers, {n} samples in chunks of {chunk_size}.")
            with netCDF4.Dataset(staged, "w", format="NETCDF4") as sofa:
                self._create_default_variables(
                    sofa,
                    m=m,
                    r=r,
                    n=n,
                    has_humidity=has_humidity,
                    receiver_position=np.asarray(hdf5["data/location/receiver"]),
                    sampling_rate=float(hdf5["metadata/sampling_rate"][()]),
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
                for row_slice in _chunk_slices(m, chunk_size):
                    source[row_slice, :] = hdf5_source[row_slice]
                    temperature[row_slice] = hdf5_temperature[row_slice].astype(np.float64) + 273.15
                    speed[row_slice, 0] = hdf5_speed[row_slice]
                    if humidity is not None:
                        humidity[row_slice, 0] = hdf5_humidity[row_slice]
            # netCDF4 must close before h5py writes externally encoded chunks.
            self._write_ir_chunks(staged, ((rows.start, ir[rows]) for rows in _chunk_slices(m, chunk_size)))

        logger.info(f"Finished SOFA file {sofa_path}.")
        return sofa_path

    @contextmanager
    def _staged_sofa(self, ingest_path: Path, sofa_path: Path, **dataset_kwargs) -> Iterator[Path]:
        """Promote a Canonical SOFA Artifact only after all writers close and checks pass."""
        sofa_path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=f".{sofa_path.stem}-", dir=sofa_path.parent) as temporary:
            staged = Path(temporary) / sofa_path.name
            yield staged
            _preserve_permissions(ingest_path, staged)
            self._verify_payload(staged, ingest_path, **dataset_kwargs)
            self._verify_sofa_convention(staged)
            staged.replace(sofa_path)

    def _write_ir_chunks(
        self, sofa_path: Path, batches: Iterable[tuple[int, np.ndarray]], *, workers: int | None = None
    ) -> None:
        """Encode owned IR chunks in workers; keep every HDF5 call on this thread."""
        workers = min(6, os.cpu_count() or 1) if workers is None else workers
        if workers <= 0:
            msg = "Compression workers must be > 0"
            raise ValueError(msg)
        with h5.File(sofa_path, "r+") as sofa, ThreadPoolExecutor(max_workers=workers) as executor:
            ir = sofa["Data.IR"]
            creation = ir.id.get_create_plist()
            filters = tuple(creation.get_filter(index)[0] for index in range(creation.get_nfilters()))
            if (
                ir.ndim != _SOFA_FIR_E_DIMS
                or filters != (h5.h5z.FILTER_SHUFFLE, h5.h5z.FILTER_DEFLATE)
                or ir.chunks != (1, ir.shape[1], ir.chunks[2], 1)
                or ir.dtype.kind != "f"
                or ir.dtype.itemsize != np.dtype(np.float64).itemsize
            ):
                msg = "Threaded IR encoding requires float64, single-measurement chunks with shuffle then gzip"
                raise ValueError(msg)
            pending = deque()
            encoding = (ir.chunks, ir.dtype, ir.fillvalue, ir.compression_opts)
            try:
                for first_row, batch in batches:
                    for row, measurement in enumerate(batch):
                        for samples in _chunk_slices(measurement.shape[1], ir.chunks[2]):
                            offset = (first_row + row, 0, samples.start, 0)
                            # Own each submitted buffer; no views retaining a whole streaming batch.
                            data = np.array(measurement[:, samples], copy=True, order="C")
                            if workers == 1:
                                ir.id.write_direct_chunk(offset, _encode_ir_chunk(data, *encoding), filter_mask=0)
                            else:
                                pending.append((offset, executor.submit(_encode_ir_chunk, data, *encoding)))
                                if len(pending) >= 2 * workers:
                                    destination, future = pending.popleft()
                                    ir.id.write_direct_chunk(destination, future.result(), filter_mask=0)
                while pending:
                    destination, future = pending.popleft()
                    ir.id.write_direct_chunk(destination, future.result(), filter_mask=0)
            finally:
                for _, future in pending:
                    future.cancel()

    def _verify_payload(self, sofa_path: Path, ingest_artifact: Path, **_dataset_kwargs) -> None:
        """Verify that a streamed ISTA SOFA matches its HDF5 ingest artifact."""
        chunk_size = int(self._chunk_size)
        if chunk_size <= 0:
            msg = "_chunk_size must be > 0"
            raise ValueError(msg)

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
                rows_per_batch = (
                    self._ir_batch_size(*hdf5_variable.shape[1:]) if variable_name == "Data.IR" else chunk_size
                )
                for row_slice in _chunk_slices(hdf5_shape[0], rows_per_batch):
                    hdf5_data = hdf5_variable[row_slice]
                    if len(sofa_variable.shape) == _SOFA_FIR_E_DIMS and sofa_variable.shape[-1] == 1:
                        sofa_data = sofa_variable[row_slice, :, :, 0]
                    else:
                        sofa_data = sofa_variable[row_slice]
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

        # Keep chunks within one measurement and 1 MiB to avoid recompressing neighboring rows.
        samples_per_chunk = max(1, min(n, 1024**2 // (r * np.dtype(np.float64).itemsize)))
        sofa.createVariable(
            "Data.IR",
            "f8",
            ("M", "R", "N", "E"),
            zlib=True,
            complevel=4,
            chunksizes=(1, r, samples_per_chunk, 1),
        )
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


def _encode_ir_chunk(data: np.ndarray, shape: tuple[int, ...], dtype: np.dtype, fill_value: float, level: int) -> bytes:
    """Apply the HDF5 byte-shuffle and deflate filters, including edge padding."""
    if data.shape == shape[1:3]:
        chunk = np.asarray(data, dtype=dtype, order="C")
    else:
        chunk = np.full(shape[1:3], fill_value, dtype=dtype)
        chunk[:, : data.shape[1]] = data
    shuffled = chunk.view(np.uint8).reshape(-1, dtype.itemsize).T.copy().tobytes()
    return zlib.compress(shuffled, level)


def _canonical_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    squeezed = tuple(dim for dim in shape if dim != 1)
    return squeezed or (1,)


def _chunk_slices(length: int, chunk_size: int):
    for start in range(0, length, chunk_size):
        yield slice(start, min(start + chunk_size, length))


def _split_row_slices(n_split_grid: int, chunk_size: int, grid_rows: range | None = None):
    """Map selected full-plane grid rows to contiguous, interleaved Split batches."""
    n_full_grid = 2 * n_split_grid
    for grid_row in range(n_full_grid) if grid_rows is None else grid_rows:
        split_row, row = divmod(grid_row, 2)
        left, right = (("C1", "C2"), ("C3", "C4"))[row]
        grid_start = grid_row * n_full_grid
        for columns in _chunk_slices(n_split_grid, chunk_size):
            src = slice(split_row * n_split_grid + columns.start, split_row * n_split_grid + columns.stop)
            dst = slice(grid_start + 2 * columns.start, grid_start + 2 * columns.stop)
            yield src, dst, left, right


def _interleave_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Interleave two Split row batches without a strided disk access."""
    merged = np.empty((2 * len(left), *left.shape[1:]), dtype=np.result_type(left, right))
    merged[::2] = left
    merged[1::2] = right
    return merged


def _canonical_array(data: np.ndarray, dtype: np.dtype) -> np.ndarray:
    array = np.squeeze(np.asarray(data))
    if array.ndim == 0:
        array = array.reshape(1)
    return np.ascontiguousarray(array.astype(dtype, copy=False))


def _verify_sriracha_rows(
    sofa_path: Path,
    split_files: dict[str, Path],
    n_split_grid: int,
    split_chunk_size: int,
    grid_rows: range,
) -> None:
    """Check bounded Split batches with read-only handles owned by this process."""
    with (
        netCDF4.Dataset(sofa_path) as sofa_dataset,
        h5.File(split_files["C1"], "r") as c1,
        h5.File(split_files["C2"], "r") as c2,
        h5.File(split_files["C3"], "r") as c3,
        h5.File(split_files["C4"], "r") as c4,
    ):
        handles = {"C1": c1, "C2": c2, "C3": c3, "C4": c4}
        for src, dst, left_name, right_name in _split_row_slices(n_split_grid, split_chunk_size, grid_rows):
            left, right = handles[left_name], handles[right_name]
            comparisons = (
                (
                    "Data.IR",
                    _interleave_rows(left["data/impulse_response"][src], right["data/impulse_response"][src]),
                    sofa_dataset.variables["Data.IR"][dst, :, :, 0],
                    np.float32,
                ),
                (
                    "SourcePosition",
                    _interleave_rows(left["data/location/source"][src], right["data/location/source"][src]),
                    sofa_dataset.variables["SourcePosition"][dst, :],
                    np.float64,
                ),
            )
            for variable_name, expected, actual, dtype in comparisons:
                expected_hash = hashlib.sha256(_canonical_array(expected, np.dtype(dtype)).view(np.uint8)).hexdigest()
                actual_hash = hashlib.sha256(_canonical_array(actual, np.dtype(dtype)).view(np.uint8)).hexdigest()
                if expected_hash != actual_hash:
                    msg = (
                        f"SOFA checksum validation failed for {sofa_path}: "
                        f"checksum differs from SRIRACHA split data: {variable_name}"
                    )
                    raise ValueError(msg)


class MiracleDataset(SonicomBaseDataset, IstaBaseDataset):
    """Download the MIRACLE database from DepositOnce/SONICOM.

    Attributes
    ----------
    name : str
        Dataset name ("miracle").
    doi : str
        Digital Object Identifier ("10.14279/depositonce-20837").
    room_volume : float
        Room volume in cubic meters (830).
    measurement_date : float
        Release date in POSIX seconds, used as the SOFA MeasurementDate
        (no per-measurement date is available).
    """

    name = "miracle"
    doi = "10.14279/depositonce-20837"
    _category = DatasetCategory.ROOM_IMPULSE_RESPONSES
    sonicom_database_id = 98
    # metadata needed for creation of sofa file
    room_volume = 830
    room_type = "shoebox"
    measurement_date = 1697068800.0

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
    room_volume = 73.5
    room_type = "shoebox"
    measurement_date = 1755648000.0

    def _create_room_variables(self, sofa: netCDF4.Dataset, **_dataset_kwargs) -> None:
        """Create SRIRACHA's current shoebox metadata."""
        sofa.RoomType = self.room_type
        room_volume = sofa.createVariable("RoomVolume", "f8", ("I",))
        room_volume.Units = "cubic metre"
        room_volume[:] = self.room_volume
        if self.room_type == "shoebox":
            room_corner_a = sofa.createVariable("RoomCornerA", "f8", ("I", "C"))
            room_corner_b = sofa.createVariable("RoomCornerB", "f8", ("I", "C"))
            room_corners = sofa.createVariable("RoomCorners", "f8", ("I", "I"))
            room_corners.Type = "cartesian"
            room_corners.Units = "metre"
            room_corner_a[:] = (0.0, 0.0, 0.0)
            room_corner_b[:] = (1.0, 1.0, 1.0)
            room_corners[:] = 0.0

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

        Notes
        -----
        Large full-plane payloads may use spawned checksum reader processes.
        In Python scripts, call Get inside an ``if __name__ == "__main__":``
        guard. CLI Get already provides this guard.
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
        # Non-dense full plane -> download 4 split files; process will then merge them
        logger.info(f"Downloading SRIRACHA scenario {scenario} (4 split files)")
        pup = _pooch_from_doi(self.doi, path=provider_dir)
        for split_file in ["C1", "C2", "C3", "C4"]:
            fname = f"{scenario}-{split_file}.h5"
            _fetch(pup, fname)
        return provider_dir

    def _process(self, provider_artifact: Path, ingest_path: Path, **dataset_kwargs) -> Path:
        """Post-process SRIRACHA file if needed.

        Non-dense full-plane scenarios retain the four Provider files as the
        ingest-ready artifact set. Otherwise promotes the single file to the
        ingest stage.

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
            Path to the processed file in the ingest directory.
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

        provider_dir = ingest_path
        split_files = {
            split_name: provider_dir / f"{scenario}-{split_name}.h5" for split_name, _ in self._split_offsets
        }
        with h5.File(split_files["C1"], "r") as first:
            ir_shape = first["data/impulse_response"].shape
            n_split = ir_shape[0]
            n_split_grid = int(np.sqrt(n_split))
            if n_split == 0 or n_split_grid**2 != n_split:
                msg = "SRIRACHA Splits must contain a nonempty square grid"
                raise ValueError(msg)
            m, r, n = (len(split_files) * n_split, *ir_shape[1:])
            has_humidity = "humidity" in first["metadata"]
            split_chunk_size = max(1, self._ir_batch_size(r, n) // 2)

        logger.info(f"Streaming SRIRACHA split files for {scenario} to SOFA {sofa_path}.")
        logger.info(f"Writing {m} measurements, {r} receivers, {n} samples.")
        with (
            self._staged_sofa(provider_dir, sofa_path, **dataset_kwargs) as staged,
            h5.File(split_files["C1"], "r") as c1,
            h5.File(split_files["C2"], "r") as c2,
            h5.File(split_files["C3"], "r") as c3,
            h5.File(split_files["C4"], "r") as c4,
        ):
            handles = {"C1": c1, "C2": c2, "C3": c3, "C4": c4}
            for handle in handles.values():
                if handle["data/impulse_response"].shape != ir_shape:
                    msg = "SRIRACHA Split IR shapes must match"
                    raise ValueError(msg)
            with netCDF4.Dataset(staged, "w", format="NETCDF4") as sofa:
                self._create_default_variables(
                    sofa,
                    m=m,
                    r=r,
                    n=n,
                    has_humidity=has_humidity,
                    receiver_position=np.asarray(c1["data/location/receiver"]),
                    sampling_rate=float(c1["metadata/sampling_rate"][()]),
                )
                self._create_room_variables(sofa, **dataset_kwargs)
                source = sofa.variables["SourcePosition"]
                temperature = sofa.variables["RoomTemperature"]
                speed = sofa.variables["SpeedOfSound"]
                humidity = sofa.variables["Humidity"] if has_humidity else None
                for src, dst, left_name, right_name in _split_row_slices(n_split_grid, split_chunk_size):
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
            batches = (
                (
                    dst.start,
                    _interleave_rows(
                        handles[left]["data/impulse_response"][src], handles[right]["data/impulse_response"][src]
                    ),
                )
                for src, dst, left, right in _split_row_slices(n_split_grid, split_chunk_size)
            )
            self._write_ir_chunks(staged, batches)

        logger.info(f"Finished SOFA file {sofa_path}.")
        return sofa_path

    def _verify_payload(
        self, sofa_path: Path, ingest_artifact: Path, *, workers: int | None = None, **dataset_kwargs
    ) -> None:
        """Check full-plane payloads in independent processes, serially for small files."""
        if not ingest_artifact.is_dir():
            super()._verify_payload(sofa_path, ingest_artifact, **dataset_kwargs)
            return

        scenario = dataset_kwargs["scenario"]
        split_files = {
            split_name: ingest_artifact / f"{scenario}-{split_name}.h5" for split_name, _ in self._split_offsets
        }
        with h5.File(split_files["C1"], "r") as first:
            n_split, r, n = first["data/impulse_response"].shape
        n_split_grid = int(np.sqrt(n_split))
        if n_split == 0 or n_split_grid**2 != n_split:
            msg = "SRIRACHA Splits must contain a nonempty square grid"
            raise ValueError(msg)
        split_chunk_size = max(1, self._ir_batch_size(r, n) // 2)
        if workers is None:
            # Process startup is not worthwhile for small payloads.
            payload_bytes = len(split_files) * n_split * r * n * np.dtype(np.float64).itemsize
            # Budget 1 GiB per reader for imported libraries, bounded batches, and HDF5 caches.
            memory_workers = max(1, psutil.virtual_memory().available // 1024**3)
            workers = (
                min(6, os.cpu_count() or 1, memory_workers) if payload_bytes >= _PARALLEL_CHECKSUM_MIN_BYTES else 1
            )
        if workers <= 0:
            msg = "Checksum workers must be > 0"
            raise ValueError(msg)
        n_full_grid = 2 * n_split_grid
        workers = min(workers, n_full_grid)

        logger.info(f"Validating SOFA file {sofa_path} with {workers} checksum workers.")
        with logger.spin(f"Running data checksum on {sofa_path.name}..."):
            if workers == 1:
                _verify_sriracha_rows(sofa_path, split_files, n_split_grid, split_chunk_size, range(n_full_grid))
                return
            # Spawn avoids inheriting HDF5 state; only paths and row ranges cross IPC.
            with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as executor:
                futures = [
                    executor.submit(
                        _verify_sriracha_rows,
                        sofa_path,
                        split_files,
                        n_split_grid,
                        split_chunk_size,
                        range(worker, n_full_grid, workers),
                    )
                    for worker in range(workers)
                ]
                for future in futures:
                    future.result()
