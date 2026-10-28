"""Datasets from the University of Electro-Communications, Tokyo, Japan.

Currently this module hosts the 3D Meshgrid Room Impulse Response Dataset.
"""

from datetime import UTC, datetime
from pathlib import Path

import h5py as h5
import netCDF4
import numpy as np

from irdl import config
from irdl._sofa import choose_ir_chunk_shape, write_ir_chunks
from irdl.base import BaseDataset, DatasetCategory
from irdl.downloader import _fetch, _pooch_from_doi
from irdl.logging import logger
from irdl.utils import _preserve_permissions


class Meshgrid3dDataset(BaseDataset):
    """Download the 3D Meshgrid Room Impulse Response Dataset."""

    name = "3dmeshgrid"
    doi = "10.5281/zenodo.17051811"
    _category = DatasetCategory.ROOM_IMPULSE_RESPONSES

    @classmethod
    def get(
        cls,
        speaker: int = 1,
        cache_dir: Path | str | None = None,
        export_dir: Path | str | None = None,
        output_format: str = "pyfar",
    ) -> dict | Path | None:
        """
        speaker : int
            Loudspeaker position to retrieve, from 1 to 4. Default is 1.

        Notes
        -----
        SOFA stores one measurement and emitter, with microphone locations along
        the receiver axis. Use 'sofa' or 'hdf5' for file-based access; 'numpy' and
        'pyfar' load all requested IRs into memory. Unknown temperature and
        measurement dates are not inferred.
        """  # noqa: D205, D403
        return cls()._get(
            speaker=speaker,
            cache_dir=cache_dir,
            export_dir=export_dir,
            output_format=output_format,
        )

    def _validate_params(self, **dataset_kwargs) -> None:
        """Validate the selected loudspeaker position."""
        speaker = dataset_kwargs["speaker"]
        if isinstance(speaker, bool) or not isinstance(speaker, int) or speaker not in range(1, 5):
            msg = "speaker must be an integer from 1 to 4"
            raise ValueError(msg)

    def _source_filename(self, **dataset_kwargs) -> str:
        """Return the Provider filename for one loudspeaker position."""
        return f"A_ir_16k_sp{dataset_kwargs['speaker']}.h5"

    def _download(self, provider_dir: Path, **dataset_kwargs) -> Path:
        """Acquire the selected checksum-verified HDF5 Provider artifact."""
        filename = self._source_filename(**dataset_kwargs)
        logger.debug("Canonical Provider artifact requested: %s", filename)
        pooch = _pooch_from_doi(self.doi, path=provider_dir)
        _fetch(pooch, filename)
        return provider_dir / filename

    def _ingest(self, ingest_path: Path, sofa_path: Path, **_dataset_kwargs) -> Path:
        """Write a single loudspeaker and many microphone positions through the shared writer."""
        batch_bytes = config._get_batch_bytes()
        with h5.File(ingest_path, "r") as source:
            info = _provider_info(source)
            receivers, samples = source["rir"].shape
            chunks = choose_ir_chunk_shape((1, receivers, samples, 1), np.dtype("f8"))
            sofa_path.parent.mkdir(parents=True, exist_ok=True)
            with netCDF4.Dataset(sofa_path, "w", format="NETCDF4") as sofa:
                self._create_skeleton(sofa, source, info, chunks, batch_bytes)
            # NetCDF4 must close before the shared h5py writer opens the skeleton.
            write_ir_chunks(sofa_path, _meshgrid_ir_chunks(source["rir"], chunks))
        _preserve_permissions(ingest_path, sofa_path)
        self._verify_payload(sofa_path, ingest_path)
        return sofa_path

    def _create_skeleton(self, sofa, source, info, chunks, batch_bytes) -> None:
        """Create Meshgrid metadata without borrowing another Dataset Group's defaults."""
        receivers, samples = source["rir"].shape
        for name, size in {"M": 1, "R": receivers, "N": samples, "E": 1, "C": 3, "I": 1}.items():
            sofa.createDimension(name, size)
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        sofa.setncatts(
            {
                "Conventions": "SOFA",
                "Version": "2.1",
                "SOFAConventions": "SingleRoomMIMOSRIR",
                "SOFAConventionsVersion": "1.0",
                "DataType": "FIR-E",
                "RoomType": "shoebox",
                "Title": "3D Meshgrid Room Impulse Response Dataset",
                "DatabaseName": "3D Meshgrid Room Impulse Response Dataset",
                "DateCreated": now,
                "DateModified": now,
                "APIName": "IRDL",
                "APIVersion": "1.0",
                "Organization": "The University of Electro-Communications",
                "AuthorContact": "yi.ren@uec.ac.jp",
                "License": "CC BY 4.0",
                "References": f"https://doi.org/{self.doi}",
                "RoomLocation": "The University of Electro-Communications, Tokyo, Japan",
                "ListenerShortName": "Microphone grid",
                "ListenerDescription": "Microphone positions sampled on a 2 cm 3D grid",
                "ReceiverShortName": "Measurement microphone",
                "ReceiverDescription": "Microphone traversed on a 2 cm 3D grid",
                "SourceShortName": "Loudspeaker",
                "SourceDescription": "Cambridge Audio MINX MIN12 loudspeaker",
                "Comment": (
                    "Each receiver is one measured microphone location in the Provider room coordinates. "
                    "Orientations are not supplied; SOFA coordinate frames use x-forward and z-up axes."
                ),
            }
        )
        sofa.createVariable("Data.IR", "f8", ("M", "R", "N", "E"), zlib=True, complevel=4, chunksizes=chunks)
        variables = {
            "ListenerPosition": (("M", "C"), np.zeros((1, 3)), True),
            "ListenerView": (("I", "C"), [[1.0, 0.0, 0.0]], True),
            "ListenerUp": (("I", "C"), [[0.0, 0.0, 1.0]], False),
            "SourcePosition": (("M", "C"), info["sp_pos"][None], True),
            "SourceView": (("I", "C"), [[1.0, 0.0, 0.0]], True),
            "SourceUp": (("I", "C"), [[0.0, 0.0, 1.0]], False),
            # The emitter is at the Source origin, not at a second absolute speaker offset.
            "EmitterPosition": (("E", "C", "I"), np.zeros((1, 3, 1)), True),
            "Data.Delay": (("M", "R", "E"), 0.0, False),
            "Data.SamplingRate": (("I",), info["fs"], False),
            "SpeedOfSound": (("M", "I"), info["c"].reshape(1, 1), False),
            "RoomVolume": (("I",), [np.prod(info["room_size"])], False),
            "RoomCornerA": (("I", "C"), np.zeros((1, 3)), False),
            "RoomCornerB": (("I", "C"), info["room_size"][None], False),
            "RoomCorners": (("I", "I"), 0.0, True),
        }
        for name, (dimensions, values, cartesian) in variables.items():
            variable = sofa.createVariable(name, "f8", dimensions)
            variable[:] = values
            if cartesian:
                variable.Type = "cartesian"
                variable.Units = "metre"
        sofa.variables["Data.SamplingRate"].Units = "hertz"
        sofa.variables["SpeedOfSound"].Units = "metre/second"
        sofa.variables["RoomVolume"].Units = "cubic metre"
        receiver = sofa.createVariable("ReceiverPosition", "f8", ("R", "C", "I"))
        receiver.Type = "cartesian"
        receiver.Units = "metre"
        batch_rows = max(1, batch_bytes // (3 * 8))
        for start in range(0, receivers, batch_rows):
            rows = slice(start, min(start + batch_rows, receivers))
            receiver[rows, :, 0] = source["grid"][:, rows].T

    def _verify_payload(self, sofa_path: Path, ingest_artifact: Path, **_dataset_kwargs) -> None:
        """Compare IRs and supplied geometry exactly in bounded receiver/sample tiles."""
        batch_bytes = config._get_batch_bytes()
        logger.info("Payload check: IRs, supplied geometry, sampling rate, and mapped metadata")
        logger.debug("Payload check: %s against %s; precision=exact float64", sofa_path, ingest_artifact)
        with h5.File(ingest_artifact, "r") as source, h5.File(sofa_path, "r") as sofa:
            info = _provider_info(source)
            rir = source["rir"]
            ir = sofa["Data.IR"]
            if ir.shape != (1, *rir.shape, 1):
                msg = f"Meshgrid payload mismatch: Data.IR shape; expected {(1, *rir.shape, 1)}, actual {ir.shape}"
                raise ValueError(msg)
            chunks = choose_ir_chunk_shape(ir.shape, ir.dtype)
            with logger.progress("Checking Meshgrid IRs", total=rir.size) as advance:
                for start in range(0, rir.shape[0], chunks[1]):
                    rows = slice(start, min(start + chunks[1], rir.shape[0]))
                    for sample in range(0, rir.shape[1], chunks[2]):
                        samples = slice(sample, min(sample + chunks[2], rir.shape[1]))
                        expected = rir[rows, samples]
                        _compare("Data.IR", expected, ir[0, rows, samples, 0])
                        advance(expected.size)
            batch_rows = max(1, batch_bytes // (3 * 8))
            for start in range(0, rir.shape[0], batch_rows):
                rows = slice(start, min(start + batch_rows, rir.shape[0]))
                _compare("ReceiverPosition", source["grid"][:, rows].T, sofa["ReceiverPosition"][rows, :, 0])
                _compare("Data.Delay", np.zeros((1, rows.stop - rows.start, 1)), sofa["Data.Delay"][:, rows, :])
            for name in ("ListenerView", "SourceView"):
                _compare(name, [[1.0, 0.0, 0.0]], sofa[name][:])
            for name in ("ListenerUp", "SourceUp"):
                _compare(name, [[0.0, 0.0, 1.0]], sofa[name][:])
            _compare("SourcePosition", info["sp_pos"][None], sofa["SourcePosition"][:])
            _compare("EmitterPosition", np.zeros((1, 3, 1)), sofa["EmitterPosition"][:])
            _compare("ListenerPosition", np.zeros((1, 3)), sofa["ListenerPosition"][:])
            _compare("Data.SamplingRate", info["fs"], sofa["Data.SamplingRate"][:])
            _compare("SpeedOfSound", info["c"].reshape(1, 1), sofa["SpeedOfSound"][:])
            _compare("RoomCornerA", np.zeros((1, 3)), sofa["RoomCornerA"][:])
            _compare("RoomCornerB", info["room_size"][None], sofa["RoomCornerB"][:])
            _compare("RoomVolume", [np.prod(info["room_size"])], sofa["RoomVolume"][:])


def _provider_info(source: h5.File) -> dict[str, np.ndarray]:
    """Validate the MATLAB matrix layout and read the small numeric attributes."""
    rir, grid = source["rir"], source["grid"]
    if rir.ndim != 2 or not all(rir.shape) or grid.shape != (3, rir.shape[0]):  # noqa: PLR2004 - MATLAB matrices.
        msg = f"Expected MATLAB-layout /rir and /grid datasets, got {rir.shape} and {grid.shape}"
        raise ValueError(msg)
    if rir.dtype.kind != "f" or not np.can_cast(rir.dtype, np.dtype("f8"), casting="safe"):
        msg = "Meshgrid /rir must contain real floating-point values representable as float64"
        raise ValueError(msg)
    info = {}
    for name, size in {"sp_pos": 3, "room_size": 3, "fs": 1, "c": 1}.items():
        values = np.asarray(source.attrs[name], dtype="f8").reshape(-1)
        if values.size != size or not np.isfinite(values).all():
            msg = f"Invalid Meshgrid attribute {name}: expected {size} finite values"
            raise ValueError(msg)
        info[name] = values
    if any(np.any(info[name] <= 0) for name in ("room_size", "fs", "c")):
        msg = "Meshgrid room_size, fs, and c must be positive"
        raise ValueError(msg)
    return info


def _meshgrid_ir_chunks(rir: h5.Dataset, chunks: tuple[int, int, int, int]):
    """Map requested microphone/sample blocks to complete stored SOFA IR chunks."""
    for receiver in range(0, rir.shape[0], chunks[1]):
        rows = slice(receiver, min(receiver + chunks[1], rir.shape[0]))
        for sample in range(0, rir.shape[1], chunks[2]):
            samples = slice(sample, min(sample + chunks[2], rir.shape[1]))
            yield (0, receiver, sample, 0), rir[rows, samples][np.newaxis, ..., np.newaxis]


def _compare(name: str, expected: np.ndarray, actual: np.ndarray) -> None:
    """Reject shape changes and any difference in the stored float64 values."""
    expected = np.asarray(expected, dtype="f8")
    actual = np.asarray(actual, dtype="f8")
    if expected.shape != actual.shape:
        msg = f"Meshgrid payload mismatch: {name}; expected shape {expected.shape}, actual {actual.shape}"
        raise ValueError(msg)
    if expected.tobytes() != actual.tobytes():
        different = np.flatnonzero(expected.view("u8").ravel() != actual.view("u8").ravel())
        index = tuple(int(value) for value in np.unravel_index(different[0], expected.shape))
        msg = (
            f"Meshgrid payload mismatch: {name} at local index {index}; "
            f"expected {expected[index]!r}, actual {actual[index]!r}"
        )
        raise ValueError(msg)
