"""Datasets from the University of Electro-Communications, Tokyo, Japan.

Currently this module hosts the 3D Meshgrid Room Impulse Response Dataset.
"""

# ruff: noqa: D205, D403

from pathlib import Path

import h5py as h5
import netCDF4
import numpy as np

from irdl.base import DatasetCategory
from irdl.downloader import _fetch, _pooch_from_doi
from irdl.ista import IstaBaseDataset, _canonical_array, _chunk_slices, _write_encoded_ir_chunks
from irdl.logging import logger

_CARTESIAN_DIMENSIONS = 3
_HDF5_MATRIX_DIMENSIONS = 2


class Meshgrid3dDataset(IstaBaseDataset):
    """Download the 3D Meshgrid Room Impulse Response Dataset."""

    name = "3dmeshgrid"
    doi = "10.5281/zenodo.17051811"
    _category = DatasetCategory.ROOM_IMPULSE_RESPONSES
    measurement_date = 1756944000.0

    @classmethod
    def get(
        cls,
        speaker: int = 1,
        cache_dir: str | Path | None = None,
        export_dir: str | Path | None = None,
        output_format: str = "pyfar",
    ) -> dict | Path | None:
        """
        speaker : int
            Loudspeaker position to retrieve, from 1 to 4. Default is 1.
        """
        return cls()._get(
            speaker=speaker,
            cache_dir=cache_dir,
            export_dir=export_dir,
            output_format=output_format,
        )

    def _validate_params(self, **dataset_kwargs) -> None:
        """Validate the selected loudspeaker position."""
        speaker = dataset_kwargs["speaker"]
        if speaker not in (1, 2, 3, 4):
            msg = "speaker must be an integer from 1 to 4"
            raise ValueError(msg)

    def _source_filename(self, **dataset_kwargs) -> str:
        """Return the provider filename for one loudspeaker position."""
        return f"A_ir_16k_sp{dataset_kwargs['speaker']}.h5"

    def _download(self, provider_dir: Path, **dataset_kwargs) -> Path:
        """Download one checksum-verified 3D Meshgrid HDF5 provider artifact."""
        filename = self._source_filename(**dataset_kwargs)
        logger.info(f"Downloading 3D Meshgrid speaker {dataset_kwargs['speaker']}")
        pup = _pooch_from_doi(self.doi, path=provider_dir)
        _fetch(pup, filename)
        return provider_dir / filename

    def _create_room_variables(self, sofa: netCDF4.Dataset, **_dataset_kwargs) -> None:
        """Create room fields after the dimensions have been established."""
        sofa.RoomType = "shoebox"
        room_volume = sofa.createVariable("RoomVolume", "f8", ("I",))
        room_corner_a = sofa.createVariable("RoomCornerA", "f8", ("I", "C"))
        room_corner_b = sofa.createVariable("RoomCornerB", "f8", ("I", "C"))
        room_corners = sofa.createVariable("RoomCorners", "f8", ("I", "I"))
        room_volume.Units = "cubic metre"
        room_corners.Type = "cartesian"
        room_corners.Units = "metre"
        room_volume[:] = 0.0
        room_corner_a[:] = 0.0
        room_corner_b[:] = 0.0
        room_corners[:] = 0.0

    def _ingest(self, ingest_path: Path, sofa_path: Path, **_dataset_kwargs) -> Path:
        """Stream a MATLAB HDF5 provider artifact into a physical SOFA representation."""
        chunk_size = int(self._chunk_size)
        if chunk_size <= 0:
            msg = "_chunk_size must be > 0"
            raise ValueError(msg)

        with self._staged_sofa(ingest_path, sofa_path) as staged, h5.File(ingest_path, "r") as hdf5:
            rir = hdf5["rir"]
            grid = hdf5["grid"]
            if (
                rir.ndim != _HDF5_MATRIX_DIMENSIONS
                or not all(rir.shape)
                or grid.shape != (_CARTESIAN_DIMENSIONS, rir.shape[0])
            ):
                msg = f"Expected MATLAB-layout /rir and /grid datasets, got {rir.shape} and {grid.shape}"
                raise ValueError(msg)
            speaker_position = np.asarray(hdf5.attrs["sp_pos"], dtype=float).reshape(-1)
            room_size = np.asarray(hdf5.attrs["room_size"], dtype=float).reshape(-1)
            if speaker_position.size != _CARTESIAN_DIMENSIONS or room_size.size != _CARTESIAN_DIMENSIONS:
                msg = "sp_pos and room_size attributes must each contain three coordinates"
                raise ValueError(msg)

            measurements, samples = rir.shape
            receiver_chunk = min(chunk_size, measurements, max(1, 1024**2 // (samples * 8)))
            sample_chunk = min(samples, 1024**2 // (receiver_chunk * 8))
            grid_positions = np.asarray(grid, dtype=float).T
            sampling_rate = float(np.asarray(hdf5.attrs["fs"]).item())
            speed_of_sound = float(np.asarray(hdf5.attrs["c"]).item())
            logger.info(f"Streaming {measurements} grid positions with {samples} samples to SOFA {sofa_path}.")
            with netCDF4.Dataset(staged, "w", format="NETCDF4") as sofa:
                self._create_default_variables(
                    sofa,
                    m=1,
                    r=measurements,
                    n=samples,
                    has_humidity=False,
                    receiver_position=grid_positions,
                    sampling_rate=sampling_rate,
                    ir_chunks=(1, receiver_chunk, sample_chunk, 1),
                )
                self._create_room_variables(sofa)
                sofa.Title = "3D Meshgrid Room Impulse Response Dataset"
                sofa.DatabaseName = "3D Meshgrid Room Impulse Response Dataset"
                sofa.Organization = "The University of Electro-Communications"
                sofa.AuthorContact = "yi.ren@uec.ac.jp"
                sofa.License = "CC BY 4.0"
                sofa.RoomLocation = "The University of Electro-Communications, Tokyo, Japan"
                sofa.ListenerShortName = "Measurement microphone array"
                sofa.ListenerDescription = "Microphone positions sampled on a 2 cm 3D grid"
                sofa.ReceiverShortName = "Measurement microphone"
                sofa.ReceiverDescription = "Microphone traversed on a 2 cm 3D grid"
                sofa.SourceShortName = "Loudspeaker"
                sofa.SourceDescription = "Cambridge Audio MINX MIN12 loudspeaker"
                sofa.Comment = "Each receiver position is one microphone location on the measured 3D grid."
                sofa.variables["RoomVolume"][:] = np.prod(room_size)
                sofa.variables["RoomCornerB"][:] = room_size
                sofa.variables["EmitterPosition"][:] = speaker_position.reshape(1, 3, 1)
                sofa.variables["SourcePosition"][:] = speaker_position.reshape(1, 3)
                sofa.variables["SpeedOfSound"][:] = speed_of_sound

            # Close netCDF4 before the shared writer opens the staged HDF5 artifact.
            with h5.File(staged, "r+") as sofa:
                chunks = (
                    ((0, receivers.start, samples_slice.start, 0), rir[receivers, samples_slice])
                    for receivers in _chunk_slices(measurements, receiver_chunk)
                    for samples_slice in _chunk_slices(samples, sample_chunk)
                )
                _write_encoded_ir_chunks(sofa["Data.IR"], chunks)

        return sofa_path

    def _verify_payload(self, sofa_path: Path, ingest_artifact: Path, **_dataset_kwargs) -> None:
        """Compare bounded receiver batches exactly before promoting the SOFA artifact."""

        def compare(name, expected, actual):
            dtype = np.dtype(np.float64)
            expected = _canonical_array(expected, dtype).view(np.uint8)
            actual = _canonical_array(actual, dtype).view(np.uint8)
            if not np.array_equal(expected, actual):
                msg = f"Meshgrid payload mismatch: {name}"
                raise ValueError(msg)

        with h5.File(ingest_artifact, "r") as source, netCDF4.Dataset(sofa_path) as sofa:
            rir = source["rir"]
            if sofa.variables["Data.IR"].shape != (1, *rir.shape, 1):
                msg = "Meshgrid payload mismatch: Data.IR shape"
                raise ValueError(msg)
            for receivers in _chunk_slices(rir.shape[0], self._ir_batch_size(1, rir.shape[1])):
                compare("Data.IR", rir[receivers], sofa.variables["Data.IR"][0, receivers, :, 0])
                compare(
                    "ReceiverPosition", source["grid"][:, receivers].T, sofa.variables["ReceiverPosition"][receivers]
                )
            compare("SourcePosition", source.attrs["sp_pos"], sofa.variables["SourcePosition"][:])
            compare("EmitterPosition", source.attrs["sp_pos"], sofa.variables["EmitterPosition"][:])
            compare("Data.SamplingRate", source.attrs["fs"], sofa.variables["Data.SamplingRate"][:])
