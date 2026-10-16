"""Datasets from the Fraunhofer Institute for Integrated Circuits IIS, Erlangen, Germany.

Currently this module hosts MP-RIR, the Multi-Purpose Room Impulse Response Dataset
measured on a 3D spatial grid.
"""

import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

import netCDF4
import numpy as np

from irdl.base import BaseDataset, DatasetCategory
from irdl.downloader import _fetch, _pooch_from_doi
from irdl.logging import logger


class MpRirDataset(BaseDataset):
    """Download MP-RIR measurements of one loudspeaker on the full 3D microphone grid."""

    name = "mprir"
    doi = "10.5281/zenodo.11148712"
    _category = DatasetCategory.ROOM_IMPULSE_RESPONSES

    @classmethod
    def get(
        cls,
        source: str = "S1",
        cache_dir: Path | str | None = None,
        export_dir: Path | str | None = None,
        output_format: str = "pyfar",
        *,
        remove_delay: bool = True,
    ) -> dict | Path | None:
        """
        source : str
            Loudspeaker to retrieve, one of 'S1' to 'S8'. S1 to S4 are subwoofers,
            S5 to S8 are full-range loudspeakers.
        remove_delay : bool
            Remove the guard time and the measurement system delay so that sample
            zero corresponds to sound emission. Default is True.
        """  # noqa: D205, D403
        return cls()._get(
            source=source,
            remove_delay=remove_delay,
            cache_dir=cache_dir,
            export_dir=export_dir,
            output_format=output_format,
        )

    def _validate_params(self, **dataset_kwargs) -> None:
        """Validate the MP-RIR loudspeaker and delay selectors."""
        if dataset_kwargs["source"] not in ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]:
            msg = "source must be one of ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']"
            raise ValueError(msg)
        if not isinstance(dataset_kwargs["remove_delay"], bool):
            msg = "remove_delay must be a boolean"
            raise TypeError(msg)

    def _source_filename(self, **dataset_kwargs) -> str:
        """Return the unique ingest directory name for the requested SOFA."""
        delay = "" if dataset_kwargs["remove_delay"] else "-nodelay"
        return f"{dataset_kwargs['source']}{delay}.mprir"

    def _download(self, provider_dir: Path, **dataset_kwargs) -> Path:
        """Download the selected loudspeaker's RIRs with the shared geometry and setup files."""
        source = dataset_kwargs["source"]
        # one folder with all three files per loudspeaker for raw export functionality
        source_dir = provider_dir / source
        files = (f"{source}_Mrir.npy", "Mxyz.npy", "Setup.npz")
        missing = [file for file in files if not (source_dir / file).exists()]
        if not missing:
            logger.info(f"MP-RIR files for {source} already cached at {source_dir}, skipping download")
            return source_dir
        logger.info(f"Downloading MP-RIR files for {source}")
        pooch = _pooch_from_doi(self.doi, path=source_dir)
        for file in missing:
            _fetch(pooch, file)
        return source_dir

    def _process(self, provider_artifact: Path, ingest_path: Path, **_dataset_kwargs) -> Path:
        """Promote the provider files to the ingest directory."""
        ingest_path.mkdir(parents=True, exist_ok=True)
        for file in provider_artifact.iterdir():
            target = ingest_path / file.name
            if target.exists():
                continue
            try:
                os.link(file, target)
            except OSError:
                shutil.copy2(file, target)
        return ingest_path

    def _ingest(self, ingest_path: Path, sofa_path: Path, **dataset_kwargs) -> Path:  # noqa: PLR0915
        """Stream the selected loudspeaker's RIRs to a SingleRoomSRIR SOFA file."""
        source = dataset_kwargs["source"]
        remove_delay = dataset_kwargs["remove_delay"]
        source_descriptions = {
            "S1": 'Custom-built 8" subwoofer',
            "S2": 'Custom-built 8" subwoofer',
            "S3": 'Custom-built 8" subwoofer',
            "S4": 'Custom-built 8" subwoofer',
            "S5": "Genelec 8250A two-way near-field monitor",
            "S6": "Neumann KH 310 three-way near-field monitor (upright)",
            "S7": "Neumann KH 80 two-way near-field monitor",
            "S8": "Neumann KH 80 two-way near-field monitor",
        }
        room_volume = 138.63

        rirs = np.load(ingest_path / f"{source}_Mrir.npy", mmap_mode="r")
        mic_positions = np.load(ingest_path / "Mxyz.npy").astype(np.float64)
        # Setup.npz stores dictionaries as pickled 0-d object arrays of plain numpy values.
        with np.load(ingest_path / "Setup.npz", allow_pickle=True) as setup:
            sampling_rate = float(setup["fs"])
            delay = int(setup["T_guard"]) + int(setup["T_system"])
            source_position = np.asarray(setup["coord_speaker_center"].item()[source], dtype=np.float64)
            azimuth = np.deg2rad(float(setup["angles_speaker"].item()[source]))
            polygon = np.asarray(setup["coord_polygon"], dtype=np.float64)

        m, r, _ = rirs.shape
        start = delay if remove_delay else 0
        n = rirs.shape[2] - start
        if remove_delay:
            logger.info(f"Removing guard time and system delay ({delay} samples) ...")
            delay_comment = f"The guard time and the measurement system delay ({delay} samples in total) were removed."
        else:
            delay_comment = f"The RIRs start with the guard time and the system delay ({delay} samples in total)."
        corners = ", ".join(f"C{q}=({x:.3f}, {y:.3f})" for q, (x, y) in enumerate(polygon))
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        sofa_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing {m} measurements, {r} receivers, {n} samples to {sofa_path}.")
        with netCDF4.Dataset(sofa_path, "w", format="NETCDF4") as sofa:
            for dimension, size in {"M": m, "R": r, "N": n, "E": 1, "C": 3, "I": 1}.items():
                sofa.createDimension(dimension, size)

            sofa.Conventions = "SOFA"
            sofa.Version = "2.1"
            sofa.SOFAConventions = "SingleRoomSRIR"
            sofa.SOFAConventionsVersion = "1.0"
            sofa.DataType = "FIR"
            sofa.RoomType = "reverberant"
            sofa.Title = f"MP-RIR loudspeaker {source}"
            sofa.DatabaseName = "Multi-Purpose Room Impulse Response Dataset (MP-RIR)"
            sofa.DateCreated = now
            sofa.DateModified = now
            sofa.APIName = "IRDL"
            sofa.APIVersion = "1.0"
            sofa.AuthorContact = "lukas.friede@iis.fraunhofer.de"
            sofa.Organization = "Fraunhofer Institute for Integrated Circuits IIS"
            sofa.License = "CC BY 4.0"
            sofa.References = (
                f"https://doi.org/{self.doi}; L. Friede et al., Multi-Purpose Room Impulse Response Dataset "
                "Measured on a 3D Spatial Grid, AES Convention Paper 10702, 2024"
            )
            sofa.RoomDescription = (
                f"Non-rectangular furnished room of {room_volume} cubic metres and 2.85 m height with wooden "
                f"structures on one wall and a slanted windowed wall. Floor polygon in metres: {corners}."
            )
            sofa.ListenerShortName = "Vertical microphone array"
            sofa.ListenerDescription = (
                "Eight microphones on a vertical rod with 15 cm spacing, moved by a robot on a 15 cm grid. "
                "ListenerPosition is the foot of the rod on the floor."
            )
            sofa.ReceiverShortName = "DPA 4061"
            sofa.ReceiverDescription = "DPA 4061 miniature omni-directional lavalier microphone"
            sofa.SourceShortName = source
            sofa.SourceDescription = (
                f"{source_descriptions[source]}. SourcePosition is the centre of the loudspeaker base."
            )
            sofa.Comment = (
                "Coordinates are given in the room coordinate system with the origin at room corner C0. "
                "The loudspeaker gains were aligned and all RIRs were scaled by the maximum absolute peak "
                f"of all measured RIRs. {delay_comment}"
            )

            listener = np.column_stack((mic_positions[:, 0, :2], np.zeros(m)))
            receiver = np.column_stack((np.zeros((r, 2)), mic_positions[0, :, 2]))[..., np.newaxis]
            variables = {
                "ListenerPosition": (("M", "C"), listener, True),
                "ListenerView": (("I", "C"), [[1.0, 0.0, 0.0]], True),
                "ListenerUp": (("I", "C"), [[0.0, 0.0, 1.0]], False),
                "ReceiverPosition": (("R", "C", "I"), receiver, True),
                "SourcePosition": (("M", "C"), np.tile(source_position, (m, 1)), True),
                "SourceView": (("I", "C"), [[np.cos(azimuth), np.sin(azimuth), 0.0]], True),
                "SourceUp": (("I", "C"), [[0.0, 0.0, 1.0]], False),
                "EmitterPosition": (("E", "C", "I"), np.zeros((1, 3, 1)), True),
                "Data.Delay": (("I", "R"), np.full((1, r), delay - start, dtype=float), False),
            }
            for variable_name, (dimensions, values, cartesian) in variables.items():
                variable = sofa.createVariable(variable_name, "f8", dimensions)
                variable[:] = values
                if cartesian:
                    variable.Type = "cartesian"
                    variable.Units = "metre"
            sampling_rate_var = sofa.createVariable("Data.SamplingRate", "f8", ("I",))
            sampling_rate_var.Units = "hertz"
            sampling_rate_var[:] = sampling_rate
            room_volume_var = sofa.createVariable("RoomVolume", "f8", ("I",))
            room_volume_var.Units = "cubic metre"
            room_volume_var[:] = room_volume

            data_ir = sofa.createVariable(
                "Data.IR", "f4", ("M", "R", "N"), zlib=True, complevel=4, chunksizes=(1, r, n)
            )
            for row in range(0, m, self._chunk_size):
                rows = slice(row, min(row + self._chunk_size, m))
                data_ir[rows] = rirs[rows, :, start:]

        logger.info(f"Finished SOFA file {sofa_path}.")
        return sofa_path
