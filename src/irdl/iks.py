"""Datasets from the Institute of Communication Systems, RWTH Aachen University, Aachen, Germany.

Currently this module hosts MIRD, the Multi-Channel Impulse Response Database.
"""

from pathlib import Path
from typing import ClassVar
from zipfile import ZipFile

import numpy as np
import sofar as sf
from scipy.io import loadmat

from irdl.base import BaseDataset, DatasetCategory
from irdl.downloader import _fetch, _pooch_from_static_registry
from irdl.logging import logger
from irdl.utils import load_hash_registry


class MirdDataset(BaseDataset):
    """Download MIRD measurements for one or both source distances."""

    name = "mird"
    doi = "10.1109/IWAENC.2014.6954309"
    _category = DatasetCategory.ROOM_IMPULSE_RESPONSES
    _t60s = frozenset({0.16, 0.36, 0.61})
    _spacings: ClassVar[dict[int, str]] = {3: "3-3-3-8-3-3-3", 4: "4-4-4-8-4-4-4", 8: "8-8-8-8-8-8-8"}
    _distances = frozenset({1, 2})
    _download_root = "https://www.iks.rwth-aachen.de/fileadmin/user_upload/downloads/forschung/tools-downloads"

    @classmethod
    def get(
        cls,
        t60: float = 0.16,
        spacing: int = 8,
        distance: int | str = "both",
        cache_dir: Path | str | None = None,
        export_dir: Path | str | None = None,
        output_format: str = "pyfar",
        *,
        remove_delay: bool = True,
    ) -> dict | Path | None:
        """
        t60 : float
            Reverberation time in seconds. One of 0.16, 0.36, or 0.61.
        spacing : int
            Nominal inter-microphone spacing in centimetres: 3, 4, or 8.
        distance : int or str
            Source distance in metres: 1, 2, or 'both'. Each selected distance
            includes all 13 measured azimuths. Default is 'both'.
        remove_delay : bool
            Remove MIRD's recorded additional delay. Default is True.
        """  # noqa: D205
        return cls()._get(
            t60=t60,
            spacing=spacing,
            distance=distance,
            remove_delay=remove_delay,
            cache_dir=cache_dir,
            export_dir=export_dir,
            output_format=output_format,
        )

    def _validate_params(self, **dataset_kwargs) -> None:
        """Validate MIRD's archive and source-distance selectors."""
        if dataset_kwargs["t60"] not in self._t60s:
            msg = f"t60 must be one of {sorted(self._t60s)}"
            raise ValueError(msg)
        if dataset_kwargs["spacing"] not in self._spacings:
            msg = f"spacing must be one of {sorted(self._spacings)}"
            raise ValueError(msg)
        if dataset_kwargs["distance"] not in {*self._distances, "both"}:
            msg = "distance must be 1, 2, or 'both'"
            raise ValueError(msg)
        if not isinstance(dataset_kwargs["remove_delay"], bool):
            msg = "remove_delay must be a boolean"
            raise TypeError(msg)

    def _source_filename(self, **dataset_kwargs) -> str:
        """Return the unique ingest artifact name for the requested SOFA."""
        t60 = {0.16: "short", 0.36: "mid", 0.61: "long"}[dataset_kwargs["t60"]]
        delay = "" if dataset_kwargs["remove_delay"] else "-nodelay"
        return f"{t60}-{dataset_kwargs['spacing']}-{dataset_kwargs['distance']}{delay}.mird"

    def _download(self, provider_dir: Path, **dataset_kwargs) -> Path:
        """Download the selected MIRD archive."""
        archive = (
            "Impulse_response_Acoustic_Lab_Bar-Ilan_University__"
            f"Reverberation_{dataset_kwargs['t60']:.3f}s__{self._spacings[dataset_kwargs['spacing']]}.zip"
        )
        archive_path = provider_dir / archive
        if archive_path.exists():
            logger.info(f"MIRD archive already cached at {archive_path}, skipping download")
            return archive_path
        logger.info(f"Downloading MIRD archive {archive}")
        pooch = _pooch_from_static_registry(
            path=provider_dir,
            registry={archive: load_hash_registry(self.name)[archive]},
            urls={archive: f"{self._download_root}/{archive}"},
        )
        _fetch(pooch, archive)
        return archive_path

    def _process(self, provider_artifact: Path, ingest_path: Path, **dataset_kwargs) -> Path:
        """Extract all MAT files at the selected source distance(s)."""
        distance = dataset_kwargs["distance"]
        distances = (1, 2) if distance == "both" else (distance,)
        with ZipFile(provider_artifact) as archive:
            members = [
                member
                for member in archive.namelist()
                if member.endswith(".mat") and any(f"_{distance}m_" in member for distance in distances)
            ]
            ingest_path.mkdir(parents=True, exist_ok=True)
            for member in members:
                archive.extract(member, ingest_path)
        return ingest_path

    def _ingest(self, ingest_path: Path, sofa_path: Path, **dataset_kwargs) -> Path:
        """Convert selected MIRD source distances to a SingleRoomMIMOSRIR SOFA file."""
        responses, source_positions = [], []
        if dataset_kwargs["remove_delay"]:
            logger.info("Removing excess measurement delay ...")
        for member in sorted(ingest_path.glob("*.mat")):
            contents = loadmat(member, simplify_cells=True)
            impulse_response = np.asarray(contents["impulse_response"], dtype=float)
            if dataset_kwargs["remove_delay"]:
                impulse_response = impulse_response[int(contents["simpar"]["int_delay"]) :]
            responses.append(impulse_response.T)
            metadata = contents["metapar"]
            angle = np.deg2rad(metadata["azimuth"])
            distance = metadata["distance"]
            source_positions.append([distance * np.cos(angle), distance * np.sin(angle), 0.0])

        ir = np.asarray(responses)[..., np.newaxis]
        measurements, receivers, _samples, _emitters = ir.shape
        spacings = self._spacings[dataset_kwargs["spacing"]].split("-")
        spacing = np.array([int(value) for value in spacings], dtype=float) / 100
        receiver_y = np.r_[0.0, np.cumsum(spacing)]
        receiver_y -= receiver_y.mean()
        source_positions = np.asarray(source_positions)

        sofa = sf.Sofa("SingleRoomMIMOSRIR")
        sofa.GLOBAL_Title = "MIRD"
        sofa.GLOBAL_DatabaseName = "Multi-Channel Impulse Response Database"
        sofa.GLOBAL_Organization = "Institute of Communication Systems, RWTH Aachen University"
        sofa.GLOBAL_References = f"https://doi.org/{self.doi}"
        sofa.GLOBAL_RoomType = "reverberant"
        sofa.GLOBAL_Comment = "MIRD source positions and inferred linear microphone-array geometry."
        sofa.Data_IR = ir
        sofa.Data_SamplingRate = 48000.0
        sofa.Data_Delay = np.zeros((measurements, receivers, 1))
        sofa.MeasurementDate = np.zeros(measurements)
        sofa.ListenerPosition = np.zeros((measurements, 3))
        sofa.SourcePosition = source_positions
        sofa.ReceiverPosition = np.column_stack((np.zeros(receivers), receiver_y, np.zeros(receivers))).reshape(
            receivers, 3, 1
        )
        sofa.ReceiverView = np.tile([1.0, 0.0, 0.0], (receivers, 1))[..., np.newaxis]
        sofa.ReceiverUp = np.tile([0.0, 0.0, 1.0], (receivers, 1))[..., np.newaxis]
        sofa.ReceiverDescriptions = np.array(["AKG CK32"] * receivers)
        sofa.EmitterPosition = source_positions[:1].reshape(1, 3, 1)
        sofa.EmitterView = np.array([[[1.0], [0.0], [0.0]]])
        sofa.EmitterUp = np.array([[[0.0], [0.0], [1.0]]])
        sofa.EmitterDescriptions = np.array(["Fostex 6301B and ADAM A3X loudspeaker"])
        sofa_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write_sofa(sofa_path, sofa)
        return sofa_path
