"""Datasets from the Aalto Acoustics Lab, Espoo, Finland.

Currently this module hosts MRTD, the Multi Room Transition Dataset.
"""

from pathlib import Path

from irdl.base import BaseDataset, DatasetCategory
from irdl.downloader import _fetch, _pooch_from_doi
from irdl.logging import logger
from irdl.raw import RawProviderDataset


class MultiRoomTransitionDataset(BaseDataset):
    """Download one SOFA file from the Multi Room Transition Dataset."""

    name = "mrtd"
    doi = "10.5281/zenodo.13341566"
    _category = DatasetCategory.ROOM_IMPULSE_RESPONSES
    _environments = frozenset({"hallways-lecturehall", "offices", "workshops"})
    _receivers = frozenset({"kemar", "zoom"})

    @classmethod
    def get(
        cls,
        environment: str = "offices",
        receiver: str = "kemar",
        loudspeaker: int = 1,
        cache_dir: Path | str | None = None,
        export_dir: Path | str | None = None,
        output_format: str = "pyfar",
    ) -> dict | Path | None:
        """
        Environment : str, optional
            Acoustic environment. One of 'hallways-lecturehall', 'offices',
            or 'workshops'. Default is 'offices'.
        receiver : str, optional
            'kemar' selects the binaural KEMAR and 'zoom' selects the
            four-channel Zoom H3-VR. Default is 'kemar'.
        loudspeaker : int, optional
            Stationary loudspeaker number from 1 through 4. Default is 1.

        Returns
        -------
        dict or Path
            For 'pyfar' / 'numpy': dict of in-memory objects.
            For 'sofa' / 'hdf5' / 'raw': Path to file on disk.
        """  # noqa: D205
        return cls()._get(
            environment=environment,
            receiver=receiver,
            loudspeaker=loudspeaker,
            cache_dir=cache_dir,
            export_dir=export_dir,
            output_format=output_format,
        )

    def _validate_params(self, **dataset_kwargs) -> None:
        """Validate the MRTD environment, receiver, and loudspeaker."""
        environment = dataset_kwargs["environment"]
        receiver = dataset_kwargs["receiver"]
        loudspeaker = dataset_kwargs["loudspeaker"]
        if environment not in self._environments:
            msg = f"environment must be one of {sorted(self._environments)}"
            raise ValueError(msg)
        if receiver not in self._receivers:
            msg = f"receiver must be one of {sorted(self._receivers)}"
            raise ValueError(msg)
        if not isinstance(loudspeaker, int) or loudspeaker not in range(1, 5):
            msg = "loudspeaker must be an integer from 1 through 4"
            raise ValueError(msg)

    def _source_filename(self, **dataset_kwargs) -> str:
        """Return the provider's SOFA filename for the requested setup."""
        environment = dataset_kwargs["environment"]
        receiver = dataset_kwargs["receiver"]
        loudspeaker = dataset_kwargs["loudspeaker"]
        return f"{environment}_{receiver}_ls_{loudspeaker}.sofa"

    def _download(self, provider_dir: Path, **dataset_kwargs) -> Path:
        """Download the requested native SOFA file from Zenodo."""
        source_filename = self._source_filename(**dataset_kwargs)
        provider_path = provider_dir / source_filename
        if provider_path.exists():
            logger.info(f"SOFA file already cached at {provider_path}, skipping download")
        else:
            logger.info(f"Downloading {self.name.upper()} file {source_filename}")
            pooch = _pooch_from_doi(self.doi, path=provider_dir)
            _fetch(pooch, source_filename)
        return provider_path


class ArniRoomImpulseResponseDataset(RawProviderDataset):
    """Expose a local copy of the Arni room impulse response dataset."""

    name = "arni"
    doi = "10.5281/zenodo.6985104"
    _category = DatasetCategory.ROOM_IMPULSE_RESPONSES
    local_source_required = True

    @classmethod
    def get(
        cls,
        source_path: Path | str | None = None,
        cache_dir: Path | str | None = None,
        export_dir: Path | str | None = None,
        output_format: str = "raw",
    ) -> Path:
        """
        source_path : Path or str
            Local provider directory containing the original WAV files.
        """  # noqa: D205
        return cls()._get(
            source_path=source_path, cache_dir=cache_dir, export_dir=export_dir, output_format=output_format
        )

    def _source_filename(self, **_dataset_kwargs) -> str:
        return "IR_Arni_upload_numClosed_0-5"

    def _download(self, _provider_dir: Path, **dataset_kwargs) -> Path:
        return self._local_source(**dataset_kwargs)


class MotusDataset(RawProviderDataset):
    """Download or expose the MOTUS raw room impulse response archive."""

    name = "motus"
    doi = "10.5281/zenodo.4923187"
    _category = DatasetCategory.ROOM_IMPULSE_RESPONSES

    @classmethod
    def get(
        cls,
        source_path: Path | str | None = None,
        cache_dir: Path | str | None = None,
        export_dir: Path | str | None = None,
        output_format: str = "raw",
    ) -> Path:
        """
        source_path : Path or str, optional
            Existing ``raw_rirs.zip`` or extracted provider directory. If omitted,
            download from Zenodo.
        """  # noqa: D205
        return cls()._get(
            source_path=source_path, cache_dir=cache_dir, export_dir=export_dir, output_format=output_format
        )

    def _source_filename(self, **_dataset_kwargs) -> str:
        return "raw_rirs.zip"

    def _download(self, provider_dir: Path, **dataset_kwargs) -> Path:
        local_source = self._local_source(**dataset_kwargs)
        if local_source is not None:
            return local_source
        archive = provider_dir / self._source_filename()
        if not archive.exists():
            _fetch(_pooch_from_doi(self.doi, path=provider_dir), archive.name)
        return archive
