"""Datasets from the Aalto Acoustics Lab, Espoo, Finland.

Currently this module hosts MRTD, the Multi Room Transition Dataset.
"""

from pathlib import Path

from irdl.base import BaseDataset, DatasetCategory
from irdl.downloader import _fetch, _pooch_from_doi
from irdl.logging import logger


class MultiRoomTransitionDataset(BaseDataset):
    """Download the MRTD dataset from Zenodo."""

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
