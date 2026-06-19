"""Shared dataset support for SOFA-backed datasets from the SONICOM ecosystem."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path

from irdl.base import SofaBaseDataset
from irdl.downloader import _fetch, _pooch_from_sonicom_database
from irdl.utils import load_hash_registry


class SonicomBaseDataset(SofaBaseDataset):
    """Abstract base class for SOFA-backed datasets served from SONICOM.

    SONICOM is modeled as a non-canonical Provider that serves SOFA artifacts
    from SONICOM database manifests. Concrete subclasses provide the requested
    file name, while the base class derives the SONICOM database URL and
    packaged checksum lookup.
    """

    _sonicom_root = "https://ecosystem.sonicom.eu/databases"
    _sonicom_database_id: int | None = None

    @abstractmethod
    def _source_filename(self, **dataset_kwargs) -> str:
        """Return the ingest-ready SOFA filename for the requested SONICOM artifact."""

    def _sonicom_database_url(self) -> str:
        """Return the SONICOM database landing URL for manifest-backed downloads."""
        if self._sonicom_database_id is None:
            msg = f"{self.__class__.__name__} must define _sonicom_database_id for manifest-backed SONICOM downloads"
            raise NotImplementedError(msg)
        return f"{self._sonicom_root}/{self._sonicom_database_id}"

    def _sonicom_registry_key(self, **dataset_kwargs) -> str:
        """Return the packaged SONICOM registry key for one artifact."""
        source_filename = self._source_filename(**dataset_kwargs)
        return f"{self.name}/{source_filename}"

    def _sonicom_checksum(self, **dataset_kwargs) -> str | None:
        """Return the checksum for one SONICOM artifact."""
        return load_hash_registry("sonicom")[self._sonicom_registry_key(**dataset_kwargs)]

    def _sonicom_pooch(self, provider_dir: Path, **dataset_kwargs):
        """Return a configured pooch instance for one SONICOM download."""
        source_filename = self._source_filename(**dataset_kwargs)
        checksum = self._sonicom_checksum(**dataset_kwargs)
        return _pooch_from_sonicom_database(
            path=provider_dir,
            database_url=self._sonicom_database_url(),
            fname=source_filename,
            checksum=checksum,
        )

    def _provider_artifact_format(self, provider: str, **_dataset_kwargs) -> str:
        """Return the Provider-side artifact Data Format for SONICOM datasets.

        SONICOM-backed Provider artifacts are expected to be SOFA files.
        """
        if provider != "sonicom":
            msg = f"Unknown provider {provider!r} for {self.name.upper()}"
            raise ValueError(msg)
        return "sofa"

    def _download(self, provider_dir: Path, provider: str, **dataset_kwargs) -> Path:
        """Download the selected SONICOM SOFA file from a direct fetch spec.

        Parameters
        ----------
        provider_dir : Path
            Provider directory for the SONICOM cache stage.
        provider : str
            Provider name. Must be ``"sonicom"``.
        **dataset_kwargs : dict
            Dataset-specific parameters used to choose one SONICOM file.

        Returns
        -------
        Path
            Path to the downloaded SOFA file inside the SONICOM Provider cache.
        """
        if provider != "sonicom":
            msg = f"Unknown provider {provider!r} for {self.name.upper()}"
            raise ValueError(msg)
        source_filename = self._source_filename(**dataset_kwargs)
        self.logger.info("provider=%r artifact=%r -> download to provider cache", provider, source_filename)
        pup = self._sonicom_pooch(provider_dir, **dataset_kwargs)
        return Path(_fetch(pup, source_filename))
