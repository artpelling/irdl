"""Native-SOFA datasets hosted by the SONICOM Ecosystem."""

from pathlib import Path

from irdl.base import BaseDataset, DatasetCategory
from irdl.utils import load_url_registry

_SONICOM_URLS = load_url_registry("sonicom")


class SonicomSofaDataset(BaseDataset):
    """Base class for SONICOM datasets supplied as individual SOFA files."""

    doi = None
    source_url = "https://ecosystem.sonicom.eu/databases"
    _category = DatasetCategory.HEAD_RELATED_IMPULSE_RESPONSES

    def direct_sofa_url(self, source_filename: str) -> str | None:
        """Return the verified SONICOM URL for a native SOFA file."""
        return _SONICOM_URLS.get(source_filename)

    def _download(self, provider_dir: Path, **dataset_kwargs) -> Path:
        """Download the requested native SOFA file from SONICOM."""
        source_filename = self._source_filename(**dataset_kwargs)
        url = self.direct_sofa_url(source_filename)
        if url is None:
            msg = f"No SONICOM SOFA URL registered for {source_filename!r}"
            raise ValueError(msg)
        return self._download_direct_sofa(provider_dir, url)


class CipicDataset(SonicomSofaDataset):
    """Download an individual CIPIC HRTF SOFA file from SONICOM."""

    name = "cipic"
    source_url = "https://ecosystem.sonicom.eu/databases/72"
    _subjects = frozenset(
        int(filename.removeprefix("subject_").removesuffix(".sofa"))
        for filename in _SONICOM_URLS
        if filename.startswith("subject_")
    )

    @classmethod
    def get(
        cls,
        subject: int = 3,
        cache_dir: str | Path | None = None,
        export_dir: str | Path | None = None,
        output_format: str = "pyfar",
    ) -> dict | Path | None:
        """
        subject : int, optional
            CIPIC subject identifier. Default is 3.

        Returns
        -------
        dict or Path
            For 'pyfar' / 'numpy': dict of in-memory objects.
            For 'sofa' / 'hdf5' / 'raw': Path to file on disk.
        """  # noqa: D205, D403
        return cls()._get(
            subject=subject,
            cache_dir=cache_dir,
            export_dir=export_dir,
            output_format=output_format,
        )

    def _validate_params(self, **dataset_kwargs) -> None:
        """Validate the CIPIC subject identifier."""
        subject = dataset_kwargs["subject"]
        if not isinstance(subject, int) or subject not in self._subjects:
            msg = "subject must identify a CIPIC SOFA file"
            raise ValueError(msg)

    def _source_filename(self, **dataset_kwargs) -> str:
        """Return the native SONICOM filename for the requested subject."""
        return f"subject_{dataset_kwargs['subject']:03d}.sofa"


class SadieDataset(SonicomSofaDataset):
    """Download an individual SADIE II HRTF SOFA file from SONICOM."""

    name = "sadie"
    source_url = "https://ecosystem.sonicom.eu/databases/92"
    _subjects = frozenset(
        filename.split("_", 1)[0] for filename in _SONICOM_URLS if filename.endswith("_48K_24bit_256tap_FIR_SOFA.sofa")
    )

    @classmethod
    def get(
        cls,
        subject: str = "H3",
        cache_dir: str | Path | None = None,
        export_dir: str | Path | None = None,
        output_format: str = "pyfar",
    ) -> dict | Path | None:
        """
        subject : str, optional
            SADIE II listener or dummy-head identifier. One of 'D1', 'D2',
            or 'H3' through 'H20'. Default is 'H3'.

        Returns
        -------
        dict or Path
            For 'pyfar' / 'numpy': dict of in-memory objects.
            For 'sofa' / 'hdf5' / 'raw': Path to file on disk.
        """  # noqa: D205, D403
        return cls()._get(
            subject=subject,
            cache_dir=cache_dir,
            export_dir=export_dir,
            output_format=output_format,
        )

    def _validate_params(self, **dataset_kwargs) -> None:
        """Validate the SADIE II listener or dummy-head identifier."""
        subject = dataset_kwargs["subject"]
        if subject not in self._subjects:
            msg = "subject must be one of 'D1', 'D2', or 'H3' through 'H20'"
            raise ValueError(msg)

    def _source_filename(self, **dataset_kwargs) -> str:
        """Return the native SONICOM filename for the requested subject."""
        return f"{dataset_kwargs['subject']}_48K_24bit_256tap_FIR_SOFA.sofa"
