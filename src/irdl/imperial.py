"""Datasets from Imperial College London's Audio Experience Design group."""

from pathlib import Path

from irdl.base import DatasetCategory
from irdl.downloader import _fetch, _pooch_from_doi
from irdl.raw import RawProviderDataset


class AceChallengeDataset(RawProviderDataset):
    """Download one receiver archive from the ACE Challenge corpus."""

    name = "ace-challenge"
    doi = "10.5281/zenodo.6257551"
    _category = DatasetCategory.ROOM_IMPULSE_RESPONSES
    _receivers = frozenset({"Single", "EM32", "Chromebook", "Lin8Ch", "Crucif", "Mobile"})

    @classmethod
    def get(
        cls,
        receiver: str = "Single",
        source_path: Path | str | None = None,
        cache_dir: Path | str | None = None,
        export_dir: Path | str | None = None,
        output_format: str = "raw",
    ) -> Path:
        """
        Receiver : str, optional
            Receiver archive: Single, EM32, Chromebook, Lin8Ch, Crucif, or Mobile.
        source_path : Path or str, optional
            Existing archive or extracted provider directory. If omitted, download from Zenodo.
        """  # noqa: D205
        return cls()._get(
            receiver=receiver,
            source_path=source_path,
            cache_dir=cache_dir,
            export_dir=export_dir,
            output_format=output_format,
        )

    def _validate_params(self, **dataset_kwargs) -> None:
        super()._validate_params(**dataset_kwargs)
        if dataset_kwargs["receiver"] not in self._receivers:
            msg = f"receiver must be one of {sorted(self._receivers)}"
            raise ValueError(msg)

    def _source_filename(self, **dataset_kwargs) -> str:
        return f"ACE_Corpus_RIRN_{dataset_kwargs['receiver']}.tbz2"

    def _download(self, provider_dir: Path, **dataset_kwargs) -> Path:
        local_source = self._local_source(**dataset_kwargs)
        if local_source is not None:
            return local_source
        archive = provider_dir / self._source_filename(**dataset_kwargs)
        if not archive.exists():
            _fetch(_pooch_from_doi(self.doi, path=provider_dir), archive.name)
        return archive
