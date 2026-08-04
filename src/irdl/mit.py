"""Datasets from the MIT McDermott Computational Audition Laboratory."""

from pathlib import Path

from irdl.base import DatasetCategory
from irdl.raw import RawProviderDataset


class MitImpulseResponseSurveyDataset(RawProviderDataset):
    """Expose a local copy of the MIT Impulse Response Survey."""

    name = "mit-ir-survey"
    doi = None
    source_url = "https://mcdermottlab.mit.edu/Reverb/IR_Survey.html"
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
            Local ``mit_audio.zip`` or extracted ``Audio`` directory.
        """  # noqa: D205
        return cls()._get(
            source_path=source_path, cache_dir=cache_dir, export_dir=export_dir, output_format=output_format
        )

    def _source_filename(self, **_dataset_kwargs) -> str:
        return "mit_audio.zip"

    def _download(self, _provider_dir: Path, **dataset_kwargs) -> Path:
        return self._local_source(**dataset_kwargs)
