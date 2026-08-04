"""Datasets from the Brno University of Technology Speech@FIT group."""

from pathlib import Path

from irdl.base import DatasetCategory
from irdl.raw import RawProviderDataset


class ButReverbDbDataset(RawProviderDataset):
    """Expose a manually acquired BUT ReverbDB RIR-only corpus."""

    name = "but-reverbdb"
    doi = None
    source_url = "https://speech.fit.vut.cz/software/but-speech-fit-reverb-database"
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
            Local RIR-only TGZ or extracted provider directory obtained under the provider's terms.
        """  # noqa: D205
        return cls()._get(
            source_path=source_path, cache_dir=cache_dir, export_dir=export_dir, output_format=output_format
        )

    def _source_filename(self, **_dataset_kwargs) -> str:
        return "BUT_ReverbDB_rel_19_06_RIR-Only.tgz"

    def _download(self, _provider_dir: Path, **dataset_kwargs) -> Path:
        return self._local_source(**dataset_kwargs)
