"""Shared support for provenance-preserving raw provider datasets."""

from abc import abstractmethod
from pathlib import Path

from irdl.base import BaseDataset


class RawProviderDataset(BaseDataset):
    """Base class for datasets retained in their provider-native representation."""

    local_source_required = False

    def _validate_params(self, **dataset_kwargs) -> None:
        """Require raw output and, where needed, an explicit local source."""
        if dataset_kwargs["output_format"] != "raw":
            msg = f"{self.name} currently supports output_format='raw' only"
            raise ValueError(msg)
        if self.local_source_required and dataset_kwargs.get("source_path") is None:
            msg = f"{self.name} requires source_path because automatic download is unavailable"
            raise ValueError(msg)

    def _local_source(self, **dataset_kwargs) -> Path | None:
        """Validate and return an optional provider artifact supplied by the user."""
        source_path = dataset_kwargs.get("source_path")
        if source_path is None:
            return None
        source = Path(source_path).expanduser()
        if not source.exists():
            msg = f"source_path does not exist: {source}"
            raise FileNotFoundError(msg)
        return source

    @abstractmethod
    def _download(self, provider_dir: Path, **dataset_kwargs) -> Path:
        """Return a local artifact or acquire one from the provider."""
