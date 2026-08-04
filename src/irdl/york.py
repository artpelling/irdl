"""Datasets from the University of York AudioLab."""

from pathlib import Path
from typing import ClassVar

import pooch

from irdl.base import DatasetCategory
from irdl.raw import RawProviderDataset


class OpenAirDataset(RawProviderDataset):
    """Download or expose one OpenAIR environment archive."""

    name = "openair"
    doi = None
    source_url = "https://www.openairlib.net/"
    _category = DatasetCategory.ROOM_IMPULSE_RESPONSES
    _environment_urls: ClassVar[dict[str, str]] = {
        "forest-in-wheldrake-wood": "https://webfiles.york.ac.uk/OPENAIR/IRs/wheldrake-wood/Forest-in-WheldrakeWood.zip",
        "first-baptist-nashville": "https://webfiles.york.ac.uk/OPENAIR/IRs/1st-baptist-nashville/1st-baptist-nashville.zip",
        "alcuin-college-university-york": "https://webfiles.york.ac.uk/OPENAIR/IRs/alcuin-college-university-york/alcuin-college-university-york.zip",
        "arthur-sykes-rymer-auditorium-university-york": "https://webfiles.york.ac.uk/OPENAIR/IRs/arthur-sykes-rymer-auditorium-university-york/arthur-sykes-rymer-auditorium-university-york.zip",
        "hendrix-hall-uoy": "https://webfiles.york.ac.uk/OPENAIR/IRs/hendrix-hall/Hendrix-Hall-UoY.zip",
        "elveden-hall-suffolk-england": "https://webfiles.york.ac.uk/OPENAIR/IRs/elveden-hall-suffolk-england/elveden-hall-suffolk-england.zip",
        "creswell-crags": "https://webfiles.york.ac.uk/OPENAIR/IRs/creswell-crags/creswell-crags.zip",
        "central-hall-university-york": "https://webfiles.york.ac.uk/OPENAIR/IRs/central-hall-university-york/central-hall-university-york.zip",
        "clifford-tower": "https://webfiles.york.ac.uk/OPENAIR/IRs/cliffords-tower/clifford-tower.zip",
    }

    @classmethod
    def get(
        cls,
        environment: str = "central-hall-university-york",
        source_path: Path | str | None = None,
        cache_dir: Path | str | None = None,
        export_dir: Path | str | None = None,
        output_format: str = "raw",
    ) -> Path:
        """
        Environment : str, optional
            Supported stable environment slug. See the dataset documentation.
        source_path : Path or str, optional
            Existing archive/directory; use this for any other OpenAIR environment.
        """  # noqa: D205
        return cls()._get(
            environment=environment,
            source_path=source_path,
            cache_dir=cache_dir,
            export_dir=export_dir,
            output_format=output_format,
        )

    def _validate_params(self, **dataset_kwargs) -> None:
        super()._validate_params(**dataset_kwargs)
        if dataset_kwargs.get("source_path") is None and dataset_kwargs["environment"] not in self._environment_urls:
            msg = "environment has no verified automatic URL; provide source_path"
            raise ValueError(msg)

    def _source_filename(self, **dataset_kwargs) -> str:
        return f"{dataset_kwargs['environment']}.zip"

    def _download(self, provider_dir: Path, **dataset_kwargs) -> Path:
        local_source = self._local_source(**dataset_kwargs)
        if local_source is not None:
            return local_source
        filename = self._source_filename(**dataset_kwargs)
        return Path(
            pooch.retrieve(
                url=self._environment_urls[dataset_kwargs["environment"]],
                known_hash=None,
                fname=filename,
                path=provider_dir,
                progressbar=True,
            )
        )
