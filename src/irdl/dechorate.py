"""dEchorate directional room impulse responses from Zenodo and SONICOM."""

from pathlib import Path

from irdl.base import BaseDataset, DatasetCategory
from irdl.downloader import _fetch, _pooch_from_doi
from irdl.sonicom import SonicomBaseDataset


class DechorateDataset(SonicomBaseDataset, BaseDataset):
    """Download the dEchorate database from Zenodo/SONICOM."""

    name = "dechorate"
    doi = "10.5281/zenodo.6576203"
    _category = DatasetCategory.ROOM_IMPULSE_RESPONSES
    sonicom_database_id = 74
    _room_codes = frozenset(
        {"000000", "010000", "011000", "011100", "011110", "011111", "001000", "000100", "000010", "000001", "020002"}
    )

    @classmethod
    def get(
        cls,
        room_code: str = "000000",
        source: int = 1,
        array: int = 1,
        cache_dir: str | Path | None = None,
        export_dir: str | Path | None = None,
        output_format: str = "pyfar",
    ) -> dict | Path | None:
        """
        room_code : str, optional
            Six-digit wall configuration code. Default is ``'000000'``.
        source : int, optional
            Directional loudspeaker index from 1 through 9. Default is 1.
        array : int, optional
            Five-microphone array index from 1 through 6. Default is 1.

        Returns
        -------
        dict or Path
            For 'pyfar' / 'numpy': dict of in-memory objects.
            For 'sofa' / 'hdf5' / 'raw': Path to file on disk.
        """  # noqa: D205
        return cls()._get(
            room_code=room_code,
            source=source,
            array=array,
            cache_dir=cache_dir,
            export_dir=export_dir,
            output_format=output_format,
        )

    def _validate_params(self, **dataset_kwargs) -> None:
        """Validate dEchorate selectors."""
        room_code = dataset_kwargs["room_code"]
        source = dataset_kwargs["source"]
        array = dataset_kwargs["array"]
        if room_code not in self._room_codes:
            msg = f"room_code must be one of {sorted(self._room_codes)}"
            raise ValueError(msg)
        if not isinstance(source, int):
            msg = "source must be an integer in the range 1 to 9"
            raise TypeError(msg)
        if source not in range(1, 10):
            msg = "source must be an integer in the range 1 to 9"
            raise ValueError(msg)
        if not isinstance(array, int):
            msg = "array must be an integer in the range 1 to 6"
            raise TypeError(msg)
        if array not in range(1, 7):
            msg = "array must be an integer in the range 1 to 6"
            raise ValueError(msg)

    def _source_filename(self, **dataset_kwargs) -> str:
        """Return the SONICOM SOFA filename for the requested slice."""
        room_code = dataset_kwargs["room_code"]
        source = dataset_kwargs["source"]
        array = dataset_kwargs["array"]
        first_mic = 5 * (array - 1) + 1
        return f"dEchorate_room{room_code}_src{source}_arr{array}_mics{first_mic}-{first_mic + 4}.sofa"

    def _download(self, provider_dir: Path, **_dataset_kwargs) -> Path:
        """Download the canonical Zenodo HDF5 archive for raw retrieval."""
        pup = _pooch_from_doi(self.doi, path=provider_dir)
        filename = "dEchorate_rirs_gzip7.hdf5"
        _fetch(pup, filename)
        return provider_dir / filename
