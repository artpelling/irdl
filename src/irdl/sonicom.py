"""Datasets available through the SONICOM Ecosystem.

Currently this module hosts CIPIC, dEchorate, and SADIE II.
"""

import json
from functools import cache
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from irdl.base import BaseDataset, DatasetCategory
from irdl.downloader import _fetch, _pooch_from_doi
from irdl.utils import load_hash_registry

_SONICOM_ROOT = "https://ecosystem.sonicom.eu"


@cache
def _sonicom_manifest(database_id: int) -> dict[str, str]:
    """Return the current SONICOM filename-to-URL manifest for one database."""
    url = f"{_SONICOM_ROOT}/databases/{database_id}/download?type=json"
    try:
        with urlopen(url) as response:  # noqa: S310 - fixed HTTPS provider endpoint
            payload = json.load(response)
    except (OSError, json.JSONDecodeError) as error:
        msg = f"Could not retrieve SONICOM manifest for database {database_id}"
        raise RuntimeError(msg) from error

    try:
        datafiles = payload["data"]
    except (KeyError, TypeError) as error:
        msg = f"Invalid SONICOM manifest for database {database_id}"
        raise ValueError(msg) from error
    if not isinstance(datafiles, list):
        msg = f"Invalid SONICOM manifest for database {database_id}"
        raise TypeError(msg)

    prefix = f"{_SONICOM_ROOT}/data/{database_id}/"
    manifest = {}
    for datafile in datafiles:
        try:
            filename = datafile["Datafile Name"]
            datafile_url = datafile["Datafile URL"]
        except (KeyError, TypeError) as error:
            msg = f"Invalid SONICOM manifest entry for database {database_id}"
            raise ValueError(msg) from error
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not isinstance(datafile_url, str)
            or not datafile_url.startswith(prefix)
            or Path(urlparse(datafile_url).path).name != filename
        ):
            msg = f"Invalid SONICOM manifest entry for database {database_id}"
            raise ValueError(msg)
        manifest[filename] = datafile_url
    return manifest


class SonicomBaseDataset:
    """Resolve SONICOM SOFA URLs dynamically while pinning checked-in digests."""

    sonicom_database_id: int

    def _direct_sofa(self, source_filename: str) -> tuple[str, str] | tuple[str, None] | tuple[None, None]:
        """Resolve a SONICOM SOFA artifact and its optional pinned digest."""
        filename = Path(source_filename).with_suffix(".sofa").name
        url = _sonicom_manifest(self.sonicom_database_id).get(filename)
        return (url, load_hash_registry("sonicom").get(filename)) if url is not None else (None, None)

    def _download(self, provider_dir: Path, **dataset_kwargs) -> Path:
        """Download the requested native SOFA file from SONICOM."""
        source_filename = self._source_filename(**dataset_kwargs)
        url, known_hash = self._direct_sofa(source_filename)
        if url is not None:
            return self._download_direct_sofa(provider_dir, url, known_hash)
        msg = f"No SONICOM SOFA available for {source_filename!r}"
        raise ValueError(msg)


class CipicDataset(SonicomBaseDataset, BaseDataset):
    """Download the CIPIC HRTF database from SONICOM."""

    name = "cipic"
    doi = "10.1109/ASPAA.2001.969552"
    _category = DatasetCategory.HEAD_RELATED_IMPULSE_RESPONSES
    sonicom_database_id = 72
    _subjects = frozenset(
        int(filename.removeprefix("subject_").removesuffix(".sofa"))
        for filename in load_hash_registry("sonicom")
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
            msg = f"subject must be one of {sorted(self._subjects)}"
            raise ValueError(msg)

    def _source_filename(self, **dataset_kwargs) -> str:
        """Return the native SONICOM filename for the requested subject."""
        return f"subject_{dataset_kwargs['subject']:03d}.sofa"


class SadieDataset(SonicomBaseDataset, BaseDataset):
    """Download the SADIE II database from SONICOM."""

    name = "sadie"
    doi = "10.3390/app8112029"
    _category = DatasetCategory.HEAD_RELATED_IMPULSE_RESPONSES
    sonicom_database_id = 92
    _subjects = frozenset(
        filename.split("_", 1)[0]
        for filename in load_hash_registry("sonicom")
        if filename.endswith("_48K_24bit_256tap_FIR_SOFA.sofa")
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
