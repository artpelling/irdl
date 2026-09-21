"""Datasets from the Audio Communications Group of TU Berlin, Berlin, Germany.

Currently this module hosts:

- BRAS-RS8: Benchmark for Room Acoustical Simulation: Reference Scene 8.
- FABIAN: The FABIAN head-related transfer function data base.
- HUTUBS: The HUTUBS head-related transfer function (HRTF) database.

"""

from pathlib import Path
from zipfile import ZipFile

from irdl.base import BaseDataset, DatasetCategory
from irdl.downloader import _fetch, _pooch_from_doi
from irdl.logging import logger
from irdl.utils import load_url_registry

_SONICOM_URLS = load_url_registry("sonicom")


class AKTZipBaseDataset(BaseDataset):
    """Base class for zipped datasets of the Audio Communications Group of TU Berlin."""

    _zipfile: str

    def _process(self, provider_artifact: Path, ingest_path: Path, **_dataset_kwargs) -> Path:
        """Extract the requested SOFA file from the ZIP into the ingest directory.

        Parameters
        ----------
        provider_artifact : :class:`pathlib.Path`
            Path to the ZIP archive in the provider directory.
        ingest_path : :class:`pathlib.Path`
            Path to the SOFA file in the ingest directory.
        **_dataset_kwargs : dict
            Unused dataset-specific parameters (accepted for compatibility).

        Returns
        -------
        :class:`pathlib.Path`
            Path to the extracted SOFA file in the ingest directory.
        """
        with ZipFile(provider_artifact, "r") as zf:
            for name in zf.namelist():
                if name.endswith(ingest_path.name):
                    # Flatten the extraction (strip any nested ZIP directory)
                    zf.getinfo(name).filename = Path(name).name
                    logger.info(f"Extracting {name} to {ingest_path.parent}")
                    zf.extract(name, path=ingest_path.parent)
                    return ingest_path

            msg = (
                f"No entry matching '{ingest_path.name}' found in archive {provider_artifact}. "
                "Check zf.namelist() for available entries."
            )
            raise FileNotFoundError(msg)

    def _download(self, provider_dir: Path, **_dataset_kwargs) -> Path:
        """Download BRAS-RS8 Scene_descriptions.zip archive to the provider directory.

        Only downloads the archive if it is not already cached in the provider
        directory. Returns the ZIP path so that ``_process`` can extract the
        requested SOFA file into the ingest directory.

        Parameters
        ----------
        provider_dir : Path
            Provider directory (e.g., ``cache/BRAS-RS8/provider/``).
        **_dataset_kwargs : dict
            Unused dataset-specific parameters (accepted for compatibility).

        Returns
        -------
        Path
            Path to the downloaded ZIP archive.
        """
        zip_path = provider_dir / self._zipfile
        if zip_path.exists():
            logger.info(f"ZIP archive already cached at {zip_path}, skipping download")
        else:
            logger.info(f"Downloading {self.name.upper()} dataset")
            pup = _pooch_from_doi(self.doi, path=provider_dir)
            _fetch(pup, self._zipfile)
        return zip_path


class BrasRs8Dataset(AKTZipBaseDataset):
    """Download the BRAS RS8 dataset from DepositOnce.

    BRAS RS8 extends the Benchmark for Room Acoustical Simulation (BRAS) by a
    finite curved reflector. The dataset contains nine scenes with various source
    and receiver configurations, covering dispersing and focusing reflections,
    as well as diffraction around the surface. In total, nearly 3,000 impulse
    responses were measured under controlled anechoic conditions.

    Attributes
    ----------
    name : str
        Dataset name ("bras-rs8").
    doi : str
        Digital Object Identifier ("10.14279/depositonce-25649").
    """

    name = "bras-rs8"
    doi = "10.14279/depositonce-25649"
    _category = DatasetCategory.ROOM_IMPULSE_RESPONSES
    _zipfile = "1_Scene_descriptions.zip"

    @classmethod
    def get(
        cls,
        scene: str = "01a",
        cache_dir: Path | str | None = None,
        export_dir: Path | str | None = None,
        output_format: str = "pyfar",
    ) -> dict | Path | None:
        """
        scene : str, optional
            Scene identifier to download. One of:
            '01a', '01b', '01c', '02',
            '03a', '03b', '03c', '03d', '03e'.
            Default is '01a'.

        Returns
        -------
        dict or Path
            For 'pyfar' / 'numpy': dict of in-memory objects.
            For 'sofa' / 'hdf5' / 'raw': Path to file on disk.
        """  # noqa: D205, D403
        return cls()._get(
            scene=scene,
            cache_dir=cache_dir,
            export_dir=export_dir,
            output_format=output_format,
        )

    def _validate_params(self, **dataset_kwargs) -> None:
        """Validate BRAS-RS8-specific parameters.

        Parameters
        ----------
        **dataset_kwargs : dict
            Must contain 'scene' (one of the valid scene identifiers).

        Raises
        ------
        ValueError
            If scene is not one of the valid scene identifiers.
        """
        scene = dataset_kwargs["scene"]
        valid_scenes = {
            "01a",
            "01b",
            "01c",
            "02",
            "03a",
            "03b",
            "03c",
            "03d",
            "03e",
        }
        if scene not in valid_scenes:
            msg = f"scene must be one of {sorted(valid_scenes)}"
            raise ValueError(msg)

    def _source_filename(self, **dataset_kwargs) -> str:
        """Construct the ingest-ready (SOFA) filename.

        Parameters
        ----------
        **dataset_kwargs : dict
            Expected key: scene.

        Returns
        -------
        str
            File name in format "RS8_{scene}.sofa".
        """
        scene = dataset_kwargs["scene"]
        return f"RS8_RIRs_{scene}.sofa"


class FabianDataset(AKTZipBaseDataset):
    """Download and extract the FABIAN HRTF database from DepositOnce.

    Attributes
    ----------
    name : str
        Dataset name ("fabian").
    doi : str
        Digital Object Identifier ("10.14279/depositonce-5718.5").
    """

    name = "fabian"
    doi = "10.14279/depositonce-5718.5"
    _category = DatasetCategory.HEAD_RELATED_IMPULSE_RESPONSES
    _zipfile = "FABIAN_HRTF_DATABASE_v4.zip"

    @classmethod
    def get(
        cls,
        kind: str = "measured",
        hato: int = 0,
        cache_dir: str | Path | None = None,
        export_dir: str | Path | None = None,
        output_format: str = "pyfar",
    ) -> dict | Path | None:
        """
        kind : str, optional
            Type of HRTF to download. Either 'measured' or 'simulated'. Default is 'measured'.
        hato : int, optional
            Head-above-torso-rotation of HRTFs in degrees.
            One of: 0, 10, 20, 30, 40, 50, 310, 320, 330, 340, 350. Default is 0.

        Returns
        -------
        dict or Path
            For 'pyfar' / 'numpy': dict of in-memory objects.
            For 'sofa' / 'hdf5' / 'raw': Path to file on disk.
        """  # noqa: D205, D403
        return cls()._get(
            kind=kind,
            hato=hato,
            cache_dir=cache_dir,
            export_dir=export_dir,
            output_format=output_format,
        )

    def _validate_params(self, **dataset_kwargs) -> None:
        """Validate FABIAN-specific parameters.

        Parameters
        ----------
        **dataset_kwargs : dict
            Parameters to validate. Expected keys: kind, hato.

        Raises
        ------
        ValueError
            If kind or hato is out of range.
        """
        kind = dataset_kwargs["kind"]
        hato = dataset_kwargs["hato"]

        if kind not in ["measured", "simulated"]:
            msg = "kind must be either 'measured' or 'simulated'"
            raise ValueError(msg)
        if hato not in [0, 10, 20, 30, 40, 50, 310, 320, 330, 340, 350]:
            msg = "hato must be one of [0, 10, 20, 30, 40, 50, 310, 320, 330, 340, 350]"
            raise ValueError(msg)

    def _source_filename(self, **dataset_kwargs) -> str:
        """Construct the ingest-ready (SOFA) filename.

        Parameters
        ----------
        **dataset_kwargs : dict
            Expected keys: kind, hato.

        Returns
        -------
        str
            File name in format "FABIAN_HRIR_{kind}_HATO_{hato}.sofa".
        """
        return f"FABIAN_HRIR_{dataset_kwargs['kind']}_HATO_{dataset_kwargs['hato']}.sofa"


class HutubsDataset(AKTZipBaseDataset):
    """Download the HUTUBS HRTF database from DepositOnce."""

    name = "hutubs"
    doi = "10.14279/depositonce-8487"
    _category = DatasetCategory.HEAD_RELATED_IMPULSE_RESPONSES
    _zipfile = "HRIRs.zip"

    @classmethod
    def get(
        cls,
        subject: int = 1,
        kind: str = "measured",
        cache_dir: str | Path | None = None,
        export_dir: str | Path | None = None,
        output_format: str = "pyfar",
    ) -> dict | Path | None:
        """
        subject : int, optional
            Subject identifier. Must be an integer in the range 1 to 96. Default is 1.
        kind : str, optional
            HUTUBS HRIR variant. Either 'measured' or 'simulated'. Default is 'measured'.

        Returns
        -------
        dict or Path
            For 'pyfar' / 'numpy': dict of in-memory objects.
            For 'sofa' / 'hdf5' / 'raw': Path to file on disk.
        """  # noqa: D205, D403
        return cls()._get(
            subject=subject,
            kind=kind,
            cache_dir=cache_dir,
            export_dir=export_dir,
            output_format=output_format,
        )

    def _validate_params(self, **dataset_kwargs) -> None:
        """Validate HUTUBS-specific parameters."""
        subject = dataset_kwargs["subject"]
        kind = dataset_kwargs["kind"]

        if not isinstance(subject, int):
            msg = "subject must be an integer in the range 1 to 96"
            raise TypeError(msg)
        if subject not in range(1, 97):
            msg = "subject must be an integer in the range 1 to 96"
            raise ValueError(msg)
        if kind not in ["measured", "simulated"]:
            msg = "kind must be either 'measured' or 'simulated'"
            raise ValueError(msg)

    def _source_filename(self, **dataset_kwargs) -> str:
        """Construct the ingest-ready SOFA file name."""
        return f"pp{dataset_kwargs['subject']}_HRIRs_{dataset_kwargs['kind']}.sofa"

    def direct_sofa_url(self, source_filename: str) -> str | None:
        """Return SONICOM's direct URL for the requested individual HUTUBS SOFA."""
        return _SONICOM_URLS.get(source_filename)
