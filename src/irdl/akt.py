"""Datasets from the Audio Communications Group of TU Berlin, Berlin, Germany.

- BRAS-RS8: Benchmark for Room Acoustical Simulation: Reference Scene 8.
- FABIAN: The FABIAN head-related transfer function data base.
- HUTUBS: The HUTUBS head-related transfer function (HRTF) database.
"""

from pathlib import Path
from zipfile import ZipFile

from irdl.base import DatasetCategory, SofaBaseDataset
from irdl.downloader import _fetch, _pooch_from_doi
from irdl.sonicom import SonicomBaseDataset


class AKTZipBaseDataset(SofaBaseDataset):
    """Base class for zipped datasets of the Audio Communications Group of TU Berlin.

    These datasets share one canonical Provider (DepositOnce), publish ZIP
    archives as Provider artifacts, and extract one SOFA file into the ingest
    stage before conversion.
    """

    canonical_provider = "depositonce"
    providers = ("depositonce",)
    _zipfile: str

    def _provider_artifact_format(self, provider: str, **_dataset_kwargs) -> str:
        """Return the Provider-side artifact format for AKT ZIP datasets.

        All current AKT datasets serve ZIP archives from their canonical
        Provider.
        """
        if provider != self.canonical_provider:
            msg = f"Unknown provider {provider!r} for {self.name.upper()}"
            raise ValueError(msg)
        return "zip"

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
                    zf.getinfo(name).filename = Path(name).name
                    self.logger.info("Extracting provider artifact %s -> %s", name, ingest_path)
                    zf.extract(name, path=ingest_path.parent)
                    return ingest_path

            msg = (
                f"No entry matching '{ingest_path.name}' found in archive {provider_artifact}. "
                "Check zf.namelist() for available entries."
            )
            raise FileNotFoundError(msg)

    def _download(self, provider_dir: Path, provider: str, **_dataset_kwargs) -> Path:
        """Download the dataset ZIP archive to the Provider directory.

        Only the canonical Provider is currently supported. The downloaded ZIP
        stays in the ``provider`` Cache Stage so ``_process`` can extract the
        requested SOFA file into ``ingest``.

        Parameters
        ----------
        provider_dir : Path
            Provider directory (for example ``cache/FABIAN/provider/depositonce``).
        provider : str
            Provider name. Must be ``"depositonce"`` for current AKT datasets.

        Returns
        -------
        Path
            Path to the downloaded ZIP archive.
        """
        if provider != self.canonical_provider:
            msg = f"Unknown provider {provider!r} for {self.name.upper()}"
            raise ValueError(msg)

        zip_path = provider_dir / self._zipfile
        if zip_path.exists():
            self.logger.info("Provider cache hit: %s", zip_path)
        else:
            self.logger.info("provider=%r artifact=%r -> download to provider cache", provider, self._zipfile)
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
        provider: str = "auto",
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
            provider=provider,
        )

    def _validate_params(self, **dataset_kwargs) -> None:
        """Validate BRAS-RS8-specific parameters.

        Parameters
        ----------
        **dataset_kwargs : dict
            Must contain ``scene`` as one of the valid BRAS-RS8 scene
            identifiers. ``provider`` and ``output_format`` may also be passed
            through the shared pipeline but do not add further restrictions
            here.

        Raises
        ------
        ValueError
            If ``scene`` is invalid.
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
        """Construct the ingest-ready SOFA filename for BRAS-RS8.

        Parameters
        ----------
        **dataset_kwargs : dict
            Expected key: ``scene``.

        Returns
        -------
        str
            File name in format ``RS8_RIRs_{scene}.sofa``.
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
        provider: str = "auto",
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
            provider=provider,
        )

    def _validate_params(self, **dataset_kwargs) -> None:
        """Validate FABIAN-specific parameters.

        Parameters
        ----------
        **dataset_kwargs : dict
            Parameters to validate. Expected keys: ``kind`` and ``hato``.

        Raises
        ------
        ValueError
            If ``kind`` or ``hato`` is out of range.
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
        """Construct the ingest-ready SOFA filename for FABIAN.

        Parameters
        ----------
        **dataset_kwargs : dict
            Expected keys: ``kind`` and ``hato``.

        Returns
        -------
        str
            File name in format ``FABIAN_HRIR_{kind}_HATO_{hato}.sofa``.
        """
        return f"FABIAN_HRIR_{dataset_kwargs['kind']}_HATO_{dataset_kwargs['hato']}.sofa"


class HutubsDataset(AKTZipBaseDataset, SonicomBaseDataset):
    """Download the HUTUBS HRTF database from DepositOnce or SONICOM.

    DepositOnce remains the canonical Provider and publishes a ZIP archive.
    SONICOM mirrors individual SOFA files and is preferred for non-raw output
    formats because it avoids the ZIP extraction path.
    """

    name = "hutubs"
    doi = "10.14279/depositonce-8487"
    providers = ("sonicom", "depositonce")
    _category = DatasetCategory.HEAD_RELATED_IMPULSE_RESPONSES
    _zipfile = "HRIRs.zip"
    _sonicom_database_id = 76

    @classmethod
    def get(
        cls,
        subject: int = 1,
        kind: str = "measured",
        cache_dir: str | Path | None = None,
        export_dir: str | Path | None = None,
        output_format: str = "pyfar",
        provider: str = "auto",
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
            provider=provider,
        )

    def _provider_artifact_format(self, provider: str, **_dataset_kwargs) -> str:
        """Return the Provider-side artifact format for HUTUBS.

        ``depositonce`` publishes a ZIP Provider artifact, while ``sonicom``
        serves SOFA files directly.
        """
        if provider == "sonicom":
            return "sofa"
        return AKTZipBaseDataset._provider_artifact_format(self, provider, **_dataset_kwargs)

    def _download(self, provider_dir: Path, provider: str, **dataset_kwargs) -> Path:
        """Download HUTUBS data from the selected Provider.

        The canonical Provider delegates to the shared AKT ZIP workflow.
        ``sonicom`` delegates to :class:`~irdl.sonicom.SonicomBaseDataset`,
        which resolves the mirrored file from the SONICOM database manifest and
        verifies it against the packaged SONICOM hash registry.
        """
        if provider == "sonicom":
            return SonicomBaseDataset._download(self, provider_dir, provider=provider, **dataset_kwargs)
        return AKTZipBaseDataset._download(self, provider_dir, provider=provider, **dataset_kwargs)

    def _validate_params(self, **dataset_kwargs) -> None:
        """Validate HUTUBS-specific parameters.

        Parameters
        ----------
        **dataset_kwargs : dict
            Parameters to validate. Expected keys: ``subject`` and ``kind``.

        Raises
        ------
        TypeError
            If ``subject`` is not an integer.
        ValueError
            If ``subject`` or ``kind`` is out of range.
        """
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
        """Construct the ingest-ready SOFA file name for HUTUBS.

        Returns
        -------
        str
            File name in format ``pp{subject}_HRIRs_{kind}.sofa``.
        """
        return f"pp{dataset_kwargs['subject']}_HRIRs_{dataset_kwargs['kind']}.sofa"
