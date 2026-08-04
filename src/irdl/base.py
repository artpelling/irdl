"""Base Dataset class and conversion utilities for IRDL.

This module provides the BaseDataset abstract base class which serves as the common interface
for all Dataset implementations. Each Dataset subclass must implement:

- validate_params()
- download()
- ingest()
- _source_filename()

The BaseDataset class handles:

- Common parameter extraction
- Path construction
- Cache checking
- Output format conversion from SOFA
"""

import os
import shutil
from abc import ABC, abstractmethod
from enum import StrEnum
from inspect import isabstract
from pathlib import Path
from types import ModuleType

import h5py as h5
import netCDF4
import numpy as np
import pyfar as pf
import sofar as sf

from irdl.cache import IRDL_CACHE_DIR
from irdl.logging import logger
from irdl.utils import _link_or_copy, _preserve_permissions

_SOFA_FIR_E_DIMS = 4
DEFAULT_CHUNK_SIZE = 256


class DatasetCategory(StrEnum):
    """Categories for grouping datasets."""

    ROOM_IMPULSE_RESPONSES = "room_impulse_responses"
    HEAD_RELATED_IMPULSE_RESPONSES = "head_related_impulse_responses"


class BaseDataset(ABC):
    """Abstract base class providing common interface for all Dataset implementations.

    Attributes
    ----------
    name : str
        Unique identifier for the Dataset.
    doi : str
        Digital Object Identifier for the Dataset.

    Methods
    -------
    _validate_params(**dataset_kwargs)
        Validate dataset-specific parameters (including output_format).
    _source_filename(**dataset_kwargs) -> str
        Construct the raw input filename with extension.
    _download(**dataset_kwargs) -> Path
        Download and return Path to raw file.
    _process(provider_artifact: Path, ingest_path: Path, **_dataset_kwargs) -> Path:
        Post-process downloaded file if needed
    _ingest(ingest_artifact: Path, sofa_path: Path) -> Path
        Write processed/provider artifact to an on-disk SOFA file.
    get() @classmethod
        Public entry point. Uses explicit type signature for CLI auto-generation.
    """

    name: str
    doi: str | None
    source_url: str | None = None
    _chunk_size = DEFAULT_CHUNK_SIZE

    # Default docstring prefix for all get() classmethods
    _get_doc_prefix = """Download {name} dataset.

Parameters
----------
cache_dir : str
    Cache directory for downloads. Defaults is the OS user cache directory.
    This default can be overridden by setting `IRDL_CACHE_DIR` environment variable.
export_dir : str, optional
    Directory for final output. If specified, the data will be exported to <export_dir/{name}/>. Else, it remains in
    <cache_dir/output/>.
output_format : str
    Output format: 'pyfar', 'numpy', 'hdf5', 'sofa', or 'raw'.
"""

    def __init_subclass__(cls, **dataset_kwargs) -> None:
        """Initialize subclass with automatic docstring composition for get() classmethod."""
        super().__init_subclass__(**dataset_kwargs)
        # Automatically compose docstrings for get() classmethod
        if hasattr(cls, "get") and hasattr(cls, "name"):
            # Get the underlying function of the classmethod
            get_func = cls.get.__func__
            # Get the first line of the class docstring for the summary
            class_doc = cls.__doc__ or ""
            doc_lines = class_doc.strip().split("\n") if class_doc.strip() else []
            summary_line = doc_lines[0] if doc_lines else ""
            # Add a DOI only when the provider actually assigns one.
            identifier_line = f"DOI: https://doi.org/{cls.doi}" if cls.doi else None
            if identifier_line is None and cls.source_url:
                identifier_line = f"Source: {cls.source_url}"
            # Format prefix with class attributes
            prefix = BaseDataset._get_doc_prefix.format(name=cls.name.upper(), doi=cls.doi)
            # If class has a docstring with a summary, replace the first line of prefix
            if summary_line:
                # Split prefix into lines and replace the first line
                prefix_lines = prefix.split("\n")
                prefix_lines[0] = summary_line
                # Insert a verified provider identifier after the summary.
                if identifier_line:
                    prefix_lines.insert(1, "")
                    prefix_lines.insert(2, identifier_line)
                prefix = "\n".join(prefix_lines)
            # Get subclass-specific docstring (from the base class _get method)
            suffix = get_func.__doc__ or ""
            # Combine: prefix + suffix
            full_doc = prefix
            if suffix:
                full_doc += suffix
            get_func.__doc__ = full_doc

    def _get(
        self,
        cache_dir: Path | str | None,
        export_dir: Path | str | None,
        output_format: str,
        **dataset_kwargs,
    ) -> dict | Path | None:
        """Internal implementation of Dataset retrieval.

        Parameters
        ----------
        cache_dir : :class:`pathlib.Path` or str or None
            Cache directory for downloads.
        export_dir : :class:`pathlib.Path` or str or None
            Directory for final output. Default is None (stays in cache_dir).
        output_format : str
            Output format: 'pyfar', 'numpy', 'hdf5', 'sofa', or 'raw'.
        **dataset_kwargs : dict
            Dataset-specific parameters.

        Returns
        -------
        dict or :class:`pathlib.Path`
            For 'pyfar' / 'numpy': a dict of in-memory objects.
            For 'sofa' / 'hdf5' / 'raw': a :class:`pathlib.Path` to the file on disk.
        """  # noqa: D401
        # Validate common parameters
        if output_format not in ("pyfar", "hdf5", "numpy", "sofa", "raw"):
            msg = "output_format must be one of 'pyfar', 'hdf5', 'numpy', 'sofa', 'raw'"
            raise ValueError(msg)

        # Validate dataset-specific parameters (including output_format)
        logger.debug(f"Validating parameters for {self.name}")
        self._validate_params(output_format=output_format, **dataset_kwargs)

        # Set up and sanitize path variables
        cache_dir = (IRDL_CACHE_DIR if cache_dir is None else Path(cache_dir)) / self.name.upper()
        export_dir = None if export_dir is None else Path(export_dir)
        output_dir = cache_dir / "output" if export_dir is None else export_dir / self.name.upper()
        provider_dir = cache_dir / "provider"
        source_filename = self._source_filename(**dataset_kwargs)
        output_path = self._output_path(output_dir, source_filename, output_format)
        ingest_path = cache_dir / "ingest" / source_filename

        # Special handling for raw output format
        if output_format == "raw":
            provider_artifact = self.download(provider_dir, **dataset_kwargs)
            if export_dir is None:
                return provider_artifact
            return self._export_raw(provider_artifact, export_dir)

        # Early exit if output file already exists (not applicable for raw format, handled above)
        if output_path is not None and output_path.exists():
            logger.info(f"Output file already exists at {output_path}, skipping download and conversion.")
            return output_path

        sofa_path = self._output_path(cache_dir / "output", source_filename, "sofa")
        if sofa_path.exists():
            logger.info(f"Cache hit: {sofa_path}.")
        else:
            if ingest_path.exists():
                logger.info(f"Ingestible file already exists at {ingest_path}, skipping download and processing.")
                ingest_artifact = ingest_path
            else:
                provider_artifact = self.download(provider_dir, **dataset_kwargs)
                ingest_artifact = self.process(provider_artifact, ingest_path, **dataset_kwargs)
            logger.debug(f"Ingesting {ingest_artifact} to SOFA file {sofa_path}")
            with logger.spin(f"Writing SOFA {sofa_path.name}..."):
                self._ingest(ingest_artifact, sofa_path, **dataset_kwargs)
            with logger.spin(f"Verifying SOFA conventions for {sofa_path.name}..."):
                self._verify_sofa_convention(sofa_path)

        return self._to_output(output_format, sofa_path, output_path)

    def _verify_sofa_convention(self, sofa_path: Path) -> None:
        """Verify SOFA convention through SofaStream."""
        try:
            with sf.SofaStream(sofa_path) as sofa, logger.as_stdout:
                sofa.verify(issue_handling="raise", mode="read")
                # the following lines can be removed once/if
                # SofaStream.upgrade_conventions exists in upstream sofar
                convention = sofa.GLOBAL_SOFAConventions
                version = sofa.GLOBAL_SOFAConventionsVersion
            with logger.as_stdout:
                sf.Sofa(convention, version=version, verify=False).upgrade_convention(verify=False)
        except (OSError, ValueError) as error:
            logger.error(
                f"SOFA convention not satisfied!\n{error}\n"
                "See https://sofar.readthedocs.io/en/stable/resources/conventions.html#conventions for details."
            )
            raise

    @abstractmethod
    def _validate_params(self, **dataset_kwargs) -> None:
        """Validate dataset-specific parameters.

        Override in subclass. This method receives dataset-specific parameters
        plus ``output_format`` (so subclasses can forbid invalid output_format /
        dataset-parameter combinations).

        Parameters
        ----------
        **dataset_kwargs : dict
            Dataset-specific parameters to validate, including ``output_format``.

        Raises
        ------
        ValueError
            If any parameter is invalid.
        """

    @abstractmethod
    def _source_filename(self, **dataset_kwargs) -> str:
        """Construct the ingest-ready filename with extension for the dataset.

        Override in subclass.

        This name is canonical: ``_get``
        treats the existence of that path as proof that download *and*
        processing are already done (if so it skips both and ingests the file
        directly). The name therefore must match the file that actually
        ends up on disk after ``_download`` + ``_process``: i.e. the *processed*
        file (merged/extracted), which is not necessarily the raw download.

        Parameters
        ----------
        **dataset_kwargs : dict
            Dataset-specific parameters used to construct the filename.

        Returns
        -------
        str
            The ingest-ready filename including extension (e.g., "A1.h5",
            "FABIAN_HRIR_measured_HATO_0.sofa").
        """

    def download(self, provider_dir: Path, **dataset_kwargs) -> Path:
        """Download raw files and return Path to the primary artifact.

        This method wraps _download to enforce provider_dir existence for all subclasses.

        Parameters
        ----------
        provider_dir : :class:`pathlib.Path`
            Target path where the file(s) should be downloaded to.
            For single-file providers, this may be the file path itself.
            For multi-file providers, this may be a directory where files are placed.
        **dataset_kwargs : dict
            Dataset-specific parameters.

        Returns
        -------
        provider_artifact : :class:`pathlib.Path`
            Path to the downloaded artifact on disk (file or directory).
        """
        provider_dir.mkdir(exist_ok=True, parents=True)
        return self._download(provider_dir, **dataset_kwargs)

    @abstractmethod
    def _download(self, provider_dir: Path, **dataset_kwargs) -> Path:
        """Concrete download logic. Override in subclass."""

    def process(self, provider_artifact: Path, ingest_path: Path, **dataset_kwargs) -> Path:
        """Post-process downloaded file if needed.

        This method wraps _process to enforce ingest_dir existence for all subclasses.

        Parameters
        ----------
        provider artifact : Path
            Path to the freshly downloaded file (or download directory).
        ingest_path : :class:`pathlib.Path`
            Path to the ingestible file in the ingest directory.
        **dataset_kwargs : dict
            Dataset-specific parameters passed through to ``_process``.

        Returns
        -------
        ingest_path : Path
            The processed, ingest-ready file at ``ingest_path``.
        """
        ingest_path.parent.mkdir(parents=True, exist_ok=True)
        return self._process(provider_artifact, ingest_path, **dataset_kwargs)

    def _process(self, provider_artifact: Path, ingest_path: Path, **_dataset_kwargs) -> Path:
        """Post-process downloaded file if needed.

        Override in subclass to extract, merge, or otherwise transform the downloaded data. Write
        the processed, ingest-ready file to ``ingest_path`` and return it.

        The default implementation promotes the provider file to the ingest
        stage. If the provider path is a file and differs from the ingest path,
        it creates a hard link (or falls back to a copy) so the ingest file
        exists.

        Parameters
        ----------
        provider artifact : Path
            Path to the freshly downloaded file (or download directory).
        ingest_path : :class:`pathlib.Path`
            Path to the ingestible file in the ingest directory.
        **dataset_kwargs : dict
            Dataset-specific parameters (unused by the default implementation).

        Returns
        -------
        ingest_path : Path
            The processed, ingest-ready file at ``ingest_path``.
        """
        if provider_artifact.is_file():
            try:
                os.link(provider_artifact, ingest_path)
            except OSError:
                shutil.copy2(provider_artifact, ingest_path)
            return ingest_path
        msg = (
            "BaseDataset._process can only handle single files."
            "Override _process with special implementation in subclass."
        )
        raise NotImplementedError(msg)

    def _ingest(self, ingest_artifact: Path, sofa_path: Path, **_dataset_kwargs) -> Path:
        """Ingest an artifact to a retained SOFA file.

        SOFA ingest-ready artifacts can be promoted directly. Other Datasets
        override this method with a Dataset-specific writer.
        """
        if ingest_artifact.suffix == ".sofa":
            return _link_or_copy(ingest_artifact, sofa_path)
        msg = f"{type(self).__name__} must implement _ingest"
        raise NotImplementedError(msg)

    def _output_path(self, output_dir: Path, source_filename: str, output_format: str) -> Path | None:
        """Return the canonical Path where a file-based output would be written.

        Returns None for formats ('pyfar', 'numpy', 'raw'). Constructs the Path based on filename,
        directory target and output format.

        Parameters
        ----------
        output_dir : Path
            The output directory. Either cache_dir/output, export_dir, or export_dir/raw.
        source_filename : str
            The name of the ingestible file. Constructed with _source_filename
        output_format : str
            One of 'pyfar', 'numpy', 'hdf5', 'sofa', 'raw'.

        Returns
        -------
        Path or None
            Canonical output path, or None for in-memory formats.
        """
        match output_format:
            case "numpy" | "pyfar" | "raw":
                return None
            case "sofa":
                suff = ".sofa"
            case "hdf5":
                suff = ".h5"
        return (output_dir / Path(source_filename).stem).with_suffix(suff)

    def _export_raw(self, provider_artifact: Path, export_dir: Path) -> Path:
        """Export raw provider artifact to export directory.

        For file artifacts, copies the file with its actual name.
        For directory artifacts, copies all contents to the output base directory.
        Raises ValueError if provider_artifact is neither a file nor a directory.

        Parameters
        ----------
        provider_artifact : Path
            Path to the downloaded artifact (file or directory).
        export_dir : Path
            Target export directory.

        Returns
        -------
        Path
            Path to the exported file or directory.
        """
        output_base = export_dir / self.name.upper() / "raw"
        output_base.mkdir(exist_ok=True, parents=True)

        if provider_artifact.is_file():
            output_path = output_base / provider_artifact.name
            if not output_path.exists():
                shutil.copy2(provider_artifact, output_path)
            return output_path
        if provider_artifact.is_dir():
            shutil.copytree(provider_artifact, output_base, dirs_exist_ok=True)
            return output_base
        msg = f"Provider artifact must be a file or directory, but {self.name} returned: {provider_artifact}"
        raise ValueError(msg)

    def _to_output(self, output_format: str, sofa_path: Path, output_path: Path | None) -> dict | Path:
        """Convert the retained SOFA file to the requested output format.

        Parameters
        ----------
        output_format : str
            One of "pyfar", "numpy", "hdf5", "sofa".
        sofa_path : :class:`pathlib.Path`
            Retained SOFA file to convert from.
        output_path : :class:`pathlib.Path` or None
            Path where file-based outputs should be written.

        Returns
        -------
        dict or :class:`pathlib.Path`
            Output depends on output_format:
            - "pyfar" : dict of :class:`pyfar.Signal` and :class:`pyfar.Coordinates` objects
            - "numpy" : dict of :class:`numpy.ndarray` arrays
            - "hdf5" : :class:`pathlib.Path` to .h5 file
            - "sofa" : :class:`pathlib.Path` to .sofa file
        """
        if output_format == "sofa":
            return self._to_sofa(sofa_path, output_path)
        if output_format == "hdf5":
            return self._to_hdf5(sofa_path, output_path)
        if output_format in ("pyfar", "numpy"):
            logger.info(f"Loading SOFA file for {output_format} conversion.")
            with logger.spin(f"Loading {sofa_path.name}..."), logger.as_stdout:
                sofa = sf.read_sofa(sofa_path, verify=False, verbose=True)
            if output_format == "pyfar":
                return self._to_pyfar(sofa)
            return self._to_numpy(sofa)
        msg = f"Unknown output_format: {output_format}"
        raise ValueError(msg)

    def _to_pyfar(self, sofa: sf.Sofa) -> dict:
        """Convert sofar.Sofa to dict of pyfar objects.

        Parameters
        ----------
        sofa : :class:`sofar.Sofa`
            SOFA object to convert.

        Returns
        -------
        dict
            Dictionary with keys:
            - "impulse_response" : :class:`pyfar.Signal`
            - "source_coordinates" : :class:`pyfar.Coordinates`
            - "receiver_coordinates" : :class:`pyfar.Coordinates`
        """
        return dict(
            zip(
                ("impulse_response", "source_coordinates", "receiver_coordinates"),
                pf.io.convert_sofa(sofa),
                strict=True,
            )
        )

    def _to_numpy(self, sofa: sf.Sofa) -> dict:
        """Convert sofar.Sofa to dict of numpy arrays.

        Parameters
        ----------
        sofa : :class:`sofar.Sofa`
            SOFA object to convert.

        Returns
        -------
        dict
            Dictionary with keys:
            - "impulse_response" : :class:`numpy.ndarray`
            - "source_coordinates" : :class:`numpy.ndarray`
            - "receiver_coordinates" : :class:`numpy.ndarray`
            - "sampling_rate" : float
        """
        return {
            "impulse_response": np.array(sofa.Data_IR),
            "source_coordinates": np.array(sofa.SourcePosition),
            "receiver_coordinates": np.array(sofa.ReceiverPosition),
            "sampling_rate": float(sofa.Data_SamplingRate),
        }

    def _to_sofa(self, sofa_path: Path, output_path: Path) -> Path:
        """Return or export the retained SOFA file."""
        if output_path == sofa_path:
            logger.info(f"Returning cached SOFA file {sofa_path}.")
            return sofa_path
        logger.info(f"Exporting SOFA file to {output_path}.")
        return _link_or_copy(sofa_path, output_path)

    def _to_hdf5(self, sofa_path: Path, output_path: Path) -> Path:
        """Convert a SOFA file to IRDL HDF5 without loading all IR data."""
        chunk_size = int(self._chunk_size)
        if chunk_size <= 0:
            msg = "_chunk_size must be > 0"
            raise ValueError(msg)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with (
            logger.spin(f"Writing HDF5 {output_path.name}..."),
            netCDF4.Dataset(sofa_path, "r") as sofa,
            h5.File(output_path, "w") as f,
        ):
            data_group = f.create_group("data")
            ir = sofa.variables["Data.IR"]
            has_single_emitter = len(ir.shape) == _SOFA_FIR_E_DIMS and ir.shape[-1] == 1
            ir_shape = ir.shape[:-1] if has_single_emitter else ir.shape
            output_ir = data_group.create_dataset("impulse_response", shape=ir_shape, dtype=ir.dtype)
            for start in range(0, ir_shape[0], chunk_size):
                row_slice = slice(start, min(start + chunk_size, ir_shape[0]))
                output_ir[row_slice] = ir[row_slice, :, :, 0] if has_single_emitter else ir[row_slice]

            loc_group = data_group.create_group("location")
            loc_group.create_dataset("source", data=sofa.variables["SourcePosition"][:])
            receiver = sofa.variables["ReceiverPosition"][:]
            loc_group.create_dataset("receiver", data=np.squeeze(receiver))

            meta_group = f.create_group("metadata")
            meta_group.create_dataset("sampling_rate", data=sofa.variables["Data.SamplingRate"][:])
            if "RoomTemperature" in sofa.variables:
                meta_group.create_dataset("temperature", data=sofa.variables["RoomTemperature"][:])
            if "SpeedOfSound" in sofa.variables:
                meta_group.create_dataset("c0", data=sofa.variables["SpeedOfSound"][:])
            if "Humidity" in sofa.variables:
                meta_group.create_dataset("humidity", data=sofa.variables["Humidity"][:])
        _preserve_permissions(sofa_path, output_path)
        return output_path


def _get_dataset_classes(module: ModuleType) -> list[type]:
    """Return concrete BaseDataset subclasses exported by module."""
    dataset_classes: list[type] = []
    for name in dir(module):
        obj = getattr(module, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, BaseDataset)
            and not isabstract(obj)
            and hasattr(obj, "name")
            and hasattr(obj, "doi")
        ):
            dataset_classes.append(obj)
    return dataset_classes
