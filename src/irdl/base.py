"""Base Dataset class and conversion utilities for IRDL.

This module provides the BaseDataset abstract base class which serves as the common interface
for all Dataset implementations. Each Dataset subclass must implement:

- _validate_params()
- _download()
- _ingest()
- _source_filename()

The BaseDataset class handles:

- Common parameter extraction
- Provider selection
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
import numpy as np
import pyfar as pf
import sofar as sf

from irdl.cache import IRDL_CACHE_DIR
from irdl.logging import get_logger, sofar_logger
from irdl.utils import _fits_in_memory


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
        Digital Object Identifier for the Dataset's canonical Provider.
    providers : tuple[str, ...]
        Provider names in priority order for ``provider="auto"`` selection.
    canonical_provider : str
        The authoritative Provider for the Dataset. This Provider shares the
        Dataset-level DOI and is the only Provider eligible for
        ``output_format="raw"``.

    Methods
    -------
    _validate_params(**dataset_kwargs)
        Validate dataset-specific parameters, including provider /
        output-format combinations the Dataset wants to forbid.
    _source_filename(**dataset_kwargs) -> str
        Construct the canonical ingest-ready filename.
    _download(provider_dir, provider, **dataset_kwargs) -> Path
        Download and return the Provider artifact for one concrete Provider.
    _process(provider_artifact, ingest_path, **dataset_kwargs) -> Path
        Post-process downloaded files into a single ingest-ready artifact when
        needed.
    _ingest(ingest_path) -> sofar.Sofa
        Convert an ingest-ready file to the internal SOFA representation.
    get() @classmethod
        Public entry point. Uses explicit type signatures for CLI
        auto-generation.
    """

    name: str
    doi: str
    providers: tuple[str, ...]
    canonical_provider: str

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
provider : str
    Provider selection. Use 'auto' (default) to try providers in the documented order.
"""

    def __init__(self) -> None:
        """Initialize per-dataset logging."""
        self.logger = get_logger(self.name.upper(), style=self._logger_style())

    def _logger_style(self) -> str:
        """Return the Rich style for this Dataset's log source."""
        if getattr(self, "_category", None) == DatasetCategory.ROOM_IMPULSE_RESPONSES:
            return "bold cyan"
        if getattr(self, "_category", None) == DatasetCategory.HEAD_RELATED_IMPULSE_RESPONSES:
            return "bold magenta"
        return "bold white"

    def __init_subclass__(cls, **dataset_kwargs) -> None:
        """Initialize subclass with automatic docstring composition for get() classmethod."""
        super().__init_subclass__(**dataset_kwargs)

        if hasattr(cls, "doi") and not hasattr(cls, "canonical_provider"):
            msg = f"{cls.__name__} must define canonical_provider"
            raise TypeError(msg)
        if hasattr(cls, "canonical_provider") and not hasattr(cls, "providers"):
            cls.providers = (cls.canonical_provider,)
        if (
            hasattr(cls, "providers")
            and hasattr(cls, "canonical_provider")
            and cls.canonical_provider not in cls.providers
        ):
            msg = f"{cls.__name__}.canonical_provider must appear in {cls.__name__}.providers"
            raise TypeError(msg)

        if hasattr(cls, "get") and hasattr(cls, "name") and hasattr(cls, "doi"):
            get_func = cls.get.__func__
            class_doc = cls.__doc__ or ""
            doc_lines = class_doc.strip().split("\n") if class_doc.strip() else []
            summary_line = doc_lines[0] if doc_lines else ""
            doi_url = f"https://doi.org/{cls.doi}"
            doi_cli_line = f"DOI: {doi_url}"
            prefix = BaseDataset._get_doc_prefix.format(name=cls.name.upper(), doi=cls.doi)
            if summary_line:
                prefix_lines = prefix.split("\n")
                prefix_lines[0] = summary_line
                prefix_lines.insert(1, "")
                prefix_lines.insert(2, doi_cli_line)
                prefix = "\n".join(prefix_lines)
            suffix = get_func.__doc__ or ""
            full_doc = prefix
            if suffix:
                full_doc += suffix
            get_func.__doc__ = full_doc

    def _get(
        self,
        cache_dir: Path | str | None,
        export_dir: Path | str | None,
        output_format: str,
        provider: str,
        **dataset_kwargs,
    ) -> dict | Path | None:
        """Retrieve Dataset data.

        Parameters
        ----------
        cache_dir : :class:`pathlib.Path` or str or None
            Cache directory for downloads.
        export_dir : :class:`pathlib.Path` or str or None
            Directory for final output. Default is None (artifact stays in the
            Cache Directory).
        output_format : str
            Output format: 'pyfar', 'numpy', 'hdf5', 'sofa', or 'raw'.
        provider : str
            Provider selection. ``"auto"`` tries provider-native Providers first,
            then ingest-derived Providers. Any explicit Provider name disables
            cross-Provider fallback. ``"raw"`` is restricted to the canonical
            Provider.
        **dataset_kwargs : dict
            Dataset-specific parameters.

        Returns
        -------
        dict or :class:`pathlib.Path`
            For 'pyfar' / 'numpy': a dict of in-memory objects.
            For 'sofa' / 'hdf5' / 'raw': a :class:`pathlib.Path` to the file on
            disk.
        """
        if output_format not in ("pyfar", "hdf5", "numpy", "sofa", "raw"):
            msg = "output_format must be one of 'pyfar', 'hdf5', 'numpy', 'sofa', 'raw'"
            raise ValueError(msg)

        self.logger.debug("Validating dataset parameters")
        self._validate_params(output_format=output_format, provider=provider, **dataset_kwargs)
        self._validate_provider_request(provider=provider, output_format=output_format)

        cache_dir = (IRDL_CACHE_DIR if cache_dir is None else Path(cache_dir)) / self.name.upper()
        export_dir = None if export_dir is None else Path(export_dir)
        output_dir = cache_dir / "output" if export_dir is None else export_dir / self.name.upper()
        source_filename = self._source_filename(**dataset_kwargs)
        output_path = self._output_path(output_dir, source_filename, output_format)
        ingest_path = cache_dir / "ingest" / source_filename

        selected_provider, mode = self._select_provider(
            provider=provider,
            output_format=output_format,
            **dataset_kwargs,
        )
        self.logger.info(
            "provider=%r requested=%r output_format=%r -> %s path",
            selected_provider,
            provider,
            output_format,
            mode,
        )
        provider_dir = cache_dir / "provider" / selected_provider

        try:
            result = self._get_from_provider(
                provider_name=selected_provider,
                mode=mode,
                provider_dir=provider_dir,
                ingest_path=ingest_path,
                output_path=output_path,
                export_dir=export_dir,
                output_format=output_format,
                **dataset_kwargs,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            first_error = exc
            if provider != "auto":
                msg = (
                    f"Provider {selected_provider!r} failed for {self.name.upper()} "
                    f"(output_format={output_format!r}): {exc}"
                )
                raise RuntimeError(msg) from exc
        else:
            return result

        provider_native_providers, ingest_derived_providers = self._provider_candidates(
            output_format=output_format,
            **dataset_kwargs,
        )
        attempted = [
            name for name in [*provider_native_providers, *ingest_derived_providers] if name == selected_provider
        ]
        errors = [f"- {selected_provider} ({mode}): {first_error}"]

        if provider == "auto":
            retry_order = [
                (name, "provider-native") for name in provider_native_providers if name != selected_provider
            ] + [(name, "ingest-derived") for name in ingest_derived_providers if name != selected_provider]
            failed_provider = selected_provider
            failed_mode = mode
            failed_error = first_error
            for provider_name, candidate_mode in retry_order:
                attempted.append(provider_name)
                self.logger.warning(
                    "provider=%r output_format=%r failed on %s path: %s. Retrying with provider=%r -> %s path.",
                    failed_provider,
                    output_format,
                    failed_mode,
                    failed_error,
                    provider_name,
                    candidate_mode,
                )
                provider_dir = cache_dir / "provider" / provider_name
                try:
                    result = self._get_from_provider(
                        provider_name=provider_name,
                        mode=candidate_mode,
                        provider_dir=provider_dir,
                        ingest_path=ingest_path,
                        output_path=output_path,
                        export_dir=export_dir,
                        output_format=output_format,
                        **dataset_kwargs,
                    )
                except (OSError, RuntimeError, ValueError) as inner_exc:
                    errors.append(f"- {provider_name} ({candidate_mode}): {inner_exc}")
                    failed_provider = provider_name
                    failed_mode = candidate_mode
                    failed_error = inner_exc
                    continue
                return result

        msg = (
            f"Unable to retrieve {self.name.upper()} with provider='auto' for output_format={output_format!r}.\n"
            f"Attempt order: {' -> '.join(attempted)}\n"
            f"Failures:\n" + "\n".join(errors)
        )
        raise RuntimeError(msg)

    def _validate_provider_request(self, *, provider: str, output_format: str) -> None:
        """Validate provider selection against common rules.

        Parameters
        ----------
        provider : str
            Requested Provider name or ``"auto"``.
        output_format : str
            Requested Output Format.

        Raises
        ------
        ValueError
            If the Provider name is unknown, or ``output_format="raw"`` is
            combined with a non-canonical explicit Provider.
        """
        valid = {"auto", *self.providers}
        if provider not in valid:
            msg = f"provider must be one of {sorted(valid)}"
            raise ValueError(msg)
        if output_format == "raw" and provider not in {"auto", self.canonical_provider}:
            msg = (
                "raw output_format is only supported with provider='auto' or the canonical provider "
                f"({self.canonical_provider!r})"
            )
            raise ValueError(msg)

    def _provider_candidates(self, output_format: str, **dataset_kwargs) -> tuple[list[str], list[str]]:
        """Return provider-native and ingest-derived candidates in priority order.

        The first list contains Providers that can serve the requested Output
        Format natively from their Provider artifact. The second contains
        Providers that can satisfy it only after ingest or conversion.
        """
        provider_native: list[str] = []
        ingest_derived: list[str] = []
        for provider in self.providers:
            if not self._provider_available(provider, output_format=output_format, **dataset_kwargs):
                continue
            if output_format in self._direct_output_formats(provider, **dataset_kwargs):
                provider_native.append(provider)
            elif output_format != "raw" and self._can_materialize_from_provider(
                provider, output_format=output_format, **dataset_kwargs
            ):
                ingest_derived.append(provider)
        return provider_native, ingest_derived

    def _select_provider(self, *, provider: str, output_format: str, **dataset_kwargs) -> tuple[str, str]:
        """Select Provider and retrieval mode.

        Returns
        -------
        tuple[str, str]
            ``(provider_name, mode)`` where ``mode`` is ``"provider-native"``
            when the requested Output Format is available directly from the
            Provider artifact and ``"ingest-derived"`` when IRDL must ingest or
            convert the artifact first.
        """
        if output_format == "raw":
            self.logger.debug("Raw output uses canonical provider %r", self.canonical_provider)
            return self.canonical_provider, "provider-native"

        provider_native_providers, ingest_derived_providers = self._provider_candidates(
            output_format=output_format,
            **dataset_kwargs,
        )

        if provider == "auto":
            self.logger.debug(
                "Provider candidates for output_format=%r: provider-native=%s, ingest-derived=%s",
                output_format,
                provider_native_providers or ["<none>"],
                ingest_derived_providers or ["<none>"],
            )
            if provider_native_providers:
                return provider_native_providers[0], "provider-native"
            if ingest_derived_providers:
                return ingest_derived_providers[0], "ingest-derived"
            msg = (
                f"No provider can satisfy output_format={output_format!r} for {self.name.upper()}. "
                "Need either a provider-native SOFA-capable provider or an ingest-capable non-SOFA provider."
            )
            raise ValueError(msg)

        if not self._provider_available(provider, output_format=output_format, **dataset_kwargs):
            msg = f"provider {provider!r} is not available for the requested parameters"
            raise ValueError(msg)
        if output_format in self._direct_output_formats(provider, **dataset_kwargs):
            return provider, "provider-native"
        if self._can_materialize_from_provider(provider, output_format=output_format, **dataset_kwargs):
            return provider, "ingest-derived"
        msg = f"provider {provider!r} cannot satisfy output_format={output_format!r} for the requested parameters"
        raise ValueError(msg)

    def _get_from_provider(
        self,
        *,
        provider_name: str,
        mode: str,
        provider_dir: Path,
        ingest_path: Path,
        output_path: Path | None,
        export_dir: Path | None,
        output_format: str,
        **dataset_kwargs,
    ) -> dict | Path | None:
        """Retrieve Dataset data from one concrete Provider.

        This method executes one Provider path after selection is complete. It
        handles raw export, provider-native SOFA-backed materialization,
        ingest reuse, Dataset-specific processing, and final conversion.
        """
        if output_format != "raw" and output_path is not None and output_path.exists():
            self.logger.info("Output cache hit: %s", output_path)
            return output_path

        provider_artifact = self.download(provider_dir, provider=provider_name, **dataset_kwargs)
        result: dict | Path | None

        if output_format == "raw":
            result = provider_artifact if export_dir is None else self._export_raw(provider_artifact, export_dir)
        elif mode == "provider-native":
            result = self._materialize_direct_output(provider_artifact, output_format, output_path)
        elif self._provider_artifact_format(provider_name, **dataset_kwargs) == "sofa":
            result = self._finalize_sofa_provider_artifact(provider_artifact, output_format, output_path)
        else:
            result = self._materialize_via_ingest(
                provider_artifact,
                ingest_path,
                provider_name=provider_name,
                output_format=output_format,
                output_path=output_path,
                **dataset_kwargs,
            )

        return result

    def _finalize_sofa_provider_artifact(
        self,
        provider_artifact: Path,
        output_format: str,
        output_path: Path | None,
    ) -> dict | Path | None:
        """Materialize output directly from a SOFA provider artifact."""
        if not _fits_in_memory(provider_artifact):
            self.logger.warning(
                "Conversion skipped for %s: dataset exceeds available memory; returning file path instead.",
                provider_artifact,
            )
            return provider_artifact
        self.logger.debug("Reading provider SOFA artifact %s directly", provider_artifact)
        sofa = sf.read_sofa(provider_artifact)
        return self._finalize_output(sofa, output_format, provider_artifact, output_path)

    def _materialize_via_ingest(
        self,
        provider_artifact: Path,
        ingest_path: Path,
        *,
        provider_name: str,
        output_format: str,
        output_path: Path | None,
        **dataset_kwargs,
    ) -> dict | Path | None:
        """Materialize output by processing a provider artifact into the ingest stage."""
        if ingest_path.exists():
            self.logger.info("Ingest cache hit: %s", ingest_path)
        else:
            self.logger.debug("Processing provider artifact %s -> %s", provider_artifact, ingest_path)
            ingest_path = self.process(provider_artifact, ingest_path, provider=provider_name, **dataset_kwargs)

        if not _fits_in_memory(ingest_path):
            self.logger.warning(
                "Conversion skipped for %s: dataset exceeds available memory; returning file path instead.",
                ingest_path,
            )
            return ingest_path

        self.logger.debug("Reading ingest artifact %s into SOFA", ingest_path)
        sofa = self._ingest(ingest_path)
        return self._finalize_output(sofa, output_format, ingest_path, output_path)

    def _finalize_output(
        self,
        sofa: sf.Sofa,
        output_format: str,
        source_path: Path,
        output_path: Path | None,
    ) -> dict | Path | None:
        """Verify internal SOFA and convert to the requested Output Format.

        The verification step is shared across all Datasets so contributors get
        immediate feedback when a new ingest path produces an invalid SOFA
        object.
        """
        try:
            sofa.verify(issue_handling="raise")
            with sofar_logger.as_stdout:
                sofa.upgrade_convention()
        except ValueError:
            self.logger.exception(
                "SOFA convention not satisfied!\n"
                "See https://sofar.readthedocs.io/en/stable/resources/conventions.html#conventions for details."
            )
            return None

        self.logger.debug("Converting to %s format", output_format)
        return self._to_output(sofa, output_format, source_path, output_path)

    @abstractmethod
    def _validate_params(self, **dataset_kwargs) -> None:
        """Validate dataset-specific parameters."""

    @abstractmethod
    def _source_filename(self, **dataset_kwargs) -> str:
        """Construct the ingest-ready filename with extension for the dataset."""

    def _provider_available(self, provider: str, **_dataset_kwargs) -> bool:
        """Return True if a Provider can serve the requested parameters.

        Subclasses may override this to express Provider-specific availability
        constraints without moving ingest semantics into the Provider model.
        """
        return provider in self.providers

    def _provider_artifact_format(self, provider: str, **_dataset_kwargs) -> str:
        """Return the Provider-side artifact Data Format.

        Examples include ``"hdf5"``, ``"sofa"``, and ``"zip"``. This describes
        what the selected Provider serves before any IRDL processing.
        """
        if provider != self.canonical_provider:
            msg = f"{self.__class__.__name__} must override _provider_artifact_format for provider {provider!r}"
            raise NotImplementedError(msg)
        msg = f"{self.__class__.__name__} must define canonical provider artifact format"
        raise NotImplementedError(msg)

    def _direct_output_formats(self, provider: str, **dataset_kwargs) -> set[str]:
        """Return Output Formats available directly from Provider artifacts.

        The default implementation treats SOFA-backed Providers as directly able
        to satisfy ``output_format="sofa"``.
        """
        if self._provider_artifact_format(provider, **dataset_kwargs) == "sofa":
            return {"sofa"}
        return set()

    def _can_materialize_from_provider(self, provider: str, output_format: str, **dataset_kwargs) -> bool:
        """Return True if a Provider can satisfy an Output Format.

        This includes both direct materialization from the Provider artifact and
        indirect satisfaction via ingest/conversion.
        """
        artifact_format = self._provider_artifact_format(provider, **dataset_kwargs)
        if artifact_format == "sofa":
            return output_format in {"pyfar", "numpy", "hdf5", "sofa"}
        return self._can_ingest_provider(provider, **dataset_kwargs)

    def _can_ingest_provider(self, provider: str, **_dataset_kwargs) -> bool:
        """Return True if Provider artifacts can enter the ingest stage.

        The default rule allows only the canonical Provider. Datasets with extra
        ingest-capable Providers should override this method.
        """
        return provider == self.canonical_provider

    def download(self, provider_dir: Path, provider: str, **dataset_kwargs) -> Path:
        """Download raw Provider files and return the primary artifact.

        This wrapper ensures the Provider directory exists before delegating to
        ``_download``.
        """
        provider_dir.mkdir(exist_ok=True, parents=True)
        return self._download(provider_dir, provider=provider, **dataset_kwargs)

    @abstractmethod
    def _download(self, provider_dir: Path, provider: str, **dataset_kwargs) -> Path:
        """Concrete download logic. Override in subclass."""

    def process(self, provider_artifact: Path, ingest_path: Path, **dataset_kwargs) -> Path:
        """Post-process downloaded files into the ingest stage if needed.

        This wrapper ensures the ingest directory exists before delegating to
        ``_process``.
        """
        ingest_path.parent.mkdir(parents=True, exist_ok=True)
        return self._process(provider_artifact, ingest_path, **dataset_kwargs)

    def _process(self, provider_artifact: Path, ingest_path: Path, **_dataset_kwargs) -> Path:
        """Post-process downloaded files into one ingest-ready artifact.

        The default implementation promotes a single Provider file to the
        ingest stage by hard-linking or copying it. Multi-file or transforming
        Datasets should override this method.
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

    @abstractmethod
    def _ingest(self, ingest_path: Path) -> sf.Sofa:
        """Convert processed or raw file to sofar.Sofa object."""

    def _output_path(self, output_dir: Path, source_filename: str, output_format: str) -> Path | None:
        """Return the canonical path for a file-backed Output Format.

        Returns ``None`` for in-memory formats and for ``raw``, which always
        returns Provider-stage artifacts instead of Output-stage files.
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
        """Export a raw Provider artifact to the Export Directory.

        File artifacts keep their original Provider filename. Directory
        artifacts are copied recursively into ``<export_dir>/<DATASET>/raw``.
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

    def _materialize_direct_output(
        self,
        provider_artifact: Path,
        output_format: str,
        output_path: Path | None,
    ) -> dict | Path:
        """Materialize Output directly from Provider artifacts.

        This path is used for SOFA-backed Providers that can bypass the ingest
        stage for non-raw retrieval.
        """
        if output_format == "sofa":
            return self._copy_or_link(provider_artifact, output_path)
        if not _fits_in_memory(provider_artifact):
            self.logger.warning(
                "Conversion skipped for %s: dataset exceeds available memory; returning file path instead.",
                provider_artifact,
            )
            return provider_artifact
        sofa = sf.read_sofa(provider_artifact)
        return self._finalize_output(sofa, output_format, provider_artifact, output_path)

    def _copy_or_link(self, source: Path, target: Path) -> Path:
        """Copy or hard-link one file to a target path."""
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            return target
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
        return target

    def _to_output(self, sofa: sf.Sofa, output_format: str, ingest_path: Path, output_path: Path | None) -> dict | Path:
        """Convert sofar.Sofa to the requested output format."""
        if output_format == "pyfar":
            return self._to_pyfar(sofa)
        if output_format == "numpy":
            return self._to_numpy(sofa)
        if output_format == "sofa":
            return self._to_sofa(sofa, ingest_path, output_path)
        if output_format == "hdf5":
            return self._to_hdf5(sofa, ingest_path, output_path)
        msg = f"Unknown output_format: {output_format}"
        raise ValueError(msg)

    def _to_pyfar(self, sofa: sf.Sofa) -> dict:
        """Convert sofar.Sofa to dict of pyfar objects."""
        return dict(
            zip(
                ("impulse_response", "source_coordinates", "receiver_coordinates"),
                pf.io.convert_sofa(sofa),
                strict=True,
            )
        )

    def _to_numpy(self, sofa: sf.Sofa) -> dict:
        """Convert sofar.Sofa to dict of numpy arrays."""
        return {
            "impulse_response": np.array(sofa.Data_IR),
            "source_coordinates": np.array(sofa.SourcePosition),
            "receiver_coordinates": np.array(sofa.ReceiverPosition),
            "sampling_rate": float(sofa.Data_SamplingRate),
        }

    def _to_sofa(self, sofa: sf.Sofa, ingest_path: Path, output_path: Path) -> Path:  # noqa: ARG002
        """Write sofar.Sofa to file and return Path."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write_sofa(output_path, sofa)
        return output_path

    def _to_hdf5(self, sofa: sf.Sofa, ingest_path: Path, output_path: Path) -> Path:  # noqa: ARG002
        """Convert sofar.Sofa to HDF5 file and return Path."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5.File(output_path, "w") as f:
            data_group = f.create_group("data")
            data_group.create_dataset("impulse_response", data=sofa.Data_IR)

            loc_group = data_group.create_group("location")
            loc_group.create_dataset("source", data=sofa.SourcePosition)
            loc_group.create_dataset("receiver", data=sofa.ReceiverPosition)

            meta_group = f.create_group("metadata")
            meta_group.create_dataset("sampling_rate", data=sofa.Data_SamplingRate)

            if hasattr(sofa, "RoomTemperature"):
                meta_group.create_dataset("temperature", data=sofa.RoomTemperature)
            if hasattr(sofa, "SpeedOfSound"):
                meta_group.create_dataset("c0", data=sofa.SpeedOfSound)
            if hasattr(sofa, "Humidity"):
                meta_group.create_dataset("humidity", data=sofa.Humidity)

        return output_path


class SofaBaseDataset(BaseDataset):
    """Base class for Datasets whose ingest-ready format is already SOFA.

    The primary distinction is that ``output_format='sofa'`` can reuse the
    ingest-ready SOFA file directly instead of rewriting it through
    :func:`sofar.write_sofa`.
    """

    def _to_sofa(self, sofa: sf.Sofa, ingest_path: Path, output_path: Path) -> Path:  # noqa: ARG002
        """Copy sofar.Sofa file from ingest_dir and return Path."""
        return self._copy_or_link(ingest_path, output_path)

    def _ingest(self, ingest_path: Path) -> sf.Sofa:
        """Load SOFA file into sofar.Sofa object."""
        return sf.read_sofa(ingest_path)


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
