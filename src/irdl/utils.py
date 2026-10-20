"""Utility functions for IRDL."""

import json
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from importlib import resources
from pathlib import Path

import psutil

from irdl.logging import logger

_TMP_PREFIX = ".tmp-"


@cache
def load_hash_registry(dataset_name: str) -> dict[str, str]:
    """Load and validate a packaged Dataset hash registry.

    Parameters
    ----------
    dataset_name : str
        Canonical lowercase Dataset name used to resolve ``<dataset_name>.json``
        from ``irdl/registry/`` package data.

    Returns
    -------
    dict
        Flat mapping of provider file identifiers to ``sha256:...`` digests.
    """
    resource = resources.files("irdl").joinpath("registry", f"{dataset_name}.json")
    with resource.open("r", encoding="utf-8") as handle:
        registry = json.load(handle)
    _validate_hash_registry(dataset_name, registry)
    return registry


def _validate_hash_registry(dataset_name: str, registry: object) -> None:
    """Validate the common flat hash-registry schema.

    Parameters
    ----------
    dataset_name : str
        Dataset identifier used in error messages.
    registry : object
        Parsed JSON payload to validate.

    Raises
    ------
    ValueError
        If the registry does not match the required flat ``dict[str, str]`` schema.
    """
    if not isinstance(registry, dict):
        msg = f"Hash registry '{dataset_name}' must be a JSON object"
        raise TypeError(msg)
    for key, value in registry.items():
        if not isinstance(key, str) or not key or Path(key).is_absolute() or ".." in Path(key).parts:
            msg = f"Hash registry '{dataset_name}' contains invalid file key: {key!r}"
            raise ValueError(msg)
        if not isinstance(value, str) or not value.startswith("sha256:"):
            msg = f"Hash registry '{dataset_name}' contains invalid digest for {key!r}: {value!r}"
            raise ValueError(msg)


def _preserve_permissions(source_path: Path, target_path: Path) -> None:
    """Copy permission bits from one file to another.

    Parameters
    ----------
    source_path : Path
        Existing file whose permission bits should be reused.
    target_path : Path
        Existing file that should receive the same permission bits.
    """
    target_path.chmod(source_path.stat().st_mode & 0o777)


def _fits_in_memory(ingest_path: Path) -> bool:
    """Check if a file can be loaded into available RAM.

    Needed when an entire dataset is loaded into memory.

    Parameters
    ----------
    ingest_path : Path
        Path to the ingestable file.

    Returns
    -------
    fits : bool
        True if the file fits into available RAM with headroom.
    """
    file_size = ingest_path.stat().st_size
    available = psutil.virtual_memory().available
    if file_size < available * 0.9:
        return True
    logger.warning(
        f"Dataset too large for available memory "
        f"({file_size / 1e9:.1f} GB needed, "
        f"{available / 1e9:.1f} GB available). "
    )
    return False


def _link_or_copy(source_path: Path, target_path: Path) -> Path:
    """Hard-link a file, falling back to copy."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.parent.parent == source_path.parent.parent:
        try:
            logger.debug(f"Linking {source_path} to {target_path}.")
            with logger.spin(f"Linking {target_path.name}..."):
                os.link(source_path, target_path)
        except OSError as error:
            logger.debug(f"Linking failed: {error!r}")
        else:
            return target_path
    logger.debug(f"Copying {source_path} to {target_path}.")
    with logger.spin(f"Copying {target_path.name}..."):
        shutil.copy2(source_path, target_path)
    return target_path


def _remove_path(path: Path) -> None:
    """Remove a file or directory if it exists."""
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


@contextmanager
def _atomic_write(final_path: Path) -> Iterator[Path]:
    """Yield a temporary sibling of final_path and move it into place only if the block succeeds.

    Cache lookups treat an existing path as complete, so interrupted or failed
    writes must never leave anything at ``final_path``.
    """
    tmp_path = final_path.with_name(f"{_TMP_PREFIX}{final_path.name}")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    _remove_path(tmp_path)
    try:
        yield tmp_path
        # Some writers (e.g. SRIRACHA full plane) use their input in place and write nothing.
        if tmp_path.exists():
            tmp_path.replace(final_path)
    finally:
        _remove_path(tmp_path)
