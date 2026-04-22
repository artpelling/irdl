import shutil
import warnings
from pathlib import Path

import psutil


def _fits_in_memory(file_path):
    """Check if a file can be loaded into available RAM.

    Needed for pyfar or numpy output formats, which load the entire dataset into memory.

    Parameters
    ----------
    file_path : Path
        Path to the HDF5 file.

    Returns
    -------
    fits : bool
        True if the file fits into available RAM with headroom.

    """
    file_size = file_path.stat().st_size
    available = psutil.virtual_memory().available
    if file_size < available * 0.9:  # Headroom
        return True
    else:
        warnings.warn(
            f"Dataset too large for available memory "
            f"({file_size / 1e9:.1f} GB needed, "
            f"{available / 1e9:.1f} GB available). "
            f"Returning HDF5 file path instead.",
            stacklevel=2,
        )
        return False


def _move_to_export_dir(cached_path, export_dir):
    """Move a file from the cache directory to a dedicated export directory.

    Parameters
    ----------
    cached_path : :class:`pathlib.Path`
    Path to the file in the cache directory.
    export_dir : :class:`str`, :class:`pathlib.Path`, or None
    Directory to move the file to. If ``None`` or identical to the file's
    parent directory, the file is not moved and ``cached_path`` is returned.

    Returns
    -------
    path : :class:`pathlib.Path`
    Path to the file, either in ``export_dir`` or unchanged if no move was needed.

    """
    # no export_dir specified
    if export_dir is None or Path(export_dir) == cached_path.parent:
        return cached_path
    dest = Path(export_dir) / cached_path.name
    # file exists already in export_dir
    if dest.exists():
        return dest
    # move file from cache to export_dir
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(cached_path, dest)
    return dest
