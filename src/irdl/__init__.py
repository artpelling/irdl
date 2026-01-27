"""Impulse Response Downloader (irdl): Download, unpack and process impulse response datasets."""

__author__ = "Art Pelling"
__date__ = "23 January 2026"
__version__ = "0.1.0"

from .downloader import CACHE_DIR as CACHE_DIR
from .fabian import get_fabian as get_fabian
from .miracle import get_miracle as get_miracle

__all__ = ["get_fabian", "get_miracle"]
