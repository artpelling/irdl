"""Impulse Response Downloader (irdl): Download, unpack and process impulse response datasets."""

import sys as _sys

from .base import _get_dataset_classes as _get_dataset_classes
from .datasets.aalto import MrtdDataset as MrtdDataset
from .datasets.akt import BrasRs8Dataset as BrasRs8Dataset
from .datasets.akt import FabianDataset as FabianDataset
from .datasets.akt import HutubsDataset as HutubsDataset
from .datasets.barilan import DechorateDataset as DechorateDataset
from .datasets.esat import MyriadDataset as MyriadDataset
from .datasets.iks import MirdDataset as MirdDataset
from .datasets.ista import MiracleDataset as MiracleDataset
from .datasets.ista import SrirachaDataset as SrirachaDataset
from .datasets.sonicom import AriDataset as AriDataset
from .datasets.sonicom import CipicDataset as CipicDataset
from .datasets.sonicom import SadieDataset as SadieDataset

__all__ = [dataset_class.__name__ for dataset_class in _get_dataset_classes(_sys.modules[__name__])]
