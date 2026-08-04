"""Impulse Response Downloader (irdl): Download, unpack and process impulse response datasets."""

import sys as _sys

from .aalto import ArniRoomImpulseResponseDataset as ArniRoomImpulseResponseDataset
from .aalto import MotusDataset as MotusDataset
from .aalto import MultiRoomTransitionDataset as MultiRoomTransitionDataset
from .akt import BrasRs8Dataset as BrasRs8Dataset
from .akt import FabianDataset as FabianDataset
from .akt import HutubsDataset as HutubsDataset
from .base import _get_dataset_classes as _get_dataset_classes
from .brno import ButReverbDbDataset as ButReverbDbDataset
from .esat import MyriadDataset as MyriadDataset
from .imperial import AceChallengeDataset as AceChallengeDataset
from .ista import MiracleDataset as MiracleDataset
from .ista import SrirachaDataset as SrirachaDataset
from .mit import MitImpulseResponseSurveyDataset as MitImpulseResponseSurveyDataset
from .york import OpenAirDataset as OpenAirDataset

__all__ = [dataset_class.__name__ for dataset_class in _get_dataset_classes(_sys.modules[__name__])]
