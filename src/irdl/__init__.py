import pooch

from .fabian import get_fabian as get_fabian
from .miracle import get_miracle as get_miracle

CACHE_DIR = pooch.os_cache("irdl")
