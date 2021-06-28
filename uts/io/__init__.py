from .hdf5 import *
from .audio import *
from .matlab import *
from .darr import *

try:
    import pyedflib
    from .edf import *
except ImportError:
    pass

__all__ = ['open', 'save']

open = opendarr
save = savedarr
load = loaddarr
cache = cachedarr