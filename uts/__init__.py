# pylint: disable-msg=W0614,W0401,W0611,W0622
from __future__ import absolute_import
__docformat__ = 'restructuredtext'

# The version is written by setuptools-scm at build/install time and read back
# from the installed package metadata, replacing the generated _version.py that
# versioneer used to keep in the source tree. It is determined before the
# submodules are imported, because uts.io.darr records it in saved signals and
# so needs it to already exist.
from importlib.metadata import version as _version, \
    PackageNotFoundError as _PackageNotFoundError

try:
    __version__ = _version('uts')
except _PackageNotFoundError:  # running from a source tree, never installed
    __version__ = 'unknown'
del _version, _PackageNotFoundError


from .core.uniformtime import *

try:
    from .core.uniformtime import *
except Exception as e:  # pragma: no cover
    if 'No module named' in str(e):
        raise ImportError('C extensions not built: if you installed already '
                          'verify that you are not importing from the source '
                          'directory')
    else:
        raise

from .core.uniformtime import *
from .core.uniformfrequency import *
from .core.utils import *
from .io import *
from .io.hdf5 import *
from .io.audio import *
from .io.matlab import *
from .io.darr import *
from .core.check_arg import check_arg

from .signalfactory.factory import *
from .signalfactory.h5factory import *
from .signalfactory.testsignals import *
from .signalfactory.examplesignals import *

from .processing.uniformtime_processing import *
from .processing.elementwise_math import *
from .processing.stats import *

from .tests import test