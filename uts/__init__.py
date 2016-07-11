# pylint: disable-msg=W0614,W0401,W0611,W0622
from __future__ import absolute_import
__docformat__ = 'restructuredtext'


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

# from uts.version import version as __version__
from .core.uniformtime import *
from .core.uniformfrequency import *
from .core.utils import *
from .io.hdf5 import *
from .io.audio import *
from .io.matlab import *
from .core.check_arg import check_arg

from .signalfactory.factory import *
from .signalfactory.h5factory import *
from .signalfactory.testsignals import *
from .signalfactory.examplesignals import *


from .processing.uniformtime_processing import *
from .processing.elementwise_math import *
from .processing.stats import *

from . import tests

from numpy.testing import Tester
test = Tester().test
bench = Tester().bench
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
