# pylint: disable-msg=W0614,W0401,W0611,W0622

__docformat__ = 'restructuredtext'

#from datetime import datetime

from uts.core.uniformtime import *

try:
    from uts.core.uniformtime import *
except Exception as e:  # pragma: no cover
    if 'No module named' in str(e):
        raise ImportError('C extensions not built: if you installed already '
                          'verify that you are not importing from the source '
                          'directory')
    else:
        raise

# from uts.version import version as __version__
from uts.info import __doc__
from uts.core.uniformtime import *
from uts.core.uniformfrequency import *
from uts.core.utils import *
from uts.io.hdf5 import *
from uts.io.audio import *
from uts.io.matlab import *
from uts.core.check_arg import check_arg

from signalfactory.factory import *
from signalfactory.h5factory import *
from signalfactory.testsignals import *
from signalfactory.examplesignals import *


from processing.uniformtime_processing import *
from processing.elementwise_math import *
from processing.stats import *

import tests

from numpy.testing import Tester
test = Tester().test
bench = Tester().bench
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
