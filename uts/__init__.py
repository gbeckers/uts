# pylint: disable-msg=W0614,W0401,W0611,W0622

__docformat__ = 'restructuredtext'

#from datetime import datetime

from uts3.core.uniformtime import *

try:
    from uts3.core.uniformtime import *
except Exception as e:  # pragma: no cover
    if 'No module named' in str(e):
        raise ImportError('C extensions not built: if you installed already '
                          'verify that you are not importing from the source '
                          'directory')
    else:
        raise

from uts3.version import version as __version__
from uts3.info import __doc__
from uts3.core.uniformtime import *
from uts3.io.hdf5 import *
from uts3.io.audio import *
from uts3.io.matlab import *
from uts3.core.check_arg import check_arg
from uts3.display.stftview import STFTView

from signalfactory.factory import *
from signalfactory.testsignals import *

from processing.processing import *
from processing.elementwise_math import *
from processing.stats import *

import tests

from numpy.testing import Tester
test = Tester().test
bench = Tester().bench