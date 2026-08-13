from unittest import TestLoader, TextTestRunner, TestSuite

from . import create_testarrays
from . import defs
from . import test_checkarg
from . import test_coreutils
from . import test_epoched
from . import test_geometry
from . import test_ifdv
from . import test_iodarr
from . import test_multichannel
from . import test_samplesource
from . import test_signatures
from . import test_stats
from . import test_uniformtime
from . import test_uniformtime_processing

modules = [create_testarrays, defs, test_checkarg, test_coreutils, test_epoched,
           test_geometry, test_ifdv, test_iodarr, test_multichannel,
           test_samplesource, test_signatures, test_stats, test_uniformtime,
           test_uniformtime_processing]

def test(verbosity=1):
    suite = TestSuite()
    for module in modules:
        suite.addTests(TestLoader().loadTestsFromModule(module))
    return TextTestRunner(verbosity=verbosity).run(suite)