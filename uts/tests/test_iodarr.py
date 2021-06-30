import uts
from unittest import TestCase
from uts.io.utils import tempdir
from numpy.random import random
from uts import UniformTimeSeries


# TODO many more tests
class test_SaveDarr(TestCase):

    def test_savedarr_uts(self):
        with tempdir(dir=None, keep=False, report=False) as tempdirname:
            s1 = UniformTimeSeries(samples=random(50)*100-50, fs=50.1, origintime=30.0,
                                   startdatetime='2021-06-29T12:00:01.', dtype='float32',
                                   readonly=True, metadata={'a':0.3})
            uts.savedarr(tempdirname, s1, accessmode='r', overwrite=True)
            s2 = uts.loaddarr(tempdirname)
            self.assertEqual(s1, s2)



