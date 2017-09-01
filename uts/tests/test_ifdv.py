import os
import time
from numpy.testing import TestCase, assert_raises, run_module_suite, \
     assert_almost_equal
import numpy as np
from scipy.io import loadmat
import uts
from uts.processing.ifdv import cifdv

TEST_DATA_DIR = os.path.join(os.path.dirname(uts.__file__),
                             'testdata')

class test_ifdv(TestCase):

    def test_compare_result_with_matlab(self):
        "comparing ifdv result with that of matlab original"
        
        # load matlab results; we need the int()'s because loadmat will read 
        # uint16, which is problematic for ifdv
        mlresults = loadmat(os.path.join(TEST_DATA_DIR, 'ifdv_test.mat'))
        s=mlresults['X'][0]
        SAMPLING=mlresults['SAMPLING'][0,0]
        N=int(mlresults['N'][0,0])
        OVERLAP=int(mlresults['OVERLAP'][0,0])
        sigma=float(mlresults['sigma'][0,0])
        ZOOMT=int(mlresults['ZOOMT'][0,0])
        ZOOMF=int(mlresults['ZOOMF'][0,0])
        TL=int(mlresults['TL'][0,0])
        FL=int(mlresults['FL'][0,0])
        
        # calculate
        ifdgram, sonogram = cifdv(s=s, SAMPLING=SAMPLING, N=N, OVERLAP=OVERLAP, 
            sigma=sigma, ZOOMT=ZOOMT, ZOOMF=ZOOMF, TL=TL, FL=FL, 
            symmetric=False)
        assert_almost_equal(sonogram, mlresults['sonogram'])
        assert_almost_equal(ifdgram, mlresults['ifdgram'])
