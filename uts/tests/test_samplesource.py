from unittest import TestCase

from numpy.testing import assert_raises
import numpy as np
from uts.core.samplesource import Samples


class test_Samples(TestCase):

    def test_dtypearg(self):
        "Samples: dtype arg"
        dtypes = [np.float32, np.float64, np.uint8, np.int8, np.uint16, 
                      np.int16, np.uint32, np.int32, np.uint64, np.int64]
        for dtype in dtypes:
            Samples(dtype=dtype, shape=(10,), sampletype=np.ndarray, 
                    axes={'time': 0}).dtype == dtype
        dtypes = [np.array([1,2]), "abc", 1, 1.0, [1,2], (1,2)]
        for dtype in dtypes:
            assert_raises(TypeError, Samples, dtype=dtype, shape=(10,), 
                          sampletype=np.ndarray, axes={'time': 0})
    
    def test_shapearg(self):
        "Samples: shape arg"
        shapes = [(10,),(10,10),(10,10,10)]
        for shape in shapes:
            axes = {}
            for i,ax in enumerate(shape):
                axes['ax%d'% i] = i
            s = Samples(dtype=np.float32, shape=shape, sampletype=np.ndarray, 
                    axes=axes)
            assert s.shape == shape
            assert s.size == np.prod(shape)
        shapes = [(None,),(10.1,),(10,10,10.1),(10.0,)]
        for shape in shapes:
            axes = {}
            for i,ax in enumerate(shape):
                axes['ax%d'% i] = i
            assert_raises(ValueError, Samples, dtype=np.float32, shape=shape, 
                          sampletype=np.ndarray, axes=axes)

if __name__ == '__main__':
    unittest.main()