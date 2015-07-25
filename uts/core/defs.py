import numpy as np

class DimensionError(Exception): pass

# np.uint64 creates troubles with cython an indexing
defined_integers = (int, long, np.uint8, np.int8, np.uint16,
        np.int16, np.uint32, np.int32, np.int64)
 
defined_floats = (float, np.float16, np.float32, np.float64)
        
eps = np.power(2.0,-52.0)

defined_sampledtypes = (np.int8, np.int16, np.int32, np.int64,
                    np.uint8, np.uint16, np.uint32, np.uint64,
                    np.float16, np.float32, np.float64, np.complex128)
