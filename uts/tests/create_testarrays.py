import tables
import numpy as np
from uts.tests.defs import floattypes, inttypes


def create_h5arrays(testf):
    
    # returns a new hdf5 file with 1D, 2D and 3D Arrays and EArrays of different
    # dtypes, inttypes and floattypes from defs.py, and bool and str. The file is
    # temporary, and will be automatically deleted when closed.
    
    # test as many Array type as we can (np.object currently not, because limitations pytables)
    for dtype in (inttypes + floattypes + [bool, np.string_]):
        ars = {'1D':np.array([1,2,3],dtype=dtype),
               '2D':np.array([[1,2,3],[4,5,6]],dtype=dtype),
               '3D':np.array([[[1,2,3],[4,5,6]],[[1,2,3],[4,5,6]]],dtype=dtype)}
        for key, ar in ars.items():
            # Array
            testf.create_array(testf.root, dtype.__name__ + '7' + key + '7Array', ar)
            # EArray
            filters = tables.Filters(complevel=0)
            atom = tables.Atom.from_dtype(dtype=np.dtype(ar.dtype))
            shape = list(ar.shape[:-1])
            shape.append(0)
            arnode = testf.create_earray(where=testf.root,
                                        name=dtype.__name__ + '7' + key + '7EArray',
                                        atom=atom,
                                        shape=shape,
                                        filters=filters)
            arnode.append(ar)
    return testf
