"""
Classes for sample objects.

Sample objects have the same interface to retrieve samples as a NumPy array, 
but may have different underlying sources, e.g. a NumPy array, h5array node, 
or some file that is accessed over a network. All classes should inherit from 
Samples.

"""
# FIXME set_readonly

import numpy as np
import tables as tb
import h5py
import darr
from .check_arg import check_arg, check_int, check_argarray
from .defs import defined_sampledtypes


class Samples(object):
    """
    Superclass for sample objects.

    Sample children should define at least a __getitem__ method that returns a
    NumPy array.

    Parameters
    ----------
    dtype: numpy dtype
      dtype of the array that is returned
    shape: a length-2 tuple
      dimensions of the array that is returned
    sampletype: string
      a short string describing where the samples are coming from
    axes: dict
      a dictionary that describes the axes of the ndarray that is returned

    Attributes
    ----------
    dtype
    ndim
    shape
    size
    nsamples
    sampletype
    axes

    """

    def __init__(self, dtype, shape, sampletype, axes, readonly=True):
        self._set_dtype(dtype)
        self.__shape, self.__size = self._check_shape(shape)
        self.__axes, self.__axisorder = self._check_axes(axes, self.__shape)
        self.__type = sampletype
        self.__ndim = len(self.__shape)
        self.__readonly = check_arg(readonly, 'readonly')
        self.__mb = (self.__size * self.__dtype.itemsize) / float(1024 * 1024)


    dtype = property(lambda self: self.__dtype, None, None,
                     "array of samples")
    ndim = property(lambda self: self.__ndim, None, None,
                    "number of dimensions")
    shape = property(lambda self: self.__shape, None, None,
                     "shape")
    size = property(lambda self: self.__size, None, None,
                    "size")
    sampletype = property(lambda self: self.__type, None, None,
                          "type of samples")
    axes = property(lambda self: self.__axes, None, None,
                    "dictionary with axisnames")
    axisorder = property(lambda self: self.__axisorder, None, None,
                         "dictionary with axisnames")
    readonly = property(lambda self: self.__readonly, None, None,
                        "are samples readonly or not")
    mb = property(lambda self: (self.__mb), None, None,
                  "size in Mb")


    def __getitem__(self, index):
        raise NotImplementedError("__getitem__ needs to be implemented by subclass")

    def __setitem__(self, key, index):
        raise NotImplementedError("__setitem__ needs to be implemented by subclass")

    def __str__(self):
        return "<Samples: %s, %s, %s>" \
               % (self.shape, self.axisorder, self.dtype)

    __repr__ = __str__

    def _instantiatefromself(self, **params):
        raise NotImplementedError("__getitem__ needs to be implemented by subclass")

    def _update_shape(self, shape):
        if (len(shape) != self.ndim):
            raise ValueError("shape parameter has wrong number of dimensions")
        self.__shape, self.__size = self._check_shape(shape)
        self.__mb = (self.__size * self.__dtype.itemsize) / float(1024 * 1024)

    def _check_shape(self, shape):
        if not isinstance(shape, tuple):
            raise TypeError("shape parameter should be a tuple")
        for ax in shape:
            if not np.issubdtype(type(ax), np.integer):
                raise ValueError("shape dimensions should be ints")
        size = np.prod(shape, dtype='int64')
        return shape, size

    def _check_axes(self, axes, shape):
        if not isinstance(axes, dict):
            raise TypeError("axes parameter should be a dictionary")
        axiskeys = list(axes.keys())
        if not len(axiskeys) == len(shape):
            raise ValueError("axes dictionary (%s) not congruent with shape parameter (%s)" \
                             % (axiskeys, shape))
        axisorder = np.array(axiskeys, copy=True)
        for axiskey in axiskeys:
            # print(axiskey, axisorder, axiskey, axes[axiskey])
            axisorder[axes[axiskey]] = axiskey
        return axes, axisorder.tolist()

    def _set_dtype(self, dtype):
        try:
            dtype = np.dtype(dtype)
        except TypeError as e:
            raise TypeError('dtype parameter is not a numpy dtype') from e
        self.__dtype = dtype

    # FIXME: it would be nice to have a parameter that would get the samples
    # with a specific axes order
    def get(self, copy=True):
        """
        Returns the actual sample values in a numpy array.

        Parameter
        ---------
        copy: {True, False}
            Determines whether or not the values should be copied to a new
            array. Note that this parameter is a request that need not always
            be granted. If `copy` is True, this should always result in a copy.
            If `copy` is False however, a view of the original samples will be
            attempted, but may not succeed. In that case a copy will be
            returned. Default is True.

        """
        ## The reason this method exists, instead of accessing the `samples`
        ## member directly, is that it is not always sure what indexed samples
        ## (samples[:]) returns: a view or a copy. The `copy` parameter in the
        ## current method makes sure that a copy is retured if requested.

        copy = check_arg(copy, 'copy')
        s = self[:]
        # we make no assumption how the samples were retrieved, so we test if
        # they happen to be a view of something
        if (not s.base is None) and copy:
            return s.copy()
        else:
            return s

class DarrSamples(Samples):

    descr = 'darr'

    def __init__(self, a, axes):
        self.__array = a
        dtype = self.__array.dtype
        shape = self.__array.shape
        sampletype = type(self.__array)
        if a.accessmode == 'r':
            readonly = True
        else:
            readonly = False
        Samples.__init__(self, dtype=dtype, shape=shape, sampletype=sampletype,
                         axes=axes, readonly=readonly)

    array = property(lambda self: (self.__array), None, None,
                       "darr array object")

    def __getitem__(self, index):
        return self.__array[index]


    def append(self, samples, axis):
        """
        Appends samples.

        Parameter
        ---------
        samples: a numpy ndarray

        """
        self.__array.append(samples)
        self._update_shape(self.__array.shape)

    def transpose(self, neworder):
        """
        Returns a NumpyArraySamples object that is a transpose of the
        h5ArraySamples object.

        Parameter
        ---------
        neworder:
          sequence of axes names that represent the desired order

        """
        return NumpyArraySamples(array=self[:], axes=self.axes,
                                 dtype=None, copy=False,
                                 readonly=self.readonly).transpose(neworder)



class PytablesArraySamples(Samples):
    # _supporteddtypes = (np.int8, np.int16, np.int32, np.int64,
    #                     np.uint8, np.uint16, np.uint32, np.uint64,
    #                     np.float16, np.float32, np.float64, np.complex128)

    descr = 'h5array'

    def __init__(self, h5array, axes, readonly=True):
        self.__h5array = self._check_h5array(h5array=h5array)
        dtype = self.__h5array.atom.dtype
        shape = self.__h5array.shape
        sampletype = type(self.__h5array)
        Samples.__init__(self, dtype=dtype, shape=shape, sampletype=sampletype,
                         axes=axes, readonly=readonly)

    h5array = property(lambda self: (self.__h5array), None, None,
                       "h5array object")

    def __getitem__(self, index):

        return self.__h5array[index]

    def _check_h5array(self, h5array):
        if not isinstance(h5array, (tb.carray.CArray, tb.array.Array,
                                    tb.earray.EArray)):
            raise TypeError("h5node should be a PyTables array node")

        if h5array.atom.dtype not in defined_sampledtypes:
            raise TypeError("input type %s not supported" % h5array.atom.dtype)

        return h5array

    def append(self, samples, axis):
        """
        Appends samples.

        Parameter
        ---------
        samples: a numpy ndarray

        """
        if not self.sampletype == tb.earray.EArray:
            raise TypeError("h5array is not extendible")
        if not (self.dtype == samples.dtype):
            raise TypeError("samples has wrong dtype")
        if not axis == self.__h5array.maindim:
            raise ValueError("for now only appending to axis 1 only")
        self.__h5array.append(samples)
        self.__h5array.flush()
        self._update_shape(self.__h5array.shape)

    def transpose(self, neworder):
        """
        Returns a NumpyArraySamples object that is a transpose of the 
        h5ArraySamples object.
        
        Parameter
        ---------
        neworder:
          sequence of axes names that represent the desired order
          
        """
        return NumpyArraySamples(array=self[:], axes=self.axes,
                                 dtype=None, copy=False, readonly=self.readonly).transpose(neworder)

h5ArraySamples = PytablesArraySamples

class H5pyArraySamples(Samples):
    # _supporteddtypes = (np.int8, np.int16, np.int32, np.int64,
    #                     np.uint8, np.uint16, np.uint32, np.uint64,
    #                     np.float16, np.float32, np.float64, np.complex128)

    descr = 'h5pyarray'

    def __init__(self, h5array, axes, readonly=True):
        self.__h5array = self._check_h5array(h5array=h5array)
        dtype = self.__h5array.dtype
        shape = self.__h5array.shape
        sampletype = type(self.__h5array)
        Samples.__init__(self, dtype=dtype, shape=shape, sampletype=sampletype,
                         axes=axes, readonly=readonly)

    h5pyarray = property(lambda self: (self.__h5array), None, None,
                       "h5array object")

    def __getitem__(self, index):

        return self.__h5array[index]

    def _check_h5array(self, h5array):
        if not isinstance(h5array, (h5py.Dataset)):
            raise TypeError("h5node should be a h5py array node")

        if h5array.dtype not in defined_sampledtypes:
            raise TypeError("input type %s not supported" % h5array.atom.dtype)

        return h5array

    def append(self, samples, axis):
         raise NotImplementedError('append not supported yet for h5py arrays')

    def transpose(self, neworder):
        """
        Returns a NumpyArraySamples object that is a transpose of the
        h5ArraySamples object.

        Parameter
        ---------
        neworder:
          sequence of axes names that represent the desired order

        """
        return NumpyArraySamples(array=self[:], axes=self.axes,
                                 dtype=None, copy=False, readonly=self.readonly).transpose(neworder)



class NumpyArraySamples(Samples):
    # _supporteddtypes = (np.int8, np.int16, np.int32, np.int64,
    #                     np.uint8, np.uint16, np.uint32, np.uint64,
    #                     np.float16, np.float32, np.float64, np.complex128)

    descr = 'numpyarray'

    def __init__(self, array, axes, dtype=None, copy=True, readonly=True):
        self.__array = self._check_array(array=array, dtype=dtype,
                                         copy=copy)
        dtype = self.__array.dtype
        shape = self.__array.shape
        sampletype = type(array)
        Samples.__init__(self, dtype=dtype, shape=shape, sampletype=sampletype,
                         axes=axes, readonly=readonly)

    array = property(lambda self: (self.__array), None, None,
                     "number of samples")

    def __getitem__(self, index):
        return self.__array[index]

    def __setitem__(self, key, item):
        if not self.readonly:
            self.__array[key] = item
        else:
            raise IOError("the samples are read-only")

    def _check_array(self, array, dtype, copy):
        # if not provided as such, try to convert samples to np array
        if not isinstance(array, np.ndarray):
            if not copy:
                raise ValueError("copy argument cannot be False if samples is not a numpy array")
            try:
                array = np.array(array)
            except (TypeError, ValueError) as e:
                raise TypeError("input type cannot be converted to np array") from e
        else:
            if copy:
                #array = np.array(array, copy=True)
                array = array.copy()

        # if not provdided as parameter, take dtype from sample array
        if dtype is None:
            dtype = array.dtype

        # if provided samples have a different dtype, try to convert
        if array.dtype != dtype:
            if not copy:
                raise ValueError("copy argument cannot be False if dtype of samples does not match dtype argument")
            array = array.astype(dtype)

        if array.dtype not in defined_sampledtypes:
            raise TypeError("input type %s not supported" % array.dtype)

        return array

    def append(self, samples, axis):
        # asarray, not array(copy=False): in NumPy 2 copy=False means "never
        # copy, raise if a copy is needed" rather than "copy only if needed".
        samples = np.asarray(samples)
        if not (self.dtype == samples.dtype):
            raise TypeError("samples has wrong dtype")
        self.__array = np.append(self.__array, samples, axis=axis)
        self._update_shape(self.__array.shape)

    def transpose(self, neworder, ignore_missingaxes=False):
        """
        Transposes the axes of the samples object
        
        Parameter
        ---------
        neworder:
          sequence of axes names that represent the desired order
        ignore_missingaxes:
            if neworder contains axes that do not exist, they will be ignored
          
        """
        # check arguments
        axesnames = list(self.axes.keys())
        if ignore_missingaxes:
            neworder = [ax for ax in neworder if ax in axesnames]
        neworder = np.array(neworder)#,  dtype=np.bytes_)
        axesnames = np.array(axesnames)#,  dtype=np.bytes_)  # we still need to order
        if not len(neworder) == len(axesnames):
            raise ValueError("neworder argument (%s) does not have the right number of axes (%s)" \
                             % (neworder, axesnames))
        for axisname, order in self.axes.items():
            axesnames[order] = axisname
        if (neworder == axesnames).all():
            return self
        else:
            t = []
            newaxes = {}
            for i, a in enumerate(neworder):
                if a not in axesnames:
                    raise ValueError("axis %s does not exist" % a)
                t.append(self.axes[a])
                newaxes[a] = i
            array = np.transpose(self.array, t)
            return self.__class__(array=array, axes=newaxes, dtype=self.dtype,
                                  copy=False, readonly=self.readonly)


def assamples(samples, axes, dtype=None, copy=True, readonly=True):
    """
    Create an apropriate Sample subclass instance.

    Parameters
    ----------
    samples: array_like or pytables Table
        Numpy arrays, lists and tuples will be converted to NumpyArraySamples
        instances. PyTable (on-disk) arrays will be converted to h5ArraySamples
        instances.
    axes: dict
        Defines the axes.
    dtype: data type, optional
        By default, the data-type is inferred from the input data.
    copy: bool, optional
        If true (default), then the samples object is copied. PyTable arrays
        will be converted to NumpyArraySamples if copy is true.

    """
    if isinstance(samples, Samples):
        if copy:
            if dtype is None:
                dtype = samples.dtype
            else:
                if not (np.dtype(dtype) is samples.dtype):
                    raise ValueError("dtypes don't match" \
                                     "(param: %s, Samples: %s)" % (dtype, samples.dtype))
            if axes != samples.axes:
                raise ValueError("axes don't match" \
                                 "(param: %s, Samples: %s)" % (axes, samples.axes))
            if readonly != samples.readonly:
                raise ValueError("readonly parameter does not match" \
                                 "(param: %s, Samples: %s)" % (readonly, samples.readonly))
            samples = samples.get(copy=True)
        else:
            return samples
    if isinstance(samples, darr.Array):
        samples = DarrSamples(samples, axes=axes)
    elif isinstance(samples, (list, tuple, np.ndarray)):
        samples = NumpyArraySamples(array=samples,
                                    dtype=dtype, copy=copy,
                                    axes=axes, readonly=readonly)
    elif isinstance(samples, (tb.carray.CArray, tb.array.Array,
                              tb.earray.EArray)):
        if not copy:
            if (dtype is not None) and (dtype != samples.atom.dtype):
                raise ValueError("dtype argument ({}) is different from h5array dtype ({})".format(dtype,samples.atom.dtype ))
            samples = PytablesArraySamples(h5array=samples, axes=axes, readonly=readonly)
        else:  # copy == True
            samples = NumpyArraySamples(array=samples[:],
                                        dtype=dtype, copy=True,
                                        axes=axes, readonly=readonly)

    elif isinstance(samples, h5py.Dataset):
        if not copy:
            if (dtype is not None) and (dtype != samples.dtype):
                raise ValueError(
                    "dtype argument ({}) is different from h5array dtype ({})".format(
                        dtype, samples.dtype))
            samples = H5pyArraySamples(h5array=samples, axes=axes,
                                           readonly=readonly)
        else:  # copy == True
            samples = NumpyArraySamples(array=samples[:],
                                        dtype=dtype, copy=True,
                                        axes=axes, readonly=readonly)
    else:
        raise TypeError("samples (%s) is not a list, tuple, or numpy / pytables array" \
                        % type(samples))
    return samples
