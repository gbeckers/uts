"""
Tests for class UniformTime

"""
#FIXME: saveh5 tests (now uncommented should be moved to io test module
from unittest import TestCase

import os
from numpy.testing import assert_raises, run_module_suite, \
     assert_almost_equal, assert_array_almost_equal, assert_equal
import numpy as np
import tables
from uts import UniformTime, UniformTimeSeries, EpochedUniformTimeSeries, \
    MultiChannelUniformTimeSeries, EpochedMultiChannelUniformTimeSeries
from uts.core.defs import DimensionError
from uts.core.check_arg import argtypes
from uts.core.errors import WriteError
from uts.tests.defs import nan, inf, pfloats, zfloats, nfloats, floattypes, inttypes, eps,\
     pints, zints, nints, float_to_floats, int_to_nints, int_to_pints
from uts.tests.create_testarrays import create_h5arrays
from tempfile import NamedTemporaryFile

goodtimeinputs = pfloats + zfloats + nfloats
sequences = [[1.0,1.0], (1.0,1.0), np.array([1.0,1.0])]

goodarraydtypes = (np.int8, np.int16, np.int32, np.int64, np.uint8,
                  np.uint16, np.uint32, np.uint64, np.float32,
                  np.float64, np.complex128)


# all allowed int types
ints = (int, long, np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.int64)
# all allowed int types that can hold negative values
nints = (int, long, np.int8, np.int16, np.int32, np.int64)
# all allowed float types
floats = (float, np.float32, np.float64)


def test_UniformTime_ntimesamples():
    "UniformTime: ntimesamples parameter"
    
    # parameter is mandatory
    assert_raises(TypeError, UniformTime, ntimesamples=None, fs=10.0)
    
    # test for all int types
    for testtype in ints:
        assert_equal(UniformTime(ntimesamples=testtype(0), fs=10.0).ntimesamples, 0)
        assert_equal(UniformTime(ntimesamples=testtype(100), fs=10.0).ntimesamples, 100)
        # type should always be np.int64
        assert_equal(type(UniformTime(ntimesamples=testtype(100), fs=10.0).ntimesamples), np.int64)
    
    # test for all int types that can represent negative values    
    for testtype in nints:
        # values that are not allowed
        assert_raises(ValueError, UniformTime, ntimesamples=testtype(-1), fs=10.0)
    # we do not support uint64
    assert_raises(TypeError, UniformTime, ntimesamples=np.uint64(100), fs=10.0)
    # test for all float types    
    for testtype in floats:
        assert_raises(TypeError, UniformTime, ntimesamples=testtype(100.0), fs=10.0)

        
def test_UniformTime_fs():
    "UniformTime: fs parameter"
    
    # parameter is mandatory
    assert_raises(TypeError, UniformTime, ntimesamples=10, fs=None)
    
    # test for all float types    
    for testtype in floats:
        assert_equal(UniformTime(ntimesamples=10, fs=testtype(10.0)).fs, 10.0)
        # type should always be np.float64
        assert_equal(type(UniformTime(ntimesamples=10, fs=testtype(10.0)).fs), np.float64)
        # values that are not allowed
        assert_raises(ValueError, UniformTime, ntimesamples=10, fs=testtype(0.0))
        assert_raises(ValueError, UniformTime, ntimesamples=10, fs=testtype(-10.0))
        assert_raises(ValueError, UniformTime, ntimesamples=10, fs=testtype(np.inf))
        assert_raises(ValueError, UniformTime, ntimesamples=10, fs=testtype(-np.inf))
        assert_raises(ValueError, UniformTime, ntimesamples=10, fs=testtype(np.nan))
    
    # test for all int types
    for testtype in ints:
        assert_raises(TypeError, UniformTime, ntimesamples=10, fs=testtype(10))
        assert_raises(TypeError, UniformTime, ntimesamples=10, fs=testtype(0))
 
        
def test_UniformTime_starttime():
    "UniformTime: starttime parameter"
    
    # parameter defaults to 0.0
    assert_equal(UniformTime(ntimesamples=10, fs=10.0).starttime, 0.0)
    for testtype in floats:
        assert_equal(UniformTime(ntimesamples=10, fs=10.0, starttime=testtype(10.0)).starttime, 10.0)
        assert_equal(UniformTime(ntimesamples=10, fs=10.0, starttime=testtype(0.0)).starttime, 0.0)
        assert_equal(UniformTime(ntimesamples=10, fs=10.0, starttime=testtype(-10.0)).starttime, -10.0)
        # type should always be np.float64
        assert_equal(type(UniformTime(ntimesamples=10, fs=10.0, starttime=testtype(10.0)).starttime), np.float64)
        # values that are not allowed
        assert_raises(ValueError, UniformTime, ntimesamples=10, fs=10.0, starttime=testtype(np.inf))
        assert_raises(ValueError, UniformTime, ntimesamples=10, fs=10.0, starttime=testtype(-np.inf))
        assert_raises(ValueError, UniformTime, ntimesamples=10, fs=10.0, starttime=testtype(-np.nan))
    
    # test for all int types
    for testtype in ints:
        assert_raises(TypeError, UniformTime, ntimesamples=10, fs=10.0, starttime=testtype(0))
    
    # None is not allowed
    assert_raises(TypeError, UniformTime, ntimesamples=10, fs=10.0, starttime=None)

    
def test_UniformTime_origintime():
    "UniformTime: origintime parameter"
    
    # parameter defaults to 0.0
    assert_equal(UniformTime(ntimesamples=10, fs=10.0).origintime, 0.0)
    for testtype in floats:
        assert_equal(UniformTime(ntimesamples=10, fs=10.0, origintime=testtype(10.0)).origintime, 10.0)
        assert_equal(UniformTime(ntimesamples=10, fs=10.0, origintime=testtype(0.0)).origintime, 0.0)
        assert_equal(UniformTime(ntimesamples=10, fs=10.0, origintime=testtype(-10.0)).origintime, -10.0)
        # type should always be np.float64
        assert_equal(type(UniformTime(ntimesamples=10, fs=10.0, origintime=testtype(10.0)).origintime), np.float64)
        # values that are not allowed
        assert_raises(ValueError, UniformTime, ntimesamples=10, fs=10.0, origintime=testtype(np.inf))
        assert_raises(ValueError, UniformTime, ntimesamples=10, fs=10.0, origintime=testtype(-np.inf))
        assert_raises(ValueError, UniformTime, ntimesamples=10, fs=10.0, origintime=testtype(-np.nan))
    
    # test for all int types
    for testtype in ints:
        assert_raises(TypeError, UniformTime, ntimesamples=10, fs=10.0, origintime=testtype(0))
    
    # None is not allowed
    assert_raises(TypeError, UniformTime, ntimesamples=10, fs=10.0, origintime=None)

    
def test_UniformTime_samplingtimes():
    "UniformTime: sampling times method"

    # dtype is float64
    assert_equal(UniformTime(ntimesamples=2, fs=2.0).samplingtimes().dtype, np.float64)
    # simple case; times are relative to origintime, which defaults to 0.0
    assert_equal(UniformTime(ntimesamples=2, fs=2.0).samplingtimes(), np.array([0.25,0.75],dtype=np.float64))
    # different origintime
    assert_equal(UniformTime(ntimesamples=2, fs=2.0, origintime=10.0).samplingtimes(), np.array([0.25,0.75],dtype=np.float64)-10.0)
    # starttime is irrelevant
    assert_equal(UniformTime(ntimesamples=2, fs=2.0, starttime=10.0).samplingtimes(), np.array([0.25,0.75],dtype=np.float64))
    # no samples
    assert_equal(UniformTime(ntimesamples=0, fs=2.0).samplingtimes(), np.array([],dtype=np.float64))

    
def test_UniformTime_dt():
    "UniformTime: dt"
    
    assert_equal(UniformTime(ntimesamples=2, fs=2.0).dt, 0.5)
    assert_equal(type(UniformTime(ntimesamples=2, fs=2.0).dt), np.float64)    

    
def test_UniformTime_duration():
    "UniformTime: duration "
    
    assert_equal(UniformTime(ntimesamples=2, fs=2.0).duration, 1.0)
    assert_equal(type(UniformTime(ntimesamples=2, fs=2.0).duration), np.float64)
    # starttime and origintime don't matter
    assert_equal(UniformTime(ntimesamples=2, fs=2.0, starttime=10.0).duration, 1.0)
    assert_equal(UniformTime(ntimesamples=2, fs=2.0, origintime=10.0).duration, 1.0)
    

def test_UniformTime_endtime():
    "UniformTime: endtime"
    
    assert_equal(UniformTime(ntimesamples=2, fs=2.0).endtime, 1.0)
    assert_equal(type(UniformTime(ntimesamples=2, fs=2.0).endtime), np.float64)
    # origintime doesn't matter
    assert_equal(UniformTime(ntimesamples=2, fs=2.0, origintime=10.0).endtime, 1.0)
    # starttime does
    assert_equal(UniformTime(ntimesamples=2, fs=2.0, starttime=10.0).endtime, 11.0)
 
    
def test_UniformTime_timerange():
    "UniformTime: timerange"
    
    assert_equal(UniformTime(ntimesamples=2, fs=2.0).timerange, (0.0,1.0))
    # depends on origintime
    assert_equal(UniformTime(ntimesamples=2, fs=2.0, origintime=10.0).timerange, (-10.0,-9.0))
    # starttime doesn't matter
    assert_equal(UniformTime(ntimesamples=2, fs=2.0, starttime=10.0).timerange, (0.0,1.0))
    
    
def test_UniformTime_time_to_index():
    "UniformTime: time to index conversion"
    # Note that the `time_to_index` method should report to which *sample center* the
    # given time is closest.
    
    s1 = UniformTime(ntimesamples=2, fs=2.0)
    s2 = UniformTime(ntimesamples=2, fs=2.0, origintime=1.0)
    s3 = UniformTime(ntimesamples=2, fs=2.0, starttime=1.0)
    # only floats should work
    for testtype in floats:
        # center of sample should be reference
        assert_equal(s1.time_to_index(testtype(0.25)),0)
        assert_equal(s1.time_to_index([testtype(0.25)]),0)
        assert_equal(s1.time_to_index([testtype(0.25),testtype(0.75)]),np.array([0,1]))
        assert_equal(s1.time_to_index([testtype(0.24999),testtype(0.25001)]),np.array([0,0]))
        assert_equal(s1.time_to_index([testtype(0.499999),testtype(0.500001)]),np.array([0,1]))
        assert_equal(s1.time_to_index([testtype(0.000001),testtype(0.999999)]),np.array([0,1]))
        assert_equal(s1.time_to_index([testtype(0.0),testtype(0.999999)]),np.array([0,1]))
        # depends on origintime
        assert_equal(s2.time_to_index([testtype(-0.75),testtype(-0.25)]),np.array([0,1]))
        # starttime doesn't matter
        assert_equal(s3.time_to_index([testtype(0.25),testtype(0.75)]),np.array([0,1]))       
        # outside time range
        assert_raises(ValueError, s1.time_to_index, time=[testtype(-0.25),testtype(0.75)])
        assert_raises(ValueError, s1.time_to_index, time=[testtype(0.25),testtype(-0.75)])        
        assert_raises(ValueError, s1.time_to_index, time=[testtype(0.25),testtype(1.1)])
        assert_raises(ValueError, s1.time_to_index, time=[testtype(1.1),testtype(1.2)])
        assert_raises(ValueError, s1.time_to_index, time=[testtype(-0.75),testtype(-0.25)])
        # weird values
        assert_raises(ValueError, s1.time_to_index, time=np.nan)
        assert_raises(ValueError, s1.time_to_index, time=-np.inf)
        assert_raises(ValueError, s1.time_to_index, time=np.inf)
    # all ints should fail
    for testtype in ints:
        assert_raises(TypeError, s1.time_to_index, time=testtype(0))
    # None should fail to
    assert_raises(TypeError, s1.time_to_index, time=None)
    # nearest_boundary = True
    for testtype in floats:
        # center of sample should be reference
        assert_equal(s1.time_to_index(testtype(0.1),True),0)
        assert_equal(s1.time_to_index([testtype(0.1)],True),0)
        assert_equal(s1.time_to_index([testtype(0.24999),testtype(0.25001)],True),np.array([0,1]))
        assert_equal(s1.time_to_index([testtype(0.499999),testtype(0.500001)],True),np.array([1,1]))
        assert_equal(s1.time_to_index([testtype(0.000001),testtype(0.999999)],True),np.array([0,2]))
        assert_equal(s1.time_to_index([testtype(0.0),testtype(0.999999)],True),np.array([0,2]))
        

def test_UniformTime_index_to_time():
    "UniformTime: index to time conversion"
    # Note that the time of a sample is defined as its average time,
    #i.e. the time of sample center.
     
    s = UniformTime(ntimesamples=2, fs=2.0)
    
    # only ints should work
    for testtype in ints:
        assert_equal(s.index_to_time(testtype(0)),0.25)
        assert_equal(s.index_to_time([testtype(0)]),0.25)
        assert_equal(s.index_to_time([testtype(0),testtype(1)]),np.array([0.25,0.75]))
        
        # depends on origintime
        assert_equal(UniformTime(ntimesamples=2, fs=2.0, origintime=1.0).index_to_time([testtype(0),testtype(1)]),np.array([-0.75,-0.25]))

        # starttime doesn't matter
        assert_equal(UniformTime(ntimesamples=2, fs=2.0, starttime=1.0).index_to_time([testtype(0),testtype(1)]),np.array([0.25,0.75]))
       
        # outside time range
        assert_raises(ValueError, s.index_to_time, index=-1)
        assert_raises(ValueError, s.index_to_time, index=2)
        assert_raises(ValueError, s.index_to_time, index=[testtype(-1),testtype(0)])
        assert_raises(ValueError, s.index_to_time, index=[testtype(-2),testtype(-1)])
        assert_raises(ValueError, s.index_to_time, index=[testtype(0),testtype(2)])        
        assert_raises(ValueError, s.index_to_time, index=[testtype(2),testtype(3)])
        assert_raises(ValueError, s.index_to_time, index=[testtype(-1),testtype(3)])
    
    # all floats should fail
    for testtype in floats:
        assert_raises(TypeError, s.index_to_time, index=testtype(0))
    
    assert_raises(TypeError, s.index_to_time, index=None)

def test_movingtimeframeindices():

    s = UniformTime(ntimesamples=5, fs=5.0)
    for testtype in ints:
        assert_equal(list(s.movingtimeframeindices(framesize=testtype(1), stepsize=None, include_remainder=False)), [(0,1),(1,2),(2,3),(3,4),(4,5)])
        assert_equal(list(s.movingtimeframeindices(framesize=testtype(1), stepsize=None, include_remainder=True)), [(0,1),(1,2),(2,3),(3,4),(4,5)])
        assert_equal(list(s.movingtimeframeindices(framesize=testtype(2), stepsize=None, include_remainder=False)), [(0,2),(2,4)])
        assert_equal(list(s.movingtimeframeindices(framesize=testtype(2), stepsize=None, include_remainder=True)), [(0,2),(2,4),(4,5)])
        assert_equal(list(s.movingtimeframeindices(framesize=testtype(2), stepsize=testtype(1), include_remainder=False)), [(0,2),(1,3),(2,4),(3,5)])
        assert_equal(list(s.movingtimeframeindices(framesize=testtype(2), stepsize=testtype(1), include_remainder=True)), [(0,2),(1,3),(2,4),(3,5)])
        assert_equal(list(s.movingtimeframeindices(framesize=testtype(2), stepsize=testtype(3), include_remainder=False)), [(0,2),(3,5)])
        assert_equal(list(s.movingtimeframeindices(framesize=testtype(2), stepsize=testtype(3), include_remainder=True)), [(0,2),(3,5)])
        assert_equal(list(s.movingtimeframeindices(framesize=testtype(1), stepsize=testtype(3), include_remainder=False)), [(0,1),(3,4)])
        assert_equal(list(s.movingtimeframeindices(framesize=testtype(1), stepsize=testtype(3), include_remainder=True)), [(0,1),(3,4)])

        assert_raises(ValueError, list, s.movingtimeframeindices(framesize=testtype(1), stepsize=testtype(0)))
        assert_raises(ValueError, list, s.movingtimeframeindices(framesize=testtype(0), stepsize=testtype(3)))

    for ntesttype in nints:
        assert_raises(ValueError, list, s.movingtimeframeindices(framesize=1, stepsize=ntesttype(-1)))
        assert_raises(ValueError, list, s.movingtimeframeindices(framesize=ntesttype(-1), stepsize=3))
        assert_raises(ValueError, list, s.movingtimeframeindices(framesize=1, stepsize=ntesttype(0)))
        assert_raises(ValueError, list, s.movingtimeframeindices(framesize=ntesttype(0), stepsize=3))

    for testtype in floats:
        assert_raises(TypeError, list, s.movingtimeframeindices(framesize=testtype(1), stepsize=3))
        assert_raises(TypeError, list, s.movingtimeframeindices(framesize=1, stepsize=testtype(3)))
        assert_raises(TypeError, list, s.movingtimeframeindices(framesize=testtype(1), stepsize=testtype(3)))
    
    for testtype in ints:
        assert_equal(list(s.movingtimeframeindices(framesize=1, startindex=testtype(0), endindex=testtype(5))), [(0,1),(1,2),(2,3),(3,4),(4,5)])
        assert_equal(list(s.movingtimeframeindices(framesize=1, startindex=testtype(1), endindex=testtype(5))), [(1,2),(2,3),(3,4),(4,5)])
        assert_equal(list(s.movingtimeframeindices(framesize=1, startindex=testtype(0), endindex=testtype(4))), [(0,1),(1,2),(2,3),(3,4)])

        assert_raises(ValueError, list, s.movingtimeframeindices(framesize=testtype(1), startindex=testtype(5)))
        assert_raises(ValueError, list, s.movingtimeframeindices(framesize=testtype(1), startindex=testtype(-1)))
        assert_raises(ValueError, list, s.movingtimeframeindices(framesize=testtype(1), endindex=testtype(-1)))
        assert_raises(ValueError, list, s.movingtimeframeindices(framesize=testtype(1), endindex=testtype(6)))
        assert_raises(ValueError, list, s.movingtimeframeindices(framesize=testtype(1), startindex=testtype(4), endindex=testtype(3)))

    s = UniformTime(ntimesamples=20, fs=5.0)
    assert_equal(list(s.movingtimeframeindices(framesize=5, stepsize=4, include_remainder=True)), [(0,5),(4,9),(8,13),(12,17),(16,20)])

        
class test_UniformTimeSeries(TestCase):

    def setUp(self):
            f = NamedTemporaryFile()
            self.temph5filename = f.name
            f.close()
            self.temph5file = tables.openFile(self.temph5filename,'w')
            self.testarrays = create_h5arrays(self.temph5file)

    def tearDown(self):
        self.temph5file.close()
        os.remove(self.temph5filename)

    def test_input_ndarray_dimension(self):
        "UniformTimeSeries: input array dimension"
        # Tests if all valid argument input types lead to correct attribute
        # types, and if invalid input raises an appropriate error.

        # Note that all inputs that are handled by TimeSampled are not tested
        # here

        ## test different good array dtypes and sizes
        for samples in [[],[1],[1,2]]:
            for dtype in goodarraydtypes:
                a = np.array(samples, dtype=dtype)
                ss = UniformTimeSeries(samples=a,fs=1.0)
                assert ss.ntimesamples == len(samples)
                assert ss.dtype == dtype
                assert ss.samples.get().size == len(samples)
                assert ss.samples.get().ndim == 1

        ## test good dtypes, bad dimensions
        for samples in [1, [[],[]],[[1],[1]],[[1,2],[1,2]]]:
            for dtype in goodarraydtypes:
                a = np.array(samples, dtype=dtype)
                assert_raises(ValueError,UniformTimeSeries,
                              samples=a, fs=1.0)

    def test_input_ndarray_none(self):
        "UniformTimeSeries: input array None"
        ## None as sample
        assert_raises(TypeError, UniformTimeSeries, samples=None, fs=1.0)
        assert_raises(TypeError, UniformTimeSeries, samples=[None], fs=1.0)
        for floattype in floattypes:
            ts =  UniformTimeSeries(samples=[None], fs=1.0, dtype=floattype)
            assert ts.ntimesamples == 1
        for inttype in inttypes:
            assert_raises(TypeError, UniformTimeSeries, samples=[None],
                          fs=1.0, dtype=inttype)

    def test_input_ndarray_dtype(self):
        "UniformTimeSeries: input array dtype"
        types = floattypes + inttypes
        types.append(np.complex128)
        for t in types:
            ts =  UniformTimeSeries(samples=[1], fs=1.0, dtype=t)
            assert ts.dtype == t
            for s in ["t", "tkjh"]:
                assert_raises(TypeError, UniformTimeSeries, samples=s,
                          fs=1.0, dtype=t)
            for s in [np.array("t"),np.array("tkjh")]:
                assert_raises(ValueError, UniformTimeSeries, samples=s,
                          fs=1.0, dtype=t)
        assert_raises(TypeError, UniformTimeSeries, samples=["hhg"],
                          fs=1.0, dtype="|S3")

    def test_input_ndarray_copy(self):
        "UniformTimeSeries: copy argument ndarray"

        a = np.random.rand(10).astype(np.float64)
        # this should al go well
        ts1 = UniformTimeSeries(samples=a, fs=1.0, copy=False)
        ts2 = UniformTimeSeries(samples=a, fs=1.0, copy=True)
        ts3 = UniformTimeSeries(samples=a, fs=1.0, copy=False, dtype=a.dtype)
        # now change a value in the original array that was supplied as samples
        a[0] = -1.0
        assert ts1.samples.get()[0] == -1.0
        assert ts2.samples.get()[0] != -1.0
        assert ts3.samples.get()[0] == -1.0

        # now the things that should go wrong
        # copy is False, but samples dtype is different from dtype argument
        assert_raises(ValueError, UniformTimeSeries, samples=a,
                          fs=1.0, dtype=np.float32, copy=False)
        # copy is False, but samples is not a numpy array
        assert_raises(ValueError, UniformTimeSeries, samples=[1.0,2.0],
                          fs=1.0, copy=False)
        assert_raises(ValueError, UniformTimeSeries, samples=(1.0,2.0),
                          fs=1.0, copy=False)

    def test_input_h5array_dimension(self):
        "UniformTimeSeries: input h5array dimension"
        """Tests if all valid argument input types lead to correct attribute
        types, and if invalid input raises an appropriate error.
        """
        # Note that all inputs that are handled by TimeSampled are not tested
        # here


        ## test different good array dimensions and bad array dimensions
        for node in iter_arnodes(self.testarrays):
            typename, dim, artype = node._v_name.split('_')
            print typename, dim
            if dim == '1D':
                if node[:].dtype in goodarraydtypes:
                    ss = UniformTimeSeries(samples=node, fs=1.0, copy=False)
                    assert ss.ntimesamples == node.shape[0]
                    assert ss.dtype == node.atom.dtype
                    assert ss.samples.get().size == node.shape[0]
                    assert ss.samples.get().ndim == 1
                else:
                    assert_raises(TypeError,UniformTimeSeries,
                          samples=node, fs=1.0)

            else: # dimension is 2D or 3D
                assert_raises(ValueError,UniformTimeSeries,
                          samples=node, fs=1.0)

    def test_input_h5array_dtypes(self):
        "UniformTimeSeries: input h5array dtype"
        ## bad dtype argument: we do not allow dtypes different from atom.dtype
        types = floattypes + inttypes
        types.append(np.complex128)
        for t in types:
            t = np.dtype(t)
            for node in iter_arnodes(self.testarrays, dimension='1D'):
                if t == node.atom.dtype:
                    ss = UniformTimeSeries(samples=node, fs=1.0,
                                    dtype=t, copy=False)
                    assert ss.dtype == node.atom.dtype
                else:
                    # should fail if copy == False
                    assert_raises(ValueError,UniformTimeSeries,
                      samples=node, fs=1.0, dtype=t, copy=False)
                    # but not if copy == True
                    if t in goodarraydtypes:
                        ss = UniformTimeSeries(samples=node, fs=1.0,
                                        dtype=t, copy=True)
                        assert ss.dtype == t

    def test_input_h5array_copy(self):
        "UniformTimeSeries: copy argument h5array"
        ## copy argument False returns sampletype h5Array, True numpy array
        for node in iter_arnodes(self.testarrays, dimension='1D'):
            if node[:].dtype in goodarraydtypes:
                # copy is False, sampletype is h5 Array
                ss = UniformTimeSeries(samples=node, fs=1.0, copy=False)
                assert ss.sampletype == type(node)
                # copy is True, samples are copied to numpy array
                ss = UniformTimeSeries(samples=node, fs=1.0, copy=True)
                assert ss.sampletype == np.ndarray
                assert ss.dtype == node.atom.dtype
            else:
                assert_raises(TypeError,UniformTimeSeries,
                          samples=node, fs=1.0, copy=False)
                assert_raises(TypeError,UniformTimeSeries,
                          samples=node, fs=1.0, copy=True)

    def test_slicing_ndarray(self):
        "UniformTimeSeries: slicing ndarray"
        ## slicing should produce a view
        for copy in (False, True):
            a = np.random.rand(10).astype(np.float64)
            ts1 = UniformTimeSeries(samples=a, fs=1.0, copy=copy)
            ts1samples = ts1.samples.get(copy=False) # view, tested above
            ts2 = ts1[:] # this should be a view, sharing sample data
            ts1samples[0] = -1.0 # also alters sample 0 of ts1
            assert ts2.samples.get()[0] == -1.0

        ## slicing with ints should be seen as sample index
        a = np.random.rand(100).astype(np.float64)
        for i in int_to_pints(1):
            for j in int_to_pints(3):
                assert UniformTimeSeries(samples=a, fs=10.)[i:j].ntimesamples == 2

        ## slicing with floats should be seen as time index
        a = np.random.rand(100).astype(np.float64)
        for i in float_to_floats(1.0):
            for j in float_to_floats(3.0):
                assert UniformTimeSeries(samples=a, fs=10.)[i:j].ntimesamples == 20

        ## end before start allowed
        a = np.random.rand(100).astype(np.float64)
        ts = UniformTimeSeries(samples=a, fs=10.)
        start,end = ts.timerange
        assert ts[2:1].ntimesamples == 0
        assert ts[end:start].ntimesamples == 0

        ## index errors
        a = np.random.rand(10).astype(np.float64)
        ts = UniformTimeSeries(samples=a, fs=10.) # 1-sec, 10 samples

        # beyond end
        assert_raises(IndexError, ts.__getitem__, index=slice(0,11))
        assert_raises(ValueError, ts.__getitem__, index=slice(0.0,1.1))
        # before start
        assert_raises(ValueError, ts.__getitem__, index=slice(-0.1,0.1))
        # both
        assert_raises(ValueError, ts.__getitem__, index=slice(-0.1,1.1))
        # step not allowed
        assert_raises(IndexError, ts.__getitem__, index=slice(0,1,1))
        # wrong dimension
        assert (ts[:].samples.get()==ts.__getitem__(slice(None,None,None)).samples.get()).all()
        assert_raises(IndexError,ts.__getitem__, (slice(None,None,None),slice(None,None,None)))

        ## correct indexing
        a = np.random.rand(10).astype(np.float64)
        ts = UniformTimeSeries(samples=a, fs=10.) # 1-sec, 10 samples
        assert ts[1:2].ntimesamples == 1
        assert ts[0.1:0.1+ts.dt].ntimesamples == 1
        assert ts[1:10].ntimesamples == 9
        assert ts[0.1:0.1].ntimesamples == 0
        assert ts[1:1].ntimesamples == 0
        assert ts[1:].ntimesamples == ts[1:ts.ntimesamples].ntimesamples
        assert ts[:5].ntimesamples == ts[0:5].ntimesamples
        assert ts[0:-1].ntimesamples == ts[0:ts.ntimesamples-1].ntimesamples
        assert ts.ntimesamples == ts[0.0:ts.duration].ntimesamples
        assert ts.ntimesamples == ts[0:ts.duration].ntimesamples

    def test_slicing_h5array(self):
        "UniformTimeSeries: slicing h5array"
        # slicing should always return a UniformTimeSeries with a numpy array
        # sample type. A lot of other tests are covered by test_slicing_ndarray
        # above
        for node in iter_arnodes(self.testarrays, dimension='1D'):
            if node[:].dtype in goodarraydtypes:
                h5s = UniformTimeSeries(samples=node, fs=1.0, copy=False)
                assert h5s.sampletype == type(node)
                ns = h5s[:]
                assert ns.sampletype == np.ndarray

    def test__to_samplesslice(self):
        "UniformTimeSeries: _to_samplesslice "
        ## partly redundant with test_slicing_ndarray above

        ts = UniformTimeSeries(samples=[1,2,3], fs=1.0)
        # allow single ints
        for i in int_to_pints(0):
            assert ts[i].samples.get(copy=True) == np.array(1)
        for i in int_to_nints(0):
            assert ts[i].samples.get(copy=True) == np.array(1)
        # allow int slice
        start, end = (0, 2)
        for istart in int_to_pints(start):
            for iend in int_to_pints(end):
                assert (ts[istart:iend].samples.get(copy=True) == np.array([1,2])).all()
        # allow floats
        start,end = ts.timerange
        for fstart in float_to_floats(start):
            for fend in float_to_floats(end):
                assert (ts[fstart:fend].samples.get(copy=True) == np.array([1,2,3])).all()
        # do not allow single float
        for f in float_to_floats(0.0):
            assert_raises(IndexError, ts.__getitem__, index=f)
        # do not allow any steps
        for i in int_to_pints(1):
            assert_raises(IndexError, ts.__getitem__, index=slice(0,2,i))
        for f in float_to_floats(1.0):
            assert_raises(IndexError, ts.__getitem__, index=slice(0,2,f))
        # do not allow extended slicing
        assert_raises(IndexError, ts.__getitem__, index=[1])
        assert_raises(IndexError, ts.__getitem__, index=np.array([1]))
        assert_raises(IndexError, ts.__getitem__, index=np.array([1.0]))
        assert_raises(IndexError, ts.__getitem__, index=[1,2])
        assert_raises(IndexError, ts.__getitem__, index=np.array([1,2]))
        assert_raises(IndexError, ts.__getitem__, index=np.array([1.0,2.0]))
        # do not allow None
        assert_raises(IndexError, ts.__getitem__, index=None)
        # allow end befor start
        start,end = ts.timerange
        assert ts[2:1].ntimesamples == 0
        assert ts[end:start].ntimesamples == 0

    def test_get_samples_ndarray_copyarg(self):
        "UniformTimeSeries: get_samples numpy array"
        ts = UniformTimeSeries(samples=[1,2,3], fs=1.0)
        # copy = True is default
        assert (ts.samples.get() == ts.samples[:].copy()).all()
        assert not (ts.samples.get().base is ts.samples.array)
        # copy = True
        assert (ts.samples.get(copy=True) == ts.samples[:].copy()).all()
        assert not (ts.samples.get().base is ts.samples.array)
        # copy = False
        assert (ts.samples.get(copy=False) == ts.samples[:]).all()
        assert (ts.samples.get(copy=False).base is ts.samples.array)

    def test_get_samples_h5array_copyarg(self):
        "UniformTimeSeries: get_samples h5 array"
        for node in iter_arnodes(self.testarrays, dimension='1D'):
            if node[:].dtype in goodarraydtypes:
                ts = UniformTimeSeries(samples=node, fs=1.0, copy=False)
                # copy = True is default
                assert (ts.samples.get() == ts.samples[:].copy()).all()
                assert not (ts.samples.get().base is ts.samples.h5array)
                # copy = True
                assert (ts.samples.get(copy=True) == ts.samples[:].copy()).all()
                assert not (ts.samples.get().base is ts.samples.h5array)
                # copy = False should still produce a copy
                assert (ts.samples.get(copy=False) == ts.samples[:]).all()
                assert not (ts.samples.get(copy=False).base is ts.samples.h5array)
    
##    # do we need this ?
##    def test_replace_samples(self):
##        "UniformTimeSeries: replace samples"
##        # readonly is False
##        ts = UniformTimeSeries(samples=[1,2,3], fs=1.0, dtype=np.int, readonly=False)
##        samples = ts.samples.array
##        ts._replace_samples(samples=[4,5,6])
##        assert (ts.samples.get() == np.array([4,5,6])).all()
##        assert (ts.samples.array is samples)
##        # readonly is True
##        ts = UniformTimeSeries(samples=[1,2,3], fs=1.0, dtype=np.int, readonly=True)
##        samples = ts.samples.array
##        assert_raises(WriteError,ts._replace_samples, samples=[4,5,6])
##                            
        
    def test_append_ndarray(self):
        "UniformTimeSeries: append"
        ts1 = UniformTimeSeries(samples=[1,2,3], fs=1400.0, starttime=5.0,
                                origintime=-3.0)
        ts2 = UniformTimeSeries(samples=[4,5,6], fs=1400.0, starttime=6.0,
                                origintime=-2.0)
        ts1.append(ts2)
        assert (ts1.samples.array[:] == np.array([1,2,3,4,5,6])).all()
        assert ts1.starttime == 5.0
        assert ts1.origintime == -3.0
        assert ts1.ntimesamples == 6
        # different dtype
        ts2 = UniformTimeSeries(samples=[4.,5.,6.], fs=1400.0, starttime=6.0,
                                origintime=-2.0)
        assert_raises(TypeError,ts1.append, ts2)
        # different shape
        ts2 = MultiChannelUniformTimeSeries(
                samples=np.linspace(0,1.0,2800).reshape(1400,2),
                fs=1400.0, channelaxis=1)
        assert_raises(TypeError,ts1.append, ts2)
        # different type
        for seq in ([1,2], (1,2), np.array([1,2])):
            assert_raises(TypeError,ts1.append, seq)

    def test_append_h5array(self):
        "UniformTimeSeries: append h5array"
        for node in iter_arnodes(self.testarrays, dimension='1D'):
            if node[:].dtype in goodarraydtypes:
                typename, dim, artype = node._v_name.split('_')
                if artype == 'EArray':
                    ts1 = UniformTimeSeries(samples=node, fs=1.0, copy=False,
                            starttime=6.0, origintime=-2.0)
                    # correct appends
                    ts2 = UniformTimeSeries(samples=[4,5,6], fs=1.0,
                            dtype=node[:].dtype,starttime=6.0, origintime=-2.0)
                    ts1.append(ts2)
                    assert ts1.ntimesamples == node.shape[0] # node shape has been updated
                    assert ts1.fs == 1.0
                    assert ts1.starttime == 6.0
                    assert ts1.origintime == -2.0
                    ts1.append(ts1)
                    assert ts1.ntimesamples == node.shape[0] # node shape has been updated
                    assert ts1.fs == 1.0
                    assert ts1.starttime == 6.0
                    assert ts1.origintime == -2.0
                    # wrong signal type
                    samples = [[4,5,6],[4,5,6]]
                    ts2 = MultiChannelUniformTimeSeries(samples=samples,
                        fs=1.0, channelaxis=0, dtype=node[:].dtype,
                        starttime=6.0, origintime=-2.0)
                    assert_raises(TypeError,ts1.append, ts2)


    def test_copy(self):
        "UniformTimeSeries: copy"
        s = UniformTimeSeries(samples=np.arange(10.), fs=1.0)
        cs = s.copy()
        assert cs.samples.array.base != s.samples.array
        assert cs.ntimesamples == s.ntimesamples
        assert cs.starttime == s.starttime
        assert cs.origintime == s.origintime
        assert cs.endtime == s.endtime
        assert cs.fs == s.fs
        
    #def test_saveh5(self):
        #"UniformTimeSeries: saveh5"
        #f = self.temph5file
        ## some unlikely parameters
        #samples = np.array([0.07436005,1.62218437,2.39154842,3.55754441])
        #fs =  48.443798
        #starttime = 0.55754441
        #origintime = 0.02353585
        #dtypes = floattypes + inttypes + [np.complex128]
        #for j, extensible in enumerate((False, True)):
            #for i,dtype in enumerate(dtypes):
                #csamples = samples.astype(dtype)
                #ts =  UniformTimeSeries(samples=csamples, fs=fs, starttime=starttime,
                                        #origintime=origintime, dtype=dtype)
                #s = ts.saveh5(f.root, "uts"+"_%d_%d"%(i,j), extensible=extensible)
                #arnode = s.samples.h5array
                #assert (arnode[:] == csamples).all()
                #assert arnode.attrs.fs == fs
                #assert arnode.attrs.starttime == starttime
                #assert arnode.attrs.origintime == origintime
                #assert arnode.dtype == csamples.dtype
            
        

class test_MultiChannelUniformTimeSeries(TestCase):

    def setUp(self):
        f = NamedTemporaryFile()
        self.temph5filename = f.name
        f.close()
        self.temph5file = tables.openFile(self.temph5filename,'w')
        self.testarrays = create_h5arrays(self.temph5file)

    def tearDown(self):
        self.temph5file.close()
        os.remove(self.temph5filename)

    def test_slicingmc(self):
        "MultiChannelUniformTimeSeries: slicing"
        # a lot of aspects of slicing are already tested in UniformTimeSeries
        # and MultiChannel (different file)

        # first index is for samples, second for channels
        channelnames = ['a','b','c']
        samples=np.arange(30.).reshape(10,3)
        for channelaxis, ar in zip((1,0),(samples,samples.T)):
            s = MultiChannelUniformTimeSeries(samples=ar, fs=1.0,
                            channelaxis=channelaxis, channels=channelnames)
            assert s.nchannels == 3
            assert s.ntimesamples == 10
            assert s[:,:2].ntimesamples == 10
            assert s[:,:2].nchannels == 2
            assert s[:5,:].ntimesamples == 5
            assert s[:5,:].nchannels == 3
            assert s[:5].ntimesamples == 5
            assert s[:5].nchannels == 3
            assert s[:5,:2].ntimesamples == 5
            assert s[:5,:2].nchannels == 2
            assert (s.channelnames==np.array(channelnames)).all()
            assert (s[:,[2,1]].samples.get()==s[:,['c','b']].samples.get()).all()
            assert (s[:,:].samples.get()==s.__getitem__((slice(None,None,None),slice(None,None,None))).samples.get()).all()
            assert_raises(IndexError,s.__getitem__, (slice(None,None,None),slice(None,None,None),slice(None,None,None)))

    def test_append_ndarray(self):
        "MultiChannelUniformTimeSeries: append ndarray"
        ts1 = MultiChannelUniformTimeSeries(
                samples=[[1,2,3],[1,2,3]],
                fs=1400.0, channelaxis=0, dtype=np.int64, starttime=5.0,
                                origintime=-3.0)
        ts2 = MultiChannelUniformTimeSeries(
                samples=[[4,5,6],[4,5,6]],
                fs=1400.0, channelaxis=0, dtype=np.int64, starttime=6.0,
                                origintime=-2.0)
        ts1.append(ts2)
        assert (ts1.samples.array[:] == np.array([[1,2,3,4,5,6],[1,2,3,4,5,6]])).all()
        assert ts1.starttime == 5.0
        assert ts1.origintime == -3.0
        assert ts1.ntimesamples == 6
        assert ts1.nchannels == 2
        ts2 = MultiChannelUniformTimeSeries(
                samples=[[4,5],[4,5]],
                fs=1400.0, channelaxis=1, dtype=np.int64, starttime=6.0,
                                origintime=-2.0)
        ts1.append(ts2)
        assert ts1.ntimesamples == 8
        assert ts1.nchannels == 2

        # different dtype
        ts2 = MultiChannelUniformTimeSeries(
                samples=[[4.,5.,6.],[4.,5.,6.]],
                fs=1400.0, channelaxis=0, dtype=np.float64, starttime=6.0,
                                origintime=-2.0)
        assert_raises(TypeError,ts1.append, ts2)
        # different shape
        ts2 = MultiChannelUniformTimeSeries(
                samples=[[4,5,6],[4,5,6]],
                fs=1400.0, channelaxis=1, dtype=np.int64, starttime=6.0,
                                origintime=-2.0)
        assert_raises(TypeError,ts1.append, ts2)
        ts2 = UniformTimeSeries(samples=[4,5,6], fs=1400.0, dtype=np.int64,
                                starttime=6.0, origintime=-2.0)
        assert_raises(TypeError,ts1.append, ts2)
        # different type
        for seq in [[[1,2],[1,2]], ((1,2),(1,2)), np.array([[1,2],[1,2]])]:
            assert_raises(TypeError,ts1.append, seq)

    def test_append_h5array(self):
        "MultiChannelUniformTimeSeries: append h5array"
        for node in iter_arnodes(self.testarrays, dimension='2D'):
            if node[:].dtype in goodarraydtypes:
                typename, dim, artype = node._v_name.split('_')
                if artype == 'EArray':
                    ts1 = MultiChannelUniformTimeSeries(samples=node, fs=1.0,
                                copy=False, channelaxis=0,starttime=6.0,
                                origintime=-2.0)
                    # correct appends
                    ts2 = MultiChannelUniformTimeSeries(
                        samples=np.array([node.shape[0]*[4,5,6]]).reshape(node.shape[0],-1),
                        fs=1.0, channelaxis=0, dtype=node[:].dtype,
                        starttime=6.0, origintime=-2.0)
                    ts1.append(ts2)
                    assert ts1.ntimesamples == node.shape[1] # node shape has been updated
                    assert ts1.nchannels == node.shape[0]
                    assert ts1.fs == 1.0
                    assert ts1.starttime == 6.0
                    assert ts1.origintime == -2.0
                    ts1.append(ts1)
                    assert ts1.ntimesamples == node.shape[1] # node shape has been updated
                    assert ts1.nchannels == node.shape[0]
                    assert ts1.fs == 1.0
                    assert ts1.starttime == 6.0
                    assert ts1.origintime == -2.0
                    # different channel axis, but nchannels correct
                    ts2 = ts2.transpose()
                    ts1.append(ts2)
                    assert ts1.ntimesamples == node.shape[1] # node shape has been updated
                    assert ts1.nchannels == node.shape[0]
                    assert ts1.fs == 1.0
                    assert ts1.starttime == 6.0
                    assert ts1.origintime == -2.0
                    # one channel too many
                    samples = np.array([(node.shape[0]+1)*[4,5,6]]).reshape((node.shape[0]+1),-1)
                    ts2 = MultiChannelUniformTimeSeries(samples=samples,
                        fs=1.0, channelaxis=0, dtype=node[:].dtype,
                        starttime=6.0, origintime=-2.0)
                    assert_raises(TypeError,ts1.append, ts2)
                    # wrong signal type
                    ts2 = UniformTimeSeries(samples=np.array([4,5,6]), fs=1.0,
                            dtype=node[:].dtype, starttime=6.0, origintime=-2.0)
                    assert_raises(TypeError,ts1.append, ts2)

    def test_copy(self):
        "MultiChannelUniformTimeSeries: copy"
        ar = np.arange(20.).reshape(10,2)
        for channelaxis, ar in zip((1,0),(ar,ar.T)):
            s = MultiChannelUniformTimeSeries(samples=ar, fs=1.0, channelaxis=1)
            cs = s.copy()
            assert cs.samples.array.base != s.samples.array
            assert cs.channelnames.base != s.channelnames
            assert cs.ntimesamples == s.ntimesamples
            assert cs.nchannels == s.nchannels
            assert cs.channelaxis == s.channelaxis
            assert cs.starttime == s.starttime
            assert cs.origintime == s.origintime
            assert cs.endtime == s.endtime
            assert cs.fs == s.fs
            assert cs.dtype == s.dtype
            
    #def test_saveh5(self):
        #"MultiChannelUniformTimeSeries: saveh5"
        #f = self.temph5file
        ## some unlikely parameters
        #samples0 = np.array([[0.07436005,1.62218437,2.39154842,3.55754441],
                            #[4.07436005,5.62218437,6.39154842,7.55754441]])
        #samples1 = samples0.T
        #fs = 48.443798
        #starttime = 0.55754441
        #origintime = 0.02353585
        #dtypes = floattypes + inttypes + [np.complex128]
        ## chanelaxis == 0
        #channelnames = np.array(['chan1','chan2'])
        #for channelaxis, samples in zip((0,1),(samples0,samples1)):
            #for i,dtype in enumerate(dtypes):
                #csamples = samples.astype(dtype)
                #ts =  MultiChannelUniformTimeSeries(samples=csamples, fs=fs, 
                            #starttime=starttime, origintime=origintime, dtype=dtype,
                            #channelnames=channelnames, channelaxis=channelaxis)
                #s = ts.saveh5(f.root, "MultiChannelUniformTimeSeries"+"_%d_%d"%(i,channelaxis),
                              #channelaxis=channelaxis)
                #arnode = s.samples.h5array
                #assert (arnode[:] == csamples).all()
                #assert arnode.attrs.fs == fs
                #assert arnode.attrs.starttime == starttime
                #assert arnode.attrs.origintime == origintime
                #assert arnode.dtype == csamples.dtype
                #assert (arnode.attrs.channelnames == channelnames).all()
                #assert arnode.attrs.channelaxis == channelaxis
     
    def test_transpose(self):
        "MultiChannelUniformTimeSeries: transpose"
        samples = {0: [[1,2,3],[4,5,6]], 1: [[1,4],[2,5],[3,6]]}
        dtypes = goodarraydtypes
        fs = 48.443798
        starttime = 0.55754441
        origintime = 0.02353585
        channelnames = ['ch1','ch2']
        for dtype in dtypes:
            for readonly in [True, False]:
                for channelaxis in [0,1]:
                    mc = MultiChannelUniformTimeSeries(samples=samples[channelaxis], fs=fs, 
                            starttime=starttime, origintime=origintime, dtype=dtype,
                            channels=channelnames, channelaxis=channelaxis)
                    tmc = mc.transpose(mc.samples.axisorder[::-1])
                    assert (tmc.samples.get() == mc.samples.get().T).all()
                    assert tmc.fs == mc.fs
                    assert tmc.starttime == mc.starttime
                    assert tmc.origintime == mc.origintime
                    assert tmc.dtype == mc.dtype
                    assert (tmc.channelnames == channelnames).all()
                    assert tmc.channelaxis == np.abs(mc.channelaxis - 1)
                    assert tmc.readonly == mc.readonly
                
            
class test_EpochedUniformTimeSeries(TestCase):
    
    def setUp(self):
        "do nothing"
        pass
    
    def tearDown(self):
        "do nothing"
        pass   

    def test_transpose(self):
        "EpochedUniformTimeSeries: transpose"
        samples = {0: [[1,2,3],[4,5,6]], 1: [[1,4],[2,5],[3,6]]}
        dtypes = goodarraydtypes
        fs = 48.443798
        origintime = 0.02353585
        epochs = np.zeros(2, dtype=[('starttime',np.float64)])
        epochs['starttime'][:] = [9.876, 30.765]
        duration = 1.0
        for dtype in dtypes:
            for readonly in [True, False]:
                for epochaxis in [0,1]:
                    es = EpochedUniformTimeSeries(samples=samples[epochaxis], fs=fs, 
                            origintime=origintime, dtype=dtype,
                            epochs=epochs, epochaxis=epochaxis)
                    tes = es.transpose(es.samples.axisorder[::-1])
                    assert (tes.samples.get() == es.samples.get().T).all()
                    assert tes.fs == es.fs
                    assert tes.origintime == es.origintime
                    assert tes.dtype == es.dtype
                    assert (tes.epochs == epochs).all()
                    assert tes.epochaxis == np.abs(es.epochaxis - 1)
                    assert tes.readonly == es.readonly
            
            
class test_EpochedMultiChannelUniformTimeSeries(TestCase):

    def setUp(self):
        "do nothing"
        pass
    
    def tearDown(self):
        "do nothing"
        pass
    
    def test_slicingmc(self):
        "EpochedMultiChannelUniformTimeSeries: slicing"
        # a lot of aspects of slicing are already tested in UniformTimeSeries
        # and MultiChannel (different file)

        # first index is for samples, second for channels, third for epochs
        channelnames = ['a','b','c']
        samples=np.arange(150.).reshape(3,5,10)
        axes = np.array(['channel','epoch','time'])
        epochs = np.zeros(5, dtype={'names': ['starttime','event'],'formats':[np.float,'|S1']})
        epochs['starttime'] = np.linspace(0.,10.,5)
        epochs['event'] = ['v','w','x','y','z']
        
        for t in ((0,1,2),(0,2,1),(1,2,0),(1,0,2),(2,0,1),(2,1,0)):
            channelaxis = np.argwhere(np.array(t)==0)[0][0]
            epochaxis = np.argwhere(np.array(t)==1)[0][0]
            s = EpochedMultiChannelUniformTimeSeries(
                samples=np.transpose(samples, t), fs=1.0, 
                channelaxis=channelaxis, epochaxis=epochaxis,
                channels=channelnames,
                epochs = epochs)
            assert s.nchannels == 3
            assert s.ntimesamples == 10
            assert s.nepochs == 5
            assert s[:,:2].ntimesamples == 10
            assert s[:,:2,:3].ntimesamples == 10
            assert s[:5].ntimesamples == 5
            assert s[:5,:2].ntimesamples == 5
            assert s[:5,:2,:3].ntimesamples == 5
            assert s[:,:2].nchannels == 2
            assert s[:,:2,:3].nchannels == 2
            assert s[:5,:2,:3].nchannels == 2
            assert s[:,:,:3].nepochs == 3
            assert s[:,:2,:3].nepochs == 3
            assert s[:5,:2,:3].nepochs == 3         
            assert (s.channelnames==np.array(channelnames)).all()
            assert s[:,[2,1]].nchannels == 2
            assert (s[:,[2,1]].samples.get()==s[:,['c','b']].samples.get()).all()
            assert (s[:,[2,1,0]][:,[2,1,0]].samples.get()==s.samples.get()).all()
            assert (s[:,:,[4,3]].epochs['event'] == np.array(['z','y'])).all()          
            
            assert (s[:,:,:].samples.get()==s.__getitem__((slice(None,None,None),slice(None,None,None),slice(None,None,None))).samples.get()).all()
            assert_raises(IndexError,s.__getitem__, (slice(None,None,None),slice(None,None,None),slice(None,None,None),slice(None,None,None)))
            

def iter_arnodes(testarrays, dimension=None):

    for node in testarrays.root._f_iterNodes():
        typename, dim, artype = node._v_name.split('_')
        if dim is not None:
            if dimension == dim:
                yield node
        else:
                yield node


if __name__ == '__main__':
    unittest.main()
