import os
from numpy.testing import TestCase, assert_raises, run_module_suite, \
     assert_almost_equal, assert_array_almost_equal
import numpy as np
from uts3 import UniformTime, UniformTimeSeries, \
    MultiChannelUniformTimeSeries, EpochedMultiChannelUniformTimeSeries, \
    noise, argmax, convolve, iter_convolve, iter_convolve_info, concatenate, \
    decimate, crosscorrelatetime


class TestConvolve(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass
        
    def test_convolve_UniformTimeSeries(self):
        "UniformTimeSeries: convolve values"
        #'does convolve produce the same samples as numpy convolve?'
        kerneleven = [[1.,2.], (1.,2.), np.array([1.,2.]),[1.,2.,3.,4.,5.,6.]]
        kernelodd = [[1.,2.,3.], (1.,2.,3.), np.array([1.,2.,3.]),[1.,2.,3.,4.,5.,6.,7.]]
        tseven = UniformTimeSeries(samples=np.random.rand(10), fs=1.0)
        tsodd = UniformTimeSeries(samples=np.random.rand(11), fs=1.0)
        for ts in (tseven, tsodd): # odd and even signals
            for signal in kerneleven + kernelodd: # odd and even kernels
                for mode in ('valid', 'same', 'full'): # all possible modes
                    c = convolve(ts, signal, mode=mode)
                    assert np.all(c.samples[:] == np.convolve(ts.samples[:],signal, mode=mode))
                    if mode == 'valid':
                        assert c.ntimesamples == (ts.ntimesamples - len(signal) + 1)
                    elif mode == 'same':
                        assert c.ntimesamples == ts.ntimesamples
                    else: # mode == 'full'
                        assert c.ntimesamples == ts.ntimesamples + len(signal) - 1


    def test_convolve_time_UniformTimeSeries(self):
        "UniformTimeSeries: convolve timing"
        # does convolve produce a result with the right timing?
        s1 = UniformTimeSeries([1,2,3,3,2,1],fs=1.0)
        s1maxtime = 3.0 # time where signal is max; result should be too.
        s2 = UniformTimeSeries([1,2,3,2,1],fs=1.0)
        s2maxtime = 2.5
        a1 = np.ones(3)
        a2 = np.ones(4)
        for mode in ('full','valid','same'):
            for a in (a1,a2):
                for s,t in zip((s1,s2),(s1maxtime,s2maxtime)):
                    b = convolve(s, a, mode=mode)
                    samples = b.samples.get()
                    index = (samples == samples.max())
                    meant = np.mean(b.samplingtimes()[index])
                    assert (meant - t) == 0.0

class TestIterconvolve(TestCase):

    def test_iter_convolve_UniformTimeSeries(self):
        "UniformTimeSeries: iter_convolve"
        "does iter_convolve produce same samples and timing as convolve?"
        s1 = UniformTimeSeries(samples=np.random.rand(20),fs=1.0)
        s2 = UniformTimeSeries(samples=np.random.rand(21),fs=1.0)
        kernels = [[1.],[1.,2.],[1.,2.,3.],[1.,2.,3.,4.]]
        chunksizes = [5,6,7,8,9]
        for s in [s1,s2]:
            for mode in ('full','same','valid'):
                for kernel in kernels:
                    for chunksize in chunksizes:
                        iterreslist = []
                        gen = iter_convolve(s, kernel, chunksize=chunksize, mode=mode)
                        info = iter_convolve_info(s, kernel, chunksize=chunksize, mode=mode)
                        frames = [frame for frame in gen]
                        assert frames[0].ntimesamples == info['firstchunksize']
                        assert info['nmidframes'] == len(frames) - 2
                        assert info['lastchunksize'] == frames[-1].ntimesamples
                        iterres = concatenate(frames)
                        normalres = convolve(s, kernel, mode=mode)
                        assert (normalres.samples.get(copy=True) == iterres.samples.get(copy=True)).all()
                        assert normalres.origintime == iterres.origintime
                        assert normalres.starttime == iterres.starttime
                        assert normalres == iterres

    def test_iter_convolve_MultiChannelUniformTimeSeries(self):
        "MultiChannelUniformTimeSeries: iter_convolve"
        "does iter_convolve produce same samples and timing as convolve?"
        s1 = MultiChannelUniformTimeSeries(samples=np.random.rand(40).reshape(2,20), fs=1.0, channelaxis=0)
        s2 = MultiChannelUniformTimeSeries(samples=np.random.rand(42).reshape(2,21), fs=1.0, channelaxis=0)
        kernels = [[1.],[1.,2.],[1.,2.,3.],[1.,2.,3.,4.]]
        chunksizes = [5,6,7,8,9]
        for s in [s1,s2]:
            for mode in ('full','same','valid'):
                for kernel in kernels:
                    for chunksize in chunksizes:
                        iterreslist = []
                        gen = iter_convolve(s, kernel, chunksize=chunksize, mode=mode)
                        info = iter_convolve_info(s, kernel, chunksize=chunksize, mode=mode)
                        frames = [frame for frame in gen]
                        assert frames[0].ntimesamples == info['firstchunksize']
                        assert info['nmidframes'] == len(frames) - 2
                        assert info['lastchunksize'] == frames[-1].ntimesamples
                        iterres = concatenate(frames)
                        normalres = convolve(s, kernel, mode=mode)
                        assert (normalres.samples.get(copy=True).flatten() == iterres.samples.get(copy=True).flatten()).all()
                        assert normalres.origintime == iterres.origintime
                        assert normalres.starttime == iterres.starttime
                        assert normalres == iterres
                    

class TestDecimate(TestCase):
    
    def test_UniformTimeSeries(self):
        "UniformTimeSeries: decimate"
        s = UniformTimeSeries(samples=np.arange(10.), fs=1.0)
        # normal case where ntimesamples is integer multiple of q
        dec = decimate(s, q=2)
        assert (dec.samples.get() ==  np.array([ 0.5,  2.5,  4.5,  6.5,  8.5])).all()
        assert dec.fs == 0.5
        # trunc at tail
        dec = decimate(s, q=3, trunc='tail')
        assert dec.ntimesamples == 3
        assert (dec.samples.get() == np.array([ 1.,  4.,  7.])).all()
        # trunc tail is default
        dec = decimate(s, q=3)
        assert (dec.samples.get() == np.array([ 1.,  4.,  7.])).all()
        #trunc at head
        dec = decimate(s, q=3, trunc='head')
        assert dec.ntimesamples == 3
        assert (dec.samples.get() == np.array([ 2.,  5.,  8.])).all()
        
    def test_MuiltiChannelUniformTimeSeries(self):
        "MultiChannelUniformTimeSeries: decimate"
        samples=np.arange(20.).reshape(10,2)
        for channelaxis, ar in zip((1,0),(samples,samples.T)):
            s = MultiChannelUniformTimeSeries(samples=ar, fs=1.0,
                                              channelaxis=channelaxis)
            # normal case where ntimesamples is integer multiple of q
            dec = decimate(s,q=2)
            assert dec.fs == 0.5
            samples = dec.samples.get(copy=True)
            if channelaxis == 0:
                samples = samples.transpose()
            assert_array_almost_equal(samples,
                    np.array([[1.,2.],[5.,6.],[9.,10.],[13.,14.],[ 17.,18.]]))
            # trunc at tail
            dec = decimate(s,q=3, trunc='tail')
            samples = dec.samples.get(copy=True)
            if channelaxis == 0:
                samples = samples.transpose()
            assert_array_almost_equal(samples,
                            np.array([[2.,3.],[8.,9.],[14.,15.]]))
            # trunc tail is default
            dec = decimate(s,q=3)
            samples = dec.samples.get(copy=True)
            if channelaxis == 0:
                samples = samples.transpose()
            assert_array_almost_equal(samples,
                            np.array([[2.,3.],[8.,9.],[14.,15.]]))
            #trunc at head
            dec = decimate(s, q=3, trunc='head')
            samples = dec.samples.get(copy=True)
            if channelaxis == 0:
                samples = samples.transpose()
            assert_array_almost_equal(samples,
                            np.array([[4.,5.],[10.,11.],[16.,17.]]))


class TestCrosscorrelatetime(TestCase):
    
    def test_UniformTimeSeries_timelag(self):
        "testingcrosscorrelatetime uts time lag"
        n = noise(ntimesamples=51, fs=51.)
        for normalized in (True, False):
            cc = crosscorrelatetime(n,n[10:41],normalized=normalized)
            assert cc.index_to_time(argmax(cc)) == 0.0
            cc = crosscorrelatetime(n,n[12:41],normalized=normalized)
            assert_almost_equal(cc.index_to_time(argmax(cc)), cc.dt)
            assert_array_almost_equal(
                crosscorrelatetime(n,n[12:41],normalized=normalized).samples[:],
                crosscorrelatetime(n[12:41],n,normalized=normalized).samples[:])



