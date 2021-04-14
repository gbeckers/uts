from numpy.testing import TestCase, assert_raises, run_module_suite, \
     assert_array_almost_equal, assert_almost_equal, assert_equal, \
     assert_array_equal

import numpy as np
from uts import UniformTime, UniformTimeSeries, \
    MultiChannelUniformTimeSeries, EpochedMultiChannelUniformTimeSeries
import uts.processing.stats as stats

class test_Mean(TestCase):

    tss = UniformTimeSeries(samples=np.linspace(-1.,1.,200), fs=200.)
    mctss1 = MultiChannelUniformTimeSeries(
        samples=np.linspace(-1.,1.,400).reshape(2,200), fs=200., channelaxis=0)
    mctss2 = MultiChannelUniformTimeSeries(
        samples=np.linspace(-1.,1.,400).reshape(2,200), fs=200., channelaxis=1)

    epochs = np.empty(3, dtype=[('starttime',np.float64)])
    epochs['starttime'] = [1.,2.,3.]
    emctss1 = EpochedMultiChannelUniformTimeSeries(
        samples=np.linspace(-1.,1.,1200).reshape(3,2,200), fs=200., 
        epochs=epochs, channelaxis=1, epochaxis=0)
    emctss2 = EpochedMultiChannelUniformTimeSeries(
        samples=np.linspace(-1.,1.,1200).reshape(3,2,200), fs=200.,
        epochs=epochs[:2], channelaxis=0, epochaxis=1)
    emctss3 = EpochedMultiChannelUniformTimeSeries(
        samples=np.linspace(-1.,1.,1200).reshape(3,200,2), fs=200., 
        epochs=epochs, channelaxis=1, epochaxis=0)
    
    def test_tss(self):
        """mean of time sampled signal"""
        m = stats.mean(self.tss, framesize=300)
        assert_almost_equal(m, 0.0)
        m = stats.mean(self.tss, axis='time', framesize=300)
        assert_almost_equal(m, 0.0)
        
    def test_tss_chunked(self):
        """mean of time sampled signal chunked"""
        m = stats.mean(self.tss, framesize=10)
        assert_almost_equal(m, 0.0)    
        m = stats.mean(self.tss, axis='time', framesize=10)
        assert_almost_equal(m, 0.0)    

    def test_mctss(self):
        """mean of multichannel time sampled signal"""
        m = stats.mean(self.mctss1, framesize=300)
        assert_array_almost_equal(m, [-0.50125313,  0.50125313])
        m = stats.mean(self.mctss1, axis='time', framesize=300)
        assert_array_almost_equal(m, [-0.50125313,  0.50125313])
        m = stats.mean(self.mctss2, axis='channel', framesize=300)
        assert_array_almost_equal(m.samples.get(), [-0.50125313,  0.50125313])
        
    def test_mctss_chunked(self):
        """mean of multichannel time sampled signal chunked"""
        m = stats.mean(self.mctss1, framesize=10)
        assert_array_almost_equal(m, [-0.50125313,  0.50125313])    
        m = stats.mean(self.mctss1, axis='time', framesize=10)
        assert_array_almost_equal(m, [-0.50125313,  0.50125313])  
        m = stats.mean(self.mctss2, axis='channel', framesize=10)
        assert_array_almost_equal(m.samples.get(), [-0.50125313,  0.50125313])
        
    def test_emctss(self):
        """mean of epoched multichannel time sampled signal"""
        m = stats.mean(self.emctss1, framesize=300)
        assert_array_almost_equal(m, [[-0.83402836, -0.50041701],
                                      [-0.16680567,  0.16680567],
                                      [ 0.50041701,  0.83402836]])
        m = stats.mean(self.emctss1, axis='time', framesize=300)
        assert_array_almost_equal(m, [[-0.83402836, -0.50041701],
                                      [-0.16680567,  0.16680567],
                                      [ 0.50041701,  0.83402836]])
        m = stats.mean(self.emctss2, axis='time', framesize=300)
        assert_array_almost_equal(m, [[-0.83402836, -0.50041701],
                                      [-0.16680567,  0.16680567],
                                      [ 0.50041701,  0.83402836]])
        m = stats.mean(self.emctss3, axis='channel', framesize=300)
        assert_array_almost_equal(m.samples.get(), [[-0.66805671, -0.66638866],
                                      [-0.00083403,  0.00083403],
                                      [ 0.66638866,  0.66805671]])
    
    def test_emctss_chunked(self):
        """mean of epoched multichannel time sampled signal chunked"""
        m = stats.mean(self.emctss1, framesize=10)
        assert_array_almost_equal(m, [[-0.83402836, -0.50041701],
                                      [-0.16680567,  0.16680567],
                                      [ 0.50041701,  0.83402836]])
        m = stats.mean(self.emctss1, axis='time', framesize=10)
        assert_array_almost_equal(m, [[-0.83402836, -0.50041701],
                                      [-0.16680567,  0.16680567],
                                      [ 0.50041701,  0.83402836]])
        m = stats.mean(self.emctss2, axis='time', framesize=10)
        assert_array_almost_equal(m, [[-0.83402836, -0.50041701],
                                      [-0.16680567,  0.16680567],
                                      [ 0.50041701,  0.83402836]])
        m = stats.mean(self.emctss3, axis='channel', framesize=10)
        assert_array_almost_equal(m.samples.get(), [[-0.66805671, -0.66638866],
                                      [-0.00083403,  0.00083403],
                                      [ 0.66638866,  0.66805671]])
    def test_wrong_input(self):
        """mean wrong input type"""
        assert_raises(TypeError, stats.mean, [1.,2.,3.])
        assert_raises(TypeError, stats.mean, np.array([1.,2.,3.]))
        assert_raises(TypeError, stats.mean, 1.)
    
    def test_wrong_axis_arg(self):
        """mean wrong axis arg"""
        assert_raises(ValueError, stats.mean, self.tss, axis='frequency')
    
    def test_wrong_framesize_arg(self):
        """mean wrong framesize arg"""
        assert_raises(ValueError, stats.mean, self.tss, framesize=0)
        assert_raises(TypeError, stats.mean, self.tss, framesize=0.)
        assert_raises(ValueError, stats.mean, self.tss, framesize=-1)
        assert_raises(TypeError, stats.mean, self.tss, framesize=-1.)
        assert_raises(TypeError, stats.mean, self.tss, framesize=1.)
    

class test_Max(TestCase):
    
    tss = UniformTimeSeries(samples=[-1.0, -0.5, 0.0, 0.5, 1.0], fs=5.)
    mctss1 = MultiChannelUniformTimeSeries(samples=[[-1.0, -0.5, 0.0, 0.5, 1.0],
                                                 [-1.0, -0.5, 0.0, 0.5, 1.0]],
                                        fs=5.,channelaxis=0)
    mctss2 = MultiChannelUniformTimeSeries(samples=[[1.0, 0.5, 0.0, -0.5, -1.0],
                                                 [-1.0, -0.5, 0.0, 0.5, 1.0]],
                                        fs=5.,channelaxis=1)
    
    def test_tss(self):
        """max of time sampled signal"""
        m = stats.max(self.tss, framesize=300)
        assert_equal(m, 1.0)
        m = stats.max(self.tss, axis='time', framesize=300)
        assert_equal(m, 1.0)
        
    def test_tss_chunked(self):
        """max of time sampled signal chunked"""
        m = stats.max(self.tss, framesize=2)
        assert_equal(m, 1.0)
        m = stats.max(self.tss, axis='time', framesize=2)
        assert_equal(m, 1.0)
        
    def test_mctss(self):
        """max of multichannel time sampled signal"""
        m = stats.max(self.mctss1, framesize=300)
        assert_array_equal(m, [1.0, 1.0])
        m = stats.max(self.mctss1, axis='time', framesize=300)
        assert_array_equal(m, [1.0, 1.0])
        m = stats.max(self.mctss2, axis='time', framesize=300)
        assert_array_equal(m, [1.0, 0.5, 0.0, 0.5, 1.0])      
        
    def test_mctss_chunked(self):
        """max of multichannel time sampled signal chunked"""
        m = stats.max(self.mctss1, framesize=2)
        assert_array_equal(m, [1.0, 1.0])
        m = stats.max(self.mctss1, axis='time', framesize=2)
        assert_array_equal(m, [1.0, 1.0])
        m = stats.max(self.mctss2, axis='time', framesize=2)
        assert_array_equal(m, [1.0, 0.5, 0.0, 0.5, 1.0])
    
    def test_wrong_input(self):
        """max wrong input type"""
        assert_raises(TypeError, stats.max, [1.,2.,3.])
        assert_raises(TypeError, stats.max, np.array([1.,2.,3.]))
        assert_raises(TypeError, stats.max, 1.)
    
    def test_wrong_axis_arg(self):
        """max wrong axis arg"""
        assert_raises(ValueError, stats.max, self.tss, axis='frequency')
    
    def test_wrong_framesize_arg(self):
        """max wrong framesize arg"""
        assert_raises(ValueError, stats.max, self.tss, framesize=0)
        assert_raises(TypeError, stats.max, self.tss, framesize=0.)
        assert_raises(ValueError, stats.max, self.tss, framesize=-1)
        assert_raises(TypeError, stats.max, self.tss, framesize=-1.)
        assert_raises(TypeError, stats.max, self.tss, framesize=1.)
    
class test_Min(TestCase):
    
    tss = UniformTimeSeries(samples=[-1.0, -0.5, 0.0, 0.5, 1.0], fs=5.)
    mctss1 = MultiChannelUniformTimeSeries(samples=[[-1.0, -0.5, 0.0, 0.5, 1.0],
                                                 [-1.0, -0.5, 0.0, 0.5, 1.0]],
                                        fs=5.,channelaxis=0)
    mctss2 = MultiChannelUniformTimeSeries(samples=[[1.0, 0.5, 0.0, -0.5, -1.0],
                                                 [-1.0, -0.5, 0.0, 0.5, 1.0]],
                                        fs=5.,channelaxis=1)
    
    def test_tss(self):
        """min of time sampled signal"""
        m = stats.min(self.tss, framesize=300)
        assert_equal(m, -1.0)
        m = stats.min(self.tss, axis='time', framesize=300)
        assert_equal(m, -1.0)
        
    def test_tss_chunked(self):
        """min of time sampled signal chunked"""
        m = stats.min(self.tss, framesize=2)
        assert_equal(m, -1.0)
        m = stats.min(self.tss, axis='time', framesize=2)
        assert_equal(m, -1.0)
        
    def test_mctss(self):
        """min of multichannel time sampled signal"""
        m = stats.min(self.mctss1, framesize=300)
        assert_array_equal(m, [-1.0, -1.0])
        m = stats.min(self.mctss1, axis='time', framesize=300)
        assert_array_equal(m, [-1.0, -1.0])
        m = stats.min(self.mctss2, axis='time', framesize=300)
        assert_array_equal(m, [-1.0, -0.5, 0.0, -0.5, -1.0])      
        
    def test_mctss_chunked(self):
        """min of multichannel time sampled signal chunked"""
        m = stats.min(self.mctss1, framesize=2)
        assert_array_equal(m, [-1.0, -1.0])
        m = stats.min(self.mctss1, axis='time', framesize=2)
        assert_array_equal(m, [-1.0, -1.0])
        m = stats.min(self.mctss2, axis='time', framesize=2)
        assert_array_equal(m, [-1.0, -0.5, 0.0, -0.5, -1.0])
    
    def test_wrong_input(self):
        """min wrong input type"""
        assert_raises(TypeError, stats.min, [1.,2.,3.])
        assert_raises(TypeError, stats.min, np.array([1.,2.,3.]))
        assert_raises(TypeError, stats.min, 1.)
    
    def test_wrong_axis_arg(self):
        """min wrong axis arg"""
        assert_raises(ValueError, stats.min, self.tss, axis='frequency')
    
    def test_wrong_framesize_arg(self):
        """min wrong framesize arg"""
        assert_raises(ValueError, stats.min, self.tss, framesize=0)
        assert_raises(TypeError, stats.min, self.tss, framesize=0.)
        assert_raises(ValueError, stats.min, self.tss, framesize=-1)
        assert_raises(TypeError, stats.min, self.tss, framesize=-1.)
        assert_raises(TypeError, stats.min, self.tss, framesize=1.)
        

class test_Ptp(TestCase):
    
    tss = UniformTimeSeries(samples=[-1.0, -0.5, 0.0, 0.5, 1.0], fs=5.)
    mctss1 = MultiChannelUniformTimeSeries(samples=[[-1.0, -0.5, 0.0, 0.5, 1.0],
                                                 [-1.0, -0.5, 0.0, 0.5, 1.0]],
                                        fs=5.,channelaxis=0)
    mctss2 = MultiChannelUniformTimeSeries(samples=[[1.0, 0.5, 0.0, -0.5, -1.0],
                                                 [-1.0, -0.5, 0.0, 0.5, 1.0]],
                                        fs=5.,channelaxis=1)
    
    def test_tss(self):
        """ptp of time sampled signal"""
        m = stats.ptp(self.tss, framesize=300)
        assert_equal(m, 2.0)
        m = stats.ptp(self.tss, axis='time', framesize=300)
        assert_equal(m, 2.0)
        
    def test_tss_chunked(self):
        """ptp of time sampled signal chunked"""
        m = stats.ptp(self.tss, framesize=2)
        assert_equal(m, 2.0)
        m = stats.ptp(self.tss, axis='time', framesize=2)
        assert_equal(m, 2.0)
        
    def test_mctss(self):
        """ptp of multichannel time sampled signal"""
        m = stats.ptp(self.mctss1, framesize=300)
        assert_array_equal(m, [2.0, 2.0])
        m = stats.ptp(self.mctss1, axis='time', framesize=300)
        assert_array_equal(m, [2.0, 2.0])
        m = stats.ptp(self.mctss2, axis='time', framesize=300)
        assert_array_equal(m, [2.0, 1.0, 0.0, 1.0, 2.0])      
        
    def test_mctss_chunked(self):
        """ptp of multichannel time sampled signal chunked"""
        m = stats.ptp(self.mctss1, framesize=2)
        assert_array_equal(m, [2.0, 2.0])
        m = stats.ptp(self.mctss1, axis='time', framesize=2)
        assert_array_equal(m, [2.0, 2.0])
        m = stats.ptp(self.mctss2, axis='time', framesize=2)
        assert_array_equal(m, [2.0, 1.0, 0.0, 1.0, 2.0])
    
    def test_wrong_input(self):
        """ptp wrong input type"""
        assert_raises(TypeError, stats.ptp, [1.,2.,3.])
        assert_raises(TypeError, stats.ptp, np.array([1.,2.,3.]))
        assert_raises(TypeError, stats.ptp, 1.)
    
    def test_wrong_axis_arg(self):
        """ptp wrong axis arg"""
        assert_raises(ValueError, stats.ptp, self.tss, axis='frequency')
    
    def test_wrong_framesize_arg(self):
        """ptp wrong framesize arg"""
        assert_raises(ValueError, stats.ptp, self.tss, framesize=0)
        assert_raises(TypeError, stats.ptp, self.tss, framesize=0.)
        assert_raises(ValueError, stats.ptp, self.tss, framesize=-1)
        assert_raises(TypeError, stats.ptp, self.tss, framesize=-1.)
        assert_raises(TypeError, stats.ptp, self.tss, framesize=1.)
        
        
class test_Rms(TestCase):
    
    samples = np.sin(2*np.pi*np.linspace(0.,1.,44100)*400.)
    tss = UniformTimeSeries(samples=samples, fs=44100.)
    mctss1 = MultiChannelUniformTimeSeries(samples=np.array([samples,samples]),
                                        fs=44100.,channelaxis=0)
    mctss2 = MultiChannelUniformTimeSeries(samples=np.array([samples,samples]).T,
                                        fs=44100.,channelaxis=1)
    
    def test_tss(self):
        """rms of time sampled signal"""
        m = stats.rms(self.tss, framesize=44101)
        assert_almost_equal(m, 0.7070987640574119)
        m = stats.rms(self.tss, axis='time', framesize=44101)
        assert_almost_equal(m, 0.7070987640574119)
        
    def test_tss_chunked(self):
        """rms of time sampled signal chunked"""
        m = stats.rms(self.tss, framesize=2000)
        assert_almost_equal(m, 0.70709552844825596)
        m = stats.rms(self.tss, axis='time', framesize=2000)
        assert_almost_equal(m, 0.70709552844825596)
        
    def test_mctss(self):
        """rms of multichannel time sampled signal"""
        m = stats.rms(self.mctss1, framesize=44101)
        assert_array_almost_equal(m, [0.7070987640574119, 0.7070987640574119])
        m = stats.rms(self.mctss1, axis='time', framesize=44101)
        assert_array_almost_equal(m, [0.7070987640574119, 0.7070987640574119])
        m = stats.rms(self.mctss2, axis='time', framesize=44101)
        assert_array_almost_equal(m, [0.7070987640574119, 0.7070987640574119])      
        
    def test_mctss_chunked(self):
        """rms of multichannel time sampled signal chunked"""
        m = stats.rms(self.mctss1, framesize=2000)
        assert_array_almost_equal(m, [0.70709552844825596,0.70709552844825596])
        m = stats.rms(self.mctss1, axis='time', framesize=2000)
        assert_array_almost_equal(m, [0.70709552844825596,0.70709552844825596])
        m = stats.rms(self.mctss2, axis='time', framesize=2000)
        assert_array_almost_equal(m, [0.70709552844825596,0.70709552844825596]) 
        
    def test_wrong_input(self):
        """rms wrong input type"""
        assert_raises(TypeError, stats.rms, [1.,2.,3.])
        assert_raises(TypeError, stats.rms, np.array([1.,2.,3.]))
        assert_raises(TypeError, stats.rms, 1.)
    
    def test_wrong_axis_arg(self):
        """rms wrong axis arg"""
        assert_raises(ValueError, stats.rms, self.tss, axis='frequency')
    
    def test_wrong_framesize_arg(self):
        """rms wrong framesize arg"""
        assert_raises(ValueError, stats.rms, self.tss, framesize=0)
        assert_raises(TypeError, stats.rms, self.tss, framesize=0.)
        assert_raises(ValueError, stats.rms, self.tss, framesize=-1)
        assert_raises(TypeError, stats.rms, self.tss, framesize=-1.)
        assert_raises(TypeError, stats.rms, self.tss, framesize=1.)
