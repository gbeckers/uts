import numpy as np
import time
from uts.core.uniformtime import UniformTimeSeries, EpochedUniformTimeSeries, \
      MultiChannelUniformTimeSeries, EpochedMultiChannelUniformTimeSeries


__all__ = ['testuts', 'testeuts','testmcuts', 'testemcuts']



def testuts(duration=0.3, fs=14000., dtype=np.float32, starttime=None):
    nsamples = fs * duration
    samples =  np.cos(2*np.pi*50*np.linspace(0.0,nsamples/float(fs), nsamples)**2.)
    if starttime is None:
        starttime = time.time()
    return UniformTimeSeries(samples=samples, fs=fs, starttime=starttime)

def testeuts(epochdur=0.3, fs=14000., dtype=np.float32, 
               nepochs=100):
    nepochsamples = fs * epochdur
    samples = np.empty((nepochs,nepochsamples), dtype=dtype)
    for i in range(nepochs):
        samples[i,:] = np.cos(2*np.pi*50*np.linspace(0.0,nepochsamples/float(fs), nepochsamples)+np.pi*i*10/float(nepochs))
    starttimes = np.linspace(0.,nepochs*epochdur*1.1,nepochs)
    epochs = np.empty(starttimes.size, dtype=[('starttime',np.float64)])
    epochs['starttime'] = starttimes
    return EpochedUniformTimeSeries(samples=samples, fs=fs, epochs=epochs,
                        epochaxis=0)

def testmcuts(duration=0.3, nchannels=5, fs=14000., dtype=np.float32, geometry=None, starttime=None):
    nsamples = fs * duration
    samples = np.empty((nchannels,nsamples), dtype=dtype)
    for i in range(nchannels):
        samples[i,:] = np.cos(2*np.pi*10*(i+1)*np.linspace(0.0,nsamples/float(fs), nsamples))
    if starttime is None:
        starttime = time.time()
    return MultiChannelUniformTimeSeries(samples=samples, fs=fs, channelaxis=0, geometry=geometry, starttime=starttime)

def testemcuts(epochdur=0.3, nchannels=5, fs=14000., dtype=np.float32, 
               nepochs=100, geometry=None):
    nepochsamples = fs * epochdur
    nsamples = nepochsamples * nepochs
    samples = np.empty((nchannels,nsamples), dtype=dtype)
    for i in range(nchannels):
        samples[i,:] = np.cos(2*np.pi*10*(i+1)*np.linspace(0.0,nsamples/float(fs), nsamples))
    samples = samples.reshape(nchannels,nepochs,nepochsamples)
    starttimes = np.linspace(0.,nepochs*epochdur*1.1,nepochs)
    epochs = np.empty(starttimes.size, dtype=[('starttime',np.float64)])
    epochs['starttime'] = starttimes
    return EpochedMultiChannelUniformTimeSeries(samples=samples, fs=fs, epochs=epochs,
                        channelaxis=0,epochaxis=1,geometry=geometry)
