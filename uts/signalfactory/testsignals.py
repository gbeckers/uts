import numpy as np
import time
from uts.core.geometry import RectangularLattice
from uts.core.multichannel import default_channelnames
from uts.core.uniformtime import UniformTimeSeries, EpochedUniformTimeSeries, \
      MultiChannelUniformTimeSeries, EpochedMultiChannelUniformTimeSeries


__all__ = ['testuts', 'testeuts','testmcuts', 'testemcuts']



def testuts(duration=0.3, fs=14000., dtype=np.float32, starttime=None):
    nsamples = int(np.round(fs * duration))
    samples =  np.cos(2*np.pi*50*np.linspace(0.0,nsamples/float(fs), nsamples)**2.)
    if starttime is None:
        starttime = time.time()
    return UniformTimeSeries(samples=samples, fs=fs, starttime=starttime,
                             dtype=dtype)

def testeuts(epochdur=0.3, fs=14000., dtype=np.float32, 
               nepochs=100):
    nepochsamples = int(fs * epochdur)
    samples = np.empty((nepochs, nepochsamples), dtype=dtype)
    for i in range(nepochs):
        samples[i,:] = np.cos(2*np.pi*50*np.linspace(0.0,nepochsamples/float(fs), nepochsamples)+np.pi*i*10/float(nepochs))
    starttimes = np.linspace(0.,nepochs*epochdur*1.1,nepochs)
    epochs = np.empty(starttimes.size, dtype=[('starttime',np.float64)])
    epochs['starttime'] = starttimes
    return EpochedUniformTimeSeries(samples=samples, fs=fs, epochs=epochs,
                                    epochaxis=0, dtype=dtype)

def testmcuts(duration=0.3, nchannels=5, fs=14000., dtype=np.float32,
            geometryshape=None, starttime=None):
    nsamples = int(fs * duration)
    samples = np.empty((nchannels,nsamples), dtype=dtype)
    for i in range(nchannels):
        samples[i,:] = np.cos(2*np.pi*10*(i+1)*np.linspace(0.0,nsamples/float(fs), nsamples))
    if starttime is None:
        starttime = time.time()
    if not geometryshape is None:
        if not nchannels == np.prod(geometryshape):
            raise ValueError('shape not congruent with nchannels')
        # the point names have to be the channel names, which is checked on
        # construction; 'channel_%.3d' never matched the defaults. A geometry
        # also requires the channel names to be passed explicitly.
        channelnames = default_channelnames(nchannels)
        geometry = RectangularLattice(*geometryshape, pointnames=channelnames)
    else:
        channelnames = None
        geometry = None
    return MultiChannelUniformTimeSeries(samples=samples, fs=fs, channelaxis=0,
                                         channelnames=channelnames,
                                         geometry=geometry, starttime=starttime,
                                         dtype=dtype)

def testemcuts(epochdur=0.3, nchannels=5, fs=14000., dtype=np.float32, 
               nepochs=100, geometryshape=None):
    nepochsamples = int(fs * epochdur)
    nsamples = nepochsamples * nepochs
    samples = np.empty((nchannels,nsamples), dtype=dtype)
    for i in range(nchannels):
        samples[i,:] = np.cos(2*np.pi*10*(i+1)*np.linspace(0.0,nsamples/float(fs), nsamples))
    samples = samples.reshape(nchannels,nepochs,nepochsamples)
    starttimes = np.linspace(0.,nepochs*epochdur*1.1,nepochs)
    epochs = np.empty(starttimes.size, dtype=[('starttime',np.float64)])
    epochs['starttime'] = starttimes
    if not geometryshape is None:
        if not nchannels == np.prod(geometryshape):
            raise ValueError('shape not congruent with nchannels')
        # the point names have to be the channel names, which is checked on
        # construction; 'channel_%.3d' never matched the defaults. A geometry
        # also requires the channel names to be passed explicitly.
        channelnames = default_channelnames(nchannels)
        geometry = RectangularLattice(*geometryshape, pointnames=channelnames)
    else:
        channelnames = None
        geometry = None
    return EpochedMultiChannelUniformTimeSeries(samples=samples, fs=fs,
                                                epochs=epochs, channelaxis=0,
                                                epochaxis=1,
                                                channelnames=channelnames,
                                                geometry=geometry,
                                                dtype=dtype)
