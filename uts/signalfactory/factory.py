import numpy as np
from ..core.uniformtime import UniformTimeSeries, EpochedUniformTimeSeries, \
    MultiChannelUniformTimeSeries, EpochedMultiChannelUniformTimeSeries, \
    isuts, iseuts, ismcuts, isemcuts, istimealigned
from uts.core.utils import timeparams, round_nearestodd
from uts.core.check_arg import check_arg, check_int, check_float, \
    check_floatarray
from uts.io.hdf5 import _newh5uts, _newh5euts, _newh5mcuts, _newh5emcuts

__all__ = ['fill_new', 'fill_like', 'fromsamples', 'ones', 'zeros', 'noise',
           'chirp', 'modfreq', 'harmonic', 'impulse', 'fromchannels']

# clean this up
def fromsamples(samples, fs, channels=None, starttime=0.0, \
                origintime=0.0, dtype=None, copy=True, channelaxis=1,
                geometry=None):
    fs = check_arg(fs, 'fs')
    starttime = check_arg(starttime, 'time')
    origintime = check_arg(origintime, 'time')
    channelaxis = check_arg(channelaxis, 'channelaxis')
    if not isinstance(samples, np.ndarray):
        samples = np.array(samples)
    if samples.ndim == 1:
        return UniformTimeSeries(samples=samples, fs=fs, starttime=starttime,
                                 origintime=origintime, dtype=dtype, copy=copy)
    else:
        return MultiChannelUniformTimeSeries(samples=samples, fs=fs,
                                             starttime=starttime,
                                             origintime=origintime,
                                             dtype=dtype, copy=copy,
                                             channelnames=channels,
                                             channelaxis=channelaxis,
                                             geometry=geometry)


def fromchannels(channels, channelnames=None, geometry=None, channelaxis=1):
    nchannels = len(channels)
    ch0 = channels[0]
    for channel in channels:
        if not isuts(channel):
            raise TypeError("channel not a UniformTimeSignal")
    if nchannels > 1:
        for i in range(1, nchannels):
            if not istimealigned(ch0, channels[i]):
                ValueError("not all channels time aligned")
    if channelaxis == 1:
        ar = np.empty((ch0.ntimesamples, nchannels), dtype=ch0.dtype)
        for i, channel in enumerate(channels):
            ar[:, i] = channel.samples.get(False)
    elif channelaxis == 0:
        ar = np.empty((nchannels, ch0.ntimesamples), dtype=ch0.dtype)
        for i, channel in enumerate(channels):
            ar[i, :] = channel.samples.get(False)
    else:
        raise ValueError("channelaxis shouls be '0' of '1'")
    return MultiChannelUniformTimeSeries(samples=ar, fs=ch0.fs,
                                         channelnames=channelnames,
                                         starttime=ch0.starttime,
                                         origintime=ch0.origintime, dtype=None,
                                         copy=False,
                                         channelaxis=channelaxis,
                                         geometry=geometry, readonly=True,
                                         attrs=ch0.attrs)


def zeros(*args, **kwargs):
    """
    See documentation of fill_new
    
    """
    return fill_new(0.0, *args, **kwargs)


def ones(*args, **kwargs):
    """
    See documentation of fill_new
    
    """
    return fill_new(1.0, *args, **kwargs)


# FIXME
def arange(*args, **kwargs):
    raise NotImplementedError('not possible yet')


#this would be nice?
#def timevalues(duration, fs, *args, **kwargs):
#    pass

def fill_new(value=0.0, ntimesamples=None, fs=None, duration=None,
             nchannels=None,
             nepochs=None, starttime=0.0, origintime=0.0, channelaxis=None,
             epochaxis=None, channels=None, epochs=None, dtype=np.float64,
             readonly=False, geometry=None, name=None, h5group=None, attrs=None,
             overwrite=False, extensible=False, expectedrows=None,
             complib='zlib', chunkshape=None, complevel=5):
    """
    Creates a uniform time series object filled with a number.

    Parameters
    ----------
    value : numeric, optional
        Fill value.
    ntimesamples : {None, int}, optional
    fs : {None, float}, optional
    duration : {None, float}, optional
    nchannels : {None, int }, optional
    nepochs : {None, int }, optional
    starttime : float, optional
    origintime : float, optional
    channelaxis : {None, int}, optional
    epochaxis: {None, int}, optional
    channels : {None, sequence of strings, structured array with 'channelname' column}, optional
    epochs : {None, structured array}, optional
        Array must have 'starttime' field of floats.
    dtype : data-type, optional
    readonly : {True, False}, optional
    geometry : geometry object, optional
    name: {None, string}, optional
    h5group: {None, pytables group}, optional
    attrs: dict, optional
    overwrite : {True, False}, optional
    extensible : {True, False}, optional
    expectedrows : {None, int}, optional
    filters : {None, pytables filters}, optional
    chunkshape : {None, sequence of ints}, optional

    """
    if attrs is None:
        attrs = {}
    ## check time params
    (ntimesamples, fs, duration) = timeparams(ntimesamples=ntimesamples, fs=fs,
                                              duration=duration)
    if not (nepochs is None):
        if epochs is None:
            epochs = np.empty(nepochs, dtype=[('starttime', np.float64)])
            epochs['starttime'] = np.arange(nepochs, dtype=np.float64)
        if not nepochs == epochs.size:
            raise ValueError("epochs size does not correspond to nepochs")

    # check if everything is correct
    if not (duration == ntimesamples / fs):
        raise ValueError(
            "duration, ntimesamples, and fs parameters do not match")
    ## check h5 params
    if not (name is None):
        h5array = True
    else:
        h5array = False
    ## now see which signal type we need
    if nchannels is None:
        if epochs is None:  # uts
            if h5array:
                samples = _newh5uts(h5group=h5group, name=name, dtype=dtype,
                                    ntimesamples=ntimesamples, fs=fs,
                                    origintime=origintime, starttime=starttime,
                                    dflt=value,
                                    overwrite=overwrite, extensible=extensible,
                                    expectedrows=expectedrows, complib=complib,
                                    complevel=complevel, chunkshape=chunkshape,
                                    attrs=attrs)
            else:
                samples = np.empty(ntimesamples, dtype=dtype)
                samples[:] = value
            s = UniformTimeSeries(samples=samples, fs=fs,
                                  origintime=origintime,
                                  starttime=starttime, dtype=None, copy=False,
                                  readonly=readonly)
        else:  # euts
            nepochs = epochs.size
            if epochaxis is None:
                epochaxis = 1
            epochaxis = check_arg(epochaxis, 'epochaxis')
            if h5array:
                samples = _newh5euts(h5group=h5group, name=name, dtype=dtype,
                                     ntimesamples=ntimesamples, epochs=epochs,
                                     epochaxis=epochaxis, fs=fs,
                                     origintime=origintime,
                                     starttime=starttime, dflt=value,
                                     overwrite=overwrite,
                                     extensible=extensible,
                                     expectedrows=expectedrows,
                                     complib=complib,
                                     complevel=complevel,
                                     chunkshape=chunkshape, attrs=attrs)
            else:
                if epochaxis == 1:
                    shape = [ntimesamples, nepochs]
                else:
                    shape = [nepochs, ntimesamples]
                samples = np.empty(shape, dtype=dtype)
                samples[:] = value
            s = EpochedUniformTimeSeries(samples=samples, fs=fs,
                                         origintime=origintime, epochs=epochs,
                                         epochaxis=epochaxis, dtype=None,
                                         copy=False,
                                         readonly=readonly)
    else:  # multichannel
        nchannels = check_arg(nchannels, 'nchannels')
        if epochs is None:  # mcuts
            if channelaxis is None:
                channelaxis = 1
            channelaxis = check_arg(channelaxis, 'channelaxis')
            if h5array:
                samples = _newh5mcuts(h5group=h5group, name=name, dtype=dtype,
                                      ntimesamples=ntimesamples,
                                      nchannels=nchannels,
                                      geometry=geometry, channels=channels,
                                      channelaxis=channelaxis, fs=fs,
                                      origintime=origintime,
                                      starttime=starttime, dflt=value,
                                      overwrite=overwrite,
                                      extensible=extensible,
                                      expectedrows=expectedrows,
                                      complib=complib,
                                      complevel=complevel,
                                      chunkshape=chunkshape, attrs=attrs)
            else:
                if channelaxis == 1:
                    shape = [ntimesamples, nchannels]
                else:
                    shape = [nchannels, ntimesamples]
                samples = np.empty(shape, dtype=dtype)
                samples[:] = value
            s = MultiChannelUniformTimeSeries(samples=samples, fs=fs,
                                              origintime=origintime,
                                              starttime=starttime,
                                              channelaxis=channelaxis,
                                              channelnames=channels,
                                              dtype=None, copy=False,
                                              readonly=readonly,
                                              geometry=geometry)
        else:  # epoched multichannel
            nepochs = records.size
            if epochaxis is None:
                if channelaxis != 2:
                    epochaxis = 2
                else:
                    epochaxis = 1
            epochaxis = check_arg(epochaxis, 'epochaxis')
            if channelaxis is None:
                if epochaxis != 1:
                    channelaxis = 1
                else:
                    channelaxis = 2
            channelaxis = check_arg(channelaxis, 'channelaxis')
            if not channelaxis in (0, 1, 2):
                raise IndexError("channelaxis should be 0, 1, 2")
            if not epochaxis in (0, 1, 2):
                raise IndexError("epochaxis should be 0, 1, 2")
            if channelaxis == epochaxis:
                raise IndexError(
                    "epochs and channels cannot be on the same axis")
            if epochaxis == 0:
                if channelaxis == 1:
                    timeaxis = 2
                else:  # channelaxis == 2
                    timeaxis = 1
            elif epochaxis == 1:
                if channelaxis == 0:
                    timeaxis = 2
                else:  # channelaxis == 2
                    timeaxis = 0
            else:  # epochaxis == 2
                if channelaxis == 1:
                    timeaxis = 0
                else:  # channelaxis == 0
                    timeaxis = 1
            if h5array:
                samples = _newh5emcuts(h5group=h5group, name=name,
                                       dtype=dtype, ntimesamples=ntimesamples,
                                       epochs=epochs, epochaxis=epochaxis,
                                       fs=fs,
                                       nchannels=nchannels, geometry=geometry,
                                       channels=channels,
                                       channelaxis=channelaxis,
                                       origintime=origintime,
                                       starttime=starttime, dflt=value,
                                       overwrite=overwrite,
                                       extensible=extensible,
                                       expectedrows=expectedrows,
                                       complib=complib,
                                       complevel=complevel,
                                       chunkshape=chunkshape, attrs=attrs)
            else:
                shape = [0, 0, 0]
                shape[timeaxis] = ntimesamples
                shape[channelaxis] = nchannels
                shape[epochaxis] = nepochs
                samples = np.empty(shape, dtype=dtype)
                samples[:] = value
            s = EpochedMultiChannelUniformTimeSeries(samples=samples, fs=fs,
                                                     origintime=origintime,
                                                     channelaxis=channelaxis,
                                                     channelnames=channels,
                                                     epochaxis=epochaxis,
                                                     epochs=epochs, dtype=None,
                                                     copy=False,
                                                     readonly=readonly,
                                                     geometry=geometry)
    return s


def fill_like(s, value=0.0, h5group=None, name=None, dtype=None,
              channelaxis=None,
              epochaxis=None, overwrite=False, extensible=False,
              expectedrows=None, filters=None, chunkshape=None, complib=None,
              complevel=None, attrs=None):
    if attrs is None:
        attrs = {}
    if isuts(s):
        return _uts_fill_like(s, value=value, h5group=h5group, name=name,
                              dtype=dtype,
                              overwrite=overwrite, extensible=extensible,
                              expectedrows=expectedrows,
                              chunkshape=chunkshape, complib=complib,
                              complevel=complevel, attrs=attrs)
    elif ismcuts(s):
        return _mcuts_fill_like(s, value=value, h5group=h5group, name=name,
                                dtype=dtype, channelaxis=channelaxis,
                                overwrite=overwrite, extensible=extensible,
                                expectedrows=expectedrows,
                                chunkshape=chunkshape, complib=complib,
                                complevel=complevel, attrs=attrs)
    elif iseuts(s):
        return _euts_fill_like(s, value=value, h5group=h5group, name=name,
                               dtype=dtype, epochaxis=epochaxis,
                               overwrite=overwrite, extensible=extensible,
                               expectedrows=expectedrows,
                               chunkshape=chunkshape, complib=complib,
                               complevel=complevel, attrs=attrs)
    elif isemcuts(s):
        return _emcuts_fill_like(s, value=value, h5group=h5group, name=name,
                                 dtype=dtype, channelaxis=channelaxis,
                                 epochaxis=epochaxis, overwrite=overwrite,
                                 extensible=extensible,
                                 expectedrows=expectedrows,
                                 chunkshape=chunkshape, complib=complib,
                                 complevel=complevel, attrs=attrs)


## FIXME if h5 array set default parameters (comlib etc) same
def _uts_fill_like(s, value=0.0, h5group=None, name=None, dtype=None,
                   overwrite=False, extensible=False, expectedrows=None,
                   chunkshape=None, complib=None, complevel=None, attrs=None):
    if attrs is None:
        attrs = {}
    if dtype is None:
        dtype = s.dtype
    else:
        dtype = np.dtype(dtype)
    params = {'dtype': dtype}
    if h5group is None:
        samples = np.zeros(s.samples.shape, dtype=dtype)
        samples[:] = value
    else:
        if hasattr(s.samples, 'h5array'):
            if chunkshape is None:
                chunkshape = s.samples.h5array.chunkshape
            if complib is None:
                complib = s.samples.h5array.filters.complib
            if complevel is None:
                complevel = s.samples.h5array.filters.complevel
        samples = _newh5uts(h5group=h5group, name=name, dtype=dtype,
                            ntimesamples=s.ntimesamples, fs=s.fs,
                            origintime=s.origintime, starttime=s.starttime,
                            dflt=value,
                            overwrite=overwrite, extensible=extensible,
                            expectedrows=expectedrows, complib=complib,
                            complevel=complevel,
                            chunkshape=chunkshape, attrs=attrs)
        if 'channelaxis' in samples.attrs:
            params['channelaxis'] = samples.attrs.channelaxis
        if 'epochaxis' in samples.attrs:
            params['epochaxis'] = samples.attrs.epochaxis
    return s._instantiatefromself(samples=samples, copy=False, **params)


#euts
def _euts_fill_like(s, value=0.0, h5group=None, name=None, dtype=None,
                    epochaxis=None,
                    overwrite=False, extensible=False, expectedrows=None,
                    chunkshape=None, complib=None,
                    complevel=None, attrs=None):
    if attrs is None:
        attrs = {}
    if epochaxis is None:
        epochaxis = s.epochaxis
    if dtype is None:
        dtype = s.samples.dtype
    if h5group is None:
        if epochaxis == s.epochaxis:
            samples = np.zeros(s.samples.shape, dtype=dtype)
        else:
            samples = np.zeros(s.samples.shape[::-1], dtype=dtype)
        samples[:] = value
    else:
        if hasattr(s.samples, 'h5array'):
            if chunkshape is None:
                chunkshape = s.samples.h5array.chunkshape
            if complib is None:
                complib = s.samples.h5array.filters.complib
            if complevel is None:
                complevel = s.samples.h5array.filters.complevel
        samples = _newh5euts(h5group=h5group, name=name, dtype=dtype,
                             ntimesamples=s.ntimesamples, fs=s.fs,
                             origintime=s.origintime, starttime=s.starttime,
                             epochs=s.epochs, epochaxis=epochaxis, dflt=value,
                             overwrite=overwrite, extensible=extensible,
                             expectedrows=expectedrows, complib=complib,
                             complevel=complevel,
                             chunkshape=chunkshape, attrs=attrs)

    return s._instantiatefromself(samples=samples, copy=False,
                                  epochaxis=epochaxis)


#mcuts
def _mcuts_fill_like(s, value=0.0, h5group=None, name=None, dtype=None,
                     channelaxis=None,
                     overwrite=False, extensible=False, expectedrows=None,
                     chunkshape=None, complib=None,
                     complevel=None, attrs=None):
    if attrs is None:
        attrs = {}
    if channelaxis is None:
        channelaxis = s.channelaxis
    if dtype is None:
        dtype = s.samples.dtype
    if h5group is None:
        if channelaxis == s.channelaxis:
            samples = np.zeros(s.samples.shape, dtype=dtype)
        else:
            samples = np.zeros(s.samples.shape[::-1], dtype=dtype)
        samples[:] = value
    else:
        if channelaxis is None:
            channelaxis = s.channelaxis
        if hasattr(s.samples, 'h5array'):
            if chunkshape is None:
                chunkshape = s.samples.h5array.chunkshape
            if complib is None:
                complib = s.samples.h5array.filters.complib
            if complevel is None:
                complevel = s.samples.h5array.filters.complevel
        samples = _newh5mcuts(h5group=h5group, name=name, dtype=dtype,
                              ntimesamples=s.ntimesamples, fs=s.fs,
                              origintime=s.origintime, starttime=s.starttime,
                              nchannels=s.nchannels, channels=s.channels,
                              geometry=s.geometry, channelaxis=channelaxis,
                              dflt=value,
                              overwrite=overwrite, extensible=extensible,
                              expectedrows=expectedrows, complib=complib,
                              complevel=complevel,
                              chunkshape=chunkshape, attrs=attrs)

    return s._instantiatefromself(samples=samples, copy=False,
                                  channelaxis=channelaxis)


#emcuts
def _emcuts_fill_like(s, value=0.0, h5group=None, name=None, dtype=None,
                      channelaxis=None,
                      epochaxis=None, overwrite=False, extensible=False,
                      expectedrows=None, chunkshape=None, complib=None,
                      complevel=None, attrs=None):
    if attrs is None:
        attrs = {}
    if channelaxis is None:
        channelaxis = s.channelaxis
    if epochaxis is None:
        epochaxis = s.epochaxis
    timeaxis, channelaxis, epochaxis = s._check_axes(channelaxis, epochaxis)
    if dtype is None:
        dtype = s.samples.dtype
    if h5group is None:
        shape = [0, 0, 0]
        shape[channelaxis] = s.nchannels
        shape[epochaxis] = s.nepochs
        shape[timeaxis] = s.ntimesamples
        samples = np.zeros(shape, dtype=dtype)
        samples[:] = value
    else:
        if hasattr(s.samples, 'h5array'):
            if chunkshape is None:
                chunkshape = s.samples.h5array.chunkshape
            if complib is None:
                complib = s.samples.h5array.filters.complib
            if complevel is None:
                complevel = s.samples.h5array.filters.complevel
        samples = _newh5emcuts(h5group=h5group, name=name, dtype=dtype,
                               ntimesamples=s.ntimesamples, fs=s.fs,
                               origintime=s.origintime, starttime=s.starttime,
                               nchannels=s.nchannels, epochaxis=epochaxis,
                               epochs=s.epochs, channels=s.channels,
                               geometry=s.geometry, channelaxis=channelaxis,
                               dflt=value,
                               overwrite=overwrite, extensible=extensible,
                               expectedrows=expectedrows, complib=complib,
                               complevel=complevel,
                               chunkshape=chunkshape, attrs=attrs)

    return s._instantiatefromself(samples=samples, copy=False,
                                  channelaxis=channelaxis)


def noise(duration=None, ntimesamples=None, fs=44100., nchannels=None,
          channels=None,
          starttime=0.0, origintime=0.0, channelaxis=1, dtype=None,
          geometry=None,
          peakamp=None, rmsamp=None):
    (ntimesamples, fs, duration) = timeparams(ntimesamples=ntimesamples, fs=fs,
                                              duration=duration)
    # nchannels is checked on instantiation
    starttime = check_arg(starttime, 'time')
    origintime = check_arg(origintime, 'time')

    nsamples = int(np.round(duration * fs))
    if nchannels is None:
        samples = np.random.rand(nsamples)
        samples -= samples.mean(0)
    else:
        if channelaxis == 1:
            samples = np.random.rand(nsamples, nchannels)
            samples -= samples.mean(0)
        else:
            samples = np.random.rand(nchannels, nsamples)
            samples -= samples.mean(1)[:, np.newaxis]
    samples = _scale(samples, peakamp=peakamp, rmsamp=rmsamp)
    return fromsamples(samples=samples, fs=fs, channels=channels,
                       starttime=starttime, origintime=origintime, dtype=dtype,
                       copy=False,
                       channelaxis=channelaxis, geometry=geometry)


def chirp(duration=None, ntimesamples=None, fs=44100., startfreq=0.,
          endfreq=10000.,
          dtype=None, peakamp=None, rmsamp=None):
    (ntimesamples, fs, duration) = timeparams(ntimesamples=ntimesamples, fs=fs,
                                              duration=duration)
    nsamples = int(round(duration * fs))
    t = np.linspace(startfreq, endfreq, num=nsamples)
    return modfreq(freqpattern=t, fs=fs, dtype=dtype, peakamp=peakamp,
                   rmsamp=rmsamp)


def impulse(duration=None, ntimesamples=None, fs=44100., dtype=None,
            peakamp=1.0):
    if dtype is None:
        dtype = np.float64

    (ntimesamples, fs, duration) = timeparams(ntimesamples=ntimesamples, fs=fs,
                                              duration=duration)
    ntimesamples = round_nearestodd(ntimesamples)
    midsample = ntimesamples // 2
    duration = ntimesamples / fs
    origintime = duration / 2.0
    ar = np.zeros(ntimesamples, dtype=dtype)
    ar[midsample] = peakamp
    return UniformTimeSeries(samples=ar, fs=fs, starttime=0.0,
                             origintime=origintime)


def modfreq(freqpattern, fs, dtype=None, peakamp=None, rmsamp=None):
    if dtype is None:
        dtype = np.float64
    freqpattern = np.asarray(freqpattern)
    assert freqpattern.ndim == 1, \
        "frequency pattern should be a 1-dimensional array"
    assert (freqpattern < (fs / 2.0)).all(), \
        "all frequencies should be smaller than fs / 2.0"
    fs = check_arg(fs, 'fs')
    ar = freqpattern.astype(dtype) / float(fs)
    ar = ar.cumsum()
    ar = np.unwrap(ar, discont=1)
    ar = np.sin(2 * np.pi * ar)
    ar = _scale(ar, peakamp=peakamp, rmsamp=rmsamp)
    return UniformTimeSeries(samples=ar, fs=fs)


def harmonic(duration=None, ntimesamples=None, fs=44100., fundfreq=500.,
             nharmonics=1, dtype=None,
             hweights=None, hphases=None, peakamp=None, rmsamp=None):
    """
    If you don't provide peakamp and rmsamp, default peakamp is 0.99.
    """
    (ntimesamples, fs, duration) = timeparams(ntimesamples=ntimesamples, fs=fs,
                                              duration=duration)
    fundfreq = check_float(fundfreq, positive=True, zero=False, negative=False)
    nharmonics = check_int(nharmonics, positive=True, zero=False,
                           negative=False, cast=int)
    if dtype is None:
        dtype = np.float64
    if hweights is None:
        hweights = np.ones(nharmonics, dtype=dtype)
    else:
        hweights = check_floatarray(hweights)
    if hphases is None:
        hphases = (np.random.random(nharmonics) * np.pi * 2.0).astype(dtype)
    else:
        hphases = check_floatarray(hphases)
    ntimebins = int(np.round(duration * fs))
    hsamples = np.empty((ntimebins, nharmonics), dtype=dtype)
    hsamples[:] = np.linspace(0., duration, ntimebins)[:, np.newaxis]  # time
    freqs = np.arange(fundfreq, fundfreq * (nharmonics + 1), fundfreq,
                      dtype=dtype)
    samples = (hweights[np.newaxis, :] * np.sin(
        2. * np.pi * hsamples * freqs + hphases[np.newaxis, :])).sum(1)
    samples = _scale(samples, peakamp=peakamp, rmsamp=rmsamp)

    return UniformTimeSeries(samples=samples, fs=fs)


def _check_amp(peakamp, rmsamp):
    if (peakamp is None) and (rmsamp is None):
        peakamp = 0.99
    if (not peakamp is None) and (not rmsamp is None):
        raise ValueError(
            "you can only set either peakamp or rmsamp, but not both")
    if not peakamp is None:
        peakamp = check_float(peakamp, positive=True, zero=False,
                              negative=False)
    if not rmsamp is None:
        rmsamp = check_float(rmsamp, positive=True, zero=False, negative=False)
    return peakamp, rmsamp


def _scale(samples, peakamp, rmsamp):
    peakamp, rmsamp = _check_amp(peakamp=peakamp,
                                 rmsamp=rmsamp)
    if rmsamp is None:
        samples /= (np.max(np.abs(samples)) / peakamp)
    else:
        rms = (samples ** 2.0).mean() ** 0.5
        samples *= rmsamp / rms
    return samples
