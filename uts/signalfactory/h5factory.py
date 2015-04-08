import numpy as np
import tables as tb
from contextlib import contextmanager
from uts.core.uniformtime import UniformTimeSeries, EpochedUniformTimeSeries, \
      MultiChannelUniformTimeSeries, EpochedMultiChannelUniformTimeSeries, \
      isuts, iseuts, ismcuts, isemcuts
from uts.core.utils import timeparams
from uts.core.check_arg import check_arg, check_int, check_float, check_floatarray
from uts.io.hdf5 import _newh5uts, _newh5euts, _newh5mcuts, _newh5emcuts, \
      _create_h5savepoint

__all__ = ['h5fill_new', 'h5ones','h5zeros','h5fill_like']


def h5zeros(*args, **kwargs):
    """
    See documentation of fill_new
    
    """
    return h5fill_new(value=0.0, *args, **kwargs)

def h5ones(*args, **kwargs):
    """
    See documentation of fill_new
    
    """
    return h5fill_new(value=1.0, *args, **kwargs)


@contextmanager
def h5fill_new(signalname, filename=None, value=0.0, ntimesamples=None, 
             fs=None, duration=None, nchannels=None, 
             nepochs=None, starttime=0.0, origintime=0.0, channelaxis=None,
             epochaxis=None, channels=None, epochs=None, dtype=np.float64,
             readonly=False, geometry=None, attrs={}, 
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
    ## check time params
    (ntimesamples, fs, duration) = timeparams(ntimesamples=ntimesamples, fs=fs,
                                               duration=duration)
    if not (nepochs is None):
        if epochs is None:
            epochs = np.empty(nepochs, dtype=[('starttime',np.float64)])
            epochs['starttime'] = np.arange(nepochs, dtype= np.float64)
        if not nepochs == epochs.size:
            raise ValueError("epochs size does not correspond to nepochs")
        
    # check if everything is correct
    if not (duration == ntimesamples / fs):
        raise ValueError(
            "duration, ntimesamples, and fs parameters do not match")
    ## now see which signal type we need
    with _create_h5savepoint(filename=filename, h5path=signalname, overwrite=overwrite) as h5group:
        if nchannels is None:
            if epochs is None: # uts
                samples = _newh5uts(h5group=h5group, dtype=dtype,
                                        ntimesamples=ntimesamples, fs=fs, 
                                        origintime=origintime, starttime=starttime, dflt=value,
                                        overwrite=overwrite, extensible=extensible, 
                                        expectedrows=expectedrows, complib=complib, 
                                        complevel=complevel, chunkshape=chunkshape, attrs=attrs)
                s = UniformTimeSeries(samples=samples, fs=fs, origintime=origintime,
                                      starttime=starttime, dtype=None, copy=False,
                                      readonly=readonly)
            else: # euts
                nepochs = epochs.size
                if epochaxis is None:
                    epochaxis = 1
                epochaxis = check_arg(epochaxis,'epochaxis')
                samples = _newh5euts(h5group=h5group, dtype=dtype,
                                         ntimesamples=ntimesamples, epochs=epochs,
                                         epochaxis=epochaxis, fs=fs, 
                                         origintime=origintime, starttime=starttime, dflt=value,
                                         overwrite=overwrite, extensible=extensible, 
                                         expectedrows=expectedrows, complib=complib, 
                                         complevel=complevel, chunkshape=chunkshape, attrs=attrs)
                s = EpochedUniformTimeSeries(samples=samples, fs=fs, 
                                             origintime=origintime, epochs=epochs,
                                             epochaxis=epochaxis, dtype=None, copy=False,
                                             readonly=readonly)
        else: # multichannel
            nchannels = check_arg(nchannels, 'nchannels')
            if epochs is None: # mcuts
                if channelaxis is None:
                    channelaxis = 1
                channelaxis = check_arg(channelaxis,'channelaxis')
                samples = _newh5mcuts(h5group=h5group, dtype=dtype,
                                          ntimesamples=ntimesamples, nchannels=nchannels,
                                          geometry=geometry, channels=channels,
                                          channelaxis=channelaxis, fs=fs, 
                                          origintime=origintime, starttime=starttime, dflt=value,
                                          overwrite=overwrite, extensible=extensible, 
                                          expectedrows=expectedrows, complib=complib, 
                                          complevel=complevel,chunkshape=chunkshape, attrs=attrs)
                s = MultiChannelUniformTimeSeries(samples=samples, fs=fs, 
                                                  origintime=origintime, starttime=starttime, 
                                                  channelaxis=channelaxis, channels=channels,
                                                  dtype=None, copy=False, readonly=readonly, 
                                                  geometry=geometry)
            else: # epoched multichannel
                nepochs = records.size
                if epochaxis is None:
                    if channelaxis != 2:
                        epochaxis = 2
                    else:
                        epochaxis = 1
                epochaxis = check_arg(epochaxis,'epochaxis')
                if channelaxis is None:
                    if epochaxis != 1:
                        channelaxis = 1
                    else:
                        channelaxis = 2
                channelaxis = check_arg(channelaxis,'channelaxis')
                if not channelaxis in (0,1,2):
                    raise IndexError("channelaxis should be 0, 1, 2")
                if not epochaxis in (0,1,2):
                    raise IndexError("epochaxis should be 0, 1, 2")
                if channelaxis == epochaxis:
                    raise IndexError(
                        "epochs and channels cannot be on the same axis")
                if epochaxis == 0:
                    if channelaxis == 1:
                        timeaxis = 2
                    else: # channelaxis == 2
                        timeaxis = 1
                elif epochaxis == 1:
                    if channelaxis == 0:
                        timeaxis = 2
                    else: # channelaxis == 2
                        timeaxis = 0
                else: # epochaxis == 2
                    if channelaxis == 1:
                        timeaxis = 0
                    else: # channelaxis == 0
                        timeaxis = 1
                samples = _newh5emcuts(h5group=h5group,
                                           dtype=dtype, ntimesamples=ntimesamples, 
                                           epochs=epochs, epochaxis=epochaxis, fs=fs, 
                                           nchannels=nchannels, geometry=geometry, 
                                           channels=channels, channelaxis=channelaxis, 
                                           origintime=origintime, starttime=starttime, dflt=value,
                                           overwrite=overwrite, extensible=extensible, 
                                           expectedrows=expectedrows, complib=complib, 
                                           complevel=complevel, chunkshape=chunkshape, attrs=attrs)
                s = EpochedMultiChannelUniformTimeSeries(samples=samples, fs=fs, 
                                                         origintime=origintime, channelaxis=channelaxis, 
                                                         channels=channels, epochaxis=epochaxis, 
                                                         epochs=epochs, dtype=None, copy=False, readonly=readonly, 
                                                         geometry=geometry)
        yield s


@contextmanager
def h5fill_like(s, name='/', filename=None, value=0.0, dtype=None, channelaxis=None,
                   epochaxis=None, overwrite=False, extensible=False, 
                   expectedrows=None, filters=None, chunkshape=None, complib=None, 
                   complevel=None, attrs={}):

    with _create_h5savepoint(filename=filename, h5path=name, overwrite=overwrite) as h5group:    
        if isuts(s):
            yield  _uts_fill_like(s, value=value, h5group=h5group, dtype=dtype, 
                     overwrite=overwrite, extensible=extensible, expectedrows=expectedrows, 
                     chunkshape=chunkshape, complib=complib, complevel=complevel, attrs=attrs)
        elif ismcuts(s):
            yield _mcuts_fill_like(s, value=value, h5group=h5group,dtype=dtype, channelaxis=channelaxis,
                     overwrite=overwrite, extensible=extensible, expectedrows=expectedrows, 
                     chunkshape=chunkshape, complib=complib, complevel=complevel, attrs=attrs)
        elif iseuts(s):
            yield _euts_fill_like(s, value=value, h5group=h5group, dtype=dtype, epochaxis=epochaxis,
                     overwrite=overwrite, extensible=extensible, expectedrows=expectedrows, 
                     chunkshape=chunkshape, complib=complib, complevel=complevel, attrs=attrs)
        elif isemcuts(s):
            yield _emcuts_fill_like(s, value=value, h5group=h5group, dtype=dtype, channelaxis=channelaxis,
                     epochaxis=epochaxis, overwrite=overwrite, extensible=extensible, expectedrows=expectedrows, 
                     chunkshape=chunkshape, complib=complib, complevel=complevel, attrs=attrs)


def _uts_fill_like(s, value=0.0, h5group='/', dtype=None, 
                 overwrite=False, extensible=False, expectedrows=None, 
                 chunkshape=None, complib=None, complevel=None, attrs={}):
    if dtype is None:
        dtype = s.dtype
    else:
        dtype = np.dtype(dtype)
    params = {'dtype': dtype}
    if hasattr(s.samples, 'h5array'):
        if chunkshape is None:
            chunkshape = s.samples.h5array.chunkshape
        if complib is None:
            complib = s.samples.h5array.filters.complib
        if complevel is None:
            complevel = s.samples.h5array.filters.complevel
    samples = _newh5uts(h5group=h5group, dtype=dtype,
                        ntimesamples=s.ntimesamples, fs=s.fs, 
                        origintime=s.origintime, starttime=s.starttime, dflt=value,
                        overwrite=overwrite, extensible=extensible, 
                        expectedrows=expectedrows, complib=complib, complevel=complevel, 
                        chunkshape=chunkshape, attrs=attrs)
    if 'channelaxis' in samples.attrs:
        params['channelaxis'] = samples.attrs.channelaxis
    if 'epochaxis' in samples.attrs:
        params['epochaxis'] = samples.attrs.epochaxis           
    return s._instantiatefromself(samples=samples, copy=False, **params)

#euts
def _euts_fill_like(s, value=0.0, h5group='/', dtype=None, epochaxis=None,
                   overwrite=False, extensible=False, expectedrows=None, 
                   chunkshape=None, complib=None, 
                   complevel=None, attrs={}):
    if epochaxis is None:
        epochaxis = s.epochaxis
    if dtype is None:
        dtype = s.samples.dtype
    if hasattr(s.samples, 'h5array'):
        if chunkshape is None:
            chunkshape = s.samples.h5array.chunkshape
        if complib is None:
            complib = s.samples.h5array.filters.complib
        if complevel is None:
            complevel = s.samples.h5array.filters.complevel
    samples = _newh5euts(h5group=h5group, dtype=dtype,
                         ntimesamples=s.ntimesamples, fs=s.fs, 
                         origintime=s.origintime, starttime=s.starttime, 
                         epochs=s.epochs, epochaxis=epochaxis, dflt=value, 
                         overwrite=overwrite, extensible=extensible, 
                         expectedrows=expectedrows, complib=complib, complevel=complevel, 
                         chunkshape=chunkshape, attrs=attrs)

    return s._instantiatefromself(samples=samples, copy=False, 
                                     epochaxis=epochaxis)


#mcuts
def _mcuts_fill_like(s, value=0.0, h5group='/', dtype=None, channelaxis=None,
                   overwrite=False, extensible=False, expectedrows=None, 
                   chunkshape=None, complib=None, 
                   complevel=None, attrs={}):
    if channelaxis is None:
        channelaxis = s.channelaxis
    if dtype is None:
        dtype = s.samples.dtype
    if channelaxis is None:
        channelaxis = s.channelaxis
    if hasattr(s.samples, 'h5array'):
        if chunkshape is None:
            chunkshape = s.samples.h5array.chunkshape
        if complib is None:
            complib = s.samples.h5array.filters.complib
        if complevel is None:
            complevel = s.samples.h5array.filters.complevel
    samples = _newh5mcuts(h5group=h5group, dtype=dtype,
                          ntimesamples=s.ntimesamples, fs=s.fs, 
                          origintime=s.origintime, starttime=s.starttime, 
                          nchannels = s.nchannels, channels=s.channels, 
                          geometry=s.geometry, channelaxis=channelaxis, dflt=value, 
                          overwrite=overwrite, extensible=extensible, 
                          expectedrows=expectedrows, complib=complib, complevel=complevel, 
                          chunkshape=chunkshape, attrs=attrs)

    return s._instantiatefromself(samples=samples, copy=False, 
                                     channelaxis=channelaxis)                                         

#emcuts
def _emcuts_fill_like(s, value=0.0, h5group='/', dtype=None, channelaxis=None,
                   epochaxis=None, overwrite=False, extensible=False, 
                   expectedrows=None, chunkshape=None, complib=None, 
                   complevel=None, attrs={}):
    if channelaxis is None:
        channelaxis = s.channelaxis
    if epochaxis is None:
        epochaxis = s.epochaxis
    timeaxis, channelaxis, epochaxis = s._check_axes(channelaxis, epochaxis)
    if dtype is None:
        dtype = s.samples.dtype
    if hasattr(s.samples, 'h5array'):
        if chunkshape is None:
            chunkshape = s.samples.h5array.chunkshape
        if complib is None:
            complib = s.samples.h5array.filters.complib
        if complevel is None:
            complevel = s.samples.h5array.filters.complevel
    samples = _newh5emcuts(h5group=h5group, dtype=dtype,
                           ntimesamples=s.ntimesamples, fs=s.fs, 
                           origintime=s.origintime, starttime=s.starttime, 
                           nchannels = s.nchannels, epochaxis=epochaxis, 
                           epochs=s.epochs, channels=s.channels, 
                           geometry=s.geometry, channelaxis=channelaxis, dflt=value,
                           overwrite=overwrite, extensible=extensible, 
                           expectedrows=expectedrows, complib=complib, complevel=complevel, 
                           chunkshape=chunkshape, attrs=attrs)

    return s._instantiatefromself(samples=samples, copy=False, 
                                     channelaxis=channelaxis)