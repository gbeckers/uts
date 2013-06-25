import numpy as np
import _stats
from uts3 import UniformTimeSeries, \
    MultiChannelUniformTimeSeries, EpochedUniformTimeSeries, \
    EpochedMultiChannelUniformTimeSeries
from frequencysampled import Spectrum
from helper import unsqueeze
from uts3.core.check_arg import check_arg, check_int, check_float, check_floatarray,\
     check_axis
import scipy.stats #import nanmean, nanstd, nanmedian

__all__ = [ 'argmax', 'argmin','extremes', 'kurtosis', 'mad', 'max', 'mean', 
            'median', 'nanmean','nanmedian', 'nanstd','min', 'ptp', 'rms', 
            'skew', 'standardize', 'std', 'sum' ]


# FIXME start, end parameters for functions (see extremes)

def _assert_signal(ts, signaltypes=None):
    allowedtypes = (UniformTimeSeries, MultiChannelUniformTimeSeries,
            EpochedUniformTimeSeries, EpochedMultiChannelUniformTimeSeries,
            Spectrum) 
    if signaltypes is None: signaltypes = allowedtypes
    if not ts.__class__ in signaltypes:
        raise TypeError("%s is not an allowed signal type" % ts.__class__)
    return True
    

def _meaniterframefunc(func, s, axis=None, framesize=1024*64, **kwargs):
    _assert_signal(s)
    axis = check_axis(axis, s)
    axissize = s.samples.shape[s.samples.axes[axis]]
    if axissize < 1:
        raise ValueError('signal has no samples along %s axis' % axis)
    if axissize <= framesize:
        m = func(s.samples.get(copy=False), axis=s.samples.axes[axis],**kwargs)
    else:
        ms = []
        weights = []
        for f in s.iter_frames(framesize=framesize):
            m = func(f.samples.get(copy=False), axis=s.samples.axes[axis],**kwargs)
            ms.append(m)
            weights.append(f.ntimesamples / float(framesize))
        ms = np.array(ms, dtype=np.float64)
        weights = np.array(weights, dtype=np.float64)
        m = np.average(ms,axis=0,weights=weights)

    return m
    

def median(s, axis=None, framesize=1024*128):
    return _meaniterframefunc(func=np.median, s=s, axis=axis, 
                              framesize=framesize)

def mean(s, axis=None, framesize=1024*128):
    return _meaniterframefunc(func=np.mean, s=s, axis=axis, 
                              framesize=framesize,  dtype=np.float64)

def nanmean(s, axis=None, framesize=1024*128):
    return _meaniterframefunc(func=scipy.stats.nanmean, s=s, axis=axis, 
                              framesize=framesize)#, dtype=np.float64)

def nanmedian(s, axis=None, framesize=1024*128):
    return _meaniterframefunc(func=scipy.stats.nanmedian, s=s, axis=axis, 
                              framesize=framesize)#, dtype=np.float64)

def std(s, axis=None, framesize=1024*128):
    return _meaniterframefunc(func=np.std, s=s, axis=axis, 
                              framesize=framesize, dtype=np.float64)

def nanstd(s, axis=None, framesize=1024*128):
    return _meaniterframefunc(func=scipy.stats.nanstd, s=s, axis=axis, 
                              framesize=framesize)#, dtype=np.float64)

def rms(s, axis=None, framesize=1024*128):
    def samplerms(samples, axis, dtype):
        return np.power(np.power(samples.astype(dtype), 2.0).mean(axis),0.5)        
    return _meaniterframefunc(func=samplerms, s=s, axis=axis, 
                              framesize=framesize, dtype=np.float64)

def kurtosis(s, axis=None, framesize=1024*128, fisher=True, bias=True):
    def samplekurtosis(samples, axis, dtype):
        return scipy.stats.kurtosis(samples.astype(dtype), axis=axis, 
                                    fisher=fisher, bias=bias)
    return _meaniterframefunc(func=samplekurtosis, s=s, axis=axis, 
                              framesize=framesize, dtype=np.float64)

def skew(s, axis=None, framesize=1024*128, bias=True):
    def sampleskew(samples, axis, dtype):
        return scipy.stats.skew(samples.astype(dtype), axis=axis, 
                                    bias=bias)
    return _meaniterframefunc(func=sampleskew, s=s, axis=axis, 
                              framesize=framesize, dtype=np.float64)


def mad(s, axis=None, framesize=1024*128, c=0.6745):
    """
    Compute the median absolute deviation (mad).

    Parameters
    ----------
    s : sampled signal
        Signal containing samples whose mad is desired.
    axis : str, optional
        Axis along which the mad is computed. The default is to compute
        the mad over the baseaxis, which is a property of s (s.baseaxis).
    framesize: int, optional
        The maximum size of the frame that is loaded in memory to compute mad. 
        If s is longer, mad will be computed over multiple frames, after which
        a weighted average is computed.
    c : float, optional
        Scale factor to adjust for asymptotically normal consistency. Defaults 
        to 0.6745.
    
    Returns
    -------
    mad : ndarray or scalar
    
    Notes
    -----
    The mad is the median of the absolute deviations from the median, and, by 
    default, adjust by a factor for asymptotically normal consistency.
    
    Examples
    --------
    >>> s = ds.signalfromsamples([0.08418439,  0.28987414, -0.05445002,  1.6779827], 1.0)
    >>> mad(s)
    >>>
    0.25524400296515937

    """
    
    return _meaniterframefunc(func=_stats.mad, s=s, axis=axis, 
                              framesize=framesize, c=c)

def sum(s, axis=None, framesize=1024*128):
    _assert_signal(s)
    axis = check_axis(axis, s)
    axissize = s.samples.shape[s.samples.axes[axis]]
    if axissize < 1:
        raise ValueError('signal has no samples along %s axis' % axis)
    if axissize <= framesize:
        m = np.sum(s.samples.get(copy=False), axis=s.samples.axes[axis])
    else:
        shape = list(s.samples.shape)
        del shape[s.samples.axes[axis]]
        if shape == []:
            shape = [1]
        m = np.zeros(shape, dtype=np.float64)
        for f in s.iter_frames(framesize=framesize):
            m += np.array(np.sum(f.samples.get(copy=False), 
                                     axis=s.samples.axes[axis]), ndmin=1)
    return m

def max(s, axis=None, framesize=1024*128):
    _assert_signal(s)
    axis = check_axis(axis, s)
    axissize = s.samples.shape[s.samples.axes[axis]]
    if axissize < 1:
        raise ValueError('signal has no samples along %s axis' % axis)
    if axissize <= framesize:
        m = np.amax(s.samples.get(copy=False), axis=s.samples.axes[axis])
    else:
        shape = list(s.samples.shape)
        del shape[s.samples.axes[axis]]
        if shape == []:
            shape = [1]
        m = np.empty(shape, dtype=np.float64)
        m[:] = -np.inf
        for f in s.iter_frames(framesize=framesize):
            currm = np.array(np.amax(f.samples.get(copy=False), 
                                     axis=s.samples.axes[axis]), ndmin=1)
            index = (currm > m)
            m[index] = currm[index]
        if m.shape == (1,):
            m = m[0]
    return m


def argmax(s, axis=None, framesize=1024*128):
    _assert_signal(s)
    axis = check_axis(axis, s)
    axissize = s.samples.shape[s.samples.axes[axis]]
    if axissize < 1:
        raise ValueError('signal has no samples along %s axis' % axis)
    if axissize <= framesize:
        m = np.argmax(s.samples.get(copy=False), axis=s.samples.axes[axis])
    else:
        shape = list(s.samples.shape)
        del shape[s.samples.axes[axis]]
        if shape == []:
            shape = [1]
        m = np.empty(shape, dtype=np.float64)
        m[:] = -np.inf
        for f in s.iter_frames(framesize=framesize):
            currm = np.array(np.argmax(f.samples.get(copy=False), 
                                     axis=s.samples.axes[axis]), ndmin=1)
            index = (currm > m)
            m[index] = currm[index]
        if m.shape == (1,):
            m = m[0]
    return m


def min(s, axis=None, framesize=1024*128):
    _assert_signal(s)
    axis = check_axis(axis, s)
    axissize = s.samples.shape[s.samples.axes[axis]]
    if axissize < 1:
        raise ValueError('signal has no samples along %s axis' % axis)
    if axissize <= framesize:
        m = np.amin(s.samples.get(copy=False), axis=s.samples.axes[axis])
    else:
        shape = list(s.samples.shape)
        del shape[s.samples.axes[axis]]
        if shape == []:
            shape = [1]
        m = np.empty(shape, dtype=np.float64)
        m[:] = np.inf
        for f in s.iter_frames(framesize=framesize):
            currm = np.array(np.amin(f.samples.get(copy=False), 
                                     axis=s.samples.axes[axis]), ndmin=1)
            index = (currm < m)
            m[index] = currm[index]
        if m.shape == (1,):
            m = m[0]
    return m

def argmin(s, axis=None, framesize=1024*128):
    _assert_signal(s)
    axis = check_axis(axis, s)
    axissize = s.samples.shape[s.samples.axes[axis]]
    if axissize < 1:
        raise ValueError('signal has no samples along %s axis' % axis)
    if axissize <= framesize:
        m = np.argmin(s.samples.get(copy=False), axis=s.samples.axes[axis])
    else:
        shape = list(s.samples.shape)
        del shape[s.samples.axes[axis]]
        if shape == []:
            shape = [1]
        m = np.empty(shape, dtype=np.float64)
        m[:] = np.inf
        for f in s.iter_frames(framesize=framesize):
            currm = np.array(np.argmin(f.samples.get(copy=False), 
                                     axis=s.samples.axes[axis]), ndmin=1)
            index = (currm < m)
            m[index] = currm[index]
        if m.shape == (1,):
            m = m[0]
    return m

def extremes(s, axis=None, framesize=1024*128, start=None, end=None):
    _assert_signal(s)
    axis = check_axis(axis, s)
    axissize = s.samples.shape[s.samples.axes[axis]]
    if axissize < 1:
        raise ValueError('signal has no samples along %s axis' % axis)
    if axissize <= framesize:
        mmin = np.min(s[slice(start,end,None)].samples.get(copy=False), axis=s.samples.axes[axis])
        mminindex = np.argmin(s[slice(start,end,None)].samples.get(copy=False), axis=s.samples.axes[axis])
        mmax = np.max(s[slice(start,end,None)].samples.get(copy=False), axis=s.samples.axes[axis])
        mmaxindex = np.argmax(s[slice(start,end,None)].samples.get(copy=False), axis=s.samples.axes[axis])
    else:
        shape = list(s.samples.shape)
        del shape[s.samples.axes[axis]]
        if shape == []:
            shape = [1]
        mmin = np.empty(shape, dtype=np.float64)
        mmin[:] = np.inf
        mminindex = np.empty(shape, dtype=np.int64)
        mmax = np.empty(shape, dtype=np.float64)
        mmax[:] = -np.inf
        mmaxindex = np.empty(shape, dtype=np.int64)
        for f in s.iter_frames(framesize=framesize, start=start, end=end):
            currmin = np.array(np.min(f.samples.get(copy=False), 
                                     axis=s.samples.axes[axis]), ndmin=1)
            currminindex = np.array(np.argmin(f.samples.get(copy=False), 
                                     axis=s.samples.axes[axis]), ndmin=1)
            index = (currmin < mmin)
            mmin[index] = currmin[index]
            mminindex[index] = currminindex[index]
            currmax = np.array(np.max(f.samples.get(copy=False), 
                                     axis=s.samples.axes[axis]), ndmin=1)
            currmaxindex = np.array(np.argmax(f.samples.get(copy=False), 
                                     axis=s.samples.axes[axis]), ndmin=1)
            index = (currmax > mmax)
            mmax[index] = currmax[index]
            mmaxindex[index] = currmaxindex[index]   
    if mmin.shape == (1,):
        mmin = mmin[0]
        mminindex = mminindex[0]
    if mmax.shape == (1,):
        mmax = mmax[0]
        mmaxindex = mmaxindex[0]
    return mmin, mmax, mminindex, mmaxindex

def ptp(s, axis=None, framesize=1024*128):
    _assert_signal(s)
    axis = check_axis(axis, s)
    axissize = s.samples.shape[s.samples.axes[axis]]
    if axissize < 1:
        raise ValueError('signal has no samples along %s axis' % axis)
    if axissize <= framesize:
        m = np.ptp(s.samples.get(copy=False), axis=s.samples.axes[axis])
    else:
        mmin, mmax, mminindex, mmaxindex = extremes(s=s, axis=axis, framesize=framesize)
        m = mmax - mmin
    return m

#FIXME this does not work anymore
def standardize(s, axis='time', sdone=True, ignorenan=True):
    """
    Standardize signal so that its mean is zero and its standard deviation 1.0.

    Parameters
    ----------
    s : sampled signal
        Signal containing samples whose mad is desired.
    axis : str, optional
        Axis along which the standardization to to be carried out. The default 
        is do this over "time".
    ignorenan: {True, False}
        Should nan values be ignored (i.e. not used in the standardization) or
        not? The default is to ignore nan values.

    Returns
    -------
    standardized signal : (epoched) (multichannel) time sampled signal

    """

    if ignorenan:
        mean = scipy.stats.nanmean
        std = scipy.stats.nanstd
    else:
        mean = np.mean
        std = np.std

    _assert_signal(s)
    axis = s.samples.axes[axis]
    samples = s.samples.get(copy=True)
    m = mean(samples, axis=axis)
    unsqueeze(data=m, axis=axis, oldshape=samples.shape)
    samples -= m
    if sdone:
        sd = std(samples, axis=axis)
        unsqueeze(data=sd, axis=axis, oldshape=samples.shape)
        samples /= sd
    return s._instantiatefromself(samples=samples)
