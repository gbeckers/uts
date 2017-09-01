#!/usr/local/bin/python
# -*- coding: utf-8 -*-

"""
The processing module holds functions that operate on time sampled signals.


"""
from __future__ import print_function

import sys
import time
import warnings
import numpy as np
import scipy.signal
import signalprocessing
from signalprocessing import segment_axis, fit_frames, multi_taper_psd, \
    iminterpolate
from ..core.utils import round_nearesteven, round_nextradix2
from ..core.uniformtime import UniformTimeSeries, \
    MultiChannelUniformTimeSeries, EpochedMultiChannelUniformTimeSeries, \
    assert_signal
from ..core.epoched import Epoched
from ..core.geometry import RectangularLattice
from ..core.uniformfrequency import Spectrum, PowerSpectrum, \
    MultiChannelPowerSpectrum
from ..core.uniformtimefrequency import Spectrogram
from .helper import convadjusttime
from ..core.check_arg import check_arg, check_int, check_float, check_bool
from ..core.uniformtime import isuts, iseuts, ismcuts, isemcuts
from .ifdv import cifdv
from ..helper.progressbar import ProgressBar
from .windows import windows

from .parallel import foreach
from ..signalfactory.factory import fromchannels

__all__ = ['bp', 'convolve', 'crosscorrelatetime',
           'crosscorrelatetime_betweenchannels',
           'crosscorrelatetime_betweenchannels_p', 'decimate', 'hp', 'ibp',
           'ica', 'ifdv', 'ihp', 'ilp', 'iter_convolve', 'iter_convolve_info',
           'iter_decimate', 'iter_map', 'fft', 'lp', 'overlap_and_add', 'ramp',
           'resample', 'specgram', 'stft', 'stmtt', 'welch']

def overlap_and_add(tss1, tss2, overlapdur):
    """
    Concatenates two time-sampled signals along the time axis with an overlap.

    At the end of the first signal and the beginning of the second signals a 
    time segments of noverlap samples are windowed with a half hanning window
    and then summed, so that the transition betweem the two sounds is smooth.

    Parameters
    ----------
    tss1: first time sampled signal
    tss2: second time sampled signal
    overlapdur: float
        The duration of the time segment where overlap occurs.

    Returns
    -------
    A new object of the same class of tss1 and tss2


    """

    assert_signal(tss1)
    assert_signal(tss2)
    window = np.hanning
    overlapdur = check_float(overlapdur)
    noverlap = round_nearesteven(overlapdur * tss1.fs)
    if noverlap > tss1.ntimesamples:
        raise ValueError("overlapdur larger than tss1")
    if noverlap > tss2.ntimesamples:
        raise ValueError("overlapdur larger than tss2")
    #FIXME we should also check non-time dimensions are equal
    #FIXME if tss2 does not have the right axes config, we could transpose
    if not (tss1.samples.axes == tss2.samples.axes):
        raise ValueError("signals do not have the same sample axes")
    if not (tss1.fs == tss2.fs):
        raise ValueError("signals do not have the same sample axes")

    w = window(noverlap * 2 + 1)
    w1 = w[noverlap:]
    w2 = w[:noverlap + 1]
    nsamples = tss1.ntimesamples + tss2.ntimesamples - len(w1)
    shape = list(tss1.samples.shape)
    shape[tss1.samples.axes['time']] = nsamples
    samples = np.zeros(shape, dtype=tss1.samples.dtype)

    s = tss1._instantiatefromself(samples=samples)
    s[:tss1.ntimesamples - len(w1)].samples.array[:] = tss1[:-len(
        w1)].samples.array[:]
    s[tss1.ntimesamples:].samples.array[:] = tss2[len(w1):].samples.array[:]
    s[tss1.ntimesamples - len(w1):tss1.ntimesamples].samples.array[:] \
        = np.apply_along_axis(np.multiply,
                              tss1.samples.axes['time'],
                              tss1[-len(w1):].samples.array[:],
                              w1)
    s[tss1.ntimesamples - len(w2):tss1.ntimesamples].samples.array[:] \
        = s[tss1.ntimesamples - len(w2):tss1.ntimesamples].samples.array[:] \
          + np.apply_along_axis(np.multiply, tss1.samples.axes['time'],
                                tss2[:len(w2)].samples.array[:],
                                w2)

    return s

def apply_to_samples(tss, f, copy=True, **nargs):
    """
    Applies a function that operates on a sample-by-sample basis on a signal.

    Parameters
    ----------
    tss: time sampled signal
    f: function
        The first argument of the function should represent the samples.
    copy: {True|False}, optional
        Determines if the f operates on the samples of ts directly, or on a
        copy. Note: if 'copy' is False, it is possible to change the original
        samples of ts. Use this parameter wisely! Default: True.
    **nargs: additional arguments, optional
        These parameters will be passed to f, if it accepts it.

    Returns
    -------
    A new object of the same class of ts, but with the new samples

    Example
    -------
    >> def mult(samples, factor):
           return samples * factor
    >>
    >> nts = apply_to_samples(ts, mult, copy=False, factor=3})


    """
    assert_signal(tss)
    samples = f(tss.samples.get(copy=copy), **nargs)
    return tss._instantiatefromself(samples=samples, copy=False)

def crosscorrelatetime(s1, s2, normalized=True):
    """
    Calculates the *valid* crosscorrelation values between two time signals, energy-corrected
    for duration (of the shortest signal).

    We assume that when the signals are not equal in duration, they are centered in time:
        s1: t1 t2 t3 t4 t5
        s2:    t2 t3 t4

    Note that the shorter signal is shifted over the longer one. A positive lag therefore
    means that the longer signal is lagging.

    """

    from uts.processing.correlate import crosscorrelate_1d, crosscorrelate_2d

    assert_signal(s1)
    assert_signal(s2)
    s1 = s1.astype(np.float64)
    s2 = s2.astype(np.float64)
    check_bool(normalized)
    if not (s1.axes == s2.axes):
        raise ValueError('axes are not the same')
    if not (s1.fs == s2.fs):
        raise ValueError('sampling rates are different')
    corrlen = np.abs(s1.ntimesamples - s2.ntimesamples) + 1
    args = {}
    args['origintime'] = corrlen / (s1.fs * 2.0)
    args['copy'] = False
    args['dtype'] = np.float64
    if isuts(s1):
        samples = crosscorrelate_1d(s1.samples.get(copy=False),
                                    s2.samples.get(copy=False),
                                    normalized=normalized)
        s = s1._instantiatefromself(samples=samples, **args)
    elif iseuts(s1):
        samples = crosscorrelate_2d(
            a=s1.transpose(['epoch', 'time']).samples.get(False),
            b=s2.transpose(['epoch', 'time']).samples.get(False),
            normalized=normalized)
        s = s1._instantiatefromself(samples=samples, epochaxis=0, **args)
    elif ismcuts(s1):
        samples = crosscorrelate_2d(
            a=s1.transpose(['channel', 'time']).samples.get(False),
            b=s2.transpose(['channel', 'time']).samples.get(False),
            normalized=normalized)
        s = s1._instantiatefromself(samples=samples, channelaxis=0, **args)
    elif isemcuts(s1):
        #samples = crosscorrelate(a=s1.transpose(['channel','epoch','time']).samples.get(False),
        #    b=s2.transpose(['channel','epoch','time']).samples.get(False), normalized=normalized)
        #s = s1._instantiatefromself(samples=samples, channelaxis=0, epochaxis=1, **args)
        raise NotImplementedError

    return s

def crosscorrelatetime_betweenchannels(s, lag, normalized=True,
                                       reportprogress=True):
    """
    If time is positive it means that the first channel is leading.
    """
    lag = check_arg(lag, 'duration')
    nlag = int(np.round(lag * s.fs))
    corrlen = nlag * 2 + 1
    if not ismcuts(s):
        raise TypeError(
            "this function works only with MultiChannelUniformTimeSeries")
    samples = np.empty((s.nchannels * s.nchannels, corrlen), s.dtype)
    k = 0
    channelnames = []
    for i, channeli in enumerate(s.channelnames):
        for j, channelj in enumerate(s.channelnames):
            if reportprogress:
                print('<%d,%d>' % (i, j), sep='', end='')
                sys.stdout.flush()
            samples[k, :] = crosscorrelatetime(s1=s[:, i], s2=s[nlag:-nlag, j],
                                               normalized=normalized).samples.get(
                False)
            channelnames.append('cc_%s_%s' % (channeli, channelj))
            k += 1
    g = RectangularLattice(s.nchannels, s.nchannels, channelnames)
    return MultiChannelUniformTimeSeries(samples=samples, fs=s.fs, geometry=g,
                                         channels=channelnames, channelaxis=0,
                                         origintime=corrlen / (s.fs * 2.0))

def _cct(args):
    s, i, j, nlag, normalized, reportprogress = args
    if reportprogress:
        print('<%d,%d>' % (i, j), sep='', end='')
        sys.stdout.flush()
    return crosscorrelatetime(s1=s[:, i], s2=s[nlag:-nlag, j],
                              normalized=normalized).samples.get(False)

def crosscorrelatetime_betweenchannels_p(s, lag, normalized=True,
                                         reportprogress=True, nprocesses=2):
    """
    If time is positive it means that the first channel is leading.

    """
    from multiprocessing import Pool

    pool = Pool(processes=nprocesses)

    lag = check_arg(lag, 'duration')
    nlag = int(np.round(lag * s.fs))
    corrlen = nlag * 2 + 1
    if not ismcuts(s):
        raise TypeError(
            "this function works only with MultiChannelUniformTimeSeries")
    # samples = np.empty((s.nchannels*s.nchannels,corrlen), s.dtype)
    channelnames = []
    args = []
    for i, channeli in enumerate(s.channelnames):
        for j, channelj in enumerate(s.channelnames):
            args.append([s, i, j, nlag, normalized, reportprogress])
            channelnames.append('cc_%s_%s' % (channeli, channelj))
    results = pool.map(_cct, args)
    samples = np.array(results, dtype=s.dtype)
    g = RectangularLattice(s.nchannels, s.nchannels, channelnames)
    return MultiChannelUniformTimeSeries(samples=samples, fs=s.fs, geometry=g,
                                         channels=channelnames, channelaxis=0,
                                         origintime=corrlen / (s.fs * 2.0))

def convolve(uts, kernel, mode='same', threads=None,
             dtype=None):  #, out=None):
    """
    Convolves self with a kernel.

    Parameters
    ----------
    uts: uniform time series
    kernel: array_like
    mode: {'full', 'valid', 'same'}, optional
        See numpy.convolve documentation for explanation. Note that default
        here is 'same', in contrast to numpy.convolve

    Returns
    -------
    A convolved version of uts. Note that sampling times are calculated so
    that uts is a process that occurs in time, while the kernel is just a
    sequence of numbers where time is irrelevant. That is: uts will not be
    shifted in time, just convolved.

    """
    assert_signal(uts)
    if dtype is None:
        dtype = uts.dtype
    else:
        uts = uts.astype(dtype)
        dtype = uts.dtype
    kernel = np.array(kernel, copy=False, dtype=dtype)
    #print(uts.dtype, kernel.dtype)
    ## is the following necessary?? We want to be flexible and upcast
    #if kernel.dtype != uts.dtype:
    #    kernel = kernel.astype(uts.dtype)

    if ismcuts(uts) and (not threads is None):
        return pmcconvolve2(uts, kernel=kernel, mode=mode, threads=threads)

    samples = np.apply_along_axis(np.convolve, uts.samples.axes['time'],
                                  uts.samples.get(copy=False), kernel, mode)
    timeshift = convadjusttime(fs=uts.fs, kernelsize=kernel.size, mode=mode)
    args = {}
    if isinstance(uts, Epoched):
        if uts.epochs is None:
            args['epochs'] = None
        else:
            epochs = uts.epochs.copy()
            epochs['starttime'] = uts.starttimes + timeshift
            args['epochs'] = epochs
    else:
        args['starttime'] = uts.starttime + timeshift
    args['origintime'] = uts.origintime - timeshift
    args['copy'] = False
    # if not (out is None):
    #     return out
    # else:
    return uts._instantiatefromself(samples=samples, dtype=samples.dtype,
                                    **args)

def pmcconvolve(mcuts, kernel, mode='same', threads=4):
    return fromchannels(foreach(convolve, mcuts.iter_channels(),
                                threads=threads, return_=True,
                                args=[kernel, mode]),
                        channelnames=mcuts.channelnames,
                        geometry=mcuts.geometry)

def pmcconvolve2(mcuts, kernel, mode='same', threads=4):
    if mcuts.channelaxis == 1:
        ars = [mcuts.samples.get(False)[:, i] for i in xrange(mcuts.nchannels)]
    else:
        ars = [mcuts.samples.get(False)[i, :] for i in xrange(mcuts.nchannels)]

    ars = foreach(np.convolve, ars, threads=threads, return_=True,
                  args=[kernel, mode])
    ars = np.array(ars, dtype=mcuts.dtype)
    if mcuts.channelaxis == 1:
        ars = ars.T
    timeshift = convadjusttime(fs=mcuts.fs, kernelsize=kernel.size, mode=mode)
    kwargs = {}

    kwargs['origintime'] = mcuts.origintime - timeshift
    kwargs['copy'] = False

    return mcuts._instantiatefromself(samples=ars, **kwargs)

def iter_convolve(uts, kernel, mode='same', chunksize=1024 * 64,
                  reportprogress=False, channels=None, dtype=None,
                  threads=None):
    '''
    Convolves uts with kernel in chunks instead of the whole signal at
    once. This may be necessary if the signal is too long to fit in memory.

    Parameters
    ----------
    uts: uniform time series
    kernel: array_like
    mode: {'full', 'valid', 'same'}, optional
        See numpy.convolve documentation for explanation. Note that default
        here is 'same', in contrast to numpy.convolve
    chunksize: int, optional
        The size in samples of signal chunks that are processed in sequence.
        Default is 1024*64 = 65536 .
    reportprogress: {True, False}
        Deteremines whether or not info on progress should be printed to
        stdout.

    Returns
    -------
    A generator that yields the convolution result in a chunks. Note that
    the size of the returned chunks is not exactly equal to the parameter
    `chunksize'. It will depend on the kernel size and `mode'. The size of
    the first and the last chunk may be different from each other and from
    the chunks in between (which are all equal in size).

    '''

    # Note on convolution modes
    #
    # 'full'  always retuns a sequence of length len(signal) + len(kernel) - 1
    # 'same'  always returns a sequence of length len(signal)
    # 'valid' always returns a sequence of length len(signal) - len(kernel) + 1

    assert mode in ('valid', 'same', 'full')
    chunksize = check_int(chunksize, negative=False, zero=False)
    kernel = np.asarray(kernel)
    if not (chunksize > kernel.size):
        chunksize = kernel.size + 1
        warnings.warn(
            "Chunksize not larger than kernel size. Setting it to %d" % chunksize)
    validntimesamples = len(kernel) - 1
    stepsize = chunksize - validntimesamples
    nframes, newsize, remainder = fit_frames(
        uts.ntimesamples, chunksize, stepsize=stepsize)
    # print "nframes", "newsize", "remainder", nframes, newsize, remainder
    if remainder > 0:
        lastframei = nframes
    else:
        lastframei = nframes - 1
    if reportprogress:
        p = ProgressBar(nframes)
    if channels is None:
        gen = enumerate(uts.iter_frames(framesize=chunksize,
                                        stepsize=stepsize,
                                        include_remainder=True))
    else:
        gen = enumerate(uts.iter_frames(framesize=chunksize,
                                        stepsize=stepsize,
                                        channels=channels,
                                        include_remainder=True))
    #print "chunksize, stepsize", chunksize, stepsize
    for i, ss in gen:
        #print 'frame', i, ss.samples[slice(None,None,None)]
        #t1 = time.time()
        # print i, ss, ss.samples[:]
        if reportprogress:
            p.animate(i)

        if (i == 0) and (mode != 'valid'):
            if mode == 'same':
                head = (kernel.size - 1) // 2
                if nframes > 0:
                    tail = 1 - kernel.size
                else:
                    tail = -(kernel.size - 1) // 2
                if tail == 0: tail = None
                result = convolve(ss, kernel, mode='full', threads=threads,
                                  dtype=dtype)[slice(head, tail, None)]
            else:  # mode == 'full'
                #tail = -(min(ss.ntimesamples, kernel.size) - 1)
                tail = -min(ss.ntimesamples,
                            kernel.size) + 1  ##IS THIS CORRECT?
                if tail == 0: tail = None
                result = convolve(ss, kernel, mode='full', threads=threads,
                                  dtype=dtype)[slice(None, tail, None)]
        elif (i == lastframei) and (mode != 'valid'):
            if mode == 'same':
                #print "I am at lastframei"
                head = kernel.size - 1
                tail = -(kernel.size - 1) // 2
                if tail == 0: tail = None
                result = convolve(ss, kernel, mode='full', threads=threads,
                                  dtype=dtype)[slice(head, tail, None)]
                #print 'head, tail, ss', head, tail, ss
            else:  # mode == 'full'
                head = min(ss.ntimesamples, kernel.size) - 1
                result = convolve(ss, kernel, mode='full', threads=threads,
                                  dtype=dtype)[slice(head, None, None)]
        else:
            result = convolve(ss, kernel, mode='valid', threads=threads,
                              dtype=dtype)
        #t2 = time.time()
        #print 'ic %d in %.3f s.' % (i,t2-t1)
        yield result


# FIXME chunks/frames choose one term
def iter_convolve_info(uts, kernel, mode='same', chunksize=1024 * 64):
    '''
    Provides data about how iter_convolve will segment uts and return chunks.

    'chunksize' * 'nmidframes' + 'firstchunksize' + 'lastchunksize' should equal
    the number of samples that is returned in total by iter_convolve.
    
    Parameters
    ----------
    uts: uniform time series
    kernel: array_like
    mode: {'full', 'valid', 'same'}, optional
        See numpy.convolve documentation for explanation. Note that default
        here is 'same', in contrast to numpy.convolve
    chunksize: int, optional
        The size in samples of signal chunks that are processed in sequence.
        Default is 1024*64 = 65536 .
    
    Returns
    -------
    A dictionary with a number of keys:

    chunksize: the size of a frame that is not the first of last

    firstchunksize: the number of time samples in the first chunk
    
    lastchunksize: the number of time samples in the last chunk
    
    nmidframes: the number of frames excluding the first and last
    
    starttime: starttime of the returned time signal
    
    origintime: origintime of the returned time signal
    
    '''

    assert mode in ('valid', 'same', 'full')
    chunksize = check_int(chunksize, negative=False, zero=False)
    kernel = np.asarray(kernel)
    if not (chunksize > kernel.size):
        chunksize = kernel.size + 1
        warnings.warn(
            "Chunksize not larger than kernel size. Setting is to %d" % chunksize)
    validntimesamples = len(kernel) - 1
    stepsize = chunksize - validntimesamples
    nframes, newsize, remainder = fit_frames(uts.ntimesamples, chunksize,
                                             stepsize=stepsize)
    #print "nframes", "newsize", "remainder", nframes, newsize, remainder
    info = {}
    if remainder > 0:
        lastframei = nframes
        info['nmidframes'] = nframes - 1  # minus last frame
    else:
        lastframei = nframes
        info['nmidframes'] = nframes - 2

    # first chunk
    if (mode == 'valid'):
        info['firstchunksize'] = chunksize - validntimesamples
    elif mode == 'same':
        head = (kernel.size - 1) // 2
        if nframes > 0:
            tail = 1 - kernel.size
        else:
            tail = -(kernel.size - 1) // 2
        info['firstchunksize'] = chunksize + kernel.size - 1 + tail - head
    else:  # mode == 'full'
        tail = min(chunksize, kernel.size) - 1
        info['firstchunksize'] = chunksize + validntimesamples - tail

    # last chunk
    if (remainder == 0):
        lastchunksize = chunksize
    else:
        lastchunksize = chunksize - stepsize + remainder
    if (mode == 'valid'):
        info['lastchunksize'] = lastchunksize - kernel.size + 1
    elif mode == 'same':
        head = kernel.size - 1
        tail = -(kernel.size - 1) // 2
        info['lastchunksize'] = lastchunksize + kernel.size - 1 - head + tail
    else:  # mode == 'full'
        head = min(lastchunksize, kernel.size) - 1
        info['lastchunksize'] = lastchunksize + kernel.size - 1 - head

    info['chunksize'] = chunksize - validntimesamples
    info['ntimesamples'] = info['nmidframes'] * info['chunksize'] \
                           + info['firstchunksize'] + info['lastchunksize']
    timeshift = convadjusttime(fs=uts.fs, kernelsize=kernel.size, mode=mode)
    info['starttime'] = uts.starttime + timeshift
    info['origintime'] = uts.origintime - timeshift
    return info

def iter_decimate(uts, q, trunc='tail', chunksize=1024 * 64,
                  reportprogress=False, dtype=None):
    assert_signal(uts)
    if not trunc in ('tail', 'head'):
        raise ValueError("trunc parameter can only be 'tail' or 'head'")
    q = check_int(q, negative=False, zero=False)
    starttime = uts.starttime
    origintime = uts.origintime
    chunksize = check_int(chunksize, negative=False, zero=False) * q
    stepsize = chunksize
    nframes, newsize, remainder = fit_frames(
        uts.ntimesamples, chunksize, stepsize=stepsize)
    # print "nframes", "newsize", "remainder", nframes, newsize, remainder
    if (remainder > 0):
        if trunc == 'tail':
            start = 0
            stop = newsize
        else:
            start = remainder
            stop = remainder + newsize
    else:
        start = 0
        stop = newsize
    if reportprogress:
        p = ProgressBar(nframes)

    gen = enumerate(uts.iter_frames(framesize=chunksize,
                                    stepsize=stepsize,
                                    include_remainder=True))
    #print "chunksize, stepsize", chunksize, stepsize
    for i, frame in gen:
        if reportprogress:
            p.animate(i)
        if dtype is None:
            yield decimate(frame, q=q)
        else:
            yield decimate(frame, q=q).astype(dtype)

def decimate(uts, q, trunc='tail'):
    """
    Decimates the signal by an integer factor q, by averaging consecutive
    frames of q samples.

    Note that, generally, the signal should be low-pass *filtered* before
    doing this. Filtering and decimation can be done in one step using the
    method `resample`.

    Parameters
    ----------
    uts: uniform time series
    q: int
        decimation factor; a value of 10 will result in a signal with a
        sampling rate that is 10 times smaller than the current one.
    trunc: {'tail', 'head'}, optional
        If there are remaining samples, because ntimesamples is not a integer
        multiple of q, the signal is truncated either at the head or the
        tail. Tail is default.

    Note
    ----
    Although the `resample` method can in principle be used for the same
    thing, but then more flexibly, this method is still convenient in cases
    where one wants to apply a special kind of filter, and then decimate the
    signal. One has to make sure however that the lowest frequency in the
    signal does not exceed the new sampling rate divided by 2. Otherwise
    aliasing will occur.

    """
    assert_signal(uts)
    if not trunc in ('tail', 'head'):
        raise ValueError("trunc parameter can only be 'tail' or 'head'")
    q = check_int(q, negative=False, zero=False)
    nframes, remainder = divmod(uts.ntimesamples, q)
    starttime = uts.starttime
    origintime = uts.origintime
    if remainder > 0:
        if trunc == 'tail':
            samples = uts[slice(None, -remainder, None)].samples.get(
                copy=False)
        else:  # trunc == 'head'
            samples = uts[slice(remainder, None, None)].samples.get(copy=False)
            remaindertime = remainder / uts.fs
            origintime -= remaindertime
            starttime += remaindertime
    else:
        samples = uts.samples.get(copy=False)
    timeaxis = uts.samples.axes['time']
    samples = segment_axis(a=samples, length=q, overlap=0,
                           axis=timeaxis, end='cut').mean(axis=timeaxis + 1)
    if uts.dtype != samples.dtype:
        samples = samples.astype(uts.dtype)
    fs = uts.fs / float(q)
    return uts._instantiatefromself(samples=samples, fs=fs, copy=False)

def iter_map(f, gen, *args, **kwargs):
    for frame in gen:
        yield f(frame, *args, **kwargs)

def resample(uts, factor, window=None):
    """
    Uses scipy.signal.resample to resample signal.

    Parameters
    ----------
    uts: uniform time series
    factor: float
        Resampling factor, i.e. a factor 0.5 will result in a signal with
        half the number of samples.
    window: windowname, float, or tuple
        (see scipy.signal.resample docstring for explanation)

    """
    assert_signal(uts)
    factor = check_float(factor, negative=False, zero=False)
    num = int(np.round(factor * uts.ntimesamples))
    timeaxis = uts.samples.axes['time']
    samples = scipy.signal.resample(x=uts.samples.get(copy=True),
                                    num=num, t=None, axis=timeaxis,
                                    window=window)
    if samples.dtype != uts.dtype:
        samples = samples.astype(uts.dtype)
    fs = factor * uts.fs
    origintime = 0.5 / fs - 0.5 / uts.fs  # this seems to be necessary to get everything aligned
    return uts._instantiatefromself(samples=samples, fs=fs,
                                    origintime=origintime, copy=False)

def fft(tss, n=None):
    assert_signal(tss, (UniformTimeSeries,))

    samples = scipy.fftpack.fftshift(
        np.fft.fft(tss.samples.get(copy=False), n=n) / float(
            tss.ntimesamples) ** 0.5)
    samplingfrequencies = scipy.fftpack.fftshift(
        scipy.fftpack.fftfreq(samples.size, d=tss.dt))
    return Spectrum(samples=samples,
                    samplingfrequencies=samplingfrequencies, copy=False)

def stft(tss, windowtype='gaussian', windowduration=None, fftduration=None,
         stepduration=None, symmetric=True):
    """
    Transform is energy-corrected, *after* windowing.
    
    """

    assert_signal(tss, (UniformTimeSeries,))

    # check parameters
    if windowtype not in windows.keys():
        raise ValueError("unsupported window type")
    if windowduration is None:
        windowduration = 512 / tss.fs
    check_arg(windowduration, 'windowduration')
    windowsize = int(np.round(windowduration * tss.fs))

    if fftduration is None:
        fftduration = windowduration
    check_arg(fftduration, 'fftduration')
    fftsize = int(np.round(fftduration * tss.fs))

    if stepduration is None:
        stepduration = 0.25 * windowduration
    check_arg(stepduration, 'stepduration')
    stepsize = int(np.round(stepduration * tss.fs))

    if not (symmetric in (True, False)):
        raise ValueError("symmetric must be either True or False")



        #stft = strft(a=tss.samples.get(copy=False), windowsize=windowsize,
        #stepsize=stepsize, windowtype=windowtype, fftsize=fftsize,
        #symmetric=symmetric, power=False)
    stft, firstsamplecenter = signalprocessing.stft(
        s=tss.samples.get(copy=False),
        windowsize=windowsize, stepsize=stepsize,
        windowtype=windowtype, fftsize=fftsize,
        symmetric=symmetric, real=False)
    fs = 1. / stepduration
    startfrequency = 0.0 - 0.5 * tss.fs / float(fftsize)
    starttime = 0.0  #FIXME
    origintime = 0.0  #FIXME
    return Spectrogram(samples=stft, fs=fs, bandwidth=tss.fs,
                       startfrequency=startfrequency, frequencyaxis=1,
                       starttime=starttime, origintime=origintime, copy=False)

def specgram(tss, windowtype='gaussian', windowduration=None, fftduration=None,
             stepduration=None, symmetric=False):
    """
    Transform is energy-corrected, *after* windowing.
    
    """

    assert_signal(tss, (UniformTimeSeries,))

    # check parameters
    if windowtype not in windows.keys():
        raise ValueError("unsupported window type")
    if windowduration is None:
        windowduration = 512 / tss.fs
    check_arg(windowduration, 'windowduration')
    windowsize = int(np.round(windowduration * tss.fs))

    if fftduration is None:
        fftduration = windowduration
    check_arg(fftduration, 'fftduration')
    fftsize = int(np.round(fftduration * tss.fs))

    if stepduration is None:
        stepduration = 0.25 * windowduration
    check_arg(stepduration, 'stepduration')
    stepsize = int(np.round(stepduration * tss.fs))

    if not (symmetric in (True, False)):
        raise ValueError("symmetric must be either True or False")

    S, firstsamplecenter = signalprocessing.specgram(
        s=tss.samples.get(copy=False),
        windowsize=windowsize, stepsize=stepsize,
        windowtype=windowtype, fftsize=fftsize,
        symmetric=symmetric)
    fs = 1. / stepduration
    samplingfrequencies = np.arange(S.shape[-1]) * (tss.fs / fftsize)
    starttime = 0.0  # FIXME
    origintime = 0.0  # FIXME
    bandwidth = tss.fs / 2.0
    return Spectrogram(samples=S, fs=fs, bandwidth=bandwidth,
                       samplingfrequencies=samplingfrequencies,
                       frequencyaxis=1, starttime=starttime,
                       origintime=origintime,
                       copy=False)

def welch(tss, windowtype='gaussian', windowduration=None, fftduration=None,
          stepduration=None, detrend='constant'):
    from scipy.signal import welch

    assert_signal(tss, (UniformTimeSeries, MultiChannelUniformTimeSeries))

    # check parameters
    if windowtype not in windows.keys():
        raise ValueError("unsupported window type")
    if windowduration is None:
        windowduration = 512 / tss.fs
    check_arg(windowduration, 'windowduration')
    windowsize = int(np.round(windowduration * tss.fs))

    if fftduration is None:
        fftduration = windowduration
    check_arg(fftduration, 'fftduration')
    fftsize = int(np.round(fftduration * tss.fs))

    if stepduration is None:
        stepduration = 0.25 * windowduration
    check_arg(stepduration, 'stepduration')
    stepsize = int(np.round(stepduration * tss.fs))
    window = windows[windowtype](windowsize)
    noverlap = windowsize - stepsize
    if ismcuts(tss):
        samples = tss.samples.transpose(['channel', 'time']).get(False)
    else:
        samples = tss.samples.get(False)
    f, Pxx = welch(samples, fs=tss.fs, window=window, nperseg=windowsize,
                   noverlap=noverlap, nfft=fftsize, detrend=detrend)
    samplingfrequencies = np.arange(Pxx.shape[-1]) * (tss.fs / fftsize)
    bandwidth = tss.fs / 2.0
    if ismcuts(tss):
        return MultiChannelPowerSpectrum(samples=Pxx,
                                         samplingfrequencies=samplingfrequencies,
                                         copy=False, channelaxis=0,
                                         channels=tss.channels,
                                         geometry=tss.geometry)
    else:
        return PowerSpectrum(samples=Pxx,
                             samplingfrequencies=samplingfrequencies,
                             copy=False)

# def periodogram(tss, windowtype='boxcar', fftduration=None, normalize=True):
#
#     assert_signal(tss, (UniformTimeSeries, MultiChannelUniformTimeSeries))
#
#     # check parameters
#     if windowtype not in windows.keys():
#         raise ValueError("unsupported window type")
#
#     windowsize = tss.ntimesamples
#     window = windows[windowtype](windowsize)
#
#     if fftduration is None:
#         fftsize = tss.ntimesamples
#     else:
#         check_arg(fftduration, 'fftduration')
#         fftsize = int(np.round(fftduration * tss.fs))
#
#     if fftsize < windowsize:
#         warnings.warn("fftsize smaller than signal. Truncating signal to %d samples." % fftsize)
#         windowsize = fftsize
#
#     if ismcuts(tss):
#         samples = tss[:windowsize].samples.transpose(['channel','time']).get(False)
#     else:
#         samples = tss[:windowsize].samples.get(False)
#     if windowtype != 'boxcar':
#         samples  = samples * window
#     samplingfrequencies, Pxx = signalprocessing.periodogram(samples, fs=tss.fs, Sk=None,
#                                 fftsize=fftsize, sides='onesided', normalize=normalize)
#     bandwidth = tss.fs / 2.0
#     if ismcuts(tss):
#         return MultiChannelPowerSpectrum(samples=Pxx,
#                     samplingfrequencies=samplingfrequencies, copy=False, channelaxis=0,
#                     channels=tss.channels, geometry=tss.geometry)
#     else:
#         return PowerSpectrum(samples=Pxx,
#                     samplingfrequencies=samplingfrequencies, copy=False)



def stmtt(tss, BW=None, windowduration=None, stepduration=None,
          symmetric=True):
    assert_signal(tss, (UniformTimeSeries,))

    # check parameters
    if windowduration is None:
        windowduration = 512 / tss.fs
    if stepduration is None:
        stepduration = 0.25 * windowduration
    check_arg(windowduration, 'windowduration')
    check_arg(stepduration, 'stepduration')
    if not (symmetric in (True, False)):
        raise ValueError("symmetric must be either True or False")
    windowsize = int(np.round(windowduration * tss.fs))
    stepsize = int(np.round(stepduration * tss.fs))
    nframes, newsize, remainder = fit_frames(
        totalsize=tss.ntimesamples, framesize=windowsize, stepsize=stepsize)
    if symmetric:
        startbin = int(remainder // 2.0)
    else:
        startbin = 0
    overlap = windowsize - stepsize
    s = segment_axis(a=tss.samples.get()[startbin:newsize],
                     length=windowsize, overlap=overlap, axis=None, end='cut',
                     endvalue=0)
    freqs, psd_est = multi_taper_psd(s=s, BW=BW,
                                     fs=tss.fs, sides='twosided')
    fs = 1. / stepduration
    startfrequency = 0.0 - 0.5 * tss.fs / windowsize
    starttime = 0.0  #FIXME
    origintime = 0.0  #FIXME
    return Spectrogram(samples=psd_est, fs=fs, bandwidth=tss.fs,
                       startfrequency=startfrequency, frequencyaxis=1,
                       starttime=starttime, origintime=origintime, copy=False)

def ifdv(tss, windowduration=None, stepduration=None,
         sigma=2.0, zoomt=1, zoomf=3, tl=5, fl=5, symmetric=True,
         consensus=False, sigmarange=np.arange(0.5, 3.6, 0.1)):
    assert_signal(tss, (UniformTimeSeries,))

    if windowduration is None:
        windowduration = 1024 / tss.fs
    if stepduration is None:
        stepduration = 0.1 * windowduration
    windowsize = int(np.round(windowduration * tss.fs))
    stepsize = int(np.round(stepduration * tss.fs))
    fftduration = round_nextradix2(windowduration * tss.fs) / tss.fs
    fftsize = int(np.round(fftduration * tss.fs))
    noverlap = fftsize - stepsize

    if not consensus:
        a, b = cifdv(tss.samples.get(copy=False), SAMPLING=tss.fs, N=fftsize,
                     OVERLAP=noverlap, sigma=sigma, ZOOMT=zoomt, ZOOMF=zoomf,
                     TL=tl, FL=fl,
                     symmetric=symmetric)
    else:
        mid = sigmarange.size // 2
        ifdgrams = []
        for i, sigma in enumerate(sigmarange):
            a, b = cifdv(tss.samples.get(copy=False), SAMPLING=tss.fs,
                         N=fftsize,
                         OVERLAP=noverlap, sigma=sigma, ZOOMT=zoomt,
                         ZOOMF=zoomf, TL=tl, FL=fl,
                         symmetric=symmetric)
            ifdgrams.append(a)
            if i == mid:
                sonogram = b.copy()
            if len(ifdgrams) == 1:
                sumifdgram = np.zeros_like(ifdgrams[0])
            elif len(ifdgrams) == 3:
                sumifdgram += ifdgrams[0] * ifdgrams[1] * ifdgrams[2]
                del ifdgrams[0]
        a, b = sumifdgram, sonogram

    # FIXME Return spectrogram object
    return a, b

def ifdv_to_image(a, b, nyquistfreq, startfreq=0.0, endfreq=None, tbins=None,
                  fbins=None,
                  maxpow=None, dynamicrange=50, powerresolution=8,
                  sonolog=0.05, ifdlog=0.5):
    if tbins is None:
        tbins = b.shape[1]
    if fbins is None:
        fbins = b.shape[0]
    if endfreq is None:
        endfreq = nyquistfreq
    if fbins is None:
        fbins = b.shape[0]


    ## calculate the powerdensity and transform to dB scale
    ## we skip the negative frequencies because we deal with 
    ## real signals, so power spectrum is symmetric

    ifdgram = 10 * np.log10(a + ifdlog)
    ifdgram = iminterpolate(ifdgram, tbins, fbins,
                            (0., 1., 0., nyquistfreq),
                            (0., 1., startfreq, endfreq))

    #sonogram = 10*numpy.log10(self.sonograms[:,startfbin:endfbin+1]+sonolog)
    sonogram = 10 * np.log10(b + sonolog)

    sonogram = iminterpolate(sonogram, tbins, fbins,
                             (0., 1., 0., nyquistfreq),
                             (0., 1., startfreq, endfreq))

    x, y = tbins, fbins

    chan0 = ifdgram
    chan0 -= chan0.min()
    chan0 /= chan0.max()
    chan2 = sonogram
    chan2 -= chan2.min()
    chan2 /= chan2.max()

    index = chan0 > 0
    chan2[index] *= (1 - chan0[index])
    #chan1 = chan2.copy()
    #chan1[index] += chan0[index]

    im = np.empty([x, y, 3])

    im[..., 1] = chan2
    im[..., 2] = chan0
    im[..., 0] = chan2

    im[im > 0.75] = 0.75
    im *= 1 / 0.75

    #if self.nchannels == 1:
    #im = im.reshape(x,y,3)

    dynamicrange = 1.0
    scaling_factor = (pow(2, powerresolution) - 1) / (dynamicrange)
    im *= scaling_factor
    if powerresolution == 8:
        return im.astype('B')  # unsiged byte
    elif powerresolution == 16:
        return im.astype('H')  # unsiged short
    elif powerresolution == 32:
        return im.astype('L')  # unsiged int
    else:
        print("unknown resolution for sonogram; returning float64",
              file=sys.stderr)
        return im

def ica(tss, **kwargs):
    assert_signal(tss, (
        MultiChannelUniformTimeSeries, EpochedMultiChannelUniformTimeSeries))
    if isinstance(tss, EpochedMultiChannelUniformTimeSeries):
        samples = tss.samples.transpose(('epoch', 'time', 'channel')).get()
        samples = samples.reshape(-1, tss.nchannels)
    elif isinstance(tss, MultiChannelUniformTimeSeries):
        if tss.channelaxis == 0:
            samples = tss.samples.transpose(('time', 'channel')).get()
        else:
            samples = tss.samples.get()
    projmat, recmat = signalprocessing.ica(samples, return_node=True, **kwargs)
    return ICA(projmat=projmat.T, recmat=recmat.T, geometry=tss.geometry,
               channelnames=tss.channelnames)

class ICA(object):
    def __init__(self, projmat, recmat, geometry, channelnames):

        """
        projmat: ndarray (2D)
            The size of the first dimension should equal the number of 
            independent components. The size of the second axes should equal the
            number of sensors.
        recmat: ndarray (2D)
            The size of the first dimension should equal the number of 
            sensors. The size of the second axes should equal the number of 
            independent components.
        geometry: geometry
            Geometry of sensors
        channelnames: sequence
            Channelnames of sensors

        """

        self.projmat = projmat
        self.recmat = recmat
        self.geometry = geometry
        self.channelnames = channelnames
        if geometry is None:
            geometry = self._calc_geometry(recmat)
        self.nics = self.projmat.shape[0]
        s = 'IC%%.%dd' % int(np.ceil(np.log10(self.nics)))
        self.icnames = [s % i for i in range(self.nics)]

    def compgeom(self):
        im = np.empty([self.nics, self.geometry.ny,
                       self.geometry.nx], dtype=self.recmat.dtype)
        for j, compvec in enumerate(self.recmat.T):
            im[j] = self.geometry.mapfrom1Dto2D(compvec, self.channelnames)
        return im

    def show_compgeom(self, ncols=4, **figargs):
        import pylab

        nrows = np.ceil(self.nics / float(ncols))
        ims = self.compgeom()
        fig = pylab.figure(**figargs)
        pylab.ioff()
        row = 1
        col = 1
        for i, im in enumerate(ims):
            pylab.subplot(nrows, ncols, i + 1)
            pylab.imshow(im, aspect='auto', interpolation='nearest',
                         cmap=pylab.cm.gray)
            pylab.title(self.icnames[i])
            col += 1
            if col > ncols:
                row += 1
                col = 1
        pylab.draw()
        pylab.ion()
        return fig

    def saveh5(self, h5group, name):
        from ..io.hdf5 import geometry_to_h5attr
        f = h5group._v_file
        # the matrices are stored in one big array
        ar = np.empty([self.nics * 2, len(self.channelnames)],
                      dtype=self.projmat.dtype)
        ar[:self.nics, :] = self.projmat  # first nics have projmat
        ar[self.nics:, :] = self.recmat.T  # rest has transposed recmat
        n = f.createArray(h5group, name, ar)
        n._f_setAttr('dstype', 'ICA')
        geometry_to_h5attr(n, self.geometry)
        n._f_setAttr('channelnames', self.channelnames)
        doc = "The first %d items of this array ([:%d,:])contain the projection matrix; " % (
            self.nics, self.nics)
        doc += "the following %s ([%d:,:]) contain the transposed recording matrix." % (
            self.nics, self.nics)
        n._f_setAttr('doc', doc)
        f.flush()
        return n


def lp(uts, freq=300., width=50., ntaps=257, window="kaiser", mode='same',
       threads=None, dtype=None):
    """low-pass filter"""
    filterkernel = signalprocessing.lpfirkernel(N=ntaps, cutoff=freq,
                                                fs=uts.fs,
                                                width=width, window=window)
    return convolve(uts, filterkernel, mode=mode, threads=threads, dtype=dtype)

def ilp(uts, freq=300., width=50., ntaps=257, chunksize=1024 * 64,
        reportprogress=True,
        channels=None, window="kaiser", dtype=None, mode='same', threads=None):
    """iter low-pass filter"""
    filterkernel = signalprocessing.lpfirkernel(N=ntaps, cutoff=freq,
                                                fs=uts.fs,
                                                width=width, window=window)
    return iter_convolve(uts, filterkernel, mode=mode, chunksize=chunksize,
                         reportprogress=reportprogress, channels=channels,
                         dtype=dtype, threads=threads)

def hp(uts, freq=450., width=50., ntaps=513, window="kaiser", mode='same',
       threads=None, dtype=None):
    """high-pass filter"""
    filterkernel = signalprocessing.hpfirkernel(N=ntaps, cutoff=freq,
                                                fs=uts.fs,
                                                width=width, window=window)
    return convolve(uts, filterkernel, mode=mode, threads=threads, dtype=dtype)

def ihp(uts, freq=450., width=50., ntaps=513, chunksize=1024 * 64,
        reportprogress=True,
        channels=None, window="kaiser", dtype=None, mode='same', threads=None):
    """iter high-pass filter"""
    filterkernel = signalprocessing.hpfirkernel(N=ntaps, cutoff=freq,
                                                fs=uts.fs,
                                                width=width, window=window)
    return iter_convolve(uts, filterkernel, mode=mode, chunksize=chunksize,
                         reportprogress=reportprogress, channels=channels,
                         dtype=dtype, threads=threads)

def bp(uts, freqs=(400., 6000.), width=300., ntaps=257, window="kaiser",
       mode='same',
       threads=None):
    """high-pass filter"""
    filterkernel = signalprocessing.bpfirkernel(N=ntaps, freqs=freqs,
                                                fs=uts.fs,
                                                width=width, window=window)
    return convolve(uts, filterkernel, mode=mode, threads=threads)

def ibp(uts, freqs=(400., 6000.), width=300., ntaps=257, chunksize=1024 * 64,
        reportprogress=True,
        channels=None, window="kaiser", dtype=None, mode='same', threads=None):
    """iter high-pass filter"""
    filterkernel = signalprocessing.bpfirkernel(N=ntaps, freqs=freqs,
                                                fs=uts.fs,
                                                width=width, window=window)
    return iter_convolve(uts, filterkernel, mode=mode, chunksize=chunksize,
                         reportprogress=reportprogress, channels=channels,
                         dtype=dtype, threads=threads)

def bp_old(uts, lfreq=500., hfreq=6000., transband=200, ntaps=257):
    '''
    defaults are good for action potentials
    '''
    filterkern = signalprocessing.design_bp_filter(ntaps=ntaps, lfreq=lfreq,
                                                   hfreq=hfreq,
                                                   fs=uts.fs,
                                                   transband=transband)
    return convolve(uts, filterkern, mode='same')

def ibp_old(uts, lfreq=500., hfreq=6000., transband=200, ntaps=257,
            chunksize=1024 * 64, reportprogress=True, channels=None,
            dtype=None):
    t1 = time.time()
    filterkern = signalprocessing.design_bp_filter(ntaps=ntaps, lfreq=lfreq,
                                                   hfreq=hfreq,
                                                   fs=uts.fs,
                                                   transband=transband)
    t2 = time.time()
    #print 'ibp in %.3f s.' % (t2-t1)
    return iter_convolve(uts, filterkern, mode='same', chunksize=chunksize,
                         reportprogress=reportprogress, channels=channels,
                         dtype=dtype)

def fftconvolve(signal, kernellist, mode='same'):
    '''
    convolves the signal using fft

    Parameters:

    kernellist - a sequence of convolution kernels. We allow for more
        kernels because it is faster to give them at once, than it is to run
        this method repeatedly.
    mode -  {'same', 'full', 'valid'}, optional. See docstring of
        convolve() method for explanation.
    '''

    fft = np.fft.fft
    ifft = np.fft.ifft

    if isinstance(kernellist, (list, tuple)):
        pass
    elif isinstance(kernellist, np.ndarray):
        if kernellist.ndim == 1:
            kernellist = [kernellist]
    else:
        raise TypeError('kernellist of wrong type: %s' % type(kernellist))
    nkernels = len(kernellist)

    kernellengths = []
    complex_result = False
    for kernel in kernellist:
        kernel = np.array(kernel)
        assert kernel.ndim == 1
        complex_result |= np.issubdtype(kernel.dtype, np.complex)
        kernellengths.append(kernel.size)
    complex_result |= np.issubdtype(signal.dtype, np.complex)

    kernellengths = np.array(kernellengths)

    n = signal.nsamples + kernellengths.max() - 1
    if complex_result:
        dtype = np.complex128
    else:
        dtype = np.float64

    kernels = np.zeros((nkernels, n), dtype=dtype)
    for i, kernel in enumerate(kernellist):
        size = len(kernel)
        kernels[i, :size] = kernel
    kernelsfft = fft(kernels, axis=-1)

    c = ifft(
        fft(signal.get_samples(copy=False), n)[np.newaxis, :] * kernelsfft)
    if not complex_result:
        c = c.real

    results = []
    for i in range(nkernels):
        kernellength = kernellengths[i]
        starttime = signal.starttime
        origintime = signal.origintime
        if mode == 'full':
            samples = c[i, :signal.nsamples + kernellength - 1]
            # we have extra samples at the head and tail
            # the total number of extra samples is signal.size - 1
            extratime = ((kernellength - 1) / signal.fs) / 2.0
            starttime -= extratime
            origintime += extratime
        elif mode == 'valid':
            samples = c[i, kernellength - 1:signal.nsamples]
            # we have less samples than our original
            # the total number of lost samples is signal.size - 1
            losttime = (kernellength - 1) / signal.fs
            starttime += losttime / 2.0
            origintime -= losttime / 2.0
        else:  # mode == 'same'
            fulllength = signal.nsamples + kernellength - 1
            remainder = fulllength - signal.nsamples
            if (remainder) % 2 == 0:
                start = remainder / 2
            else:
                start = int(np.floor(remainder / 2.0))
            samples = c[i, start:start + signal.nsamples]
            # we have the same number of samples, but need a small time shift
            # correction if signal is even
            if (kernellength % 2) == 0:
                shifttime = signal.dt / 2.0
                starttime -= shifttime
                origintime += shifttime

        params = {}
        params['samples'] = samples
        params['starttime'] = starttime
        params['origintime'] = origintime
        params['copy'] = False

        results.append(signal._instantiatefromself(params))

    if nkernels == 1:
        return results[0]
    else:
        return results

def ramp(uts, duration=8e-3, window=np.hanning):
    assert_signal(uts)
    duration = check_arg(duration, 'duration')
    ruts = uts.copy()
    axis = ruts.samples.axes['time']
    ntimebins = int(np.round(duration * ruts.fs))
    w = window(ntimebins * 2)
    w1 = w[-ntimebins:]
    w2 = w[:ntimebins]
    samples = uts.samples.get(copy=True)
    ruts[:ntimebins].samples.array[:] = np.apply_along_axis(np.multiply, axis,
                                                            ruts[
                                                            :ntimebins].samples.array[
                                                            :], w2)
    ruts[-ntimebins:].samples.array[:] = np.apply_along_axis(np.multiply, axis,
                                                             ruts[
                                                             -ntimebins:].samples.array[
                                                             :], w1)
    return ruts
