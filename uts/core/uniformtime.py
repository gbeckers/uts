"""MultiChannelUniformTimeSeries
Contains classes representing uniform time.

Classes
=======

UniformTime, UniformTimeSeries, MultiChannelUniformTimeSeries,
EpochedUniformTimeSeries, EpochedMultiChannelUniformTimeSeries

"""
__all__ = ['UniformTime', 'UniformTimeSeries',
           'MultiChannelUniformTimeSeries', 'EpochedUniformTimeSeries',
           'EpochedMultiChannelUniformTimeSeries',
           'isuts', 'iseuts', 'ismcuts', 'isemcuts', 'concatenate',
           'assert_signal', 'istimealigned']

import warnings

import numpy as np

from .multichannel import MultiChannel
from .epoched import Epoched
from .defs import defined_integers
from .utils import fit_frames
from .samplesource import assamples
from .check_arg import check_arg, check_argarray, check_int, \
    check_intarray


class UniformTime(object):
    """
    Represents uniform time.

    Mainly intended to be used as a super class for classes that represent
    time-sampled signals, but may also be useful on its own.

    Parameters
    ----------
    ntimesamples: int
        Number of sampling times.
    fs: float
        Sampling rate in Herz (number of samples per second).
    starttime: float, optional
        Absolute time at which the signal starts relative to some user-known
        standard (e.g. Epoch, or start of recording). Defaults to 0.0.

    origintime: float, optional
    Time of origin relative to beginning of the first sample. 
    Defaults to 0.0.

    Attributes
    ----------
    ntimesamples: int64
        Number of samples.
    fs: float64
        Sampling rate in Herz (number of samples per second).
    dt: float64
        Sampling period in seconds.
    duration: float64
        Duration in seconds.
    starttime: float64
        Absolute time at which the signal starts relative to some user-known
        standard (e.g. Epoch, or start of recording).
    endtime: float64
        Absolute time at which the signal ends relative to some user-known
        standard (e.g. Epoch, or start of recording).
    origintime: float64
        Time of origin relative to beginning of the first sample. Defaults to
        0.0.
    timerange: (float64, float64)
        Start and end times, *relative to origintime*


    Methods (see method doc for explanation)
    ----------------------------------------
    time_to_index
    index_to_time
    samplingtimes

    Note
    ----
    Example that explains definitions of time (all in seconds):

                 |  s0  |  s1  |  s2  |  s3  |  s4  |  s5  |  s6  |
                 ^                                                ^
    absolute: starttime                                        endtime
            (e.g. 1452.3)                       (e.g. 1452.3 + duration)

                                 ^
    relative:                origintime
                          (defined as 0.0s)

    The methods `index_to_time` and `time_to_index`, and the attribute
    `timerange` all use relative time. `origintime` does not have to coincide
    with sample boundaries or centers.

    """

    def __init__(self, ntimesamples, fs, starttime=0.0, origintime=0.0):

        self.__fs = check_arg(fs, 'fs')
        self.__dt = 1.0 / self.__fs
        self.__starttime = check_arg(starttime, 'time')
        self.__origintime = check_arg(origintime, 'time')
        self.__ntimesamples = check_arg(ntimesamples, 'ntimesamples')
        self.__duration = self.__ntimesamples / self.__fs
        self.__endtime = self.__starttime + self.__duration
        self.__timerange = (0.0 - self.__origintime,
                            0.0 - self.__origintime + self.__duration)


    ################
    ## Properties ##
    ################
    ntimesamples = property(lambda self: (self.__ntimesamples), None,
                            None, "number of time samples")
    nt = ntimesamples
    fs = property(lambda self: self.__fs, None, None,
                  "sampling rate in Herz")
    dt = property(lambda self: self.__dt, None, None,
                  "sampling period in seconds")
    duration = property(lambda self: self.__duration, None, None,
                        "duration in seconds")
    starttime = property(lambda self: self.__starttime, None, None,
                         "absolute start time (begin of first sample) "
                         "in seconds")
    endtime = property(lambda self: self.__endtime, None, None,
                       "absolute end time (end of last sample) in seconds")
    origintime = property(lambda self: self.__origintime, None, None,
                          "time of origin in seconds *relative to starttime*")
    timerange = property(lambda self: self.__timerange, None, None,
                         "tuple representing timerange (start, end) in seconds"
                         " *relative to origintime*")


    ######################
    ## Standard methods ##
    ######################

    def __str__(self):
        return "<timesampled: %.3f seconds, %d times>" \
               % (self.duration, self.ntimesamples)

    __repr__ = __str__

    def info(self):
        from utils import epochtimetotext

        s = ''
        s += 'ntimesamples: %d\n' % self.ntimesamples
        s += 'duration: %.14f\n' % self.duration
        s += 'sampling rate: %.14f\n' % self.fs
        s += 'sampling period: %.14f\n' % self.dt
        s += 'starttime (in seconds since Epoch): %.14f\n' % self.starttime
        s += 'local starttime : %s\n' % epochtimetotext(self.starttime)
        s += 'origintime: %.14f\n' % self.origintime
        return s

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            result = False
        else:
            result = True
            result &= (self.__fs == other.fs)
            result &= (self.__dt == other.dt)
            result &= (self.__starttime == other.starttime)
            result &= (self.__origintime == other.origintime)
            result &= (self.__ntimesamples == other.ntimesamples)
            result &= (self.__duration == other.duration)
            result &= (self.__endtime == other.endtime)
            result &= (self.__timerange == other.timerange)
        return result


    ################
    ## Methods    ##
    ################

    def _update_ntimesamples(self, ntimesamples):
        """
        Update the number of samples. The `starttime` and `origintime`
        attributes will keep their original values, but `endtime`, `duration`
        and `samplingtimes` will be updated.

        Parameter
        ---------
        ntimesamples: int
            number of samples

        """
        self.__ntimesamples = check_arg(ntimesamples, 'ntimesamples')
        self.__duration = self.__ntimesamples / self.__fs
        self.__endtime = self.__starttime + self.__duration
        self.__timerange = (0.0 - self.__origintime,
                            0.0 - self.__origintime + self.__duration)

    def samplingtimes(self):
        """
        Returns the times of *sample centers*, relative to origin time.

        To get the sampling times relative to starttime add the starttime and
        origintime to the result of this method (e.g. signal.samplingtimes +
        signal.starttime + signal.origintime). But if you do this, note that
        the samplingtimes property is a float array of 64-bit precision. If
        starttime is a large number (e.g. Epoch) you may want to upcast to
        higher-precision floats first (96- or 128-bit float) if possible on
        your platform, at least if you want to retain precision.

        """
        return ((np.arange(self.__ntimesamples, dtype=np.float64) + 0.5)
                / self.__fs) - self.__origintime

    def samplingboundarytimes(self):
        """
        Returns the times of *sample boundaries*, relative to origin time.

        Note that the size of the return array is ntimesamples + 1

        """
        return ((np.arange(self.__ntimesamples + 1, dtype=np.float64))
                / self.__fs) - self.__origintime

    def time_to_index(self, time, nearest_boundary=False):
        """
        Converts a time, or array of times, to their corresponding sample
        indices. Reference time is `origintime`.

        Note that the default is to return to which *sample center* the given
        time is closest. When nearest_boundary is True, this will be to which
        *sample start* the time is closest.

        Parameter
        ---------

        time: <float, float sequence>
        nearest_boundary: bool
            Determines whether or not the the sample index is given of which
            the *center* is closest to the parameter time. If True, this will
            be the closest sample *start*.

        Example: if the time parameter is [t0, t1, t2],

        |    s0    |    s1    |    s2    |    s3    |    s4    | ...
                     ^       ^  ^
                     t0      t1 t2

        then the result will be the sample of indices [s1, s1, s2] if
        nearest_boundary is False (default) or [s1, s2, s2] if nearest_boundary
        is True.

        Note that it is possible to get an index number of ntimesamples 
        if the time is endtime.

        """
        ## for very high sampling numbers we might want to use a different
        ## approach (mpmath?), but for now this will do
        # check parameter
        time = check_argarray(time, 'time')
        nearest_boundary = check_arg(nearest_boundary, 'nearest_boundary')
        outsiderange = (time < self.__timerange[0]) | \
                       (time > self.__timerange[1])
        if outsiderange.any():
            offendingtimes = time[outsiderange]
            raise ValueError(
                "%s s. are outside range of possible times (%f,%f)" \
                % (offendingtimes, self.__timerange[0], self.__timerange[1]))
        # calculate
        if nearest_boundary:
            index = (np.floor((time + self.__origintime + self.dt / 2.0)
                             * self.__fs)).astype(np.int64)
        else:
            index = (np.floor((time + self.__origintime)
                              * self.__fs)).astype(np.int64)
        #index = ((time + self.__origintime) // self.__dt).astype(np.int64)
        #idx = (ta.view(np.ndarray) - self.t0) // self.sampling_interval
        if index.size == 1:
            return index.item()
        else:
            return index

    def index_to_time(self, index):
        """
        Converts a sample index, or array of indices, to their 
        corresponding times. Note that the *time of sample center* 
        is given, not the time of the sample start.

        Parameter
        ---------
        index: an integer or array of integers

        Note
        ----
        Center of samples define their time of occurrence

        |    0    |    1    |    2    |    3    |    4    | ...
             ^         ^         ^         ^         ^
             t0        t1        t2        t3        t4

        """
        index = check_intarray(index, cast=np.int64)
        if ((index < 0) | (index >= self.__ntimesamples)).any():
            raise ValueError("index (%s) cannot be smaller than zero or higher"
                             " than ntimesamples-1 (%d)" \
                             % (index, self.__ntimesamples - 1))
        result = ((index + 0.5) / self.__fs) - self.__origintime
        if result.size == 1:
            return result.item()
        else:
            return result

    def hastime(self, time):
        """
        Checks if a sample exists at a time, or array of times.

        Parameter
        ---------
        time: a float or array of floats


        """

        time = check_argarray(time, 'time')
        starttime, endtime = self.__timerange
        result = (time >= starttime) & (time <= endtime)
        if result.size == 1:
            return result.item()
        else:
            return result

    def movingtimeframeindices(self, framesize, stepsize=None,
                               include_remainder=True, startindex=None,
                               endindex=None):
        """
        Parameters
        ----------

        framesize: int
            Size of the frame in timesamples. Note that the last frame may be
            smaller than `framesize`, depending on the number of timesamples.
        stepsize: <int, None>
            Size of the shift in time per iteration in number of timesamples.
            Default is None, which means that `stepsize` equals `framesize`.
        include_remainder: <bool, True>
            Determines whether remainder (< framesize) should be included.
        startindex: <int, None>
            Start sample number.
            Default is None, which means to start at the beginning (sample 0)
        endindex: <int, None>
            End sample number.
            Default is None, which means to end at the end.

        Returns
        -------

        An iterator that yield tuples (start, end) representing the start and
        end indices of a time frame of size framesize that moves in stepsize
        steps. If include_remainder is True, it ends with a tuple representing
        the remainder, if present.

        """

        framesize = check_arg(framesize, 'framesize')
        if stepsize is None:
            stepsize = framesize
        stepsize = check_arg(stepsize, 'stepsize')
        if startindex is None:
            startindex = 0
        startindex = check_arg(startindex, 'startindex')
        if endindex is None:
            endindex = self.ntimesamples
        endindex = check_arg(endindex, 'startindex')

        if startindex > (self.ntimesamples - 1):
            raise ValueError("startindex too high")
        if endindex > self.ntimesamples:
            raise ValueError("endindex is too high")
        if startindex >= endindex:
            raise ValueError("startindex should be lower than endindex")

        nframes, newsize, remainder = fit_frames(
            totalsize=(endindex - startindex),
            framesize=framesize,
            stepsize=stepsize)
        framestart = startindex
        frameend = framestart + framesize
        for i in range(nframes):
            yield framestart, frameend
            framestart += stepsize
            frameend = framestart + framesize
        if include_remainder and (remainder > 0) and (
            framestart < self.ntimesamples):
            yield framestart, self.ntimesamples


class UniformTimeSeries(UniformTime):
    """
    Represents a single channel time sampled signal.

    Parameters
    ----------

    samples: 1-D sequence of numbers (list, tuple, numpy array, pytables array)
        Sample values.
    fs: float
        Sampling rate in Herz (number of samples per second).
    starttime: float, optional
        Absolute time at which the signal starts relative to some user-known
        standard (e.g. Epoch, or start of recording). Defaults to 0.0.
    origintime: float, optional
        Time of origin relative to beginning of the first sample. Defaults to
        0.0.
    dtype: numpy dtype, optional
        The samples provided by the `samples` parameter will be converted to
        `dtype`. The defaults is to keep the dtype of `samples`, or to use
        numpy defaults if `samples` is a list or tuple.
        See the `_supporteddtypes` attribute of the class to see which dtypes
        are allowed.
    copy: True or False, optional
        Determines whether or not the input array `samples` will be copied to a
        new memory space. If False, a change in the sample values of the
        `samples` will lead to changes of the samples of our object because both
        object share the same array in memory. `False` only works if `samples`
        is a numpy array and the `dtype` parameter is None. Default is True.

    Attributes (all read-only properties)
    -------------------------------------

    ntimesamples: int64
        Number of samples.
    fs: float64
        Sampling rate in Herz (number of samples per second).
    dt: float64
        Sampling period in seconds.
    duration: float64
        Duration in seconds.
    starttime: float64
        Absolute time at which the signal starts relative to some user-known
        standard (e.g. Epoch, or start of recording).
    endtime: float64
        Absolute time at which the signal ends relative to some user-known
        standard (e.g. Epoch, or start of recording).
    origintime: float64
        Time of origin relative to beginning of the first sample. Defaults to
        0.0.
    timerange: (float64, float64)
        Start and end times, *relative to origintime*
    dtype: numpy dtype
        The dtype of the samples
    samples: instance of Samples object (or subclass)
        Provides direct access to the samples, using indexing.


    Methods (see method doc for explanation)
    ----------------------------------------

    time_to_index
    index_to_time
    samplingtimes
    decimate
    copy


    Note
    ----

    Example that explains definitions of time (all in seconds):

                 |  s0  |  s1  |  s2  |  s3  |  s4  |  s5  |  s6  |
                 ^                                                ^
    absolute: starttime                                        endtime
            (e.g. 1452.3)                              (e.g. 1452.3 + duration)

                                 ^
    relative:                origintime
                          (defined as 0.0s)

    The methods `index_to_time` and `time_to_index`, and the attribute
    `timerange` all use relative time. `origintime` does not have to coincide
    with sample boundaries or centers.

    """
    # _supporteddtypes = (np.int8, np.int16, np.int32, np.int64,
    # np.uint8, np.uint16, np.uint32, np.uint64,
    #                     np.float16, np.float32, np.float64, np.complex128)

    def __init__(self, samples, fs, starttime=0.0, origintime=0.0, dtype=None,
                 copy=True, readonly=True, attrs=None):

        copy = check_arg(copy, 'copy')
        self.__samples = self._check_samples(samples, dtype=dtype, copy=copy,
                                             readonly=readonly)
        self.__sampletype = self.__samples.sampletype
        self.__dtype = self.__samples.dtype
        ntimesamples = self.__samples.shape[self.__samples.axes['time']]
        if attrs is None:
            self.attrs = {}
        else:
            self.attrs = attrs
        UniformTime.__init__(self, ntimesamples=ntimesamples, fs=fs,
                             starttime=starttime, origintime=origintime)

    samples = s = property(lambda self: self.__samples, None, None,
                       "Samples object")

    dtype = property(lambda self: self.__samples.dtype, None, None,
                     "")
    axes = property(lambda self: self.__samples.axes, None, None,
                    "")
    readonly = property(lambda self: self.__samples.readonly, None, None,
                        "")
    sampletype = property(lambda self: self.__sampletype, None, None,
                          "type of samples object")
    baseaxis = property(lambda self: 'time', None, None,
                        "this tells functions the most likely axis along which "
                        "something should be calculated")
    mb = property(lambda self: self.__samples.mb, None, None,
                  "size in Mb")

    _instantiationparams = property(
        lambda self: ({'samples': self.samples,
                       'fs': self.fs,
                       'starttime': self.starttime,
                       'origintime': self.origintime,
                       'dtype': self.dtype,
                       'copy': True,
                       'readonly': self.readonly,
                       'attrs': self.attrs}),
        None, None, "parameter dictionary for instantiation")

    ######################
    ## Standard methods ##
    ######################

    def __str__(self):
        return "<UniformTimeSeries: %.3f seconds, %d times>" \
               % (self.duration, self.ntimesamples)

    __repr__ = __str__

    def _get_index(self, index):
        sslice, starttime, origintime = self._to_timeslice(
            timeindex=index)
        return sslice

    def __getitem__(self, index):

        sslice, starttime, origintime = self._to_timeslice(
            timeindex=index)

        return UniformTimeSeries(samples=self.samples[sslice], fs=self.fs,
                                 starttime=starttime, copy=False,
                                 origintime=origintime,
                                 attrs=self.attrs)

    def __eq__(self, other):
        result = UniformTime.__eq__(self, other)
        result &= (self.__dtype == other.dtype)
        result &= (self.baseaxis == other.baseaxis)
        if result:  # we check samples
            framesize = int(np.ceil(self.mb / 64.0))  # 64 Mb chunks
            axisorder = self.samples.axisorder
            startindex = 0
            for frame in self.iter_frames(framesize=framesize,
                                          include_remainder=True):
                result &= \
                    np.all(other[
                           startindex:startindex + frame.ntimesamples].samples.transpose(
                        axisorder).get(
                        copy=False) == \
                           frame.samples.get(copy=False))
                startindex += framesize
        return result

    def __del__(self):
        if hasattr(self, 'close'):
            # print "closing uts signal"
            self.close()

    #############
    ## Methods ##
    #############

    def _instantiatefromself(self, **params):
        args = self._instantiationparams
        for key, item in params.items():
            args[key] = item
        return self.__class__(**args)

    def _check_samples(self, samples, dtype=None, copy=True, readonly=True):
        return assamples(samples, axes={'time': 0}, dtype=dtype, copy=copy,
                         readonly=readonly)

    def _to_timeslice(self, timeindex):
        """
        There is only one slicing dimension. Samples can only be indexed by an
        int or slice, with the latter having the restriction that step can
        only be 1 and start may not be larger than stop.

        Returns a slice, start time and orgin
        time.
        """
        ## check that the index is proper
        if isinstance(timeindex, defined_integers):
            start, stop = timeindex, timeindex + type(timeindex)(1)
            timeindex = slice(start, stop)
        elif isinstance(timeindex, slice):
            start = timeindex.start
            if start is None:
                start = 0
            # if index is not int, it should indicate a time; convert

            if not isinstance(start, defined_integers):
                start = self.time_to_index(start)
            stop = timeindex.stop
            if stop is None:
                stop = self.ntimesamples
            # if index is not int, it should indicate a time; convert
            if not isinstance(stop, defined_integers):
                stop = self.time_to_index(stop)
            if not timeindex.step is None:
                raise IndexError("sample dimension cannot have a step index")
        else:
            raise IndexError(
                "index of sample dimension can only be an int or slice")
            # numpy allows the following, but we don't for now
        # this test must be done here, before the slice indices method below
        if stop > self.ntimesamples:
            raise IndexError(
                "sample index (%d) is higher than ntimesamples (%d)" % (
                stop, self.ntimesamples))
        start, stop, step = slice(start, stop).indices(self.ntimesamples)
        samplesslice = slice(start, stop)
        # if start is not 0, starttime and origintime are not correct anymore
        timeshift = start / self.fs
        starttime = self.starttime + timeshift
        origintime = self.origintime - timeshift

        return samplesslice, starttime, origintime

    def copy(self, starttime=None, origintime=None):
        """
        Makes a copy of the UniformTimeSeries, with the option of
        resetting starttime and origintime.

        """
        if starttime is None: starttime = self.starttime
        if origintime is None: origintime = self.origintime
        starttime = check_arg(starttime, 'time')
        origintime = check_arg(origintime, 'time')
        return self._instantiatefromself(starttime=starttime,
                                         origintime=origintime, copy=True,
                                         attrs=self.attrs)

    def astype(self, dtype):
        if not self.dtype.type is dtype:
            samples = self.samples.get(copy=False).astype(dtype)
            return self._instantiatefromself(samples=samples, dtype=dtype)
        else:
            return self

    def to_multichannel(self, channelname=None):
        samples = self.samples.get(copy=False).reshape(-1, 1)
        fs = self.fs
        if channelname is None:
            channelnames = None
        else:
            channelnames = [channelname]
        starttime = self.starttime
        origintime = self.origintime
        channelaxis = 1
        return MultiChannelUniformTimeSeries(samples=samples, fs=fs,
                                             channels=channelnames,
                                             starttime=starttime,
                                             origintime=origintime,
                                             channelaxis=1, copy=True,
                                             attrs=self.attrs)

    def iter_frames(self, framesize, stepsize=None, include_remainder=True,
                    startindex=None, endindex=None):
        """
        Iterate over time axis.

        Parameters
        ----------

        framesize: int
            Size of the frame in timesamples. Note that the last frame may be
            smaller than `framesize`, depending on the number of timesamples.
        stepsize: <int, None>
            Size of the shift in time per iteration in number of timesamples. 
            Default is None, which means that `stepsize` equals `framesize`.
        startindex: <int, None>
            Start sample number.
            Default is None, which means to start at the beginning (sample 0)
        endindex: <int, None>
            End sample number.
            Default is None, which means to end at the end.

        Returns
        -------
        A generator that yields time frames of self.

        """

        for framestart, frameend in self.movingtimeframeindices(
                framesize=framesize,
                stepsize=stepsize,
                include_remainder=include_remainder,
                startindex=startindex,
                endindex=endindex):
            yield self[framestart:frameend]

    def show(self, zoom=False, labels={}, indexrange=None):

        if zoom:
            from old_code.sampled_zoom_plot import SampledZoomPlot

            return SampledZoomPlot(self.samplingtimes(), self.samples.get(),
                labels=labels)
        else:
            from old_code.singlechannel import PlotView

            p = PlotView(self.get_plot(labels=labels, indexrange=indexrange))
            p.configure_traits()
            return p

    def plot(self, ax=None, **kwargs):
        from ..viz.plots import plot_uts
        return plot_uts(self, ax=ax, **kwargs)

    def append(self, signal):
        """
        note that append can be very inefficient, depending on the type of
        sample source. h5ArraySamples can be appended efficiently, but
        NumpyArraySamples not, as each append involves copying the whole signal.

        """
        if not isinstance(signal, UniformTimeSeries):
            raise TypeError("signal (%s) should be a UniformTimeSeries" % type(
                signal).__name__)
        if not signal.fs == self.fs:
            raise TypeError(
                "cannot append a signal with different sampling rate")
        if not signal.samples.ndim == 1:
            raise TypeError("signal does not have a dimension of 1")
        self.samples.append(samples=signal.samples.get(copy=False),
                            axis=self.samples.axes['time'])
        ntimesamples = self.samples.shape[self.samples.axes['time']]
        self._update_ntimesamples(ntimesamples)

    # def get_epochs(self, starttimes, duration):
    #     """
    #     get_epochs(self, starttimes, duration)
    #     """
    #     startsamples = self.time_to_index(starttimes)
    #     duration = check_arg(duration, 'duration')
    #     nsamples = int(np.round(duration * self.fs))
    #     ar = np.empty((len(startsamples), nsamples), dtype=self.samples.dtype)
    #     for i, sample in enumerate(startsamples):
    #         ar[i] = self.samples[sample:sample + nsamples]
    #     epochs = np.array(list(starttimes), dtype=[('starttime', np.float64)])
    #     return EpochedUniformTimeSeries(samples=ar, fs=self.fs, epochs=epochs,
    #                                     epochaxis=0,
    #                                     attrs=self.attrs)



    def get_epochs(self, duration, starttimes=None, epochs=None,
                   origintime=0.0):
        """
        get_epochs(self, starttimes, duration)
        """
        duration = check_arg(duration, 'duration')
        if (starttimes is None) and (epochs is None):
            raise ValueError(
                "Either 'starttimes' or 'epochs' should not be None")
        elif (not starttimes is None) and (not epochs is None):
            warnings.warn(
                "Both 'starttimes' and 'epochs' are provided; using only 'epochs'")
        elif epochs is None:  # starttimes have been provided
            epochs = np.array(list(starttimes),
                              dtype=[('starttime', np.float64)])
        startsamples = self.time_to_index(epochs['starttime'])
        nsamples = int(np.round(duration * self.fs))
        ar = np.empty((len(startsamples), nsamples),
                      dtype=self.samples.dtype)
        for i, sample in enumerate(startsamples):
            ar[i] = self[sample:sample + nsamples].samples.get(True)

        return EpochedUniformTimeSeries(samples=ar, fs=self.fs,
                                        epochs=epochs,
                                        epochaxis=0,
                                        origintime=origintime)


class EpochedUniformTimeSeries(Epoched, UniformTimeSeries):
    def __init__(self, samples, fs, epochs, origintime=0.0, dtype=None,
                 copy=True, epochaxis=0, readonly=True, attrs=None):
        """
        samples: a (events x samples) array
        epochs: a numpy structured array or h5 table, with at least a 'starttime' column

        """
        self.__epochaxis, self.__timeaxis = self._check_epochaxis(epochaxis)
        UniformTimeSeries.__init__(self, samples=samples, fs=fs, starttime=0.0,
                                   origintime=origintime, dtype=dtype,
                                   copy=copy,
                                   readonly=readonly, attrs=attrs)
        duration = self.samples.shape[self.samples.axes['time']] / self.fs
        nepochs = self.samples.shape[self.samples.axes['epoch']]
        Epoched.__init__(self, duration=duration, epochs=epochs,
                         origintime=origintime)

    timeaxis = property(
        lambda self: (self.__timeaxis), None, None,
        "the axis of the samples array that corresponds to samples")
    epochaxis = property(
        lambda self: (self.__epochaxis), None, None,
        "the axis of the samples array that corresponds to epochs")

    _instantiationparams = property(
        lambda self: ({'samples': self.samples,
                       'fs': self.fs,
                       'origintime': self.origintime,
                       'dtype': self.dtype,
                       'copy': True,
                       'epochaxis': self.samples.axes['epoch'],
                       'epochs': self.epochs,
                       'readonly': self.readonly,
                       'attrs': self.attrs}),
        None, None, "parameter dictionary for instantiation")

    def __eq__(self, other):
        result = UniformTimeSeries.__eq__(self, other)
        result &= (self.nepochs == other.nepochs)
        result &= np.all(self.epochs == other.epochs)
        return result


    def _get_index(self, index):

        index = np.index_exp[index]
        if not len(index) <= 2:
            raise IndexError(
                "index can maximally have two dimensions [sampledim, epochdim]")

        if len(index) == 2:
            epochindex = self._to_epochsindex(index[1])
        else:
            epochindex = slice(None, None, None)

        # the first part refers to samples
        samplesindex, starttime, origintime = \
            super(self.__class__, self)._to_timeslice(index[0])

        if self.__epochaxis == 0:
            return (epochindex, samplesindex)
        else:
            return (samplesindex, epochindex)


    def __getitem__(self, index):

        index = np.index_exp[index]
        samplesindex, starttime, origintime = \
             super(self.__class__, self)._to_timeslice(index[0])

        if len(index) == 2:
            epochindex = self._to_epochsindex(index[1])
        else:
            epochindex = slice(None, None, None)

        samples = self.samples[self._get_index(index)]
        if isinstance(epochindex, defined_integers):
            return UniformTimeSeries(samples=samples,
                                     fs=self.fs,
                                     starttime=self.starttimes[epochindex],
                                     origintime=origintime, copy=False,
                                     attrs=self.attrs)
        else:
            epochs = self.epochs[epochindex]
            return self._instantiatefromself(samples=samples, fs=self.fs,
                                             origintime=origintime,
                                             epochs=epochs, copy=True)

    def __str__(self):
        return "<EpochedUniformTimeSeries: %.3f seconds, %d times, %d epochs>" \
               % (self.duration, self.ntimesamples, self.nepochs)

    __repr__ = __str__

    ###########################
    ## methods for internal use
    ###########################

    def _check_samples(self, samples, dtype=None, copy=True, readonly=True):
        return assamples(samples, axes={'time': self.__timeaxis,
                                        'epoch': self.__epochaxis},
                         dtype=dtype, copy=copy, readonly=readonly)

    def _check_epochaxis(self, epochaxis):
        epochaxis = check_arg(epochaxis, 'epochaxis')
        if not epochaxis in (0, 1):
            raise IndexError("epochaxis should be 0 or 1")
        if epochaxis == 0:
            timeaxis = 1
        else:
            timeaxis = 0
        return epochaxis, timeaxis

    def _collapse_axis(self, samples, axis='epoch'):
        if not axis == 'epoch':
            raise ValueError(
                "%s is not a valid axis, should be 'epoch'" % axis)
        return self[slice(None, None, None), 0]._instantiatefromself(
            samples=samples)

    def copy(self):
        return self._instantiatefromself(copy=True)

    def iter_epochs(self):
        for i in range(self.nepochs):
            yield self[slice(None, None, None), i]

    def show(self):
        p = self.get_plot()
        p.configure_traits()
        return p

    def append(self, euts):
        if not isinstance(euts, EpochedUniformTimeSeries):
            raise TypeError("euts (%s) should be a EpochedUniformTimeSeries" \
                            % type(euts).__name__)
        if not euts.fs == self.fs:
            raise TypeError("cannot append a euts with different sampling "
                            "rate")
        if not euts.ntimesamples == self.ntimesamples:
            raise TypeError("euts does not have the same number of time "
                            "samples")
        samples = euts.samples.transpose(self.samples.axisorder).get(
            copy=False)
        self.samples.append(samples, axis=self.epochaxis)
        self._append_epochs(euts.epochs)

    def transpose(self, axes=None):
        if axes is None:
            axes = self.samples.axisorder[::-1]
        samples = self.samples.transpose(axes)
        return self._instantiatefromself(samples=samples,
                                         epochaxis=samples.axes['epoch'])


class MultiChannelUniformTimeSeries(MultiChannel, UniformTimeSeries):
    """
    Represents a multi-channel time sampled signal.

    Parameters
    ----------
    samples: 2-D sequence of numbers (list, tuple, numpy array, pytables array)
        Sample values.
    fs: float
        Sampling rate in Herz (number of samples per second).
    channels: sequence, optional
        A sequence (list, tuple, numpy array) with the names of individual
        channels. If None, default names will be created (i.e. this object 
        must always have named channels.)
    starttime: float, optional
        Absolute time at which the signal starts relative to some user-known
        standard (e.g. Epoch, or start of recording). Defaults to 0.0.
    origintime: float, optional
        Time of origin relative to beginning of the first sample. Defaults to
        0.0.
    dtype: numpy dtype, optional
        The samples provided by the `samples` parameter will be converted to
        `dtype`. The defaults is to keep the dtype of `samples`, or to use
        numpy defaults if `samples` is a list or tuple.
        See the `_supporteddtypes` attribute of the class to see which dtypes
        are allowed.
    copy: True or False, optional
        Determines whether or not the input array `samples` will be copied to a
        new memory space. If False, a change in the sample values of the
        `samples` will lead to changes of the samples of our object because both
        object share the same array in memory. `False` only works if `samples`
        is a numpy array and the `dtype` parameter is None. Default is True.
    channelaxis: int
        default 1
    geometry:
        default None

    Attributes
    ----------
    samples: instance of Samples object (or subclass)
          Provides direct access to the samples, using indexing.

      - samples: acces to the actual samples object
      - fs: sampling rate
      - dt: sampling period
      - starttime: start of the first sample re epoch (1970-01-01, 12:00:00 AM)
        in seconds, may be None.
      - endtime: end of the last sample re epoch (1970-01-01, 12:00:00 AM)
        in seconds, may be none.
      - duration: duration of the sound in seconds.
      - nchannels: number of channels
      - ntimesamples: number of samples per channel

    Methods
    -------
      - time_to_index(time); short form is tti(time)
      - index_to_time(index); short form is itt(index)
      - samplingtimes()

    Slicing
    -------

    MultiChannelUniformTimeSeriess can be sliced the following way:

      - slicing (e.g. sound[100:500,1]) works in principle like np array
        slicing
      - indices correspond to channel numbers and sample numbers, counting
        from zero
      - e.g.: sound[:,1] returns a sound object with only channel 1 (second channel)
              sound[:,[0,2,5]] returns a sound with channels 0,2, and 5
              sound[100:500,[0,1]] returns channels 0 and 1, samples 100 to 499
      - slicing returns a new sound object, with the samples copied (not a
        view), and timing information adjusted.

    """

    def __init__(self, samples, fs, channels=None, starttime=0.0,
                 origintime=0.0, dtype=None, copy=True, channelaxis=1,
                 geometry=None, readonly=True, attrs=None):
        self.__channelaxis, self.__timeaxis = self._check_channelaxis(
            channelaxis)
        UniformTimeSeries.__init__(self, samples=samples, fs=fs,
                                   starttime=starttime, origintime=origintime,
                                   dtype=dtype, copy=copy, readonly=readonly,
                                   attrs=attrs)
        nchannels = self.samples.shape[self.__channelaxis]
        MultiChannel.__init__(self, nchannels=nchannels,
                              channels=channels, geometry=geometry)

    channelaxis = property(
        lambda self: (self.__channelaxis), None, None,
        "the axis of the samples array that corresponds to channels")
    timeaxis = property(
        lambda self: (self.__timeaxis), None, None,
        "the axis of the samples array that corresponds to time")

    _instantiationparams = property(
        lambda self: ({'samples': self.samples,
                       'fs': self.fs,
                       'channels': self.channels,
                       'starttime': self.starttime,
                       'origintime': self.origintime,
                       'dtype': self.dtype,
                       'copy': True,
                       'channelaxis': self.samples.axes['channel'],
                       'geometry': self.geometry,
                       'readonly': self.readonly,
                       'attrs': self.attrs}),
        None, None, "parameter dictionary for instantiation")

    ######################
    ## Standard methods ##
    ######################

    def __str__(self):
        return "<MultiChannelUniformTimeSeries: %.3f seconds, %d times, %d channels>" \
               % (self.duration, self.ntimesamples, self.nchannels)

    __repr__ = __str__

    def info(self):
        s = UniformTimeSeries.info(self)
        s += MultiChannel.info(self)
        return s

    def __eq__(self, other):
        result = UniformTimeSeries.__eq__(self, other)
        result &= np.all(self.nchannels == other.nchannels)
        result &= np.all(self.channelnames == other.channelnames)
        result &= (self.geometry == other.geometry)
        return result


    def _get_index(self, index):
        index = np.index_exp[index]
        if not len(index) <= 2:
            raise IndexError(
                "index can maximally have two dimensions [sampledim,channeldim]")
        # if there are two parts to index, the second one refers to channels
        if len(index) == 2:
            channelindex = self._to_channelsslice(index[1])
        else:
            channelindex = slice(None, None, None)
        channels = self.channels[channelindex]
        # the first part refers to samples
        timeindex, starttime, origintime = \
            super(self.__class__, self)._to_timeslice(index[0])
        return tuple([{'channel': channelindex, 'time': timeindex}[axis]
            for axis in self.samples.axisorder])


    def __getitem__(self, index):
        """
        There are maximally two slicing dimensions, the first one for samples
        and the second one for channels. This is always true, irrespective of
        the channelaxis parameter. Channels can be indexed in any way that
        np allows, but samples can only be indexed by an int or slice, with
        the latter having the restriction that step is not allowed and start may
        not be larger than stop.

        Returns a new MultiChannelUniformTimeSeries object. Sampling rate,
        starttime and orgintime are recalculated if necessary.
        """
        fullindex =  self._get_index(index)
        axisslices = dict(zip(self.samples.axisorder ,fullindex))
        samplesindex, starttime, origintime = \
             super(self.__class__, self)._to_timeslice(axisslices['time'])
        channelnames = np.array(self.channelnames[axisslices['channel']],ndmin=1)
        nselchannels = len(channelnames)
        samples = self.samples[fullindex]
        #FIXME can it be optimized to test for int type only?
        if nselchannels == self.nchannels:
            geometry = self.geometry
        else:
            geometry = None
        if nselchannels == 1:  # one channel only
            return UniformTimeSeries(samples=samples.flatten(), fs=self.fs,
                                     copy=False, starttime=starttime,
                                     origintime=origintime,
                                     attrs=self.attrs)
        else:
            return MultiChannelUniformTimeSeries(samples=samples,
                                                 channels=channelnames,
                                                 fs=self.fs,
                                                 starttime=starttime,
                                                 copy=False,
                                                 origintime=origintime,
                                                 channelaxis=self.__channelaxis,
                                                 geometry=geometry,
                                                 attrs=self.attrs)



    ###########################
    ## methods for internal use
    ###########################

    def _check_samples(self, samples, dtype=None, copy=True, readonly=True):
        return assamples(samples, axes={'time': self.__timeaxis,
                                        'channel': self.__channelaxis},
                         dtype=dtype, copy=copy, readonly=readonly)

    def _check_channelaxis(self, channelaxis):
        channelaxis = check_arg(channelaxis, 'channelaxis')
        if not channelaxis in (0, 1):
            raise IndexError("channelaxis should be 0 or 1")
        if channelaxis == 0:
            timeaxis = 1
        else:
            timeaxis = 0
        return channelaxis, timeaxis

    def _collapse_axis(self, samples, axis='channel'):
        if not axis == 'channel':
            raise ValueError(
                "%s is not a valid axis, should be 'channel'" % axis)
        return self[slice(None, None, None), 0]._instantiatefromself(
            samples=samples)


    ##########################
    ## user accessible methods
    ##########################

    def copy(self, starttime=None, origintime=None, channels=None):
        """
        Makes a copy of the UniformTimeSeries, with the option of
        resetting starttime, origintime, and channelnames.

        """
        if starttime is None: starttime = self.starttime
        if origintime is None: origintime = self.origintime
        if channels is None: channels = self.channels.copy()
        starttime = check_arg(starttime, 'time')
        origintime = check_arg(origintime, 'time')
        return self._instantiatefromself(starttime=starttime,
                                         origintime=origintime,
                                         channels=channels,
                                         copy=True)

    def dot(self, m, channels=None):
        """
        Matrix multiplication m * signal. The number of new channels equals the
        size of the first axis of m. Hence the size of the second axis of m 
        should equal the number of channels of the signal.
        """

        if self.samples.axes['channel'] == 0:
            samples = np.dot(m, self.samples.get(copy=False))
        else:
            samples = np.dot(self.samples.get(copy=False), m.T)

        nchannels = m.shape[0]
        if channels is None:
            s = 'dotprod%%.%dd' % max(1, int(np.ceil(np.log10(nchannels))))
            channels = [s % i for i in range(nchannels)]
        if samples.dtype == self.samples.dtype:
            copy = False
        else:
            copy = True
        return self._instantiatefromself(samples=samples,
                                         channels=channels,
                                         copy=copy, dtype=samples.dtype,
                                         geometry=None)

    def iter_frames(self, framesize, stepsize=None, include_remainder=True,
                    startindex=None, endindex=None, channels=None):
        """
        Iterate over time axis.

        Parameters
        ----------

        framesize: int
            Size of the frame in timesamples. Note that the last frame may be
            smaller than `framesize`, depending on the number of timesamples.
        stepsize: <int, None>
            Size of the shift in time per iteration in number of timesamples. 
            Default is None, which means that `stepsize` equals `framesize`.
        startindex: <int, None>
            Start sample number.
            Default is None, which means to start at the beginning (sample 0)
        endindex: <int, None>
            End sample number.
            Default is None, which means to end at the end.

        Returns
        -------
        A generator that yields time frames of self.

        """

        if channels is None:
            channels = range(self.nchannels)
        for framestart, frameend in self.movingtimeframeindices(
                framesize=framesize,
                stepsize=stepsize,
                include_remainder=include_remainder,
                startindex=startindex,
                endindex=endindex):
            yield self[framestart:frameend, channels]

    def iter_channels(self):
        for i in range(self.nchannels):
            yield self[:, i]

    def to_epoched(self, nepochs, epochs=None, starttime=None):
        """
        Divides signal in nepochs maximally and equally long epochs.

        The last samples will be truncated if they don't make up a full epoch.

        See get_epochs() method to get arbitrary epochs.

        """
        nepochs = check_int(nepochs, positive=True, zero=False, negative=False)
        samples = self.samples.get(copy=False)
        if self.timeaxis == 0:
            samples = samples.T
        channelaxis = 0
        nepochsamples = self.ntimesamples // nepochs
        samples = samples.reshape(self.nchannels, nepochsamples, nepochs)
        epochaxis = 2
        fs = self.fs
        channelnames = self.channelnames
        origintime = self.origintime
        geometry = self.geometry
        if not starttime:
            starttime = self.starttime
        if not epochs:
            epochs = np.empty(nepochs, dtype=([('starttime', np.float64)]))
            epochs['starttime'] = starttime \
                                  + (
            np.arange(nepochs, dtype=np.float64) * (nepochsamples / self.fs))
        return EpochedMultiChannelUniformTimeSeries(samples=samples, fs=fs,
                                                    epochs=epochs,
                                                    channels=channelnames,
                                                    origintime=origintime,
                                                    geometry=geometry,
                                                    dtype=None, copy=False,
                                                    channelaxis=channelaxis,
                                                    epochaxis=epochaxis)

    def show(self, plottype='lineplot', labels=None, include_imageplot=False,
             clip=None, **plotparams):
        from uts.display.traitsmultichannel import \
            MultiChannelUniformTimeSeriesPlot as Plot
        from uts.display.traitsmultichannel import \
            MultiChannelImagePlot as ImagePlot

        p1 = Plot(self, plottype=plottype, geometry=self.geometry,
                  labels=labels, **plotparams)
        p1.configure_traits()
        if include_imageplot:
            p2 = ImagePlot(self, clip=clip, mcutsp=p1,
                           colormap='jet', inversecolor=False,
                           time=self.timerange[0], **plotparams)
            p2.configure_traits()
        else:
            p2 = None
        return p1, p2

    def plot(self, ga=None, **kwargs):
        from ..viz.plots import plot_mcuts
        return plot_mcuts(self, ga=ga, **kwargs)

    def append(self, signal):
        if not isinstance(signal, MultiChannelUniformTimeSeries):
            raise TypeError(
                "signal (%s) should be a MultiChannelUniformTimeSeries" % type(
                    signal).__name__)
        if not signal.fs == self.fs:
            raise TypeError(
                "cannot append a signal with different sampling rate")
        if not signal.nchannels == self.nchannels:
            raise TypeError("signals do not have the same number of channels")
        samples = signal.samples.get(copy=False)
        if signal.channelaxis != self.channelaxis:
            samples = samples.transpose()
        self.samples.append(samples, axis=self.timeaxis)
        ntimesamples = self.samples.shape[self.__timeaxis]
        self._update_ntimesamples(ntimesamples)

    def transpose(self, axes=None):
        if axes is None:
            axes = self.samples.axisorder[::-1]
        samples = self.samples.transpose(axes)
        return self._instantiatefromself(samples=samples,
                                         channelaxis=samples.axes['channel'])

    def get_epochs(self, duration, starttimes=None, epochs=None,
                   origintime=0.0):
        """
        get_epochs(self, starttimes, duration)
        """
        duration = check_arg(duration, 'duration')
        if (starttimes is None) and (epochs is None):
            raise ValueError(
                "Either 'starttimes' or 'epochs' should not be None")
        elif (not starttimes is None) and (not epochs is None):
            warnings.warn(
                "Both 'starttimes' and 'epochs' are provided; using only 'epochs'")
        elif epochs is None:  # starttimes have been provided
            epochs = np.array(list(starttimes),
                              dtype=[('starttime', np.float64)])
        startsamples = self.time_to_index(epochs['starttime'])
        nsamples = int(np.round(duration * self.fs))
        ar = np.empty((len(startsamples), self.nchannels, nsamples),
                      dtype=self.samples.dtype)
        for i, sample in enumerate(startsamples):
            ar[i] = self[sample:sample + nsamples].transpose(
                ['channel', 'time']).samples.get(True)

        return EpochedMultiChannelUniformTimeSeries(samples=ar, fs=self.fs,
                                                    epochs=epochs,
                                                    epochaxis=0, channelaxis=1,
                                                    channels=self.channels,
                                                    geometry=self.geometry,
                                                    origintime=origintime)


class EpochedMultiChannelUniformTimeSeries(Epoched,
                                           MultiChannelUniformTimeSeries):
    def __init__(self, samples, fs, epochs, channels=None,
                 origintime=0.0, dtype=None, copy=True, geometry=None,
                 channelaxis=2, epochaxis=0, readonly=True, attrs=None):

        '''
        samples: a (events x samples x channels) array
        epochs: a numpy structured array or h5 table, with at least a 'starttime' column

        '''
        self.__timeaxis, self.__channelaxis, self.__epochaxis = \
            self._check_axes(channelaxis, epochaxis)

        MultiChannelUniformTimeSeries.__init__(self, samples=samples,
                                               fs=fs,
                                               channelaxis=self.__channelaxis,
                                               channels=channels,
                                               origintime=origintime,
                                               starttime=0.0, dtype=dtype,
                                               copy=copy, geometry=geometry,
                                               readonly=readonly, attrs=attrs)
        duration = self.samples.shape[self.samples.axes['time']] / self.fs
        nepochs = self.samples.shape[self.samples.axes['epoch']]

        Epoched.__init__(self, duration=duration, epochs=epochs,
                         origintime=origintime)

    epochaxis = property(
        lambda self: (self.__epochaxis), None, None,
        "the axis of the samples array that corresponds to epochs")

    _instantiationparams = property(
        lambda self: ({'samples': self.samples,
                       'fs': self.fs,
                       'channels': self.channels,
                       'origintime': self.origintime,
                       'epochs': self.epochs,
                       'dtype': self.dtype,
                       'copy': True,
                       'geometry': self.geometry,
                       'channelaxis': self.samples.axes['channel'],
                       'epochaxis': self.samples.axes['epoch'],
                       'readonly': self.readonly,
                       'attrs': self.attrs}),
        None, None, "parameter dictionary for instantiation")

    def _check_samples(self, samples, dtype=None, copy=True, readonly=True):
        return assamples(samples, axes={'time': self.__timeaxis,
                                        'epoch': self.__epochaxis,
                                        'channel': self.__channelaxis},
                         dtype=dtype, copy=copy,
                         readonly=readonly)

    def _check_axes(self, channelaxis, epochaxis):
        channelaxis = check_arg(channelaxis, 'channelaxis')
        epochaxis = check_arg(epochaxis, 'epochaxis')
        if not channelaxis in (0, 1, 2):
            raise IndexError("channelaxis should be 0, 1, 2")
        if not epochaxis in (0, 1, 2):
            raise IndexError("epochaxis should be 0, 1, 2")
        if channelaxis == epochaxis:
            raise IndexError("epochs and channels cannot be on the same axis")
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
        return timeaxis, channelaxis, epochaxis

    def _check_channelaxis(self, channelaxis):
        # overrides method from parent; we have _checkaxes
        return self.__channelaxis, self.__timeaxis

    def __eq__(self, other):
        result = MultiChannelUniformTimeSeries.__eq__(self, other)
        result &= np.all(self.epochs == other.epochs)
        # result &= (self.duration == other.duration)
        return result

    def _get_index(self, index):
        index = np.index_exp[index]
        # index is now a tuple of max 3 with possibly slices, ints, lists, tuples arrays
        if not len(index) <= 3:
            raise IndexError(
                "index can maximally have three dimensions [time,channel,epoch]")
        # create indices
        if len(index) == 3:
            timeindex, starttime, origintime = self._to_timeslice(
                index[0])
            channelindex = self._to_channelsslice(index[1])
            epochindex = self._to_epochsindex(index[2])
        elif len(index) == 2:
            timeindex, starttime, origintime = self._to_timeslice(
                index[0])
            channelindex = self._to_channelsslice(index[1])
            epochindex = slice(None, None, None)
        else:
            timeindex, starttime, origintime = self._to_timeslice(
                index[0])
            channelindex = slice(None, None, None)
            epochindex = slice(None, None, None)
        return tuple([{'channel': channelindex, 'time': timeindex, 'epoch': epochindex}[axis]
            for axis in self.samples.axisorder])


    def __getitem__(self, index):
        """
        This object can be indexed in three dimensions, in the order: time, channel, epoch.
        """

        ## FIXME look if this can be optimized; indexing now works per axis at the time
        ## which ones should be first? Can it be done at once?
        fullindex =  self._get_index(index)
        axisslices = dict(zip(self.samples.axisorder, fullindex))
        samplesindex, starttime, origintime = \
             super(self.__class__, self)._to_timeslice(axisslices['time'])
        # channels = self.channelnames[axisslices['channel']]
        epochs = np.array(self.epochs[axisslices['epoch']], ndmin=1)
        samples = np.squeeze(self.samples[fullindex])
        # print samples, samples.shape, self.samples.axisorder, fullindex
        # print axisslices
        # print epochs
        # print axisslices['epoch']
        channelnames = np.array(self.channelnames[axisslices['channel']], ndmin=1)
        # print channelnames
        nselchannels = len(channelnames)
        # print nselchannels
        # print len(epochs)
        if nselchannels == self.nchannels:
            geometry = self.geometry
        else:
            geometry = None

        if (nselchannels==1) and (len(epochs)==1):
            if starttime is None: starttime = 0.0
            res = UniformTimeSeries(samples=samples, fs=self.fs,
                                    starttime=starttime, origintime=origintime,
                                    copy=False,
                                    attrs=self.attrs)
        elif len(epochs)==1:
            if self.channelaxis > self.timeaxis:
                channelaxis = 1
            else:
                channelaxis = 0
            res = MultiChannelUniformTimeSeries(samples=samples, fs=self.fs,
                                                starttime=starttime,
                                                origintime=origintime,
                                                channels=channelnames,
                                                copy=False, geometry=geometry,
                                                channelaxis=channelaxis,
                                                attrs=self.attrs)
        elif nselchannels==1:
            if self.epochaxis > self.timeaxis:
                epochaxis = 1
            else:
                epochaxis = 0
            res = EpochedUniformTimeSeries(samples=samples, fs=self.fs,
                                           epochs=epochs,
                                           origintime=origintime, copy=False,
                                           epochaxis=epochaxis,
                                           attrs=self.attrs)
        else:  # remaining_axes == ['time','channel,'epoch']
            res = EpochedMultiChannelUniformTimeSeries(samples=samples,
                                                       fs=self.fs,
                                                       channels=channelnames,
                                                       origintime=origintime,
                                                       dtype=None, copy=False,
                                                       epochs=epochs,
                                                       geometry=geometry,
                                                       epochaxis=self.epochaxis,
                                                       channelaxis=self.channelaxis,
                                                       attrs=self.attrs)
        return res

    def __iter__(self):
        for i in range(self.nepochs):
            yield self[:, :, i]

    iter_epochs = __iter__

    def __str__(self):
        return "<EpochedMultiChannelUniformTimeSeries: " \
               "%.3f seconds, %d times, %d channels, %d epochs>" \
               % (
        self.duration, self.ntimesamples, self.nchannels, self.nepochs)

    __repr__ = __str__

    def _collapse_axis(self, samples, axis):
        if not axis in ('channel', 'epoch'):
            raise ValueError(
                "%s is not a valid axis, should be 'channel' or 'epoch'" % axis)
        caxes = self.axes
        if axis == 'channel':
            return self[slice(None, None, None), 0]._instantiatefromself(
                samples=samples)
        else:
            return self[slice(None, None, None), slice(None, None,
                                                       None), 0]._instantiatefromself(
                samples=samples)

    def copy(self, channels=None):
        """
        Makes a copy of the UniformTimeSeries, with the option of
        resetting channelnames.

        """
        if channels is None:
            channels = self.channels.copy()
        return self._instantiatefromself(channels=channels, copy=True)

    # def show(self, autovrange=False, vrangelow=None, vrangehigh=None,
    #          zmapper=None, **plotparams):
    #     from uts.display.traitsmultichannel import \
    #         EpochedMultiChannelUniformTimeSeriesPlot as Plot
    #
    #     p = Plot(self, geometry=self.geometry, zmapper=zmapper,
    #              autovrange=autovrange, vrangelow=vrangelow,
    #              vrangehigh=vrangehigh, **plotparams)
    #     p.configure_traits()
    #     return p

    # FIXME: make this resistent against appending large h5 signals
    def append(self, signal):
        """
        Append epochs

        """
        if not isinstance(signal, EpochedMultiChannelUniformTimeSeries):
            raise TypeError(
                "signal (%s) should be a EpochedMultiChannelUniformTimeSeries" % type(
                    signal).__name__)
        if not signal.fs == self.fs:
            raise TypeError(
                "cannot append a signal with different sampling rate")
        if not signal.nchannels == self.nchannels:
            raise TypeError("signals do not have the same number of channels")
        if not signal.ntimesamples == self.ntimesamples:
            raise TypeError("signals do not have the same number of samples")
        samples = signal.samples.transpose(self.samples.axisorder).get(
            copy=False)
        self.samples.append(samples, axis=self.epochaxis)
        self._append_epochs(signal.epochs)

    def concatenate_epochs(self):
        """
        Concatenates epochs so that the epoch dimension disappears and a
        MultiChannelUniformTimeSeries is created.

        """
        s = self.samples.transpose(['epoch', 'time', 'channel']).get(
            copy=False)
        s = s.reshape(-1, self.nchannels)
        if self.starttimes is None:
            starttime = 0.0
        else:
            starttime = self.starttimes[0]
        return MultiChannelUniformTimeSeries(samples=s, fs=self.fs,
                                             channels=self.channelnames,
                                             starttime=starttime,
                                             origintime=0.0, dtype=self.dtype,
                                             copy=True, channelaxis=1,
                                             geometry=self.geometry)

    def dot(self, m, channelnames=None):
        """
        Matrix multiplication m * signal. The number of new channels equals the
        size of the first axis of m. Hence the size of the second axis of m 
        should equal the number of channels of the signal.

        """
        ## the channelaxis must be 1
        samples = self.samples.transpose(['epoch', 'channel', 'time']).get(
            copy=False)
        channelaxis = 0
        epochaxis = 1
        samples = np.dot(m, samples)

        nchannels = m.shape[0]
        if channelnames is None:
            s = 'dotprod%%.%dd' % max(1, int(np.ceil(np.log10(nchannels))))
            channelnames = [s % i for i in range(nchannels)]
        if samples.dtype == self.samples.dtype:
            copy = False
        else:
            copy = True
        return self._instantiatefromself(samples=samples,
                                         channels=channelnames,
                                         copy=copy, dtype=samples.dtype,
                                         geometry=None,
                                         channelaxis=channelaxis,
                                         epochaxis=epochaxis)

    def transpose(self, axes=None):
        if axes is None:
            axes = self.samples.axisorder[::-1]
        samples = self.samples.transpose(axes)
        return self._instantiatefromself(samples=samples,
                                         channelaxis=samples.axes['channel'],
                                         epochaxis=samples.axes['epoch'])


def concatenate(sequence):
    if not len(sequence) > 0:
        raise ValueError('signal sequence cannot be empty')
    sig = sequence[0].copy()
    for frame in sequence[1:]:
        sig.append(frame)
    return sig

def isuts(s):
    return type(s) is UniformTimeSeries

def iseuts(s):
    return type(s) is EpochedUniformTimeSeries

def ismcuts(s):
    return type(s) is MultiChannelUniformTimeSeries

def isemcuts(s):
    return type(s) is EpochedMultiChannelUniformTimeSeries

definedtypes = (UniformTimeSeries,
                MultiChannelUniformTimeSeries,
                EpochedUniformTimeSeries,
                EpochedMultiChannelUniformTimeSeries)

def assert_signal(s, signaltypes=None):
    if signaltypes is None:
        signaltypes = definedtypes
    if not type(s) in signaltypes:
        raise TypeError(
            "object (%s) is not an allowed uniform time series type (%s)" \
            % (s.__class__, signaltypes))
    return True

def istimealigned(s1, s2):
    return (s1.ntimesamples == s2.ntimesamples) & \
           (s1.fs == s2.fs) & \
           (s1.origintime == s2.origintime)
