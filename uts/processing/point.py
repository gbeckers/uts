import numpy
from helper import LazyProperty
from uts.core.multichannel import MultiChannel
from uts.core.epoched import Epoched
from uts import UniformTimeSeries, MultiChannelUniformTimeSeries
from check_arg import check_arg, check_argarray

class DimensionError(Exception): pass
class TimeRangeError(Exception): pass

class PointProcess(object):

    _supporteddtypes = (numpy.float32, numpy.float64)


    def __init__(self, timings, starttime, duration, origintime=0.0, dtype=None,
                 copy=True):
        '''
        timings are given *relative to origintime*, which itself is relative to
        starttime.
        '''

        self.__origintime = check_arg(origintime, 'time')
        self.__timings = self._timings_to_arrays(timings=timings, dtype=dtype,
                                            copy=copy)
        self.__starttime = check_arg(starttime, 'time')
        self.__duration = check_arg(duration, 'duration')
        self.__timerange = (check_arg(0.0-self.__origintime, 'time'),
                            check_arg(self.__duration-self.__origintime, 'time'))
        if (self.__timings < self.__timerange[0]).any():
            raise TimeRangeError("not all points occur after starttime")
        if (self.__timings > self.__timerange[1]).any():
            raise TimeRangeError("not all points occur before endtime")


        self.__npoints = self.__timings.size
        self.__dtype = self.__timings.dtype

    dtype = property(lambda self: (self.__dtype), None, None,
                       "")

    timings = property(lambda self: (self.__timings), None, None,
                  "timing of points")

    starttime = property(lambda self: (self.__starttime), None, None,
                  "starttime of point process")

    timerange = property(lambda self: (self.__timerange), None, None,
                  "endtime of point process")

    origintime = property(lambda self: (self.__origintime), None, None,
                  "origin time of point process, relative to start")

    npoints = property(lambda self: (self.__npoints), None, None,
                  "number of points")

    duration = property(lambda self: (self.__duration), None, None,
                         "")

    def __getitem__(self, index):

        pointsslice, starttime, origintime, duration = self._to_pointsslice(index=index)

        return self.__class__(timings=self.__timings[pointsslice].copy(),
                starttime=starttime, duration=duration, copy=False,
                origintime=origintime)

    def _to_pointsslice(self, index):

        allowedfloats = (float, numpy.float32, numpy.float64)

        if not isinstance(index, slice):
            raise IndexError("index of point dimension can only be a slice")
        if not index.step is None:
            raise IndexError("point dimension cannot have a step index")

        if index.start is None:
            start = 0
            startrange = self.timerange[0]
        else:
            if not isinstance(index.start, allowedfloats):
                raise IndexError("start index should be float")
            if index.start < self.timerange[0]:
                raise IndexError(
                    "starttime may not be smaller than start process")
            start = numpy.searchsorted(self.timings, index.start, side='right')
            startrange = index.start

        if index.stop is None:
            stop = self.npoints
            endrange = self.timerange[1]
        else:
            if not isinstance(index.stop, allowedfloats):
                raise IndexError("stop index should be float")
            if index.stop > self.timerange[1]:
                raise IndexError("endtime may not be larger than end process")
            stop = numpy.searchsorted(self.timings, index.stop, side='right')
            endrange = index.stop


        start,stop,step = slice(start, stop).indices(self.npoints)
        if not start <= stop:
            raise IndexError("point index start may not be higher than end")

        pointsslice = slice(start,stop)
        starttime = self.starttime + (startrange + self.origintime)
        origintime = -startrange
        duration = endrange - startrange

        return pointsslice, starttime, origintime, duration


    def _timings_to_arrays(self, timings, dtype, copy):

        if dtype is None:
            dtype = numpy.float64

        if dtype not in self._supporteddtypes:
            raise TypeError("input dtype %s not supported" % dtype)

        if not isinstance(timings, numpy.ndarray):
            try:
                timings = numpy.array(timings, dtype=dtype)
            except (TypeError, ValueError) as e:
                # This used to read `TypeError, "..."`, python 2 syntax for
                # raising with a message. Without the `raise` python 3 reads it
                # as a tuple expression that does nothing, so the conversion
                # failure was swallowed and the unconverted input passed on.
                raise TypeError(
                    "input type cannot be converted to numpy array") from e
        else:
            if copy:
                timings = numpy.array(timings, dtype=dtype)
            else:
                if not dtype == timings.dtype:
                    timings = timings.astype(dtype)

        # in case we received a 0-dimensional array
        if timings.ndim == 0:
            raise DimensionError("input array has no dimensions")

        timings.sort()

        return timings

    def to_histogram(self, binduration):
        nbins = self.duration // binduration
        start = self.timerange[0]
        end = start + nbins * binduration
        fs = 1/float(binduration)
        y,x = numpy.histogram(self[start:end].timings, bins=nbins, new=True)
        return UniformTimeSeries(samples=y, fs=fs, starttime=self.starttime,
                origintime=self.origintime, dtype=None,copy=False)

        

    
def interpointintervals(p1, p2, maxintervaldur, excludesimultaneous=False):
    globalintervals = []
    laststartt2 = 0
    for t1 in p1.timings:        
        localintervals = []
        for t2 in p2.timings[laststartt2:]:
            curinterval = t2 - t1
            if  curinterval > maxintervaldur:
                break
            elif curinterval >= -maxintervaldur:
                if not excludesimultaneous:
                    localintervals.append(curinterval)
                else:
                    if curinterval != 0.0:
                        localintervals.append(curinterval)
            else:
                laststartt2 +=1
        globalintervals.append(localintervals)
    return globalintervals

def interpointintervalhist(p1,p2, maxintervaldur, nbins=100,excludesimultaneous=False):
    intervals = interpointintervals(p1, p2, maxintervaldur,excludesimultaneous=excludesimultaneous)
    rows = len(intervals)
    hist = numpy.empty((rows,nbins), dtype=int)
    for i,interval in enumerate(intervals):
        hist[i,:] = numpy.histogram(interval,bins=nbins,
                        range=(-maxintervaldur,maxintervaldur))[0]
    return hist
                
    

class MultiChannelPointProcess(MultiChannel, PointProcess):


    '''
    timings: a list of timing sequences
    '''

    def __init__(self, timings, starttime, duration, origintime=0.0,
                 channelnames=None, dtype=None, copy=True, mcshape=None):

        #fixed attributes
        self.__duration = check_arg(duration, 'duration')
        self.__starttime = check_arg(starttime, 'time')
        self.__origintime = check_arg(origintime, 'time')
        self.__timerange = (check_arg(0.0-self.__origintime, 'time'),
                            check_arg(self.__duration-self.__origintime, 'time'))
        self.__timings = self._timings_to_pplist(timings=timings,
                            dtype=dtype, copy=copy)
        nchannels = len(self.__timings)
        MultiChannel.__init__(self, nchannels=nchannels,
                    channelnames=channelnames, mcshape=mcshape)
        npoints = []
        for channel in range(nchannels):
            npoints.append(self.__timings[channel].size)

        self.__npoints = numpy.array(npoints)



    starttime = property(lambda self: (self.__starttime), None, None,
                         "")

    timerange = property(lambda self: (self.__timerange), None, None,
                         "")

    duration = property(lambda self: (self.__duration), None, None,
                         "")

    origintime = property(lambda self: (self.__origintime), None, None,
                         "")

    timings = property(lambda self: (self.__timings), None, None,
                         "")

    npoints = property(lambda self: (self.__npoints), None, None,
                         "")


    def __getitem__(self, index):

        index = numpy.index_exp[index]
        if not len(index) <= 2:
            raise IndexError(
                "index can maximally have two dimensions [pointdim,channeldim]")

        # if there are two parts to index, the second one refers to channels
        if len(index) == 2:
            channelindex = self._to_channelsslice(index[1])
        else:
            channelindex = slice(None,None,None)

        #channelindices = numpy.arange(self.nchannels)[channelindex]


        #if isinstance(channelindices,(int,long)):
        #    channelindices = [channelindices]

        pointsslice = index[0]
        pch1 = PointProcess(self.timings[channelindex],
                    starttime=self.starttime, duration=self.duration,
                    origintime=self.origintime,copy=False)[pointsslice]
        starttime = pch1.starttime
        duration = pch1.duration
        origintime = pch1.origintime
        timings = [pch1.timings]
        if not isinstance(channelindex,(int,long)):
            channelindices = numpy.arange(self.nchannels)[channelindex]
            for i in channelindices[1:]:
                p = PointProcess(self.timings[i],
                    starttime=self.starttime, duration=self.duration,
                    origintime=self.origintime,copy=False)
                timings.append(p[pointsslice].timings)

        if self.channelnames is not None:
            channelnames = self.channelnames[channelindex]
        else:
            channelnames = None

        if len(timings)==1: # one channel only
            return PointProcess(timings=timings[0], starttime=starttime,
                    duration=duration, origintime=origintime, copy=False)
        else:
            return MultiChannelPointProcess(timings=timings,
                    channelnames=channelnames, starttime=starttime,
                    duration=duration, copy=False, origintime=origintime)


    def _timings_to_pplist(self, timings, dtype, copy):
        pplist = []
        for ptimings in timings:
            pplist.append(self._timings_to_arrays(timings=ptimings,
                                dtype=dtype, copy=copy))
        return pplist

    def to_histogram(self, binduration):
        samples = []
        for i in range(self.nchannels):
            ss = self[:,i].to_histogram(binduration=binduration)
            samples.append(ss.get_samples())
        return MultiChannelUniformTimeSeries(samples=numpy.array(samples).T,
                fs=ss.fs, starttime=ss.starttime, origintime=ss.origintime,
                channelaxis=1, copy=False, channelnames=self.channelnames)


class h5MultiChannelPointProcess(MultiChannelPointProcess):
    '''
    A h5-based (i.e. on-disk) multichannel point process. Represents a PyTables
    VLArray node, with at least four attributes: starttime, endtime, origin,
    and fieldorder.
    '''

    def __init__(self, h5node, mcshape=None):

        self.__h5node = h5node

        channelnames = h5node.attrs.fieldorder
        nchannels = h5node.shape[0]
        MultiChannel.__init__(self, nchannels=nchannels,
                    channelnames=channelnames, mcshape=mcshape)

        starttime = h5node.attrs.starttime
        endtime = h5node.attrs.endtime
        self.__origintime = check_arg(h5node.attrs.origintime,'time')
        self.__starttime = check_arg(starttime, 'time')
        self.__endtime = check_arg(endtime, 'time')
        self.__duration = check_arg(self.__endtime - self.__starttime, 'duration')

    node = property(lambda self: (self.__h5node), None, None,
                       "array of samples")


    starttime = property(lambda self: (self.__starttime), None, None,
                         "")

    endtime = property(lambda self: (self.__endtime), None, None,
                         "")

    duration = property(lambda self: (self.__duration), None, None,
                         "")

    origintime = property(lambda self: (self.__origintime), None, None,
                         "")

    @LazyProperty
    def timings(self):
        '''
        Returns a list of arrays
        '''
        return self.__h5node[:]


    def _timings_to_pplist(self, timings, starttime, endtime, origintime,
                           dtype, copy):
        '''Overrides method of superclass. Should do nothing.
        '''
        pass


#FIXME hier verder
class EpochedMultiChannelPointProcess(Epoched, MultiChannel, PointProcess):

    def __init__(self, timings, starttimes, epochduration, origintime=0.0,
                channelnames=None, dtype=None, copy=True, mcshape=None,
                records=None, geometry=None):
        '''
        timings is a list of lists of points (channels x epochs x points)
        '''
        #fixed attributes
        self.__timings = self._timings_to_mcpplist(timings=timings,
            starttimes=starttimes, duration=epochduration, origintime=origintime,
            dtype=dtype, copy=copy)

        Epoched.__init__(self, starttimes=starttimes, duration=epochduration,
                         origintime=origintime, records=records)

        nchannels = len(self.__timings)
        MultiChannel.__init__(self, nchannels=nchannels,
                    channelnames=channelnames, mcshape=mcshape, geometry=geometry)
        nepochs = len(self.__timings[0])
        if not nepochs == len(starttimes):
            raise DimensionError(
                'number of starttimes does not match number of epochs')

        self.__dtype = self.__timings[0][0].dtype

    timings = property(lambda self: (self.__timings), None, None,
                       "")

    dtype = property(lambda self: (self.__dtype), None, None,
                       "")

    def __getitem__(self, index):

        index = numpy.index_exp[index]
        if not len(index) <= 3:
            raise IndexError(
                "index can maximally have three dimensions [epochdim, pointsdim, channeldim]")

        if len(index) == 3:
            channelindex = self._to_channelsslice(index[2])
            pointsindex = index[1]
        elif len(index) == 2:
            channelindex = slice(None,None,None)
            pointsindex = index[1]
        else:
            channelindex = slice(None,None,None)
            pointsindex = slice(None,None,None)
        epochsindex = self._to_epochsindex(index[0])

        epochsindices = numpy.arange(self.nepochs)[epochsindex]
        if isinstance(epochsindices,(int,long)):
            epochsindices = [epochsindices]
        channelindices = numpy.arange(self.nchannels)[channelindex]
        if isinstance(channelindices,(int,long)):
            channelindices = [channelindices]


        timings = []
        for i in channelindices:
            timings.append([])
            channel = self.__timings[i]
            for j in epochsindices:
                timings[-1].append([])
                epoch = channel[j]
                timings[-1][-1].append(self.timings[i,j][pointsindex])


    def _timings_to_mcpplist(self, timings, starttimes, duration, dtype=None,
                    copy=None, origintime=0.0):
        # (channels x epochs x points)

        if not isinstance(timings,list):
            raise TypeError('timings parameter should be a list')

        points = []
        nepochschannel0 = len(timings[0])
        nepochs = []
        for channel in timings:
            if not isinstance(channel, list):
                raise TypeError('timings parameter should be a list of lists')
            points.append([])
            for starttime, epoch in zip(starttimes,channel):
                epoch = numpy.array(epoch) - (starttime + origintime)
                points[-1].append(PointProcess(timings = epoch,
                    starttime=starttime, duration=duration, dtype=dtype,
                    copy=copy, origintime=origintime))
            nepochs.append(len(points[-1]))
        if not (numpy.array(nepochs) == nepochschannel0).all():
            raise DimensionError(
                'not all channels have the same number of epochs')
        return points

    def show(self):
        from display.multichannel import EpochedMultiChannelPointProcessPlot as Plot
        return Plot(self, plotparams={}, geometry=self.geometry)
