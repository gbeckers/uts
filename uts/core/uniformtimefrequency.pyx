import numpy as np

from .uniformtime import UniformTimeSeries
from .uniformfrequency import FrequencySampled
from .samplesource import Samples, assamples #FIXME
from .check_arg import check_arg


class Spectrogram(UniformTimeSeries, FrequencySampled):

    def __init__(self, samples, fs, bandwidth, samplingfrequencies, frequencyaxis=1,
                 starttime=0.0, origintime=0.0, copy=True):

        self.__frequencyaxis, self.__timeaxis = \
            self._check_frequencyaxis(frequencyaxis)

        UniformTimeSeries.__init__(self, samples=samples, fs=fs, starttime=0.0,
                                   origintime=0.0, dtype=None, copy=copy)

        nfrequencysamples = self.samples.shape[self.__frequencyaxis]

        FrequencySampled.__init__(self, bandwidth=bandwidth, samplingfrequencies=samplingfrequencies)

    frequencyaxis = property(
        lambda self: (self.__frequencyaxis), None, None,
        "the axis of the samples array that corresponds to frequency")
    timeaxis = property(
        lambda self: (self.__timeaxis), None, None,
        "the axis of the samples array that corresponds to time")
    
    
    def __str__(self):
        return "<TimeFrequencySampledSignal: %.3f seconds, %d time x %d frequency samples>" \
                            % (self.duration, self.ntimesamples, self.nfrequencysamples)
    __repr__ = __str__

    def _check_samples(self, samples, dtype=None, copy=True, readonly=False):
        if not isinstance(samples, Samples):
            samples = assamples(samples, axes={'time': self.__timeaxis,
                        'frequency': self.__frequencyaxis}, dtype=dtype,
                                         copy=copy, readonly=readonly)
        #if not samples.dtype == np.complex128:
        #    raise TypeError, "samples of a spectrogram should be complex"
        return samples

    def _check_frequencyaxis(self, frequencyaxis):
        channelaxis = check_arg(frequencyaxis, 'frequencyaxis')
        if not channelaxis in (0,1):
            raise IndexError("frequencyaxis should be 0 or 1")
        if frequencyaxis == 0:
            timeaxis = 1
        else:
            timeaxis = 0
        return frequencyaxis, timeaxis

    def _check_dimension(self, samples):
        if samples.ndim != 2:
            raise ValueError("input samples array has wrong dimensionality")

    #FIXME klopt niet
    def get_plot(self, log=True, dynamicrange=50.):
        
        from old_code.singlechannel import imageplot
        
        samples = self.samples.get(copy=False)
        if self.timeaxis == 0:
            endfreqindex = samples.shape[1]//2
            samples = samples[:,:endfreqindex].T
        else:
            endfreqindex = samples.shape[0]//2
            samples = samples[:endfreqindex,:]
        if samples.dtype == np.complex128:
            samples = np.absolute(samples * samples.conjugate())
        if log:
            samples = 10*np.log10(samples)
            high = np.max(samples)
            np.clip(samples, high-dynamicrange, high, samples)
        x = self.samplingboundarytimes()
        y = self.samplingfrequencies[:endfreqindex+1]
        return imageplot(img=samples, x=x, y=y, xlabel="time (s)", ylabel="Frequency (Hz)",
                 title="Power Spectrum")

    def show(self, log=True, dynamicrange=50.):

        from old_code.singlechannel import PlotView
        
        PlotView(self.get_plot(log=log,dynamicrange=dynamicrange)).configure_traits()

