__all__ = ['FrequencySampled', 'Spectrum', 'PowerSpectrum',
           'MultiChannelPowerSpectrum', 'isspectrum', 'ispowerspectrum']

import numpy as np
from .check_arg import check_arg, check_argarray, check_intarray
from .samplesource import assamples, Samples  # FIXME
from .multichannel import MultiChannel

class FrequencySampled(object):
    """
    Represents a frequency sampled process.

    Mainly intended to be used as a super class for classes that represent
    frequency-sampled signals, but may also be useful on its own.

    Parameters
    ----------
    nfrequencysamples: int
        Number of samples.
    bandwidth: float
        Bandwidth in Herz.
    startfrequency: float, optional
        Frequency at which the first sample starts.

    Attributes
    ----------
    nfrequencysamples: int64
        Number of samples.
    bandwidth: float64
        Bandwidth in Herz.
    df: float64
        Sampling period in Herz.
    startfrequency: float64
        Frequency at which the first sample starts.
    endfrequency: float64
        Frequency at which the last sample ends.
    frequencyrange: (float64, float64)
        Start and end frequencies.


    Methods (see method doc for explanation)
    ----------------------------------------
    frequency_to_index
    index_to_frequency
    samplingfrequencies

    """

    def __init__(self, samplingfrequencies):
        self.__samplingfrequencies, self.__samplingboundaryfrequencies, self.__bandwidth = self._check_samplingfrequencies(
            samplingfrequencies)
        self.__nfrequencysamples = len(self.__samplingfrequencies)
        self.__startfrequency = samplingfrequencies[0]
        self.__df = self.__bandwidth / self.__nfrequencysamples
        self.__endfrequency = samplingfrequencies[-1]
        self.__frequencyrange = \
            (self.__startfrequency, self.__endfrequency)

    ################
    ## Properties ##
    ################

    nfrequencysamples = property(lambda self: (self.__nfrequencysamples), None,
                                 None,
                                 "number of samples")
    nf = nfrequencysamples
    bandwidth = property(lambda self: (self.__bandwidth), None, None,
                         "bandwidth in Herz")
    df = property(lambda self: (self.__df), None, None,
                  "sampling period in Herz")
    startfrequency = property(lambda self: (self.__startfrequency), None, None,
                              "endfrequency in Herz")
    endfrequency = property(lambda self: (self.__endfrequency), None, None,
                            "endfrequency in Herz")
    frequencyrange = property(lambda self: (self.__frequencyrange), None, None,
                              "frequency range in Herz")
    samplingfrequencies = property(lambda self: (self.__samplingfrequencies),
                                   None, None,
                                   "sampling frequencies in Herz")
    baseaxis = property(lambda self: ('frequency'), None, None, "this tells "
                                                                "functions the most likely axis along which something "
                                                                "should be calculated")

    ######################
    ## Standard methods ##
    ######################

    def __str__(self):
        return "<FrequencySampled: %.2f Hz to %.2f Hz, %d samples>" \
               % (
        self.startfrequency, self.endfrequency, self.nfrequencysamples)

    __repr__ = __str__

    #############
    ## Methods ##
    #############

    # FIXME CHECK
    def _check_samplingfrequencies(self, samplingfrequencies):
        """
        Returns the frequencies of *sample centers* and the boundaries

        """
        samplingfrequencies = np.array(samplingfrequencies)
        dfs = np.diff(samplingfrequencies)
        mean_df = np.mean(dfs)
        if not np.allclose(mean_df, dfs):
            raise ValueError("Sampling frequencies not equally spaced")
        samplingboundaryfrequencies = np.empty(samplingfrequencies.size + 1,
                                               dtype=samplingfrequencies.dtype)
        samplingboundaryfrequencies[:-1] = samplingfrequencies - mean_df / 2.
        samplingboundaryfrequencies[-1] = samplingfrequencies[
                                              -1] + mean_df / 2.
        bandwidth = samplingboundaryfrequencies[-1] - \
                    samplingboundaryfrequencies[0]
        return samplingfrequencies, samplingboundaryfrequencies, bandwidth

    def frequency_to_index(self, frequency, quantize=True):
        """
        Converts a frequency, or array of frequencies, to their corresponding
        sample indices. Note that the sample index is given of which the
        *start frequency*, not the sample center, is closest to the parameter
        frequency.

        Parameter
        ---------
        frequency: a float or float array-like

        Note
        ----
        We choose sample start frequency as a reference, because when one needs
        to select a frequency range on the basis of frequency info, one can only
        select at whole sample intervals. If we calculated using the frequency
        of sample center, then a frequency from both the begin and end part of
        the sample would result in the same index, while we would want to cut
        at the begin or end of the sample, respectively.

        Example: if the frequency parameter is [f0, f1, f2],

        |    s0    |    s1    |    s2    |    s3    |    s4    | ...
                     ^       ^  ^
                     f0      f1 f2

        then the result will be the sample of indices [s1, s2, s2].

        Please note that it is possible to get an index number of nfrequencysamples.

        """

        # check parameters
        frequency = check_argarray(frequency, 'frequency')
        quantize = check_arg(quantize, 'quantize')
        if ((frequency < self.__startfrequency) |
                (frequency > self.__endfrequency)).any():
            raise ValueError(
                "%.2f Hz is outside range of possible frequencies (%f,%f)" \
                % (frequency, self.__startfrequency, self.__endfrequency))
        # calculate
        index = ((frequency - self.__startfrequency) / self.__bandwidth) \
                * self.__nfrequencysamples
        if quantize:
            index = np.round(index).astype(np.uint64)
        if index.size == 1:
            return index[0]
        else:
            return index

    def index_to_frequency(self, index):
        """
        Converts a sample index, or array of indices, to their corresponding
        frequencies. 

        Parameter
        ---------
        index: an integer or array of integers

        """
        index = check_intarray(index, cast=np.uint64)
        if ((index < 0) | (index >= self.__nfrequencysamples)).any():
            raise ValueError(
                "index cannot be smaller than zero or higher than nfrequencysamples-1")
        result = index * self.__bandwidth / self.__nfrequencysamples \
                 + self.__startfrequency
        if result.size == 1:
            return result.item()
        else:
            return result

class Spectrum(FrequencySampled):
    _supporteddtypes = (np.int8, np.int16, np.int32, np.int64, np.uint8,
                        np.uint16, np.uint32, np.uint64, np.float16,
                        np.float32,
                        np.float64, np.complex128)

    def __init__(self, samples, samplingfrequencies, dtype=None,
                 copy=True, readonly=True):
        copy = check_arg(copy, 'copy')
        self.__samples = self._check_samples(samples, dtype=dtype, copy=copy,
                                             readonly=readonly)
        self._check_dimension(self.__samples)
        self.__sampletype = self.__samples.sampletype

        self.__dtype = self.__samples.dtype

        nfrequencysamples = self.__samples.shape[
            self.__samples.axes['frequency']]
        FrequencySampled.__init__(self,
                                  samplingfrequencies=samplingfrequencies)

    samples = property(lambda self: (self.__samples), None, None,
                       "Samples object")
    dtype = property(lambda self: (self.__dtype), None, None,
                     "array of samples")
    sampletype = property(lambda self: (self.__sampletype), None, None,
                          "type of samples object")

    def _check_samples(self, samples, dtype=None, copy=True, readonly=True):
        if not isinstance(samples, Samples):
            samples = assamples(samples, axes={'frequency': 0}, dtype=dtype,
                                copy=copy, readonly=readonly)
        return samples

    def _check_dimension(self, samples):
        if samples.ndim != 1:
            raise ValueError("input samples array has wrong dimensionality")

    def plot(self, ga=None, **kwargs):
        from ..viz.plots import plot_spectrum

        return plot_spectrum(self, ga=ga, **kwargs)


            # def ifft(self):
            #     samples = np.fft.ifft(self.samples.get(copy=False)
            #             * float(self.nfrequencysamples)**0.5).astype(np.float64)
            #     from uniformtime import UniformTimeSeries
            #     return UniformTimeSeries(samples=samples,
            #                         fs=self.bandwidth, copy=False)

            # def get_plot(self):
            #     from uts.display.singlechannel import waveplot
            #     samples = self.samples.get(copy=False)
            #     samples = np.absolute(samples * samples.conjugate())
            #     samples = 10*np.log10(samples)
            #     return waveplot(x=self.samplingfrequencies(), y=samples,
            #              xlabel="Frequency (Hz)", ylabel="Power (dB)",
            #              title="Power Spectrum")
            #
            #
            # def show(self, labels={}):
            #     from uts.display.singlechannel import PlotView
            #     PlotView(self.get_plot()).configure_traits()

class PowerSpectrum(FrequencySampled):
    _supporteddtypes = (np.int8, np.int16, np.int32, np.int64, np.uint8,
                        np.uint16, np.uint32, np.uint64, np.float16,
                        np.float32,
                        np.float64)

    def __init__(self, samples, samplingfrequencies, dtype=None,
                 copy=True, readonly=True):
        copy = check_arg(copy, 'copy')
        self.__samples = self._check_samples(samples, dtype=dtype, copy=copy,
                                             readonly=readonly)
        self.__sampletype = self.__samples.sampletype
        self.__dtype = self.__samples.dtype

        nfrequencysamples = self.__samples.shape[
            self.__samples.axes['frequency']]
        FrequencySampled.__init__(self,
                                  samplingfrequencies=samplingfrequencies)

    samples = property(lambda self: (self.__samples), None, None,
                       "Samples object")
    dtype = property(lambda self: (self.__dtype), None, None,
                     "array of samples")
    sampletype = property(lambda self: (self.__sampletype), None, None,
                          "type of samples object")

    def _check_samples(self, samples, dtype=None, copy=True, readonly=True):
        if not isinstance(samples, Samples):
            samples = assamples(samples, axes={'frequency': 0}, dtype=dtype,
                                copy=copy, readonly=readonly)
        return samples

    def plot(self, db=True, ax=None, **kwargs):
        from ..viz.plots import plot_powerspectrum

        return plot_powerspectrum(self, db=db, ax=ax, **kwargs)




        # def get_plot(self):
        #     from uts.display.singlechannel import waveplot
        #     samples = self.samples.get(copy=False)
        #     samples = 10*np.log10(samples)
        #     return waveplot(x=self.samplingfrequencies(), y=samples,
        #              xlabel="Frequency (Hz)", ylabel="Power (dB)",
        #              title="Power Spectrum")
        #
        # def show(self, labels={}):
        #     from uts.display.singlechannel import PlotView
        #     PlotView(self.get_plot()).configure_traits()


class MultiChannelPowerSpectrum(MultiChannel, PowerSpectrum):
    def __init__(self, samples, samplingfrequencies, dtype=None,
                 copy=True, channelnames=None, channelaxis=1,
                 geometry=None, readonly=True):

        self.__channelaxis, self.__frequencyaxis = self._check_channelaxis(
            channelaxis)
        PowerSpectrum.__init__(self, samples=samples,
                               samplingfrequencies=samplingfrequencies,
                               dtype=dtype, copy=copy, readonly=readonly)
        nchannels = self.samples.shape[self.__channelaxis]
        MultiChannel.__init__(self, nchannels=nchannels,
                              channelnames=channelnames, geometry=geometry)

    channelaxis = property(
        lambda self: (self.__channelaxis), None, None,
        "the axis of the samples array that corresponds to channels")
    frequencyaxis = property(
        lambda self: (self.__frequencyaxis), None, None,
        "the axis of the samples array that corresponds to time")

    _instantiationparams = property(
        lambda self: ({'samples': self.samples,
                       'bandwidth': self.bandwidth,
                       'startfrequency': self.startfrequency,
                       'dtype': self.dtype,
                       'copy': True,
                       'channels': self.channels,
                       'channelaxis': self.samples.axes['channel'],
                       'geometry': self.geometry,
                       'readonly': self.readonly}),
        None, None, "parameter dictionary for instantiation")

    def _check_channelaxis(self, channelaxis):
        channelaxis = check_arg(channelaxis, 'channelaxis')
        if not channelaxis in (0, 1):
            raise IndexError("channelaxis should be 0 or 1")
        if channelaxis == 0:
            frequencyaxis = 1
        else:
            frequencyaxis = 0
        return channelaxis, frequencyaxis

    def _check_samples(self, samples, dtype=None, copy=True, readonly=True):
        return assamples(samples, axes={'frequency': self.__frequencyaxis,
                                        'channel': self.__channelaxis},
                         dtype=dtype, copy=copy, readonly=readonly)

    def __str__(self):
        return "<MultiChannelPowerSpectrum: %.2f Hz to %.2f Hz, %d samples, %d channels>" \
               % (
        self.startfrequency, self.endfrequency, self.nfrequencysamples,
        self.nchannels)

    __repr__ = __str__


def isspectrum(s):
    return type(s) is Spectrum

def ispowerspectrum(s):
    return type(s) is PowerSpectrum
