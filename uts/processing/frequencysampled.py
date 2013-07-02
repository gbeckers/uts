__all__ = ['FrequencySampled', 'Spectrum']

import numpy as np
from uts.core.check_arg import check_arg, check_argarray, check_intarray
from uts.core.samplesource import assamples, Samples # FIXME

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

    def __init__(self, nfrequencysamples, bandwidth, startfrequency):

        self.__nfrequencysamples = check_arg(nfrequencysamples, 'nfrequencysamples')
        self.__bandwidth = check_arg(bandwidth, 'bandwidth')
        self.__startfrequency = check_arg(startfrequency, 'frequency')
        self.__df = self.__bandwidth / self.__nfrequencysamples
        self.__endfrequency = self.__startfrequency + self.__bandwidth
        self.__frequencyrange = \
            (self.__startfrequency, self.__endfrequency)

    ################
    ## Properties ##
    ################

    nfrequencysamples = property(lambda self: (self.__nfrequencysamples), None, None,
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
    baseaxis = property(lambda self: ('frequency'), None, None, "this tells " 
                       "functions the most likely axix along which something "
                       "should be calculated")

    ######################
    ## Standard methods ##
    ######################

    def __str__(self):
        return "<FrequencySampled: %.2f Hz to %.2f Hz, %d samples>" \
                    % (self.startfrequency, self.endfrequency, self.nfrequencysamples)

    __repr__ = __str__

    #############
    ## Methods ##
    #############

    def samplingfrequencies(self):
        """
        Returns the frequencies of *sample centers*.

        """
        return (np.arange(self.__nfrequencysamples, dtype=np.float64) + 0.5) \
               * self.__bandwidth / self.__nfrequencysamples + self.__startfrequency
               
    def samplingboundaryfrequencies(self):
          return np.arange(self.__nfrequencysamples+1, dtype=np.float64) \
               * self.__bandwidth / self.__nfrequencysamples + self.__startfrequency
               
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
        frequencies. Note that the *frequency of sample center* is given, not
        the frequency of the sample start.

        Parameter
        ---------
        index: an integer or array of integers

        Note
        ----
        Center of samples define their frequency of occurrence

        |    0    |    1    |    2    |    3    |    4    | ...
             ^         ^         ^         ^         ^
             f0        f1        f2        f3        f4

        """
        index = check_intarray(index, cast=np.uint64)
        if ((index < 0) | (index >= self.__nfrequencysamples)).any():
            raise ValueError(
                "index cannot be smaller than zero or higher than nfrequencysamples-1")
        result = (index + 0.5) * self.__bandwidth / self.__nfrequencysamples \
               + self.__startfrequency
        if result.size == 1:
            return result.item()
        else:
            return result

class Spectrum(FrequencySampled):

    _supporteddtypes = (np.int8, np.int16, np.int32, np.int64, np.uint8,
                        np.uint16, np.uint32, np.uint64, np.float32,
                        np.float64, np.complex128)

    def __init__(self, samples, bandwidth, startfrequency, dtype=None,
                 copy=True):
        copy = check_arg(copy, 'copy')
        self.__samples = self._check_samples(samples, dtype=dtype, copy=copy)
        self._check_dimension(self.__samples)
        self.__sampletype = self.__samples.sampletype

        self.__dtype = self.__samples.dtype

        nfrequencysamples = self.__samples.shape[self.__samples.axes['frequency']]
        FrequencySampled.__init__(self, nfrequencysamples=nfrequencysamples, bandwidth=bandwidth,
                             startfrequency=startfrequency)

    samples = property(lambda self: (self.__samples), None, None,
                       "Samples object")
    dtype = property(lambda self: (self.__dtype), None, None,
                       "array of samples")
    sampletype = property(lambda self: (self.__sampletype), None, None,
                       "type of samples object")

    def _check_samples(self, samples, dtype=None, copy=True):
        if not isinstance(samples, Samples):
            samples = assamples(samples, axes={'frequency': 0}, dtype=dtype,
                                       copy=copy)
        return samples

    def _check_dimension(self, samples):
        if samples.ndim != 1:
            raise ValueError("input samples array has wrong dimensionality")

    def ifft(self):
        samples = np.fft.ifft(self.samples.get(copy=False)
                * float(self.nfrequencysamples)**0.5).astype(np.float64)
        from uniformtime import UniformTimeSeries
        return UniformTimeSeries(samples=samples,
                            fs=self.bandwidth, copy=False)

    def get_plot(self):
        from uts.display.singlechannel import waveplot
        samples = self.samples.get(copy=False)
        samples = np.absolute(samples * samples.conjugate())
        samples = 10*np.log10(samples)
        return waveplot(x=self.samplingfrequencies(), y=samples,
                 xlabel="Frequency (Hz)", ylabel="Power (dB)",
                 title="Power Spectrum")
        

    def show(self, labels={}):
        from uts.display.singlechannel import PlotView
        PlotView(self.get_plot()).configure_traits()


