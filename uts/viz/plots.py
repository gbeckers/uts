__author__ = 'gabriel'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec
import matplotlib.cm
import uts
from ..core.geometry import RectangularLattice
from core import GridAxes


def plot_uts(s, ax=None, **kwargs):
    defaults = {'color': (0.3,0.3,0.3)}
    for key, item in defaults.iteritems():
        if not key in kwargs:
            kwargs[key] = item
    if ax is None:
        ax = plt.gca(frameon=False)

    pl = ax.plot(s.samplingtimes(),
        s.samples.get(False), **kwargs)
    ax.tick_params(direction='out', top=False, right=False)
    return ax


def plot_mcuts(mcuts, ylim=None, wspace=0.06,
                   hspace=0.06, left=None, right=None, bottom=None, top=None,
                   ylabel=None, xlabel='Time (s)', yticks=None, xticks=None,
                ga=None, **kwargs):
    if ga is None:
        ga = GridAxes(mcuts.geometry, wspace=wspace, hspace=hspace,
                      left=left, right=right,
                      bottom=bottom, top=top,
                      ylabel=ylabel, xlabel=xlabel,
                      yticks=yticks, xticks=xticks, fig=None)

    defaults = {'color': (0.3,0.3,0.3)}
    # if plot_params is None:
    #     plot_params = {}
    for key, item in defaults.iteritems():
        if not key in kwargs:
            kwargs[key] = item
    if ylim is None:
        low = np.min(uts.min(mcuts))
        high = np.max(uts.max(mcuts))
        low -= abs(low * 0.05)
        high += abs(high * 0.05)
    else:
        low, high = ylim
    for i, channelname in enumerate(mcuts.channelnames):
        ax = ga.axes[channelname]
        pl = ax.plot(mcuts.samplingtimes(),
            mcuts[:, channelname].samples.get(False), **kwargs)
        ax.set_ylim([low, high])
    return ga


def euts_image(euts, ranges=None, clip=None, zeroincenter=False,
               invert=False, colormap=None, includebar=False,
               transparantcolor=None):
    from scipy.stats import scoreatpercentile

    if (not (clip is None)) and (not (ranges is None)):
        raise ValueError("You cannot set ranges and clip parameters at the"
                         " same time")
    if colormap is None:
        colormap = matplotlib.cm.jet
    samples = euts.samples.get(copy=True)
    if euts.epochaxis == 1:
        samples = np.rot90(samples)
    low = samples.min()
    high = samples.max()
    if not clip is None:
        low, high = clip
    if not ranges is None:
        low = scoreatpercentile(samples.flatten(), ranges[0])
        high = scoreatpercentile(samples.flatten(), ranges[1])
    np.clip(samples, low, high, samples)
    if zeroincenter:
        high = max(abs(low), abs(high))
        low = -high
    samples -= low
    high -= low
    samples /= high
    if invert:
        samples -= 1.0
        samples[:] = np.absolute(samples)
    colors = colormap(samples)
    if not transparantcolor is None:
        i = (colors[:, :, :3] == transparantcolor)
        j = (i.sum(-1) == 3)
        colors[j, 3] = 0.0
    return colors


def save_euts_image(euts, basefilename, path='.', ranges=None, clip=None,
                    zeroincenter=False, invert=False, colormap=None,
                    includebar=False, transparantcolor=None):
    """
    ranges: 2-item sequence or None
        The range in *percentiles*, (pmin, pmax).
    clip: 2-item sequence or None
        The values at which clipping should take place (cmin, cmax).

    """
    from scipy.misc import toimage
    from os.path import abspath, join
    from pylab import cm

    if colormap is None:
        colormap = cm.jet
    if (not (clip is None)) and (not (ranges is None)):
        raise ValueError("You cannot set ranges and clip parameters at the"
                         " same time")
    path = abspath(path)
    im = euts_image(euts=euts, ranges=ranges, clip=clip,
                    zeroincenter=zeroincenter, invert=invert,
                    colormap=colormap, includebar=includebar,
                    transparantcolor=transparantcolor)
    im = toimage(im)
    absfilename = join(path, '%s.png' % basefilename)
    im.save(absfilename)
    if includebar:
        if invert:
            bar = np.linspace(1.0, 0.0, 512).reshape(1, -1)
        else:
            bar = np.linspace(0.0, 1.0, 512).reshape(1, -1)
        barim = toimage(colormap(bar))
        absfilename = join(path, '%s_bar.png' % basefilename)
        barim.save(absfilename)
    return absfilename

    ## uts
    # def get_plot(self, labels={}, indexrange=None):
    # from uts.display.singlechannel import waveplot
    #
    # samples = self.samples.get(copy=False)
    #     return waveplot(x=self.samplingtimes(), y=samples,
    #                     xlabel="Time (s)", ylabel="Amplitude",
    #                     title="Wave Plot", labels=labels, indexrange=indexrange)

    ## euts
    # def get_plot(self):
    #        from uts.display.singlechannel import ImagePlot
    #
    #        samples = self.samples.get(copy=False)
    #        if self.epochaxis == 1:
    #            samples = samples.T
    #        x = self.samplingboundarytimes()
    #        y = np.arange(self.nepochs + 1)
    #        return ImagePlot(img=samples, x=x, y=y, xlabel="time (s)",
    #                         ylabel="Epoch", title="")


def plot_spectrum(s, xlim=None, wspace=0.06,
                   hspace=0.1, left=None, right=None, bottom=None, top=None,
                   yticks=None, xticks=None, ga=None, **kwargs):
    defaults = {'color': 'black'}
    for key, item in defaults.iteritems():
        if not key in kwargs:
            kwargs[key] = item
    if ga is None:
        g = RectangularLattice(1, 2, [['magnitude'],['phase']])
        ga = GridAxes(g, wspace=wspace, hspace=hspace,
                      left=left, right=right,
                      bottom=bottom, top=top,
                      ylabel='', xlabel='',
                      yticks=yticks, xticks=xticks, fig=None)
    if xlim is None:
        xlim = s.frequencyrange
    ax1 = ga.axes['magnitude']
    samples = np.abs(s.samples.get(False))
    ylim =  (samples.min(), samples.max())
    pl1 = ax1.plot(s.samplingfrequencies, samples, **kwargs)
    ax1.set_ylabel('Magnitude')
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_yticks(_ticks(ylim))
    ax1.tick_params(direction='out', top=False, right=False)

    ax2 = ga.axes['phase']
    pl2 = ax2.plot(s.samplingfrequencies,
                   np.angle(s.samples.get(False)), **kwargs)
    ax2.set_ylabel('Phase')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.tick_params(direction='out', top=False, right=False)
    ax2.set_xlim(xlim)
    ax2.set_ylim((-np.pi, np.pi))
    ax2.set_yticks((-np.pi, 0., np.pi))
    ax2.set_xticks(_ticks(xlim))

    return ga


def plot_powerspectrum(s, ax=None, db=True, **kwargs):
    defaults = {'color': (0.3,0.3,0.3)}
    for key, item in defaults.iteritems():
        if not key in kwargs:
            kwargs[key] = item
    if ax is None:
        ax = plt.gca()
    samples = s.samples.get(False) ** 2.
    if db:
        samples = 10 * np.log10(samples)
    pl = ax.plot(s.samplingfrequencies,
                 samples, **kwargs)
    ax.tick_params(direction='out', top=False, right=False)
    return ax

def _ticks(extremes):
    if extremes[0] < 0. and extremes[1] > 0.:
        return (extremes[0], 0., extremes[1])
    else:
        return extremes