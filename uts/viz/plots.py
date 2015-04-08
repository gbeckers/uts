__author__ = 'gabriel'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec
import matplotlib.cm
import uts


def mcuts_subplots(mcuts, figsize=(12, 12), ylim=None, spacing=0.06,
                   plot_params=None, yticks=((), ()), ylabel='', fig=None):
    defaults = {'color': 'black'}

    if plot_params is None:
        plot_params = {}
    for key, item in defaults.iteritems():
        if not key in plot_params:
            plot_params[key] = item
    if fig is None:
        fig = plt.figure(figsize=figsize)
    g = mcuts.geometry
    nx, ny = g.nx, g.ny
    gs = matplotlib.gridspec.GridSpec(ny, nx, wspace=spacing, hspace=spacing,
                                      left=spacing, right=1.0 - spacing,
                                      bottom=spacing, top=1.0 - spacing)
    if ylim is None:
        low = np.min(uts.min(mcuts))
        high = np.max(uts.max(mcuts))
    else:
        low, high = ylim
    for i, channelname in enumerate(mcuts.channelnames):
        point = rown, coln = g[channelname]
        ax = fig.add_subplot(gs[rown, coln], frameon=False)

        pl = ax.plot(mcuts.samplingtimes(),
            mcuts[:, channelname].samples.get(False), **plot_params)
        ax.tick_params(direction='out', top=False, right=False)
        ax.set_ylim([low, high])
        if (coln == 0) and (rown == ny - 1):
            ax.set_ylabel(ylabel)
            ax.set_yticks(yticks[0])
            ax.set_yticklabels(yticks[1])
            ax.set_xticks(())
        elif (rown == ny - 1) and (coln == nx - 1):
            ax.set_xlabel('Time')
            xticks = list(mcuts.timerange)
            if (xticks[0] < 0.0) and (xticks[1] > 0.0):
                xticks.insert(1, 0.0)
            ax.set_xticks(xticks)
            ax.set_yticks(())
        else:
            ax.set_xticks(())
            ax.set_yticks(())
    fig.canvas.draw()
    return fig


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
    #     samples = self.samples.get(copy=False)
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