import matplotlib.pyplot as plt
import matplotlib.gridspec
import matplotlib.pyplot as plt
from uts import isuts, ismcuts, iseuts, isemcuts, isspectrum, ispowerspectrum

__all__ = ['image', 'imshow', 'imsave', 'plot']


def plot(s, **kwargs):
    import plots

    if isuts(s):
        return plots.plot_uts(s, **kwargs)
    elif ismcuts(s):
        return plots.plot_mcuts(s, **kwargs)
    elif isemcuts(s):
        return plots.plot_emcuts(s, **kwargs)
    elif isspectrum(s):
        return plots.spectrum_plot(s, **kwargs)
    elif ispowerspectrum(s):
        return plots.plot_powerspectrum(s, **kwargs)
    else:
        raise NotImplementedError(
            "'plot' not implemented for {}".format(type(s)))


def image(s, **kwargs):
    if iseuts(s):
        from plots import euts_image

        return euts_image(s, **kwargs)
    else:
        raise NotImplementedError(
            "'implot' not implemented for {}".format(type(s)))


def imshow(s, image_kwargs=None, imshow_kwargs=None):
    if image_kwargs is None:
        image_kwargs = {}
    if imshow_kwargs is None:
        imshow_kwargs = {}
    if iseuts(s):
        from plots import euts_image
        from matplotlib.pyplot import imshow
        return imshow(image(s, **image_kwargs), **imshow_kwargs)
    else:
        raise NotImplementedError(
            "'imshow' not implemented for {}".format(type(s)))


def imsave(s, **kwargs):
    if iseuts(s):
        from plots import save_euts_image

        return save_euts_image(s, **kwargs)
    else:
        raise NotImplementedError(
            "'imsave' not implemented for {}".format(type(s)))


class GridAxes(object):
    def __init__(self, g, wspace=0.06, hspace=0.06,
                 left=None, right=None,
                 bottom=None, top=None,
                 ylabel=None, xlabel=None,
                 yticks=(), xticks=(), fig=None):

        if left is None:
            left = 0
        if right is None:
            right = 1.0 
        if bottom is None:
            bottom = 0
        if top is None:
            top = 1.0
        if fig is None:
            fig = plt.gcf()
        gs = matplotlib.gridspec.GridSpec(nrows=g.ny, ncols=g.nx,
                                          wspace=wspace, hspace=hspace,
                                          left=left, right=right,
                                          bottom=bottom, top=top)
        nx = g.nx
        ny = g.ny
        axes = {}
        for pointnamerow in g.pointnames:
            for pointname in pointnamerow:
                rown, coln = g[pointname]
                axes[pointname] = ax = fig.add_subplot(gs[rown, coln],
                                                       frameon=False)
                ax.set_xticks(())
                ax.set_yticks(())
                if (coln == 0) and (rown == ny - 1):
                    ax.tick_params(direction='out', top=False, right=False)
                    self.ylabelsubplot = ax
                    ax.set_yticks(yticks)
                if (rown == ny - 1) and (coln == nx - 1):
                    ax.tick_params(direction='out', top=False, right=False)
                    self.xlabelsubplot = ax
                    ax.set_xticks(xticks)

        self.axes = axes
        self._geometry = g
        self._gridspec = gs
        self._fig = fig

        if not ylabel is None:
            self.set_ylabel(ylabel)
        if not xlabel is None:
            self.set_xlabel(xlabel)

    def set_ylabel(self, ylabel):
        self.ylabelsubplot.set_ylabel(ylabel)

    def set_xlabel(self, xlabel):
        self.xlabelsubplot.set_xlabel(xlabel)

    def set_yticks(self, yticks):
        self.ylabelsubplot.set_yticks(yticks)

    def set_xticks(self, xticks):
        self.xlabelsubplot.set_xticks(xticks)

