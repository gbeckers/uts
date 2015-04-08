import numpy as np
import scipy
import warnings
from uts.core.check_arg import check_arg, check_int

defined_integers = (int, long, np.uint8, np.int8, np.uint16,
        np.int16, np.uint32, np.int32, np.uint64, np.int64)

eps = np.power(2.0,-52.0)


class DimensionError(Exception): pass


class LazyProperty(object):
    '''
    Nice trick to avoid calculating the same properties over and over again.

    From Rick Copeland, who desribed it at:
    http://blog.pythonisito.com/2008/08/lazy-descriptors.html
    '''

    def __init__(self, func):
        self._func = func
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__

    def __get__(self, obj, klass=None):
        if obj is None: return None
        result = obj.__dict__[self.__name__] = self._func(obj)
        return result


def iminterpolate(im, newxsize, newysize, xyvalranges, newxyvalranges):
    '''
    Interpolates image (rows,cols,channels)
    '''
    needxinterp = (newxsize != im.shape[0])
    needyinterp = (newysize != im.shape[1])

    if needxinterp or needyinterp: # we need to interpolate image
        xfrom, xto, yfrom, yto = xyvalranges
        newxfrom, newxto, newyfrom, newyto = newxyvalranges
        xsize, ysize = im.shape

        x,y = scipy.ogrid[xfrom:xto:xsize*1j,yfrom:yto:ysize*1j]
        newx,newy = scipy.mgrid[newxfrom:newxto:newxsize*1j,newyfrom:newyto:newysize*1j]
        x0 = x[0,0]
        y0 = y[0,0]
        dx = x[1,0]-x0
        dy = y[0,1]-y0
        ivals = (newx-x0)/dx
        jvals = (newy-y0)/dy
        coords = np.array([ivals,jvals])
        im =  scipy.ndimage.map_coordinates(im, coords)
    return im


def convadjusttime(fs, kernelsize, mode):
    """
    Calculates time shift in a signal that results from a convolving it with a
    kernel.

    Parameters
    ----------
    fs: int
        Sampling rate.
    kernelsize: int
        Size of the convolution kernel
    mode: {'full', 'same', 'valid'}
        Convolution mode (see numpy.convolve for more info)

    Returns
    -------
    A float that represents the time shift in seconds.

    """

    fs = check_arg(fs, 'fs')
    if not mode in ('full', 'same', 'valid'):
        raise ValueError("mode has to be 'full', 'same' or 'valid'")
    kernelsize = check_int(kernelsize, zero=False, negative=False)

    # if mode == 'same' and kernel has an uneven number of samples, then
    # nothing changes; else we need to correct the starttime and origintime

    timeshift = 0.0
    if mode == 'full':
        # we have extra samples at the head and tail
        # the total number of extra samples is kernel.size - 1
        timeshift = -((kernelsize - 1) / fs) / 2.0
    elif mode == 'valid':
        # we have less samples than our original
        # the total number of lost samples is kernel.size - 1
        timeshift = ((kernelsize - 1)/ fs) / 2.0
    else: # mode == 'same'
        # we have the same number of samples, but need a small time shift
        # correction if kernel is even
        if (kernelsize % 2) == 0:
            timeshift = -(1.0 / (2.0 * fs))
    return timeshift


def fit_frames(totalsize, framesize, stepsize=None):

    if ((totalsize % 1) != 0) or (totalsize < 1):
        raise ValueError("invalid totalsize")
    if ((framesize % 1) != 0) or (framesize < 1):
        raise ValueError("invalid framesize")
    
    if framesize > totalsize:
        return 0, 0, totalsize

    if stepsize is None:
        stepsize = framesize
    else:
        if ((stepsize % 1) != 0) or (stepsize < 1):
            raise ValueError("invalid stepsize")

    totalsize = int(totalsize)
    framesize = int(framesize)
    stepsize = int(stepsize)
    
    nframes = ((totalsize - framesize) // stepsize) + 1
    newsize = nframes * stepsize + (framesize - stepsize)
    remainder = totalsize - newsize
    
    return nframes, newsize, remainder


def unsqueeze(data, axis, oldshape):
    """
    unsqueeze a collapsed array
 
    >>> from numpy import mean
    >>> from numpy.random import standard_normal
    >>> x = standard_normal((3,4,5))
    >>> m = mean(x, axis=1)
    >>> m.shape
    (3, 5)
    >>> unsqueeze(m, 1, x.shape)
    >>> m.shape
    (3, 1, 5)
    >>>                       
    """
    
    newshape = list(oldshape)
    newshape[axis] = 1
    return data.reshape(newshape)
