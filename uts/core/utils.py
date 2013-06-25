import numpy as np
import scipy
import scipy.ndimage
import warnings
from uts3.core.check_arg import check_arg, check_int

eps = np.power(2.0, -52.0)

def round_nextradix2(number):
    return int(pow(2,np.ceil(np.log2(number))))

def round_nearestradix2(number):
    return int(pow(2,np.round(np.log2(number))))

def round_nearestodd(number):
    n = int(np.round(number)) # nearest int
    if n % 2 == 0: # n is even
        if n - number < 0:
            n += 1
        else:
            n -= 1
    return n

def round_nearesteven(number):
    n = int(np.round(number)) # nearest int
    if n % 2 == 1: # n is odd
        if n - number < 0:
            n += 1
        else:
            n -= 1
    return n

def iseven(number):
    return (number % 2.0) == 0.0

def isodd(number):
    return (number % 2.0) == 1.0

def alleven(numbers):
    return np.all((np.asarray(numbers)%2.0) == 0.0)

def allodd(numbers):
    return np.all((np.asarray(numbers)%2.0) == 1.0)

def fit_frames(totalsize, framesize, stepsize=None):

    """
    Calculates how many frames of 'framesize' fit in 'totalsize',
    given a step size of 'stepsize'.

    Parameters
    ----------
    totalsize: int
        Size of total
    framesize: int
        Size of frame
    stepsize: int
        Step size, defaults to framesize (i.e. no overlap)

    Returns a tuple (nframes, newsize, remainder)
    """
    
    if ((totalsize % 1) != 0) or (totalsize < 1):
        raise ValueError("invalid totalsize (%d)" % totalsize)
    if ((framesize % 1) != 0) or (framesize < 1):
        raise ValueError("invalid framesize (%d)" % framesize)
    
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

#FIXME possibility of setting nan to number
def to_image(samples, ranges=None, clip=None, zeroincenter=False, 
                  invert=False, colormap=None, includebar=False, nancolor=None):
        from pylab import cm
        from scipy.stats import scoreatpercentile
        if (not(clip is None)) and (not(ranges is None)):
            raise ValueError("You cannot set ranges and clip parameters at the same time")
        if colormap is None:
            colormap = cm.jet
        samples = np.array(samples,np.float64)
        low = np.nanmin(samples)
        high = np.nanmax(samples)
        if not clip is None:
            low,high = clip
        if not ranges is None:
            low = scoreatpercentile(samples.flatten(),ranges[0])
            high = scoreatpercentile(samples.flatten(),ranges[1])
        np.clip(samples, low, high, samples)  
        if zeroincenter:
            high = max(abs(low),abs(high))
            low = -high
        samples -= low
        high -= low
        samples /= high
        if invert:
            samples -= 1.0
            samples[:] = np.absolute(samples)
        if not nancolor is None:
            nans = np.isnan(samples)
        else:
            nans = np.zeros(samples.shape, '?')
        samples[nans] = 0.0
        im = colormap(samples)
        im[nans] = nancolor
        return im

def epochtimetotext(epochtime):
    from time import strftime, localtime
    return strftime('%Y/%M/%d %H:%m:%S',localtime(epochtime))

def iminterpolate(im, newxsize, newysize, xyvalranges=(0,1,0,1), 
                  newxyvalranges=(0,1,0,1)):
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

def timeparams(ntimesamples=None, fs=None, duration=None):
        #  we need enough info from duration, fs and ntimesamples
    havents = not (ntimesamples is None)
    havefs = not (fs is None)
    havedur = not (duration is None)
    timeparams = np.array([havents, havefs,havedur])
    if not (timeparams.sum()>= 2):
        raise ValueError("zeros needs at least 2 values for duration, ntimesamples, and fs")
    if havents:
        ntimesamples = check_int(ntimesamples, negative=False)
    if havefs:
        fs = check_arg(fs, 'fs')
    if havedur:
        duration = check_arg(duration, 'duration')
    if timeparams.sum()== 2:
        #  now calculate what's missing
        if havents:
            if havefs:
                duration = ntimesamples / fs
            else: # have duration

                fs = ntimesamples / duration
        else: # have duration and have fs
            ntimesamples = fs * duration
            if divmod(ntimesamples,1.0)[1] != 0.0:
                raise ValueError("duration and fs do not correspond to integer ntimesamples")
            else:
                ntimesamples = int(ntimesamples)
    return (ntimesamples, fs, duration)


def smoothmat(mat, sd=(3,3), mode='same'):

    '''smoothmat -- smooth matrix by convolution with a 2-dimensional
    Gaussian kernel. The size of the Gaussian kernel is truncated at 10 times
    the standard deviation or the size of the input matrix, whichever is
    smaller.

    Inputs:
      mat -- matrix to be smoothed
      sd -- tuple with standard deviation (in points) of Gaussian in two
            dimensions: (rows, columns). A value of zero indicates no
            smoothing.
      mode -- a flag indicating the size of the output
              'valid'  (0): The output consists only of those elements that
                              are computed by scaling the larger array with all
                              the values of the smaller array.
              'same'   (1): The output is the same size as the largest input
                              centered with respect to the 'full' output.
              'full'   (2): The output is the full discrete linear convolution

    Outputs:  (out,)

      out -- a 2-dimensional array containing a smoothed version of the input.
             matrix
    '''

    sdy,sdx = sd

    if sdy > 0:
        my = int(np.round(sdy * 10))
        my = min(my, mat.shape[0])
        kerny = scipy.signal.gaussian(my,sdy).reshape(my,1)
    else:
        kerny = np.ones([1,1])

    if sdx > 0:
        mx = int(np.round(sdx * 10))
        mx = min(mx, mat.shape[1])
        kernx = scipy.signal.gaussian(mx,sdx).reshape(1,mx)
    else:
        kernx = np.ones([1,1])

    kern = kerny * kernx

    factor = float(kern.mean() * kern.size)
    return scipy.signal.convolve(mat, kern, mode=mode) / factor