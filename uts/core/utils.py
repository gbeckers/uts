import numpy as np
from uts.core.check_arg import check_arg, check_int

eps = np.power(2.0, -52.0)

def round_nextradix2(number):
    return int(pow(2, np.ceil(np.log2(number))))


def round_nearestradix2(number):
    return int(pow(2, np.round(np.log2(number))))


def round_nearestodd(number):
    n = int(np.round(number))  # nearest int
    if n % 2 == 0:  # n is even
        if n - number < 0:
            n += 1
        else:
            n -= 1
    return n


def round_nearesteven(number):
    n = int(np.round(number))  # nearest int
    if n % 2 == 1:  # n is odd
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
    return np.all((np.asarray(numbers) % 2.0) == 0.0)


def allodd(numbers):
    return np.all((np.asarray(numbers) % 2.0) == 1.0)

def isradix2(number):
    if not (number & (number - 1)):
        return True
    else:
        return False

def allradix2(numbers):
    numbers = np.asarray(numbers)
    if not (numbers & (numbers- 1)).any():
        return True
    else:
        return False

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



def epochtimetotext(epochtime):
    from time import strftime, localtime

    return strftime('%Y/%m/%d %H:%M:%S', localtime(epochtime))


def timeparams(ntimesamples=None, fs=None, duration=None):
    # we need enough info from duration, fs and ntimesamples
    havents = not (ntimesamples is None)
    havefs = not (fs is None)
    havedur = not (duration is None)
    timeparams = np.array([havents, havefs, havedur])
    if not (timeparams.sum() >= 2):
        raise ValueError(
            "at least 2 values are required for duration, ntimesamples, and fs")
    if havents:
        ntimesamples = check_int(ntimesamples, negative=False)
    if havefs:
        fs = check_arg(fs, 'fs')
    if havedur:
        duration = check_arg(duration, 'duration')
    if timeparams.sum() == 2:
        #  now calculate what's missing
        if havents:
            if havefs:
                duration = ntimesamples / fs
            else:  # have duration

                fs = ntimesamples / duration
        else:  # have duration and have fs
            ntimesamples = fs * duration
            if divmod(ntimesamples, 1.0)[1] != 0.0:
                raise ValueError(
                    "duration and fs do not correspond to integer ntimesamples")
            else:
                ntimesamples = int(ntimesamples)
    return (ntimesamples, fs, duration)
