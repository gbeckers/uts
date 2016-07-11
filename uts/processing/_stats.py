from __future__ import absolute_import
import numpy as np
from .helper import unsqueeze


def mad(a, axis=None, c=0.6745):
    """
    Compute the median absolute deviation (mad).

    Parameters
    ----------
    a : array_like
        Array containing numbers whose mad is desired. If `a` is not an
        array, a conversion is attempted.
    c : float, optional
        Scale factor to adjust for asymptotically normal consistency. Defaults 
        to 0.6745.
    axis : {None, int}, optional
        Axis along which the mad is computed. The default is to compute
        the mad of the flattened array.
    
    Returns
    -------
    mad : ndarray or scalar
    
    Notes
    -----
    The mad is the median of the absolute deviations from the median, and, by 
    default, adjust by a factor for asymptotically normal consistency.
    
    Examples
    --------
    >>> a = [0.08418439,  0.28987414, -0.05445002,  1.6779827]
    >>> mad(a)
    0.25524400296515937
    >>> a = [[0.08418439,  0.28987414, -0.05445002,  1.6779827],
    ...      [0.28987414, -0.05445002,  1.6779827, 0.08418439],
    ...      [-0.05445002,  1.6779827, 0.08418439,  0.28987414]]
    >>> mad(a)
    0.25524400296515937
    >>> mad(a, axis=0)
    array([ 0.20553656,  0.51048801,  0.20553656,  0.30495145])
    >>> mad(a, axis=1)
    array([ 0.255244,  0.255244,  0.255244])

    """
 
    a = np.asarray(a, np.float64)
    d = np.median(a, axis=axis)
    if not (axis is None or axis==0):
        unsqueeze(d, axis, a.shape)
    return np.median(np.fabs(a - d) / c, axis=axis)

def moment(distribution, order, axis=-1, startval=0.0, endval=1.0):
    '''
    Calculates moment from a given distribution. If array has more than one
    dimension, the last one is used. Note that the input is the values of the
    distribution, not the actual samples.
    '''
    d = np.asarray(distribution, np.float64)
    variable = np.linspace(startval, endval, d.shape[axis])
    shape = np.ones(len(d.shape), np.int).tolist()
    shape[axis] = d.shape[axis]
    variable.shape = shape
    moment = (d * np.power(variable, order)).sum(axis=axis) / d.sum(axis=axis)
    return moment


def meand(distribution, startval, endval, axis=None):
    if axis is None:
        distribution = np.asarray(distribution).flatten()
        axis = 0
    mean = moment(distribution=distribution, order=1, axis=axis, startval=startval, 
                  endval=endval)
    return mean

def varianced(distribution, startval, endval, axis=None):
    if axis is None:
        distribution = np.asarray(distribution).flatten()
        axis = 0
    variance = moment(distribution=distribution, order=2, axis=axis,startval=startval, 
                  endval=endval)\
             - np.power(moment(distribution=distribution, order=1, axis=axis,
                  startval=startval, endval=endval), 2)
    return variance

def skewnessd(distribution, startval, endval, axis=None):
    if axis is None:
        distribution = np.asarray(distribution).flatten()
        axis = 0
    mean = meand(distribution=distribution, startval=startval, endval=endval,axis=axis)
    skewness = moment(distribution=distribution, order=3, axis=axis, startval=startval-mean,
                      endval=endval-mean) \
             / pow(varianced(distribution=distribution, startval=startval-mean,
                      endval=endval-mean, axis=axis), 1.5)
    return skewness

def kurtosisd(distribution, startval, endval, axis=None):
    if axis is None:
        distribution = np.asarray(distribution).flatten()
        axis = 0
    mean = meand(distribution=distribution, startval=startval, endval=endval, axis=axis)
    kurtosis = moment(distribution=distribution, order=4, axis=axis, startval=startval-mean,
                      endval=endval-mean) \
             / pow(varianced(distribution=distribution, startval=startval-mean,
                      endval=endval-mean, axis=axis), 2.0)
    return kurtosis

if __name__ == "__main__":
    import doctest
    doctest.testmod()

    
    