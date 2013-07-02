import numpy as np
from numpy import arange
from numpy.random import shuffle
cimport numpy as np
cimport cython
from cpython cimport bool

from uts import UniformTimeSeries, isuts, iseuts, ismcuts, isemcuts
from uts.core.check_arg import check_int

FLOATDTYPE = np.float32
ctypedef np.float32_t FLOATDTYPE_t
DOUBLEDTYPE = np.float64
ctypedef np.float64_t DOUBLEDTYPE_t



# def crosscorrelate(a,b, covariance=True, normalized=True):
#     assert a.ndim == b.ndim, "dtypes are not the same"
#     if covariance and not normalized:
#         m = a - np.expand_dims(np.mean(a, -1), -1)
#         n = b - np.expand_dims(np.mean(b, -1), -1)
#     elif normalized:
#         m = a - np.expand_dims(np.mean(a, -1), -1)
#         n = b - np.expand_dims(np.mean(b, -1), -1)
#         m /= np.expand_dims(np.std(m,-1), -1)
#         n /= np.expand_dims(np.std(n,-1), -1)
#     else:
#         n = a
#         m = b
#     minsize = min(n.shape[-1],m.shape[-1])
#     if n.ndim == 1:
#         result = np.correlate(n, m, mode='valid')
#     elif n.ndim == 2:
#         l = n.shape[0]
#         corrlen = abs(n.shape[1]-m.shape[1]) + 1
#         result = np.empty((l, corrlen), dtype=np.float64)
#         samples = []
#         for i in range(l):
#             result[i,:] = np.correlate(n[i], m[i], mode='valid')
#     elif n.ndim == 3:
#         l = n.shape[0]
#         k = n.shape[1]
#         corrlen = abs(n.shape[2]-m.shape[2]) + 1
#         result = np.empty((l, k, corrlen), dtype=np.float64)
#         samples = []
#         for i in range(l):
#             for j in range(k):
#                 result[i,j,:] = np.correlate(n[i,j], m[i,j], mode='valid')
#     else:
#         raise TypeError, 'incorrect number of dimensions'
#     return result / float(minsize)




def crosscovariance_sc(s1, s2, nlag, normalized=True):
    """
    Shuffle-corrected!!
    """

    nlag = check_int(nlag, positive=True, zero=True, negative=False, cast=int)

    if iseuts(s1) and iseuts(s2):
        if not (s1.fs == s2.fs):
            raise ValueError("s1 and s2 have different sampling rates")
        if not (s1.ntimesamples == s2.ntimesamples):
            raise ValueError("s1 and s2 have different number of time samples")
        if not (s1.nepochs == s2.nepochs):
            raise ValueError("s1 and s2 have different dtypes")
        a = s1.transpose(['epoch','time']).samples.get(False).astype(np.float64)
        b = s2.transpose(['epoch','time']).samples.get(False).astype(np.float64)
        origintime = (nlag*2+1)/(s1.fs*2.0)
        return UniformTimeSeries(
                   samples=_crosscovariance_sc(a, b, nlag, normalized),
                   fs=s1.fs, origintime=origintime, copy=False)


@cython.boundscheck(False)
@cython.wraparound(False) 
@cython.cdivision(True)
cdef np.ndarray[DOUBLEDTYPE_t, ndim=1] _crosscovariance_sc(
        np.ndarray[DOUBLEDTYPE_t, ndim=2] a,
        np.ndarray[DOUBLEDTYPE_t, ndim=2] b, 
        int nlag, bool normalize=True):
    """
    It is assumed that s1 and s2 are 2-d float64 arrays, identical in shape, 
    and have time on the last axis.

    """
    # we move b along a; hence b needs to be shorter
    if nlag > 0:
        b = b[:,nlag:-nlag]
    # create a shuffled version of b
    s = arange(b.shape[0])
    shuffle(s)
    sb = b[s]

    cdef int complen = min(a.shape[1], b.shape[1])
    cdef DOUBLEDTYPE_t fcomplen = np.float64(complen)
    cdef int ntau = a.shape[1] - b.shape[1] + 1
    cdef DOUBLEDTYPE_t corr, varra, varrb
    cdef np.ndarray[DOUBLEDTYPE_t, ndim=1] result = np.empty(ntau,
                                                             dtype=DOUBLEDTYPE)
    cdef np.ndarray[DOUBLEDTYPE_t, ndim=2] varsa
    if normalize:
        varb = np.mean(np.mean(np.power(b-np.mean(b,0),2.0),1))
        varsa = np.power(a-np.mean(a,0),2.0)
    for tau in range(ntau):
        ra = a[:,tau:tau+complen]
        corr = np.mean((1/fcomplen)*(np.sum(ra*b,1) - np.sum(ra*sb,1)))
        if normalize:
            varra = np.mean(np.mean(varsa[:,tau:tau+complen],1))
            result[tau] = corr / (varra*varb)**0.5
        else:
            result[tau] = corr
    return result

# Does this work??
def coherency(a,b):
    """
    s1 and s2 should be 2-d with time on the last axis
    """

    # i : trial (epoch)
    # j : time bin
    # tau : time shift in bins
    # t : time bin
    # we assume rb <= ra in duration
    fft = np.fft.rfft
    ifft = np.fft.irfft
    # create a shuffled version of b
    rb = b
    s = np.arange(b.shape[0])
    np.random.shuffle(s)
    sb = b[s]
    np.random.shuffle(s)
    sa = a[s]

    # how many time lags?
    cdef int ntau = a.shape[1] - b.shape[1] + 1
    cdef int complen = min(a.shape[1], b.shape[1])
    cdef int mar = (a.shape[1] - b.shape[1]) / 2
    cdef DOUBLEDTYPE_t fcomplen = np.float64(complen)
    cdef DOUBLEDTYPE_t corr
    # for every time lag
    cdef np.ndarray[DOUBLEDTYPE_t, ndim=1] resultab = np.empty(ntau, dtype=DOUBLEDTYPE)
    cdef np.ndarray[DOUBLEDTYPE_t, ndim=1] resultaa = np.empty(ntau, dtype=DOUBLEDTYPE)
    cdef np.ndarray[DOUBLEDTYPE_t, ndim=1] resultbb = np.empty(ntau, dtype=DOUBLEDTYPE)
    for tau in range(ntau):
        ra = a[:,tau:tau+complen]
        #resultab[tau] = np.mean((1/float(dt))*(1/fcomplen)*(np.sum(ra*rb,1) - np.sum(ra*sb,1)))
    for tau in range(ntau):
        ra1 = a[:,tau:tau+complen]
        ra2 = a[:,mar:-mar]
        ras = sa[:,mar:-mar]
        #resultaa[tau] = np.mean((1/float(dt))*(1/fcomplen)*(np.sum(ra1*ra2,1) - np.sum(ra1*ras,1)))
            

        
        # shuffle-corrected cross-correlogram
        #resultaa[tau] = np.mean((1/float(dt))*(1/fcomplen)*(np.sum(ra*ra,1) - np.sum(ra*ras,1)))
        #resultbb[tau] = np.mean((1/float(dt))*(1/fcomplen)*(np.sum(rb*rb,1) - np.sum(rb*sb,1)))
    print resultab
    print resultaa
    print resultbb
    print fft(resultab)
    print fft(resultaa)
    print fft(resultbb)
    div = (fft(resultaa)*fft(resultbb))**0.5
    div[div==0+0j] = np.power(2.0,-52.0)+0j
    return ifft(fft(resultab)/div)

def crosscorrelate_2d(a, b, normalized=True):
    """We assume correlation axes is 1"""
    assert a.shape[0] == b.shape[0]
    if not a.flags['C_CONTIGUOUS']:
        a = np.ascontiguousarray(a)
    if not b.flags['C_CONTIGUOUS']:
        b = np.ascontiguousarray(b)
    
    cdef int nrows = a.shape[0]
    cdef int lena = a.shape[1]
    cdef int lenb = b.shape[1]
    cdef DOUBLEDTYPE_t flenb = float(lenb)
    cdef int i
    if lenb > lena:
        (a, b) = (b, a)
        (lena, lenb) = (lenb, lena)
    cdef np.ndarray[DOUBLEDTYPE_t, ndim=2] out = np.empty((a.shape[0],abs(lena-lenb)+1),
                                                          dtype=np.float64)
    if normalized:
        for i in range(nrows):
            _crosscorrelate_normalized(a[i], b[i], out[i])
    else:
        for i in range(nrows):
            out[i] = np.correlate(a[i], b[i], mode='valid') / flenb
    return out

def crosscorrelate_1d(np.ndarray[DOUBLEDTYPE_t, ndim=1] a, np.ndarray[DOUBLEDTYPE_t, ndim=1] b,
                      normalized=True):

    cdef int lena = a.shape[0]
    cdef int lenb = b.shape[0]
    if lenb > lena:
        (a, b) = (b, a)
        (lena, lenb) = (lenb, lena)
    cdef np.ndarray[DOUBLEDTYPE_t, ndim=1] out = np.empty(abs(lena-lenb+1), dtype=np.float64)
    
    if not normalized:
        return np.correlate(a, b, mode='valid') / float(lenb)
    else:
        if not a.flags['C_CONTIGUOUS']:
            a = np.ascontiguousarray(a)
        if not b.flags['C_CONTIGUOUS']:
            b = np.ascontiguousarray(b)
        _crosscorrelate_normalized(a,b,out)
    return out
        


@cython.boundscheck(False)
@cython.wraparound(False) 
@cython.cdivision(True)
cdef np.ndarray[DOUBLEDTYPE_t, ndim=1] _crosscorrelate_normalized(
                                            np.ndarray[DOUBLEDTYPE_t, ndim=1, mode='c'] a,
                                            np.ndarray[DOUBLEDTYPE_t, ndim=1, mode='c'] b,
                                            np.ndarray[DOUBLEDTYPE_t, ndim=1, mode='c'] out):
    """Valid normalized crosscorrelation of two signals

    We assume len(a) >= len(b)
    """
    cdef int asize = len(a)
    cdef int bsize = len(b)
    cdef DOUBLEDTYPE_t fbsize = bsize
    cdef int corrlen = asize - bsize + 1
    cdef int tau
    cdef int time
    cdef int i # loop counter
    cdef DOUBLEDTYPE_t meana, meanb
    cdef DOUBLEDTYPE_t stda, stdb
    cdef DOUBLEDTYPE_t s # sum
    cdef DOUBLEDTYPE_t scorr

    # mean b
    s = 0.0
    for i in range(bsize):
        s += b[i]
    meanb = s / fbsize
    # std b
    s = 0.0
    for i in range(bsize):
        s += (b[i] - meanb)**2.0
    stdb = (s / fbsize)**0.5

    # moving std and mean of a
    cdef np.ndarray[DOUBLEDTYPE_t, ndim=1] meana_ar = np.empty(corrlen, np.float64)
    cdef np.ndarray[DOUBLEDTYPE_t, ndim=1] stda_ar = np.empty(corrlen, np.float64)
    _moving_std_and_mean(a, bsize, stda_ar, meana_ar, asize, corrlen)

    for tau in range(corrlen):
        meana = meana_ar[tau]
        stda = stda_ar[tau]
        scorr = 0.0
        for time in range(bsize):
            scorr += (a[tau+time] - meana) * (b[time] - meanb)
        out[tau] =  scorr / (stda * stdb * bsize)


def moving_mean(np.ndarray[DOUBLEDTYPE_t, ndim=1] a, 
                int windowsize,
                np.ndarray[DOUBLEDTYPE_t, ndim=1] out=None):

    cdef int arsize = a.shape[0]
    cdef int outsize = arsize - windowsize + 1
    if (windowsize > arsize) or (windowsize) < 1:
        raise ValueError(
            'windowsize must be larger than 0 but not larger than array size')
    if out is None:
        out = np.empty(arsize - windowsize + 1, dtype=np.float64)
    if not a.flags['C_CONTIGUOUS']:
        a = np.ascontiguousarray(a)
    _moving_mean(a, windowsize, out, arsize, outsize)
    return out

def moving_std_and_mean(np.ndarray[DOUBLEDTYPE_t, ndim=1] a, 
                int windowsize,
                np.ndarray[DOUBLEDTYPE_t, ndim=1] out_mean=None,
                np.ndarray[DOUBLEDTYPE_t, ndim=1] out_std=None):

    cdef int arsize = a.shape[0]
    cdef int outsize = arsize - windowsize + 1
    if out_mean is None:
        out_mean = np.empty(arsize - windowsize + 1, dtype=np.float64)
    if out_std is None:
        out_std = np.empty(arsize - windowsize + 1, dtype=np.float64)
    if not a.flags['C_CONTIGUOUS']:
        a = np.ascontiguousarray(a)
    _moving_std_and_mean(a, windowsize, out_std, out_mean, arsize, outsize)
    return out_std, out_mean



# FIXME this does not work for larger arrays
@cython.boundscheck(False)
@cython.wraparound(False) 
@cython.cdivision(True)
cdef np.ndarray[DOUBLEDTYPE_t, ndim=1] _moving_mean(
                                            np.ndarray[DOUBLEDTYPE_t, ndim=1, mode='c'] a,
                                            int windowsize,
                                            np.ndarray[DOUBLEDTYPE_t, ndim=1, mode='c'] out,
                                            int arsize,
                                            int outsize):

    cdef int i
    cdef DOUBLEDTYPE_t s
    cdef DOUBLEDTYPE_t fwindowsize = np.float64(windowsize)
    
    # first the mean of the first window
    s = 0.0
    for i in range(windowsize):
        s += a[i]
    out[0] = s / fwindowsize
    for i in range(1, outsize):
        s = s - a[i - 1] + a[i + windowsize - 1]
        out[i] = s / fwindowsize


@cython.boundscheck(False)
@cython.wraparound(False) 
@cython.cdivision(True)
cdef np.ndarray[DOUBLEDTYPE_t, ndim=1] _moving_std_and_mean(
                                            np.ndarray[DOUBLEDTYPE_t, ndim=1, mode='c'] a,
                                            int windowsize,
                                            np.ndarray[DOUBLEDTYPE_t, ndim=1, mode='c'] out_std,
                                            np.ndarray[DOUBLEDTYPE_t, ndim=1, mode='c'] out_mean,
                                            int asize,
                                            int outsize):

    cdef int i,j
    cdef DOUBLEDTYPE_t s,m
    cdef DOUBLEDTYPE_t fwindowsize = np.float64(windowsize)
    
    # first moving mean
    _moving_mean(a, windowsize, out_mean, asize, outsize)

    # then moving std    
    for i in range(outsize):
        s = 0.0
        m = out_mean[i]
        for j in range(windowsize):
            s += (a[i+j] - m)**2.0
        out_std[i] = (s / fwindowsize)**0.5
