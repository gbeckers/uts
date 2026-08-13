import numpy as np
cimport numpy as np
from windows import windows
from cpython cimport bool
from numpy.fft import rfft

def fit_frames(int totalsize, int framesize, stepsize=None):
    
    cdef int istepsize
    cdef int nframes
    cdef int newsize
    cdef int remainder

    if stepsize is None:
        stepsize = framesize
    else:
        if ((stepsize % 1) != 0) or (stepsize < 1):
            raise ValueError("invalid stepsize")

    if ((totalsize % 1) != 0) or (totalsize < 1):
        raise ValueError("invalid totalsize")
    if ((framesize % 1) != 0) or (framesize < 1):
        raise ValueError("invalid framesize")
    
    if framesize > totalsize:
        return 0, 0, totalsize

  
    nframes = ((totalsize - framesize) // stepsize) + 1
    newsize = nframes * stepsize + (framesize - stepsize)
    remainder = totalsize - newsize
    
    return nframes, newsize, remainder


FLOATDTYPE = np.float32
ctypedef np.float32_t FLOATDTYPE_t
DOUBLEDTYPE = np.float64
ctypedef np.float64_t DOUBLEDTYPE_t
INTDTYPE = np.int_
ctypedef np.int_t INTDTYPE_t
COMPLEXTYPE = np.complex128
ctypedef np.complex128_t COMPLEX_t

cimport cython

#FIXME we should also take non-float arrays and convert them
# AND fix framesize fftsize naming
def strft(np.ndarray a, int windowsize, int stepsize, str windowtype='gaussian',
          int fftsize=0, bool symmetric=False, bool power=False):

    cdef int newsize
    cdef int remainder
    cdef int offset
    
    assert a.ndim == 1
    
    if fftsize == 0:
        framesize = windowsize
    else:
        framesize = fftsize
    if fftsize < windowsize:
        raise ValueError("fftsize smaller than windowsize")
    
    nwindows, newsize, remainder = fit_frames(totalsize=a.size, framesize=windowsize, stepsize=stepsize)
    if symmetric:
        offset = int(remainder // 2.0)
    else:
        offset = 0
    if a.dtype == FLOATDTYPE:
        if power:
            return _strftf32p(a=a, windowsize=windowsize, nwindows=nwindows,
                        stepsize=stepsize, framesize=framesize, offset=offset, 
                        windowtype=windowtype)
        else:
            return _strftf32(a=a, windowsize=windowsize, nwindows=nwindows,
                        stepsize=stepsize, framesize=framesize, offset=offset, 
                        windowtype=windowtype)
            
    elif a.dtype == DOUBLEDTYPE:
        if power: 
            return _strftf64p(a=a, windowsize=windowsize, nwindows=nwindows,
                        stepsize=stepsize, framesize=framesize, offset=offset,
                        windowtype=windowtype)
        else:
            return _strftf64(a=a, windowsize=windowsize, nwindows=nwindows,
                stepsize=stepsize, framesize=framesize, offset=offset,
                windowtype=windowtype)

    else:
        raise TypeError("array has wrong dtype (%s)" % a.dtype)

@cython.boundscheck(False)
def _strftf32p(np.ndarray[FLOATDTYPE_t, ndim=1] a, int windowsize, int nwindows, int stepsize, 
              int framesize, int offset, str windowtype='gaussian'):
    
    cdef int iwindow
    
    cdef np.ndarray[FLOATDTYPE_t, ndim=1] window = windows[windowtype](windowsize, dtype=FLOATDTYPE)     
    cdef np.ndarray[FLOATDTYPE_t, ndim=2] matrix = np.empty([framesize,nwindows], dtype=FLOATDTYPE)
    for iwindow in range(nwindows):
        matrix[:windowsize,iwindow] = window * a[offset+iwindow*stepsize:offset+iwindow*stepsize+windowsize]
    cdef np.ndarray[COMPLEX_t, ndim=2] even = rfft(matrix,n=framesize, axis=0)
    cdef np.ndarray[DOUBLEDTYPE_t, ndim=2] result = np.absolute(even*even.conjugate()) / windowsize
    return result

@cython.boundscheck(False)            
def _strftf64p(np.ndarray[DOUBLEDTYPE_t, ndim=1] a, int windowsize, int nwindows, int stepsize, 
              int framesize, int offset, str windowtype='gaussian'):
    
    cdef int iwindow
    
    cdef np.ndarray[DOUBLEDTYPE_t, ndim=1] window = windows[windowtype](windowsize, dtype=DOUBLEDTYPE)     
    cdef np.ndarray[DOUBLEDTYPE_t, ndim=2] matrix = np.empty([framesize,nwindows], dtype=DOUBLEDTYPE)
    for iwindow in range(nwindows):
        matrix[:windowsize,iwindow] = window * a[offset+iwindow*stepsize:offset+iwindow*stepsize+windowsize]
    cdef np.ndarray[COMPLEX_t, ndim=2] even = rfft(matrix,n=framesize, axis=0)
    cdef np.ndarray[DOUBLEDTYPE_t, ndim=2] result = np.absolute(even*even.conjugate()) / windowsize
    return result

@cython.boundscheck(False)
def _strftf32(np.ndarray[FLOATDTYPE_t, ndim=1] a, int windowsize, int nwindows, int stepsize, 
              int framesize, int offset, str windowtype='gaussian'):
    
    cdef int iwindow
    
    cdef np.ndarray[FLOATDTYPE_t, ndim=1] window = windows[windowtype](windowsize, dtype=FLOATDTYPE)     
    cdef np.ndarray[FLOATDTYPE_t, ndim=2] matrix = np.empty([framesize,nwindows], dtype=FLOATDTYPE)
    for iwindow in range(nwindows):
        matrix[:windowsize,iwindow] = window * a[offset+iwindow*stepsize:offset+iwindow*stepsize+windowsize]
    cdef np.ndarray[COMPLEX_t, ndim=2] result = rfft(matrix,n=framesize, axis=0) / windowsize
    return result

@cython.boundscheck(False)            
def _strftf64(np.ndarray[DOUBLEDTYPE_t, ndim=1] a, int windowsize, int nwindows, int stepsize, 
              int framesize, int offset, str windowtype='gaussian'):
    
    cdef int iwindow
    
    cdef np.ndarray[DOUBLEDTYPE_t, ndim=1] window = windows[windowtype](windowsize, dtype=DOUBLEDTYPE)     
    cdef np.ndarray[DOUBLEDTYPE_t, ndim=2] matrix = np.empty([framesize,nwindows], dtype=DOUBLEDTYPE)
    for iwindow in range(nwindows):
        matrix[:windowsize,iwindow] = window * a[offset+iwindow*stepsize:offset+iwindow*stepsize+windowsize]
    cdef np.ndarray[COMPLEX_t, ndim=2] result = rfft(matrix,n=framesize, axis=0) / windowsize
    return result

