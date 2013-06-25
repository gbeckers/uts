import numpy as np
cimport numpy as np

from signalprocessing import fit_frames, segment_axis, eps

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
COMPLEXTYPE = np.complex128
ctypedef np.complex128_t COMPLEX_t

def cifdv(np.ndarray s, SAMPLING, N, OVERLAP, sigma=2.0, ZOOMT=1, ZOOMF=3, 
          TL=5, FL=5, symmetric=False):
    """
    Calculates remapped sonograms, as described in Gardner & Magnasco PNAS 2006
    
    This function follows the one provided by Timothy Gardner in a  Matlab 
    script as closely as possible.
    
    Parameters
    ----------
    s: 1-D numpy array
    SAMPLING: float
    N: int
    OVERLAP: int
    sigma: float
    ZOOMT: int
    ZOOMF: int
    TL: int
    FL: int
    symmetric: {True, False}
    
    Reference
    ---------

    Gardner, T.J. and Magnasco, M.O. (2006) Sparse time-frequency 
    representations. Proc. Natl. Acad. Sci. USA, 103(16): 6094-6099.
    doi: 10.1073/pnas.0601707103
    
    """
    #Remapped sonograms, as described in
    #Gardner & Magnasco PNAS 2006
    #
    #For lack of a better name, at the moment, "ifdgram" is the new sonogram
    # 
    # [s] Signal to be analyzed.
    # [SAMPLING] Sampling rate of the signal. 
    # [N] Number of filters in the filter-bank.
    #------
    # [OVERLAP] Number of samples to overlap in each successive window;
    #
    # The number of time points in the final image is proportional to
    # length(s)/(N-OVERLAP);
    # Higher overlap will result in sharper lines.
    #------
    # [sigma] is the temporal resolution of the analysis in milliseconds.
    # N should be larger than 5*SAMPLING*(sigma/1000);
    #
    # Choose sigma small to represent sound in a time-like fashion - as a
    # series of clicks, or sigma large, to represent sound in a frequency-like
    # fasion, as a series of tones. For most signals of interest, intermediate
    # values of sigma are best.
    #-------
    # [ZOOMT] sets the temporal resolution of the final image typically ZOOMT=1.
    #-------
    #[ZOOMF] sets the resultion of final image in Frequency. Res=ZOOMF*(N/2)
    # In contrast to ZOOMT, it is typically useful to set ZOOMF great than one,
    #------
    #
    #[TL] temporal locking window in pixels
    #[FL] frequency locking window in pixels
    #
    #When the remapping moves a pixel by more than TL, or FL, that pixel
    #acquires zero weight. For discussion of the locking window, see
    #Gardner & Magnasco, J. Acoust. Soc. Am. 2005
    #
    #When these parameters are small (order 1) "stray" points are removed and the lines are sharpened.
    #If these parameters are too small, lines become too thin, and appear
    #discontinuous.
    #
    #Typical parameters: SAMPLING 44100, N=1024, OVERLAP=1010, 
    #sigma=2, ZOOMT=1, ZOOMF=3; FL=5;TL=5
    #---
    # Implementation note:
    #The best results will come from calculating an ifdgram for many values of
    #sigma, (.5:.1:3.5) for example, then combining by multiplying together images
    #with neighboring values of sigma, and adding them all together.
    #Rational for this is given in Gardner & Magnasco PNAS 2006.
    #---------------
    #example: Compute an ifdgram of 100ms of white noise
    #s=rand(4000,1)-0.5;
    #[ifdgram,sonogram]=ifdv(s,44100,1024,1020,1,1,1,2,2);
    #colormap(hot)
    #imagesc(log(ifdgram+3));
    #-------
    #Comment: As in the previous example, log scaling of the ifdgram,
    #may be optimal for most sounds.
    
    N = int(N) # for some reason this is important
    
    FACTOR=np.float64(N - OVERLAP)
    nsamples = s.size
    nt, nsoundsamples, remainder = fit_frames(nsamples, N, int(FACTOR))
    if symmetric:
        fromsoundsample = remainder // 2
        tosoundsample = fromsoundsample + nsoundsamples
    else:
        fromsoundsample = 0
        tosoundsample = nsoundsamples
    
    segmentedsound = segment_axis(a=s[fromsoundsample:tosoundsample],
            length=N, overlap=OVERLAP)
    
    #t=-N/2+1:N/2;
    t = np.arange(-N//2+1,N//2+1)

    # Gaussian and first derivative as windows.
    sigma = (sigma/1000.0) * SAMPLING
    w = np.exp(-np.power((t/sigma),2.0))
    dw = w * t/(np.power(sigma,2)) * -2
    
    cdef int F = (segmentedsound.shape[1] // 2) + 1
    cdef int T = segmentedsound.shape[0]
    cdef np.ndarray[COMPLEX_t, ndim=2] q = np.empty([F, T], dtype=COMPLEXTYPE)
    cdef np.ndarray[COMPLEX_t, ndim=2] q2 = np.empty([F, T], dtype=COMPLEXTYPE)
    cdef np.ndarray[COMPLEX_t, ndim=2] dx = np.empty([F, T], dtype=COMPLEXTYPE)

    q = np.fft.rfft(segmentedsound * w, axis=-1).T + eps
    q2 = np.fft.rfft(segmentedsound * dw, axis=-1).T + eps

    dx =(q2/q)/(2*np.pi) # displacement according to the remapping algorithm

    sonogram = q.copy()
    
    cdef np.ndarray[DTYPE_t, ndim=2] fo = np.zeros([F,T], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] to = np.zeros([F,T], dtype=DTYPE)

    to[:,:] = np.arange(1,T+1)[np.newaxis,:]
    fo[:,:] = ((0.5 * np.arange(1,F+1)-1)/(F-1))[:,np.newaxis]

    cdef np.ndarray[DTYPE_t, ndim=2] f_est = ((fo-np.imag(dx)) * N) + 1 # calculate frequency displacement factors
    cdef np.ndarray[DTYPE_t, ndim=2] t_est = to - (np.pi * sigma * sigma) * np.real(dx) / FACTOR # calculate temporal displacement

    cdef np.ndarray[DTYPE_t, ndim=2] tref = ZOOMT * np.tile(np.arange(1,T+1,dtype=DTYPE), (F, 1))
    cdef np.ndarray[DTYPE_t, ndim=2] fref = ZOOMF * np.tile(np.arange(1,F+1,dtype=DTYPE), (T, 1)).T

    F = F * ZOOMF # rescale the image by the resolution stretching factors.
    T = T * ZOOMT
    cdef np.ndarray[DTYPE_t, ndim=2] f_e = np.round(ZOOMF * f_est)
    cdef np.ndarray[DTYPE_t, ndim=2] t_e = np.round(ZOOMT * t_est)

    q[(f_e < 1) | (f_e > F)] = 0  #set to zero the points that are mapped out of the image
    q[(t_e < 1) | (t_e > T)] = 0 

    f_e[f_e < 1] = 1 #keep all points in the image
    f_e[f_e > F] = F
    t_e[t_e < 1] = 1
    t_e[t_e > T] = T

    q[np.abs(f_e-fref) > FL] = 0 #remove "stray points"
    q[np.abs(t_e-tref) > TL] = 0

    cdef int a = f_e.shape[0]
    cdef int b = f_e.shape[1]
    
    f_e -= 1
    t_e -= 1
    cdef np.ndarray[DTYPE_t, ndim=2] newq = np.zeros([f_e.max()+1, t_e.max()+1], dtype=DTYPE)
    #cdef int x
    #cdef int y
    cdef DTYPE_t x
    cdef DTYPE_t y

    for i in range(a):
        for j in range(b):
            x = f_e[i,j]
            y = t_e[i,j]
            newq[x, y] += abs(q[i,j]);
    
    ifdgram = np.abs(np.rot90(newq.T))
    sonogram = np.abs(np.rot90(sonogram.T))
    
    return ifdgram, sonogram
