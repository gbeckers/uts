"""
The signalprocessing module contains stuff for signal processing 
operations.

It is intended for basic functions etc., and does not depend on 
other modules in the ds package. Thus, the time sampled signal 
types are not known here. We also don't want extensive type testing 
etc. here. This should be done in signal methods or functions that 
accept signals.

"""


import numpy as np
# from time import time
# import warnings
import scipy.fftpack
from scipy.signal import cheby1, firwin, lfilter, remez
from helper import fit_frames
import scipy
from windows import windows
# from scipy.weave import converters, inline
from uts.core.utils import round_nearestodd, eps



def segment_axis(a, length, overlap=0, axis=None, end='cut', endvalue=0):
    # segment_axis originates from the SciPy website (http://www.scipy.org).
    # I couldn't find the name of the original author
    """
    Generate a new array that chops the given array along the given axis into
    overlapping frames.

    example:
    >>> segment_axis(arange(10), 4, 2)
    array([[0, 1, 2, 3],
           [2, 3, 4, 5],
           [4, 5, 6, 7],
           [6, 7, 8, 9]])

    arguments:
    a       The array to segment
    length  The length of each frame
    overlap The number of array elements by which the frames should overlap
    axis    The axis to operate on; if None, act on the flattened array
    end     What to do with the last frame, if the array is not evenly
            divisible into pieces. Options are:

            'cut'   Simply discard the extra values
            'wrap'  Copy values from the beginning of the array
            'pad'   Pad with a constant value

    endvalue    The value to use for end='pad'

    The array is not copied unless necessary (either because it is
    unevenly strided and being flattened or because end is set to
    'pad' or 'wrap').
    """

    if axis is None:
        a = np.ravel(a) # may copy
        axis = 0

    l = a.shape[axis]

    if overlap>=length:
        raise ValueError("frames cannot overlap by more than 100%")
    if overlap<0 or length<=0:
        raise ValueError(
            "overlap must be nonnegative and length must be positive")

    if l<length or (l-length)%(length-overlap):
        if l>length:
            roundup = length + (1+(l-length)//(length-overlap))*(length-overlap)
            rounddown = length + ((l-length)//(length-overlap))*(length-overlap)
        else:
            roundup = length
            rounddown = 0
        assert rounddown<l<roundup
        assert roundup==rounddown+(length-overlap) or (roundup==length and rounddown==0)
        a = a.swapaxes(-1,axis)

        if end=='cut':
            a = a[...,:rounddown]
        elif end in ['pad','wrap']: # copying will be necessary
            s = list(a.shape)
            s[-1]=roundup
            b = np.empty(s,dtype=a.dtype)
            b[...,:l] = a
            if end=='pad':
                b[...,l:] = endvalue
            elif end=='wrap':
                b[...,l:] = a[...,:roundup-l]
            a = b

        a = a.swapaxes(-1,axis)


    l = a.shape[axis]
    if l==0:
        raise ValueError("Not enough data points to segment array in 'cut' " \
                         "mode; try 'pad' or 'wrap'")
    assert l>=length
    assert (l-length)%(length-overlap) == 0
    n = 1+(l-length)//(length-overlap)
    s = a.strides[axis]
    newshape = a.shape[:axis]+(n,length)+a.shape[axis+1:]
    newstrides = a.strides[:axis]+((length-overlap)*s,s) + a.strides[axis+1:]

    try:
        return np.ndarray.__new__(np.ndarray,strides=newstrides,
                                     shape=newshape,buffer=a,dtype=a.dtype)
    except TypeError:
        #warnings.warn("Problem with ndarray creation forces copy.")
        a = a.copy()
        # Shape doesn't change but strides does
        newstrides = a.strides[:axis]+((length-overlap)*s,s) + a.strides[axis+1:]
        return np.ndarray.__new__(np.ndarray,strides=newstrides,
                                     shape=newshape,buffer=a,dtype=a.dtype)
    
def stft(s, windowsize, windowtype, fftsize, stepsize, symmetric=False, real=False):
    
    FFTDTYPE = np.float64
    if real:
        fft = np.fft.rfft
    else:
        fft = np.fft.fft

    window = windows[windowtype](windowsize)
    noverlap = windowsize - stepsize
    
    # how to window
    nframes, newsize, remainder = fit_frames(s.shape[0], windowsize, stepsize)
    if symmetric:
        fromsample = int(remainder // 2.0)
        tosample = fromsample + int(newsize)
    else:
        fromsample = 0
        tosample = int(newsize)

    # creat input matrix
    segmentmatrix = np.empty([nframes, fftsize], dtype=FFTDTYPE)

    if windowsize == fftsize:
        segmentmatrix[:] =\
                segment_axis(a=s[fromsample:tosample], length=windowsize, overlap=noverlap)
    else:
        segmentmatrix[:,0:windowsize] = \
                segment_axis(a=s[fromsample:tosample], length=windowsize, overlap=noverlap)
        segmentmatrix[:,windowsize:] = 0.0

    # window and fft
    segmentmatrix[:,:windowsize] *= window
    stft = fft(segmentmatrix, axis=-1) / np.sqrt(fftsize)
    firstsamplecenter = float(windowsize + fromsample) / 2.0

    return stft, firstsamplecenter

def specgram(s, windowsize, windowtype, fftsize, stepsize, symmetric=False):
    S, firstsamplecenter = stft(s=s, windowsize=windowsize, windowtype=windowtype, 
                               fftsize=fftsize, stepsize=stepsize, symmetric=symmetric, real=False)

    # putative Nyquist freq
    Fn = fftsize / 2 + 1
    # last duplicate freq
    Fl = (fftsize + 1) / 2
    pshape = list(S.shape)
    pshape[-1] = Fn
    P = np.zeros(pshape, 'd')
    P[..., 0] = (S[..., 0] * S[..., 0].conj()).real
    P[..., 1:Fl] = 2 * (S[..., 1:Fl] * S[..., 1:Fl].conj()).real
    if Fn > Fl: # fftsize is even
        P[..., Fn - 1] = (S[..., Fn - 1] * S[..., Fn - 1].conj()).real
    return P, firstsamplecenter


from numpy import vstack, hstack, eye, ones, zeros, linalg, \
newaxis, r_, flipud, convolve, matrix, array
from scipy.signal import lfilter

def lfilter_zi(b,a):
    #compute the zi state from the filter parameters. see [Gust96].

    #Based on:
    # [Gust96] Fredrik Gustafsson, Determining the initial states in forward-backward
    # filtering, IEEE Transactions on Signal Processing, pp. 988--992, April 1996,
    # Volume 44, Issue 4

    n=max(len(a),len(b))

    zin = (  eye(n-1) - hstack( (-a[1:n,newaxis],
                                 vstack((eye(n-2), zeros(n-2))))))

    zid=  b[1:n] - a[1:n]*b[0]

    zi_matrix=linalg.inv(zin)*(matrix(zid).transpose())
    zi_return=[]

    #convert the result into a regular array (not a matrix)
    for i in range(len(zi_matrix)):
        zi_return.append(float(zi_matrix[i][0]))

    return array(zi_return)


def filtfilt(b,a,x):
    #For now only accepting 1d arrays
    ntaps=max(len(a),len(b))
    edge=ntaps*3

    if x.ndim != 1:
        raise ValueError("Filtfilt is only accepting 1 dimension arrays.")

    #x must be bigger than edge
    if x.size < edge:
        raise ValueError(
            "Input vector needs to be bigger than 3 * max(len(a),len(b).")

    if len(a) < ntaps:
        a=r_[a,zeros(len(b)-len(a))]

    if len(b) < ntaps:
        b=r_[b,zeros(len(a)-len(b))]

    zi=lfilter_zi(b,a)

    #Grow the signal to have edges for stabilizing
    #the filter with inverted replicas of the signal
    s=r_[2*x[0]-x[edge:1:-1],x,2*x[-1]-x[-1:-edge:-1]]
    #in the case of one go we only need one of the extrems
    # both are needed for filtfilt

    (y,zf)=lfilter(b,a,s,-1,zi*s[0])

    (y,zf)=lfilter(b,a,flipud(y),-1,zi*y[-1])

    return flipud(y[edge-1:-edge+1])

def wavelet_gabor(freq, fs, bandwidthfactor):
    freq = float(freq)
    bandwidth = bandwidthfactor/freq

    # let's take 6 cycles in total
    duration = 6/float(freq)
    size = int(np.round(duration *fs))
    t = np.linspace (0., duration, size) - (duration/2.0)
    gaussian = 1.0/np.power(bandwidth*bandwidth*np.pi,0.25) \
      * np.exp(-(t*t)/(2*bandwidth*bandwidth))

    return gaussian * np.exp(1j*2*np.pi*freq*t)

def window_gaussian(M, sigma):
    sigma=float(sigma)
    M = int(np.round(M))
    n = np.linspace(-(M-1)/2.0, (M-1)/2.0, M)
    return np.exp(-n*n / (2 * sigma * sigma))

def decimate_signal(x, q, n=None, ftype='iir', axis=-1):
    """downsample the signal x by an integer factor q, using an order n filter

        By default, an order 8 Chebyshev type I filter is used or a 30 point FIR
        filter with hamming window if ftype is 'fir'.

        (port to python of the GNU Octave function decimate.)

        Inputs:
            x -- the signal to be downsampled (N-dimensional array)
            q -- the downsampling factor
            n -- order of the filter (1 less than the length of the filter for a
                 'fir' filter)
            ftype -- type of the filter; can be 'iir' or 'fir'
            axis -- the axis along which the filter should be applied

        Outputs:
            y -- the downsampled signal

       """

    if type(q) != type(1):
        raise TypeError("q should be an integer")

    if n is None:
        if ftype == 'fir':
            n = 30
        else:
            n = 8
    if ftype == 'fir':
        b = firwin(n+1, 1./q, window='hamming')
        if x.ndim != 1:
            y = lfilter(b, 1., x, axis=axis)
        else:
            y = filtfilt(b, 1., x)

    else:
        (b, a) = cheby1(n, 0.05, 0.8/q)
        if x.ndim != 1:
            y = lfilter(b, a, x, axis=axis)
        else:
            y = filtfilt(b, a, x)

    return y.swapaxes(0,axis)[::q].swapaxes(0,axis)



def lpfirkernel(N, cutoff, fs, width, window='kaiser'):
    # we want odd filter lengths; results in nice convolution timing and allows
    # for spectral inversion
    Nodd = round_nearestodd(N)
    if Nodd != N:
        raise ValueError("only odd kernel lengths allowed")
    nyquist = fs/2.
    if not width is None:
        width = width/nyquist
    return firwin(numtaps=Nodd, cutoff=cutoff/nyquist, 
                  width=width, window=window)

design_lpfirfilter = lpfirkernel

def hpfirkernel(N, cutoff, fs, width, window='kaiser'):
    kernel = lpfirkernel(N=N, cutoff=cutoff, fs=fs, width=width, window=window)
    kernel = -kernel
    kernel[N // 2] += 1
    return kernel

def bpfirkernel(N, freqs, fs, width, window='kaiser'):
    cutofflow, cutoffhigh = freqs
    kernela = lpfirkernel(N=N, cutoff=cutofflow, fs=fs, width=width, window=window)
    kernelb = hpfirkernel(N=N, cutoff=cutoffhigh, fs=fs, width=width, window=window)
    kernelc = -(kernela + kernelb)
    kernelc[N//2] = kernelc[N//2] + 1.0
    return kernelc

def design_bp_filter(ntaps, lfreq, hfreq, fs, transband=300):
    #print [0, lfreq-transband, lfreq, hfreq, hfreq+transband, fs/2]
    filt = remez(ntaps,[0, lfreq-transband, lfreq, hfreq, hfreq+transband, fs/2],[0,1,0],Hz=fs)
    return filt


def ica(array, outputdim=None, method='fastica', dtype=None, return_node=False, **icaargs):

    import mdp

    assert method in ('fastica', 'cubica', 'pca', 'jade', 'tdsep')#, 'isfa', 'sfa2', 'xsfa')
    assert ~np.any(np.isnan(array)), \
           "your array contains nan's; remove them first with the exclude_nan method"
    ## if outputdim is None:
        ## outputdim = self.nsites
    if dtype is not None:
        #        dim1 = self.nsamplespererp
        matrix = array.astype(dtype)
    else:
        matrix = array

    # ica
    if method == 'fastica':
        icanode = mdp.nodes.FastICANode(whitened=False, white_comp=outputdim,\
                                        white_parm={'svd': True}, dtype=dtype, **icaargs)
    elif method == 'cubica':
        icanode = mdp.nodes.CuBICANode(whitened=False, white_comp=outputdim,\
                                       white_parm={'svd': True}, dtype=dtype, **icaargs)
    elif method == 'isfa':
        icanode = mdp.nodes.ISFANode(whitened=False, white_comp=outputdim,
                                     white_parm={'svd': True}, dtype=dtype, **icaargs)
    elif method == 'sfa2':
        icanode = mdp.nodes.SFA2Node(output_dim=outputdim, dtype=dtype, **icaargs)
    elif method == 'jade':
        icanode = mdp.nodes.JADENode(whitened=False, white_comp=outputdim,
                                     white_parm={'svd': True}, dtype=dtype, **icaargs)
    elif method == 'pca':
        icanode = mdp.nodes.PCANode(output_dim=outputdim, svd=True, dtype=dtype, **icaargs)
    elif method == 'tdsep':
        icanode = mdp.nodes.TDSEPNode(whitened=False, white_comp=outputdim,
                                      dtype=dtype, **icaargs)
    elif method == 'xsfa':
        icanode = mdp.nodes.XSFANode(output_dim=outputdim,
                                      dtype=dtype, **icaargs)
    
    icanode.train(matrix)
    #mat = icanode.execute(matrix)
    return icanode.get_projmatrix(), icanode.get_recmatrix()

def sfa(array, outputdim=None, method='fastica', dtype=None, return_node=False, **sfaargs):

    import mdp
    
    assert method in ('isfa', 'sfa2','xsfa')
    assert ~np.any(np.isnan(array)), \
           "your array contains nan's; remove them first with the exclude_nan method"
    if dtype is not None:
        #        dim1 = self.nsamplespererp
        matrix = array.astype(dtype)
    else:
        matrix = array

    if method == 'isfa':
        sfanode = mdp.nodes.ISFANode(whitened=False, white_comp=outputdim,
                                     white_parm={'svd': True}, dtype=dtype, **sfaargs)
    elif method == 'sfa2':
        sfanode = mdp.nodes.SFA2Node(output_dim=outputdim, dtype=dtype, **sfaargs)
    elif method == 'xsfa':
        sfanode = mdp.nodes.XSFANode(dtype=dtype, output_dim=outputdim, **sfaargs)    
    sfanode.train(matrix)
    mat = sfanode.execute(matrix)
    if return_node:
        return mat, sfanode
    else:
        return mat
    
def iminterpolate(im, newxsize, newysize, xyvalranges, newxyvalranges):
    '''
    Interpolates image (rows,cols,channels)
    '''
    needxinterp = (newxsize != im.shape[0])
    needyinterp = (newysize != im.shape[1])
    # print newxsize, newysize, xyvalranges, newxyvalranges, needxinterp, needyinterp
    
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
    
def periodogram(s, fs=2 * np.pi, Sk=None, fftsize=None,
                sides='default', normalize=True):

    """Takes an N-point periodogram estimate of the PSD function. The
    number of points N, or a precomputed FFT Sk may be provided. By default,
    the PSD function returned is normalized so that the integral of the PSD
    is equal to the mean squared amplitude (mean energy) of s (see Notes).

    Parameters
    ----------
    s : ndarray
    Signal(s) for which to estimate the PSD, time dimension in the last
    axis

    Fs : float (optional)
    The sampling rate. Defaults to 2*pi

    Sk : ndarray (optional)
    Precomputed FFT of s

    N : int (optional)
    Indicates an N-point FFT where N != s.shape[-1]

    sides : str (optional) [ 'default' | 'onesided' | 'twosided' ]
    This determines which sides of the spectrum to return.
    For complex-valued inputs, the default is two-sided, for real-valued
    inputs, default is one-sided Indicates whether to return a one-sided
    or two-sided

    PSD normalize : boolean (optional, default=True) Normalizes the PSD

    Returns
    -------
    (f, psd) : tuple
    f: The central frequencies for the frequency bands
    PSD estimate for each row of s

    The code for this function has been copied from the nitime package.
   

    """
    if Sk is not None:
        fftsize = Sk.shape[-1]
    else:
        fftsize = s.shape[-1] if not fftsize else fftsize
        Sk = scipy.fftpack.fft(s, n=fftsize)
    pshape = list(Sk.shape)

    # if the time series is a complex vector, a one sided PSD is invalid:
    if (sides == 'default' and np.iscomplexobj(s)) or sides == 'twosided':
        sides = 'twosided'
    elif sides in ('default', 'onesided'):
        sides = 'onesided'

    if sides == 'onesided':
        # putative Nyquist freq
        Fn = fftsize / 2 + 1
        # last duplicate freq
        Fl = (fftsize + 1) / 2
        pshape[-1] = Fn
        P = np.zeros(pshape, 'd')
        freqs = np.linspace(0, fs / 2, Fn)
        P[..., 0] = (Sk[..., 0] * Sk[..., 0].conj()).real
        P[..., 1:Fl] = 2 * (Sk[..., 1:Fl] * Sk[..., 1:Fl].conj()).real
        if Fn > Fl:
            P[..., Fn - 1] = (Sk[..., Fn - 1] * Sk[..., Fn - 1].conj()).real
    else:
        P = (Sk * Sk.conj()).real
        freqs = np.linspace(0, fs, fftsize, endpoint=False)
    if normalize:
        P /= (fs * s.shape[-1])
    return freqs, P

def nDPSS(fs, N, BW):
    """
    Given a sampling frequency, number of samples, and an approximate
    DPSS window bandwidth, return the number of DPSSs to use in a
    multi-taper PSD estimate: K = N*BW/fs = BW/f0
    Also returns the true bandwidth.

    Paramters
    ---------
    fs : float
        sampling frequency
    N : int
        sequence length
    BW : float
        window bandwidth to match

    Returns
    -------
    K, true_BW : int, float
        The optimal number of DPSS windows to use in PSD estimation, and
        the true bandwidth of the windows.

    Notes
    -----
	The bandwidth parameter reflects the length of the interval 
	[-BW/2, BW/2].
    
    The code for this function has been copied from the nitime package.

    """
    K = int(N*BW/fs + 0.5)
    true_BW = fs*K/N
    return K, true_BW

def DPSS_windows(N, W, Kmax):
    """
    Returns the first Kmax-1 Discrete Prolate Spheroidal Sequences for
    a given frequency-spacing multiple W and sequence length N. 

    Paramters
    ---------
    N : int
        sequence length
    W : float
        half bandwidth corresponding to 2W = Kmax*f0 = (Kmax/T)
    Kmax : int
        number of DPSS windows to return is Kmax-1

    Returns
    -------
    v : ndarray
        an array of DPSS windows shaped (Kmax-1, N)

    Notes
    -----
    Tridiagonal form of DPSS calculation from:

    Slepian, D. Prolate spheroidal wave functions, Fourier analysis, and
    uncertainty V: The discrete case. Bell System Technical Journal,
    Volume 57 (1978), 1371430
    
    The code for this function has been copied from the nitime package.
    
    """
    
    # here we want to set up an optimization problem to find a sequence
    # whose energy is maximally concentrated within band [-W,W].
    # Thus, the measure lambda(T,W) is the ratio between the energy within
    # that band, and the total energy. This leads to the eigen-system
    # (A - (l1)I)v = 0, where the eigenvector corresponding to the largest
    # eigenvalue is the sequence with maximally concentrated energy. The
    # collection of eigenvectors of this system are called Slepian sequences,
    # or discrete prolate spheroidal sequences (DPSS). Only the first K-1,
    # K = 2NW orders of DPSS will exhibit good spectral concentration
    # [see http://en.wikipedia.org/wiki/Spectral_concentration_problem]
    
    # Here I set up an alternative symmetric tri-diagonal eigenvalue problem
    # such that
    # (B - (l2)I)v = 0, and v are our DPSS (but eigenvalues l2 != l1)
    # the main diagonal = ([N-1-2*t]/2)**2 cos(2PIW), t=[0,1,2,...,N-1]
    # and the first off-diangonal = t(N-t)/2, t=[1,2,...,N-1]
    # [see Percival and Walden, 1993]
    ab = np.zeros((2,N), 'd')
    nidx = np.arange(N)
    ab[0,1:] = nidx[1:]*(N-nidx[1:])/2.
    ab[1] = ((N-1-2*nidx)/2.)**2 * np.cos(2*np.pi*W)
    # only calculate the highest Kmax-1 eigenvectors
    l,v = scipy.linalg.eig_banded(ab, select='i', select_range=(N+1-Kmax, N-1))
    return v.transpose()[::-1]


def multi_taper_psd(s, BW=None, fs=2*np.pi, sides='onesided'):
    """
    Returns an estimate of the PSD function of s using the multitaper
    method. If BW and fs are not specified by the user, a bandwidth of 4
    times the fundamental frequency, corresponding to K = 8.

    Parameters
    ----------
    s : ndarray
        An array of sampled random processes, where the time axis is
        assumed to be on the last axis
    BW : float (optional)
        The bandwidth of the windowing function will determine the number
        tapers to use. Normal values are in the range [3/2,5] * 2f0, where
        f0 is the fundamental frequency of an N-length sequence.
        This parameters represents trade-off between frequency resolution (BW)
        and variance reduction (number of tapers).
    fs : float (optional)
        The sampling frequency
    sides : str (optional)
        Indicates whether to return a one-sided or two-sided PSD

    Returns
    -------
    (freqs, psd_est) : ndarrays
        The estimatated PSD and the frequency points vector

    Note
    ----
    The code for this function has been copied from the nitime package.

    """
    # have last axis be time series for now
    N = s.shape[-1]

    # choose bw 2W to be a small multiple of the fundamental freq f0=fs/N
    if not BW:
        W = 4 * fs / N # 2W = 4 * 2f0
        Kmax = 8
    else:
        Kmax, true_BW = nDPSS(fs, N, BW)
        W = true_BW/2.

    v = DPSS_windows(N, W, Kmax)

    sig_sl = [slice(None)]*len(s.shape)
    sig_sl.insert(-1, np.newaxis)

    # tapers.shape is (..., Kmax-1, N)
    tapers = s[sig_sl] * v
    # don't normalize the periodograms by 1/N as normal.. since the taper
    # windows are orthonormal, they effectively scale the signal by 1/N
    freqs,tapers_sdf = periodogram(tapers, fs=fs, sides=sides, normalize=False)
    psd_est = tapers_sdf.mean(axis=-2)
    return freqs, psd_est

