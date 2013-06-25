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
import mdp
from time import time
import warnings
from scipy.signal import cheby1, firwin, lfilter, remez
from helper import fit_frames
import scipy
from windows import windows
from scipy.weave import converters, inline
from uts3.core.utils import round_nextradix2, round_nearestodd, round_nearesteven, eps



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
        warnings.warn("Problem with ndarray creation forces copy.")
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



def ifdv(s, SAMPLING, N, OVERLAP, sigma=2.0, ZOOMT=1, ZOOMF=3, TL=5, FL=5, 
         symmetric=False, weave=True):
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
    weave: {True, False}
    
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
    nt, nsoundsamples, remainder = fit_frames(nsamples, N, FACTOR)
    
    if symmetric:
        fromsoundsample = remainder // 2.0
        tosoundsample = fromsoundsample + nsoundsamples
    else:
        fromsoundsample = 0
        tosoundsample = nsoundsamples
    
    segmentedsound = segment_axis(a=s[fromsoundsample:tosoundsample],
            length=N, overlap=OVERLAP)
    
    #t=-N/2+1:N/2;
    t = np.arange(-N//2+1,N//2+1)

    #Gaussian and first derivative as windows.

    #sigma=(sigma/1000)*SAMPLING;
    sigma = (sigma/1000.0) * SAMPLING
    #w=exp(-(t/sigma).^2);
    w = np.exp(-np.power((t/sigma),2.0))
    #dw=(w).*((t)/(sigma^2))*-2;
    dw = w * t/(np.power(sigma,2)) * -2
    #q=specgram(s,N,[],w,OVERLAP)+eps; %gaussian windowed spectrogram
    
    q = np.fft.rfft(segmentedsound * w, axis=-1).T + eps
    q2 = np.fft.rfft(segmentedsound * dw, axis=-1).T + eps
    #q,f,t = rspecgram(soundar=s, window=w, noverlap=OVERLAP).T + eps
    #q2=specgram(s,N,[],dw,OVERLAP)+eps; %deriv gaussian windowed spectrogram
    #q2,f,t = rspecgram(soundar=s, window=dw, noverlap=OVERLAP).T + eps

    #[F,T]=size(q);
    F,T= q.shape
    #dx=(q2./q)/(2*pi); %displacement according to the remapping algorithm
    dx=(q2/q)/(2*np.pi) #displacement according to the remapping algorithm

    #sonogram=q;
    sonogram = q.copy()
    #fo=zeros(size(q));
    fo = np.zeros(q.shape)
    #to=zeros(size(q));
    to = np.zeros(q.shape)

    #for(k=1:T) to(:,k)=k; end
    to[:,:] = np.arange(1,T+1)[np.newaxis,:]
    #for(n=1:F) fo(n,:)=(.5*n-1)/(F-1); end
    fo[:,:] = ((0.5 * np.arange(1,F+1)-1)/(F-1))[:,np.newaxis]
    #f_est= ((fo-imag(dx))*N)+1; %calculate frequency displacement factors
    f_est= ((fo-np.imag(dx)) * N) + 1 #calculate frequency displacement factors
    #t_est= to-(pi*sigma*sigma)*real(dx)/FACTOR; %calculate temporal displacement
    t_est= to - (np.pi * sigma * sigma) * np.real(dx) / FACTOR# %calculate temporal displacement
    #print t_est

    #tref=ZOOMT*repmat(1:T,F,1);
    tref = ZOOMT * np.tile(np.arange(1,T+1), (F, 1))
    #fref=ZOOMF*repmat(1:F,T,1)';
    fref = ZOOMF * np.tile(np.arange(1,F+1), (T, 1)).T

    #F=F*ZOOMF;T=T*ZOOMT; %rescale the image by the resolution stretching factors.
    F = F * ZOOMF #rescale the image by the resolution stretching factors.
    T = T * ZOOMT
    #f_e=round(ZOOMF*f_est);
    f_e= np.round(ZOOMF * f_est)#.astype(np.int)
    #t_e=round(ZOOMT*t_est);
    t_e=np.round(ZOOMT * t_est)#.astype(np.int)

    #q(f_e<1 | f_e>F)=0; %set to zero the points that are mapped out of the image
    q[(f_e < 1) | (f_e > F)] = 0  #set to zero the points that are mapped out of the image
    #q(t_e<1 | t_e>T)=0;
    q[(t_e < 1) | (t_e > T)] = 0 

    #f_e(f_e<1)=1; f_e(f_e>F)=F; t_e(t_e<1)=1; t_e(t_e>T)=T;%keep all points in the image
    f_e[f_e < 1] = 1 #keep all points in the image
    f_e[f_e > F] = F
    t_e[t_e < 1] = 1
    t_e[t_e > T] = T
    #q(abs(f_e-fref)>FL)=0; q(abs(t_e-tref)>TL)=0; %remove "stray points"
    q[np.abs(f_e-fref) > FL] = 0 #remove "stray points"
    q[np.abs(t_e-tref) > TL] = 0
    #[a,b]=size(f_e);n=a*b;
    [a,b] = f_e.shape
    n = a * b

    #newq=accumarray([f_e(1:n)' t_e(1:n)'],abs(q(1:n)')); %build the remapped sonogram
    #indices = np.array([f_e.flatten(), t_e.flatten()]).T

    #indices -= 1
    q = q.flatten()

    f_e = f_e.flatten() - 1
    t_e = t_e.flatten() - 1
    newq = np.zeros([f_e.max()+1,t_e.max()+1], dtype=np.float64)
    if weave:
        code = """
             int a, b;
             for (int i = 0; i < n; ++i) {
             a = f_e(i);
             b = t_e(i);
             newq(a, b) += abs(q(i));
             }
             return_val = 0;
             """
        inline(code, ['n', 'q', 'newq','f_e', 't_e'],
               type_converters=converters.blitz, compiler = 'gcc')
    else:
        for i,index in enumerate(zip(f_e,t_e)):
            newq[index] += abs(q[i])
    
    
        #newq = (q[f_e] + q[t_e]).reshape(a,b)
        #newq[f_e,t_e] = q 
        #newq.resize(q.shape)
    
    
        #newq = accumarray(indices , np.abs(q[0:n].T)) #build the remapped sonogram
    
    #%the Frequency axis of the final image is zero to Nyquist freqeuncy, that is
    #%(SAMPLING/2) Hz.

    #ifdgram=abs(flipdim(newq,1)); %flip image, so low frequency is at the bottom.
    #sonogram=abs(flipdim(sonogram,1)); %this is the standard sonogram
    ifdgram = np.abs(np.rot90(newq.T))
    sonogram = np.abs(np.rot90(sonogram.T))
    
    return ifdgram, sonogram

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
    nyquist = fs/2.0
    return firwin(numtaps=Nodd, cutoff=cutoff/nyquist, 
                  width=width/nyquist, window=window)

design_lpfirfilter = lpfirkernel

def hpfirkernel(N, cutoff, fs, width, window='kaiser'):
    kernel = lpfirkernel(N=N, cutoff=cutoff, fs=fs, width=width, window=window)
    kernel = -kernel
    kernel[N // 2] += 1
    return kernel

def bpfirkernel(N, cutoff, fs, width):
    kernela = lpfirkernel(N=N, cutoff=cutoff, fs=fs, width=width)
    kernelb = hpfirkernel(N=N, cutoff=cutoff, fs=fs, width=width)
    kernelc = -(kernela + kernelb)
    kernelc[N//2] = kernelc[n//2] + 1.0

def design_bp_filter(ntaps, lfreq, hfreq, fs, transband=300):
    #print [0, lfreq-transband, lfreq, hfreq, hfreq+transband, fs/2]
    filt = remez(ntaps,[0, lfreq-transband, lfreq, hfreq, hfreq+transband, fs/2],[0,1,0],Hz=fs)
    return filt


def ica(array, outputdim=None, method='fastica', dtype=None, return_node=False, **icaargs):

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
    
def periodogram(s, Sk=None, N=None,  fs=2*np.pi, sides='onesided', normalize=True):
    """
	Takes an N-point periodogram estimate of the PSD function. The 
	number of points N, or a precomputed FFT Sk may be provided. By 
	default, the PSD function returned is normalized so that the 
	integral of the PSD is equal to the mean squared amplitude 
	(mean energy) of s (see Notes).

    Parameters
    ----------
    s : ndarray
        Signal(s) for which to estimate the PSD, time dimension in the last axis
    Sk : ndarray (optional)
        Precomputed FFT of s
    N : int (optional)
        Indicates an N-point FFT where N != s.shape[-1]
    sides : str (optional)        
        Indicates whether to return a one-sided or two-sided PSD
    normalize : boolean (optional)
        Normalizes the PSD

    Returns
    -------
    PSD estimate for each row of s

    Notes
    -----
    setting dw = 2*PI/N, then the integral from -PI, PI (or 0,PI) of PSD/(2PI)
    will be nearly equal to sxx(0), where sxx is the autocorrelation function
    of s(n). By definition, sxx(0) = E{s(n)s*(n)} ~ (s*s.conj()).mean()
    
    The code for this function has been copied from the nitime package.
    
    """
    if Sk is not None:
        N = Sk.shape[-1]
    else:
        N = s.shape[-1] if not N else N
        Sk = np.fft.fft(s, n=N)
    pshape = list(Sk.shape)
    norm = float(s.shape[-1])
    # if the time series is a complex vector, a one sided PSD is invalid..
    # should we check for that?
    if sides=='onesided':
        # putative Nyquist freq
        Fn = N/2 + 1
        # last duplicate freq
        Fl = (N+1)/2
        pshape[-1] = Fn
        P = np.zeros(pshape, 'd')
        freqs = np.linspace(0, fs/2.0, Fn)
        P[...,0] = (Sk[...,0]*Sk[...,0].conj())
        P[...,1:Fl] = 2 * (Sk[...,1:Fl]*Sk[...,1:Fl].conj())
        if Fn > Fl:
            P[...,Fn-1] = (Sk[...,Fn-1]*Sk[...,Fn-1].conj())
    else:
        P = (Sk*Sk.conj()).real
        freqs = np.linspace(0, fs, N, endpoint=False)
    if normalize:
        P /= norm
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

