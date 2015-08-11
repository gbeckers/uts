import numpy as np

def gaussian(size, dtype=np.float64):
    """
    A gaussian windows as is used by the phonetics program Praat.

    Its -3 dB bandwidth is 2*sqrt(6*ln(2))/(pi*size)
    """

    imid = 0.5 * (size + 1)
    edge = np.exp (-12.0)
    window = np.arange(1.,size+1).astype(dtype)

    return (np.exp (-48.0 * (window - imid) * (window - imid) / (size + 1)\
                       / (size + 1)) - edge) / (1 - edge)

def boxcar(size, dtype=np.float64):
    """
    A rectangular window.
    """
    return np.ones([size],dtype=dtype)

def hanning(size, dtype=np.float64):
    return np.hanning(size).astype(dtype)

def hamming(size, dtype=np.float64):
    return np.hamming(size).astype(dtype)
    
def blackman(size, dtype=np.float64):
    return np.blackman(size).astype(dtype)

def bartlett(size, dtype=np.float64):
    return np.bartlett(size).astype(dtype)

# define the window function that can be used
windows = {'hanning': hanning, 'hamming': hamming,
               'blackman': blackman, 'bartlett': bartlett,
               'gaussian': gaussian, 'boxcar': boxcar}
