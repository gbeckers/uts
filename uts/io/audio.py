import warnings
import numpy as np
from contextlib import contextmanager, closing
from ..core.uniformtime import UniformTimeSeries, MultiChannelUniformTimeSeries, isuts, ismcuts
from ..core.utils import fit_frames
from ..core.check_arg import check_arg
from ..processing.stats import extremes
from ..processing.signalprocessing import segment_axis



__all__ = ['load_audiofile', 'iter_load_audiofile', 'save_audiofile', 'save_wav', 'load_wav']


def load_audiofile(filename, dtype=np.float32, starttime=0.0, correct_audiolabbug=True):
    """
    reads audio files

    Uses the scikit audiolab, which is a wrapper for libsndfile from 
    Erik Castro de Lopo. This function supports all formats supported by 
    libsndfile, including wav, aiff, ircam files, and flac. Supportded formats 
    are listed by the supportedaudioformats() function.

    Parameters:
    -----------
    filename: the name of the file to be read.
    starttime: defines the start of the sound in seconds since epoch;
        defaults to 0.0
    dtype: numeric format of returned sound samples; defaults to numpy.float32.

    Returns:
    --------
    a Sound object.

    """
    import scikits.audiolab as audiolab
    a = audiolab.Sndfile(filename, 'r')
    nchannels = a.channels
    fs = a.samplerate
    nframes = a.nframes
    data = a.read_frames(nframes, dtype=dtype)
    if correct_audiolabbug and (nchannels > 1):
        warnings.warn('correcting audiolab interleaving bug')
        data = segment_axis(data.T, nchannels)
    a.close()
    if nchannels == 1:
        s = UniformTimeSeries(samples=data, fs=float(fs), starttime=starttime)
    else:
        s = MultiChannelUniformTimeSeries(samples=data, fs=float(fs),starttime=starttime)
    return s

def iter_load_audiofile(filename, readsize=1024**2, stepsize=None, dtype=np.float32,
                        correct_audiolabbug=True):
    """
    iter reads audio files

    Uses the scikit audiolab, which is a wrapper for libsndfile from 
    Erik Castro de Lopo. This function supports all formats supported by 
    libsndfile, including wav, aiff, ircam files, and flac. Supportded formats 
    are listed by the supportedaudioformats() function.

    Parameters:
    -----------
    filename: the name of the file to be read.
    dtype: numeric format of returned sound samples; defaults to numpy.float32.

    Returns:
    --------
    a Sound object.

    """
    import scikits.audiolab as audiolab
    import sys
    a = audiolab.Sndfile(filename, 'r')
    nchannels = a.channels
    fs = float(a.samplerate)
    nframes = a.nframes
    if stepsize is None:
        stepsize = readsize
    nloops, newsize, remainder = fit_frames(nframes, readsize, stepsize=stepsize)
    startsample = 0
    if nchannels == 1:
        data = np.empty(readsize, dtype=dtype)
    else:
        data = np.empty((readsize, nchannels), dtype=dtype)
    for i in range(nloops):
        samples = a.read_frames(readsize, dtype=dtype)
        if correct_audiolabbug and (nchannels > 1):
            warnings.warn('correcting audiolab interleaving bug')
            samples = segment_axis(samples.T, nchannels)
        if (i == 0) or (stepsize==readsize):   
            data[:] = samples
        else:
            data[:-stepsize] = data[stepsize:]
            data[-stepsize:] = samples
        starttime = startsample/fs
        if nchannels == 1:
            yield UniformTimeSeries(samples=data, fs=float(fs),starttime=starttime)
        else:
            yield MultiChannelUniformTimeSeries(samples=data, fs=float(fs),starttime=starttime,
                                                channelaxis=1)
        startsample += (data.size / nchannels)
    if remainder > 0:
        samples = a.read_frames(remainder, dtype=dtype)
        if correct_audiolabbug and (nchannels > 1):
            samples = segment_axis(samples.T, nchannels)
        if (stepsize==readsize):
            data = samples
        else:
            data[:-stepsize] = data[stepsize:]
            data[-stepsize:-(stepsize-remainder)] = samples
            data = data[:-(stepsize-remainder)]
        if nchannels == 1:
            yield UniformTimeSeries(samples=data, fs=float(fs), starttime=starttime)
        else:
            yield MultiChannelUniformTimeSeries(samples=data, fs=float(fs), starttime=starttime,
                                                channelaxis=1)
    a.close()
    

# FIXME allow specific channels
def save_audiofile(tss, filename, fmt=('flac', 'pcm24', 'file'),
                   writesize=1024**2, start=None, end=None,
                   overwrite=False, maxamp=None, fs=None):
    import os.path
    import scikits.audiolab as audiolab
    #FIXME check if tss is tss
    fmt = audiolab.Format(*fmt)
    if os.path.exists(filename) and not overwrite:
        raise IOError("file exists, use overwrite parameter if desired")
    if 'channel' in tss.samples.axes:
        nchannels = tss.nchannels
    else:
        nchannels = 1
    if fs is None:
        fs = tss.fs
    if not maxamp is None:
        mmin, mmax, mminindex, mmaxindex = extremes(tss, start=start, end=end)
        #FIXME do this only for selected channels
        m = max(abs(min(np.array(mmin, ndmin=1))),abs(max(np.array(mmax, ndmin=1))))
        ampfactor = maxamp / float(m)
    else:
        ampfactor = None
    if fmt.file_format == 'flac':
        fs = int(np.round(fs))
        warnings.warn("flac only supports integer sampling rate => will round to %d"%fs)
    f = audiolab.Sndfile(filename, 'w', fmt, nchannels, fs)
    for frame in tss.iter_frames(framesize=writesize/nchannels,
                                 startindex=start, endindex=end):
        samples = frame.samples.get(copy=False)
        if ampfactor:
            samples *= ampfactor
        if np.any((samples >= 1.0) | (samples <= -1.0)):
            warnings.warn('not all samples are > -1.0 and < 1.0')
        f.write_frames(samples)
    f.close()
    
#def write_csv(tss, filename, overwrite=False):
#    
#    def _write_file(ar, filename):
#        if os.path.exists(filename) and not overwrite:
#            raise IOError, "file exists, use overwrite parameter if desired"
#        numpy.savetxt(filename, ar,delimeter=',')
        

@contextmanager
def _open_wavefile(filename, mode='r'):
    import wave
    with closing(wave.open(filename, mode=mode)) as f:
        yield f
    
def load_wav(filename, dtype=np.int16, starttime=0.0):
    with _open_wavefile(filename=filename, mode='r') as f:
        samplewidth = f.getsampwidth()
        nframes = f.getnframes()
        nchannels = f.getnchannels()
        fs = f.getframerate()
        conv_code, audiotofloatfactor = packing_code(samplewidth)
        data = np.frombuffer(f.readframes(f.getnframes()), dtype=conv_code)
        if np.issubdtype(dtype, np.float_):
            data = (data / audiotofloatfactor).astype(dtype)
        if nchannels == 1:
            return UniformTimeSeries(samples=data, fs=float(fs), starttime=starttime)
        else:
            return MultiChannelUniformTimeSeries(samples=data.reshape(-1, nchannels), fs=float(fs), starttime=starttime, channelaxis=1)

# def reset_fs_wavefile(filename, newfs):
#     with _open_wavefile(filename, mode='a') as f:
#         f.setframerate(newfs)


# FIXME make multichannel capability
def save_wav(tss, filename, override_fs=None,
                   writesize=1024**2, start=None, end=None,
                   overwrite=False, maxamp=None):
    import os.path
    
    isuts(tss)
    if os.path.exists(filename) and not overwrite:
        raise IOError("file exists, use overwrite parameter if desired")
    if not maxamp is None:
        mmin, mmax, mminindex, mmaxindex = extremes(tss, start=start, end=end)
        m = max(abs(min(np.array(mmin, ndmin=1))),abs(max(np.array(mmax, ndmin=1))))
        ampfactor = maxamp / float(m)
    else:
        ampfactor = None
    if not (override_fs is None):
        fs = check_arg(override_fs, 'fs')
    else:
        fs = tss.fs
    if (fs % 1) >  0:
        warnings.warn("wav only supports integer sampling rate => will round %d"%fs)
        fs = np.round(tss.fs)
    fs = int(fs)
    
    with _open_wavefile(filename,'w') as f:
        f.setnchannels(1)
        f.setframerate(fs)
        f.setsampwidth(2)
        floattoaudiofactor = packing_code(2)[1]
        for frame in tss.iter_frames(framesize=writesize,
                                     startindex=start, endindex=end):
            samples = frame.samples.get(copy=False)
            if np.issubdtype(samples.dtype, np.float_):
                if np.any((samples >= 1.0) | (samples <= -1.0)):
                    warnings.warn('not all samples are > -1.0 and < 1.0')
                samples *= floattoaudiofactor
            if ampfactor:
                samples *= ampfactor
            f.writeframes(samples.astype('h').tostring())


#FIXME now only single channel

def singlechannelaudio_to_multichannelaudio(filenames, newfilename):
    import scikits.audiolab as audiolab
    if not type(filenames) in (list, tuple):
        raise ValueError('filenames parameter not a list or tuple')
    if not len(filenames) > 1:
        raise ValueError('filenames parameter does not contain multiple names')
    sndfiles = []
    for filename in filenames:
        sndfiles.append(audiolab.Sndfile(filename, 'r'))
    for sndfile in sndfiles[1:]:
        if not sndfile.channels == 1:
            raise ValueError('not all files are single channel')
        if not sndfile.samplerate == sndfiles[0].samplerate:
            raise ValueError('not all sampling rates are equal')
        if not sndfile.nframes == sndfiles[0].nframes:
            raise ValueError('not all durations are equal')
    nchannels = len(sndfiles)
    newsndfile = audiolab.Sndfile(newfilename, 'w', channels=nchannels,
                                  samplerate=sndfiles[0].samplerate,
                                  format=sndfiles[0].format)
    dtype = np.float32
    nframes = sndfiles[0].nframes
    if nframes > 1024**2:
        readsize = 1024**2
    else:
        readsize = nframes
    nloops, newsize, remainder = fit_frames(nframes, readsize, stepsize=readsize)
    startsample = 0
    data = np.empty([readsize,nchannels], dtype=dtype)
    for i in range(nloops):
        for j,sndfile in enumerate(sndfiles):
            data[:,j] = sndfile.read_frames(readsize)
        newsndfile.write_frames(data)
    if remainder > 0:
        for i,sndfile in enumerate(sndfiles):
            data[:remainder,i]= sndfile.read_frames(remainder, dtype=dtype)
        newsndfile.write_frames(data[:remainder])
    for sndfile in sndfiles:
        sndfile.close()
        newsndfile.close()
            
    
    
    

def convert_audiotoh5(filename, h5group, h5arrayname, origintime=0.0,
                      starttime=0.0, channelaxis=1, overwrite=False, 
                      extensible=False, expectedrows=None, filters=None, 
                      chunkshape=None, attrs={}, readsize=4096):
    
    from uts.io.hdf5 import loadh5

    import scikits.audiolab as audiolab
    a = audiolab.Sndfile(filename, 'r')
    nchannels = a.channels
    fs = a.samplerate
    nframes = a.nframes
    enc = a.encoding
    dtype = np.float32
    if nchannels == 1:
        arnode = _newh5uts(h5group=h5group, name=h5arrayname, dtype=dtype, 
                        ntimesamples=nframes, fs=float(fs), origintime=origintime, 
                        starttime=starttime, dflt=0.0, overwrite=overwrite, 
                        extensible=extensible, expectedrows=expectedrows, 
                        filters=filters, chunkshape=chunkshape, attrs=attrs)
    else:
        raise ValueError("multichannel not supported yet")
    
    nloops, newsize, remainder = fit_frames(nframes, readsize, stepsize=None)
    start = 0
    for i in range(nloops):
        samples = a.read_frames(readsize, dtype=dtype)
        arnode[start:start+samples.size] = samples
        start += samples.size
    if remainder > 0:
        samples = a.read_frames(remainder, dtype=dtype)
        arnode[start:start+samples.size] = samples
        
    arnode.flush()
    return loadh5(arnode, inmemory=False)


def packing_code(samplewidth):
    if samplewidth == 1:            # 8 bits are unsigned, 16 & 32 signed
        return 'B', 128.0        # unsiged 8 bits
    elif samplewidth == 2:
        return 'h', 32768.0        # signed 16 bits
    elif samplewidth == 4:
        return 'i',  32768.0 * 65536.0    # signed 32 bits
    raise('WaveIO Packing Error, Not able to parse %s bytes' % samplewidth)
    