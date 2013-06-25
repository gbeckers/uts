import os
from types import GeneratorType

__all__ = ['savemat']

def savemat(s, filename, appendmat=True, format='5', do_compression=False, 
            oned_as=None, overwrite=False):
    '''
        Saves a signal to mat file (Matlab).

        Parameters
        ----------
        s: a ((epoched) multi-channel) time sampled signal
        appendmat: bool, optional
            True to append the .mat extension to the end of the given
            filename, if not already present.
        format : {'5', '4'}, string, optional
            '5' (the default) for MATLAB 5 and up (to 7.2),
            '4' for MATLAB 4 .mat files
        overwrite: bool, optional 
            True to overwrite an existing signal if it has the same name.

    '''
    try:
        import scipy.io
    except:
        raise ImportError(
            "could not import 'scipy.io' module, which is needed " \
            "to write matlab files")
    if os.path.exists(filename):
        if overwrite:
            os.remove(filename)
        else:
            raise IOError('filename exists')
    elif (appendmat and os.path.exists(filename + '.mat')):
        if overwrite:
            os.remove(filename + '.mat')
        else:
            raise IOError('filename exists')
    if s._typeid == 'uts':
        mdict = {'samples': s.samples.get(False), 'fs': s.fs, 'starttime': s.starttime, 
                 'origintime': s.origintime}
    elif s._typeid == 'mcuts':
        mdict = {'samples': s.samples.get(False), 'fs': s.fs, 'starttime': s.starttime, 
                 'origintime': s.origintime, 'channelnames': s.channelnames.copy()}
    elif s._typeid == 'euts':
        mdict = {'samples': s.samples.get(False), 'fs': s.fs, 'starttime': s.starttime, 
                 'origintime': s.origintime, 'starttimes': s.starttimes}
    elif s._typeid == 'emcuts':
        mdict = {'samples': s.samples.get(False), 'fs': s.fs, 'starttime': s.starttime, 
                 'origintime': s.origintime, 'starttimes': s.starttimes.copy(), 
                 'channelnames': s.channelnames.copy()}
    elif isinstance(s, GeneratorType):
        raise NotImplementedError(
            "Cannot save generators to matlab format yet")
    else:
        raise TypeError("signal is not a sampled signal, or generator")
    scipy.io.savemat(file_name=filename, mdict=mdict, appendmat=appendmat,
        format=format, do_compression=do_compression, oned_as=oned_as)