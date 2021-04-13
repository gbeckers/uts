import numpy as np
import pyedflib

from ..core.uniformtime import MultiChannelUniformTimeSeries

__all__ = ['load_edf', 'read_edfinfo']


def read_edfinfo(filepath):
    with pyedflib.EdfReader(str(filepath)) as f:
        return {
            'admincode': f.getAdmincode(),
            'birthdate': f.getBirthdate(),
            'digitalmaximum': f.getDigitalMaximum(),
            'digitalminimum': f.getDigitalMinimum(),
            'fileduration': f.getFileDuration(),
            'gender': f.getGender(),
            'equipment': f.getEquipment(),
            'nsamples': f.getNSamples(),
            'nsignals': f.signals_in_file,
            'nbits_recorded': np.log2(f.getDigitalMaximum()-f.getDigitalMinimum()+1).astype('int'),
            'patientadditional': f.getPatientAdditional(),
            'patientcode': f.getPatientCode(),
            'patientname': f.getPatientName(),
            'samplefrequencies': f.getSampleFrequencies(),
            'startdatetime': f.getStartdatetime().isoformat(),
            'signallabels': f.getSignalLabels(),
            'signalheaders': f.getSignalHeaders(),
            'technician': f.getTechnician(),
            'recordingadditional': f.getRecordingAdditional()
        }

def load_edf(filepath, dtype='float32', channels=None, nanchannels=None,
                    geometry=None, digital=False):
    metadata = read_edfinfo(filepath=filepath)
    if channels is not None:
        chindices = [metadata['signallabels'].index(ch) for ch in channels]
    else:
        channels = metadata['signallabels']
        chindices = [i for i in range(len(channels))]
    nframes = metadata['nsamples'][chindices]
    fss = metadata['samplefrequencies'][chindices]
    if not np.all(nframes[0] == nframes):
        raise ValueError(f'Not all channel lengths are the same: {nframes} for'
                         f'{channels}.')
    if not np.all(fss[0] == fss):
        raise ValueError(f'Not all sampling frequencies are the same: {fss} for'
                         f'{channels}.')
    nframes = nframes[0]
    fs = fss[0]
    nchannels = len(chindices)
    # replace entries because channels may be selection
    for key in ['samplefrequencies', 'nsamples', 'signallabels', 'signalheaders']:
        metadata[key] = [metadata[key][i] for i in chindices]
    metadata['nsignals'] = len(channels)
    if nanchannels is not None:
        nchannels += len(nanchannels)
        channels += nanchannels
    sigbufs = np.zeros((nframes,nchannels), dtype=dtype)
    if nanchannels is not None:
        sigbufs[:,-len(nanchannels):] = np.nan
    with pyedflib.EdfReader(str(filepath)) as f:
        for i,j in enumerate(chindices):
            sigbufs[:,i] = f.readSignal(j, digital=digital)
    startdatetime = metadata['startdatetime']
    s = MultiChannelUniformTimeSeries(samples=sigbufs, fs=float(fs),channelnames=channels,
                                      geometry=geometry, startdatetime=startdatetime, dtype=dtype,
                                      channelaxis=1, metadata=metadata)
    return s


