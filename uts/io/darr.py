import numpy as np
import darr
from darr.basedatadir import BaseDataDir, create_basedatadir
from ..core.uniformtime import UniformTimeSeries, \
    MultiChannelUniformTimeSeries, EpochedMultiChannelUniformTimeSeries, \
    EpochedUniformTimeSeries
from ..core.geometry import RectangularLattice
from .._version import get_versions

# FIXME epoched
# FIXME Darr should become standard open and save

__all__ = ['opendarr', 'savedarr']


def savedarr(path, s, accessmode='r', dtype=None, overwrite=False,
             framesize=1024*1000):
    bdd = create_basedatadir(path, overwrite=overwrite)
    if overwrite:
        bdd._delete_files(('signaldata.json', 'channeldata.json',
                           'epochdata.json', 'attrs.json'))

    framegenerator = (f.samples[:] for f in s.iter_frames(framesize=framesize))
    attrs = dict(((k, i) if not hasattr(i, 'decode') else (k, i.decode("utf-8"))
                     for (k, i) in s.attrs.items()))
    signaldata = ({'fs': s.fs,
                 'starttime': s.starttime,
                 'origintime': s.origintime,
                 })
    if hasattr(s, 'channelnames'):
        signaldata['channelaxis'] =  int(s.channelaxis)
        signaldata['timeaxis'] = int(s.timeaxis)
        channeldata = {'channelnames': [str(n) for n in s.channelnames]}
        if hasattr(s, 'geometry'):
            if type(s.geometry) is RectangularLattice:
                g = {'type': 'RectangularLattice'}
                g['instantiationparameters'] = s.geometry.instantiationparameters()
                channeldata['geometry'] = g
            else:
                raise TypeError(f'Geometry type "{type(g)}" not supported for Darr')
        bdd._write_jsondict(filename='channeldata.json', d=channeldata,
                            overwrite=overwrite)
    if hasattr(s, 'epochs'):
        signaldata['epochaxis'] = int(s.epochaxis)
        signaldata['timeaxis'] = int(s.timeaxis)
        # starttimes = s.epochs['starttime']
        # epochtimespath = bdd.path / 'epochstarttimes.darr'
        # darr.asarray(path=epochtimespath, array=starttimes,
        #              accessmode=accessmode, dtype=dtype, overwrite=overwrite)
        header = ",".join(s.epochs.dtype.names)
        np.savetxt(bdd.path / 'epochs.csv', s.epochs, delimiter=',', header=header)
    signaldata['utsversion'] = get_versions()['version']
    signaldata['utstype'] = type(s).__name__
    bdd._write_jsondict(filename='signaldata.json', d=signaldata,
                        overwrite=overwrite)
    bdd._write_jsondict(filename='attrs.json', d=attrs, overwrite=overwrite)
    bdd._write_txt(filename='README.txt', text="Not implemented yet!",
                   overwrite=overwrite)
    samplepath = bdd.path / 'samples.darr'
    a = darr.asarray(path=samplepath, array=framegenerator,
                     accessmode=accessmode, dtype=dtype, overwrite=overwrite)
    return opendarr(path, accessmode=accessmode)


def opendarr(path, accessmode='r'):
    bdd = BaseDataDir(path)
    a = darr.Array(bdd.path / 'samples.darr' , accessmode=accessmode)
    sd = bdd._read_jsondict('signaldata.json')
    attrs = bdd._read_jsondict('attrs.json')
    utstype = sd['utstype']
    if utstype == 'UniformTimeSeries':
        return UniformTimeSeries(samples=a, fs=sd['fs'],
                                 starttime=sd['starttime'],
                                 origintime=sd['origintime'], copy=False,
                                 dtype=a.dtype, attrs=attrs)
    elif utstype == 'MultiChannelUniformTimeSeries':
        sd = bdd._read_jsondict('signaldata.json')
        chd = bdd._read_jsondict('channeldata.json')
        channelnames = chd['channelnames']
        geometry = _darrmetadatatogeometry(chd)
        return MultiChannelUniformTimeSeries(samples=a,
                                      fs=sd['fs'], channelnames=channelnames,
                                      channelaxis=sd['channelaxis'],
                                      starttime=sd['starttime'],
                                      origintime=sd['origintime'], copy=False,
                                      dtype=a.dtype, geometry=geometry,
                                      attrs=attrs)
    elif utstype == 'EpochedUniformTimeSeries':
        sd = bdd._read_jsondict('signaldata.json')
        epochs = np.genfromtxt(bdd._path / 'epochs.csv', names=True)
        return EpochedUniformTimeSeries(samples=a,
                                        fs=sd['fs'],
                                        origintime=sd['origintime'],
                                        epochs=epochs,
                                        epochaxis=sd['epochaxis'],
                                        copy=False,
                                        dtype=a.dtype,
                                        attrs=attrs)
    elif utstype == 'EpochedMultiChannelUniformTimeSeries':
        sd = bdd._read_jsondict('signaldata.json')
        chd = bdd._read_jsondict('channeldata.json')
        channelnames = chd['channelnames']
        geometry = _darrmetadatatogeometry(chd)
        epochs = np.genfromtxt(bdd._path / 'epochs.csv', names=True)
        return EpochedMultiChannelUniformTimeSeries(samples=a,
                                             fs=sd['fs'],
                                             channelnames=channelnames,
                                             channelaxis=sd['channelaxis'],
                                             origintime=sd['origintime'],
                                             epochs=epochs,
                                             epochaxis=sd['epochaxis'],
                                             copy=False,
                                             dtype=a.dtype, geometry=geometry,
                                             attrs=attrs)
    else:
        raise TypeError(f'"Type of data is not understood or not supported"')


def _darrmetadatatogeometry(metadata):
    if not 'geometry' in metadata:
        return None
    g = metadata['geometry']
    if g['type'] == 'RectangularLattice':
        return RectangularLattice(**g['instantiationparameters'])
    else:
        raise ValueError('geometry data in darr metadata not understood')
