import numpy as np
import darr
import textwrap
from darr.basedatadir import DataDir, create_basedatadir
from ..core.uniformtime import UniformTimeSeries, \
    MultiChannelUniformTimeSeries, EpochedMultiChannelUniformTimeSeries, \
    EpochedUniformTimeSeries, isuts, iseuts, ismcuts, isemcuts
from ..core.geometry import RectangularLattice
from .._version import get_versions

# FIXME Darr should become standard open and save


__all__ = ['opendarr', 'savedarr']

def wrap(s):
    return textwrap.fill(s, width=78, replace_whitespace=False)

readmetxt =  wrap("This directory contains time signal data that is accessible with the Python library uts, "
                  "but that is also easily accessible in other data analysis / programming platforms.")
readmetxt += "\n\n"
readmetxt += wrap("The JSON text files 'metadata.json', 'channeldata.json', 'epochdata.json' and "
                  "'signaldata.json', if present, contain metadata. The subdirectory 'samples.darr' contains "
                  "the numeric data array, and has its own README file detailing how it can be accessed.")
readmetxt += "\n"

# TODO perhaps an iter based version and a non-iter for time not last axis
def savedarr(path, s, accessmode='r', dtype=None, overwrite=False,
             framesize=1024*1000, readmetxt=readmetxt):
    bdd = create_basedatadir(path, overwrite=overwrite)
    if overwrite:
        bdd._delete_files(('signaldata.json', 'channeldata.json',
                           'epochdata.json', 'metadata.json'))
    # if iseuts(s) and s.timeaxis != 0:
    #     axes = ['time','epoch']
    # elif ismcuts(s) and s.timeaxis != 0:
    #     axes = ['time','channel']
    # elif isemcuts(s) and s.timeaxis != 0:
    #     axes = list(s.samples.axisorder)
    #     del(axes[axes.index('time')])
    #     axes.insert(0, 'time')
    # elif isuts(s): # isuts
    #     axes = ['time']
    axisorder = s.samples.axisorder
    firstaxis = axisorder[0]
    if firstaxis == 'time':
        framegenerator = (f.samples[:] for f in s.iter_frames(1024*100))
    elif firstaxis == 'channel':
        print('am here')
        framegenerator = (np.expand_dims(ch.samples[:],0) for ch in s.iter_channels()) #
    elif firstaxis == 'epoch':
        framegenerator = (np.expand_dims(e.samples[:],0) for e in s.iter_epochs()) # FIXME check this!!
    else:
        raise ValueError(f"Do not understand last axis '{firstaxis}'")
    metadata = dict(((k, i) if not hasattr(i, 'decode') else (k, i.decode("utf-8"))
                      for (k, i) in s.metadata.items()))
    signaldata = ({'fs': s.fs,
                 'startdatetime': str(s.startdatetime),
                 'origintime': s.origintime,
                 })
    if hasattr(s, 'channelnames'):
        signaldata['channelaxis'] =  int(s.channelaxis)
        signaldata['timeaxis'] = int(s.timeaxis)
        channeldata = {'channelnames': [str(n) for n in s.channelnames]}
        if hasattr(s, 'geometry'):
            if type(s.geometry) is RectangularLattice:
                g = {'type': 'RectangularLattice',
                     'instantiationparameters': s.geometry.instantiationparameters()}
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
    bdd.write_jsondict(filename='signaldata.json', d=signaldata,
                        overwrite=overwrite)
    bdd.write_jsondict(filename='metadata.json', d=metadata, overwrite=overwrite)
    bdd.write_txt(filename='README.txt', text=readmetxt,
                   overwrite=overwrite)
    samplepath = bdd.path / 'samples.darr'
    a = darr.asarray(path=samplepath, array=framegenerator,
                     accessmode=accessmode, dtype=dtype, overwrite=overwrite)
    return opendarr(path, accessmode=accessmode)

# we need to allow the old `starttime` parameter to be read
def opendarr(path, accessmode='r'):
    bdd = DataDir(path)
    # bdd._write_txt(filename='README.txt', text=readmetxt,
    #                overwrite=True)
    a = darr.Array(bdd.path / 'samples.darr' , accessmode=accessmode)
    sd = bdd.read_jsondict('signaldata.json')
    if (bdd.path / 'metadata.json').exists():
        metadata = bdd.read_jsondict('metadata.json')
    else:
        metadata = None
    # if (bdd.path / 'attrs.json').exists(): # for compatibility with old code
    #     attrs = bdd._read_jsondict('attrs.json')
    #     if metadata is None:
    #         metadata = attrs
    #     else:
    #         metadata.update(attrs)
    #     bdd._write_jsondict(filename='metadata.json', d=metadata, overwrite=True)
    #     (bdd.path / 'attrs.json').unlink()
    utstype = sd['utstype']
    if utstype == 'UniformTimeSeries':
        return UniformTimeSeries(samples=a, fs=sd['fs'],
                                 startdatetime=sd.get('startdatetime', 'NaT'),
                                 origintime=sd['origintime'], copy=False,
                                 dtype=a.dtype, metadata=metadata,
                                 starttime=sd.get('starttime', 0.0))
    elif utstype == 'MultiChannelUniformTimeSeries':
        sd = bdd.read_jsondict('signaldata.json')
        chd = bdd.read_jsondict('channeldata.json')
        channelnames = chd['channelnames']
        geometry = _darrmetadatatogeometry(chd)
        return MultiChannelUniformTimeSeries(samples=a,
                                      fs=sd['fs'], channelnames=channelnames,
                                      channelaxis=sd['channelaxis'],
                                      startdatetime=sd.get('startdatetime','NaT'),
                                      origintime=sd['origintime'], copy=False,
                                      dtype=a.dtype, geometry=geometry,
                                      metadata=metadata,
                                      starttime=sd.get('starttime', 0.0))
    elif utstype == 'EpochedUniformTimeSeries':
        sd = bdd.read_jsondict('signaldata.json')
        epochs = np.genfromtxt(bdd._path / 'epochs.csv', names=True)
        return EpochedUniformTimeSeries(samples=a,
                                        fs=sd['fs'],
                                        origintime=sd['origintime'],
                                        epochs=epochs,
                                        epochaxis=sd['epochaxis'],
                                        copy=False,
                                        dtype=a.dtype,
                                        metadata=metadata)
    elif utstype == 'EpochedMultiChannelUniformTimeSeries':
        sd = bdd.read_jsondict('signaldata.json')
        chd = bdd.read_jsondict('channeldata.json')
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
                                             metadata=metadata)
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
