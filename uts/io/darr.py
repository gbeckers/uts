import numpy as np
import pandas as pd
import darr
import textwrap
from contextlib import contextmanager
from darr.datadir import DataDir, create_datadir
from pathlib import Path
from ..core.uniformtime import UniformTimeSeries, \
    MultiChannelUniformTimeSeries, EpochedMultiChannelUniformTimeSeries, \
    EpochedUniformTimeSeries, isuts, iseuts, ismcuts, isemcuts
from ..core.geometry import RectangularLattice
from .._version import get_versions
from .utils import tempdir

# FIXME Darr should become standard open and save

__all__ = ['opendarr', 'savedarr', 'loaddarr', 'cachedarr']

def wrap(s):
    return textwrap.fill(s, width=78, replace_whitespace=False)

readmetxt =  wrap("This directory contains time signal data that is accessible with the Python library uts, "
                  "but that is also easily accessible in other data analysis / programming platforms.")
readmetxt += "\n\n"
readmetxt += wrap("The JSON text files 'metadata.json', 'channeldata.json', 'epochdata.json' and "
                  "'signaldata.json', if present, contain metadata. The subdirectory 'samples.darr' contains "
                  "the numeric data array, and has its own README file detailing how it can be accessed.")
readmetxt += "\n"


def savedarr(path, s, accessmode='r', dtype=None, overwrite=False,
             axisorder=None, float_format='%.16e'):
    """Saves uts objects in a s self-explanatory and tool-independent format.

    Note that sample information is written without any loss of precision. Epoch
    information, however, is written as csv text file. Floating point numbers in
    epoch tables will therefore only be represented with a given precision (see
    `float_format` parameter).

    path:
    s:
    accessmode:
    dtype:
    overwrite:
    axisorder:
    float_format: str
      Float text format that is used to write the csv file with epoch information.
      Note that all sample information is written in binary format and will not
      lead to any loss of information. Epoch information, however is saved as text,
      and precision of float numbers depends on this parameter. Default is '%.18e'.

    """
    bdd = create_datadir(path, overwrite=overwrite)
    if overwrite:
        bdd._delete_files(('signaldata.json', 'channeldata.json',
                           'epochdata.json', 'metadata.json'))
    if axisorder is None:
        axisorder = s.samples.axisorder
    if len(axisorder) > 1:
        framegenerator = (f.transpose(axisorder).samples[:] for f in s.iter_frames(1024*100))
    else:
        framegenerator = (f.samples[:] for f in s.iter_frames(1024 * 100))

    metadata = dict(((k, i) if not hasattr(i, 'decode') else (k, i.decode("utf-8"))
                      for (k, i) in s.metadata.items()))
    signaldata = ({'fs': s.fs,
                 'startdatetime': str(s.startdatetime),
                 'origintime': s.origintime,
                 })
    sel = s[:2].transpose(axisorder) # two timesamples
    if hasattr(s, 'channelnames'):
        signaldata['channelaxis'] =  int(sel.channelaxis)
        signaldata['timeaxis'] = int(sel.timeaxis)
        channeldata = {'channelnames': [str(n) for n in s.channelnames]}
        if hasattr(s, 'geometry'):
            if type(s.geometry) is RectangularLattice:
                g = {'type': 'RectangularLattice',
                     'instantiationparameters': s.geometry.instantiationparameters()}
                channeldata['geometry'] = g
            else:
                raise TypeError(f'Geometry type "{type(s.geometry)}" not supported for Darr')
        bdd._write_jsondict(filename='channeldata.json', d=channeldata,
                            overwrite=overwrite)
    if hasattr(s, 'epochs'):
        signaldata['epochaxis'] = int(sel.epochaxis)
        signaldata['timeaxis'] = int(sel.timeaxis)
        if not isinstance(s.epochs, pd.DataFrame):
            epochs = pd.DataFrame(s.epochs)
        else:
            epochs = s.epochs
        epochs.to_csv(bdd.path / 'epochs.csv', index=False, float_format=float_format)
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
        epochs = pd.read_csv(bdd._path / 'epochs.csv').to_records(index=False)
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
        epochs = pd.read_csv(bdd._path / 'epochs.csv').to_records(index=False)
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


def loaddarr(path):
    return opendarr(path)[:]



@contextmanager
def cachedarr(s, axisorder=None, dtype=None, path=None, report=False,
            filename='cached.darr', keep=False, float_format='%.16e'):
    with tempdir(dir=path, keep=keep, report=report) as tempdirname:
        filepath = Path(tempdirname) /  filename
        s = savedarr(path=filepath, s=s, accessmode='r', dtype=dtype, overwrite=False,
                 axisorder=axisorder, float_format=float_format)
        yield s



def _darrmetadatatogeometry(metadata):
    if not 'geometry' in metadata:
        return None
    g = metadata['geometry']
    if g['type'] == 'RectangularLattice':
        return RectangularLattice(**g['instantiationparameters'])
    else:
        raise ValueError('geometry data in darr metadata not understood')
