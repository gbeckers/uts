import darr
from ..core.uniformtime import UniformTimeSeries, \
    MultiChannelUniformTimeSeries, isuts, iseuts, ismcuts, isemcuts
from ..core.geometry import RectangularLattice
from .._version import get_versions

# FIXME epoched

__all__ = ['opendarr', 'savedarr']

def savedarr(path, s, accessmode='r', dtype=None, overwrite=False,
             framesize=1024*1000):
    framegenerator = (f.samples[:] for f in s.iter_frames(framesize=framesize))
    attrs = dict(((k, i) if not hasattr(i, 'decode') else (k, i.decode("utf-8"))
                     for (k, i) in s.attrs.items()))
    metadata = ({'fs': s.fs,
                 'starttime': s.starttime,
                 'origintime': s.origintime,
                 'attrs': attrs})
    if hasattr(s, 'channelnames'):
        metadata.update({'channelnames': [str(n) for n in s.channelnames],
                         'channelaxis': int(s.channelaxis)})
    if hasattr(s, 'geometry'):
        if type(s.geometry) is RectangularLattice:
            g = {'type': 'RectangularLattice'}
            g['instantiationparameters'] = s.geometry.instantiationparameters()
            metadata.update({'geometry': g})
        else:
            raise TypeError(f'Geometry type "{type(g)}" not supported for Darr')
    if hasattr(s, 'epochs'):
        raise TypeError('cannot save epochs in diskarray yet')
    metadata['utsversion'] = get_versions()['version']
    return darr.asarray(path=path, array=framegenerator, metadata=metadata,
                        accessmode=accessmode, dtype=dtype,
                        overwrite=overwrite)

def opendarr(path, accessmode='r'):
    a = darr.Array(path, accessmode=accessmode)
    md = dict(a.metadata) # read in memory
    if (len(a.shape) == 1):
        return UniformTimeSeries(samples=a, fs=md['fs'],
                                 starttime=md['starttime'],
                                 origintime=md['origintime'], copy=False,
                                 dtype=a.dtype, attrs=md['attrs'])
    elif (len(a.shape) == 2) and ('channelaxis' in md):
        channelnames = md['channelnames']
        geometry = _darrmetadatatogeometry(md)
        return MultiChannelUniformTimeSeries(samples=a,
                                      fs=md['fs'], channelnames=channelnames,
                                      channelaxis=md['channelaxis'],
                                      starttime=md['starttime'],
                                      origintime=md['origintime'], copy=False,
                                      dtype=a.dtype, geometry=geometry,
                                      attrs=md['attrs'])
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
