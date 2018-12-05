import diskarray as da

__all__ = ['asdiskarray']

# FIXME all this decoding utf-8 should be improved
def asdiskarray(path, s, accessmode='r', dtype=None, scalefactor=None,
                overwrite=False, framesize=1024*1000):
    if scalefactor is None:
        framegenerator = (f.samples[:] for f in s.iter_frames(framesize=framesize))
    else:
        framegenerator = (f.samples[:]*scalefactor for f in s.iter_frames(framesize=framesize))
    attrs = dict(((k, i) if not hasattr(i, 'decode') else (k, i.decode("utf-8"))
                     for (k, i) in s.attrs.items()))
    metadata = ({'fs': s.fs,
                 'starttime': s.starttime,
                 'origintime': s.origintime,
                 'attrs': attrs})
    if hasattr(s, 'channels'):
        metadata.update({'channels': [(chn, int(p)) for (chn, p) in s.channels],
                         'channelaxis': int(s.channelaxis)})
    if hasattr(s, 'geometry'):
        g = {'pointnames': s.geometry.pointnames.tolist()}
        g.update({'instantiationcode': s.geometry.instantiationcode()})
        metadata.update({'geometry': g})
    if hasattr(s, 'epochs'):
        raise TypeError('cannot save epochs in diskarray yet')
    return da.asdiskarray(path=path, array=framegenerator, metadata=metadata,
                          accessmode=accessmode, dtype=dtype, overwrite=overwrite)