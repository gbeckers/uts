import diskarray as da

def as_diskarray(path, s, mode='r', overwrite=False):
    framegenerator = (f.samples[:] for f in s.iter_frames(framesize=1024*10))
    metadata = {'fs': s.fs,
                'starttime': s.starttime,
                'origintime': s.origintime}
    da.asdiskarray(path=path, array=framegenerator, metadata=metadata,
                   mode=mode, overwrite=overwrite)