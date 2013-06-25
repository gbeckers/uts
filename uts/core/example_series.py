from io import load_UniformTimeSeriesfromh5
import tables
import os.path
from uts3.core.check_arg import check_arg

__all__ = ["example_emcuts", "example_mcuts", "example_uts", "example_euts"]

def example_emcuts(inmemory=True):
    return _load_series('emcuts.h5', 'lfp', inmemory=inmemory)

def example_mcuts(inmemory=True):
    return _load_series('mcuts.h5', 'lfp_epoch10', inmemory=inmemory)

def example_uts(inmemory=True):
    return _load_series('uts.h5', 'lfp_epoch10_electrode_26', inmemory=inmemory)

def example_euts(inmemory=True):
    return _load_series('euts.h5', 'lfp_electrode26', inmemory=inmemory)

def _load_series(filename, arrayname,inmemory=True):
    inmemory = check_arg(inmemory, 'inmemory')
    dirname = os.path.dirname(__file__)
    f = tables.openFile(os.path.join(dirname,'example_data', filename),'r')
    s = load_UniformTimeSeriesfromh5(f.getNode(f.root, arrayname), inmemory=inmemory)
    if inmemory:
        f.close()
    return s
