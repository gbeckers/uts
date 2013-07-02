from numpy import fromfile
from uts import UniformTimeSeries

def load_UniformTimeSeriesfromrawfile(filename, fs, dtype, starttime=0.0, origintime=0.0):
    return UniformTimeSeries(fromfile(filename, dtype=dtype), fs=fs,
                         starttime=starttime, origintime=origintime)