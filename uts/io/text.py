import numpy as np
from uts import MultiChannelUniformTimeSeries
## can be handy if you have EEG data with channels in different textfile
def iter_loadfromtxtfiles(filenames, channelnames, fs, chunksize=1024*10):
    
    def iter_lines(filenames):
        fileobjects = []
        for filename in filenames:
            fileobjects.append(open(filename,'r'))
        for l in fileobjects[0]:
            vals = [float(l)]
            for chan in range(1,len(fileobjects)):
                vals.append(float(fileobjects[chan].readline()))
            yield vals
        for fileobject in fileobjects:
            fileobject.close()
    
    block=[]
    i = 0
    for values in iter_lines(filenames):
        block.append(values)
        i += 1
        if i == chunksize:
            yield MultiChannelUniformTimeSeries(samples=block,fs=fs,
                                    channelnames=channelnames, channelaxis=1,
                                    dtype=np.float32)
            block = []
            i = 0
    if not i == 0:
        yield MultiChannelUniformTimeSeries(samples=block,fs=fs,
                                channelnames=channelnames, channelaxis=1,
                                dtype=np.float32)


