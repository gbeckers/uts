import os.path
import warnings
from types import GeneratorType
import numpy as np
import tables
from numpy import fromfile
from uts3.core.uniformtime import UniformTimeSeries, EpochedUniformTimeSeries, \
    MultiChannelUniformTimeSeries, EpochedMultiChannelUniformTimeSeries
from uts3.core.multichannel import MultiChannel
from uts3.core.epoched import Epoched
from utils import h5attr_to_geometry
from uts3.core.check_arg import check_arg, check_int
from uts3.core.utils import fit_frames

__all__ = ['save', 'load']

#FIXME file/node/group handling for load and save should be improved in a separate function


def save(s, signalname, h5group=None, filename=None, channelaxis=None, epochaxis=None,attrs={},
         overwrite=False, extensible=False, expectedrows=None, dtype=None, 
            complib='zlib', complevel=5, chunkshape=None):
    '''
        Saves a signal to h5 file. This function can also be used for generator objects that
        yield signals.

        Parameters
        ----------
        s: may be a ((epoched) multi-channel) time sampled signal or a 
          generator that produces such signals. The latter can be handy if one 
          has very large signals on-disk, that don't fit into working memory.
        signalname: a valid name that can later be used to retrieve the
          signal.
        h5group: a pytables h5 group instance
        filename:
        channelaxis:
        epochaxis:
        overwrite: if True overwrites an existing signal if it has the same
          name.
        extensible: if True makes the signal that is stored enlargeable. That
          is: samples can be appended later. If signal is a generator, the saved
          signal is always enlargeable.

    '''
    if filename is None:
        if not isinstance(h5group,tables.group.Group):
            raise ValueError("h5group argument is not a pytables Group")
    else:
        f = tables.openFile(filename, 'a')
        if h5group is None:
            h5group = f.root
        else:
            if isinstance(h5group, str):
                h5group = f.root._f_getChild(h5group)

    if isinstance(s, GeneratorType):
        s1 = s.next()
        if not extensible:
            warnings.warn("Saving generator to h5 array: setting extensible to True")
        extensible = True
        # FIXME use standard function from uniformtime
        if not isinstance(s1, (EpochedMultiChannelUniformTimeSeries,
                        MultiChannelUniformTimeSeries, UniformTimeSeries)):
            raise TypeError("signal is not a sampled signal")
        h5mcss = save(s1, signalname=signalname, h5group=h5group, filename=filename, 
            channelaxis=channelaxis, epochaxis=epochaxis, attrs=attrs, overwrite=overwrite, 
                        extensible=extensible,  expectedrows=expectedrows, dtype=dtype, 
                        complib=complib, complevel=complevel, chunkshape=chunkshape)
        for sn in s:
            h5mcss.append(sn)
        return h5mcss
    elif s._typeid == 'uts':
        return _save_uts(s, h5group=h5group, name=signalname, attrs=attrs, overwrite=overwrite,
                extensible=extensible, expectedrows=expectedrows, dtype=dtype, 
                complib=complib, complevel=complevel, chunkshape=chunkshape)
    elif s._typeid == 'mcuts':
        return _save_mcuts(s, h5group=h5group, name=signalname, channelaxis=channelaxis, attrs=attrs, 
                           overwrite=overwrite, extensible=extensible, expectedrows=expectedrows, 
                           dtype=dtype, complib=complib, complevel=complevel, chunkshape=chunkshape)
    elif s._typeid == 'euts':
        return _save_euts(s, h5group=h5group, name=signalname, epochaxis=epochaxis, attrs=attrs, 
                          overwrite=overwrite, extensible=extensible, expectedrows=expectedrows, 
                          dtype=dtype, complib=complib, complevel=complevel, chunkshape=chunkshape)
    elif s._typeid == 'emcuts':
        return _save_emcuts(s, h5group=h5group, name=signalname, channelaxis=channelaxis,
                            epochaxis=epochaxis, attrs=attrs, overwrite=overwrite,
                extensible=extensible, expectedrows=expectedrows, dtype=dtype, 
                complib=complib, complevel=complevel, chunkshape=chunkshape)
    else:
        raise TypeError("signal is not a sampled signal, or generator")


def load(h5group=None, filename=None, inmemory=False, mode='r'):

    if not mode in ('r', 'a'):
        raise ValueError('Only mode "r" and "a" allowed')
    if filename is None:
        if not isinstance(h5group, (tables.group.Group, tables.array.Array)):
            raise ValueError("h5group argument is not a pytables Group")
    else:
        f = tables.openFile(filename, mode)
        if h5group is None:
            h5group = f.root
        else:
            if isinstance(h5group, str):
                h5group = f.root._f_getChild(h5group)


    if isinstance(h5group,tables.group.Group):
        h5array = h5group._f_getChild('samples')
    else:
        h5array = h5group # for backward compatibility
    if not inmemory in (False,True):
        raise TypeError("inmemory parameter should be True or False")
    if inmemory:
        samples = h5array[:]
    else:
        samples = h5array
    fs = h5array.attrs.fs
    origintime = h5array.attrs.origintime
    
    if len(h5array.shape) == 1: # uts
        starttime = h5array.attrs.starttime
        signal = UniformTimeSeries(samples=samples, fs=fs, starttime=starttime,
                                     origintime=origintime, copy=False,
                                     dtype=None)
    #FIXME use the attr to find name of channel and epoch tables, not hard-code strings
    elif len(h5array.shape) == 2: # euts or mcuts
        if "channelaxis" in h5array.attrs._v_attrnames: # mcuts
            starttime = h5array.attrs.starttime
            channelaxis = h5array.attrs.channelaxis
            # there can be a table with channels info
            if h5array._v_name + "_channels" in h5array._v_parent._v_children.keys():
                name = h5array._v_pathname + "_channels"
                channeltable = h5array._v_file.getNode(name)
                channels = channeltable[:]
                channelnames = channels['channelname']
                if "geometry" in channeltable.attrs._v_attrnames:
                    pointnames = np.empty_like(channelnames)
                    pointnames[channels['geometrypoint']] = channelnames
                    g = h5attr_to_geometry(channeltable.attrs.geometry, pointnames=pointnames)
                else:
                    g = None
            elif "channelnames" in h5array.attrs._v_attrnames: # for compatibility with older data
                channels = h5array.attrs.channelnames
                if "geometry" in h5array.attrs._v_attrnames:
                    g = h5attr_to_geometry(h5array.attrs.geometry, pointnames=None)
                else:
                    g = None
            else:
                raise IOError(
                    "no channelnames or channels found in %s" % h5array._v_pathname)
                #channels = None
            signal = MultiChannelUniformTimeSeries(samples=samples,
                fs=fs, channels=channels, starttime=starttime,
                origintime=origintime, copy=False, dtype=None,
                channelaxis=channelaxis, geometry=g)
        else: #euts
            epochaxis = h5array.attrs.epochaxis
            # there should be a table with epoch info
            if h5array._v_name + "_records" in h5array._v_parent._v_children.keys():
                name = h5array._v_pathname + "_records"
            elif h5array._v_name + "_epochs" in h5array._v_parent._v_children.keys():
                name = h5array._v_pathname + "_epochs"
            else:
                raise IOError(
                    "no epochs table found in %s" % h5array._v_pathname)
            epochs = h5array._v_file.getNode(name)
            if inmemory: epochs = epochs[:]
            signal = EpochedUniformTimeSeries(samples=samples,
                fs=fs, epochs=epochs,
                origintime=origintime, copy=False, dtype=None,
                epochaxis=epochaxis)       
    else: # len(h5array.shape) == 3: # emcuts
        epochaxis = h5array.attrs.epochaxis
        channelaxis = h5array.attrs.channelaxis
        # there should be a table with epoch info
        if h5array._v_name + "_records" in h5array._v_parent._v_children.keys():
            name = h5array._v_pathname + "_records"
        elif h5array._v_name + "_epochs" in h5array._v_parent._v_children.keys():
            name = h5array._v_pathname + "_epochs"
        else:
            raise IOError("no epochs table found in %s" % h5array._v_pathname)
        epochs = h5array._v_file.getNode(name)
        if inmemory: epochs = epochs[:]
        # there can be a table with channels info
        if h5array._v_name + "_channels" in h5array._v_parent._v_children.keys():
            name = h5array._v_pathname + "_channels"
            channeltable = h5array._v_file.getNode(name)
            channels = channeltable[:]
            channelnames = channels['channelname']
            if "geometry" in channeltable.attrs._v_attrnames:
                pointnames = np.empty_like(channelnames)
                pointnames[channels['geometrypoint']] = channelnames
                g = h5attr_to_geometry(channeltable.attrs.geometry, pointnames=pointnames)
            else:
                g = None
        elif "channelnames" in h5array.attrs._v_attrnames:
            channels = h5array.attrs.channelnames
            if "geometry" in h5array.attrs._v_attrnames:
                g = h5attr_to_geometry(h5array.attrs.geometry, pointnames=None)
            else:
                g = None
        else:
            #raise IOError, "no channelnames found in %s" % h5array._v_pathname
            channels = None
        signal = EpochedMultiChannelUniformTimeSeries(samples=samples,
            fs=fs, channels=channels, origintime=origintime, 
            copy=False, dtype=None, epochs=epochs, geometry=g, 
            channelaxis=channelaxis, epochaxis=epochaxis)
    return signal
    
def _newh5uts(h5group, name, dtype, ntimesamples, fs, origintime, 
              starttime, dflt=0.0, overwrite=False, extensible=False, expectedrows=None, 
              complib='zlib', complevel=5,  chunkshape=None, attrs={}):

    """
    Returns an pytables CArray or EArray node with dimensions of ntimesamples, 
    filled with zeros.

    """
    fs = check_arg(fs, 'fs')
    origintime = check_arg(origintime, 'time')
    starttime = check_arg(starttime, 'time')
    ntimesamples = check_int(ntimesamples, negative=False)
    dtype = np.dtype(dtype)

    h5file = h5group._v_file
    if overwrite:
        if h5group._v_children.has_key(name):
            h5group._v_children[name]._f_remove()
            h5file.flush()
    filters = tables.Filters(complevel=complevel, complib=complib)
    atom = tables.Atom.from_dtype(dtype, dflt=dflt)
    if extensible:    
        arnode = h5file.createEArray(where=h5group, name=name, atom=atom,
                                     shape=(0,), filters=filters,
                                     expectedrows=expectedrows, chunkshape=chunkshape)
        framesize = int(8*1024*1024/dtype.itemsize)# 8 Mb chunks
        framesize = max(framesize, 1)
        nframes, newsize, remainder = fit_frames(totalsize=ntimesamples,
                                                 framesize=framesize, stepsize=framesize)
        zf = np.zeros(framesize, dtype=dtype)
        for i in range(nframes):
            arnode.append(zf)
        arnode.append(np.zeros(remainder, dtype=dtype))
    else:
        shape = (ntimesamples,)
        arnode = h5file.createCArray(where=h5group, name=name, 
                                     atom=atom, shape=shape, title='', filters=filters, 
                                     chunkshape=chunkshape, byteorder=None, createparents=False)
    attrs['fs'] = fs
    attrs['starttime'] = starttime
    attrs['origintime'] = origintime
    for attrname, value in attrs.items():
        arnode._f_setAttr(attrname, value)
    arnode.flush()
    return arnode

def _newh5mcuts(h5group, name, dtype, ntimesamples, fs, origintime, 
                starttime, channelaxis, nchannels, channels, geometry,
                dflt=0.0, overwrite=False, extensible=False, expectedrows=None, 
                complib='zlib', complevel=5,  chunkshape=None, attrs={}):

    """
    Returns an pytables CArray or EArray node with dimensions of self.samples, 
    filled with zeros.

    """
    
    fs = check_arg(fs, 'fs')
    origintime = check_arg(origintime, 'time')
    starttime = check_arg(starttime, 'time')
    channelaxis = check_arg(channelaxis, 'channelaxis')
    mc = MultiChannel(nchannels=nchannels, channels=channels, 
                      geometry=geometry)
    ntimesamples = check_int(ntimesamples, negative=False)
    dtype = np.dtype(dtype)
    h5file = h5group._v_file
    if overwrite:
        if h5group._v_children.has_key(name):
            h5group._v_children[name]._f_remove()
            h5file.flush()
    filters = tables.Filters(complevel=complevel, complib=complib)
    atom = tables.Atom.from_dtype(dtype,dflt=dflt)
    if extensible:
        if channelaxis is None:
            channelaxis = 0
        if channelaxis != 0:
            raise ValueError("Extensible mcuts can only have channelaxis 0")
        shape = (mc.nchannels, 0)
        arnode = h5file.createEArray(where=h5group, name=name, atom=atom,
                                     shape=shape, filters=filters,
                                     expectedrows=expectedrows, chunkshape=chunkshape)
        framesize = int(8*1024*1024/(dtype.itemsize*mc.nchannels))# 8 Mb chunks
        framesize = max(framesize, 1)
        nframes, newsize, remainder = fit_frames(totalsize=ntimesamples,
                                                 framesize=framesize, stepsize=framesize)
        zf = np.zeros((mc.nchannels,framesize), dtype=dtype)
        for i in range(nframes):
            arnode.append(zf)
        arnode.append(np.zeros((mc.nchannels,remainder), dtype=dtype))
    else:
        if channelaxis is None:
            channelaxis = 1 # most likely efficient if retrieving time chunks
        else:
            if channelaxis not in (0,1):
                raise ValueError("channelaxis should be 0 or 1")
        if channelaxis == 0:
            shape = (mc.nchannels, ntimesamples)
        else:
            shape = (ntimesamples, mc.nchannels)
        arnode = h5file.createCArray(where=h5group, name=name, 
                                     atom=atom, shape=shape, title='', filters=filters, 
                                     chunkshape=chunkshape, byteorder=None, createparents=False)
    attrs['fs'] = fs
    attrs['starttime'] = starttime
    attrs['origintime'] = origintime
    attrs['channelaxis'] = channelaxis
    attrs['channeltable'] = name + "_channels"
    for attrname, value in attrs.items():
        arnode._f_setAttr(attrname, value)
    _save_multichannel(multichannel=mc, h5group=h5group, name=name+"_channels", overwrite=overwrite)
    arnode.flush()
    return arnode

def _newh5euts(h5group, name, dtype, ntimesamples, fs, origintime, 
               starttime, epochaxis, epochs, dflt=0.0, overwrite=False, extensible=False, 
               expectedrows=None, complib='zlib', complevel=5,  chunkshape=None, attrs={}):

    """
    Returns an pytables CArray or EArray node with dimensions of self.samples, 
    filled with zeros.

    If extensible, then in the dimension of epoch; epochaxis must be 1 then,
    and the argument value is ignored. When nepochs >> ntimesamples then 
    choose epochaxis = 0.

    """

    fs = check_arg(fs, 'fs')
    ntimesamples = check_int(ntimesamples, negative=False)
    origintime = check_arg(origintime, 'time')
    starttime = check_arg(starttime, 'time')
    epochaxis = check_arg(epochaxis,'epochaxis')
    ep = Epoched(duration=ntimesamples/fs, epochs=epochs, origintime=origintime)
    dtype = np.dtype(dtype)
    h5file = h5group._v_file
    if overwrite:
        if h5group._v_children.has_key(name):
            h5group._v_children[name]._f_remove()
            h5file.flush()
    filters = tables.Filters(complevel=complevel, complib=complib)
    atom = tables.Atom.from_dtype(dtype,dflt=dflt)
    if extensible:
        if epochaxis is None:
            epochaxis = 1
        if epochaxis != 1:
            raise ValueError("Extensible euts can only have epochaxis 1")
        shape = (ntimesamples,0)
        arnode = h5file.createEArray(where=h5group, name=name, atom=atom,
                                     shape=shape, filters=filters,
                                     expectedrows=expectedrows, chunkshape=chunkshape)
        nepochs = ep.nepochs
        zf = np.zeros((ntimesamples,1), dtype=dtype)
        for i in range(ep.nepochs):
            arnode.append(zf)
    else:
        if epochaxis is None:
            epochaxis = 0 # most likely efficient if retrieving whole epochs
        else:
            if epochaxis not in (0,1):
                raise ValueError("epochaxis should be 0 or 1")
        if epochaxis == 0:
            shape = (ep.nepochs, ntimesamples)
        else:
            shape = (ntimesamples, ep.nepochs)
        arnode = h5file.createCArray(where=h5group, name=name, 
                                     atom=atom, shape=shape, title='', filters=filters, 
                                     chunkshape=chunkshape, byteorder=None, createparents=False)
    attrs['fs'] = fs
    attrs['starttime'] = starttime
    attrs['origintime'] = origintime
    attrs['epochaxis'] = epochaxis
    attrs['epochtable'] = name+"_epochs"
    for attrname, value in attrs.items():
        arnode._f_setAttr(attrname, value)
    _save_epoched(epoched=ep, h5group=h5group, name=name+"_epochs", overwrite=overwrite)
    arnode.flush()
    return arnode

def _newh5emcuts(h5group, name, dtype, ntimesamples, fs, origintime, 
                 starttime, epochaxis, channelaxis, nchannels, channels, geometry,
                 epochs, dflt=0.0, overwrite=False, extensible=False, expectedrows=None, 
                 complib='zlib', complevel=5,  chunkshape=None, attrs={}):

    """
    Returns an pytables CArray or EArray node with dimensions of self.samples, 
    filled with zeros.

    """

    fs = check_arg(fs, 'fs')
    ntimesamples = check_int(ntimesamples, negative=False)    
    origintime = check_arg(origintime, 'time')
    starttime = check_arg(starttime, 'time')
    channelaxis = check_arg(channelaxis, 'channelaxis')
    mc = MultiChannel(nchannels=nchannels, channels=channels, 
                      geometry=geometry)    
    epochaxis = check_arg(epochaxis,'epochaxis')
    ep = Epoched(duration=ntimesamples/fs, epochs=epochs, origintime=origintime)
    dtype = np.dtype(dtype)
    h5file = h5group._v_file
    if overwrite:
        if h5group._v_children.has_key(name):
            h5group._v_children[name]._f_remove()
            h5file.flush()
    filters = tables.Filters(complevel=complevel, complib=complib)
    atom = tables.Atom.from_dtype(dtype,dflt=dflt)
    if extensible:
        if epochaxis is None:
            epochaxis = 2
        if epochaxis != 2:
            raise ValueError("epochaxis should be 2 for extensible emcuts")
        if channelaxis is None:
            channelaxis = 1
        if not channelaxis in (0,1):
            raise ValueError(
                "channelaxis should be 0 or 1 for extensible emcuts")
        if channelaxis == 1:
            shape=(ntimesamples, mc.nchannels, 0)
        else:
            shape=(mc.nchannels, ntimesamples, 0)
        arnode = h5file.createEArray(where=h5group, name=name, atom=atom,
                                     shape=shape, filters=filters,
                                     expectedrows=expectedrows, chunkshape=chunkshape)
        nepochs = ep.nepochs
        zf = np.zeros((shape[0],shape[1],1), dtype=dtype)
        for i in range(nepochs):
            arnode.append(zf)
    else:
        # first figure out which axis is where. The default is to put
        # epoch at low axes and channel at high
        shape = [ntimesamples,ntimesamples,ntimesamples]
        if epochaxis is None:
            if channelaxis != 0:
                epochaxis = 0
            else:
                epochaxis = 1
        if not epochaxis in (0, 1, 2):
            raise ValueError("epochaxis should be 0, 1 or 2")
        shape[epochaxis] = ep.nepochs
        if channelaxis is None:
            if epochaxis != 2:
                channelaxis = 2
            else:
                channelaxis = 1
        if not channelaxis in (0, 1, 2):
            raise ValueError("channelaxis should be 0, 1 or 2")
        if channelaxis == epochaxis:
            raise ValueError("channelaxis is same as epochaxis")
        shape[channelaxis] = mc.nchannels
        arnode = h5file.createCArray(where=h5group, name=name, 
                                     atom=atom, shape=shape, title='', filters=filters, 
                                     chunkshape=chunkshape, byteorder=None, createparents=False)
    attrs['fs'] = fs
    attrs['starttime'] = starttime
    attrs['origintime'] = origintime
    attrs['epochaxis'] = epochaxis
    attrs['channelaxis'] = channelaxis
    attrs['channeltable'] = name + "_channels"
    attrs['epochtable'] = name + "_epochs"
    for attrname, value in attrs.items():
        arnode._f_setAttr(attrname, value)
    _save_epoched(epoched=ep, h5group=h5group, name=name+"_epochs", overwrite=overwrite)
    _save_multichannel(multichannel=mc, h5group=h5group, name=name+"_channels", overwrite=overwrite)
    arnode.flush()
    return arnode


# def save_UniformTimeSeriestoh5(h5group, signal, signalname, overwrite=False,
#                 extensible=False, expectedrows=None, filters=None, chunkshape=None):
#         if isinstance(signal, (EpochedMultiChannelUniformTimeSeries,
#                            MultiChannelUniformTimeSeries, UniformTimeSeries)):
#             h5mcss = save(signal,h5group=h5group, name=signalname,
#                              overwrite=overwrite, extensible=extensible,
#                              expectedrows=expectedrows, filters=filters,
#                              chunkshape=chunkshape)
#         elif isinstance(signal, GeneratorType):
#             s1 = signal.next()
#             assert isinstance(s1, (EpochedMultiChannelUniformTimeSeries,
#                             MultiChannelUniformTimeSeries, UniformTimeSeries))
#             h5mcss = save(s1,h5group=h5group, name=signalname,
#                              overwrite=overwrite, extensible=True,
#                               filters=filters, chunkshape=chunkshape)
#             for sn in signal:
#                 h5mcss.append(sn)
#         else:
#             raise TypeError, "signal is not a sampled signal, or generator"

#         return h5mcss

# def create_emptyh5MultiChannelUniformTimeSeries(h5group, signalname, nsamples, 
#                 nchannels, fs, starttime=0.0, origintime=0.0, channelaxis=0,
#                 dtype=np.float64, geometry=None, channels=None, 
#                 overwrite=False, extensible=False, expectedrows=None, complevel=0,
#                 chunkshape=None, attrdict={}):
    
#     h5file = h5group._v_file
#     if (channelaxis == 0) or extensible:
#         shape = (nchannels, nsamples)
#         channelaxis = 0
#     else:
#         shape = (nsamples,nchannels)
#     filters = tables.Filters(complevel=complevel)
#     atom = tables.Atom.from_dtype(dtype)
#     if overwrite:
#         if h5group._v_children.has_key(signalname):
#             h5group._v_children[signalname]._f_remove()
#             h5file.flush()
#     if not extensible:
#         arnode = h5file.createCArray(where=h5group, name=signalname, 
#                                     atom=atom, shape=shape,filters=filters,
#                                     chunkshape=chunkshape)
#     else:
#         if expectedrows is None:
#             expectedrows = nsamples
#         arnode = h5file.createEArray(where=h5group, name=name, atom=atom,
#                 shape=(self.nchannels, 0), filters=filters,
#                 expectedrows=expectedrows,chunkshape=chunkshape)
#         blocksize = 1024
#         nblocks,remainder = divmode(nsamples,blocksize)
#         sampleblock = np.zeros([channels,blocksize],dtype=dtype)
#         for i in range(nblocks):
#             arnode.append(samples)
#         if remainder > 0:
#             arnode.append(samples[:,:remainder])
#     attrs = {'fs': fs,
#              'starttime': starttime,
#              'origintime': origintime,
#              'channelaxis': channelaxis,
#              'channeltable': signalname + "_channels"}
#     for attrname, value in attrs.items():
#         arnode._f_setAttr(attrname, value)
#     # if not geometry is None:
#     #     geometry.saveh5(arnode)
#     mc = MultiChannel(channels=channels,geometry=geometry)
#     _save_multichannel(multichannel=mc, h5group=h5group, name=signalname+"_channels", overwrite=overwrite)
#     arnode.flush()
    
#     return loadh5(arnode)

# def create_h5MultiChannelUniformTimeSeries_emptylike(mctss, h5group, signalname,
#                 overwrite=False, extensible=False, expectedrows=None, complevel=0,
#                 attrdict={}):
    
#     fs = mctss.fs
#     nsamples = mctss.ntimesamples
#     nchannels = mctss.nchannels
#     channels = mctss.channels
#     geometry = mctss.geometry
#     dtype = mctss.dtype
#     starttime = mctss.starttime
#     origintime = mctss.origintime
#     channelaxis = mctss.channelaxis
    
#     return create_emptyh5MultiChannelUniformTimeSeries(h5group, signalname, nsamples, 
#                 nchannels, fs, starttime=0.0, origintime=0.0, channelaxis=channelaxis,
#                 dtype=dtype, geometry=geometry, channels=channels, 
#                 overwrite=overwrite, extensible=False, expectedrows=None, complevel=0,
#                 attrdict=attrdict)

def _create_signalgroup(h5group, name, overwrite):
    if overwrite and h5group._v_children.has_key(name):
        h5group._v_children[name]._f_remove(recursive=True)
        h5group._v_file.flush()
    return h5group._v_file.createGroup(h5group, name)


def _save_uts(uts, h5group, name, attrs={}, overwrite=False,
            extensible=False, expectedrows=None, dtype=None, 
            complib='zlib', complevel=5, chunkshape=None):
    '''
    Saves the signal array in a hdf5 file.

    parameter
    ---------

    - h5group: the group object under which the signal array should be saved

    returns
    -------

    the h5 arraynode in which the array is saved

    '''
    
    signalgroup = _create_signalgroup(h5group=h5group, name=name, overwrite=overwrite)
    if dtype is None:
        dtype = uts.samples.dtype
    arnode = _newh5uts(h5group=signalgroup, name='samples',
                       ntimesamples=uts.ntimesamples, fs=uts.fs,
                       origintime=uts.origintime, starttime=uts.starttime, dtype=dtype, 
                       overwrite=overwrite, extensible=extensible, expectedrows=expectedrows, 
                       complib=complib, complevel=complevel, chunkshape=chunkshape, attrs=attrs)
    # fill it with our samples; we take chunks of maximum 8 Mb
    framesize = int(8*1024*1024/uts.samples.dtype.itemsize)
    framesize = max(1, framesize)
    start = 0
    for i,frame in enumerate(uts.iter_frames(framesize=framesize)):
        samples = frame.samples.get(copy=False)
        if dtype != samples.dtype:
            samples = samples.astype(dtype)
        arnode[start:start+frame.ntimesamples] = samples
        start += frame.ntimesamples
    arnode.flush()
    return uts._instantiatefromself(samples=arnode, dtype=arnode.dtype, copy=False)

def _save_mcuts(mcuts, h5group, name, attrs={}, overwrite=False, channelaxis=None,
               extensible=False, expectedrows=None, dtype=None, 
               complib='zlib', complevel=5, chunkshape=None):
    '''
    Saves the signal array in a hdf5 file.

    parameter:

    - h5group: the group object under which the signal array should be saved

    returns:

    the h5 arraynode in which the array is saved

    Notes
    -----

    - extensible arrays should have channelaxis 1, because we want to be able
    to append along the time dimension, not channel domension, and the 
    underlying hdf5 object only allows to append along dimension 0.

    '''
    signalgroup = _create_signalgroup(h5group=h5group, name=name, overwrite=overwrite)
    if extensible:
        if channelaxis == 1:
            raise ValueError("Extensible mcuts must have channelaxis 0")
        else:
            channelaxis = 0 # in case the param wasn't provided
    if channelaxis is None:
        channelaxis = mcuts.channelaxis
    if dtype is None:
        dtype = mcuts.samples.dtype
    arnode = _newh5mcuts(h5group=signalgroup, name='samples', dtype=dtype,
                         ntimesamples=mcuts.ntimesamples, fs=mcuts.fs, 
                         origintime=mcuts.origintime, starttime=mcuts.starttime, 
                         nchannels = mcuts.nchannels,
                         channels=mcuts.channelnames, geometry=mcuts.geometry, 
                         channelaxis=channelaxis, overwrite=overwrite, 
                         extensible=extensible, expectedrows=expectedrows, 
                         complib=complib, complevel=complevel, chunkshape=chunkshape,
                         attrs=attrs)

    # loop over big chunks of first axis
    chunksize = 8*1024*1024 # 8 Mb
    if channelaxis == 0:
        channelsize = (mcuts.ntimesamples*mcuts.dtype.itemsize)
        nchannelsperchunk = int(chunksize//float(channelsize))
        nchannelsperchunk = max(nchannelsperchunk,1)
        nchunks, remainder = divmod(mcuts.nchannels,nchannelsperchunk)
        start = 0
        for i in range(nchunks):
            arnode[start:start+nchannelsperchunk,:] =\
                  mcuts[:,start:start+nchannelsperchunk].samples.transpose(['channel','time']).get(copy=False)
            start = start + nchannelsperchunk
        if remainder > 0:
            arnode[start:start+remainder,:] =\
                  mcuts[:,start:start+remainder].samples.transpose(['channel','time']).get(copy=False)
    else: # channelaxis == 1
        timesize = (mcuts.nchannels*mcuts.dtype.itemsize)
        ntimesperchunk = int(chunksize//float(timesize))
        ntimesperchunk = max(ntimesperchunk,1)
        nchunks, remainder = divmod(mcuts.ntimesamples,ntimesperchunk)
        start = 0
        for i in range(nchunks):
            arnode[start:start+ntimesperchunk,:] =\
                  mcuts[start:start+ntimesperchunk].samples.transpose(['time','channel']).get(copy=False)
            start = start + ntimesperchunk
        if remainder > 0:
            arnode[start:start+remainder,:] =\
                  mcuts[start:start+remainder,:].samples.transpose(['time','channel']).get(copy=False)

    arnode.flush()
    dtype = arnode.dtype
    return mcuts._instantiatefromself(samples=arnode, channelaxis=channelaxis, 
                                     dtype=dtype, copy=False)

def _save_euts(euts, h5group, name, epochaxis=None,attrs={}, overwrite=False,
               extensible=False, expectedrows=None, dtype=None, 
               complib='zlib', complevel=5, chunkshape=None):
    '''
    Saves the signal array in a hdf5 file.

    parameter:

    - h5group: the group object under which the signal array should be saved

    returns:

    the h5 arraynode in which the array is saved

    '''
    signalgroup = _create_signalgroup(h5group=h5group, name=name, overwrite=overwrite)
    if dtype is None:
        dtype = euts.dtype
    if epochaxis is None:
        epochaxis = euts.epochaxis
    arnode = _newh5euts(h5group=signalgroup, name='samples', dtype=dtype, 
                        ntimesamples=euts.ntimesamples, fs=euts.fs, 
                        origintime=euts.origintime, starttime=euts.starttime, 
                        epochaxis=epochaxis, epochs=euts.epochs, 
                        overwrite=overwrite, extensible=False, 
                        expectedrows=None, complib=complib, complevel=complevel,
                        chunkshape=None, attrs=attrs)
    epochs = arnode._v_parent._f_getChild("samples_epochs")

    # loop over big chunks of first axis
    chunksize = 8*1024*1024 # 8 Mb
    if epochaxis == 0:
        epochsize = (euts.ntimesamples*euts.dtype.itemsize)
        nepochsperchunk = int(chunksize//float(epochsize))
        nepochsperchunk = max(nepochsperchunk,1)
        nchunks, remainder = divmod(euts.nepochs,nepochsperchunk)
        start = 0
        for i in range(nchunks):
            arnode[start:start+nepochsperchunk,:] =\
                  euts[:,start:start+nepochsperchunk].samples.transpose(['epoch','time']).get(copy=False)
            start = start + nepochsperchunk
        if remainder > 0:
            arnode[start:start+remainder,:] =\
                  euts[:,start:start+remainder].samples.transpose(['epoch','time']).get(copy=False)
    else: # epochaxis == 1
        timesize = (euts.nepochs*euts.dtype.itemsize)
        ntimesperchunk = int(chunksize//float(timesize))
        ntimesperchunk = max(ntimesperchunk,1)
        nchunks, remainder = divmod(euts.ntimesamples,ntimesperchunk)
        start = 0
        for i in range(nchunks):
            arnode[start:start+ntimesperchunk,:] =\
                  euts[start:start+ntimesperchunk].samples.transpose(['time','epoch']).get(copy=False)
            start = start + ntimesperchunk
        if remainder > 0:
            arnode[start:start+remainder,:] =\
                  euts[start:start+remainder,:].samples.transpose(['time','epoch']).get(copy=False)
    arnode.flush()
    dtype = arnode.dtype
    return euts._instantiatefromself(samples=arnode, dtype=dtype, epochs=epochs,
                                     epochaxis=epochaxis, copy=False)

def _save_emcuts(emcuts, h5group, name, attrs={}, overwrite=False,
        extensible=False, expectedrows=None, dtype=None, epochaxis=None, 
        channelaxis=None, complib='zlib', complevel=5, chunkshape=None):
    '''
    Saves the signal array in a hdf5 file.
   
    parameter:
   
    - h5group: the group object under which the signal array should be saved
   
    returns:
   
    the h5 arraynode in which the array is saved
   
    Notes
    -----
   
    - extensible arrays should have channelaxis 1, because we want to be able
    to append along the time dimension, not channel domension, and the 
    underlying hdf5 object only allows to append along dimension 0.
   
    '''
    signalgroup = _create_signalgroup(h5group=h5group, name=name, overwrite=overwrite)
    if channelaxis is None:
        channelaxis = emcuts.channelaxis
    if epochaxis is None:
        epochaxis = emcuts.epochaxis
    if dtype is None:
        dtype = emcuts.samples.dtype
    arnode = _newh5emcuts(h5group=signalgroup, name='samples', dtype=dtype,
                          ntimesamples=emcuts.ntimesamples, fs=emcuts.fs, 
                          origintime=emcuts.origintime, starttime=emcuts.starttime, 
                          nchannels = emcuts.nchannels, epochaxis=epochaxis, 
                          epochs=emcuts.epochs, channels=emcuts.channelnames, 
                          geometry=emcuts.geometry, channelaxis=channelaxis, 
                          overwrite=overwrite, extensible=extensible, 
                          expectedrows=expectedrows, complib=complib, complevel=complevel, 
                          chunkshape=chunkshape, attrs=attrs)
    
    epochs = arnode._v_parent._f_getChild("samples_epochs")
   
    # loop over big chunks of first axis
    chunksize = 8*1024*1024 # 8 Mb
    if epochaxis == 0:
        epochsize = (emcuts.ntimesamples*emcuts.nchannels*emcuts.dtype.itemsize)
        nepochsperchunk = int(chunksize//float(epochsize))
        nepochsperchunk = max(nepochsperchunk,1)
        nchunks, remainder = divmod(emcuts.nepochs,nepochsperchunk)
        start = 0
        if channelaxis == 1:
            trans = ['epoch','channel','time']
        else: # channelaxis == 2
            trans = ['epoch','time','channel']
        for i in range(nchunks):
            arnode[start:start+nepochsperchunk,:] =\
                  emcuts[:,:,start:start+nepochsperchunk].samples.transpose(trans).get(copy=False)
            start = start + nepochsperchunk
        if remainder > 0:
            arnode[start:start+remainder,:] =\
                  emcuts[:,:,start:start+remainder].samples.transpose(trans).get(copy=False)
    elif channelaxis == 0:
        channelsize = (emcuts.ntimesamples*emcuts.nepochs*emcuts.dtype.itemsize)
        nchannelsperchunk = int(chunksize//float(channelsize))
        nchannelsperchunk = max(nchannelsperchunk,1)
        nchunks, remainder = divmod(emcuts.nchannels,nchannelsperchunk)
        start = 0
        if epochaxis == 1:
            trans = ['channel','epoch','time']
        else: # epochaxis == 2
            trans = ['channel','time','epoch']
        for i in range(nchunks):
            arnode[start:start+nchannelsperchunk,:,:] =\
                  emcuts[:,start:start+nchannelsperchunk].samples.transpose(trans).get(copy=False)
            start = start + nchannelsperchunk
        if remainder > 0:
            arnode[start:start+remainder,:,:] =\
                  emcuts[:,start:start+remainder].samples.transpose(trans).get(copy=False)
    else: # timeaxis == 0
        timesize = (emcuts.nchannels*emcuts.nepochs*emcuts.dtype.itemsize)
        ntimesperchunk = int(chunksize//float(timesize))
        ntimesperchunk = max(ntimesperchunk,1)
        nchunks, remainder = divmod(emcuts.ntimesamples,ntimesperchunk)
        start = 0
        if epochaxis == 1:
            trans = ['time','epoch','channel']
        else: # epochaxis == 2
            trans = ['time','channel','epoch']
        for i in range(nchunks):
            arnode[start:start+ntimesperchunk,:,:] =\
                  emcuts[start:start+ntimesperchunk].samples.transpose(trans).get(copy=False)
            start = start + ntimesperchunk
        if remainder > 0:
            arnode[start:start+remainder,:,:] =\
                  emcuts[start:start+remainder].samples.transpose(trans).get(copy=False)
    arnode.flush()
    dtype = arnode.dtype
    return emcuts._instantiatefromself(samples=arnode, epochaxis=epochaxis, epochs=epochs,
                                     dtype=dtype, channelaxis=channelaxis, copy=False)

def _save_epoched(epoched, h5group, name, overwrite=False, attrs={}):
    h5file = h5group._v_file
    if overwrite:
        if h5group._v_children.has_key(name):
            h5group._v_children[name]._f_remove()
            h5file.flush()
    table = h5file.createTable(where=h5group, name=name, 
                               description=epoched.epochs)
    attrs['origintime'] = epoched.origintime
    attrs['duration'] =  epoched.duration
    for attrname, value in attrs.items():
        table._f_setAttr(attrname, value)
        table.flush()
    return table

#FIXME!
def _save_multichannel(multichannel, h5group, name, overwrite=False, attrs={}):
    h5file = h5group._v_file
    if overwrite:
        if h5group._v_children.has_key(name):
            h5group._v_children[name]._f_remove()
            h5file.flush()
    table = h5file.createTable(where=h5group, name=name, 
                               description=multichannel.channels)
    for attrname, value in attrs.items():
        table._f_setAttr(attrname, value)
    multichannel.geometry.saveh5(table)
    table.flush()
    return table