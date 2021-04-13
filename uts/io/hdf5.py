import os.path
import warnings
import numpy as np
import tables as tb
from contextlib import contextmanager
from ..core.uniformtime import UniformTimeSeries, EpochedUniformTimeSeries, \
    MultiChannelUniformTimeSeries, EpochedMultiChannelUniformTimeSeries, \
    isuts, iseuts, ismcuts, isemcuts
from ..core.multichannel import MultiChannel
from ..core.geometry import Geometry
from ..core.epoched import Epoched
from ..core.check_arg import check_arg, check_int
from ..core.utils import fit_frames
from ..core.utils import timeparams
from .utils import tempdir

# uts, euts, mcuts and emcuts signals are saved as a h5 *Group*. The name/path of that 
# group can be determined by the user and is the 'name' of that signal. The h5 Group 
# should contain at least one array named 'samples', but may also contain the tables
# 'epochs' or 'channels'

# FIXME allow for new startdatetime parameter

__all__ = ['openh5', 'saveh5', 'loadh5', 'cacheh5', 'removeh5', 'renameh5', 'h5fillnew',
           'h5filllike', 'saveh5_itered', 'cacheh5_itered']

SAMPLEARRAYNAME = 'samples'
EPOCHTABLENAME = 'epochs'
CHANNELTABLENAME = 'channels'

def h5attr_to_geometry(attr, pointnames=None):
    from uts.core.geometry import RectangularLattice

    code = '%s(' % attr[0]  #class
    for key, item in attr[1].items():  #parameters
        code += '%s=%s,' % (key, item)
    if not pointnames is None:
        code += '%s=%s,' % ('pointnames', list(pointnames))
    code = code[:-1] + ')'
    return eval(code)

def _h5userattrs_to_dict(h5node, attrs=None):
    """
    Gets h5 user attrs as dict. If attrs dict is provided as parameter,
    it will be updated. If item is present already, it will be overwritten.
    """
    if attrs is None:
        attrs = {}
    for attrname in h5node._v_attrs._v_attrnamesuser:
        attr = h5node._v_attrs[attrname]
        if hasattr(attr, 'tolist'):
            attr = attr.tolist()
        attrs[attrname] = attr
    return attrs

def _isgenerator(iterable):
    return hasattr(iterable, '__iter__') and not hasattr(iterable, '__len__')

def _remove(h5fileobject, h5path):
    f = h5fileobject
    if h5path == '/':
        for child in f.root._v_children.keys():
            f.remove_node('/', child, recursive=True)
    else:
        f.remove_node(h5path, recursive=True)

def removeh5(filename, signalname):
    with tb.open_file(filename, 'a') as f:
        _remove(h5fileobject=f, h5path=signalname)

def _rename(h5fileobject, signalname, newsignalname):
    h5fileobject.rename_node(signalname, newsignalname, overwrite=False)

def renameh5(filename, signalname, newsignalname):
    with tb.open_file(filename, 'a') as f:
        _rename(h5fileobject=f, signalname=signalname, newsignalname=newsignalname)


@contextmanager
def _create_h5savepoint(filename, h5path='/', overwrite=False, mode='a'):
    with tb.open_file(filename, mode=mode) as f:
        if h5path in f and overwrite:
            _remove(h5fileobject=f, h5path=h5path)
        grouppath = h5path.split('/')
        '/'.join(grouppath[:-1])
        h5group = f.create_group(where='/'.join(grouppath[:-1]),
                                 name=grouppath[-1],
                                 createparents=True)
        yield h5group

def saveh5(s, signalname='/', filename=None, channelaxis=None,
           epochaxis=None, overwrite=False, extensible=False,
           expectedrows=None, dtype=None, complib='zlib', complevel=5,
           chunkshape=None, warn=True):
    '''
        Saves a signal to h5 file. This function can also be used for generator objects that
        yield signals.

        Parameters
        ----------
        s: may be a ((epoched) multi-channel) time sampled signal or a 
          generator that produces such signals. The latter can be handy if one 
          has very large signals on-disk, that don't fit into working memory.
        signalname: a valid name that can later be used to retrieve the
          signal. Note that this is a hdf5 groupname, with path, so it should
          start with '/'
        channelaxis:
        epochaxis:
        overwrite: if True overwrites an existing signal if it has the same
          name.
        extensible: if True makes the signal that is stored enlargeable. That
          is: samples can be appended later. If signal is a generator, the saved
          signal is always enlargeable.

    '''
    with _create_h5savepoint(filename=filename,
                             h5path=signalname,
                             overwrite=overwrite) as h5group:
        _save(s=s, h5group=h5group,
              channelaxis=channelaxis, epochaxis=epochaxis,
              overwrite=overwrite, extensible=extensible,
              expectedrows=expectedrows, dtype=dtype, complib=complib,
              complevel=complevel, chunkshape=chunkshape, warn=warn)

def _save(s, h5group, channelaxis=None, epochaxis=None,
          overwrite=False, extensible=False, expectedrows=None, dtype=None,
          complib='zlib', complevel=5, chunkshape=None, warn=True):
    if _isgenerator(s):
        s1 = next(s)
        if (not extensible) and warn:
            warnings.warn(
                "Saving generator to h5 array: setting extensible to True")
        extensible = True
        # FIXME use standard function from uniformtime
        if not isinstance(s1, (EpochedMultiChannelUniformTimeSeries,
                               MultiChannelUniformTimeSeries,
                               UniformTimeSeries)):
            raise TypeError("signal is not a sampled signal")
        h5mcss = _save(s1, h5group=h5group,
                       channelaxis=channelaxis, epochaxis=epochaxis,
                       overwrite=overwrite,
                       extensible=extensible, expectedrows=expectedrows,
                       dtype=dtype,
                       complib=complib, complevel=complevel,
                       chunkshape=chunkshape)
        for sn in s:
            h5mcss.append(sn)
        return h5mcss
    elif isuts(s):
        return _save_uts(s, h5group=h5group, overwrite=overwrite,
                         extensible=extensible, expectedrows=expectedrows,
                         dtype=dtype,
                         complib=complib, complevel=complevel,
                         chunkshape=chunkshape)
    elif ismcuts(s):
        return _save_mcuts(s, h5group=h5group, channelaxis=channelaxis,
                           overwrite=overwrite, extensible=extensible,
                           expectedrows=expectedrows,
                           dtype=dtype, complib=complib, complevel=complevel,
                           chunkshape=chunkshape)
    elif iseuts(s):
        return _save_euts(s, h5group=h5group, epochaxis=epochaxis,
                          overwrite=overwrite, extensible=extensible,
                          expectedrows=expectedrows,
                          dtype=dtype, complib=complib, complevel=complevel,
                          chunkshape=chunkshape)
    elif isemcuts(s):
        return _save_emcuts(s, h5group=h5group, channelaxis=channelaxis,
                            epochaxis=epochaxis, overwrite=overwrite,
                            extensible=extensible, expectedrows=expectedrows,
                            dtype=dtype,
                            complib=complib, complevel=complevel,
                            chunkshape=chunkshape)
    else:
        raise TypeError(
            "signal is not a sampled signal, or generator (%s)" % type(s))

def saveh5_itered(smodel, gen, name='/', filename=None, ntimesamples=None,
                  fs=None, value=0.0, dtype=None, channelaxis=None,
                  epochaxis=None, overwrite=False, extensible=False,
                  expectedrows=None, filters=None, chunkshape=None, complib=None,
                  complevel=None, attrs=None):
    if attrs is None:
        attrs = smodel.attrs
    else:
        attrs.update(smodel.attrs)
    with h5filllike(smodel, name=name, filename=filename,
                    ntimesamples=ntimesamples, fs=fs, value=value, dtype=dtype,
                    channelaxis=channelaxis,
                    epochaxis=epochaxis, overwrite=overwrite,
                    extensible=extensible,
                    complevel=complevel, attrs=attrs) as newh5s:
        start = 0
        for frame in gen:
            end = start + frame.ntimesamples
            index = newh5s._get_index(slice(start, end))
            newh5s.samples.h5array[index] = frame.samples.transpose(
                newh5s.samples.axisorder).get(False)
            start += frame.ntimesamples

@contextmanager
def cacheh5(s, channelaxis=None, epochaxis=None,
            extensible=True, expectedrows=None, dtype=None, complib='blosc',
            complevel=5, chunkshape=None, path=None, report=False,
            signalname='/cached', filename='cached.h5', warn=True, keep=False):


    with tempdir(dir=path, keep=keep, report=report) as tempdirname:
        filename = os.path.join(tempdirname, filename)
        saveh5(s, signalname=signalname, filename=filename,
               channelaxis=channelaxis, epochaxis=epochaxis,
               overwrite=False, extensible=extensible, expectedrows=expectedrows,
               dtype=dtype, complib=complib, complevel=complevel,
               chunkshape=chunkshape,
               warn=warn)
        s = loadh5(signalname=signalname, filename=filename)
        s.path = tempdirname
        s.filename = filename
        yield s

@contextmanager
def cacheh5_itered(smodel, gen, ntimesamples=None, fs=None, channelaxis=None,
                   epochaxis=None,
                   extensible=False, expectedrows=None, dtype=None,
                   complib='blosc',
                   complevel=5, chunkshape=None, path=None, report=False,
                   signalname='/cached', filename='cached.h5', attrs=None,
                   mode='a', keep=False):
    with tempdir(dir=path, keep=keep, report=report) as tempdirname:
        filename = os.path.join(tempdirname, filename)
        if attrs is None:
            attrs = smodel.attrs
        else:
            attrs.update(smodel.attrs)
        with h5filllike(smodel, name=signalname, filename=filename,
                        ntimesamples=ntimesamples, fs=fs, value=0.0,
                        dtype=dtype, channelaxis=channelaxis,
                        epochaxis=epochaxis, extensible=extensible,
                        expectedrows=expectedrows, chunkshape=chunkshape,
                        complib=complib,
                        complevel=complevel, attrs=attrs) as newh5s:
            start = 0
            for frame in gen:
                end = start + frame.ntimesamples
                index = newh5s._get_index(slice(start, end))
                newh5s.samples.h5array[index] = frame.samples.transpose(
                    newh5s.samples.axisorder).get(False)
                start += frame.ntimesamples
        with openh5(signalname=signalname, filename=filename, mode=mode) as s:
            s.path = tempdirname
            s.filename = filename
            yield s


def loadh5(signalname='/', filename=None, start=None, end=None):
    with openh5(signalname=signalname, filename=filename) as s:
        return s[start:end]

@contextmanager
def openh5(signalname='/', filename=None, mode='r'):
    with tb.open_file(filename, mode=mode) as f:
        h5group = f.get_node(signalname)
        yield _open(h5group, mode=mode)

def _open(h5group, mode='r'):
    f = h5group._v_file
    samples = h5array = f.get_node(h5group, SAMPLEARRAYNAME)
    attrs = _h5userattrs_to_dict(h5group)
    attrs = _h5userattrs_to_dict(samples, attrs=attrs)
    fs = h5array.attrs.fs
    attrs.pop('fs')
    origintime = h5array.attrs.origintime
    attrs.pop('origintime')
    if 'startdatetime' in attrs:
        startdatetime = attrs['startdatetime']
    elif 'starttime' in attrs: # compatibility old versions
        starttime = attrs.pop('starttime')
        if starttime != 0.0:
            startdatetime = np.datetime64(int(round(starttime * 1e9)), 'ns')
        else:
            startdatetime = 'NaT'
    if len(h5array.shape) == 1:  # uts
        signal = UniformTimeSeries(samples=samples, fs=fs, startdatetime=startdatetime,
                                   origintime=origintime, copy=False,
                                   dtype=None, attrs=attrs)
    #FIXME use the attr to find name of channel and epoch tables, not hard-code strings
    elif len(h5array.shape) == 2:  # euts or mcuts
        if "channelaxis" in h5array.attrs._v_attrnames:  # mcuts
            channelaxis = h5array.attrs.channelaxis
            attrs.pop('channelaxis')
            attrs.pop('channeltable')
            # there can be a table with channels info
            if CHANNELTABLENAME in h5group:
                channeltable = f.get_node(h5group, CHANNELTABLENAME)
                channels = channeltable[:]
                channelnames = channels['channelname']
                if "geometry" in channeltable.attrs._v_attrnames:
                    pointnames = channeltable.attrs.pointnames
                    g = h5attr_to_geometry(channeltable.attrs.geometry,
                                           pointnames=pointnames)
                else:
                    g = None
            else:
                raise IOError(
                    "no channels found in %s" % h5array._v_pathname)
                #channels = None
            signal = MultiChannelUniformTimeSeries(samples=samples,
                                                   fs=fs,
                                                   channelnames=channelnames,
                                                   startdatetime=startdatetime,
                                                   origintime=origintime,
                                                   copy=False, dtype=None,
                                                   channelaxis=channelaxis,
                                                   geometry=g, attrs=attrs)
        else:  #euts
            epochaxis = h5array.attrs.epochaxis
            attrs.pop('epochaxis')
            attrs.pop('epochtable')
            # there should be a table with epoch info
            if EPOCHTABLENAME in h5group:
                epochtable = f.get_node(h5group, EPOCHTABLENAME)
            else:
                raise IOError(
                    "no epochs table found in %s" % h5array._v_pathname)
            signal = EpochedUniformTimeSeries(samples=samples,
                                              fs=fs, epochs=epochtable,
                                              origintime=origintime,
                                              copy=False, dtype=None,
                                              epochaxis=epochaxis, attrs=attrs)
    else:  # len(h5array.shape) == 3: # emcuts
        epochaxis = h5array.attrs.epochaxis
        attrs.pop('epochaxis')
        attrs.pop('epochtable')
        channelaxis = h5array.attrs.channelaxis
        attrs.pop('channelaxis')
        attrs.pop('channeltable')
        # there should be a table with epoch info
        if EPOCHTABLENAME in h5group:
            epochtable = f.get_node(h5group, EPOCHTABLENAME)
        else:
            raise IOError("no epochs table found in %s" % h5array._v_pathname)
        # there can be a table with channels info
        if CHANNELTABLENAME in h5group:
            channeltable = f.get_node(h5group, CHANNELTABLENAME)
            channels = channeltable[:]
            channelnames = channels['channelname']
            if "geometry" in channeltable.attrs._v_attrnames:
                pointnames = channeltable.attrs.pointnames
                g = h5attr_to_geometry(channeltable.attrs.geometry,
                                       pointnames=pointnames)
            else:
                g = None
        else:
            raise IOError("no channels found in %s" % h5array._v_pathname)
        signal = EpochedMultiChannelUniformTimeSeries(samples=samples,
                                                      fs=fs, channelnames=channels,
                                                      origintime=origintime,
                                                      copy=False, dtype=None,
                                                      epochs=epochtable,
                                                      geometry=g,
                                                      channelaxis=channelaxis,
                                                      epochaxis=epochaxis,
                                                      attrs=attrs)
    return signal

def _newh5uts(h5group, dtype, ntimesamples, fs, origintime,
              startdatetime='NaT', dflt=0.0,
              overwrite=False, extensible=False, expectedrows=None,
              complib='zlib', complevel=5, chunkshape=None, attrs=None):
    """
    Returns an pytables CArray or EArray node with dimensions of ntimesamples, 
    filled with zeros.

    """
    fs = check_arg(fs, 'fs')
    origintime = check_arg(origintime, 'time')
    startdatetime = str(np.datetime64(startdatetime, 'ns'))
    ntimesamples = check_int(ntimesamples, negative=False)
    dtype = np.dtype(dtype)
    if attrs is None:
        attrs = {}
    f = h5group._v_file
    if overwrite:
        if SAMPLEARRAYNAME in h5group:
            f.get_node(h5group, SAMPLEARRAYNAME)._f_remove()
            f.flush()
    filters = tb.Filters(complevel=complevel, complib=complib)
    atom = tb.Atom.from_dtype(dtype, dflt=dflt)
    if extensible:
        arnode = f.create_earray(where=h5group, name=SAMPLEARRAYNAME,
                                 atom=atom,
                                 shape=(0,), filters=filters,
                                 expectedrows=expectedrows,
                                 chunkshape=chunkshape)
        framesize = int(8 * 1024 * 1024 / dtype.itemsize)  # 8 Mb chunks
        framesize = max(framesize, 1)
        nframes, newsize, remainder = fit_frames(totalsize=ntimesamples,
                                                 framesize=framesize,
                                                 stepsize=framesize)
        zf = np.zeros(framesize, dtype=dtype)
        for i in range(nframes):
            arnode.append(zf)
        arnode.append(np.zeros(remainder, dtype=dtype))
    else:
        shape = (ntimesamples,)
        arnode = f.create_carray(where=h5group, name=SAMPLEARRAYNAME,
                                 atom=atom, shape=shape, title='',
                                 filters=filters,
                                 chunkshape=chunkshape, byteorder=None,
                                 createparents=False)
    for attrname, value in attrs.items():
        h5group._f_setattr(attrname, value)
    h5group._f_setattr('fs', fs)
    h5group._f_setattr('startdatetime', startdatetime)
    h5group._f_setattr('origintime', origintime)
    for attrname, value in attrs.items():
        arnode.set_attr(attrname, value)
    arnode.set_attr('fs', fs)
    arnode.set_attr('startdatetime', startdatetime)
    arnode.set_attr('origintime', origintime)

    arnode.flush()

    return arnode

def _newh5mcuts(h5group, dtype, ntimesamples, fs, origintime,
                channelaxis, nchannels, channels, geometry, startdatetime='NaT',
                dflt=0.0, overwrite=False, extensible=False, expectedrows=None,
                complib='zlib', complevel=5, chunkshape=None, attrs=None):
    """
    Returns an pytables CArray or EArray node with dimensions of self.samples, 
    filled with zeros.

    """

    fs = check_arg(fs, 'fs')
    origintime = check_arg(origintime, 'time')
    startdatetime = str(np.datetime64(startdatetime, 'ns'))
    channelaxis = check_arg(channelaxis, 'channelaxis')
    mc = MultiChannel(nchannels=nchannels, channelnames=channels,
                      geometry=geometry)
    ntimesamples = check_int(ntimesamples, negative=False)
    dtype = np.dtype(dtype)
    if attrs is None:
        attrs = {}
    f = h5group._v_file
    if overwrite:
        if SAMPLEARRAYNAME in h5group:
            h5group._v_children[SAMPLEARRAYNAME]._f_remove()
            f.flush()
        if CHANNELTABLENAME in h5group._v_children:
            h5group._v_children[CHANNELTABLENAME]._f_remove()
            f.flush()
    filters = tb.Filters(complevel=complevel, complib=complib)
    atom = tb.Atom.from_dtype(dtype, dflt=dflt)
    if extensible:
        if channelaxis is None:
            channelaxis = 0
        if channelaxis != 0:
            raise ValueError("Extensible mcuts can only have channelaxis 0")
        shape = (mc.nchannels, 0)
        arnode = f.create_earray(where=h5group, name=SAMPLEARRAYNAME,
                                 atom=atom,
                                 shape=shape, filters=filters,
                                 expectedrows=expectedrows,
                                 chunkshape=chunkshape)
        framesize = int(
            8 * 1024 * 1024 / (dtype.itemsize * mc.nchannels))  # 8 Mb chunks
        framesize = max(framesize, 1)
        nframes, newsize, remainder = fit_frames(totalsize=ntimesamples,
                                                 framesize=framesize,
                                                 stepsize=framesize)
        zf = np.zeros((mc.nchannels, framesize), dtype=dtype)
        for i in range(nframes):
            arnode.append(zf)
        arnode.append(np.zeros((mc.nchannels, remainder), dtype=dtype))
    else:
        if channelaxis is None:
            channelaxis = 1  # most likely efficient if retrieving time chunks
        else:
            if channelaxis not in (0, 1):
                raise ValueError("channelaxis should be 0 or 1")
        if channelaxis == 0:
            shape = (mc.nchannels, ntimesamples)
        else:
            shape = (ntimesamples, mc.nchannels)
        arnode = f.create_carray(where=h5group, name=SAMPLEARRAYNAME,
                                 atom=atom, shape=shape, title='',
                                 filters=filters,
                                 chunkshape=chunkshape, byteorder=None,
                                 createparents=False)

    for attrname, value in attrs.items():
        h5group._f_setattr(attrname, value)
    h5group._f_setattr('fs', fs)
    h5group._f_setattr('startdatetime', startdatetime)
    h5group._f_setattr('origintime', origintime)
    h5group._f_setattr('channelaxis', channelaxis)
    h5group._f_setattr('channeltable', CHANNELTABLENAME)
    for attrname, value in attrs.items():
        arnode.set_attr(attrname, value)
    arnode.set_attr('fs', fs)
    arnode.set_attr('startdatetime', startdatetime)
    arnode.set_attr('origintime', origintime)
    arnode.set_attr('channelaxis', channelaxis)
    arnode.set_attr('channeltable', CHANNELTABLENAME)

    _save_multichannel(multichannel=mc, h5group=h5group, name=CHANNELTABLENAME,
                       overwrite=overwrite)
    arnode.flush()
    return arnode

def _newh5euts(h5group, dtype, ntimesamples, fs, origintime,
               epochaxis, epochs, startdatetime='NaT', dflt=0.0, overwrite=False,
               extensible=False,
               expectedrows=None, complib='zlib', complevel=5, chunkshape=None,
               attrs=None):
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
    startdatetime = str(np.datetime64(startdatetime, 'ns'))
    epochaxis = check_arg(epochaxis, 'epochaxis')
    ep = Epoched(duration=ntimesamples / fs, epochs=epochs,
                 origintime=origintime)
    dtype = np.dtype(dtype)
    if attrs is None:
        attrs = {}
    f = h5group._v_file
    if overwrite:
        if SAMPLEARRAYNAME in h5group:
            h5group._v_children[SAMPLEARRAYNAME]._f_remove()
            f.flush()
        if EPOCHTABLENAME in h5group:
            h5group._v_children[EPOCHTABLENAME]._f_remove()
            f.flush()
    filters = tb.Filters(complevel=complevel, complib=complib)
    atom = tb.Atom.from_dtype(dtype, dflt=dflt)
    if extensible:
        if epochaxis is None:
            epochaxis = 1
        if epochaxis != 1:
            raise ValueError("Extensible euts can only have epochaxis 1")
        shape = (ntimesamples, 0)
        arnode = f.create_earray(where=h5group, name=SAMPLEARRAYNAME,
                                 atom=atom,
                                 shape=shape, filters=filters,
                                 expectedrows=expectedrows,
                                 chunkshape=chunkshape)
        nepochs = ep.nepochs
        zf = np.zeros((ntimesamples, 1), dtype=dtype)
        for i in range(ep.nepochs):
            arnode.append(zf)
    else:
        if epochaxis is None:
            epochaxis = 0  # most likely efficient if retrieving whole epochs
        else:
            if epochaxis not in (0, 1):
                raise ValueError("epochaxis should be 0 or 1")
        if epochaxis == 0:
            shape = (ep.nepochs, ntimesamples)
        else:
            shape = (ntimesamples, ep.nepochs)
        arnode = f.create_carray(where=h5group, name=SAMPLEARRAYNAME,
                                 atom=atom, shape=shape, title='',
                                 filters=filters,
                                 chunkshape=chunkshape, byteorder=None,
                                 createparents=False)

    for attrname, value in attrs.items():
        h5group._f_setattr(attrname, value)
    h5group._f_setattr('fs', fs)
    h5group._f_setattr('startdatetime', startdatetime)
    h5group._f_setattr('origintime', origintime)
    h5group._f_setattr('epochaxis', epochaxis)
    h5group._f_setattr('epochtable', EPOCHTABLENAME)
    for attrname, value in attrs.items():
        arnode.set_attr(attrname, value)
    arnode.set_attr('fs', fs)
    arnode.set_attr('startdatetime', startdatetime)
    arnode.set_attr('origintime', origintime)
    arnode.set_attr('epochaxis', epochaxis)
    arnode.set_attr('epochtable', EPOCHTABLENAME)

    _save_epoched(epoched=ep, h5group=h5group, name=EPOCHTABLENAME,
                  overwrite=overwrite)
    arnode.flush()
    return arnode

def _newh5emcuts(h5group, dtype, ntimesamples, fs, origintime,
                 epochaxis, channelaxis, nchannels, channels,
                 geometry, epochs, startdatetime='NaT', dflt=0.0,
                 overwrite=False, extensible=False,
                 expectedrows=None,
                 complib='zlib', complevel=5, chunkshape=None, attrs=None):
    """
    Returns an pytables CArray or EArray node with dimensions of self.samples, 
    filled with zeros.

    """

    fs = check_arg(fs, 'fs')
    ntimesamples = check_int(ntimesamples, negative=False)
    origintime = check_arg(origintime, 'time')
    startdatetime = str(np.datetime64(startdatetime, 'ns'))
    channelaxis = check_arg(channelaxis, 'channelaxis')
    mc = MultiChannel(nchannels=nchannels, channelnames=channels,
                      geometry=geometry)
    epochaxis = check_arg(epochaxis, 'epochaxis')
    ep = Epoched(duration=ntimesamples / fs, epochs=epochs,
                 origintime=origintime)
    dtype = np.dtype(dtype)
    if attrs is None:
        attrs = {}
    f = h5group._v_file
    if overwrite:
        if SAMPLEARRAYNAME in h5group:
            h5group._v_children[SAMPLEARRAYNAME]._f_remove()
            f.flush()
        if CHANNELTABLENAME in h5group:
            h5group._v_children[CHANNELTABLENAME]._f_remove()
            f.flush()
        if EPOCHTABLENAME in h5group:
            h5group._v_children[EPOCHTABLENAME]._f_remove()
            f.flush()
    filters = tb.Filters(complevel=complevel, complib=complib)
    atom = tb.Atom.from_dtype(dtype, dflt=dflt)
    if extensible:
        if epochaxis is None:
            epochaxis = 2
        if epochaxis != 2:
            raise ValueError("epochaxis should be 2 for extensible emcuts")
        if channelaxis is None:
            channelaxis = 1
        if not channelaxis in (0, 1):
            raise ValueError(
                "channelaxis should be 0 or 1 for extensible emcuts")
        if channelaxis == 1:
            shape = (ntimesamples, mc.nchannels, 0)
        else:
            shape = (mc.nchannels, ntimesamples, 0)
        arnode = f.create_earray(where=h5group, name=SAMPLEARRAYNAME,
                                 atom=atom,
                                 shape=shape, filters=filters,
                                 expectedrows=expectedrows,
                                 chunkshape=chunkshape)
        nepochs = ep.nepochs
        zf = np.zeros((shape[0], shape[1], 1), dtype=dtype)
        for i in range(nepochs):
            arnode.append(zf)
    else:
        # first figure out which axis is where. The default is to put
        # epoch at low axes and channel at high
        shape = [ntimesamples, ntimesamples, ntimesamples]
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
        arnode = f.create_carray(where=h5group, name=SAMPLEARRAYNAME,
                                 atom=atom, shape=shape, title='',
                                 filters=filters,
                                 chunkshape=chunkshape, byteorder=None,
                                 createparents=False)

    for attrname, value in attrs.items():
        h5group._f_setattr(attrname, value)
    h5group._f_setattr('fs', fs)
    h5group._f_setattr('startdatetime', startdatetime)
    h5group._f_setattr('origintime', origintime)
    h5group._f_setattr('channelaxis', channelaxis)
    h5group._f_setattr('channeltable', CHANNELTABLENAME)
    h5group._f_setattr('epochaxis', epochaxis)
    h5group._f_setattr('epochtable', EPOCHTABLENAME)
    for attrname, value in attrs.items():
        arnode.set_attr(attrname, value)
    arnode.set_attr('fs', fs)
    arnode.set_attr('startdatetime', startdatetime)
    arnode.set_attr('origintime', origintime)
    arnode.set_attr('channelaxis', channelaxis)
    arnode.set_attr('channeltable', CHANNELTABLENAME)
    arnode.set_attr('epochaxis', epochaxis)
    arnode.set_attr('epochtable', EPOCHTABLENAME)

    _save_epoched(epoched=ep, h5group=h5group, name=EPOCHTABLENAME,
                  overwrite=overwrite)
    _save_multichannel(multichannel=mc, h5group=h5group, name=CHANNELTABLENAME,
                       overwrite=overwrite)
    arnode.flush()
    return arnode

def _create_signalgroup(h5group, name, overwrite):
    f = h5group._v_file
    if overwrite and (name in h5group):
        f.get_node(h5group, name)._f_remove(recursive=True)
        f.flush()
    return f.create_group(h5group, name)

def _save_uts(uts, h5group, overwrite=False,
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

    if dtype is None:
        dtype = uts.samples.dtype
    arnode = _newh5uts(h5group=h5group,
                       ntimesamples=uts.ntimesamples, fs=uts.fs,
                       origintime=uts.origintime, startdatetime=uts.startdatetime,
                       dtype=dtype,
                       overwrite=overwrite, extensible=extensible,
                       expectedrows=expectedrows,
                       complib=complib, complevel=complevel,
                       chunkshape=chunkshape, attrs=uts.attrs)
    # fill it with our samples; we take chunks of maximum 8 Mb
    framesize = int(8 * 1024 * 1024 / uts.samples.dtype.itemsize)
    framesize = max(1, framesize)
    start = 0
    for i, frame in enumerate(uts.iter_frames(framesize=framesize)):
        samples = frame.samples.get(copy=False)
        if dtype != samples.dtype:
            samples = samples.astype(dtype)
        arnode[start:start + frame.ntimesamples] = samples
        start += frame.ntimesamples
    arnode.flush()
    return uts._instantiatefromself(samples=arnode, dtype=arnode.dtype,
                                    copy=False)

def _save_mcuts(mcuts, h5group, overwrite=False, channelaxis=None,
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
    if extensible:
        if channelaxis == 1:
            raise ValueError("Extensible mcuts must have channelaxis 0")
        else:
            channelaxis = 0  # in case the param wasn't provided
    if channelaxis is None:
        channelaxis = mcuts.channelaxis
    if dtype is None:
        dtype = mcuts.samples.dtype
    arnode = _newh5mcuts(h5group=h5group, dtype=dtype,
                         ntimesamples=mcuts.ntimesamples, fs=mcuts.fs,
                         origintime=mcuts.origintime,
                         startdatetime=mcuts.startdatetime,
                         nchannels=mcuts.nchannels,
                         channels=mcuts.channelnames, geometry=mcuts.geometry,
                         channelaxis=channelaxis, overwrite=overwrite,
                         extensible=extensible, expectedrows=expectedrows,
                         complib=complib, complevel=complevel,
                         chunkshape=chunkshape,
                         attrs=mcuts.attrs)

    # loop over big chunks of first axis
    chunksize = 1 * 1024 * 1024  # 8 Mb
    if channelaxis == 0:
        channelsize = (mcuts.ntimesamples * mcuts.dtype.itemsize)
        nchannelsperchunk = int(chunksize // float(channelsize))
        nchannelsperchunk = max(nchannelsperchunk, 1)
        nchunks, remainder = divmod(mcuts.nchannels, nchannelsperchunk)
        start = 0
        for i in range(nchunks):
            arnode[start:start + nchannelsperchunk, :] = \
                mcuts[:, start:start + nchannelsperchunk].samples.transpose(
                    ['channel', 'time'],
                    ignore_missingaxes=True).get(copy=False)
            start = start + nchannelsperchunk
        if remainder > 0:
            arnode[start:start + remainder, :] = \
                mcuts[:, start:start + remainder].samples.transpose(
                    ['channel', 'time'],
                    ignore_missingaxes=True).get(copy=False)
    else:  # channelaxis == 1
        timesize = (mcuts.nchannels * mcuts.dtype.itemsize)
        ntimesperchunk = int(chunksize // float(timesize))
        ntimesperchunk = max(ntimesperchunk, 1)
        nchunks, remainder = divmod(mcuts.ntimesamples, ntimesperchunk)
        start = 0
        for i in range(nchunks):
            arnode[start:start + ntimesperchunk, :] = \
                mcuts[start:start + ntimesperchunk].samples.transpose(
                    ['time', 'channel']).get(copy=False)
            start = start + ntimesperchunk
        if remainder > 0:
            arnode[start:start + remainder, :] = \
                mcuts[start:start + remainder, :].samples.transpose(
                    ['time', 'channel']).get(copy=False)

    arnode.flush()
    dtype = arnode.dtype
    return mcuts._instantiatefromself(samples=arnode, channelaxis=channelaxis,
                                      dtype=dtype, copy=False)

def _save_euts(euts, h5group, epochaxis=None, overwrite=False,
               extensible=False, expectedrows=None, dtype=None,
               complib='zlib', complevel=5, chunkshape=None):
    '''
    Saves the signal array in a hdf5 file.

    parameter:

    - h5group: the group object under which the signal array should be saved

    returns:

    the h5 arraynode in which the array is saved

    '''
    if dtype is None:
        dtype = euts.dtype
    if epochaxis is None:
        epochaxis = euts.epochaxis
    arnode = _newh5euts(h5group=h5group, dtype=dtype,
                        ntimesamples=euts.ntimesamples, fs=euts.fs,
                        origintime=euts.origintime, startdatetime=euts.startdatetime,
                        epochaxis=epochaxis, epochs=euts.epochs,
                        overwrite=overwrite, extensible=False,
                        expectedrows=None, complib=complib,
                        complevel=complevel,
                        chunkshape=None, attrs=euts.attrs)
    epochs = h5group._f_get_child(EPOCHTABLENAME)

    # loop over big chunks of first axis
    chunksize = 4 * 1024 * 1024  # 8 Mb
    if epochaxis == 0:
        epochsize = (euts.ntimesamples * euts.dtype.itemsize)
        nepochsperchunk = int(chunksize // float(epochsize))
        nepochsperchunk = max(nepochsperchunk, 1)
        nchunks, remainder = divmod(euts.nepochs, nepochsperchunk)
        start = 0
        for i in range(nchunks):
            arnode[start:start + nepochsperchunk, :] = \
                euts[:, start:start + nepochsperchunk].samples.transpose(
                    ['epoch', 'time']).get(copy=False)
            start = start + nepochsperchunk
        if remainder > 0:
            arnode[start:start + remainder, :] = \
                euts[:, start:start + remainder].samples.transpose(
                    ['epoch', 'time']).get(copy=False)
    else:  # epochaxis == 1
        timesize = (euts.nepochs * euts.dtype.itemsize)
        ntimesperchunk = int(chunksize // float(timesize))
        ntimesperchunk = max(ntimesperchunk, 1)
        nchunks, remainder = divmod(euts.ntimesamples, ntimesperchunk)
        start = 0
        for i in range(nchunks):
            arnode[start:start + ntimesperchunk, :] = \
                euts[start:start + ntimesperchunk].samples.transpose(
                    ['time', 'epoch']).get(copy=False)
            start = start + ntimesperchunk
        if remainder > 0:
            arnode[start:start + remainder, :] = \
                euts[start:start + remainder, :].samples.transpose(
                    ['time', 'epoch']).get(copy=False)
    arnode.flush()
    dtype = arnode.dtype
    return euts._instantiatefromself(samples=arnode, dtype=dtype,
                                     epochs=epochs,
                                     epochaxis=epochaxis, copy=False)

def _save_emcuts(emcuts, h5group, overwrite=False,
                 extensible=False, expectedrows=None, dtype=None,
                 epochaxis=None,
                 channelaxis=None, complib='zlib', complevel=5,
                 chunkshape=None):
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
    if channelaxis is None:
        channelaxis = emcuts.channelaxis
    if epochaxis is None:
        epochaxis = emcuts.epochaxis
    if dtype is None:
        dtype = emcuts.samples.dtype
    arnode = _newh5emcuts(h5group=h5group, dtype=dtype,
                          ntimesamples=emcuts.ntimesamples, fs=emcuts.fs,
                          origintime=emcuts.origintime,
                          startdatetime=emcuts.startdatetime,
                          nchannels=emcuts.nchannels, epochaxis=epochaxis,
                          epochs=emcuts.epochs, channels=emcuts.channelnames,
                          geometry=emcuts.geometry, channelaxis=channelaxis,
                          overwrite=overwrite, extensible=extensible,
                          expectedrows=expectedrows, complib=complib,
                          complevel=complevel,
                          chunkshape=chunkshape, attrs=emcuts.attrs)

    epochs = h5group._f_get_child(EPOCHTABLENAME)

    # loop over big chunks of first axis
    chunksize = 1 * 1024 * 1024  # 8 Mb
    if epochaxis == 0:
        epochsize = (
            emcuts.ntimesamples * emcuts.nchannels * emcuts.dtype.itemsize)
        nepochsperchunk = int(chunksize // float(epochsize))
        nepochsperchunk = max(nepochsperchunk, 1)
        nchunks, remainder = divmod(emcuts.nepochs, nepochsperchunk)
        start = 0
        if channelaxis == 1:
            trans = ['epoch', 'channel', 'time']
        else:  # channelaxis == 2
            trans = ['epoch', 'time', 'channel']
        for i in range(nchunks):
            arnode[start:start + nepochsperchunk, :] = \
                emcuts[:, :, start:start + nepochsperchunk].samples.transpose(
                    trans).get(copy=False)
            start = start + nepochsperchunk
        if remainder > 0:
            arnode[start:start + remainder, :] = \
                emcuts[:, :, start:start + remainder].samples.transpose(
                    trans).get(copy=False)
    elif channelaxis == 0:
        channelsize = (
            emcuts.ntimesamples * emcuts.nepochs * emcuts.dtype.itemsize)
        nchannelsperchunk = int(chunksize // float(channelsize))
        nchannelsperchunk = max(nchannelsperchunk, 1)
        nchunks, remainder = divmod(emcuts.nchannels, nchannelsperchunk)
        start = 0
        if epochaxis == 1:
            trans = ['channel', 'epoch', 'time']
        else:  # epochaxis == 2
            trans = ['channel', 'time', 'epoch']
        for i in range(nchunks):
            arnode[start:start + nchannelsperchunk, :, :] = \
                emcuts[:, start:start + nchannelsperchunk].samples.transpose(
                    trans).get(copy=False)
            start = start + nchannelsperchunk
        if remainder > 0:
            arnode[start:start + remainder, :, :] = \
                emcuts[:, start:start + remainder].samples.transpose(
                    trans).get(copy=False)
    else:  # timeaxis == 0
        timesize = (emcuts.nchannels * emcuts.nepochs * emcuts.dtype.itemsize)
        ntimesperchunk = int(chunksize // float(timesize))
        ntimesperchunk = max(ntimesperchunk, 1)
        nchunks, remainder = divmod(emcuts.ntimesamples, ntimesperchunk)
        start = 0
        if epochaxis == 1:
            trans = ['time', 'epoch', 'channel']
        else:  # epochaxis == 2
            trans = ['time', 'channel', 'epoch']
        for i in range(nchunks):
            arnode[start:start + ntimesperchunk, :, :] = \
                emcuts[start:start + ntimesperchunk].samples.transpose(
                    trans).get(copy=False)
            start = start + ntimesperchunk
        if remainder > 0:
            arnode[start:start + remainder, :, :] = \
                emcuts[start:start + remainder].samples.transpose(trans).get(
                    copy=False)
    arnode.flush()
    dtype = arnode.dtype
    return emcuts._instantiatefromself(samples=arnode, epochaxis=epochaxis,
                                       epochs=epochs,
                                       dtype=dtype, channelaxis=channelaxis,
                                       copy=False)

def _save_epoched(epoched, h5group, name, overwrite=False, attrs={}):
    h5file = h5group._v_file
    if overwrite and (name in h5group):
        h5group._v_children[name]._f_remove()
        h5file.flush()
    table = h5file.create_table(where=h5group, name=name,
                                description=epoched.epochs)
    attrs['origintime'] = epoched.origintime
    attrs['duration'] = epoched.duration
    for attrname, value in attrs.items():
        table.set_attr(attrname, value)
        table.flush()
    return table

def geometry_to_h5attr(node, geometry):
    """
    This does not include the pointnames array
    """
    if not isinstance(geometry, Geometry):
        raise TypeError("geometry argument is not a Geometry instance")
    node.set_attr('geometry', geometry.instantiationcode())
    node.set_attr('pointnames', geometry.pointnames.flatten())
    node._v_file.flush()

def _save_multichannel(multichannel, h5group, name, overwrite=False, attrs={}):
    h5file = h5group._v_file
    dtypedescr = multichannel.channelnames.dtype.descr
    dtypedescr[0] = ('channelname', dtypedescr[0][1].replace('U', 'S'))
    if overwrite and (name in h5group):
        h5group._v_children[name]._f_remove()
        h5file.flush()
    channels = multichannel.channelnames.astype(dtypedescr)
    table = h5file.create_table(where=h5group, name=name,
                                description=channels)
    for attrname, value in attrs.items():
        table.set_attr(attrname, value)
    geometry_to_h5attr(table, multichannel.geometry)
    table.flush()
    return table

@contextmanager
def h5fillnew(signalname, filename=None, value=0.0, ntimesamples=None,
              fs=None, duration=None, nchannels=None,
              nepochs=None, startdatetime='NaT', origintime=0.0, channelaxis=None,
              epochaxis=None, channels=None, epochs=None, dtype=np.float64,
              readonly=False, geometry=None, attrs={},
              overwrite=False, extensible=False, expectedrows=None,
              complib='zlib', chunkshape=None, complevel=5):
    """
    Creates a uniform time series object filled with a number.

    Parameters
    ----------
    value : numeric, optional
        Fill value.
    ntimesamples : {None, int}, optional
    fs : {None, float}, optional
    duration : {None, float}, optional
    nchannels : {None, int }, optional
    nepochs : {None, int }, optional
    startdatetime : numpy datetime, optional
    origintime : float, optional
    channelaxis : {None, int}, optional
    epochaxis: {None, int}, optional
    channels : {None, sequence of strings, structured array with 'channelname' column}, optional
    epochs : {None, structured array}, optional
        Array must have 'starttime' field of floats.
    dtype : data-type, optional
    readonly : {True, False}, optional
    geometry : geometry object, optional
    name: {None, string}, optional
    h5group: {None, pytables group}, optional
    attrs: dict, optional
    overwrite : {True, False}, optional
    extensible : {True, False}, optional
    expectedrows : {None, int}, optional
    filters : {None, pytables filters}, optional
    chunkshape : {None, sequence of ints}, optional

    """
    ## check time params
    (ntimesamples, fs, duration) = timeparams(ntimesamples=ntimesamples, fs=fs,
                                              duration=duration)
    if not (nepochs is None):
        if epochs is None:
            epochs = np.empty(nepochs, dtype=[('starttime', np.float64)])
            epochs['starttime'] = np.arange(nepochs, dtype=np.float64)
        if not nepochs == epochs.size:
            raise ValueError("epochs size does not correspond to nepochs")

    # check if everything is correct
    if not (duration == ntimesamples / fs):
        raise ValueError(
            "duration, ntimesamples, and fs parameters do not match")
    ## now see which signal type we need
    with _create_h5savepoint(filename=filename, h5path=signalname,
                             overwrite=overwrite) as h5group:
        if nchannels is None:
            if epochs is None:  # uts
                samples = _newh5uts(h5group=h5group, dtype=dtype,
                                    ntimesamples=ntimesamples, fs=fs,
                                    origintime=origintime, startdatetime=startdatetime,
                                    dflt=value,
                                    overwrite=overwrite, extensible=extensible,
                                    expectedrows=expectedrows, complib=complib,
                                    complevel=complevel, chunkshape=chunkshape,
                                    attrs=attrs)
                s = UniformTimeSeries(samples=samples, fs=fs,
                                      origintime=origintime,
                                      startdatetime=startdatetime, dtype=None,
                                      copy=False,
                                      readonly=readonly)
            else:  # euts
                nepochs = epochs.size
                if epochaxis is None:
                    epochaxis = 1
                epochaxis = check_arg(epochaxis, 'epochaxis')
                samples = _newh5euts(h5group=h5group, dtype=dtype,
                                     ntimesamples=ntimesamples, epochs=epochs,
                                     epochaxis=epochaxis, fs=fs,
                                     origintime=origintime,
                                     startdatetime=startdatetime, dflt=value,
                                     overwrite=overwrite,
                                     extensible=extensible,
                                     expectedrows=expectedrows,
                                     complib=complib,
                                     complevel=complevel,
                                     chunkshape=chunkshape, attrs=attrs)
                s = EpochedUniformTimeSeries(samples=samples, fs=fs,
                                             origintime=origintime,
                                             epochs=epochs,
                                             epochaxis=epochaxis, dtype=None,
                                             copy=False,
                                             readonly=readonly)
        else:  # multichannel
            nchannels = check_arg(nchannels, 'nchannels')
            if epochs is None:  # mcuts
                if channelaxis is None:
                    channelaxis = 1
                channelaxis = check_arg(channelaxis, 'channelaxis')
                samples = _newh5mcuts(h5group=h5group, dtype=dtype,
                                      ntimesamples=ntimesamples,
                                      nchannels=nchannels,
                                      geometry=geometry, channels=channels,
                                      channelaxis=channelaxis, fs=fs,
                                      origintime=origintime,
                                      startdatetime=startdatetime, dflt=value,
                                      overwrite=overwrite,
                                      extensible=extensible,
                                      expectedrows=expectedrows,
                                      complib=complib,
                                      complevel=complevel,
                                      chunkshape=chunkshape, attrs=attrs)
                s = MultiChannelUniformTimeSeries(samples=samples, fs=fs,
                                                  origintime=origintime,
                                                  startdatetime=startdatetime,
                                                  channelaxis=channelaxis,
                                                  channelnames=channels,
                                                  dtype=None, copy=False,
                                                  readonly=readonly,
                                                  geometry=geometry)
            else:  # epoched multichannel
                nepochs = epochs.size
                if epochaxis is None:
                    if channelaxis != 2:
                        epochaxis = 2
                    else:
                        epochaxis = 1
                epochaxis = check_arg(epochaxis, 'epochaxis')
                if channelaxis is None:
                    if epochaxis != 1:
                        channelaxis = 1
                    else:
                        channelaxis = 2
                channelaxis = check_arg(channelaxis, 'channelaxis')
                if not channelaxis in (0, 1, 2):
                    raise IndexError("channelaxis should be 0, 1, 2")
                if not epochaxis in (0, 1, 2):
                    raise IndexError("epochaxis should be 0, 1, 2")
                if channelaxis == epochaxis:
                    raise IndexError(
                        "epochs and channels cannot be on the same axis")
                if epochaxis == 0:
                    if channelaxis == 1:
                        timeaxis = 2
                    else:  # channelaxis == 2
                        timeaxis = 1
                elif epochaxis == 1:
                    if channelaxis == 0:
                        timeaxis = 2
                    else:  # channelaxis == 2
                        timeaxis = 0
                else:  # epochaxis == 2
                    if channelaxis == 1:
                        timeaxis = 0
                    else:  # channelaxis == 0
                        timeaxis = 1
                samples = _newh5emcuts(h5group=h5group,
                                       dtype=dtype, ntimesamples=ntimesamples,
                                       epochs=epochs, epochaxis=epochaxis,
                                       fs=fs,
                                       nchannels=nchannels, geometry=geometry,
                                       channels=channels,
                                       channelaxis=channelaxis,
                                       origintime=origintime,
                                       startdatetime=startdatetime, dflt=value,
                                       overwrite=overwrite,
                                       extensible=extensible,
                                       expectedrows=expectedrows,
                                       complib=complib,
                                       complevel=complevel,
                                       chunkshape=chunkshape, attrs=attrs)
                s = EpochedMultiChannelUniformTimeSeries(samples=samples,
                                                         fs=fs,
                                                         origintime=origintime,
                                                         channelaxis=channelaxis,
                                                         channelnames=channels,
                                                         epochaxis=epochaxis,
                                                         epochs=epochs,
                                                         dtype=None,
                                                         copy=False,
                                                         readonly=readonly,
                                                         geometry=geometry)
        yield s

@contextmanager
def h5filllike(s, name='/', filename=None, value=0.0, ntimesamples=None,
               fs=None, dtype=None, channelaxis=None,
               epochaxis=None, overwrite=False, extensible=False,
               expectedrows=None, filters=None, chunkshape=None, complib=None,
               complevel=None, attrs=None):
    if not hasattr(s.samples, 'h5array'):
        if complib is None:
            complib = 'zlib'
        if complevel is None:
            complevel = 5
    if attrs is None:
        attrs = s.attrs
    with _create_h5savepoint(filename=filename, h5path=name,
                             overwrite=overwrite) as h5group:
        if isuts(s):
            yield _uts_fill_like(s, value=value, h5group=h5group, dtype=dtype,
                                 ntimesamples=ntimesamples, fs=fs,
                                 overwrite=overwrite, extensible=extensible,
                                 expectedrows=expectedrows,
                                 chunkshape=chunkshape, complib=complib,
                                 complevel=complevel, attrs=attrs)
        elif ismcuts(s):
            yield _mcuts_fill_like(s, value=value, h5group=h5group,
                                   dtype=dtype, ntimesamples=ntimesamples,
                                   fs=fs, channelaxis=channelaxis,
                                   overwrite=overwrite, extensible=extensible,
                                   expectedrows=expectedrows,
                                   chunkshape=chunkshape, complib=complib,
                                   complevel=complevel, attrs=attrs)
        elif iseuts(s):
            yield _euts_fill_like(s, value=value, h5group=h5group, dtype=dtype,
                                  ntimesamples=ntimesamples, fs=fs,
                                  epochaxis=epochaxis,
                                  overwrite=overwrite, extensible=extensible,
                                  expectedrows=expectedrows,
                                  chunkshape=chunkshape, complib=complib,
                                  complevel=complevel, attrs=attrs)
        elif isemcuts(s):
            yield _emcuts_fill_like(s, value=value, h5group=h5group,
                                    dtype=dtype, ntimesamples=ntimesamples,
                                    fs=fs, channelaxis=channelaxis,
                                    epochaxis=epochaxis, overwrite=overwrite,
                                    extensible=extensible,
                                    expectedrows=expectedrows,
                                    chunkshape=chunkshape, complib=complib,
                                    complevel=complevel, attrs=attrs)

def _uts_fill_like(s, value=0.0, h5group='/', ntimesamples=None, fs=None,
                   dtype=None,
                   overwrite=False, extensible=False, expectedrows=None,
                   chunkshape=None, complib=None, complevel=None, attrs={}):
    if ntimesamples is None:
        ntimesamples = s.ntimesamples
    if fs is None:
        fs = s.fs
    if dtype is None:
        dtype = s.dtype
    else:
        dtype = np.dtype(dtype)
    if hasattr(s.samples, 'h5array'):
        if chunkshape is None:
            chunkshape = s.samples.h5array.chunkshape
        if complib is None:
            complib = s.samples.h5array.filters.complib
        if complevel is None:
            complevel = s.samples.h5array.filters.complevel
    samples = _newh5uts(h5group=h5group, dtype=dtype,
                        ntimesamples=ntimesamples, fs=fs,
                        origintime=s.origintime, startdatetime=s.startdatetime,
                        dflt=value,
                        overwrite=overwrite, extensible=extensible,
                        expectedrows=expectedrows, complib=complib,
                        complevel=complevel,
                        chunkshape=chunkshape, attrs=attrs)

    return s._instantiatefromself(samples=samples, copy=False,
                                  dtype=samples.atom.dtype, fs=fs)

#euts
def _euts_fill_like(s, value=0.0, h5group='/', ntimesamples=None, fs=None,
                    dtype=None, epochaxis=None,
                    overwrite=False, extensible=False, expectedrows=None,
                    chunkshape=None, complib=None,
                    complevel=None, attrs={}):
    if ntimesamples is None:
        ntimesamples = s.ntimesamples
    if epochaxis is None:
        epochaxis = s.epochaxis
    if fs is None:
        fs = s.fs
    if dtype is None:
        dtype = s.samples.dtype
    if hasattr(s.samples, 'h5array'):
        if chunkshape is None:
            chunkshape = s.samples.h5array.chunkshape
        if complib is None:
            complib = s.samples.h5array.filters.complib
        if complevel is None:
            complevel = s.samples.h5array.filters.complevel
    samples = _newh5euts(h5group=h5group, dtype=dtype,
                         ntimesamples=ntimesamples, fs=fs,
                         origintime=s.origintime, startdatetime=s.startdatetime,
                         epochs=s.epochs, epochaxis=epochaxis, dflt=value,
                         overwrite=overwrite, extensible=extensible,
                         expectedrows=expectedrows, complib=complib,
                         complevel=complevel,
                         chunkshape=chunkshape, attrs=attrs)

    return s._instantiatefromself(samples=samples, copy=False,
                                  epochaxis=epochaxis,
                                  dtype=samples.atom.dtype, fs=fs)


#mcuts
def _mcuts_fill_like(s, value=0.0, h5group='/', ntimesamples=None, fs=None,
                     dtype=None, channelaxis=None,
                     overwrite=False, extensible=False, expectedrows=None,
                     chunkshape=None, complib=None,
                     complevel=None, attrs={}):
    if ntimesamples is None:
        ntimesamples = s.ntimesamples
    if channelaxis is None:
        channelaxis = s.channelaxis
    if fs is None:
        fs = s.fs
    if dtype is None:
        dtype = s.samples.dtype
    if channelaxis is None:
        channelaxis = s.channelaxis
    if hasattr(s.samples, 'h5array'):
        if chunkshape is None:
            chunkshape = s.samples.h5array.chunkshape
        if complib is None:
            complib = s.samples.h5array.filters.complib
        if complevel is None:
            complevel = s.samples.h5array.filters.complevel
    samples = _newh5mcuts(h5group=h5group, dtype=dtype,
                          ntimesamples=ntimesamples, fs=fs,
                          origintime=s.origintime, startdatetime=s.startdatetime,
                          nchannels=s.nchannels, channels=s.channelnames,
                          geometry=s.geometry, channelaxis=channelaxis,
                          dflt=value,
                          overwrite=overwrite, extensible=extensible,
                          expectedrows=expectedrows, complib=complib,
                          complevel=complevel,
                          chunkshape=chunkshape, attrs=attrs)

    return s._instantiatefromself(samples=samples, copy=False,
                                  channelaxis=channelaxis, fs=fs,
                                  dtype=samples.atom.dtype)

#emcuts
def _emcuts_fill_like(s, value=0.0, h5group='/', ntimesamples=None, fs=None,
                      dtype=None, channelaxis=None,
                      epochaxis=None, overwrite=False, extensible=False,
                      expectedrows=None, chunkshape=None, complib=None,
                      complevel=None, attrs={}):
    if ntimesamples is None:
        ntimesamples = s.ntimesamples
    if channelaxis is None:
        channelaxis = s.channelaxis
    if epochaxis is None:
        epochaxis = s.epochaxis
    timeaxis, channelaxis, epochaxis = s._check_axes(channelaxis, epochaxis)
    if fs is None:
        fs = s.fs
    if dtype is None:
        dtype = s.samples.dtype
    if hasattr(s.samples, 'h5array'):
        if chunkshape is None:
            chunkshape = s.samples.h5array.chunkshape
        if complib is None:
            complib = s.samples.h5array.filters.complib
        if complevel is None:
            complevel = s.samples.h5array.filters.complevel
    samples = _newh5emcuts(h5group=h5group, dtype=dtype,
                           ntimesamples=ntimesamples, fs=fs,
                           origintime=s.origintime, startdatetime=s.startdatetime,
                           nchannels=s.nchannels, epochaxis=epochaxis,
                           epochs=s.epochs, channels=s.channelnames,
                           geometry=s.geometry, channelaxis=channelaxis,
                           dflt=value,
                           overwrite=overwrite, extensible=extensible,
                           expectedrows=expectedrows, complib=complib,
                           complevel=complevel,
                           chunkshape=chunkshape, attrs=attrs)

    return s._instantiatefromself(samples=samples, copy=False,
                                  channelaxis=channelaxis, fs=fs,
                                  dtype=samples.atom.dtype)