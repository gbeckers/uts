import numpy as np
from .geometry import Geometry, RectangularLattice
from .defs import defined_integers, DimensionError
from .check_arg import check_arg

class ChannelIndexError(IndexError): pass

class ChannelFormatError(Exception): pass

class MultiChannel(object):

    """Super class for multichannel classes.

    Parameters
    ----------
    nchannels: int
        number of channels

    channelnames: array-like sequence of strings
       a sequence of channelnames

    Properties
    ----------
    nchannels

    channelnames

    """

    def __init__(self, nchannels=None, channels=None, geometry=None):

        if (nchannels is None) and (channels is None):
            raise ValueError("'nchannels' and 'channels' parameters cannot both be None")
        if (channels is None) and (not geometry is None):
            raise ValueError("if 'geometry' is specified, 'channels' should be specified as well")
        if nchannels is None:
            nchannels = len(channels)

        self.__nchannels = check_arg(nchannels, 'nchannels')
        self.__channels, self.__geometry = \
                self._set_channels(nchannels=self.__nchannels,
                                   channels=channels, geometry=geometry)
        
    channels = property(lambda self: self.__channels, None, None,
                         "structured array witch channel info")
    nchannels = property(lambda self: self.__nchannels, None, None,
                         "number of channels")
    channelnames = property(
        lambda self: (self.__channels['channelname']), None, None,
        "names of channels")
    geometry = property(lambda self: self.__geometry, None, None,
                         "geometry of channel set")

    def info(self):
        s = ''
        s += 'nchannels: %d\n' % self.nchannels
        chnamess = ''
        for chname in self.channelnames:
            chnamess += ('%s ' %chname)
        s += 'channelorder: %s\n' % chnamess[:-1]
        s += self.geometry.info()
        return s

    def _set_channels(self, nchannels, channels, geometry):
        """Checks if channels parameter is properly formatted. Will return
        a structured array with at least a column 'channelname'.
        """
        # we first take care of the channels
        if channels is None:
            s = 'channel_%%.%dd' % max(1, int(np.ceil(np.log10(nchannels))))
            channelnames = np.array([s % i for i in range(nchannels)])
            channels = np.empty(channelnames.size,
                                dtype=[('channelname', channelnames.dtype),('geometrypoint', np.int64)])
            channels['channelname'] = channelnames
        else:
            channels = np.array(channels)
        if not channels.dtype.fields is None:
            if not 'channelname' in channels.dtype.names:
                raise ChannelFormatError("'channels' structured array does not have 'channelname' field")
            if not 'geometrypoint' in channels.dtype.names:
                raise ChannelFormatError("'channels' structured array does not have 'geometrypoint' field")
        else: # we assume we got a sequence of channelnames
            if not channels.ndim == 1:
                raise ChannelFormatError("channels not a 1-dimensional sequence" \
                                         ", %s" % channels)
            sa = np.empty(channels.size,
                          dtype=[('channelname', channels.dtype),('geometrypoint', np.int64)])
            sa['channelname'] = channels
            channels = sa
        channelnames = channels['channelname']
        if not channelnames.ndim == 1:
            raise ChannelFormatError("channel names not a 1-dimensional sequence" \
                                     ", %s" % channelnames)
        if not np.issubdtype(channelnames.dtype, np.string_):
            raise ChannelFormatError("channel name(s) not of string type")
        if not np.unique(channelnames).size == channelnames.size:
            raise ChannelFormatError("channel names not unique")
        if not channelnames.size == nchannels:
            raise ChannelFormatError("number of channelnames does not correspond to nchannels: %s, %d" \
                                     % (channelnames, nchannels))
        # now geometry
        if geometry is None:
            geometry =  RectangularLattice(nx=1, ny=nchannels, pointnames=channelnames)
            channels['geometrypoint'] = np.arange(channelnames.size)
        else:
            if not isinstance(geometry, Geometry):
                raise ChannelFormatError("geometry argument (%s) is not a Geometry instance" \
                                         % type(geometry).__name__)
            pointnames = geometry.pointnames.flatten()
            if not geometry.npoints == nchannels:
                raise ChannelFormatError("number of geometry points does not match number of channels")
            if len(set(pointnames) ^ set(channelnames)) != 0:
                raise ChannelFormatError("names geometry points does not match those of channels" \
                                         "\n%s" % (set(pointnames) ^ set(channelnames)))
            channels['geometrypoint'] = geometry.indexfrom1Dto1D(channelnames)
        return channels, geometry

    def _channels_exist(self, channels):
        '''
        Checks if a channel index contains valid channel names.
        '''
        channelnames = np.array(channels)
        if not channelnames.ndim == 1:
            raise DimensionError("channel names not a 1-dimensional sequence")
        if not np.issubdtype(channelnames.dtype, np.string_):
            raise ChannelIndexError("channel name(s) not of string type")
        if np.union1d(channelnames, self.channelnames).size > self.channelnames.size:
            raise ChannelIndexError("not all requested channels exist (requested: %s, available: %s)" \
                                    % (channelnames, self.channelnames))
        return channelnames

    def _to_channelsslice(self, channelindex):
        """
        Converts a channelindex to an array of channel indices.

        Parameter
        ---------
        channelindex: a slice with int (or None) start, stop, step, or
            an array_like, containing the channelnames, channelindices, or a
            boolean mask.

        Returns
        -------
        a 1D array of dtype int with channel indices

        """

        channelindex = np.index_exp[channelindex]
        if len(channelindex) != 1:
            raise DimensionError("dimension of channel index != 1")
        channelindex = channelindex[0]

        if isinstance(channelindex, slice):
            for i in (channelindex.start, channelindex.stop, channelindex.step):
                if not (isinstance(i, defined_integers) or (i is None)):
                    raise ChannelIndexError("channel slice must have start, stop, step of type int")
            return channelindex
            #channelindices = np.arange(self.nchannels, dtype=np.int)[channelindex]
        elif isinstance(channelindex, defined_integers):
            if (channelindex >= self.nchannels) or (channelindex < -self.nchannels):
                raise ChannelIndexError("channel index out of range")
            return int(channelindex)
        else: # channelindex probably a sequence
            channelindex = np.array(channelindex, ndmin=1)
            if np.issubdtype(channelindex.dtype, np.string_):
                channelindices = []
                chindnumbers = np.arange(self.nchannels)
                for channelname in self._channels_exist(channelindex):
                    channelindices.append(chindnumbers[self.channelnames == channelname].item())
            else:
                if not ((channelindex.dtype in defined_integers) or \
                   channelindex.dtype == np.bool):
                    raise ChannelIndexError("channel index (%s) is of wrong type (%s)" \
                                            % (channelindex, channelindex.dtype))
                channelindices = np.arange(self.nchannels, dtype=np.int)[channelindex]

            if len(channelindices) == 1:
                channelindices = channelindices[0]
            else:
                channelindices = np.array(channelindices, copy=False, ndmin=1)
                # see if we can make a slice to speed things up down the road (e.g. hdf5)
                if (np.diff(channelindices)==1).all():
                    channelindices = slice(channelindices[0],channelindices[-1]+1)
            return channelindices

    def set_geometry(self, geometry, channels):
        self.__channels, self.__geometry = \
                self._set_channels(nchannels=self.nchannels,
                                   channels=channels, geometry=geometry)
    
