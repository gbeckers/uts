from unittest import TestCase

from numpy.testing import assert_raises, run_module_suite
import numpy as np
from uts.core.multichannel import MultiChannel, ChannelIndexError, ChannelFormatError
from uts.core.defs import DimensionError
from uts.core.geometry import RectangularLattice, GeometryError
from uts.tests.defs import nan, inf, pfloats, zfloats, nfloats, floattypes, inttypes, eps,\
     pints, zints, nints, float_to_floats, int_to_nints, int_to_pints


class test_MultiChannel(TestCase):

    def test_nchannels_arg(self):
        "MultiChannel: nchannels argument"
        # good number of channels
        for nchannels in [1,2,3]:
            # good nchannels type
            for intnchannels in int_to_pints(nchannels):
                mc = MultiChannel(nchannels=nchannels, channels=None,
                                  geometry=None)
                assert mc.nchannels == nchannels
                assert isinstance(mc.nchannels, int)
            # bad nchannels type
            for floatnchannel in float_to_floats(nchannels):
                assert_raises(TypeError, MultiChannel, nchannels=floatnchannel,
                              channels=None, geometry=None)
            for nchannels in [[0],[1,2], None, '1']:
                assert_raises(TypeError, MultiChannel, nchannels=floatnchannel,
                              channels=None, geometry=None)

        # bad number of channels:
        for nchannels in [-1,0]:
            assert_raises(ValueError, MultiChannel, nchannels=nchannels,
                          channels=None, geometry=None)

    def test_channels_arg(self):
        "MultiChannel: channels argument"
        nchannels = 3
        # sequences
        goodchannels = [["channel1","channel2","channel3"],
                        ("channel1","channel2","channel3"),
                         np.array(["channel1","channel2","channel3"])]
        # structured arrays
        channelnames = np.array(["channel1","channel2","channel3"])
        star1 = np.empty(3, dtype=[('channelname', channelnames.dtype),('geometrypoint', np.int64)])
        star1['channelname'] = channelnames
        star1['geometrypoint'] = np.array([0,1,2])
        goodchannels.append(star1)
        star2 = np.empty(3, dtype=[('channelname', channelnames.dtype),('geometrypoint', np.int64), ('impedance',np.float64)])
        star2['channelname'] = channelnames
        star2['geometrypoint'] = np.array([0,1,2])
        star2['impedance'] = np.array([1.0, 2.0, 3.0])
        goodchannels.append(star2)
        for channels in goodchannels:
            mc = MultiChannel(nchannels=nchannels, channels=channels,
                              geometry=None)
            assert mc.nchannels == nchannels
            assert (mc.channelnames == np.array(channelnames)).all()
            assert mc.channelnames.size == nchannels

        # duplicate names, or nchannels != channelname.size
        badchannels = [["channel1","channel1","channel3"],
                       ("channel1","channel1","channel3"),
                       np.array(["channel1","channel1","channel3"]),
                       ["channel1","channel3"],
                       ("channel1","channel3"),
                       np.array(["channel1","channel3"]),
                       [1, 2, 3], 
                       [1.0, 2.0, 3.0]]
        # structured arrays
        star3 = np.empty(3, dtype=[('geometrypoint', np.int64), ('impedance', np.int64)]) # no channelname column
        star3['impedance'] = [1.0, 2.0, 3.0]
        star3['geometrypoint'] = np.array([0,1,2])
        badchannels.append(star3)
        star4 = np.empty(3, dtype=[('channelname', channelnames.dtype), ('impedance', np.int64)]) # no geometrypoint column
        star4['impedance'] = [1.0, 2.0, 3.0]
        star4['channelname'] = channelnames
        badchannels.append(star4)
        for channels in badchannels:
            assert_raises(ChannelFormatError, MultiChannel, nchannels=nchannels,
                          channels=channels, geometry=None)

        # bad dimension
        assert_raises(ChannelFormatError, MultiChannel, nchannels=6,
                      channels=[['1','2','3'],['1','2','3']],
                      geometry=None)
        assert_raises(ChannelFormatError, MultiChannel, nchannels=6,
                      channels='123456',
                      geometry=None)

    def test_geometry_arg(self):
        "MultiChannel: geometry argument"
        # correct Geometry
        gm = RectangularLattice(nx=2,ny=2, pointnames=['a','b','c','d'])
        mc = MultiChannel(nchannels=4, channels=['a','b','c','d'], geometry=gm)
        assert mc.nchannels == mc.geometry.npoints
        # wrong number of points
        assert_raises(ChannelFormatError, MultiChannel, nchannels=3,
                      channels=['a','b','c','d'], geometry=gm)
        # wrong type
        assert_raises(ChannelFormatError, MultiChannel, nchannels=3,
                      channels=['a','b','c','d'], geometry=(2,2))
        # None is acceptable and should lead to a default geometry
        mc = MultiChannel(nchannels=4, channels=None, geometry=None)
        assert isinstance(mc.geometry, RectangularLattice)

    def test_set_geometry(self):
        "MultiChannel: set geometry"
        mc = MultiChannel(nchannels=4, channels=None, geometry=None)
        gm = RectangularLattice(nx=2, ny=2, pointnames=mc.channelnames)
        assert isinstance(mc.geometry, RectangularLattice)
        mc.set_geometry(gm, mc.channels)
        assert mc.geometry == gm
        mc.set_geometry(None, mc.channels)
        assert isinstance(mc.geometry, RectangularLattice)
        badgm = RectangularLattice(nx=3,ny=2, pointnames=['a','b','c','d','e','f'])
        assert_raises(ChannelFormatError, mc.set_geometry, geometry=badgm, channels=mc.channels)

    def test_to_channelsslicevalid(self):
        "MultiChannel: valid slicing"

        channels = ["channel1","channel2","channel3","channel4"]
        mc = MultiChannel(nchannels=4, channels=channels, geometry=None)
        # int
        for i in int_to_pints(1):
            assert mc._to_channelsslice(i) == 1
        # negative int
        for i in int_to_nints(-1):
            assert mc._to_channelsslice(i) == -1
        # slice
        assert mc._to_channelsslice(slice(1,3,None)) == slice(1,3,None)
        # slice with negative step
        assert mc._to_channelsslice(slice(None,None,-1)) == slice(None,None,-1)
        # list
        assert mc._to_channelsslice([0,1,2,3]) == slice(0, 4, None)
        # list with decreasing indices
        assert (mc._to_channelsslice([3,2,1,0]) == np.array([3,2,1,0])).all()
        # same for int arrays
        assert mc._to_channelsslice(np.array([0,1,2,3])) == slice(0, 4, None)
        assert (mc._to_channelsslice(np.array([3,2,1,0])) == np.array([3,2,1,0])).all()
        # boolean arrays
        assert (mc._to_channelsslice(np.array([0,0,1,0],'?')) == np.array([2])).all()
        assert (mc._to_channelsslice(np.array([1,0,1,0],'?')) == np.array([0,2])).all()
        # name indices
        assert (mc._to_channelsslice(np.array(["channel1","channel3"])) == np.array([0,2])).all()
        assert (mc._to_channelsslice(np.array(["channel3","channel1"])) == np.array([2,0])).all()
        # duplicates allowed
        assert (mc._to_channelsslice(np.array(["channel1","channel3","channel3"])) == np.array([0,2,2])).all()
        # slicing beyond range allowed
        assert mc._to_channelsslice(slice(0,5)) == slice(0,5)

    def test_to_channelssliceinvalid(self):
        "MultiChannel: invalid slicing"

        channels = ["channel1","channel2","channel3","channel4"]
        mc = MultiChannel(nchannels=4, channels=channels, geometry=None)
        # float
        for i in float_to_floats(1.0):
            assert_raises(ChannelIndexError, mc._to_channelsslice, channelindex=i)
        # non-existing channels
        for i in int_to_pints(4):
            assert_raises(ChannelIndexError, mc._to_channelsslice, channelindex=i)
        assert_raises(ChannelIndexError, mc._to_channelsslice, channelindex=["channel5"])
        for i in int_to_nints(-5):
            assert_raises(ChannelIndexError, mc._to_channelsslice, channelindex=i)
        # None
        assert_raises(ChannelIndexError, mc._to_channelsslice, channelindex=None)


if __name__ == '__main__':
    unittest.main()