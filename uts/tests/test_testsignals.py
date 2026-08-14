import unittest

import numpy as np

from uts.core.multichannel import default_channelnames
from uts.signalfactory.testsignals import (testemcuts, testeuts, testmcuts,
                                           testuts)


class TestDefaultChannelNames(unittest.TestCase):

    def test_width(self):
        self.assertEqual(list(default_channelnames(5)),
                         ['channel_%d' % i for i in range(5)])
        self.assertEqual(list(default_channelnames(100))[:2],
                         ['channel_00', 'channel_01'])

    def test_matchesasignalthatgetsnonames(self):
        signal = testmcuts(duration=0.05, fs=1000., nchannels=6)
        self.assertEqual(list(signal.channelnames),
                         list(default_channelnames(6)))


class TestTestMcuts(unittest.TestCase):

    def test_nogeometry(self):
        signal = testmcuts(duration=0.05, fs=1000., nchannels=6)
        self.assertEqual(signal.nchannels, 6)
        self.assertEqual(signal.geometry.shape, (1, 6))

    def test_geometryshape(self):
        # this used to raise NameError: name 'uts' is not defined, and then
        # ChannelFormatError once that was fixed, because the point names were
        # generated as 'channel_%.3d' and so never matched the channel names
        for shape in [(3, 2), (2, 3), (6, 1), (1, 6)]:
            signal = testmcuts(duration=0.05, fs=1000., nchannels=6,
                               geometryshape=shape)
            self.assertEqual(signal.geometry.shape, shape)
            self.assertEqual(set(signal.geometry.pointnames.flatten()),
                             set(signal.channelnames))

    def test_incongruentshaperaises(self):
        self.assertRaises(ValueError, testmcuts, nchannels=5,
                          geometryshape=(3, 2))

    def test_samples(self):
        signal = testmcuts(duration=0.05, fs=1000., nchannels=3)
        self.assertEqual(signal.samples[:].shape, (3, 50))


class TestTestEmcuts(unittest.TestCase):

    def test_geometryshape(self):
        signal = testemcuts(epochdur=0.05, fs=1000., nepochs=4, nchannels=6,
                            geometryshape=(3, 2))
        self.assertEqual(signal.geometry.shape, (3, 2))
        self.assertEqual(set(signal.geometry.pointnames.flatten()),
                         set(signal.channelnames))
        self.assertEqual(signal.nepochs, 4)

    def test_incongruentshaperaises(self):
        self.assertRaises(ValueError, testemcuts, nchannels=5,
                          geometryshape=(3, 2))


class TestTestUtsAndEuts(unittest.TestCase):

    def test_uts(self):
        signal = testuts(duration=0.05, fs=1000.)
        self.assertEqual(signal.ntimesamples, 50)

    def test_euts(self):
        signal = testeuts(epochdur=0.05, fs=1000., nepochs=4)
        self.assertEqual(signal.nepochs, 4)

    def test_dtype(self):
        self.assertEqual(testuts(duration=0.05, fs=1000.,
                                 dtype=np.float64).samples[:].dtype,
                         np.float64)


if __name__ == '__main__':
    unittest.main()
