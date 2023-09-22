from unittest import TestCase

from numpy.testing import assert_raises, \
     assert_almost_equal, assert_array_almost_equal
import numpy as np
from uts.core.utils import round_nextradix2, round_nearestradix2, \
                            round_nearestodd, round_nearesteven, \
                            allodd, alleven, isodd, iseven
from uts.tests.defs import int_to_pints, float_to_floats

class test_Rounding(TestCase):

    def setUp(self):
        pass

    def test_round_nextradix2(self):
        "Utils: round_nextradix2"
        assert round_nextradix2(1) == 1
        assert round_nextradix2(2) == 2
        assert round_nextradix2(3) == 4
        assert round_nextradix2(4) == 4
        assert round_nextradix2(5) == 8
        assert round_nextradix2(2048) == 2048
        assert round_nextradix2(2049) == 4096

    def test_round_nearestradix2(self):
        "Utils: round_nearestradix2"
        assert round_nearestradix2(1) == 1
        assert round_nearestradix2(2) == 2
        assert round_nearestradix2(3) == 4
        assert round_nearestradix2(4) == 4
        assert round_nearestradix2(5) == 4
        assert round_nearestradix2(7) == 8
        assert round_nearestradix2(2048) == 2048
        assert round_nearestradix2(2049) == 2048

    def test_round_nearestodd(self):
        "Utils: round_nearestodd"
        assert round_nearestodd(0.0) in (-1, 1)
        assert round_nearestodd(0.1) == 1
        assert round_nearestodd(1.0) == 1
        assert round_nearestodd(1.1) == 1
        assert round_nearestodd(1.9) == 1
        assert round_nearestodd(2.1) == 3
        assert round_nearestodd(2.9) == 3
        assert round_nearestodd(3.1) == 3
        assert round_nearestodd(3.9) == 3
        assert round_nearestodd(4.1) == 5
        assert round_nearestodd(-0.1) == -1
        assert round_nearestodd(-1.0) == -1
        assert round_nearestodd(-1.1) == -1
        assert round_nearestodd(-1.9) == -1
        assert round_nearestodd(-2.1) == -3
        assert round_nearestodd(-2.9) == -3
        assert round_nearestodd(-3.1) == -3
        assert round_nearestodd(-3.9) == -3
        assert round_nearestodd(-4.1) == -5

    def test_round_nearesteven(self):
        "Utils: round_nearesteven"
        assert round_nearesteven(0.0) == 0
        assert round_nearesteven(0.1) == 0
        assert round_nearesteven(0.9) == 0
        assert round_nearesteven(1.1) == 2
        assert round_nearesteven(1.9) == 2
        assert round_nearesteven(2.1) == 2
        assert round_nearesteven(2.9) == 2
        assert round_nearesteven(3.1) == 4
        assert round_nearesteven(3.9) == 4
        assert round_nearesteven(4.1) == 4
        assert round_nearesteven(-0.1) == 0
        assert round_nearesteven(-1.1) == -2
        assert round_nearesteven(-1.9) == -2
        assert round_nearesteven(-2.1) == -2
        assert round_nearesteven(-2.9) == -2
        assert round_nearesteven(-3.1) == -4
        assert round_nearesteven(-3.9) == -4
        assert round_nearesteven(-4.1) == -4

class test_OddEven(TestCase):

    def setUp(self):

        self.evens = [2, 2., 0, 0., -2, -2., 4, -4, 4., -4.]
        self.odds = [1, 1., -1, -1., 3, 3., -3, -3.]

    def test_iseven(self):
        "Utils: iseven"
        for even in self.evens:
            assert iseven(even)
        for odd in self.odds:
            assert not iseven(odd)


    def test_isodd(self):
        "Utils: isodd"
        for even in self.evens:
            assert not isodd(even)
        for odd in self.odds:
            assert isodd(odd)
        

    def test_allodd(self):
        "Utils: allodd"
        assert allodd([1,-1,1,1,1])
        assert allodd([1.,-1.,1,-1,1])
        assert allodd([1.,-1.,1.,1.,1.])
        assert allodd([5,-5,5,5,5])
        assert allodd([5.,-5.,5,-5,5])
        assert allodd([5.,-5.,5.,5.,5.])
        assert allodd([1,3,5,7,9])

        assert not allodd([2,-2,2,2,2])
        assert not allodd([2.,-2.,2,-2,2])
        assert not allodd([2.,-2.,2.,2.,2.])
        assert not allodd([6,-6,6,6,6])
        assert not allodd([6.,-6.,6,-6,6])
        assert not allodd([6.,-6.,6.,6.,6.])
        assert not allodd([0,2,4,6,8,10])

        assert not allodd([1,3,5,7,10])

    def test_alleven(self):
        "Utils: alleven"
        assert alleven([2,-2,2,2,2])
        assert alleven([2.,-2.,2,-2,2])
        assert alleven([2.,-2.,2.,2.,2.])
        assert alleven([6,-6,6,6,6])
        assert alleven([6.,-6.,6,-6,6])
        assert alleven([6.,-6.,6.,6.,6.])
        assert alleven([0,2,4,6,8,10])

        assert not alleven([1,-1,1,1,1])
        assert not alleven([1.,-1.,1,-1,1])
        assert not alleven([1.,-1.,1.,1.,1.])
        assert not alleven([5,-5,5,5,5])
        assert not alleven([5.,-5.,5,-5,5])
        assert not alleven([5.,-5.,5.,5.,5.])
        assert not alleven([1,3,5,7,9])

        assert not alleven([2,4,6,8,9])


if __name__ == '__main__':
    unittest.main()