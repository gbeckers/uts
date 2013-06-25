from unittest import TestCase

from numpy.testing import assert_raises, run_module_suite, \
     assert_almost_equal, assert_array_almost_equal
import numpy as np
from uts3.core.geometry import RectangularLattice, GeometryError
from uts3.tests.defs import int_to_pints, float_to_floats


def pn(n):
    """create n pointnames"""
    return ['point_%d'%i for i in range(n)] 


class test_RectangularLattice(TestCase):

    def setUp(self):
        pass

    def test_nx_ny_args(self):
        "RectangularLattice: nx ny arguments"
        for n in [1,2,3]:
            pointnames = pn(n*2)
            # good nchannels type
            for intn in int_to_pints(n):
                g = RectangularLattice(nx=n, ny=2, pointnames=pointnames)
                assert g.nx == n
                assert isinstance(g.nx, int)
                g = RectangularLattice(nx=2, ny=n, pointnames=pointnames)
                assert g.ny == n
                assert isinstance(g.ny, int)
            # bad type
            for floatn in float_to_floats(n):
                assert_raises(TypeError, RectangularLattice, nx=floatn,
                              ny=2, pointnames=pointnames)
            for ntype in [[0],[1,2], None, '1']:
                assert_raises(TypeError, RectangularLattice, nx=ntype,
                              ny=2, pointnames=pointnames)
        # bad number of n
        for n in [-1,0]:
            assert_raises(ValueError, RectangularLattice, nx=n,
                              ny=2, pointnames=pointnames)
            assert_raises(ValueError, RectangularLattice, nx=2,
                              ny=n, pointnames=pointnames)

    def test_pointnames_arg(self):
        "RectangularLattice: pointname argument"
        def _pn_generator(nx,ny):
            for i in range(nx*ny):
                yield "point_%d" % i

        nx = 2; ny = 3

        # OK
        pointnames_list = pn(nx*ny)
        pointnames_tuple = tuple(pointnames_list)
        pointnames_array = np.array(pointnames_list)
        for pointnames in (pointnames_list, pointnames_tuple,
                           pointnames_array):
            g = RectangularLattice(nx=nx, ny=ny, pointnames=pointnames)
            assert (g.pointnames.flatten() == pointnames_array).all()
            assert g.pointnames.shape[0] == ny
            assert g.pointnames.shape[1] == nx

        # bad sequences
        pointnames_generator = _pn_generator(nx,ny)
        pointnames_int = [i for i in range(nx,ny)]
        for pointnames in (pointnames_generator, pointnames_int):
            assert_raises(GeometryError, RectangularLattice, nx=nx,
                              ny=ny, pointnames=pointnames)
        # other bad stuff
        for pointnames in (1, 1.1, 'abc', None):
            assert_raises(GeometryError, RectangularLattice, nx=nx,
                              ny=ny, pointnames=pointnames)
        # wrong number
        assert_raises(GeometryError, RectangularLattice, nx=nx,
                              ny=ny+1, pointnames=pointnames_list)
        assert_raises(GeometryError, RectangularLattice, nx=nx+1,
                              ny=ny, pointnames=pointnames_list)
        assert_raises(GeometryError, RectangularLattice, nx=nx+1,
                              ny=ny+1, pointnames=pointnames_list)

    def test_equals(self):
        pass

    def test_get_item(self):
        pass

    def test_mapfrom1Dto1D(self):
        pass

    def test_mapfrom1Dto2D(self):
        pass

    def test_mapfrom2Dto1D(self):
        pass

    def test_indexfrom1Dto1D(self):
        pass

    def test_indexfrom1Dto2D(self):
        pass

    def test_indexfrom2Dto1D(self):
        pass

    def test_instantiationcode(self):
        pass

    def test_distance(self):
        "RectangularLattice: distance method"
        # 'point_0'  'point_1'
        # 'point_2'  'point_3'
        # 'point_4'  'point_5',
        g = RectangularLattice(nx=2, ny=3, pointnames=pn(6), dx=1.0, dy=2.0)
        assert g.distance('point_0','point_0') == 0.0
        assert g.distance('point_0','point_1') == 1.0
        assert g.distance('point_0','point_2') == 2.0
        assert g.distance('point_0','point_4') == 4.0
        assert_almost_equal(g.distance('point_0','point_3'), (1.0**2.0+2.0**2.0)**0.5)
        assert_almost_equal(g.distance('point_3','point_0'), (1.0**2.0+2.0**2.0)**0.5)
        assert_almost_equal(g.distance('point_0','point_5'), (1.0**2.0+4.0**2.0)**0.5)
        assert_almost_equal(g.distance('point_5','point_0'), (1.0**2.0+4.0**2.0)**0.5)
        assert_almost_equal(g.distance('point_4','point_1'), (1.0**2.0+4.0**2.0)**0.5)
        assert_almost_equal(g.distance('point_1','point_4'), (1.0**2.0+4.0**2.0)**0.5)

    def test_distance_matrix(self):
        "RectangularLattice: distance matrix"
        mat = np.array([[ 0., 1., 2., 5**0.5 ],
                        [ 1., 0., 5**0.5, 2. ],
                        [ 2., 5**0.5, 0., 1. ],
                        [ 5**0.5, 2., 1., 0. ]])
        g = RectangularLattice(nx=2, ny=2, pointnames=pn(4), dx=1.0, dy=2.0)
        assert_array_almost_equal(g.distance_matrix(g.pointnames.flatten()), mat)



if __name__ == '__main__':
    unittest.main()