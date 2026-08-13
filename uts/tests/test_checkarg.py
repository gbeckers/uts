from __future__ import absolute_import
from unittest import TestCase

from numpy.testing import assert_raises, assert_almost_equal
import numpy as np
from .defs import float_to_floats, int_to_nints, int_to_pints
from uts.core.check_arg import check_bool, check_float, check_int


__all__ = ['TestCheck_bool', 'TestCheck_float', 'TestCheck_int']

class TestCheck_bool(TestCase):

    def test_correctinput(self):
        "bool argument: correct input"
        for val in (True, False, bool(1), bool(0)):
            assert check_bool(val) == val

    def test_incorrectinputtype(self):
        "bool argument: incorrect input"
        nodicts(check_bool)
        nosequences(check_bool)
        nochar(check_bool)
        nocomplex(check_bool)
        noint(check_bool)
        nonone(check_bool)
        nofloat(check_bool)


class TestCheck_float(TestCase):

    def test_correctdefaultinput(self):
        "float argument: correct input"
        for val in (-1.0, 0.0, 1.0):
            for f in float_to_floats(val):
                assert check_float(val).dtype == np.float64
                assert check_float(val) == val

    def test_correctcasting(self):
        "float argument: correct casting"
        for val in float_to_floats(1.0):
            for dtype in (float, np.float32, np.float64):
                assert isinstance(check_float(val, cast=dtype), type(dtype(val)))
                assert check_float(val,cast=dtype) == val

    def test_nanarg(self):
        "float argument: nan argument"
        #assert_raises(ValueError, check_float, arg=np.nan)
        assert_raises(ValueError, check_float, arg=np.nan, nan=False)
        assert np.isnan(check_float(arg=np.nan))
        assert np.isnan(check_float(arg=np.nan, nan=True))

    def test_infarg(self):
        "float argument: inf argument"
        #assert_raises(ValueError, check_float, arg=np.inf)
        #assert_raises(ValueError, check_float, arg=-np.inf)
        assert_raises(ValueError, check_float, arg=np.inf, inf=False)
        assert_raises(ValueError, check_float, arg=-np.inf, inf=False)
        assert np.isinf(check_float(arg=np.inf, inf=True))
        assert np.isinf(check_float(arg=-np.inf, inf=True))
        assert np.isinf(check_float(arg=np.inf))
        assert np.isinf(check_float(arg=-np.inf))

    def test_zeroarg(self):
        "float argument: zero argument"
        for val in float_to_floats(0.0):
            for dtype in (float, np.float32, np.float64):
                assert check_float(arg=val, cast=dtype) == 0.0
                assert check_float(arg=val, zero=True, cast=dtype) == 0.0
                assert_raises(ValueError, check_float, arg=val, cast=dtype, zero=False)

    def test_positivearg(self):
        "float argument: positive arg"
        for val in float_to_floats(1.0):
            for dtype in (float, np.float32, np.float64):
                assert check_float(arg=val, cast=dtype) == 1.0
                assert check_float(arg=val, cast=dtype, positive=True) == 1.0
                assert_raises(ValueError, check_float, arg=val, cast=dtype, positive=False)

    def test_negativearg(self):
        "float argument: negative arg"
        for val in float_to_floats(-1.0):
            for dtype in (float, np.float32, np.float64):
                assert check_float(arg=val, cast=dtype) == -1.0
                assert check_float(arg=val, cast=dtype, negative=True) == -1.0
                assert_raises(ValueError, check_float, arg=val, cast=dtype, negative=False)

    def test_incorrectinputtype(self):
        "float argument: incorrect input type"
        nodicts(check_float)
        nosequences(check_float)
        nochar(check_float)
        nocomplex(check_float)
        nobool(check_float)
        noint(check_float)
        nonone(check_float)


class TestCheck_int(TestCase):

    def test_correctinput(self):
        "int argument: correct input and casting"
        for val in int_to_pints(1):
            assert isinstance(check_int(val), (np.int64))
            assert check_int(val) == val
        for dtype in (int, np.uint8, np.int8, np.uint16, np.int16,
                      np.uint32, np.int32, np.int64, np.int64):
            assert isinstance(check_int(val, cast=dtype), type(dtype(val)))
            assert check_int(val,cast=dtype) == val

    def test_incorrectinputtype(self):
        "int argument: incorrect inputtypes"
        nodicts(check_int)
        nosequences(check_int)
        nochar(check_int)
        nocomplex(check_int)
        nobool(check_int)
        nofloat(check_int)
        nonone(check_int)

    def test_negativearg(self):
        "int argument: negative arg"
        for val in int_to_nints(-1.0):
            for dtype in (int, np.int8, np.int16, np.int32, np.int64):
                assert check_int(arg=val, cast=dtype) == -1.0
                assert check_int(arg=val, cast=dtype, negative=True) == -1.0
                assert_raises(ValueError, check_int, arg=val, cast=dtype, negative=False)

    def test_zeroarg(self):
        "int argument: zero arg"
        for val in int_to_pints(0):
            for dtype in (int, np.uint8, np.int8, np.uint16, np.int16,
                      np.uint32, np.int32, np.int64):
                assert check_int(arg=val, cast=dtype) == 0
                assert check_int(arg=val, zero=True, cast=dtype) == 0
                assert_raises(ValueError, check_int, arg=val, cast=dtype, zero=False)

    def test_positivearg(self):
        "int argument: positive arg"
        for val in int_to_pints(1.0):
            for dtype in (int, np.uint8, np.int8, np.uint16, np.int16,
                      np.uint32, np.int32, np.int64):
                assert check_int(arg=val, cast=dtype) == 1
                assert check_int(arg=val, cast=dtype, positive=True) == 1
                assert_raises(ValueError, check_int, arg=val, cast=dtype, positive=False)




def nodicts(func):
    # We do not accept dicts
    assert_raises(TypeError, func, {'a': 1.0})

def nosequences(func):
    # We do not accept sequences
    for val in ((1.0,2.0), [1.0,2.0], np.array([1.0,2.0])):
        assert_raises(TypeError, func, val)

def nochar(func):
    # We do not accept a char
    for val in ('a', np.array('a')):
        assert_raises(TypeError, func, val)

def nocomplex(func):
    # Complex numbers are not allowed as input
    for val in (1+5j, complex(1,5)):
        assert_raises(TypeError, func, val)

def nobool(func):
    # Booleans are not allowed as input
    for val in (True, False, bool(1.0),bool(0.0)):
        assert_raises(TypeError, func, val)

def noint(func):
    # Ints are not allowed as input
    for val in int_to_nints(-1):
        assert_raises(TypeError, func, val)
    for val in int_to_pints(0):
        assert_raises(TypeError, func, val)
    for val in int_to_pints(1):
        assert_raises(TypeError, func, val)

def nonone(func):
    # We do not accept None
    assert_raises(TypeError, func, None)

def nofloat(func):
    # floats are not allowed as input
    for val in float_to_floats(1.0):
        assert_raises(TypeError, func, val)
    for val in float_to_floats(-1.0):
        assert_raises(TypeError, func, val)
    for val in float_to_floats(0.0):
        assert_raises(TypeError, func, val)

if __name__ == '__main__':
    unittest.main()