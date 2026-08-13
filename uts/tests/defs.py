import numpy as np

nan = np.nan
inf = np.inf

eps = np.power(2.0,-52.0)

# floats, positive, negative, zero
pfloats = [1.0, np.float32(1.0), np.float64(1.0)]
nfloats = [-1.0, np.float32(-1.0), np.float64(-1.0)]
zfloats = [0.0, np.float32(0.0), np.float64(0.0)]

floattypes = [float, np.float32, np.float64]
inttypes = [int, np.uint8, np.int8, np.uint16, np.int16, np.uint32,
            np.int32, np.uint64, np.int64]

def int_to_pints(x):
    """Returns a list of positive ints of different types. Takes care of
    overflow issues.

    Parameter:
    ----------
    x: int

    """
    if x < 0: raise ValueError("x < 0")
    ints = []
    if x <= 127: ints.append(np.int8(x))
    if x <= 255: ints.append(np.uint8(x))
    if x <= 32767: ints.append(np.int16(x))
    if x <= 65535: ints.append(np.uint16(x))
    if x <= 2147483647: ints.extend([np.int32(x), int(x)])
    if x <= 4294967295: ints.extend([np.uint32(x)])
    if x <= 9223372036854775807: ints.append(np.int64(x))
    #if x <= 18446744073709551615: ints.append(np.uint64(x))
    return ints

def int_to_nints(x):
    """Returns a list of negative ints of different types. Takes care of
    overflow issues.

    Parameter:
    ----------
    x: int

    """
    if x > 0: raise ValueError("x > 0")
    ints = []
    if x >= (-127 - 1): ints.append(np.int8(x))
    if x >= (-32767 - 1): ints.append(np.int16(x))
    if x >= (-2147483647 - 1): ints.extend([np.int32(x), int(x)])
    if x >= (-9223372036854775807 - 1): ints.append(np.int64(x))
    return ints

def float_to_floats(x):
    return [float(x), np.float32(x), np.float64(x)]


# ints, positive, negative, zero
pints = int_to_pints(1) + [int(np.power(np.int64(2),32))]
zints = int_to_pints(0)
nints = int_to_nints(-1) + [-int(np.power(np.int64(2),32))]

