import numpy as np
import numexpr
from uts3 import UniformTimeSeries, \
    MultiChannelUniformTimeSeries, EpochedUniformTimeSeries, \
    EpochedMultiChannelUniformTimeSeries
from uts3.core.samplesource import NumpyArraySamples, h5ArraySamples
from frequencysampled import Spectrum
from timefrequencysampled import Spectrogram
from uts3.core.check_arg import check_arg, check_float, check_floatarray, check_argarray


__all__ = ['add', 'power', 'scale', 'abs', 'sin', 'cos', 'tan', 
           'arcsin', 'arccos', 'arctan', 'arctan2',
           'sinh', 'cosh', 'tanh',
           'arcsinh', 'arccosh', 'arctanh', 
           'log', 'log10', 'log1p',
           'exp', 'expm1', 'sqrt',
           'real','imag', 'eval_expr']

def _assert_signal(ts, signaltypes=None):
    allowedtypes = (UniformTimeSeries, MultiChannelUniformTimeSeries,
            EpochedUniformTimeSeries, EpochedMultiChannelUniformTimeSeries,
            Spectrum, Spectrogram) 
    if signaltypes is None: signaltypes = allowedtypes
    if not ts.__class__ in signaltypes:
        raise TypeError(
            "object is not the correct ds signal type (allowed: %s)" \
            % signaltypes)
    return True

## Functions that are applied on element-wise on samples

def eval_expr(s, expr, out=None, **kwargs):
    """
    'a' in expr refers to the samples of s. Provide all other
    variables as keyword arguments.
    
    """
    _assert_signal(s)
    if not isinstance(s.samples, (NumpyArraySamples, h5ArraySamples)):
        NotImplemented, "at the moment only numpy or hdf5 samples allowed" 
    if isinstance(s.samples, NumpyArraySamples):
        kwargs['a'] = s.samples.array
    else:
        kwargs['a'] = s.samples.h5array
    if not (out is None):
        _assert_signal(out)        
        if isinstance(out.samples, NumpyArraySamples):
            r = out.samples.array
        elif isinstance(out.samples, h5ArraySamples):
            r = out.samples.h5array
        else:
            NotImplemented, "at the moment only numpy or hdf5 samples allowed"
        ## This should be done in place, but numexpr does not allow this anymore
        r[:] = numexpr.evaluate(expr, kwargs)
        return None
    else:
        return s._instantiatefromself(samples=numexpr.evaluate(expr, kwargs))

def add(s, val, out=None):
    val = np.array(val,ndmin=1)
    if val.size == 1:
        val = val[0]
    expr = "a+val"
    return eval_expr(s=s, expr=expr, out=out, val=val)

def power(s, p, out=None):
    """
    Returns signal raised to the power p.
    
    Parameters
    ----------
    s : sampled signal
        The bases.
    p : float
        The exponent.
    
    Returns
    -------
    Sampled signal of the same type and dimensions
    
    """
    p = np.array(p,ndmin=1) 
    if p.size == 1:
        p = p[0]
    expr = "a**p"
    return eval_expr(s=s, expr=expr, out=out, p=p)

def scale(s, factor, out=None):
    factor = np.array(factor,ndmin=1)
    if factor.size == 1:
        factor = factor[0]
    expr = "a*factor"
    return eval_expr(s=s, expr=expr, out=out, factor=factor)

def abs(s, out=None):
    expr = "abs(a)"
    return eval_expr(s=s, expr=expr, out=out)

def sin(s, out=None):
    expr = "sin(a)"
    return eval_expr(s=s, expr=expr, out=out)

def cos(s, out=None):
    expr = "cos(a)"
    return eval_expr(s=s, expr=expr, out=out)

def tan(s, out=None):
    expr = "tan(a)"
    return eval_expr(s=s, expr=expr, out=out)

def arcsin(s, out=None):
    expr = "arcsin(a)"
    return eval_expr(s=s, expr=expr, out=out)

def arccos(s, out=None):
    expr = "arccos(a)"
    return eval_expr(s=s, expr=expr, out=out)

def arctan(s, out=None):
    expr = "arctan(a)"
    return eval_expr(s=s, expr=expr, out=out)

def arctan2(s, out=None):
    expr = "arctan2(a)"
    return eval_expr(s=s, expr=expr, out=out)

def sinh(s, out=None):
    expr = "sinh(a)"
    return eval_expr(s=s, expr=expr, out=out)

def cosh(s, out=None):
    expr = "cosh(a)"
    return eval_expr(s=s, expr=expr, out=out)

def tanh(s, out=None):
    expr = "tanh(a)"
    return eval_expr(s=s, expr=expr, out=out)

def arcsinh(s, out=None):
    expr = "arcsinh(a)"
    return eval_expr(s=s, expr=expr, out=out)

def arccosh(s, out=None):
    expr = "arccosh(a)"
    return eval_expr(s=s, expr=expr, out=out)

def arctanh(s, out=None):
    expr = "arctanh(a)"
    return eval_expr(s=s, expr=expr, out=out)

def log(s, out=None):
    expr = "log(a)"
    return eval_expr(s=s, expr=expr, out=out)

def log10(s, out=None):
    expr = "log10(a)"
    return eval_expr(s=s, expr=expr, out=out)

def log1p(s, out=None):
    expr = "log1p(a)"
    return eval_expr(s=s, expr=expr, out=out)

def exp(s, out=None):
    expr = "exp(a)"
    return eval_expr(s=s, expr=expr, out=out)

def expm1(s, out=None):
    expr = "expm1(a)"
    return eval_expr(s=s, expr=expr, out=out)

def sqrt(s, out=None):
    expr = "sqrt(a)"
    return eval_expr(s=s, expr=expr, out=out)

def real(s, out=None):
    expr = "real(a)"
    return eval_expr(s=s, expr=expr, out=out)

def imag(s, out=None):
    expr = "imag(a)"
    return eval_expr(s=s, expr=expr, out=out)

