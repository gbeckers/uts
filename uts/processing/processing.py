#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from uts import UniformTime
import uniformtime_processing
import correlate

__all__ = ['bp', 'convolve', 'crosscorrelatetime','crosscorrelatetime_betweenchannels','crosscorrelatetime_betweenchannels_p',
           'decimate', 'iter_decimate', 'fft', 'hp', 'ibp', 'ica',
           'ifdv', 'ihp', 'ilp', 'iter_convolve', 'iter_convolve_info', 'lp',
           'overlap_and_add', 'ramp', 'resample', 'stft', 'stmtt', 'welch', 'specgram']


bp = uniformtime_processing.bp

convolve = uniformtime_processing.convolve
crosscorrelatetime = uniformtime_processing.crosscorrelatetime
crosscorrelatetime_betweenchannels = uniformtime_processing.crosscorrelatetime_betweenchannels
crosscorrelatetime_betweenchannels_p = uniformtime_processing.crosscorrelatetime_betweenchannels_p

# def decimate(s, *args, **kwargs):
#     if isinstance(s, UniformTime):
#         return uniformtime_processing.decimate(s, *args, **kwargs)
#     else:
#         raise NotImplementedError(
#             "cannot decimate signal of type '%s'" % type(s))
# decimate.__doc__ = uniformtime_processing.decimate.__doc__

decimate = uniformtime_processing.decimate
iter_decimate = uniformtime_processing.iter_decimate


fft = uniformtime_processing.fft

welch = uniformtime_processing.welch

specgram = uniformtime_processing.specgram
# periodogram = uniformtime_processing.periodogram

ibp = uniformtime_processing.ibp

ica = uniformtime_processing.ica

ifdv = uniformtime_processing.ifdv

ilp = uniformtime_processing.ilp
ihp = uniformtime_processing.ihp

iter_convolve = uniformtime_processing.iter_convolve
iter_convolve_info = uniformtime_processing.iter_convolve_info

lp = uniformtime_processing.lp
hp = uniformtime_processing.hp

overlap_and_add = uniformtime_processing.overlap_and_add

ramp = uniformtime_processing.ramp

def resample(s, *args, **kwargs):
    if isinstance(s, UniformTime):
        return uniformtime_processing.resample(s, *args, **kwargs)
    else:
        raise NotImplementedError("cannot resample this signal")
resample.__doc__ = uniformtime_processing.resample.__doc__

stft = uniformtime_processing.stft

stmtt = uniformtime_processing.stmtt



    
    


