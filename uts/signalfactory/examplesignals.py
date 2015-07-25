import os
from ..io.hdf5 import load
__all__ = ['load_doughnutexampleraw', 'load_doughnutexamplelfp','load_doughnutexampleamua',
           'load_doughnutexamplemua']

here = os.path.dirname(__file__)

def load_doughnutexampleraw():
    filename = os.path.join(here, '..', 'exampledata', 'examples.h5')
    return load(signalname='/doughnut', filename=filename)

def load_doughnutexamplelfp():
    filename = os.path.join(here, '..', 'exampledata', 'examples.h5')
    return load(signalname='/doughnut_lfp', filename=filename)

def load_doughnutexampleamua():
    filename = os.path.join(here, '..', 'exampledata', 'examples.h5')
    return load(signalname='/doughnut_amua', filename=filename)

def load_doughnutexamplemua():
    filename = os.path.join(here, '..', 'exampledata', 'examples.h5')
    return load(signalname='/doughnut_mua', filename=filename)