#!/usr/bin/env python
try:
    import setuptools
except ImportError:
    pass
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy.distutils.misc_util
import versioneer

DESCRIPTION = 'uts, a python package for uniform time series representation'
LONG_DESCRIPTION = """Uniform Time Series

Uniform Time Series is a python package that defines uniform time
series objects."""

DISTNAME = 'uts'
LICENSE = 'commercial'
AUTHOR = "Gabriel Beckers"
MAINTAINER = "Gabriel Beckers"
MAINTAINER_EMAIL = "gabriel@gbeckers.nl"
URL = "http://gbeckers.nl"
DOWNLOAD_URL = ''
CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Environment :: Console',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 3',
    'Programming Language :: Cython',
    'Topic :: Scientific/Engineering',
]

include_dirs = numpy.distutils.misc_util.get_numpy_include_dirs()

extensionfiles = (
    ('uts.processing.cstft',                  'uts/processing/cstft.pyx'),
    ('uts.processing.ifdv',                   'uts/processing/ifdv.pyx'),
    ('uts.processing.correlate',              'uts/processing/correlate.pyx'),
)

extensions = [Extension(extname, [filename], include_dirs=include_dirs) for extname, filename in extensionfiles]

cmdclass = versioneer.get_cmdclass()
cmdclass['build_ext'] =  build_ext

setup(
  name = DISTNAME,
  ext_modules = extensions,
  version=versioneer.get_version(),
  cmdclass=cmdclass,
  maintainer=MAINTAINER,
  packages=['uts', 'uts.core', 'uts.processing',
            'uts.tests','uts.signalfactory', 'uts.helper',
           'uts.io', 'uts.exampledata'],
  package_data={'uts.exampledata': ['*.h5']},
  include_package_data=True,
  maintainer_email=MAINTAINER_EMAIL,
  description=DESCRIPTION,
  license=LICENSE,
  url=URL,
  download_url=DOWNLOAD_URL,
  long_description=LONG_DESCRIPTION,
  classifiers=CLASSIFIERS,
  platforms='any', requires=['scipy', 'pytables', 'h5py']
)