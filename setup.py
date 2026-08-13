"""
Only the Cython extensions are configured here; all packaging metadata lives
in pyproject.toml.

The previous version of this file used distutils, numpy.distutils and
Cython.Distutils.build_ext, all of which are gone or deprecated on current
Python, and it needed setuptools pinned to 58.0.4 to work at all.

"""
import numpy
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

extensionnames = (
    ('uts.processing.ifdv', 'uts/processing/ifdv.pyx'),
    ('uts.processing.correlate', 'uts/processing/correlate.pyx'),
)

extensions = [
    Extension(name, [filename], include_dirs=[numpy.get_include()])
    for name, filename in extensionnames
]

setup(ext_modules=cythonize(extensions, language_level='3'))
