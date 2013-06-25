#!/usr/bin/env python

"""
Parts of this file were taken from the pandas project
and hence are subject to the terms of the
Lesser GNU General Public License.
Parts are from lxml (https://github.com/lxml/lxml)
"""

from datetime import datetime
from glob import glob
import os
import sys
import shutil
import warnings

# may need to work around setuptools bug by providing a fake Pyrex
try:
    import Cython
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "fake_pyrex"))
except ImportError:
    pass

# try bootstrapping setuptools if it doesn't exist
try:
    import pkg_resources
    try:
        pkg_resources.require("setuptools>=0.6c5")
    except pkg_resources.VersionConflict:
        from ez_setup import use_setuptools
        use_setuptools(version="0.6c5")
    from setuptools import setup, Command
    _have_setuptools = True
except ImportError:
    # no setuptools installed
    from distutils.core import setup, Command
    _have_setuptools = False

setuptools_kwargs = {}
if sys.version_info[0] >= 3:

    setuptools_kwargs = {'use_2to3': True,
                         'zip_safe': False,
                         'install_requires': ['python-dateutil >= 2','numpy'],
                        }
    if not _have_setuptools:
        sys.exit("need setuptools/distribute for Py3k"
            "\n$ pip install distribute")

else:
    setuptools_kwargs = {
        'install_requires': ['python-dateutil < 2', 'numpy'],
        'zip_safe' : False,
    }
    if not _have_setuptools:
        try:
            import numpy
            import dateutil
            setuptools_kwargs = {}
        except ImportError:
            sys.exit("install requires: 'python-dateutil < 2','numpy'."
                     "  use pip or easy_install."
                     "\n   $ pip install 'python-dateutil < 2' 'numpy'")

try:
    import numpy as np
except ImportError:
    nonumpy_msg = ("# numpy needed to finish setup.  run:\n\n"
    "    $ pip install numpy  # or easy_install numpy\n")
    sys.exit(nonumpy_msg)


from distutils.extension import Extension
from distutils.command.build import build
from distutils.command.build_ext import build_ext
from distutils.command.sdist import sdist

from os.path import splitext, basename, join as pjoin

DESCRIPTION = 'uts, a python package for unifrom time series representation'
LONG_DESCRIPTION = """Uniform Time Series

Uniform Time Series is a python package that defines uniform time 
series objects."""

DISTNAME = 'uts'
LICENSE = 'BSD'
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

MAJOR = 0
MINOR = 2
MICRO = 0
ISRELEASED = True
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
QUALIFIER = 'b'

FULLVERSION = VERSION
if not ISRELEASED:
    FULLVERSION += '.dev'
    try:
        import subprocess
        pipe = subprocess.Popen(["git", "rev-parse", "--short", "HEAD"],
                                stdout=subprocess.PIPE).stdout
        rev = pipe.read().strip()

        # makes distutils blow up on Python 2.7
        if sys.version_info[0] >= 3:
            rev = rev.decode('ascii')

        FULLVERSION += "-%s" % rev
    except:
        warnings.warn("WARNING: Couldn't get git revision")
else:
    FULLVERSION += QUALIFIER

def write_version_py(filename='uts/version.py'):
    cnt = """\
version = '%s'
"""
    a = open(filename, 'w')
    try:
        a.write(cnt % FULLVERSION)
    finally:
        a.close()

class CleanCommand(Command):
    """Custom distutils command to clean the .so and .pyc files."""

    user_options = [("all", "a", "") ]

    def initialize_options(self):
        self.all = True
        self._clean_me = []
        self._clean_trees = []
        for root, dirs, files in list(os.walk('uts')):
            for f in files:
                if os.path.splitext(f)[-1] in ('.pyc', '.so', '.o',
                                               '.pyd', '.c'):
                    self._clean_me.append(pjoin(root, f))
            for d in dirs:
                if d == '__pycache__':
                    self._clean_trees.append(pjoin(root, d))

        for d in ('build',):
            if os.path.exists(d):
                self._clean_trees.append(d)

    def finalize_options(self):
        pass

    def run(self):
        for clean_me in self._clean_me:
            try:
                os.unlink(clean_me)
            except Exception:
                pass
        for clean_tree in self._clean_trees:
            try:
                shutil.rmtree(clean_tree)
            except Exception:
                pass

class CheckSDist(sdist):
    """Custom sdist that ensures Cython has compiled all pyx files to c."""

    _pyxfiles = ['uts/core/uniformtime.pyx'
                 'uts/core/geometry.pyx']

    def initialize_options(self):
        sdist.initialize_options(self)

        '''
        self._pyxfiles = []
        for root, dirs, files in os.walk('uts'):
            for f in files:
                if f.endswith('.pyx'):
                    self._pyxfiles.append(pjoin(root, f))
        '''

    def run(self):
        if 'cython' in cmdclass:
            self.run_command('cython')
        else:
            for pyxfile in self._pyxfiles:
                cfile = pyxfile[:-3]+'c'
                msg = "C-source file '%s' not found."%(cfile)+\
                " Run 'setup.py cython' before sdist."
                assert os.path.isfile(cfile), msg
        sdist.run(self)

class CheckingBuildExt(build_ext):
    """Subclass build_ext to get clearer report if Cython is neccessary."""

    def check_cython_extensions(self, extensions):
        for ext in extensions:
          for src in ext.sources:
            if not os.path.exists(src):
                raise Exception("""Cython-generated file '%s' not found.
                Cython is required to compile pandas from a development branch.
                Please install Cython or download a release package of pandas.
                """ % src)

    def build_extensions(self):
        self.check_cython_extensions(self.extensions)
        self.check_extensions_list(self.extensions)

        for ext in self.extensions:
            self.build_extension(ext)

cmdclass = {'clean': CleanCommand,
            'build': build}

try:
    from Cython.Distutils import build_ext
    cython=True
except ImportError:
    cython=False
    suffix = '.c'
    cmdclass['build_ext'] = CheckingBuildExt
else:
    suffix = '.pyx'
    class CythonCommand(build_ext):
        """Custom distutils command subclassed from Cython.Distutils.build_ext
        to compile pyx->c, and stop there. All this does is override the
        C-compile method build_extension() with a no-op."""
        def build_extension(self, ext):
            pass

    class DummyBuildSrc(Command):
        """ numpy's build_src command interferes with Cython's build_ext.
        """
        user_options = []
        def initialize_options(self):
            self.py_modules_dict = {}
        def finalize_options(self):
            pass
        def run(self):
            pass

    cmdclass['build_src'] = DummyBuildSrc
    cmdclass['cython'] = CythonCommand
    cmdclass['build_ext'] =  build_ext
    cmdclass['sdist'] =  CheckSDist


uniformtime_depends = ['geometry']

def srcpath(name=None, suffix='.pyx', subdir='core'):
    return pjoin('uts', subdir, name+suffix)

if suffix == '.pyx':
    uniformtime_depends = [srcpath(f, suffix='.pyx')
                       for f in uniformtime_depends]
else:
    uniformtime_depends = []

uniformtime_ext = Extension('uts.core.uniformtime',
                        depends=uniformtime_depends + ['uts/core/numpy_helper.h'],
                        sources=[srcpath('uniformtime', suffix=suffix)],
                        include_dirs=[np.get_include()],
                        # extra_compile_args=['-Wconversion']
                        )
# check_libraries_ext = Extension('uts.core.check_libraries',
#                        depends=['uts/core/numpy_helper.h'],
#                        sources=[srcpath('check_libraries', suffix=suffix)],
#                        include_dirs=[np.get_include()])
geometry_ext = Extension('uts.core.geometry',
                       depends=['uts/core/numpy_helper.h'],
                       sources=[srcpath('geometry', suffix=suffix)],
                       include_dirs=[np.get_include()])
epoched_ext = Extension('uts.core.epoched',
                       depends=['uts/core/numpy_helper.h'],
                       sources=[srcpath('epoched', suffix=suffix)],
                       include_dirs=[np.get_include()])
multichannel_ext = Extension('uts.core.multichannel',
                       depends=['uts/core/numpy_helper.h'],
                       sources=[srcpath('multichannel', suffix=suffix)],
                       include_dirs=[np.get_include()])
uniformtime_processing_ext = Extension('uts.processing.uniformtime_processing',
                       depends=['uts/core/numpy_helper.h'],
                       sources=[srcpath('uniformtime_processing', suffix=suffix, subdir='processing')],
                       include_dirs=[np.get_include()])

cstft_ext = Extension('uts.processing.cstft',
                       depends=['uts/core/numpy_helper.h'],
                       sources=[srcpath('cstft', suffix=suffix, subdir='processing')],
                       include_dirs=[np.get_include()])

ifdv_ext = Extension('uts.processing.ifdv',
                       depends=['uts/core/numpy_helper.h'],
                       sources=[srcpath('ifdv', suffix=suffix, subdir='processing')],
                       include_dirs=[np.get_include()])
correlate_ext = Extension('uts.processing.correlate',
                       depends=['uts/core/numpy_helper.h'],
                       sources=[srcpath('correlate', suffix=suffix, subdir='processing')],
                       include_dirs=[np.get_include()])
static_ext = Extension('uts.display.static',
                       depends=['uts/core/numpy_helper.h'],
                       sources=[srcpath('static', suffix=suffix, subdir='display')],
                       include_dirs=[np.get_include()])
hdf5_ext = Extension('uts.io.hdf5',
                       depends=['uts/core/numpy_helper.h'],
                       sources=[srcpath('hdf5', suffix=suffix, subdir='io')],
                       include_dirs=[np.get_include()])         

extensions = [uniformtime_ext, geometry_ext, epoched_ext,
              multichannel_ext, uniformtime_processing_ext, cstft_ext, ifdv_ext,
              correlate_ext, static_ext, hdf5_ext]

# if _have_setuptools:
#     setuptools_kwargs["test_suite"] = "nose.collector"

write_version_py()
setup(name=DISTNAME,
      version=FULLVERSION,
      maintainer=MAINTAINER,
      packages=['uts', 'uts.display', 'uts.core', 'uts.processing',
               'uts.config', 'uts.tests','uts.signalfactory', 'uts.helper', 'uts.io'],
      package_data={'uts/example_data' : ['*.h5',
                                   ],
                    'uts/fonts' : ['*.ttf']
                    },
      include_package_data=True,
      ext_modules=extensions,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      cmdclass = cmdclass,
      url=URL,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      classifiers=CLASSIFIERS,
      platforms='any',
      **setuptools_kwargs)
