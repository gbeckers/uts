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

cmdclass = versioneer.get_cmdclass()

DESCRIPTION = 'uts, a python package for unifrom time series representation'
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


# MAJOR = 1
# MINOR = 0
# MICRO = 2
# ISRELEASED = False
# VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
# QUALIFIER = 'b'
#
# FULLVERSION = VERSION
# if not ISRELEASED:
#     FULLVERSION += '.dev'
# else:
#     FULLVERSION += QUALIFIER

include_dirs = numpy.distutils.misc_util.get_numpy_include_dirs()

extensionfiles = (
    ('uts.core.uniformtime',                  'uts/core/uniformtime.pyx'),
    ('uts.core.geometry',                     'uts/core/geometry.pyx'),
    ('uts.core.epoched',                      'uts/core/epoched.pyx'),
    ('uts.core.multichannel',                 'uts/core/multichannel.pyx'),
    ('uts.core.uniformfrequency',             'uts/core/uniformfrequency.pyx'),
    ('uts.core.uniformtimefrequency',         'uts/core/uniformtimefrequency.pyx'),
    ('uts.processing.uniformtime_processing', 'uts/processing/uniformtime_processing.pyx'),
    ('uts.processing.cstft',                  'uts/processing/cstft.pyx'),
    ('uts.processing.ifdv',                   'uts/processing/ifdv.pyx'),
    ('uts.processing.correlate',              'uts/processing/correlate.pyx'),
    ('uts.viz.static',                    'uts/viz/static.pyx'),
    ('uts.viz.movies',                    'uts/viz/movies.pyx'),
    ('uts.io.hdf5',                           'uts/io/hdf5.pyx')
)

extensions = [Extension(extname, [filename], include_dirs=include_dirs) for extname, filename in extensionfiles]

# uniformtime_ext = Extension('uts.core.uniformtime',
#                             ['uts/core/uniformtime.pyx'],
#                         include_dirs=[include_dirs],
#                         )

def get_configdir(create=True):
    """
Returns the uts configuration directory name.
:param bool create:
If True, the directory will be created if it doesn't exist.
:returns: The absolute path to the config directory as a string
"""
    import os,sys
    from os import environ as env

    #First find the home directory - this is inspired by the scheme ipython
    #uses to identify "home"
    if os.name == 'posix':
        # Linux, Unix, AIX, OS X
        if 'HOME' in env:
            homedir = env['HOME'].decode(sys.getfilesystemencoding())
        else:
            raise OSError('Could not find unix home directory to search for astropysics config dir')
    elif os.name == 'nt': # This is for all modern Windows (NT or after)
        #Try for a network home first
        if 'HOMESHARE' in env:
            homedir = env['HOMESHARE'].decode(sys.getfilesystemencoding())
        #See if there's a local home
        elif 'HOMEDRIVE' in env and 'HOMEPATH' in env:
            homedir = os.path.join(env['HOMEDRIVE'],env['HOMEPATH'])
            homedir = homedir.decode(sys.getfilesystemencoding())
        #Maybe a user profile?
        elif 'USERPROFILE' in env:
            homedir = os.path.join(env['USERPROFILE']).decode(sys.getfilesystemencoding())
        else:
            try:
                import _winreg as wreg
                key = wreg.OpenKey(
                    wreg.HKEY_CURRENT_USER,
                    "Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders"
                )
                homedir = wreg.QueryValueEx(key,'Personal')[0]
                homedir = homedir.decode(sys.getfilesystemencoding())
                key.Close()
            except:
                #As a final possible resort, see if HOME is present for some reason
                if 'HOME' in env:
                    homedir = env['HOME'].decode(sys.getfilesystemencoding())
                else:
                    raise OSError('Could not find windows home directory to search for utd config dir')
    else:
        #for other platforms, try HOME, although it probably isn't there
        if 'HOME' in env:
            homedir = env['HOME'].decode(sys.getfilesystemencoding())
        else:
            raise OSError('Could not find a home directory to search for uts config dir - are you on an unspported platform?')
        
    configdir = os.path.realpath(os.path.join(homedir,'.uts'))
        
    if create and not os.path.isdir(configdir):
        #try to create it
        os.mkdir(configdir)
    return configdir


configdir=get_configdir(True)

data_files = [(configdir, ['uts/config/configrc'])]


setup(
  name = DISTNAME,
  # cmdclass = {'build_ext': build_ext},
  ext_modules = extensions,
  version=versioneer.get_version(),
  cmdclass=versioneer.get_cmdclass(),
  maintainer=MAINTAINER,
  packages=['uts', 'uts.display', 'uts.core', 'uts.processing',
           'uts.config', 'uts.tests','uts.signalfactory', 'uts.helper',
           'uts.io', 'uts.viz', 'uts.exampledata'],
  data_files=data_files,
  package_data={'uts.display' : ['*.ttf'],
                'uts.exampledata': ['*.h5']},
  include_package_data=True,
  maintainer_email=MAINTAINER_EMAIL,
  description=DESCRIPTION,
  license=LICENSE,
  url=URL,
  download_url=DOWNLOAD_URL,
  long_description=LONG_DESCRIPTION,
  classifiers=CLASSIFIERS,
  platforms='any', requires=['matplotlib', 'scipy']
)