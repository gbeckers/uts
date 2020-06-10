try:
    import setuptools
except ImportError:
    pass
import sys
import versioneer
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy.distutils.misc_util
import numpy

if sys.version_info < (3,6):
    print("uts requires Python 3.6 or higher please upgrade")
    sys.exit(1)

description = 'uts, a python package for uniform time series representation'
long_description = """Uniform Time Series

Uniform Time Series is a python package that defines uniform time
series objects."""

include_dirs = numpy.distutils.misc_util.get_numpy_include_dirs()

extensionfiles = (
    ('uts.processing.cstft', 'uts/processing/cstft.pyx'),
    ('uts.processing.ifdv', 'uts/processing/ifdv.pyx'),
    ('uts.processing.correlate', 'uts/processing/correlate.pyx'),
)

extensions = [Extension(extname, [filename], include_dirs=include_dirs) for extname, filename in extensionfiles]

cmdclass = versioneer.get_cmdclass()
cmdclass['build_ext'] =  build_ext

setup(
    name='uts',
    ext_modules = extensions,
    version=versioneer.get_version(),
    cmdclass=cmdclass,
    packages = ['uts', 'uts.core', 'uts.processing',
                'uts.tests', 'uts.signalfactory', 'uts.helper',
                'uts.io', 'uts.exampledata', 'uts.testdata'],
    package_data={'uts.exampledata': ['*.h5'],
                  'uts.testdata':['*.mat']},
    include_package_data=True,
    license='BSD-3',
    author='Gabriel J.L. Beckers',
    author_email='gabriel@gbeckers.nl',
    description=description,
    long_description=long_description,
    long_description_content_type="text/x-rst",
    python_requires='>=3.6',
    install_requires=['numpy','cython'],
    data_files = [("", ["LICENSE"])],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
    ],
    project_urls={  # Optional
        'Source': 'https://github.com/gbeckers/',
    },
)
