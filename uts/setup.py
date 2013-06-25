#!/usr/bin/env python

import numpy

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('uts', parent_package, top_path)
    config.add_subpackage('core')
    config.add_subpackage('processing')
    config.add_subpackage('io')
    config.add_subpackage('signalfactory')
    config.add_subpackage('display')
    config.add_subpackage('config')
    config.add_subpackage('helper')
    config.add_data_dir('tests')
    config.add_data_dir('example_data')
    config.add_data_dir('fonts')

    config.add_extension('core/uniformtime',
                         sources=['core/uniformtime.c'],
                         include_dirs=[numpy.get_include()])
    # config.add_extension('core/check_libraries',
    #                      sources=['core/check_libraries.c'],
    #                      include_dirs=[numpy.get_include()])
    config.add_extension('core/geometry',
                         sources=['core/geometry.c'],
                         include_dirs=[numpy.get_include()])
    config.add_extension('core/multichannel',
                         sources=['core/multichannel.c'],
                         include_dirs=[numpy.get_include()])
    config.add_extension('core/epoched',
                         sources=['core/epoched.c'],
                         include_dirs=[numpy.get_include()])
    config.add_extension('processing/uniformtime_processing',
                         sources=['processing/uniformtime_processing.c'],
                         include_dirs=[numpy.get_include()])
    config.add_extension('processing/cstft',
                         sources=['processing/cstft.c'],
                         include_dirs=[numpy.get_include()])
    config.add_extension('processing/ifdv',
                         sources=['processing/ifdv.c'],
                         include_dirs=[numpy.get_include()])
    config.add_extension('processing/correlate',
                         sources=['processing/correlate.c'],
                         include_dirs=[numpy.get_include()])
    config.add_extension('display/static',
                         sources=['display/static.c'],
                         include_dirs=[numpy.get_include()])
    config.add_extension('io/hdf5',
                         sources=['io/hdf5.c'],
                         include_dirs=[numpy.get_include()])
    return config

if __name__ == '__main__':
    print('This is the wrong setup.py file to run')

