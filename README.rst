UTS
===

A python package for uniform time series manipulation and analysis. Works efficiently
with very large signals that may be multichannel and/or epoched.

Installation
------------

The best is to create a conda environment in Anaconda, with a number of packages with specific versions.

On Windows, you'll first need to install visual C++ build tools if you haven't done so yet:

https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019

You only need the "Build Tools for Visual Studio 2019", but if you install the whole studio, that is fine as well.

Most Python packages you need are in the default conda channel. But you
will also need two additional packages. One package, darr, is in the
conda-forge channel. This channel may need to be appended to conda's
channel list (if it is already appended, then the next is still safe): ::

    $ conda config --append channels conda-forge

In a terminal, use the following to create an environment called "sl2020a": ::

    $  conda create -n Sl2020a python=3.8 scipy darr cython h5py pytables

You can do the above also in the Anaconda GUI, but you need the terminal for the next anyway:

To install the latest uts from Github: ::

    $ pip install git+https://github.com/gbeckers/uts@master