UTS
===

A python package for uniform time series manipulation and analysis. Works efficiently
with very large signals that may be multichannel and/or epoched.

Installation
------------

uts contains Cython extensions, so a C compiler is needed. On Windows, install
the Visual C++ build tools first if you have not already:

https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019

You only need the "Build Tools for Visual Studio 2019", but installing the whole
studio is fine as well.

Reproducing an analysis
^^^^^^^^^^^^^^^^^^^^^^^

``environment.yml`` pins the exact versions the test suite passes with. Use it
if you want the same environment an analysis was run in: ::

    $ conda env create -f environment.yml
    $ conda activate uts
    $ pip install -e . --no-build-isolation

``--no-build-isolation`` makes the extensions compile against the numpy pinned
in the environment. Without it, the build fetches its own numpy, which may be a
different major version from the one the extensions are then imported alongside.

Installing normally
^^^^^^^^^^^^^^^^^^^

darr is on conda-forge, so that channel may need appending (safe to repeat): ::

    $ conda config --append channels conda-forge

Create an environment and install the dependencies: ::

    $ conda create -n uts python=3.11 numpy scipy pandas pytables h5py numexpr darr cython
    $ conda activate uts

Then install the latest uts from GitHub: ::

    $ pip install git+https://github.com/gbeckers/uts@master

Both numpy 1 and numpy 2 work. The suite is run against python 3.11 with
numpy 1.25 and pandas 2.0, and against python 3.14 with numpy 2.5 and
pandas 3.0.

Running the tests
-----------------

::

    >>> import uts
    >>> uts.test()

pytest also works: ::

    $ pytest --pyargs uts

Note that pytest collects ten more tests than ``uts.test()`` does, because the
latter loads only ``TestCase`` subclasses and so skips module-level test
functions. Two of those extra tests currently fail; both date from an earlier
version of the time API and have not yet been reconciled with it.
