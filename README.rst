UTS
===

.. image:: https://github.com/gbeckers/uts/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/gbeckers/uts/actions/workflows/tests.yml
   :alt: test status

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

pytest is the fuller run: it collects ten module-level test functions that
``uts.test()`` skips, because that runner loads only ``TestCase`` subclasses.
Both runs should be entirely green.

Every push runs the suite in three configurations: the exact pins in
``environment.yml``, the same python with current numpy 2 and pandas 3, and
the newest python with everything current. Linux only.

Authors
-------

The `uts` library has been written by *Gabriel Beckers* up to version 2.4.5. That
version has been audited by *Claude Opus 5* with the request to identify and fix
any bugs, one by one after inspection and approval by Gabriel. Further, the
library was made compatible with modern versions of Numpy and Python. Each change
was inspected by Gabriel. The library had an extensive tests suite before the
audit, which ran fine after the changes by Claude. After this version 2.5.0 was
released.

AI Statement
------------
Starting with version 2.5.0 Claude is used to audit the human-designed and -written
code for bugs and compatibility. Identified issues are fixed, with regression tests,
by Claude after approval and under supervision of the human programmer who wrote the
code.