from __future__ import print_function
from contextlib import contextmanager

__all__ = ['tempdir']

@contextmanager
def tempdir(dir='.', keep=False, report=False):
    '''
    :param dir:
    :param keep: can be set to True for testing purposes
    :return:
    '''
    import tempfile
    import shutil
    # Created before the try: if mkdtemp itself fails there is nothing to clean
    # up, and running the finally block would raise NameError on the unbound
    # tempdirname, hiding the real error.
    tempdirname = tempfile.mkdtemp(dir=dir)
    if report:
        print('created cache dir {}'.format(tempdirname))
    try:
        yield tempdirname
    finally:
        if not keep:
            shutil.rmtree(tempdirname)
            if report:
                print('removed cache dir {}'.format(tempdirname))

def make_dirs(path, mode=0o777, existserror=False):
    from os import makedirs
    from os.path import exists
    if not exists(path):
        makedirs(path, mode=mode)
    elif existserror:
        raise IOError('Path "{}" already exists'.format(path))
