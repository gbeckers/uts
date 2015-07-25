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
    try:
        tempdirname = tempfile.mkdtemp(dir=dir)
        if report:
            print('created cache file {}'.format(tempdirname))
        yield tempdirname
    except:
        raise
    finally:
        if not keep:
            shutil.rmtree(tempdirname)
            if report:
                print('removed cache file {}'.format(tempdirname))

def make_dirs(path, mode=0777, existserror=False):
    from os import makedirs
    from os.path import exists
    if not exists(path):
        makedirs(path, mode=mode)
    elif existserror:
        raise IOError('Path "{}" already exists'.format(path))
