import sys
import time
import threading
from itertools import count

def foreach(f,l,threads=3,return_=False, args=()):
    """
    Apply f to each element of l, in parallel
    """

    if threads>1:
        iteratorlock = threading.Lock()
        exceptions = []
        if return_:
            n = 0
            d = {}
            i = zip(count(),l.__iter__())
        else:
            i = l.__iter__()


        def runall(args):

            while True:
                iteratorlock.acquire()
                try:
                    try:
                        if exceptions:
                            return
                        v = next(i)#.next()
                    finally:
                        iteratorlock.release()
                except StopIteration:
                    return
                try:
                    if return_:
                        n,x = v
                        d[n] = f(x,*args)
                    else:
                        f(v,*args)
                except Exception:
                    # Deliberately broad: a worker must not die silently, or
                    # the caller would see a partial result and no error. Only
                    # Exception though, so KeyboardInterrupt and SystemExit
                    # still end the run rather than being collected.
                    e = sys.exc_info()
                    iteratorlock.acquire()
                    try:
                        exceptions.append(e)
                    finally:
                        iteratorlock.release()
        
        threadlist = [threading.Thread(target=runall,args=(args,)) for j in range(threads)]
        for t in threadlist:
            t.start()
        for t in threadlist:
            t.join()
        if exceptions:
            # `raise (type, value, traceback)` is python 2 syntax. In python 3
            # it passes a tuple to raise, which fails with "exceptions must
            # derive from BaseException", so every worker failure surfaced as
            # that TypeError instead of the actual error.
            exctype, excvalue, exctraceback = exceptions[0]
            raise excvalue.with_traceback(exctraceback)
        if return_:
            r = sorted(d.items())
            return [v for (n,v) in r]
    else:
        if return_:
            return [f(v,*args) for v in l]
        else:
            for v in l:
                f(v, *args)
            return

def parallel_map(f,l,threads=3):
    return foreach(f,l,threads=threads,return_=True)
