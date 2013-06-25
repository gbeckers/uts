import numpy as np
import tables
from uts3.core.check_arg import check_arg, check_argarray

class Epoched(object):
    """
    Represents an epoched time series.

    Parameters
    ----------
    duration: float
        Duration of 1 epoch in seconds.
    epochs: structured array
        Information about the epochs. Should at least contain the field 
        "starttime".
    origintime: float, optional
           Time of origin relative to beginning of the first sample of epoch. 
           Defaults to 0.0.
    
    """

    def __init__(self, duration, epochs, origintime=0.0):
        
        self.__duration = check_arg(duration, 'duration')
        self.__origintime = check_arg(origintime, 'time')
        self.__epochs = self._check_epochs(epochs=epochs)
        self.__nepochs = len(self.__epochs)
        self.__starttimes = self._check_starttimes(epochs)
        self.__endtimes = self.__starttimes + self.__duration

    starttimes = property(lambda self: self.__starttimes, None, None,
                        "starttimes of epochs")

    endtimes = property(lambda self: self.__endtimes, None, None,
                        "endtimes of epochs")

    duration = property(lambda self: self.__duration, None, None,
                        "duration of epochs")

    origintime = property(lambda self: self.__origintime, None, None,
                        "origintime of epochs, relative to their starttimes")

    nepochs = property(lambda self: self.__nepochs, None, None,
                        "number of epochs")

    epochs = property(lambda self: self.__epochs, None, None,
                        "epoch info")

    def _to_epochsindex(self, index):
        index = np.index_exp[index]
        if len(index) != 1:
            raise IndexError("index of epoch dimension is not equal to one")
        index = index[0]

        # first see if index is an int
        if isinstance(index, (int, long)):
            return index

        indices = np.arange(self.nepochs,dtype=np.int)[index]
        return indices

    def _check_epochs(self, epochs):
        if not isinstance(epochs, (np.ndarray, tables.table.Table)) or (epochs.dtype.fields is None):
            raise TypeError('epochs parameter is not a numpy structured array or h5 table')
        if not epochs.dtype.fields.has_key('starttime'):
            raise ValueError('epochs do not have a "starttime" field')
        return epochs
    
    def _check_starttimes(self, epochs):
        if isinstance(epochs, tables.table.Table):
            return check_argarray(epochs.cols.starttime[:], 'time')
        else:
            return check_argarray(epochs['starttime'], 'time')
    
    def _append_epochs(self, epochs):
        epochs = self._check_epochs(epochs)
        if isinstance(self.__epochs, tables.table.Table):
            self.__epochs.append(epochs)
        else:
            self.__epochs = np.concatenate([self.epochs, epochs])
        self.__nepochs = len(self.__epochs)
        self.__starttimes = np.concatenate([self.__starttimes, self._check_starttimes(epochs)])
        self.__endtimes = self.__starttimes + self.__duration
            
    
    #this has not been tested
    def append_recordcolumns(self, epochs):
        if not isinstance(epochs, np.ndarray) or (epochs.dtype.fields is None):
            raise TypeError('epochs parameter is not a numpy structured array')
        if not len(epochs) == len(self.__epochs):
            raise ValueError('number of epochs do not match (%s vs %s)' % (len(self.__epochs), len(epochs)))
        newdescr = self.__epochs.dtype.descr + epochs.dtype.descr
        size = len(self.__epochs)
        newepochs = np.empty(size,dtype=newdescr)
        for descr in self.__epochs.dtype.descr:
            newepochs[descr[0]] = self.__epochs[descr[0]]
        for descr in epochs.dtype.descr:
            newepochs[descr[0]] = epochs[descr[0]]
        self.__epochs = newepochs
            
        
        
        
        
        
        