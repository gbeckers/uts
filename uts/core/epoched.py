import numpy as np
import tables as tb
import pandas as pd
from .check_arg import check_arg, check_argarray


def epochs_to_array(epochs):
    """
    Returns epoch info as a numpy structured array, whichever of the three
    supported containers it is held in.

    Epoch info may be a numpy structured array, a pandas DataFrame, or a
    pytables Table. Code that has to treat all three alike -- appending, and
    writing to hdf5 -- goes through this function rather than special-casing
    each container.

    Text columns are returned as fixed-width bytes ('S'), not as object or
    unicode columns. pandas represents text as object dtype, which pytables
    cannot store, and the rest of the package already stores text as bytes
    (see `_save_multichannel` in uts.io.hdf5).

    Parameter
    ---------
    epochs: a numpy structured array, pandas DataFrame, or pytables Table

    """
    if isinstance(epochs, pd.DataFrame):
        columndtypes = {}
        for name in epochs.columns:
            dtype = epochs[name].dtype
            if (dtype == object) or np.issubdtype(dtype, np.str_):
                if len(epochs) == 0:
                    width = 1
                else:
                    width = max(1, int(epochs[name].astype(str).str.len().max()))
                columndtypes[name] = 'S%d' % width
        return epochs.to_records(index=False, column_dtypes=columndtypes)
    elif isinstance(epochs, tb.table.Table):
        return epochs[:]
    elif isinstance(epochs, np.ndarray):
        return epochs
    else:
        raise TypeError(f'epochs type ({type(epochs)}) is not a pandas '
                        'dataframe, a numpy structured array, or h5 table')


def _textlength(dtype):
    """Length in characters of a text dtype ('U' stores 4 bytes per char)."""
    return dtype.itemsize // 4 if dtype.kind == 'U' else dtype.itemsize


def _coerce_epochs_dtype(epochs, dtype):
    """
    Returns structured array `epochs` as exactly `dtype`, matching columns by
    name.

    Needed when appending to a pytables Table, which is fixed-width on disk and
    accepts only rows of its own dtype. numpy's `concatenate` widens text
    columns as needed, so the in-memory containers do not need this.

    Columns are matched by name rather than with `astype`, which casts
    structured arrays by *position* and so silently produces nonsense when the
    column order differs. Text columns that would be truncated raise instead of
    losing characters, which numpy assignment would do without complaint.

    """
    if epochs.dtype == dtype:
        return epochs
    sourcenames = epochs.dtype.names or ()
    missing = [name for name in dtype.names if name not in sourcenames]
    if missing:
        raise ValueError(f'appended epochs lack column(s) {missing} that the '
                         f'existing epoch info has')
    for name in dtype.names:
        target, source = dtype[name], epochs.dtype[name]
        if (target.kind in 'SU') and (source.kind in 'SU') \
                and (_textlength(source) > _textlength(target)):
            raise ValueError(
                f"column '{name}' of the appended epochs holds up to "
                f"{_textlength(source)} characters, but the existing epoch "
                f"info only holds {_textlength(target)}; appending would "
                f"silently truncate it")
    coerced = np.empty(len(epochs), dtype=dtype)
    for name in dtype.names:
        coerced[name] = epochs[name]
    return coerced


class Epoched:
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
            raise IndexError("dimension of epoch index is not equal to one")
        index = index[0]

        # first see if index is an int
        if isinstance(index, int):
            return index

        indices = np.arange(self.nepochs,dtype=np.int64)[index]
        return indices

    def _get_epochs(self, index):
        if isinstance(self.epochs, (pd.DataFrame,)):
            return self.epochs.iloc[index]
        else:
            return np.array(self.epochs[index], ndmin=1)

    def _check_epochs(self, epochs):
        if not isinstance(epochs, (np.ndarray, tb.table.Table, pd.DataFrame)):
            raise TypeError(f'epochs type ({type(epochs)}) is not a pandas dataframe, '
                            'a numpy structured array, or h5 table')
        if isinstance(epochs,np.ndarray) and (epochs.dtype.fields is None):
            raise TypeError(f'epochs type ({type(epochs)}) is not '
                            'a numpy structured array (or a pandas dataframe or h5 table)')

        return epochs
    
    def _check_starttimes(self, epochs):
        if isinstance(epochs, tb.table.Table):
            return check_argarray(epochs.cols.starttime[:], 'time')
        else:
            return check_argarray(epochs['starttime'], 'time')
    
    def _append_epochs(self, epochs):
        """
        Appends epoch info, keeping the container type this object was built
        with. The appended `epochs` may be held in any of the three supported
        containers, regardless of which one this object uses.

        """
        epochs = self._check_epochs(epochs)
        # Read everything needed out of `epochs` before mutating anything: it
        # may be the very table this object holds, and a pytables Table is
        # extended in place, so reading afterwards would read back the rows
        # just written and count them twice.
        newstarttimes = self._check_starttimes(epochs)
        if isinstance(self.__epochs, tb.table.Table):
            # An on-disk table needs rows of its own dtype, not a container.
            self.__epochs.append(
                _coerce_epochs_dtype(epochs_to_array(epochs),
                                     self.__epochs.description._v_dtype))
            self.__epochs.flush()
        elif isinstance(self.__epochs, pd.DataFrame):
            if not isinstance(epochs, pd.DataFrame):
                epochs = pd.DataFrame(epochs_to_array(epochs))
            # pandas.DataFrame.append was removed in pandas 2.0. It never
            # appended in place either, so its return value used to be
            # discarded here, leaving nepochs and starttimes inconsistent.
            self.__epochs = pd.concat([self.__epochs, epochs],
                                      ignore_index=True)
        else:
            self.__epochs = np.concatenate([self.__epochs,
                                            epochs_to_array(epochs)])
        self.__nepochs = len(self.__epochs)
        self.__starttimes = np.concatenate([self.__starttimes, newstarttimes])
        self.__endtimes = self.__starttimes + self.__duration
            
    
    # #this has not been tested
    # def append_recordcolumns(self, epochs):
    #     if not isinstance(epochs, np.ndarray) or (epochs.dtype.fields is None):
    #         raise TypeError('epochs parameter is not a numpy structured array')
    #     if not len(epochs) == len(self.__epochs):
    #         raise ValueError('number of epochs do not match (%s vs %s)' % (len(self.__epochs), len(epochs)))
    #     newdescr = self.__epochs.dtype.descr + epochs.dtype.descr
    #     size = len(self.__epochs)
    #     newepochs = np.empty(size,dtype=newdescr)
    #     for descr in self.__epochs.dtype.descr:
    #         newepochs[descr[0]] = self.__epochs[descr[0]]
    #     for descr in epochs.dtype.descr:
    #         newepochs[descr[0]] = epochs[descr[0]]
    #     self.__epochs = newepochs
    #
        
        
        
        
        
        