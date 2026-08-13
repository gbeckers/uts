"""
Tests for the Epoched class and epoch info handling.

Epoch info may be held in a numpy structured array, a pandas DataFrame, or a
pytables Table. These tests exercise all three, because only the structured
array path was covered before.

"""
import os
from unittest import TestCase

import numpy as np
import pandas as pd
import tables as tb
from numpy.testing import assert_raises, assert_array_equal

import uts
from uts import EpochedUniformTimeSeries, saveh5, loadh5
from uts.core.epoched import Epoched, epochs_to_array
from uts.io.utils import tempdir


def structured_epochs():
    a = np.zeros(2, dtype=[('starttime', np.float64), ('event', '|S5')])
    a['starttime'] = [0., 1.]
    a['event'] = [b'left', b'right']
    return a


def dataframe_epochs():
    return pd.DataFrame({'starttime': [0., 1.], 'event': ['left', 'right']})


class TestEpochsToArray(TestCase):

    def test_structured_array_passes_through(self):
        "epochs_to_array: a structured array is returned unchanged"
        a = structured_epochs()
        assert epochs_to_array(a) is a

    def test_dataframe(self):
        "epochs_to_array: a DataFrame becomes a structured array"
        r = epochs_to_array(dataframe_epochs())
        assert r.dtype.names == ('starttime', 'event')
        assert_array_equal(r['starttime'], [0., 1.])

    def test_dataframe_text_is_bytes_not_object(self):
        "epochs_to_array: text columns become fixed-width bytes"
        # pandas holds text as object dtype, which pytables cannot store.
        r = epochs_to_array(dataframe_epochs())
        assert r.dtype['event'].kind == 'S'
        assert_array_equal(r['event'], [b'left', b'right'])

    def test_empty_dataframe_with_text_column(self):
        "epochs_to_array: an empty DataFrame with a text column converts"
        df = pd.DataFrame({'starttime': [], 'event': []}, dtype=object)
        r = epochs_to_array(df)
        assert len(r) == 0

    def test_rejects_unsupported_type(self):
        "epochs_to_array: an unsupported container raises TypeError"
        assert_raises(TypeError, epochs_to_array, [(0., 'left')])


class TestAppendEpochs(TestCase):
    """_append_epochs must keep nepochs, epochs and starttimes consistent for
    every combination of container held and container appended."""

    def _assert_consistent(self, e, expected):
        assert e.nepochs == expected, (e.nepochs, expected)
        assert len(e.epochs) == expected, (len(e.epochs), expected)
        assert len(e.starttimes) == expected, (len(e.starttimes), expected)
        assert len(e.endtimes) == expected

    def test_append_into_structured_array(self):
        "_append_epochs: structured-array epochs accept every container"
        e = Epoched(duration=1.0, epochs=structured_epochs())
        e._append_epochs(structured_epochs())
        self._assert_consistent(e, 4)
        e._append_epochs(dataframe_epochs())
        self._assert_consistent(e, 6)

    def test_append_into_dataframe(self):
        "_append_epochs: DataFrame epochs accept every container"
        # DataFrame.append was removed in pandas 2.0, and before that its
        # return value was discarded here, so this never worked.
        e = Epoched(duration=1.0, epochs=dataframe_epochs())
        e._append_epochs(dataframe_epochs())
        self._assert_consistent(e, 4)
        e._append_epochs(structured_epochs())
        self._assert_consistent(e, 6)
        assert_array_equal(e.starttimes, [0., 1., 0., 1., 0., 1.])

    def test_append_into_pytables_table(self):
        "_append_epochs: pytables Table epochs accept every container"
        with tempdir(dir=None, keep=False, report=False) as dirname:
            with tb.open_file(os.path.join(dirname, 't.h5'), 'w') as h5:
                table = h5.create_table(h5.root, 'ep',
                                        description=structured_epochs())
                e = Epoched(duration=1.0, epochs=table)
                e._append_epochs(structured_epochs())
                self._assert_consistent(e, 4)
                e._append_epochs(dataframe_epochs())
                self._assert_consistent(e, 6)

    def test_self_append_does_not_double_count(self):
        "_append_epochs: appending an object's own epochs counts them once"
        # A pytables Table is extended in place, so reading its starttimes
        # after the append would read back the rows just written.
        with tempdir(dir=None, keep=False, report=False) as dirname:
            with tb.open_file(os.path.join(dirname, 't.h5'), 'w') as h5:
                table = h5.create_table(h5.root, 'ep',
                                        description=structured_epochs())
                e = Epoched(duration=1.0, epochs=table)
                e._append_epochs(e.epochs)
                self._assert_consistent(e, 4)
        for epochs in (structured_epochs(), dataframe_epochs()):
            e = Epoched(duration=1.0, epochs=epochs)
            e._append_epochs(e.epochs)
            self._assert_consistent(e, 4)

    def test_columns_matched_by_name_not_position(self):
        "_append_epochs: columns are matched by name, not by position"
        reversed_ = np.zeros(1, dtype=[('event', '|S5'),
                                       ('starttime', np.float64)])
        reversed_['event'] = [b'down']
        reversed_['starttime'] = [9.]
        with tempdir(dir=None, keep=False, report=False) as dirname:
            with tb.open_file(os.path.join(dirname, 't.h5'), 'w') as h5:
                table = h5.create_table(h5.root, 'ep',
                                        description=structured_epochs())
                e = Epoched(duration=1.0, epochs=table)
                e._append_epochs(reversed_)
                self._assert_consistent(e, 3)
                assert e.epochs[-1]['starttime'] == 9.
                assert e.epochs[-1]['event'] == b'down'

    def test_refuses_to_truncate_text(self):
        "_append_epochs: a too-wide text column raises instead of truncating"
        wide = pd.DataFrame({'starttime': [3.], 'event': ['verylongname']})
        with tempdir(dir=None, keep=False, report=False) as dirname:
            with tb.open_file(os.path.join(dirname, 't.h5'), 'w') as h5:
                table = h5.create_table(h5.root, 'ep',
                                        description=structured_epochs())
                e = Epoched(duration=1.0, epochs=table)
                assert_raises(ValueError, e._append_epochs, wide)

    def test_refuses_missing_column(self):
        "_append_epochs: epochs lacking a required column raise"
        lacking = np.zeros(1, dtype=[('starttime', np.float64)])
        with tempdir(dir=None, keep=False, report=False) as dirname:
            with tb.open_file(os.path.join(dirname, 't.h5'), 'w') as h5:
                table = h5.create_table(h5.root, 'ep',
                                        description=structured_epochs())
                e = Epoched(duration=1.0, epochs=table)
                assert_raises(ValueError, e._append_epochs, lacking)


class TestEpochedSignalWithDataFrame(TestCase):

    def _signal(self, epochs):
        return EpochedUniformTimeSeries(samples=np.zeros((2, 10)), fs=10.,
                                        epochs=epochs)

    def test_append_signal_with_dataframe_epochs(self):
        "EpochedUniformTimeSeries.append: works with DataFrame epochs"
        s = self._signal(dataframe_epochs())
        s.append(self._signal(dataframe_epochs()))
        assert s.nepochs == 4
        assert len(s.epochs) == 4
        assert len(s.starttimes) == 4
        assert s.samples.shape[s.epochaxis] == 4

    def test_saveh5_with_dataframe_epochs(self):
        "saveh5: DataFrame epochs, including text columns, round-trip"
        # create_table needs a structured array; passing the DataFrame raised
        # TypeError inside pytables.
        with tempdir(dir=None, keep=False, report=False) as dirname:
            filename = os.path.join(dirname, 'sig.h5')
            s = self._signal(dataframe_epochs())
            saveh5(s, signalname='/sig', filename=filename)
            r = loadh5(signalname='/sig', filename=filename)
            assert r.nepochs == 2
            assert_array_equal(r.starttimes, [0., 1.])
            assert_array_equal(r.epochs[:]['event'], [b'left', b'right'])

    def test_saveh5_does_not_pollute_metadata(self):
        "saveh5: saving does not write origintime/duration into metadata"
        with tempdir(dir=None, keep=False, report=False) as dirname:
            filename = os.path.join(dirname, 'sig.h5')
            s = EpochedUniformTimeSeries(samples=np.zeros((2, 10)), fs=10.,
                                         epochs=structured_epochs(),
                                         metadata={'subject': 'a'})
            saveh5(s, signalname='/sig', filename=filename)
            assert s.metadata == {'subject': 'a'}
