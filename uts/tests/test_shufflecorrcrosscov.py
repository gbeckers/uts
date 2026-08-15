import itertools
from unittest import TestCase

import numpy as np
import pandas as pd

from uts.processing.shufflecorrcrosscov import shuffle_corrected_crosscovariance
from uts.core.uniformtime import EpochedUniformTimeSeries

def _stte(starttimes):
    """Helper fucntion that converts a sequence of starttimes into a Pandas DataFrame"""
    return pd.DataFrame(starttimes, columns=['starttime'])


class CrossCovarianceTests(TestCase):
    def test_exact_matches_average_of_all_unrestricted_shuffles(self):
        a = np.array([[1.0, 4.0, 2.0], [3.0, 1.0, 5.0], [6.0, 2.0, 0.0]]) # (n_epochs, n_samples)
        b = np.array([[2.0, 8.0, 1.0], [7.0, 3.0, 4.0], [5.0, 0.0, 9.0]])
        s1 = EpochedUniformTimeSeries(a, fs=1., epochs=_stte(np.linspace(0,1,len(a))))
        s2 = EpochedUniformTimeSeries(b, fs=1., epochs=_stte(np.linspace(0,1,len(b))))
        result = shuffle_corrected_crosscovariance(
            s1, s2, max_lag=1, normalize=False
        )
        expected = []
        for lag in result.lag_samples:
            if lag >= 0:
                av, bv = a[:, : a.shape[1] - lag], b[:, lag:]
            else:
                av, bv = a[:, -lag:], b[:, : b.shape[1] + lag]
            observed = np.mean(av * bv)
            predictors = [
                np.mean(av * bv[list(p)])
                for p in itertools.permutations(range(3))
            ]
            expected.append(observed - np.mean(predictors))
        np.testing.assert_allclose(result.cross_covariance, expected)

    def test_positive_lag_means_b_follows_a(self):
        rng = np.random.default_rng(4)
        a = rng.normal(size=(100, 20))
        b = rng.normal(size=(100, 20))
        delay = 3
        b[:, delay:] = a[:, :-delay]
        s1 = EpochedUniformTimeSeries(a, fs=2., epochs=_stte(np.linspace(0, 1, len(a))))
        s2 = EpochedUniformTimeSeries(b, fs=2., epochs=_stte(np.linspace(0, 1, len(b))))
        result = shuffle_corrected_crosscovariance(s1, s2, max_lag=5)
        peak_value, peak_lag, peak_time = result.peak()

        self.assertAlmostEqual(peak_value, 1.0)
        self.assertEqual(peak_lag, delay)
        self.assertAlmostEqual(peak_time, delay * 0.5)

    def test_fixed_mean_response_is_removed(self):
        rng = np.random.default_rng(12)
        a = rng.normal(size=(50, 15))
        b = 0.4 * a + rng.normal(size=(50, 15))
        mean_a = np.linspace(-20.0, 30.0, a.shape[1])
        mean_b = np.sin(np.linspace(0.0, 4.0, b.shape[1])) * 100.0
        s1 = EpochedUniformTimeSeries(a, fs=1., epochs=_stte(np.linspace(0, 1, len(a))))
        s2 = EpochedUniformTimeSeries(b, fs=1., epochs=_stte(np.linspace(0, 1, len(b))))
        s1_mean = EpochedUniformTimeSeries(a+mean_a[None,:], fs=1., epochs=_stte(np.linspace(0, 1, len(a))))
        s2_mean = EpochedUniformTimeSeries(b+mean_b[None,:], fs=1., epochs=_stte(np.linspace(0, 1, len(b))))

        before = shuffle_corrected_crosscovariance(s1, s2, max_lag=2).values
        after = shuffle_corrected_crosscovariance(
            s1_mean, s2_mean, max_lag=2
        ).values

        np.testing.assert_allclose(before, after, atol=1e-14)

    def test_overlap_and_lag_arrays_are_explicit(self):
        a = np.arange(24.0).reshape(4, 6)
        s1 = EpochedUniformTimeSeries(a, fs=4., epochs=_stte(np.linspace(0, 1, len(a))))
        result = shuffle_corrected_crosscovariance(s1, s1, max_lag=2)

        np.testing.assert_array_equal(result.lag_samples, [-2, -1, 0, 1, 2])
        np.testing.assert_allclose(
            result.lag_times, [-0.5, -0.25, 0.0, 0.25, 0.5]
        )
        np.testing.assert_array_equal(result.n_overlap, [4, 5, 6, 5, 4])

    def test_zero_variance_can_return_nan_or_raise(self):
        a = np.ones((4, 10))
        b = np.arange(40.0).reshape(4, 10)
        s1 = EpochedUniformTimeSeries(a, fs=1., epochs=_stte(np.linspace(0, 1, len(a))))
        s2 = EpochedUniformTimeSeries(b, fs=1., epochs=_stte(np.linspace(0, 1, len(b))))
        result = shuffle_corrected_crosscovariance(s1, s2, max_lag=0)
        self.assertTrue(np.isnan(result.values[0]))

        with self.assertRaisesRegex(ValueError, "Zero residual variance"):
            shuffle_corrected_crosscovariance(
                s1, s2, max_lag=0, zero_variance="raise"
            )

    def test_rejects_bad_inputs(self):
        with self.assertRaisesRegex(ValueError, "same number of epochs"):
            s1 = EpochedUniformTimeSeries(np.zeros((3, 5)), fs=1., epochs=_stte(np.linspace(0, 1, 3)))
            s2 = EpochedUniformTimeSeries(np.zeros((4, 5)), fs=1., epochs=_stte(np.linspace(0, 1, 4)))
            shuffle_corrected_crosscovariance(
                s1, s2, max_lag=1
            )
        with self.assertRaisesRegex(ValueError, "smaller than"):
            s1 = EpochedUniformTimeSeries(np.zeros((3, 5)), fs=1., epochs=_stte(np.linspace(0, 1, 3)))
            s2 = EpochedUniformTimeSeries(np.zeros((3, 5)), fs=1., epochs=_stte(np.linspace(0, 1, 3)))
            shuffle_corrected_crosscovariance(
                s1, s2, max_lag=5
            )


if __name__ == "__main__":
    unittest.main()
