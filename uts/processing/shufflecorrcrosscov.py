"""Shuffle-corrected cross-covariance for EpochedUniformTimeSeries.

The estimator removes at every time sample the mean across epochs and
then calculates the cross-covariance of the residual signals.  This
implementation cannot contain accidental self-pairings.

The lag convention is that a positive lag means that activity at s2 follows
activity at s1.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from uts import EpochedUniformTimeSeries, iseuts
from uts.core.check_arg import check_int


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class CrossCovarianceResult:
    """Result returned by :func:`shuffle_corrected_crosscovariance`.

    ``values`` contains normalized coefficients when ``normalized`` is true,
    and otherwise contains the cross-covariances.  ``lag_times`` has the same
    unit as the supplied ``dt``.
    """

    values: FloatArray
    cross_covariance: FloatArray
    variance_a: FloatArray
    variance_b: FloatArray
    lag_samples: IntArray
    lag_times: FloatArray
    n_overlap: IntArray
    normalized: bool

    def peak(self) -> tuple[float, int, float]:
        """Return ``(peak_value, lag_in_samples, lag_time)``.

        Will be (np.nan, None, np.nan) when there were no non-nan
        pairings in calculating the shuffle-corrected crosscovariance.

        """

        if np.all(np.isnan(self.values)):
            return (np.nan, None, np.nan)
        index = int(np.nanargmax(self.values))
        return (float(self.values[index]),
                int(self.lag_samples[index]),
                float(self.lag_times[index]))


def _as_epoch_matrix(x: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(x, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array (epochs x samples).")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return np.ascontiguousarray(array)


def _lagged_views(
        a: FloatArray, b: FloatArray, lag: int
) -> tuple[FloatArray, FloatArray]:
    """Return overlapping views for C_AB(lag) = cov(A(t), B(t + lag))."""

    if lag >= 0:
        stop = a.shape[1] - lag
        return a[:, :stop], b[:, lag:]
    start = -lag
    return a[:, start:], b[:, : a.shape[1] - start]


def shuffle_corrected_crosscovariance(
        s1: EpochedUniformTimeSeries,
        s2: EpochedUniformTimeSeries,
        max_lag: int,
        normalize: bool = True,
        zero_variance: Literal["nan", "raise"] = "nan",
) -> CrossCovarianceResult:
    """Estimate shuffle-corrected AMUA coupling over symmetric lags.

    Parameters
    ----------
    a, b
        Matrices with shape (n_epochs, n_samples).  Corresponding
        rows must be simultaneously recorded epochs.
    max_lag
        Largest absolute lag, in samples.  Returned lags run from
        -max_lag through +max_lag.
    dt
        Sampling period in any desired time unit (for example, milliseconds).
    normalize
        If true, divide the residual cross-covariance by the geometric mean of
        the two residual variances.  The result is then dimensionless and lies
        in [-1, 1].
    zero_variance
        Return NaN or raise an error when either residual variance is zero.

    This function implements the *inner* shuffle correction that removes the
    across-epoch mean response.  It does not correct the positive bias caused
    by selecting the maximum across lags.  That separate permutation analysis
    must repeat the full calculation, including :meth:`CrossCovarianceResult.peak`.

    """
    max_lag = check_int(max_lag, positive=True, zero=True, negative=False, cast=int)
    if not (iseuts(s1) and iseuts(s2)):
        raise ValueError("`s1` and `s2` must both be EpochedUniformTimeSignals.")
    if not (s1.fs == s2.fs):
        raise ValueError("s1 and s2 have different sampling rates")
    if not (s1.ntimesamples == s2.ntimesamples):
        raise ValueError("s1 and s2 have different number of time samples")
    if not (s1.nepochs == s2.nepochs):
        raise ValueError("s1 and s2 have do not have the same number of epochs")
    a_array = s1.transpose(['epoch', 'time']).samples.get(False).astype(np.float64)
    b_array = s2.transpose(['epoch', 'time']).samples.get(False).astype(np.float64)
    origintime = (max_lag * 2 + 1) / (s1.fs * 2.0)

    dt = s1.dt
    n_epochs, n_samples = a_array.shape
    if n_epochs < 2:
        raise ValueError("At least two epochs are required.")
    if not isinstance(max_lag, (int, np.integer)) or max_lag < 0:
        raise ValueError("max_lag must be a non-negative integer.")
    if max_lag >= n_samples:
        raise ValueError("max_lag must be smaller than the number of samples.")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive.")
    if zero_variance not in {"nan", "raise"}:
        raise ValueError("zero_variance must be 'nan' or 'raise'.")

    lag_samples = np.arange(-max_lag, max_lag + 1, dtype=np.int64)
    n_lags = lag_samples.size
    covariance = np.empty(n_lags, dtype=np.float64)
    variance_a = np.empty(n_lags, dtype=np.float64)
    variance_b = np.empty(n_lags, dtype=np.float64)
    values = np.empty(n_lags, dtype=np.float64)
    n_overlap = n_samples - np.abs(lag_samples)

    # Subtract mean over epochs (centering) onec, before taking lagged views.
    # Because the centering is over epochs, slicing these residual matrices is
    # identical to centering each lagged view separately and avoids repeated allocations.
    a_residual_full = a_array - np.mean(a_array, axis=0, keepdims=True)
    b_residual_full = b_array - np.mean(b_array, axis=0, keepdims=True)

    for index, lag_value in enumerate(lag_samples):
        lag = int(lag_value)
        a_view, b_view = _lagged_views(a_array, b_array, lag)
        a_residual, b_residual = _lagged_views(
            a_residual_full, b_residual_full, lag
        )
        var_a = float(np.mean(a_residual * a_residual))
        var_b = float(np.mean(b_residual * b_residual))

        cov = float(np.mean(a_residual * b_residual))
        covariance[index] = cov
        variance_a[index] = var_a
        variance_b[index] = var_b

        if not normalize:
            values[index] = cov
            continue

        denominator = np.sqrt(var_a * var_b)
        if denominator == 0.0:
            if zero_variance == "raise":
                raise ValueError(
                    f"Zero residual variance at lag {lag}; normalized "
                    "coupling is undefined."
                )
            values[index] = np.nan
        else:
            coefficient = cov / denominator
            # The exact coefficient is bounded by Cauchy-Schwarz; clipping
            # removes only tiny floating-point excursions
            coefficient = float(np.clip(coefficient, -1.0, 1.0))
            # Do not clip Monte Carlo derangement results: excursions outside
            # [-1, 1] are useful evidence that n_shuffles is too small.
            values[index] = coefficient

    return CrossCovarianceResult(
        values=values,
        cross_covariance=covariance,
        variance_a=variance_a,
        variance_b=variance_b,
        lag_samples=lag_samples,
        lag_times=lag_samples.astype(np.float64) * dt,
        n_overlap=n_overlap,
        normalized=normalize,
    )

