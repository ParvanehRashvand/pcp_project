"""Estimators for EEG signal processing."""

import numpy as np
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import mne
import pyriemann
import warnings


class BandPassFilter(BaseEstimator, TransformerMixin):
    """Filter EEG signals to keep only specific frequency bands.

    Applies one or more bandpass filters and sums their outputs.

    Parameters
    ----------
    frequency_bands : list of [float, float]
        List of [low, high] frequency pairs in Hz.
        Example: [[5, 10], [13, 35]]
    sfreq : float, default=256.0
        Sampling frequency of the EEG signal in Hz.

    Attributes
    ----------
    fitted_ : bool
        True after fit() has been called.

    Examples
    --------
    >>> import numpy as np
    >>> filt = BandPassFilter(frequency_bands=[[5, 10]])
    >>> X = np.random.randn(61, 1000)
    >>> X_filtered = filt.fit_transform(X)
    >>> X_filtered.shape
    (61, 1000)
    """

    def __init__(self, frequency_bands, sfreq=256.0):
        self.frequency_bands = frequency_bands
        self.sfreq = sfreq

    def fit(self, X, y=None):
        """Validate input and return self.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_channels, n_samples)
            EEG signal.
        y : ignored

        Returns
        -------
        self : BandPassFilter
        """
        self.fitted_ = True
        return self

    def transform(self, X):
        """Apply bandpass filter to EEG signal.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_channels, n_samples)
            Raw EEG signal.

        Returns
        -------
        X_filtered : numpy.ndarray of shape (n_channels, n_samples)
            Filtered EEG signal in float64.
        """
        check_is_fitted(self)
        X = X.astype(np.float64)

        filters = np.array(
            [
                signal.butter(
                    N=5,
                    Wn=[low, high],
                    btype="bandpass",
                    fs=self.sfreq,
                    output="sos",
                )
                for low, high in self.frequency_bands
            ]
        )

        filtered = np.stack([signal.sosfiltfilt(f, X, axis=-1) for f in filters])

        return filtered.sum(axis=0)


class NotchFilter(BaseEstimator, TransformerMixin):
    """Filter EEG signals to remove power line noise (50/60 Hz) using MNE.

    Applies a notch filter to the signal to attenuate specific frequencies
    while leaving other frequencies intact.

    Parameters
    ----------
    freqs : float or list of float, default=50.0
        Frequencies to notch filter. Can be a single frequency (e.g., 50.0)
        or a list of frequencies (e.g., [50.0, 100.0]) to remove harmonics.
    sfreq : float, default=256.0
        Sampling frequency of the EEG signal in Hz.
    notch_widths : float or array-like, default=None
        Width of the notch at each frequency. If None, MNE uses freqs / 200.
    n_jobs : int or str, default=None
        Number of jobs to run in parallel. Useful for fast vectorized computation.

    Attributes
    ----------
    fitted_ : bool
        True after fit() has been called.

    Examples
    --------
    >>> import numpy as np
    >>> filt = NotchFilter(freqs=50.0)
    >>> Y = np.random.randn(61, 1000)
    >>> Y_filtered = filt.fit_transform(Y)
    >>> Y_filtered.shape
    (61, 1000)
    """

    def __init__(self, freqs=50.0, sfreq=256.0, notch_widths=None, n_jobs=None):
        self.freqs = freqs
        self.sfreq = sfreq
        self.notch_widths = notch_widths
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Validate input and return self.

        Parameters
        ----------
        X : numpy.ndarray of shape (..., n_times)
            EEG signal array.
        y : ignored

        Returns
        -------
        self : NotchFilter
        """
        self.fitted_ = True
        return self

    def transform(self, X):
        """Apply MNE notch filter to the EEG signal.

        Parameters
        ----------
        X : numpy.ndarray of shape (..., n_times)
            Raw EEG signal array.

        Returns
        -------
        X_filtered : numpy.ndarray of shape (..., n_times)
            Filtered EEG signal in float64.
        """
        check_is_fitted(self)

        X_copied = np.asarray(X, dtype=np.float64)

        freqs_array = (
            np.atleast_1d(self.freqs)
            if not isinstance(self.freqs, list)
            else self.freqs
        )

        X_filtered = mne.filter.notch_filter(
            X_copied,
            Fs=self.sfreq,
            freqs=freqs_array,
            notch_widths=self.notch_widths,
            n_jobs=self.n_jobs,
            method="fir",
            phase="zero",
            verbose=False,
        )

        return X_filtered


# TODO: vectorize Oracle Approximation Shrinkage and add it to estimator
class BatchCovariances(pyriemann.estimation.Covariances):
    def __init__(self, estimator="scm", **kwds):
        super().__init__(estimator=estimator, **kwds)

    def transform(self, X):
        """Estimate covariance matrices.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_features, n_samples)
            Multi-channel time-series.

        Returns
        -------
        X_new : ndarray, shape (n_matrices, n_features, n_features)
            Covariance matrices.
        """
        covmats, _ = batch_ledoit_wolf(X, **self.kwds)
        return covmats


def batch_empirical_covariance(X):
    """Compute the empirical covariance of several matrices.

    Parameters
    ----------
    X : ndarray of shape (n_matrices, n_features, n_samples)
        Data from which to compute the batched covariance estimate.

    Returns
    -------
    covariance : ndarray of shape (n_features, n_features)"""
    covariance = X @ X.transpose(0, 2, 1) / X.shape[2]
    return covariance


def batch_ledoit_wolf_shrinkage(X, block_size=1000):
    """Estimate the Ledoit Wolf shrinkage parameter for several matrices

    X : ndarray, shape (n_matrices, n_features, n_samples)

    Returns
    -------
    shrinkage : ndarray, shape (n_matrices,)
    """
    n_matrices, n_features, n_samples = X.shape

    X = X.astype(float, copy=True)
    X = X.transpose(0, 2, 1)

    X2 = X**2
    emp_cov_trace = X2.sum(axis=1) / n_samples
    mu = emp_cov_trace.sum(axis=1) / n_features

    Xt = X.transpose(0, 2, 1)
    XtX2 = X2.transpose(0, 2, 1) @ X2
    beta_ = XtX2.sum(axis=(1, 2))

    XtX = Xt @ X
    delta_ = (XtX**2).sum(axis=(1, 2)) / n_samples**2

    beta = (beta_ / n_samples - delta_) / (n_features * n_samples)
    delta = (
        delta_ - 2.0 * mu * emp_cov_trace.sum(axis=1) + n_features * mu**2
    ) / n_features
    beta = np.minimum(beta, delta)
    shrinkage = np.where(beta == 0, 0.0, beta / delta)
    return shrinkage


def batch_ledoit_wolf(X, *, assume_centered, block_size):
    """Estimate the shrunk Ledoit-Wolf covariance matrix."""
    if X.ndim != 3:
        raise ValueError(
            f"X must have shape (n_matrices, n_features, n_samples), got {X.shape}"
        )

    if X.shape[2] == 1:
        warnings.warn(
            "Only one sample available. You may want to reshape your data array"
        )

    if not assume_centered:
        X -= np.mean(X, axis=2, keepdims=True)

    n_features = X.shape[1]

    # get Ledoit-Wolf shrinkage
    shrinkages = batch_ledoit_wolf_shrinkage(X, block_size=block_size)

    emp_cov = batch_empirical_covariance(X)
    mu = np.linalg.trace(emp_cov) / n_features

    shrunk_cov = (1.0 - shrinkages)[:, None, None] * emp_cov
    i = np.arange(n_features)
    shrunk_cov[:, i, i] += (shrinkages * mu)[:, None]
    return shrunk_cov, shrinkages

class MeanProbabilityAggregator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # nothing to learn
        self.fitted_ = True
        return self

    def transform(self, X, groups=None):
        """
        Parameters
        ----------
        X : array-like, shape (n_windows,)
            Probabilities or predictions per window.

        groups : array-like, shape (n_windows,)
            Subject ID for each window.
        """
        check_is_fitted(self)

        X = np.asarray(X)

        if groups is None:
            raise ValueError("groups must be provided to aggregate per subject")

        groups = np.asarray(groups)

        unique_groups = np.unique(groups)

        aggregated = np.array([
            X[groups == g].mean()
            for g in unique_groups
        ])

        return aggregated