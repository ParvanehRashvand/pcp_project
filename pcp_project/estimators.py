"""Estimators for EEG signal processing."""

import numpy as np
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import mne

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

        filters = np.array([
            signal.butter(
                N=5,
                Wn=[low, high],
                btype="bandpass",
                fs=self.sfreq,
                output="sos",
            )
            for low, high in self.frequency_bands
        ])

        filtered = np.stack([
            signal.sosfiltfilt(f, X, axis=-1)
            for f in filters
        ])

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

