"""Estimators for EEG signal processing."""

import numpy as np
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


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