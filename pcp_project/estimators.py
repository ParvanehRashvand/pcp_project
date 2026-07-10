"""Estimators for EEG signal processing (Fully Compatible with SubjectPipeline)."""

import numpy as np
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import mne
import warnings


class StateSelector(BaseEstimator):
    """Select specific recording states from EEG signal.

    This estimator selects only the timepoints belonging
    to the requested states.

    Parameters
    ----------
    states : list, default=None
        List of state values to keep.
        Values must match what appears in y.
        If None, all timepoints are returned unchanged.

    Attributes
    ----------
    fitted_ : bool
        True after fit() has been called.

    Examples
    --------
    >>> import numpy as np
    >>> selector = StateSelector(states=[1])
    >>> X = np.random.randn(61, 1000)
    >>> y = np.random.randint(0, 2, 1000)
    >>> X_selected = selector.fit_transform(X, y=y)
    >>> X_selected.shape[0]
    61
    """

    def __init__(self, states=None):
        self.states = states

    def fit(self, X, y=None):
        """Validate input and return self.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_channels, n_samples)
            EEG signal.
        y : numpy.ndarray of shape (n_samples,), default=None
            State labels.

        Returns
        -------
        self : StateSelector
        """
        self.fitted_ = True
        return self

    def transform(self, X, y=None, groups=None):
        """Select timepoints from EEG signal.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_channels, n_samples)
            EEG signal.
        y : numpy.ndarray of shape (n_samples,), default=None
            State labels.
            Required if states parameter was set in __init__.

        Returns
        -------
        X_selected : numpy.ndarray of shape (n_channels, n_selected)
            EEG signal with only requested state timepoints.
            n_selected depends on selected states.
        """
        check_is_fitted(self, "fitted_")

        if isinstance(X, tuple):
            if len(X) > 1 and groups is None:
                groups = X[1]
            X = X[0]

        X_copied = X.astype(np.float64)

        if self.states is None or groups is None:
            return X_copied

        # Convert the incoming groups array
        eye_states = np.asarray(groups).squeeze()

        if len(eye_states) != X_copied.shape[1]:
            import __main__

            if hasattr(__main__, "eye_states_mask"):
                eye_states = np.asarray(__main__.eye_states_mask).squeeze()
            else:
                raise ValueError(
                    f"Length mismatch: 'groups' has length {len(eye_states)} "
                    f"but X has {X_copied.shape[1]} samples. Ensure 'eye_states_mask' "
                    "is defined in your main notebook workspace."
                )

        mask = np.isin(eye_states, self.states)

        X_selected = X_copied[:, mask]
        groups_selected = eye_states[mask]

        return X_selected, groups_selected

    def fit_transform(self, X, y=None, groups=None, **fit_params):
        """Fit and transform in one step.

        Overrides TransformerMixin.fit_transform to ensure
        y is passed to transform() for state selection.
        sklearn's default fit_transform does not pass y
        to transform, but StateSelector needs y to select
        the correct timepoints.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_channels, n_samples)
            EEG signal.
        y : numpy.ndarray of shape (n_samples,), default=None
            State labels.

        Returns
        -------
        X_selected : numpy.ndarray
            EEG signal with only requested state timepoints.
        """
        self.fit(X, y)
        return self.transform(X, y, groups=groups)


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

    def fit(self, X, y=None, groups=None):
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
        self.fitted_ = True
        return self

    def transform(self, X, y=None, groups=None):
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
        check_is_fitted(self, "fitted_")

        if isinstance(X, tuple):
            if len(X) > 1 and groups is None:
                groups = X[1]
            X = X[0]

        X = X.astype(np.float64)

        mask = ~np.isnan(X).any(axis=0)
        X_valid = X[:, mask]

        if X_valid.shape[1] == 0:
            return X, groups

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

        filtered_valid = np.stack(
            [signal.sosfiltfilt(f, X_valid, axis=-1) for f in filters]
        )
        summed_valid = filtered_valid.sum(axis=0)

        X_filtered = np.full_like(X, fill_value=np.nan)
        X_filtered[:, mask] = summed_valid

        return X_filtered, groups

    def fit_transform(self, X, y=None, groups=None, **fit_params):
        self.fit(X, y, groups=groups)
        return self.transform(X, y, groups=groups)


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

    def fit(self, X, y=None, groups=None):
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

    def transform(self, X, y=None, groups=None):
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
        check_is_fitted(self, "fitted_")

        if isinstance(X, tuple):
            if len(X) > 1 and groups is None:
                groups = X[1]
            X = X[0]

        X_copied = np.asarray(X, dtype=np.float64)

        mask = ~np.isnan(X_copied).any(axis=0)
        X_valid = X_copied[:, mask]

        if X_valid.shape[1] == 0:
            return X_copied, groups

        freqs_array = (
            np.atleast_1d(self.freqs)
            if not isinstance(self.freqs, list)
            else self.freqs
        )

        X_filtered_valid = mne.filter.notch_filter(
            X_valid,
            Fs=self.sfreq,
            freqs=freqs_array,
            notch_widths=self.notch_widths,
            n_jobs=self.n_jobs,
            method="fir",
            phase="zero",
            verbose=False,
        )

        X_filtered = np.full_like(X_copied, fill_value=np.nan)
        X_filtered[:, mask] = X_filtered_valid

        return X_filtered, groups

    def fit_transform(self, X, y=None, groups=None, **fit_params):
        self.fit(X, y, groups=groups)
        return self.transform(X, y, groups=groups)


class BatchCovariances(BaseEstimator, TransformerMixin):
    """Estimate covariance matrices for a batch of signals.

    Parameters
    ----------
    estimator : {"lwf", "oas"}, default="lwf"
        Covariance estimator to use.
    **kwds
        Additional keyword arguments passed to the covariance estimator.
    """

    def __init__(self, estimator="lwf", assume_centered=False):
        self.covariance_methods = {"lwf": batch_ledoit_wolf, "oas": batch_oas}
        if estimator not in self.covariance_methods.keys():
            raise ValueError(
                f"Invalid method: '{estimator}'. "
                f"Available methods: {list(self.covariance_methods.keys())}"
            )
        self.estimator = self.covariance_methods[estimator]
        self.assume_centered = assume_centered

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X, y=None, groups=None):
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
        check_is_fitted(self, "fitted_")

        if isinstance(X, tuple):
            X_data = X[0]
        else:
            X_data = X

        if X_data.ndim != 3:
            raise ValueError(
                f"X must have shape (n_matrices, n_features, n_samples), got {X.shape}"
            )

        if X_data.shape[2] == 1:
            warnings.warn(
                "Only one sample available. You may want to reshape your data array"
            )

        X_copied = X_data.copy()

        covmats, _ = self.estimator(X_copied, assume_centered=self.assume_centered)
        return covmats


def batch_empirical_covariance(X):
    """Compute the empirical covariance of several matrices.

    Parameters
    ----------
    X : ndarray of shape (n_matrices, n_features, n_samples)
        Data from which to compute the batched covariance estimate.

    Returns
    -------
    covariance : ndarray of shape (n_matrices, n_features, n_features)"""
    return X @ X.transpose(0, 2, 1) / X.shape[2]


def batch_ledoit_wolf_shrinkage(X):
    """Estimate the Ledoit Wolf shrinkage parameter for several matrices

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n_features, n_samples)

    Returns
    -------
    shrinkage : ndarray, shape (n_matrices,)
    """
    n_matrices, n_features, n_samples = X.shape
    X = X.astype(float, copy=True).transpose(0, 2, 1)

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


def batch_ledoit_wolf(X, *, assume_centered):
    """Estimate batch Ledoit-Wolf covariance matrices.

    Parameters
    ----------
    X : ndarray of shape (n_matrices, n_features, n_samples)
        Input data.
    assume_centered : bool, default=False
        If False, center each signal before estimating the covariance.

    Returns
    -------
    covariance : ndarray of shape (n_matrices, n_features, n_features)
        Estimated covariance matrices.
    shrinkage : ndarray of shape (n_matrices,)
        Ledoit-Wolf shrinkage coefficients.
    """

    if not assume_centered:
        X -= np.mean(X, axis=2, keepdims=True)

    n_features = X.shape[1]
    shrinkages = batch_ledoit_wolf_shrinkage(X)
    emp_cov = batch_empirical_covariance(X)
    mu = np.linalg.trace(emp_cov) / n_features

    shrunk_cov = (1.0 - shrinkages)[:, None, None] * emp_cov
    i = np.arange(n_features)
    shrunk_cov[:, i, i] += (shrinkages * mu)[:, None]
    return shrunk_cov, shrinkages


def batch_oas(X, *, assume_centered=False):
    """Estimate batch OAS covariance matrices.

    Parameters
    ----------
    X : ndarray of shape (n_matrices, n_features, n_samples)
        Input data.
    assume_centered : bool, default=False
        If False, center each signal before estimating the covariance.

    Returns
    -------
    covariance : ndarray of shape (n_matrices, n_features, n_features)
        Estimated covariance matrices.
    shrinkage : ndarray of shape (n_matrices,)
        OAS shrinkage coefficients.
    """

    n_matrices, n_features, n_samples = X.shape

    if not assume_centered:
        X -= np.mean(X, axis=2, keepdims=True)

    emp_cov = batch_empirical_covariance(X)

    alpha = np.mean(emp_cov**2, axis=(1, 2))

    mu = np.linalg.trace(emp_cov) / n_features
    mu_squared = mu**2

    num = alpha + mu_squared
    den = (n_samples + 1) * (alpha - mu_squared / n_features)
    shrinkage = np.where(
        den == 0, np.ones(den.shape), np.minimum(num / den, np.ones(den.shape))
    )

    shrunk_cov = (1.0 - shrinkage[:, None, None]) * emp_cov
    i = np.arange(n_features)
    shrunk_cov[:, i, i] += (shrinkage * mu)[:, None]

    return shrunk_cov, shrinkage


class MeanProbabilityAggregator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X, y=None, groups=None):
        check_is_fitted(self, "fitted_")
        X = np.asarray(X)

        if isinstance(X, tuple):
            X = X[0]

        if groups is None:
            raise ValueError("groups must be provided to aggregate per subject")

        groups = np.asarray(groups)
        unique_groups = np.unique(groups)

        aggregated = np.array([X[groups == g].mean() for g in unique_groups])
        return aggregated


class SlidingWindow(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        length=200,
        step_size=50,
        padding_policy="valid",
        label_strategy="majority",
    ):
        self.length = length
        self.step_size = step_size
        self.padding_policy = padding_policy
        self.label_strategy = label_strategy

    def fit(self, X, y=None):
        if self.length <= 0 or self.step_size <= 0:
            raise ValueError("Length and step_size must be positive integers.")
        if self.padding_policy not in ["valid", "zero", "edge"]:
            raise ValueError(f"Unknown padding_policy: {self.padding_policy}")
        if self.label_strategy not in ["majority", "last", "first"]:
            raise ValueError(f"Unknown label_strategy: {self.label_strategy}")

        self.fitted_ = True
        return self

    def transform(self, X, y=None, groups=None):
        check_is_fitted(self, "fitted_")

        # 1. Unpack pipeline structural tuple leakage safely
        if isinstance(X, tuple):
            if len(X) > 1 and groups is None:
                groups = X[1]
            X = X[0]

        n_channels, n_samples = X.shape

        if n_samples < self.length:
            raise ValueError(
                f"Data length ({n_samples}) is shorter than window length ({self.length})."
            )

        remainder = (n_samples - self.length) % self.step_size

        # 2. Manage padding configurations on both signal and eye state markers
        if remainder != 0 and self.padding_policy != "valid":
            pad_size = self.step_size - remainder

            if self.padding_policy == "zero":
                X = np.pad(
                    X, ((0, 0), (0, pad_size)), mode="constant", constant_values=0
                )
                if groups is not None:
                    groups = np.pad(
                        groups,
                        (0, pad_size),
                        mode="constant",
                        constant_values=groups[-1],
                    )

            elif self.padding_policy == "edge":
                X = np.pad(X, ((0, 0), (0, pad_size)), mode="edge")
                if groups is not None:
                    groups = np.pad(groups, (0, pad_size), mode="edge")

            n_samples = X.shape[1]

        start_idx = np.arange(0, n_samples - self.length + 1, self.step_size)
        n_windows = len(start_idx)

        indexer = start_idx[:, None] + np.arange(self.length)
        X_windows = X[:, indexer].transpose(1, 0, 2)

        # 4. Generate window-level summary trackers out of sample-level eye state groups
        groups_windows = None
        if groups is not None:
            groups_arr = np.asarray(groups)
            groups_windows = np.empty(n_windows, dtype=groups_arr.dtype)
            for i, start in enumerate(start_idx):
                w_groups = groups_arr[start : start + self.length]
                if len(w_groups) == 0:
                    continue
                if self.label_strategy == "majority":
                    vals, counts = np.unique(w_groups, return_counts=True)
                    groups_windows[i] = vals[np.argmax(counts)]
                elif self.label_strategy == "last":
                    groups_windows[i] = w_groups[-1]
                elif self.label_strategy == "first":
                    groups_windows[i] = w_groups[0]

        return X_windows, groups_windows

    def fit_transform(self, X, y=None, groups=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y, groups=groups)
