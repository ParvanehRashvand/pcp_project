"""Estimators for EEG signal processing (Fully Compatible with SubjectPipeline)."""

import numpy as np
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import mne
import pyriemann
import warnings


class StateSelector(BaseEstimator):

    def __init__(self, states=None):
        self.states = states

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X, y=None, groups=None):
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
            if hasattr(__main__, 'eye_states_mask'):
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
        self.fit(X, y)
        return self.transform(X, y, groups=groups)


class BandPassFilter(BaseEstimator, TransformerMixin):

    def __init__(self, frequency_bands, sfreq=256.0):
        self.frequency_bands = frequency_bands
        self.sfreq = sfreq

    def fit(self, X, y=None, groups=None):
        self.fitted_ = True
        return self

    def transform(self, X, y=None, groups=None):
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

        filtered_valid = np.stack([signal.sosfiltfilt(f, X_valid, axis=-1) for f in filters])
        summed_valid = filtered_valid.sum(axis=0)

        X_filtered = np.full_like(X, fill_value=np.nan)
        X_filtered[:, mask] = summed_valid

        return X_filtered, groups

    def fit_transform(self, X, y=None, groups=None, **fit_params):
        self.fit(X, y, groups=groups)
        return self.transform(X, y, groups=groups)


class NotchFilter(BaseEstimator, TransformerMixin):

    def __init__(self, freqs=50.0, sfreq=256.0, notch_widths=None, n_jobs=None):
        self.freqs = freqs
        self.sfreq = sfreq
        self.notch_widths = notch_widths
        self.n_jobs = n_jobs

    def fit(self, X, y=None, groups=None):
        self.fitted_ = True
        return self

    def transform(self, X, y=None, groups=None):
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


class BatchCovariances(pyriemann.estimation.Covariances):
    def __init__(self, estimator="scm", assume_centered=False, block_size=1000):
        super().__init__(estimator=estimator)
        self.assume_centered = assume_centered
        self.block_size = block_size

        if hasattr(self, "_set_output"):
            self._set_output(transform="bypass")

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X, y=None, groups=None):
        check_is_fitted(self, "fitted_")

        if isinstance(X, tuple):
            X_data = X[0]
        else:
            X_data = X

        X_copied = X_data.copy()

        if np.isnan(X_copied).any():
            X_copied = np.nan_to_num(X_copied, nan=0.0)

        covmats, _ = batch_ledoit_wolf(
            X_copied,
            assume_centered=self.assume_centered,
            block_size=self.block_size
        )
        return covmats


def batch_empirical_covariance(X):
    return X @ X.transpose(0, 2, 1) / X.shape[2]


def batch_ledoit_wolf_shrinkage(X, block_size=1000):
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
    delta = (delta_ - 2.0 * mu * emp_cov_trace.sum(axis=1) + n_features * mu**2) / n_features
    beta = np.minimum(beta, delta)
    shrinkage = np.where(beta == 0, 0.0, beta / delta)
    return shrinkage


def batch_ledoit_wolf(X, *, assume_centered, block_size):
    if X.ndim != 3:
        raise ValueError(f"X must have shape (n_matrices, n_features, n_samples), got {X.shape}")

    if X.shape[2] == 1:
        warnings.warn("Only one sample available.")

    if not assume_centered:
        X -= np.mean(X, axis=2, keepdims=True)

    n_features = X.shape[1]
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

        aggregated = np.array([
            X[groups == g].mean()
            for g in unique_groups
        ])
        return aggregated


class SlidingWindow(BaseEstimator, TransformerMixin):

    def __init__(self, length=200, step_size=50, padding_policy="valid", label_strategy="majority"):
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
            raise ValueError(f"Data length ({n_samples}) is shorter than window length ({self.length}).")

        remainder = (n_samples - self.length) % self.step_size

        # 2. Manage padding configurations on both signal and eye state markers
        if remainder != 0 and self.padding_policy != "valid":
            pad_size = self.step_size - remainder

            if self.padding_policy == "zero":
                X = np.pad(X, ((0, 0), (0, pad_size)), mode='constant', constant_values=0)
                if groups is not None:
                    groups = np.pad(groups, (0, pad_size), mode='constant', constant_values=groups[-1])

            elif self.padding_policy == "edge":
                X = np.pad(X, ((0, 0), (0, pad_size)), mode='edge')
                if groups is not None:
                    groups = np.pad(groups, (0, pad_size), mode='edge')

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