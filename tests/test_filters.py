"""Tests for BandPassFilter estimator.

These tests check that BandPassFilter works correctly.
Run with: uv run pytest tests/test_filters.py -v
"""

import numpy as np
import pytest
from pcp_project.estimators import BandPassFilter


@pytest.fixture
def eeg_signal():
    """Create fake EEG signal for testing.

    We use fake data because:
    - Real EEG files are too large for testing
    - We only care that the CODE works correctly
    - Shape (61, 1000) matches real EEG structure
      (61 channels, 1000 timepoints)

    Returns
    -------
    numpy.ndarray of shape (61, 1000)
        Fake EEG signal.
    """
    return np.random.randn(61, 1000)


def test_output_shape(eeg_signal):
    """Output shape must be same as input shape.

    BandPassFilter only filters frequencies.
    It must NOT change the shape of the signal.
    Input (61, 1000) must give output (61, 1000).
    """
    filt = BandPassFilter(frequency_bands=[[5, 10]])
    result = filt.fit_transform(eeg_signal)
    assert result.shape == eeg_signal.shape


def test_output_is_float64(eeg_signal):
    """Output must always be float64.

    We always convert to float64 because float32
    can cause numerical errors in covariance matrices.
    This is a project rule from lecture 2.
    """
    filt = BandPassFilter(frequency_bands=[[5, 10]])
    result = filt.fit_transform(eeg_signal)
    assert result.dtype == np.float64


def test_multiple_bands(eeg_signal):
    """Filter must work with multiple frequency bands.

    Example: [[5, 10], [13, 35]] means keep both
    theta waves (5-10 Hz) AND beta waves (13-35 Hz).
    Output shape must still match input shape.
    """
    filt = BandPassFilter(frequency_bands=[[5, 10], [13, 35]])
    result = filt.fit_transform(eeg_signal)
    assert result.shape == eeg_signal.shape


def test_fit_returns_self(eeg_signal):
    """fit() must always return self.

    This is a scikit-learn rule (lecture 3).
    fit() must return the estimator itself
    so we can chain: filt.fit(X).transform(X)
    """
    filt = BandPassFilter(frequency_bands=[[5, 10]])
    result = filt.fit(eeg_signal)
    assert result is filt


def test_fitted_attribute_exists(eeg_signal):
    """fitted_ attribute must exist after fit().

    After fit() is called, the estimator must have
    a fitted_ attribute. This follows the scikit-learn
    convention that learned attributes end with _.
    check_is_fitted() checks for this attribute.
    """
    filt = BandPassFilter(frequency_bands=[[5, 10]])
    filt.fit(eeg_signal)
    assert hasattr(filt, "fitted_")


def test_transform_before_fit_raises_error(eeg_signal):
    """transform() without fit() must raise an error.

    If someone calls transform() before fit(),
    check_is_fitted() raises NotFittedError.
    This protects against incorrect usage.
    """
    filt = BandPassFilter(frequency_bands=[[5, 10]])
    with pytest.raises(Exception):
        filt.transform(eeg_signal)


def test_does_not_modify_input(eeg_signal):
    """transform() must never modify the original signal.
 uv run pytest tests/test_filters.py -v
    We save a copy before filtering.
    After filtering, original must be unchanged.
    This is a scikit-learn rule (lecture 3):
    transform() must never modify X in place.
    Inside transform() we use X.astype(np.float64)
    which creates a new copy automatically.
    """
    original = eeg_signal.copy()
    filt = BandPassFilter(frequency_bands=[[5, 10]])
    filt.fit_transform(eeg_signal)
    np.testing.assert_array_equal(eeg_signal, original)