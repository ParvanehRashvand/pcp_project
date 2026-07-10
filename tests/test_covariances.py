"""Tests for BatchCovariances estimator.

These tests check that BatchCovariances works correctly for every
supported estimator ("lwf" and "oas"), by comparing it against
pyRiemann's own Covariances class.

Run with: uv run pytest tests/test_covariances.py -v
"""

import numpy as np
import pytest

from pyriemann.estimation import Covariances
from pcp_project.estimators import BatchCovariances

ESTIMATORS = ["lwf", "oas"]


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize(
    "shape",
    [
        (5, 8, 10),
        (10, 15, 20),
        (20, 30, 15),
        (50, 10, 8),
    ],
)
def test_batch_covariances_matches_pyriemann(estimator, shape):
    """Batch implementation should match pyRiemann's implementation."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal(shape)
    ours = BatchCovariances(estimator=estimator, assume_centered=True).fit_transform(X)
    theirs = Covariances(estimator=estimator, assume_centered=True).fit_transform(X)
    assert np.allclose(ours, theirs, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("seed", range(10))
def test_batch_covariances_matches_pyriemann_random(estimator, seed):
    """Multiple random realizations should agree."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((10, 15, 20))
    ours = BatchCovariances(estimator=estimator, assume_centered=True).fit_transform(X)
    theirs = Covariances(estimator=estimator, assume_centered=True).fit_transform(X)
    assert np.allclose(ours, theirs, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_batch_covariances_matches_pyriemann_float32(estimator):
    """Check float32 inputs."""
    rng = np.random.default_rng(1234)
    X = rng.standard_normal((25, 12, 18)).astype(np.float32)
    ours = BatchCovariances(estimator=estimator, assume_centered=True).fit_transform(X)
    theirs = Covariances(estimator=estimator, assume_centered=True).fit_transform(X)
    assert np.allclose(ours, theirs, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_batch_covariances_matches_pyriemann_not_centered(estimator):
    """Match pyRiemann when the data is not centered."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((10, 15, 20))
    ours = BatchCovariances(estimator=estimator, assume_centered=False).fit_transform(X)
    theirs = Covariances(estimator=estimator, assume_centered=False).fit_transform(X)
    assert np.allclose(ours, theirs, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_batch_covariances_preserves_shape(estimator):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((17, 9, 11))
    ours = BatchCovariances(estimator=estimator, assume_centered=True).fit_transform(X)
    assert ours.shape == (17, 9, 9)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_batch_covariances_single_feature_returns_matrix(estimator):
    """Single-feature input should still return a (1, 1) covariance."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 1, 20))
    cov = BatchCovariances(estimator=estimator, assume_centered=True).fit_transform(X)
    assert cov.shape == (5, 1, 1)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_batch_covariances_invalid_dimension(estimator):
    """Invalid input dimensions should raise a ValueError."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 20))
    batch_cov = BatchCovariances(estimator=estimator, assume_centered=True)
    with pytest.raises(ValueError, match="shape"):
        batch_cov.fit_transform(X)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_batch_covariances_warns_on_single_sample(estimator):
    """A warning should be emitted when only one sample is provided."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 10, 1))
    batch_cov = BatchCovariances(estimator=estimator, assume_centered=True)
    with pytest.warns(UserWarning, match="Only one sample"):
        batch_cov.fit_transform(X)


def test_batch_covariances_invalid_estimator():
    """An unknown estimator name should raise a ValueError."""
    with pytest.raises(ValueError, match="Invalid method"):
        BatchCovariances(estimator="not_a_real_estimator")


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_batch_covariances_accepts_tuple_input(estimator):
    """Tuple inputs should use the first element as the data."""
    rng = np.random.default_rng(42)

    X = rng.standard_normal((10, 15, 20))
    y = np.arange(10)

    ours_tuple = BatchCovariances(
        estimator=estimator,
        assume_centered=True,
    ).fit_transform((X, y))

    ours_array = BatchCovariances(
        estimator=estimator,
        assume_centered=True,
    ).fit_transform(X)

    assert np.allclose(ours_tuple, ours_array)
