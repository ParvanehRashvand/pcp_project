"""Tests for BatchCovariance estimator.

These tests check that BatchCovariance works correctly.
Run with: uv run pytest tests/test_covariances.py -v
"""

import numpy as np
import pytest
from pyriemann.estimation import Covariances
from pcp_project.estimators import BatchCovariances, batch_ledoit_wolf


@pytest.mark.parametrize(
    "shape",
    [
        (5, 8, 10),
        (10, 15, 20),
        (20, 30, 15),
        (50, 10, 8),
    ],
)
def test_batch_ledoit_wolf_matches_pyriemann(shape):
    """Batch implementation should match pyRiemann's implementation."""

    rng = np.random.default_rng(42)

    X = rng.standard_normal(shape)

    ours = BatchCovariances(
        estimator=batch_ledoit_wolf,
        assume_centered=True,
        block_size=1000,
    ).fit_transform(X)

    theirs = Covariances(
        estimator="lwf",
        assume_centered=True,
        block_size=1000,
    ).fit_transform(X)

    assert np.allclose(ours, theirs, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("seed", range(10))
def test_batch_ledoit_wolf_matches_pyriemann_random(seed):
    """Multiple random realizations should agree."""

    rng = np.random.default_rng(seed)

    X = rng.standard_normal((10, 15, 20))

    ours = BatchCovariances(
        estimator=batch_ledoit_wolf,
        assume_centered=True,
        block_size=1000,
    ).fit_transform(X)

    theirs = Covariances(
        estimator="lwf",
        assume_centered=True,
        block_size=1000,
    ).fit_transform(X)

    assert np.allclose(ours, theirs, rtol=1e-6, atol=1e-8)


def test_batch_ledoit_wolf_matches_pyriemann_float32():
    """Check float32 inputs."""

    rng = np.random.default_rng(1234)

    X = rng.standard_normal((25, 12, 18)).astype(np.float32)

    ours = BatchCovariances(
        estimator=batch_ledoit_wolf,
        assume_centered=True,
        block_size=1000,
    ).fit_transform(X)

    theirs = Covariances(
        estimator="lwf",
        assume_centered=True,
        block_size=1000,
    ).fit_transform(X)

    assert np.allclose(ours, theirs, rtol=1e-5, atol=1e-6)


def test_batch_ledoit_wolf_preserves_shape():
    rng = np.random.default_rng(0)

    X = rng.standard_normal((17, 9, 11))

    ours = BatchCovariances(
        estimator=batch_ledoit_wolf,
        assume_centered=True,
        block_size=1000,
    ).fit_transform(X)

    assert ours.shape == (17, 9, 9)


def test_batch_ledoit_wolf_matches_pyriemann_not_centered():
    """Match pyRiemann when centering is requested."""

    rng = np.random.default_rng(42)
    X = rng.standard_normal((10, 15, 20))

    ours = BatchCovariances(
        estimator=batch_ledoit_wolf, assume_centered=False, block_size=1000
    ).fit_transform(X)

    theirs = Covariances(
        estimator="lwf", assume_centered=False, block_size=1000
    ).fit_transform(X)

    assert np.allclose(ours, theirs, rtol=1e-6, atol=1e-8)


def test_batch_ledoit_wolf_single_feature_returns_matrix():
    """Single-feature input should still return a (1, 1) covariance."""

    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 1, 20))

    cov = BatchCovariances(
        estimator=batch_ledoit_wolf, assume_centered=True, block_size=1000
    ).fit_transform(X)

    assert cov.shape == (5, 1, 1)


def test_batch_ledoit_wolf_invalid_dimension():
    """Invalid input dimensions should raise a ValueError."""

    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 20))

    estimator = BatchCovariances(
        estimator=batch_ledoit_wolf,
        assume_centered=True,
        block_size=1000,
    )

    with pytest.raises(ValueError, match="shape"):
        estimator.fit_transform(X)


def test_batch_ledoit_wolf_warns_on_single_sample():
    """A warning should be emitted when only one sample is provided."""

    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 10, 1))

    estimator = BatchCovariances(
        estimator=batch_ledoit_wolf, assume_centered=True, block_size=1000
    )

    with pytest.warns(UserWarning, match="Only one sample"):
        estimator.fit_transform(X)
