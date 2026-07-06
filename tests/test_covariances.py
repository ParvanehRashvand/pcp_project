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
