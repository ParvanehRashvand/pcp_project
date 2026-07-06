"""Tests for MeanProbabilityAggregator.

Run with:
uv run pytest tests/test_mean_probability_aggregator.py -v
"""

import numpy as np
import pytest

from pcp_project.estimators import MeanProbabilityAggregator


@pytest.fixture
def sample_data():
    """Create simple probability data with known grouping."""
    # 6 windows total
    X = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.7])

    # 2 subjects:
    # subject A -> first 3 windows
    # subject B -> last 3 windows
    groups = np.array(["A", "A", "A", "B", "B", "B"])

    return X, groups


def test_fit_returns_self(sample_data):
    """.fit() should return self

     We do not want the fit function to change anything in the data
     """
    X, _ = sample_data
    agg = MeanProbabilityAggregator()

    result = agg.fit(X)

    assert result is agg


def test_fitted_attribute_exists(sample_data):
    """fitted attribute must exist

     We want to make sure, that our class
     has the fitted_ attribute.
     """
    X, _ = sample_data
    agg = MeanProbabilityAggregator()

    agg.fit(X)

    assert hasattr(agg, "fitted_")


def test_transform_requires_groups(sample_data):
    """Transform must require groups

     Transforming should not work, if we
     do not provide any groups.
     """
    X, _ = sample_data
    agg = MeanProbabilityAggregator().fit(X)

    with pytest.raises(ValueError):
        agg.transform(X)


def test_output_shape_is_number_of_groups(sample_data):
    def test_transform_requires_groups(sample_data):
        """Output shape must correspond to number of groups"""
        X, _ = sample_data
        agg = MeanProbabilityAggregator().fit(X)

        with pytest.raises(ValueError):
            agg.transform(X)
    X, groups = sample_data
    agg = MeanProbabilityAggregator().fit(X)

    result = agg.transform(X, groups=groups)

    # 2 unique subjects → 2 outputs
    assert result.shape == (2,)


def test_correct_group_means(sample_data):
    """Means must be correctly calculated

    We test, whether the mean of the previously
    created sample groups is calculated correctly.
    """
    X, groups = sample_data
    agg = MeanProbabilityAggregator().fit(X)

    result = agg.transform(X, groups=groups)

    expected = np.array([
        np.mean([0.1, 0.2, 0.3]),  # A
        np.mean([0.9, 0.8, 0.7])   # B
    ])

    np.testing.assert_allclose(result, expected)


def test_group_order_is_sorted_unique(sample_data):
    """np.unique() sorts output groups, so order is deterministic."""
    X, groups = sample_data
    agg = MeanProbabilityAggregator().fit(X)

    result = agg.transform(X, groups=groups)

    # unique groups = ['A', 'B']
    # so order must match sorted order
    expected_first = np.mean([0.1, 0.2, 0.3])
    expected_second = np.mean([0.9, 0.8, 0.7])

    assert result[0] == expected_first
    assert result[1] == expected_second


def test_single_group_case():
    """Edge case: all samples belong to one subject.

    Mean should be aggregated for just one subject.
    """
    X = np.array([0.2, 0.4, 0.6])
    groups = np.array(["A", "A", "A"])

    agg = MeanProbabilityAggregator().fit(X)

    result = agg.transform(X, groups=groups)

    assert result.shape == (1,)
    assert np.isclose(result[0], np.mean(X))


def test_groups_length_mismatch():
    """Number of labels must match number of samples.    """
    X = np.array([0.1, 0.2, 0.3])
    groups = np.array(["A", "B"])  # <- wrong length

    agg = MeanProbabilityAggregator().fit(X)

    # Current implementation will crash implicitly
    with pytest.raises(Exception):
        agg.transform(X, groups=groups)