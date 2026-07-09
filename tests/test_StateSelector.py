import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from pcp_project.estimators import StateSelector


@pytest.fixture
def eeg_signal():
    rng = np.random.default_rng(42)
    return rng.normal(size=(61, 1000))


@pytest.fixture
def states():
    rng = np.random.default_rng(42)
    return rng.integers(0, 2, size=1000)


# shape
def test_state_selector_output_shape(eeg_signal, states):
    selector_all = StateSelector([0, 1])
    res_all = selector_all.fit_transform(eeg_signal, y=states)
    assert res_all.shape == eeg_signal.shape

    selector_zero = StateSelector([0])
    res_zero = selector_zero.fit_transform(eeg_signal, y=states)
    expected_zero_cols = np.sum(states == 0)
    assert res_zero.shape[1] == expected_zero_cols

    selector_none = StateSelector()
    res_none = selector_none.fit_transform(eeg_signal, y=states)
    assert res_none.shape == eeg_signal.shape

    selector_none_y = StateSelector([1])
    res_none_y = selector_none_y.fit_transform(eeg_signal)
    assert res_none_y.shape == eeg_signal.shape

    selector_out = StateSelector([3])
    res_out = selector_out.fit_transform(eeg_signal, y=states)
    assert res_out.shape[1] == 0


# dtype
def test_state_selector_output_is_float64(eeg_signal, states):
    selector = StateSelector([0])
    result = selector.fit_transform(eeg_signal, y=states)
    assert result.dtype == np.float64


# fit returns self
def test_state_selector_fit_returns_self(eeg_signal, states):
    selector = StateSelector([0])
    result = selector.fit(eeg_signal)
    assert result is selector


# fitted attribute exists
def test_state_selector_fitted_attribute_exists(eeg_signal, states):
    selector = StateSelector([0])
    selector.fit(eeg_signal)
    assert hasattr(selector, "fitted_")


# The class does not modify the input in place
def test_state_selector_not_modify_input(eeg_signal, states):
    original = eeg_signal.copy()
    selector = StateSelector([0])
    selector.fit_transform(eeg_signal, y=states)
    np.testing.assert_array_equal(eeg_signal, original)


#  Calling transform() before fit() must raise an exception
def test_state_selector_transform_before_fit_raises_error():
    selector = StateSelector([0])
    with pytest.raises(Exception):
        selector.transform(eeg_signal)


def test_state_selector_correct_values_selected():
    X_simple = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
    y_simple = np.array([0, 1, 0, 1])

    selector = StateSelector(states=[0])
    result = selector.fit_transform(X_simple, y=y_simple)

    expected_output = np.array([[1.0, 3.0], [1.0, 3.0]])
    np.testing.assert_array_equal(result, expected_output)


@pytest.mark.xfail(strict=False, reason="Some sklearn versions drop y in fit_transform")
def test_sklearn_transformer_mixin_bug_directly():
    """
    Does the default fit_transform method pass y to the transform method?
    """

    # Define a mock class
    class BuggyTransformer(BaseEstimator, TransformerMixin):
        def __init__(self):
            self.y_received_in_transform = None

        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None):
            # Store whatever value is received as y
            self.y_received_in_transform = y
            return X

    # Create dummy data
    X = np.zeros((5, 5))
    y = np.array([1, 2, 3, 4, 5])

    transformer = BuggyTransformer()

    # Call scikit-learn's default fit_transform method
    transformer.fit_transform(X, y=y)

    # Check whether y reached transform or became None
    # This assert proves that in older scikit-learn versions, y_received_in_transform will be None!
    assert transformer.y_received_in_transform is not None
