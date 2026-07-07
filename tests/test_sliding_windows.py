import pytest
import numpy as np
from pcp_project.estimators import SlidingWindow  # Adjust import based on your exact structure

def test_sliding_window_standard():
    # 2 channels, 500 samples
    X = np.random.randn(2, 500)
    y = np.array([0] * 250 + [1] * 250)
    groups = np.array([7] * 500)
    
    sw = SlidingWindow(length=100, step_size=50, label_strategy="majority")
    sw.fit(X)
    X_w, y_w, g_w = sw.transform(X, y=y, groups=groups)
    
    # Expected windows: (500 - 100) / 50 + 1 = 9
    assert X_w.shape == (9, 2, 100)
    assert len(y_w) == 9
    assert np.all(g_w == 7)

def test_sliding_window_strategies():
    X = np.random.randn(1, 10)
    # Alternating labels to easily test "first" vs "last"
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    
    # "first" strategy
    sw_first = SlidingWindow(length=5, step_size=5, label_strategy="first").fit(X)
    _, y_first = sw_first.transform(X, y=y)
    assert y_first[0] == 0  # First element of first window is 0
    
    # "last" strategy
    sw_last = SlidingWindow(length=5, step_size=5, label_strategy="last").fit(X)
    _, y_last = sw_last.transform(X, y=y)
    assert y_last[0] == 1  # Last element of first window is 1

def test_sliding_window_errors():
    X = np.random.randn(2, 50)
    
    # Test window length larger than data length
    sw = SlidingWindow(length=100, step_size=10).fit(X)
    with pytest.raises(ValueError, match="is shorter than window length"):
        sw.transform(X)
        
    # Test parameter validations
    with pytest.raises(ValueError, match="Window length must be greater than 0"):
        SlidingWindow(length=0).fit(X)
    with pytest.raises(ValueError, match="Unknown label_strategy"):
        SlidingWindow(label_strategy="invalid").fit(X)