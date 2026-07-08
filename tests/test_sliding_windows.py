import pytest
import numpy as np
from pcp_project.estimators import SlidingWindow 

def test_sliding_window_standard():
    # 2 channels, 500 samples
    X = np.random.randn(2, 500)
    y = np.array([0] * 250 + [1] * 250)
    groups = np.array([7] * 500)
    
    sw = SlidingWindow(length=100, step_size=50, label_strategy="majority")
    sw.fit(X)
    X_w, y_w, g_w = sw.transform(X, y=y, groups=groups)
    
    assert X_w.shape == (9, 2, 100)
    assert len(y_w) == 9
    assert np.all(g_w == 7)

def test_sliding_window_padding_valid():
    X = np.ones((1, 13))
    sw = SlidingWindow(length=10, step_size=2, padding_policy="valid").fit(X)
    X_w = sw.transform(X)
    assert X_w.shape[0] == 2  # Exactly 2 windows

def test_sliding_window_padding_zero():
    X = np.ones((1, 13))
    y = np.ones(13)
    groups = np.ones(13) * 5
    
    sw = SlidingWindow(length=10, step_size=2, padding_policy="zero").fit(X)
    X_w, y_w, g_w = sw.transform(X, y=y, groups=groups)
    
    assert X_w.shape[0] == 3
    assert np.any(X_w[-1] == 0) 
    assert len(y_w) == 3
    assert len(g_w) == 3

def test_sliding_window_padding_edge():
    X = np.ones((1, 13)) * 42.0
    sw = SlidingWindow(length=10, step_size=2, padding_policy="edge").fit(X)
    X_w = sw.transform(X)
    
    assert X_w.shape[0] == 3
    assert np.all(X_w[-1] == 42.0) 

def test_sliding_window_no_y_no_groups():
    X = np.random.randn(2, 200)
    sw = SlidingWindow(length=100, step_size=50).fit(X)
    X_w = sw.transform(X)
    
    assert isinstance(X_w, np.ndarray)
    assert X_w.shape == (3, 2, 100)

def test_sliding_window_strategies():
    X = np.random.randn(1, 10)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    
    sw_first = SlidingWindow(length=5, step_size=5, label_strategy="first").fit(X)
    _, y_first = sw_first.transform(X, y=y)
    assert y_first[0] == 0  
    
    sw_last = SlidingWindow(length=5, step_size=5, label_strategy="last").fit(X)
    _, y_last = sw_last.transform(X, y=y)
    assert y_last[0] == 1 

def test_sliding_window_errors():
    X = np.random.randn(2, 50)
    
    sw = SlidingWindow(length=100, step_size=10).fit(X)
    with pytest.raises(ValueError, match="is shorter than window length"):
        sw.transform(X)
        
    with pytest.raises(ValueError, match="Length and step_size must be positive integers."):
        SlidingWindow(length=0).fit(X)
        
    with pytest.raises(ValueError, match="Unknown label_strategy"):
        SlidingWindow(label_strategy="invalid").fit(X)