import numpy as np
from training.src.models.ensemble import hard_vote, threshold_tuning

# Test for the hard_vote function
def test_hard_vote_majority():
    a = np.array([0, 1, 1, 0])
    b = np.array([1, 1, 0, 0])
    c = np.array([1, 1, 1, 0])
    voted = hard_vote([a, b, c])
    assert np.array_equal(voted, np.array([1, 1, 1, 0]))  # Majority voting should result in [1, 1, 1, 0]

# Test for threshold_tuning function
def test_threshold_tuning_returns_threshold():
    y_true = np.array([0, 0, 1, 0, 1, 0, 1, 0])
    y_proba = np.array([0.1, 0.2, 0.9, 0.3, 0.8, 0.2, 0.7, 0.4])
    thr, metr = threshold_tuning(y_proba, y_true, min_recall=0.5)
    assert 0.0 <= thr <= 1.0  # Threshold should be between 0 and 1
    assert "recall" in metr and "precision" in metr and "f1" in metr  # Ensure metrics contain recall, precision, and f1
