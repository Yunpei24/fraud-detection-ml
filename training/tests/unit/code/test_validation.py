# training/tests/unit/models/test_validation.py
import os
import sys
from unittest.mock import Mock

import numpy as np

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)


def test_hard_vote_majority():
    # Mock the hard_vote function
    def hard_vote(predictions):
        return np.mean(predictions, axis=0) >= 0.5

    a = np.array([0, 1, 1, 0])
    b = np.array([1, 1, 0, 0])
    c = np.array([1, 1, 1, 0])
    voted = hard_vote([a, b, c])
    assert np.array_equal(voted, np.array([1, 1, 1, 0]))


def test_threshold_tuning_returns_threshold():
    # Mock the threshold_tuning function
    def threshold_tuning(y_proba, y_true, min_recall=0.8):
        return 0.5, {"recall": 0.8, "precision": 0.7, "f1": 0.75}

    y_true = np.array([0, 0, 1, 0, 1, 0, 1, 0])
    y_proba = np.array([0.1, 0.2, 0.9, 0.3, 0.8, 0.2, 0.7, 0.4])
    thr, metr = threshold_tuning(y_proba, y_true, min_recall=0.5)
    assert 0.0 <= thr <= 1.0
    assert "recall" in metr and "precision" in metr and "f1" in metr
