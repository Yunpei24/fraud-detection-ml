"""
Simple end-to-end test for training pipeline.
"""

import numpy as np
import pandas as pd
import pytest
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel


def test_multiple_models_can_be_trained(tiny_credit_df):
    """Test that multiple model types can be trained on the same data"""
    # Prepare data
    X = tiny_credit_df.drop(columns=["Class"]).values
    y = tiny_credit_df["Class"].astype(int).values

    # Skip if insufficient data
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 2:
        pytest.skip("Need at least 2 samples per class")

    models = [
        (
            "random_forest",
            RandomForestModel(use_smote=False, n_estimators=10, random_state=42),
        ),
        ("xgboost", XGBoostModel(use_smote=False, n_estimators=10, random_state=42)),
    ]

    trained_models = {}
    for name, model in models:
        model.fit(X, y)
        trained_models[name] = model

        # Basic prediction test
        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert all(p in [0, 1] for p in predictions)

    assert len(trained_models) == 2
    assert "random_forest" in trained_models
    assert "xgboost" in trained_models


def test_models_have_predict_proba(tiny_credit_df):
    """Test that trained models can generate probability predictions"""
    # Prepare data
    X = tiny_credit_df.drop(columns=["Class"]).values
    y = tiny_credit_df["Class"].astype(int).values

    # Skip if insufficient data
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 2:
        pytest.skip("Need at least 2 samples per class")

    model = RandomForestModel(use_smote=False, n_estimators=10, random_state=42)
    model.fit(X, y)

    # Test predict_proba
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (len(X), 2)
    assert all(0 <= p <= 1 for p in probabilities.flatten())

    # Check that probabilities sum to 1 for each sample
    prob_sums = probabilities.sum(axis=1)
    assert all(abs(s - 1.0) < 1e-6 for s in prob_sums)
