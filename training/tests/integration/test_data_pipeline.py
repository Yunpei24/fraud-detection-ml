"""
Simple integration tests for data processing pipeline.
"""

import numpy as np
import pandas as pd
import pytest
from src.evaluation.metrics import calculate_all_metrics
from src.models.random_forest import RandomForestModel


def test_model_training_and_prediction(tiny_credit_df):
    """Test that a model can be trained and make predictions"""
    # Simple data preparation
    X = tiny_credit_df.drop(columns=["Class"]).values
    y = tiny_credit_df["Class"].astype(int).values

    # Skip if insufficient data
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 2:
        pytest.skip("Need at least 2 samples per class")

    # Train model
    model = RandomForestModel(use_smote=False, n_estimators=10, random_state=42)
    model.fit(X, y)

    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    # Basic assertions
    assert len(predictions) == len(X)
    assert probabilities.shape == (len(X), 2)
    assert all(p in [0, 1] for p in predictions)
    assert all(0 <= p <= 1 for p in probabilities.flatten())


def test_model_evaluation(tiny_credit_df):
    """Test that model evaluation works"""
    # Simple data preparation
    X = tiny_credit_df.drop(columns=["Class"]).values
    y = tiny_credit_df["Class"].astype(int).values

    # Skip if insufficient data
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 2:
        pytest.skip("Need at least 2 samples per class")

    # Train model
    model = RandomForestModel(use_smote=False, n_estimators=10, random_state=42)
    model.fit(X, y)

    # Get predictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]  # fraud probability

    # Calculate metrics
    metrics = calculate_all_metrics(y, y_proba, y_pred)

    # Check that key metrics exist
    assert "auc" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics

    # Check metric ranges
    for metric_name in ["auc", "precision", "recall", "f1"]:
        value = metrics[metric_name]
        assert 0.0 <= value <= 1.0, f"{metric_name} = {value} not in [0,1]"
