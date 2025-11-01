"""
Unit Tests for Training Models
Tests XGBoost, Random Forest, Neural Network, and Isolation Forest models
"""

import numpy as np
import pandas as pd
import pytest
from src.models.isolation_forest import IsolationForestModel
from src.models.neural_network import NeuralNetworkModel
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel


class TestXGBoostModel:
    """Test suite for XGBoost fraud detection model"""

    def test_init_default_params(self):
        """Test XGBoost initialization with default parameters"""
        model = XGBoostModel()
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_fit_and_predict(self, X_train_balanced, y_train_balanced, X_test_small):
        """Test fitting and prediction"""
        model = XGBoostModel()
        model.fit(X_train_balanced, y_train_balanced)
        predictions = model.predict(X_test_small)

        assert len(predictions) == len(X_test_small)
        assert all(pred in [0, 1] for pred in predictions)

    def test_predict_proba(self, X_train_balanced, y_train_balanced, X_test_small):
        """Test that predict_proba returns valid probabilities"""
        model = XGBoostModel()
        model.fit(X_train_balanced, y_train_balanced)
        probabilities = model.predict_proba(X_test_small)

        assert probabilities.shape == (len(X_test_small), 2)
        assert np.all((probabilities >= 0) & (probabilities <= 1))
        assert np.allclose(probabilities.sum(axis=1), 1.0)


class TestRandomForestModel:
    """Test suite for Random Forest fraud detection model"""

    def test_init_default_params(self):
        """Test Random Forest initialization"""
        model = RandomForestModel()
        assert model is not None
        assert model.n_estimators > 0

    def test_fit_and_predict(self, X_train_balanced, y_train_balanced, X_test_small):
        """Test fitting and prediction"""
        model = RandomForestModel(n_estimators=50)
        model.fit(X_train_balanced, y_train_balanced)
        predictions = model.predict(X_test_small)

        assert len(predictions) == len(X_test_small)
        assert all(pred in [0, 1] for pred in predictions)

    def test_predict_proba(self, X_train_balanced, y_train_balanced, X_test_small):
        """Test predict_proba returns probabilities"""
        model = RandomForestModel()
        model.fit(X_train_balanced, y_train_balanced)
        probabilities = model.predict_proba(X_test_small)

        assert probabilities.shape == (len(X_test_small), 2)
        assert np.all((probabilities >= 0) & (probabilities <= 1))


class TestNeuralNetworkModel:
    """Test suite for Neural Network fraud detection model"""

    def test_init_creates_model(self):
        """Test that initialization creates neural network"""
        model = NeuralNetworkModel()
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_fit_and_predict(self, X_train_balanced, y_train_balanced, X_test_small):
        """Test training and prediction"""
        model = NeuralNetworkModel(epochs=5)
        model.fit(X_train_balanced, y_train_balanced)
        predictions = model.predict(X_test_small)

        assert len(predictions) == len(X_test_small)
        assert all(pred in [0, 1] for pred in predictions)


class TestIsolationForestModel:
    """Test suite for Isolation Forest (anomaly detection)"""

    def test_init_default_params(self):
        """Test Isolation Forest initialization"""
        model = IsolationForestModel()
        assert model is not None

    def test_fit_and_predict(self, X_train_balanced, X_test_small):
        """Test unsupervised training and prediction"""
        model = IsolationForestModel()
        model.fit(X_train_balanced)
        predictions = model.predict(X_test_small)

        assert len(predictions) == len(X_test_small)
        # Predictions should be 0 or 1 (converted from -1/1)
        assert all(pred in [0, 1] for pred in predictions)


class TestModelComparison:
    """Test suite for comparing all models"""

    def test_all_models_can_be_created(self):
        """Test that all 4 model types can be instantiated"""
        models = {
            "xgboost": XGBoostModel(),
            "random_forest": RandomForestModel(),
            "neural_network": NeuralNetworkModel(),
            "isolation_forest": IsolationForestModel(),
        }

        assert len(models) == 4
        for name, model in models.items():
            assert model is not None

    def test_all_models_can_train_and_predict(
        self, X_train_balanced, y_train_balanced, X_test_small
    ):
        """Test that all models can train and make predictions"""
        models = {
            "xgboost": XGBoostModel(),
            "random_forest": RandomForestModel(n_estimators=50),
            "neural_network": NeuralNetworkModel(epochs=5),
            "isolation_forest": IsolationForestModel(),
        }

        # Train all models
        for name, model in models.items():
            if name == "isolation_forest":
                model.fit(X_train_balanced)  # No y needed
            else:
                model.fit(X_train_balanced, y_train_balanced)

        # Test predictions
        for name, model in models.items():
            predictions = model.predict(X_test_small)
            assert len(predictions) == len(X_test_small)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def X_train_balanced():
    """Fixture providing balanced training features"""
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    return pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )


@pytest.fixture
def y_train_balanced():
    """Fixture providing balanced training labels"""
    np.random.seed(42)
    return pd.Series(np.random.choice([0, 1], 1000, p=[0.5, 0.5]))


@pytest.fixture
def X_test_small():
    """Fixture providing small test set"""
    np.random.seed(42)
    return pd.DataFrame(
        np.random.randn(50, 10), columns=[f"feature_{i}" for i in range(10)]
    )
