# training/src/models/ensemble.py
import numpy as np

def hard_vote(models_predictions: list) -> np.ndarray:
    """
    Perform hard voting based on majority rule.
    Args:
        models_predictions (list): List of model predictions (each model returns an array of predictions).
    Returns:
        np.ndarray: Voted predictions based on majority.
    """
    return np.array([np.bincount(model_predictions).argmax() for model_predictions in zip(*models_predictions)])

def threshold_tuning(y_pred_proba: np.ndarray, y_true: np.ndarray, min_recall: float = 0.95) -> tuple:
    thresholds = np.linspace(0, 1, 100)
    best_threshold = 0
    best_metrics = {}

    for threshold in thresholds:
        predictions = (y_pred_proba >= threshold).astype(int)
        recall = np.sum((predictions == 1) & (y_true == 1)) / np.sum(y_true == 1)
        
        # Prevent division by zero for precision calculation
        precision = np.sum((predictions == 1) & (y_true == 1)) / (np.sum(predictions == 1) if np.sum(predictions == 1) > 0 else 1)
        
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        if recall >= min_recall:
            best_threshold = threshold
            best_metrics = {"recall": recall, "precision": precision, "f1": f1}

    return best_threshold, best_metrics
