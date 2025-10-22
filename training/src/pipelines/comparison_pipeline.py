# training/src/pipelines/comparison_pipeline.py
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy import stats

from training.src.evaluation.metrics import calculate_all_metrics


def compare_models(champion, challenger, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Returns metrics for both models on the same test set.
    """
    def _score(m):
        if hasattr(m, "predict_proba"):
            proba = m.predict_proba(X_test)
        elif hasattr(m, "decision_function"):
            scores = m.decision_function(X_test)
            proba = _to_rank_prob(scores)
        else:
            proba = m.predict(X_test)

        yhat = m.predict(X_test)
        return calculate_all_metrics(y_test, np.asarray(proba), np.asarray(yhat))

    return {"champion": _score(champion), "challenger": _score(challenger)}


def statistical_test(champion_metrics: Dict[str, float], challenger_metrics: Dict[str, float]) -> bool:
    """
    Simple significance check on F1 (placeholder).
    In practice, prefer a paired test using fold-wise scores.
    """
    f1_c = champion_metrics.get("f1", 0.0)
    f1_n = challenger_metrics.get("f1", 0.0)
    # Dummy test: if challenger improves by >= 0.01, consider it “significant”
    return (f1_n - f1_c) >= 0.01


def decide_deployment(comparison_results: Dict[str, Dict[str, float]]) -> str:
    """
    If challenger significantly better by our rule, choose challenger; else keep champion.
    """
    champ = comparison_results["champion"]
    chal = comparison_results["challenger"]
    if statistical_test(champ, chal):
        return "promote_challenger"
    return "keep_champion"


# ---- helpers ----
def _to_rank_prob(scores: np.ndarray) -> np.ndarray:
    import pandas as pd
    ranks = pd.Series(scores).rank(method="average") / len(scores)
    return 1.0 - ranks.values
