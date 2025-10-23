# training/src/utils/helpers.py
from __future__ import annotations

import json
import os
import pickle
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Sequence, Tuple

import numpy as np
import yaml
from sklearn.metrics import precision_recall_fscore_support, roc_curve


def set_seed(seed: int = 42):
    """Set the seed for reproducibility."""
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def ensure_dir(path: os.PathLike | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Dict[str, Any], path: os.PathLike | str, indent: int = 2) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def load_json(path: os.PathLike | str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_yaml(obj: Dict[str, Any], path: os.PathLike | str) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def load_yaml(path: os.PathLike | str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_pickle(obj: Any, path: os.PathLike | str) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: os.PathLike | str) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)


@contextmanager
def timer(label: str) -> Generator[None, None, None]:
    t0 = time.time()
    yield
    dt = time.time() - t0
    print(f"[TIMER] {label}: {dt:.3f}s")


def chunked(iterable: Sequence[Any] | Iterable[Any], n: int) -> Iterable[List[Any]]:
    """Yield successive chunks of size n from an iterable/sequence."""
    chunk: List[Any] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Simple inverse-frequency class weights for binary labels.
    Returns {0: w0, 1: w1}
    """
    y = np.asarray(y).astype(int)
    n = len(y)
    n_pos = max(int((y == 1).sum()), 1)
    n_neg = max(n - n_pos, 1)
    w1 = n / (2.0 * n_pos)
    w0 = n / (2.0 * n_neg)
    return {0: w0, 1: w1}


def tune_threshold_for_recall(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_recall: float = 0.95,
    min_precision: float | None = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Find the lowest threshold achieving >= target_recall on validation.
    Optionally enforce a minimum precision. Returns (threshold, metrics_dict).
    """
    fpr, tpr, thr = roc_curve(y_true, y_proba)  # thresholds aligned with fpr/tpr
    # roc_curve returns thresholds descending; iterate to find earliest meeting recall
    best_thr = 0.5
    best = {"recall": 0.0, "precision": 0.0, "f1": 0.0}

    # Convert TPR -> recall; evaluate precision at each threshold
    for t, recall in zip(thr, tpr):
        y_hat = (y_proba >= t).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_hat, average="binary", zero_division=0)
        if rec >= target_recall and (min_precision is None or prec >= min_precision):
            best_thr = float(t)
            best = {"recall": float(rec), "precision": float(prec), "f1": float(f1)}
            break

    return best_thr, best
