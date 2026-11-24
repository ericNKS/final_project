"""Exact binary ranking metrics and F1 threshold selection on CUDA, including ties."""

import numpy as np
import pandas as pd

from src.utils.cuda import initialize


def curve(y, probability):
    """Return cumulative true/false positives at the end of each tied score group."""
    import cupy as cp
    initialize()
    labels = cp.asarray(np.asarray(y), dtype=cp.int64)
    scores = cp.asarray(np.asarray(probability), dtype=cp.float64)
    if labels.size == 0 or labels.size != scores.size:
        raise ValueError("Rótulos e escores devem ter o mesmo tamanho não vazio")
    if not bool(cp.isfinite(scores).all()) or not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("Rótulos ou escores inválidos")
    order = cp.argsort(-scores)
    ordered = scores[order]
    endpoints = cp.flatnonzero(cp.r_[ordered[1:] != ordered[:-1], True])
    tp = cp.cumsum(labels[order])[endpoints].astype(cp.float64)
    fp = (endpoints + 1).astype(cp.float64) - tp
    return labels, scores, ordered[endpoints], tp, fp


def metrics(y: pd.Series, probability: np.ndarray, threshold: float) -> dict:
    """Match sklearn binary metrics; absent-class AUC/AP remains explicitly unavailable."""
    import cupy as cp
    labels, scores, _, true_positive, false_positive = curve(y, probability)
    prediction = scores >= threshold
    tp = int(cp.count_nonzero(prediction & (labels == 1)))
    fp = int(cp.count_nonzero(prediction & (labels == 0)))
    fn = int(cp.count_nonzero(~prediction & (labels == 1)))
    tn = int(labels.size) - tp - fp - fn
    positive, negative = tp + fn, tn + fp
    if positive and negative:
        recall = true_positive / positive
        precision = true_positive / (true_positive + false_positive)
        ap = float(cp.sum(cp.diff(cp.r_[0., recall]) * precision))
        auc = float(cp.trapezoid(cp.r_[0., recall], cp.r_[0., false_positive / negative]))
    else:
        ap = auc = None
    return {"accuracy": (tp + tn) / int(labels.size), "precision": tp / (tp + fp) if tp + fp else 0.,
            "recall": tp / positive if positive else 0., "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            "average_precision": ap, "roc_auc": auc, "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def calibrate(y: pd.Series, probability: np.ndarray) -> float:
    """Maximize F1 on the calibration period; choose the highest tied threshold."""
    import cupy as cp
    labels, _, thresholds, tp, fp = curve(y, probability)
    positive = int(cp.sum(labels))
    if not 0 < positive < len(labels):
        raise ValueError("Calibração temporal exige as duas classes; não haverá fallback aleatório")
    f1 = 2 * tp / (positive + tp + fp)
    # thresholds are descending, so the first optimum is the highest cutoff.
    return float(thresholds[cp.flatnonzero(f1 == f1.max())[0]])


def average_precision(y: pd.Series, probability: np.ndarray) -> float:
    """Ranking selection uses the same CUDA implementation as the final evaluation."""
    result = metrics(y, probability, .5)["average_precision"]
    if result is None:
        raise ValueError("Janela de validação sem ambas as classes")
    return result
