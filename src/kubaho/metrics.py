"""Evaluation metrics, threshold calibration and McNemar's test."""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from . import CLASSES, FLOOR_LEVEL

FALL = CLASSES.index("fall")
FLOOR_IDX = [CLASSES.index(c) for c in FLOOR_LEVEL]


def fall_metrics(y: np.ndarray, fall_pred: np.ndarray) -> dict:
    """Binary fall-alert metrics from class labels y and boolean alert decisions."""
    is_fall = y == FALL
    floor = np.isin(y, FLOOR_IDX)
    tp = int((fall_pred & is_fall).sum())
    fp = int((fall_pred & ~is_fall).sum())
    fn = int((~fall_pred & is_fall).sum())
    sens = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    return {
        "fall_sensitivity": sens,
        "fall_precision": prec,
        "fall_f1": 2 * prec * sens / max(prec + sens, 1e-9),
        "specificity_floor_level": float((~fall_pred[floor]).mean()) if floor.any() else float("nan"),
        "specificity_all_non_fall": float((~fall_pred[~is_fall]).mean()),
        "false_alarms": fp,
    }


def evaluate(y: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    """Multi-class metrics (argmax) plus fall-alert metrics at a calibrated threshold."""
    pred = proba.argmax(1)
    out = {
        "accuracy": accuracy_score(y, pred),
        "macro_precision": precision_score(y, pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y, pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "threshold": threshold,
    }
    out.update(fall_metrics(y, proba[:, FALL] >= threshold))
    return out


def calibrate_threshold(y: np.ndarray, p_fall: np.ndarray, target_sensitivity: float = 0.95) -> float:
    """Highest P(fall) threshold that keeps validation sensitivity >= target (fewest false alarms)."""
    falls = np.sort(p_fall[y == FALL])
    if len(falls) == 0:
        return 0.5
    k = int(np.floor((1 - target_sensitivity) * len(falls)))
    return float(min(falls[k], 0.5))


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact McNemar test on paired per-window correctness."""
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    return {"a_only_correct": b, "b_only_correct": c, "p_value": p}
