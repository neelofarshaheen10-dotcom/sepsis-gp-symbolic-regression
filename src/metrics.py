"""Shared metric utilities for NB10, NB11, NB12, NB13.

Import in notebooks:
    from src.metrics import compute_ece, compute_metrics, calibration_bins
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    roc_curve,
    f1_score,
)
from sklearn.linear_model import LogisticRegression as _CalLR


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (uniform-width bins, correct last-bin boundary).

    The final bin uses <= so that predictions clipped to exactly 1.0 are included.
    All other bins use [lo, hi).  This matches NB10's original implementation and
    is the canonical version used across all notebooks.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (y_prob >= edges[i]) & (y_prob < edges[i + 1])
        else:
            mask = (y_prob >= edges[i]) & (y_prob <= edges[i + 1])
        if mask.sum() > 0:
            ece += (mask.sum() / len(y_prob)) * abs(
                y_true[mask].mean() - y_prob[mask].mean()
            )
    return float(ece)


def calibration_bins(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean_predicted, fraction_positive, bin_size) for calibration plots."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    conf, acc, sizes = [], [], []
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (y_prob >= edges[i]) & (y_prob < edges[i + 1])
        else:
            mask = (y_prob >= edges[i]) & (y_prob <= edges[i + 1])
        if mask.sum() > 0:
            conf.append(y_prob[mask].mean())
            acc.append(y_true[mask].mean())
            sizes.append(mask.sum())
    return np.array(conf), np.array(acc), np.array(sizes)


def compute_metrics(
    y_true: np.ndarray,
    y_raw: np.ndarray,
    label: str = "",
    fitness_alpha: float = 1.0,
    fitness_beta: float = 0.5,
) -> dict:
    """Full metric suite used for model evaluation across all notebooks.

    Parameters
    ----------
    y_true      : binary ground-truth labels
    y_raw       : raw model output (clipped internally to (1e-7, 1-1e-7))
    label       : identifier string stored in returned dict
    fitness_alpha/beta : weights for fitness = alpha*AUROC - beta*ECE
    """
    y_true = np.asarray(y_true)
    p = np.clip(np.asarray(y_raw), 1e-7, 1 - 1e-7)

    auroc = roc_auc_score(y_true, p)
    auprc = average_precision_score(y_true, p)

    null_brier = float(np.mean((y_true - y_true.mean()) ** 2))
    brier = brier_score_loss(y_true, p)
    brier_skill = 1.0 - brier / null_brier

    ece = compute_ece(y_true, p)

    logit_p = np.log(p / (1.0 - p))
    cal = _CalLR(solver="lbfgs", max_iter=1000)
    cal.fit(logit_p.reshape(-1, 1), y_true)
    cal_slope = float(cal.coef_[0][0])
    cal_intercept = float(cal.intercept_[0])

    fpr, tpr, thresh = roc_curve(y_true, p)
    best_idx = int(np.argmax(tpr - fpr))
    best_thr = float(thresh[best_idx])
    y_pred = (p >= best_thr).astype(int)
    sensitivity = float(tpr[best_idx])
    specificity = float(1.0 - fpr[best_idx])
    f1 = float(f1_score(y_true, y_pred))

    fitness = fitness_alpha * auroc - fitness_beta * ece

    return {
        "label": label,
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "brier": round(brier, 4),
        "brier_skill_pct": round(brier_skill * 100, 1),
        "ece_10bin": round(ece, 4),
        "cal_slope": round(cal_slope, 4),
        "cal_intercept": round(cal_intercept, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
        "fitness": round(fitness, 4),
        "threshold": round(best_thr, 4),
    }
