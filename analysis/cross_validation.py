"""Cross-validation, metrics and stacking utilities for the SHL grammar task.

The label is a 0-5 score in 0.5 steps with a block of 37 synthetic-noise clips
labelled 0. Folds are stratified on the score so every fold sees the full
range, and repeated with different seeds to reduce variance on 769 samples.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import nnls
from scipy.stats import pearsonr
from sklearn.model_selection import RepeatedStratifiedKFold


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def pearson(y, p) -> float:
    return float(pearsonr(y, p)[0])


def report(y, p, mask=None) -> dict:
    """RMSE / Pearson overall and, if given, on a subset (e.g. speech-only clips)."""
    out = {"rmse": rmse(y, p), "pearson": pearson(y, p)}
    if mask is not None:
        out["rmse_sub"] = rmse(y[mask], p[mask])
        out["pearson_sub"] = pearson(y[mask], p[mask])
    return out


def make_folds(y, n_splits=5, n_repeats=3, seed=42, strata=None, groups=None):
    """Repeated stratified K-fold on the (rounded) score, optionally crossed with extra strata.

    With `groups` (e.g. speaker clusters) uses StratifiedGroupKFold so a speaker never
    appears in both the training and validation side of a split.
    """
    y2 = np.round(np.asarray(y) * 2).astype(int)
    y2 = np.where(y2 < 4, 4, y2)  # merge the 4 clips at 1.0-1.5 into 2.0
    key = y2.astype(str)
    if strata is not None:
        key = np.char.add(np.char.add(key, "_"), np.asarray(strata).astype(str))
    vals, counts = np.unique(key, return_counts=True)
    rare = set(vals[counts < n_splits])
    if rare:
        key = np.array([k if k not in rare else "rare" for k in key])
    if groups is None:
        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
        return list(rskf.split(np.zeros(len(key)), key))
    from sklearn.model_selection import StratifiedGroupKFold

    folds = []
    for r in range(n_repeats):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        folds += list(sgkf.split(np.zeros(len(key)), key, groups))
    return folds


@dataclass
class OOFResult:
    name: str
    oof: np.ndarray  # (n_train,) averaged over repeats
    test: np.ndarray  # (n_test,) averaged over all fold models
    fold_scores: list = field(default_factory=list)


def cross_val_predict(name: str, fit_predict: Callable, X, y, X_test, folds, n_repeats: int) -> OOFResult:
    """fit_predict(X_tr, y_tr, X_va, X_te) -> (pred_va, pred_te). Averages over repeats."""
    n = len(y)
    oof_sum, oof_cnt = np.zeros(n), np.zeros(n)
    test_preds, scores = [], []
    for tr_idx, va_idx in folds:
        p_va, p_te = fit_predict(X[tr_idx], y[tr_idx], X[va_idx], X_test)
        oof_sum[va_idx] += p_va
        oof_cnt[va_idx] += 1
        test_preds.append(p_te)
        scores.append(rmse(y[va_idx], p_va))
    assert (oof_cnt == n_repeats).all(), "every sample must be predicted once per repeat"
    return OOFResult(name, oof_sum / oof_cnt, np.mean(test_preds, axis=0), scores)


def nnls_blend(oof_matrix: np.ndarray, y: np.ndarray, intercept: bool = True):
    """Non-negative least-squares blend weights (+ free intercept) on OOF predictions."""
    A = oof_matrix
    if intercept:
        # centre so the intercept is unconstrained
        mu_a, mu_y = A.mean(0), y.mean()
        w, _ = nnls(A - mu_a, y - mu_y)
        b = mu_y - mu_a @ w
    else:
        w, _ = nnls(A, y)
        b = 0.0
    return w, b


def nested_blend_score(oof_matrix: np.ndarray, y: np.ndarray, folds) -> tuple[np.ndarray, dict]:
    """Honest estimate of a NNLS blend: weights learned on train folds of the OOF matrix."""
    n = len(y)
    pred_sum, cnt = np.zeros(n), np.zeros(n)
    for tr_idx, va_idx in folds:
        w, b = nnls_blend(oof_matrix[tr_idx], y[tr_idx])
        pred_sum[va_idx] += oof_matrix[va_idx] @ w + b
        cnt[va_idx] += 1
    pred = np.clip(pred_sum / cnt, 0, 5)
    return pred, report(y, pred)
