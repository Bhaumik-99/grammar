"""Fast ridge regression for many (layer x fold) probes on small n.

Uses the dual (kernel) form with a linear kernel: one Gram matrix per feature
view, one eigendecomposition per training fold, and the closed-form
leave-one-out (GCV) error to choose alpha *inside* the training fold - the
same model RidgeCV fits (centred intercept), at a fraction of the cost when
features >> samples.
"""
from __future__ import annotations

import numpy as np


def standardize(X: np.ndarray) -> np.ndarray:
    """Column z-scoring (unsupervised; fitted on train+test features, no labels)."""
    mu, sd = X.mean(0), X.std(0)
    return (X - mu) / np.where(sd > 1e-8, sd, 1.0)


def ridge_fold(K_tr, K_va_tr, y_tr, alphas):
    """Fit centred kernel ridge on one fold; alpha by exact LOO on the training part."""
    n = len(y_tr)
    ym = y_tr.mean()
    yc = y_tr - ym
    # centre the kernel (equivalent to an unpenalised intercept / centred features)
    one = np.full((n, n), 1.0 / n)
    Kc = K_tr - one @ K_tr - K_tr @ one + one @ K_tr @ one
    lam, V = np.linalg.eigh(Kc)
    lam = np.clip(lam, 0, None)
    Vy = V.T @ yc
    best, best_a = np.inf, alphas[0]
    for a in alphas:
        shrink = lam / (lam + a)
        fit = V @ (shrink * Vy)
        h = (V ** 2) @ shrink  # diag of hat matrix
        loo = (yc - fit) / np.clip(1 - h, 1e-6, None)
        err = np.mean(loo ** 2)
        if err < best:
            best, best_a = err, a
    coef = V @ (Vy / (lam + best_a))  # dual coefficients
    kv = K_va_tr - K_va_tr.mean(1, keepdims=True) - K_tr.mean(0, keepdims=True) + K_tr.mean()
    return kv @ coef + ym, best_a


def _centered_eig(K_tr):
    rm, cm, tm = K_tr.mean(1, keepdims=True), K_tr.mean(0, keepdims=True), K_tr.mean()
    lam, V = np.linalg.eigh(K_tr - rm - cm + tm)
    return np.clip(lam, 0, None), V, (rm, cm, tm)


def ridge_fold_grouped(K_tr, K_pred_list, y_tr, groups_tr, alphas, n_inner=4):
    """Kernel ridge with alpha chosen by *speaker-grouped* inner CV (not LOO).

    LOO inside a training fold that contains same-speaker twins favours too little
    regularisation (a clip's twin "predicts" it); grouped inner folds avoid that.
    """
    from sklearn.model_selection import GroupKFold

    errs = np.zeros(len(alphas))
    for itr, iva in GroupKFold(n_splits=n_inner).split(K_tr, y_tr, groups_tr):
        lam, V, (rm, cm, tm) = _centered_eig(K_tr[np.ix_(itr, itr)])
        ym = y_tr[itr].mean()
        Vy = V.T @ (y_tr[itr] - ym)
        Kp = K_tr[np.ix_(iva, itr)]
        Kp = Kp - Kp.mean(1, keepdims=True) - cm + tm
        KpV = Kp @ V
        for ai, a in enumerate(alphas):
            pred = KpV @ (Vy / (lam + a)) + ym
            errs[ai] += np.sum((y_tr[iva] - pred) ** 2)
    best_a = alphas[int(np.argmin(errs))]
    lam, V, (rm, cm, tm) = _centered_eig(K_tr)
    ym = y_tr.mean()
    coef = V @ ((V.T @ (y_tr - ym)) / (lam + best_a))
    outs = [(Kp - Kp.mean(1, keepdims=True) - cm + tm) @ coef + ym for Kp in K_pred_list]
    return outs, best_a


def ridge_oof_grouped(X, y, folds, groups, X_test, alphas=None):
    """OOF / test / in-sample predictions with grouped-inner-CV alpha selection."""
    if alphas is None:
        alphas = np.logspace(-1, 6, 36) / 1000.0
    Z = standardize(np.vstack([X, X_test]).astype(np.float64))
    K = Z @ Z.T / Z.shape[1]
    n = len(y)
    Ktr, Kte = K[:n, :n], K[n:, :n]
    oof, cnt, test, chosen = np.zeros(n), np.zeros(n), [], []
    for tr, va in folds:
        (p_va, p_te), a = ridge_fold_grouped(Ktr[np.ix_(tr, tr)], [Ktr[np.ix_(va, tr)], Kte[:, tr]],
                                             y[tr], groups[tr], alphas)
        oof[va] += p_va
        cnt[va] += 1
        test.append(p_te)
        chosen.append(a)
    (p_full,), _ = ridge_fold_grouped(Ktr, [Ktr], y, groups, alphas)
    return oof / cnt, np.mean(test, 0), p_full, chosen


def ridge_oof(X: np.ndarray, y: np.ndarray, folds, alphas=None, X_test: np.ndarray | None = None):
    """OOF predictions (averaged over repeats) and mean test prediction for a feature matrix."""
    if alphas is None:
        alphas = np.logspace(-1, 6, 36) * X.shape[1] / 1000.0
    Xall = X if X_test is None else np.vstack([X, X_test])
    Z = standardize(Xall.astype(np.float64))
    K = Z @ Z.T / Z.shape[1]
    n = len(y)
    Ktr_all = K[:n, :n]
    oof_sum, cnt = np.zeros(n), np.zeros(n)
    test_preds, chosen = [], []
    for tr, va in folds:
        p, a = ridge_fold(Ktr_all[np.ix_(tr, tr)], Ktr_all[np.ix_(va, tr)], y[tr], alphas / Z.shape[1])
        oof_sum[va] += p
        cnt[va] += 1
        chosen.append(a)
        if X_test is not None:
            pt, _ = ridge_fold(Ktr_all[np.ix_(tr, tr)], K[n:, :n][:, tr], y[tr], alphas / Z.shape[1])
            test_preds.append(pt)
    oof = oof_sum / np.maximum(cnt, 1)
    test = np.mean(test_preds, 0) if test_preds else None
    return oof, test, chosen


if __name__ == "__main__":
    # sanity check against sklearn RidgeCV on random data
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 500))
    w = rng.normal(size=500) * (rng.random(500) < 0.05)
    y = X @ w + rng.normal(size=200)
    Z = standardize(X)
    K = Z @ Z.T / Z.shape[1]
    tr, va = np.arange(150), np.arange(150, 200)
    p, a = ridge_fold(K[np.ix_(tr, tr)], K[np.ix_(va, tr)], y[tr], np.array([1.0]) / 500)
    ref = Ridge(alpha=1.0).fit(Z[tr], y[tr]).predict(Z[va])
    print("max |dual - sklearn| =", float(np.abs(p - ref).max()))
