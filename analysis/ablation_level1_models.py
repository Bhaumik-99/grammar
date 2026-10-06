"""Second-level CV ablation of the final notebook's stack from its saved level-1 OOF matrix.

Reproduces section 7 of the notebook (same NNLS-with-intercept, same outer folds:
StratifiedGroupKFold 5 x 3, seed SEED + 1000), once with all base columns and once
without the columns matching --drop, then compares the two cross-validated stack predictions with a
paired speaker bootstrap of the composite score (RMSE + 1 - Pearson) / 2.

Fold generator: the Kaggle image runs a scikit-learn release whose
StratifiedGroupKFold(shuffle=True) shuffles the per-group class counts without remapping
the groups (fixed in later releases). `--folds kaggle` (default) re-implements that
behaviour, so `--seeds 1` reproduces the notebook's Kaggle numbers exactly;
`--folds local` uses the installed scikit-learn.

Usage: python analysis/ablate_level1.py <oof_level1.csv> --drop voxtral_mini_lora [--seeds 5]
"""
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import check_random_state

SEED, N_SPLITS, N_REPEATS = 42, 5, 3
KEY_COLS = ["filename", "speaker", "is45", "label"]


class KaggleSGKF(StratifiedGroupKFold):
    """StratifiedGroupKFold as shipped in the Kaggle image: with shuffle=True the rows of the per-group class counts
    are shuffled but the group indices are not remapped (groups stay intact; the stratification is scrambled)."""

    def _iter_test_indices(self, X, y, groups):
        rng = check_random_state(self.random_state)
        _, y_inv, y_cnt = np.unique(np.asarray(y), return_inverse=True, return_counts=True)
        _, groups_inv, groups_cnt = np.unique(groups, return_inverse=True, return_counts=True)
        counts = np.zeros((len(groups_cnt), len(y_cnt)))
        for c, gi in zip(y_inv, groups_inv):
            counts[gi, c] += 1
        per_fold, members = np.zeros((self.n_splits, len(y_cnt))), defaultdict(set)
        if self.shuffle:
            rng.shuffle(counts)
        for gi in np.argsort(-np.std(counts, axis=1), kind="mergesort"):
            best = self._find_best_fold(y_counts_per_fold=per_fold, y_cnt=y_cnt, group_y_counts=counts[gi])
            per_fold[best] += counts[gi]
            members[best].add(gi)
        for k in range(self.n_splits):
            yield [i for i, gi in enumerate(groups_inv) if gi in members[k]]


def make_folds(y, groups, seed, generator="kaggle"):
    strata = np.maximum(np.round(y * 2).astype(int), 4)
    cls = KaggleSGKF if generator == "kaggle" else StratifiedGroupKFold
    folds = []
    for r in range(N_REPEATS):
        sgkf = cls(n_splits=N_SPLITS, shuffle=True, random_state=seed + r)
        folds += list(sgkf.split(np.zeros(len(y)), strata, groups))
    return folds


def nnls_fit(A, y):
    mu_a, mu_y = A.mean(0), y.mean()
    w, _ = nnls(A - mu_a, y - mu_y)
    return w, mu_y - mu_a @ w


def stack_cv(A, y, folds):
    pred, cnt = np.zeros(len(y)), np.zeros(len(y))
    for tr, va in folds:
        w, b = nnls_fit(A[tr], y[tr])
        pred[va] += A[va] @ w + b
        cnt[va] += 1
    return np.clip(pred / cnt, 1, 5)


def scores(y, p):
    e = float(np.sqrt(np.mean((y - p) ** 2)))
    r = float(np.corrcoef(y, p)[0, 1])
    return e, r, (e + 1 - r) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("level1")
    ap.add_argument("--drop", required=True, help="comma-separated substrings of base columns to remove")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=1,
                    help="outer-fold seeds to average (1 = the notebook's folds only; >1 damps fold-assignment noise)")
    ap.add_argument("--folds", default="kaggle", choices=["kaggle", "local"],
                    help="fold generator: 'kaggle' reproduces the notebook's Kaggle runs, 'local' uses installed sklearn")
    args = ap.parse_args()
    d = pd.read_csv(args.level1)
    cols = [c for c in d.columns if c not in KEY_COLS]
    drops = [s for s in args.drop.split(",") if s]
    reduced = [c for c in cols if not any(s in c for s in drops)]
    print("dropped columns:", sorted(set(cols) - set(reduced)))
    assert len(reduced) < len(cols), f"--drop {args.drop!r} matched no column"
    y, g = d.label.values, d.speaker.values
    p_full, p_red = np.zeros(len(y)), np.zeros(len(y))
    for k in range(args.seeds):  # seed SEED + 1000 first: exactly the notebook's outer folds
        folds2 = make_folds(y, g, SEED + 1000 + 100 * k, args.folds)
        p_full += stack_cv(d[cols].values, y, folds2) / args.seeds
        p_red += stack_cv(d[reduced].values, y, folds2) / args.seeds
    print(f"outer-fold seeds averaged: {args.seeds} | fold generator: {args.folds}")
    sf, sr = scores(y, p_full), scores(y, p_red)
    print(f"without: rmse {sr[0]:.4f}  r {sr[1]:.4f}  composite {sr[2]:.4f}  ({len(reduced)} columns)")
    print(f"with   : rmse {sf[0]:.4f}  r {sf[1]:.4f}  composite {sf[2]:.4f}  ({len(cols)} columns)")
    w, b = nnls_fit(d[cols].values, y)
    print("full-data weights of the dropped columns:",
          {c: round(float(x), 3) for c, x in zip(cols, w) if c not in reduced})
    spk = np.unique(g)
    by = {s: np.flatnonzero(g == s) for s in spk}
    rng = np.random.default_rng(0)
    diff = []
    for _ in range(args.n):
        idx = np.concatenate([by[s] for s in rng.choice(spk, len(spk))])
        diff.append(np.subtract(scores(y[idx], p_full[idx]), scores(y[idx], p_red[idx])))
    diff = np.array(diff)
    for k, name in enumerate(["rmse", "pearson", "composite"]):
        lo, hi = np.percentile(diff[:, k], [2.5, 97.5])
        better = (diff[:, k] > 0).mean() if name == "pearson" else (diff[:, k] < 0).mean()
        print(f"  delta {name:9s} {sf[k] - sr[k]:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  P(with better) {better:.3f}")
    m45 = d.is45.values.astype(bool)
    print(f"  45-s batch (n={m45.sum()}): rmse without {scores(y[m45], p_red[m45])[0]:.4f}  "
          f"with {scores(y[m45], p_full[m45])[0]:.4f}")


if __name__ == "__main__":
    main()
