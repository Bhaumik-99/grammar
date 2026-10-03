"""Paired speaker-bootstrap comparison of two stack runs (second-level CV predictions of the stack).

Both runs must cover the same training clips. Speakers are resampled with
replacement; the composite score (RMSE + 1 - Pearson) / 2 is recomputed for both
runs on every resample, so the spread of the *difference* reflects how much of an
apparent gain could come from which speakers happen to be in the data.

Usage: python analysis/compare_stacks.py <old> <new> [--n 2000]
       (<old>/<new>: a tag under outputs/stack, a run directory, or a stack_oof.csv file)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUTS = Path(__file__).parents[1] / "outputs" / "stack"


def load(ref):
    p = Path(ref)
    if not p.exists():
        p = OUTS / ref
    return pd.read_csv(p / "stack_oof.csv" if p.is_dir() else p)


def scores(y, p):
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    r = float(np.corrcoef(y, p)[0, 1])
    return rmse, r, (rmse + 1 - r) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()
    a, b = load(args.old), load(args.new)
    m = a.merge(b[["filename", "stack_oof"]], on="filename", suffixes=("_old", "_new"), validate="1:1")
    assert len(m) == len(a) == len(b), "runs cover different clips"
    y, po, pn, g = m.label.values, m.stack_oof_old.values, m.stack_oof_new.values, m.speaker.values
    so, sn = scores(y, po), scores(y, pn)
    print(f"old {args.old:28s} rmse {so[0]:.4f}  r {so[1]:.4f}  composite {so[2]:.4f}")
    print(f"new {args.new:28s} rmse {sn[0]:.4f}  r {sn[1]:.4f}  composite {sn[2]:.4f}")
    spk = np.unique(g)
    by = {s: np.flatnonzero(g == s) for s in spk}
    rng = np.random.default_rng(0)
    d = []
    for _ in range(args.n):
        idx = np.concatenate([by[s] for s in rng.choice(spk, len(spk))])
        d.append(np.subtract(scores(y[idx], pn[idx]), scores(y[idx], po[idx])))
    d = np.array(d)
    for k, name in enumerate(["rmse", "pearson", "composite"]):
        lo, hi = np.percentile(d[:, k], [2.5, 97.5])
        better = (d[:, k] > 0).mean() if name == "pearson" else (d[:, k] < 0).mean()
        print(f"  delta {name:9s} {sn[k] - so[k]:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  P(new better) {better:.3f}")
    for flag, lbl in ((1, "45-s batch"), (0, "other")):
        mk = m.is45.values == flag
        print(f"  {lbl:10s} n={mk.sum():3d}  rmse old {scores(y[mk], po[mk])[0]:.4f}  new {scores(y[mk], pn[mk])[0]:.4f}")


if __name__ == "__main__":
    main()
