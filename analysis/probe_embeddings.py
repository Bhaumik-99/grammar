"""Layer-wise ridge probes on frozen speech/audio-LLM embeddings.

Usage: python analysis/probe_embeddings.py [--groups spk_60]
For every encoder file in outputs/audio_emb (emb_*.npz) and every layer: z-scored
mean (or mean+std) pooled states -> ridge with alpha chosen by inner LOO, scored by
repeated (speaker-grouped) stratified 5-fold CV on speech (non-noise) train clips.
Writes outputs/audio_emb/probe_results.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from cv import make_folds, report  # noqa: E402
from fastridge import ridge_oof  # noqa: E402

ROOT = Path(__file__).parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--groups", default=None)
ap.add_argument("--repeats", type=int, default=1)
ap.add_argument("--with-std", action="store_true")
args = ap.parse_args()

labels = pd.read_csv(ROOT / "data/train.csv").assign(split="train")
stats = pd.read_csv(ROOT / "outputs/eda/audio_stats.csv")
stats["is_noise"] = (stats.dyn_range_db < 5) & (stats.peak > 0.9)
spk = None
if args.groups:
    gfile = "pseudo_speakers.csv" if args.groups == "pseudo_spk" else "speaker_clusters.csv"
    spk = pd.read_csv(ROOT / "outputs/speakers" / gfile)[["split", "filename", args.groups]]

rows = []
for f in sorted((ROOT / "outputs/audio_emb").glob("emb_*.npz")):
    z = np.load(f, allow_pickle=True)
    key = pd.DataFrame({"split": z["split"], "filename": z["filename"]})
    key = key.merge(stats[["split", "filename", "is_noise", "duration"]], on=["split", "filename"], how="left")
    key = key.merge(labels.drop(columns=[]), on=["split", "filename"], how="left")
    if spk is not None:
        key = key.merge(spk, on=["split", "filename"], how="left")
    sel = ((key.split == "train") & ~key.is_noise).values
    y = key.label.values[sel]
    groups = key[args.groups].values[sel] if spk is not None else None
    folds = make_folds(y, n_splits=5, n_repeats=args.repeats, seed=7, groups=groups)
    is45 = (np.abs(key.duration.values[sel] - 45.06) < 0.2)
    views = {k: z[k].astype(np.float32) for k in ("mean", "audio_mean", "last") if k in z.files}
    std = z["std"].astype(np.float32) if "std" in z.files else None
    name = f.stem[4:]
    print(f"{f.name}: views {[(k, v.shape) for k, v in views.items()]}, speech clips {sel.sum()}", flush=True)
    for vname, arr in views.items():
        for layer in range(arr.shape[1]):
            pools = [("mean", arr[sel, layer])]
            if std is not None and vname == "mean" and args.with_std:
                pools.append(("mean+std", np.hstack([arr[sel, layer], std[sel, layer]])))
            for pool, X in pools:
                oof, _, alphas = ridge_oof(X, y, folds)
                p = np.clip(oof, 0, 5)
                r = report(y, p)
                rows.append({"encoder": name, "view": vname, "layer": layer, "pool": pool,
                             "rmse": r["rmse"], "pearson": r["pearson"],
                             "composite": (r["rmse"] + 1 - r["pearson"]) / 2,
                             "rmse_45s": float(np.sqrt(np.mean((y[is45] - p[is45]) ** 2))),
                             "alpha_med": float(np.median(alphas))})
    res = pd.DataFrame(rows)
    best = res[res.encoder == name].sort_values("rmse").head(6)
    print(best.round(4).to_string(index=False), flush=True)

out = pd.DataFrame(rows)
out.to_csv(ROOT / "outputs/audio_emb/probe_results.csv", index=False)
print("\nBest per encoder/view:")
print(out.loc[out.groupby(["encoder", "view"]).rmse.idxmin()].round(4).to_string(index=False))
