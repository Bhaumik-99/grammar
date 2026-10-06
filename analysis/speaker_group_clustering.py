"""Pseudo-speaker groups from low-layer WavLM statistics (voice timbre, not content).

Clips are linked when their cosine similarity exceeds the 99.5th percentile of
test-to-train nearest-neighbour similarities (an empirical "different speaker"
reference), then grouped by connected components. Used for GroupKFold so that
the same voice never sits on both sides of a CV split.

Usage: python analysis/speaker_groups.py  -> outputs/speakers/pseudo_speakers.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).parents[1]
z = np.load(ROOT / "outputs/audio_emb/emb_wavlm_large.npz", allow_pickle=True)
X = np.hstack([z["mean"][:, 3:7].reshape(len(z["mean"]), -1), z["std"][:, 3:7].reshape(len(z["std"]), -1)])
X = X.astype(np.float64)
X = (X - X.mean(0)) / (X.std(0) + 1e-8)
X /= np.linalg.norm(X, axis=1, keepdims=True)
S = X @ X.T
np.fill_diagonal(S, -1)
split = z["split"]
tr, te = split == "train", split == "test"
nn_te_tr = S[np.ix_(te, tr)].max(1)
thr = float(np.percentile(nn_te_tr, 99.5))
nn_tr_tr = S[np.ix_(tr, tr)].max(1)
print(f"test->train NN sim: median {np.median(nn_te_tr):.3f}, p99.5 {thr:.3f} | "
      f"train->train NN sim median {np.median(nn_tr_tr):.3f}")
A = csr_matrix(S > thr)
n_comp, lab = connected_components(A, directed=False)
out = pd.DataFrame({"split": split, "filename": z["filename"], "pseudo_spk": lab})
sizes = out.pseudo_spk.value_counts()
lbl = pd.read_csv(ROOT / "data/train.csv")
m = out[out.split == "train"].merge(lbl, on="filename")
multi = m.groupby("pseudo_spk").filter(lambda g: len(g) > 1)
print(f"groups {n_comp} | multi-clip groups {int((sizes > 1).sum())} | largest {int(sizes.max())} | "
      f"train clips in multi-clip groups {len(multi)} / {len(m)}")
print(f"within-group label SD {multi.groupby('pseudo_spk').label.std().mean():.3f} vs overall {m.label.std():.3f}")
print("train-test shared groups:", int(out.groupby("pseudo_spk").split.nunique().gt(1).sum()))
(ROOT / "outputs/speakers").mkdir(parents=True, exist_ok=True)
out.to_csv(ROOT / "outputs/speakers/pseudo_speakers.csv", index=False)
