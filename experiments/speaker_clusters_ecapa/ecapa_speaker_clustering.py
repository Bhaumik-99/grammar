"""SHL Grammar Scoring 2026 - speaker embeddings and speaker clusters (CPU).

Public write-ups report repeated speakers across clips. If the same voice
appears in several training clips, random K-fold leaks speaker identity and
overstates CV. We embed every clip with SpeechBrain ECAPA-TDNN (VoxCeleb,
Apache-2.0) and cluster by cosine similarity to obtain speaker groups for
GroupKFold, and to measure train/test speaker overlap.

Output: speaker_ecapa.npz (split, filename, emb[N,192]), speaker_clusters.csv
"""
import glob
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

T0 = time.time()
OUT = "/kaggle/working"


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


subprocess.run([sys.executable, "-m", "pip", "install", "-q", "speechbrain"], check=True)
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from speechbrain.inference.speaker import EncoderClassifier  # noqa: E402

torch.set_num_threads(os.cpu_count())
ROOT = os.path.dirname(glob.glob("/kaggle/input/**/Dataset_Final/train.csv", recursive=True)[0])
df = pd.concat([pd.read_csv(f"{ROOT}/train.csv").assign(split="train"),
                pd.read_csv(f"{ROOT}/test.csv").assign(split="test", label=np.nan)], ignore_index=True)
enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="/tmp/ecapa",
                                     run_opts={"device": "cpu"})
embs = []
for i, r in enumerate(df.itertuples()):
    x, sr = sf.read(f"{ROOT}/{r.split}/{r.filename}", dtype="float32")
    with torch.no_grad():
        e = enc.encode_batch(torch.from_numpy(x).unsqueeze(0)).squeeze().numpy()
    embs.append(e / (np.linalg.norm(e) + 1e-9))
    if i % 100 == 0:
        log(i)
E = np.stack(embs)
np.savez_compressed(f"{OUT}/speaker_ecapa.npz", split=df.split.values, filename=df.filename.values, emb=E)

# ---------------------------------------------------------------- clustering at several thresholds
from sklearn.cluster import AgglomerativeClustering  # noqa: E402

S = E @ E.T
iu = np.triu_indices(len(E), 1)
log("cosine similarity quantiles:", np.round(np.quantile(S[iu], [0.5, 0.9, 0.99, 0.999]), 3))
out = df[["split", "filename", "label"]].copy()
for thr in (0.5, 0.6, 0.7):
    lab = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                  distance_threshold=1 - thr).fit_predict(E)
    out[f"spk_{int(thr * 100)}"] = lab
    sizes = pd.Series(lab).value_counts()
    cross = out.groupby(f"spk_{int(thr * 100)}").split.nunique().gt(1).sum()
    tr = out[out.split == "train"]
    within_sd = tr.groupby(f"spk_{int(thr * 100)}").label.std().dropna()
    log(f"thr {thr}: clusters {len(sizes)}, multi-clip {int((sizes > 1).sum())}, largest {int(sizes.max())}, "
        f"train-test shared {int(cross)}, mean within-speaker label SD {within_sd.mean():.3f} "
        f"(n={len(within_sd)}) vs overall {tr.label.std():.3f}")
out.to_csv(f"{OUT}/speaker_clusters.csv", index=False)
log("done")
