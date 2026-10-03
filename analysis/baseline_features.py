"""Build the hand-crafted feature table and score quick baselines with repeated CV.

Usage: python analysis/baseline_features.py <eda_output_dir>
Writes <eda_output_dir>/features_handcrafted.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from lightgbm import LGBMRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from cv import cross_val_predict, make_folds, report  # noqa: E402
from textfeats import linguistic_features, timing_features  # noqa: E402

eda = Path(sys.argv[1])
data_dir = Path(__file__).parents[1] / "data"
train = pd.read_csv(data_dir / "train.csv").assign(split="train")
test = pd.read_csv(data_dir / "test.csv").assign(split="test", label=np.nan)
stats = pd.read_csv(eda / "audio_stats.csv")
asr = {(r["split"], r["filename"]): r for r in map(json.loads, open(eda / "asr_whisper_large_v3.jsonl", encoding="utf-8"))}

df = pd.concat([train, test], ignore_index=True).merge(
    stats.drop(columns=["label"]), on=["split", "filename"], how="left")
df["is_noise"] = (df.dyn_range_db < 5) & (df.peak > 0.9)

nlp = spacy.load("en_core_web_sm")
rows = []
for r, doc in zip(df.itertuples(), nlp.pipe([asr[(s, f)]["text"] for s, f in zip(df.split, df.filename)])):
    rec = asr[(r.split, r.filename)]
    rows.append({**timing_features(rec, r.duration), **linguistic_features(doc)})
feat = pd.concat([df, pd.DataFrame(rows)], axis=1)
feat.to_csv(eda / "features_handcrafted.csv", index=False)

AUDIO = ["duration", "rms_dbfs", "noise_floor_db", "speech_level_db", "dyn_range_db", "active_ratio",
         "pause_total", "n_pauses", "lead_sil", "trail_sil"]
TEXT = [c for c in pd.DataFrame(rows).columns]
cols = AUDIO + TEXT

tr = feat[(feat.split == "train") & ~feat.is_noise].reset_index(drop=True)
te = feat[feat.split == "test"].reset_index(drop=True)
X = tr[cols].fillna(0).values
Xt = te[cols].fillna(0).values
y = tr.label.values
N_REP = 3
folds = make_folds(y, n_splits=5, n_repeats=N_REP, seed=42)


def ridge(Xa, ya, Xv, Xte):
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))).fit(Xa, ya)
    return m.predict(Xv), m.predict(Xte)


def lgbm(Xa, ya, Xv, Xte):
    m = LGBMRegressor(n_estimators=600, learning_rate=0.02, num_leaves=15, min_child_samples=15,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0,
                      verbose=-1).fit(Xa, ya)
    return m.predict(Xv), m.predict(Xte)


print(f"train speech clips: {len(tr)} | test: {len(te)} | features: {len(cols)}")
for name, fn in [("ridge", ridge), ("lgbm", lgbm)]:
    res = cross_val_predict(name, fn, X, y, Xt, folds, N_REP)
    p = np.clip(res.oof, 0, 5)
    print(name, {k: round(v, 4) for k, v in report(y, p).items()},
          "| fold rmse sd", round(float(np.std(res.fold_scores)), 3))

# univariate signal check
corr = tr[cols + ["label"]].corr(method="spearman")["label"].drop("label").sort_values()
print("\nTop |spearman| features:")
print(pd.concat([corr.head(8), corr.tail(12)]).round(3).to_string())
