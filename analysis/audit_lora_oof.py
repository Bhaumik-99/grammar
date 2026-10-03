"""Audit the LoRA regressor's out-of-fold predictions before they enter the stack.

Checks: every scorable training clip has exactly one OOF prediction; no pseudo-speaker
spans two folds (the kernel rebuilt the groups itself, so they must match the local
ones); the test file covers all test clips; and reports OOF quality by fold.

Usage: python analysis/audit_lora_oof.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
OUTS = ROOT / "outputs"

oof = pd.read_csv(OUTS / "voxtral_lora/voxtral_lora_oof.csv")
tst = pd.read_csv(OUTS / "voxtral_lora/voxtral_lora_test.csv")
sp = pd.read_csv(OUTS / "speakers/pseudo_speakers.csv")
st = pd.read_csv(OUTS / "eda/audio_stats.csv")
lab = pd.read_csv(ROOT / "data/train.csv")
test_ids = pd.read_csv(ROOT / "data/test.csv")

st["is_noise"] = (st.dyn_range_db < 5) & (st.peak > 0.9)
speech = st[(st.split == "train") & ~st.is_noise].filename
assert oof.filename.is_unique, "duplicate OOF rows"
assert set(oof.filename) == set(speech), f"OOF covers {len(set(oof.filename) & set(speech))} of {len(speech)} speech clips"
assert set(tst.filename) == set(test_ids.filename) and tst.filename.is_unique, "test predictions incomplete"
assert np.isfinite(oof.pred).all() and np.isfinite(tst.pred).all()

m = oof.merge(sp[sp.split == "train"][["filename", "pseudo_spk"]], on="filename", how="left")
m = m.merge(lab.rename(columns={"label": "label_ref"}), on="filename", how="left")
assert np.allclose(m.label, m.label_ref), "labels in the OOF file differ from train.csv"
spread = m.groupby("pseudo_spk").fold.nunique()
print(f"pseudo-speakers: {spread.size} | spanning >1 fold: {int((spread > 1).sum())}")
assert (spread == 1).all(), "speaker groups leak across LoRA folds"

r = float(np.corrcoef(m.pred, m.label)[0, 1])
e = float(np.sqrt(np.mean((m.pred - m.label) ** 2)))
ec = float(np.sqrt(np.mean((m.pred.clip(1, 5) - m.label) ** 2)))
print(f"LoRA OOF: rmse {e:.4f} (clipped {ec:.4f}) | pearson {r:.4f} | composite {(ec + 1 - r) / 2:.4f}")
print(m.groupby("fold").apply(lambda g: pd.Series({"n": len(g), "rmse": np.sqrt(np.mean((g.pred - g.label) ** 2)),
                                                   "r": np.corrcoef(g.pred, g.label)[0, 1]})).round(4))
print("test pred summary:", tst.pred.describe().round(3).to_dict())
