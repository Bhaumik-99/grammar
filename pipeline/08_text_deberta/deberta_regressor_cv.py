"""SHL Grammar Scoring 2026 - DeBERTa-v3-large Regression on ASR Transcripts.

Speaker-grouped StratifiedGroupKFold (5 folds, WavLM pseudo-speakers), one
training seed per GPU, run in parallel.

Noise clips (label 0, flat ~3 dB dynamic range at peak ≈ 0.95) are handled
by a downstream rule and are excluded from training here.

The model input is the concatenation of two transcript views:
  [whisper_clean_transcript] [SEP] [parakeet_ctc_transcript]
Each transcript occupies roughly 384 tokens at max_length.

Inputs (attached notebook outputs):
  asr_whisper_large_v3.jsonl    Clean Whisper transcripts  (pipeline 01)
  asr_parakeet_ctc_1.1b.jsonl   LM-free CTC transcripts    (pipeline 02)
  audio_stats.csv               Per-clip audio stats        (pipeline 01)
  emb_wavlm_large.npz           WavLM states for speaker groups (pipeline 03)

Outputs (/kaggle/working):
  deberta_oof.csv    split, filename, label, pred  (train OOF, averaged over seeds)
  deberta_test.csv   split, filename, pred          (test, averaged over all fold models)
"""
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
T0       = time.time()
OUT      = "/kaggle/working"
MODEL_ID = "microsoft/deberta-v3-large"
ASR_FILE = os.environ.get("ASR_FILE", "asr_whisper_large_v3.jsonl")

N_SPLITS = 5
EPOCHS   = 4
LR       = 1.5e-5
BATCH    = 8
MAX_LEN  = 384


def log(*args):
    """Timestamped console log."""
    print(f"[{time.time() - T0:7.1f}s]", *args, flush=True)


def find(pattern: str) -> str:
    """Return the first file matching pattern under /kaggle/input."""
    hits = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    assert hits, f"{pattern} not found under /kaggle/input"
    return hits[0]


# ---------------------------------------------------------------------------
# Training worker script (one subprocess per GPU / seed)
# ---------------------------------------------------------------------------
WORKER_SCRIPT = r'''
import json, math, os, sys, time
import numpy as np, pandas as pd, torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedGroupKFold

cfg = json.loads(sys.argv[1])
torch.manual_seed(cfg["seed"])
np.random.seed(cfg["seed"])

data = pd.read_csv(cfg["data"])
tr   = data[data.split == "train"].reset_index(drop=True)
te   = data[data.split == "test"].reset_index(drop=True)
tok  = AutoTokenizer.from_pretrained(cfg["model"])


class MeanPoolRegressor(nn.Module):
    """DeBERTa backbone with attention-mask mean pooling and a linear head."""

    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(cfg["model"], torch_dtype=torch.float32)
        self.backbone.config.hidden_dropout_prob = 0.0
        self.backbone.config.attention_probs_dropout_prob = 0.0
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)

    def forward(self, ids, mask):
        h = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        return self.head((h * m).sum(1) / m.sum(1).clamp(min=1)).squeeze(-1)


def encode(texts):
    enc = tok(list(texts), truncation=True, max_length=cfg["max_len"],
              padding=True, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


@torch.no_grad()
def predict(model, texts):
    model.eval()
    out = []
    for i in range(0, len(texts), 16):
        ids, mask = encode(texts[i:i + 16])
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(ids.cuda(), mask.cuda()).float().cpu().numpy())
    return np.concatenate(out)


y     = tr.label.values.astype(np.float32)
strat = np.maximum(np.round(y * 2).astype(int), 4)  # merge the few grades below 2.0
skf   = StratifiedGroupKFold(n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["seed"])
oof   = np.zeros(len(tr))
test_pred = np.zeros(len(te))
mu    = float(y.mean())

for fold, (tr_idx, va_idx) in enumerate(skf.split(tr, strat, tr.speaker.values)):
    model = MeanPoolRegressor().cuda()
    # Gradient checkpointing: saves activation memory (two transcript views fill 384 tokens).
    model.backbone.gradient_checkpointing_enable()
    nn.init.zeros_(model.head.weight)
    nn.init.constant_(model.head.bias, mu)

    # Layer-wise LR decay: top encoder layer gets lr, each layer below × 0.9.
    layers   = model.backbone.encoder.layer
    n_layers = len(layers)
    params   = [
        {"params": model.head.parameters(),               "lr": 1e-3},
        {"params": model.backbone.embeddings.parameters(),"lr": cfg["lr"] * 0.9 ** n_layers},
    ]
    for i, layer in enumerate(layers):
        params.append({"params": layer.parameters(), "lr": cfg["lr"] * 0.9 ** (n_layers - 1 - i)})
    rest = [p for n, p in model.backbone.named_parameters()
            if not n.startswith(("embeddings.", "encoder.layer."))]
    params.append({"params": rest, "lr": cfg["lr"]})

    opt    = torch.optim.AdamW(params, weight_decay=0.01)
    steps  = cfg["epochs"] * math.ceil(len(tr_idx) / cfg["batch"])
    sch    = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler()

    texts_tr, y_tr = tr.text.values[tr_idx], y[tr_idx]
    for ep in range(cfg["epochs"]):
        model.train()
        perm, losses = np.random.permutation(len(tr_idx)), []
        for i in range(0, len(tr_idx), cfg["batch"]):
            idx  = perm[i:i + cfg["batch"]]
            ids, mask = encode(texts_tr[idx])
            with torch.autocast("cuda", dtype=torch.float16):
                pred = model(ids.cuda(), mask.cuda())
            loss = nn.functional.mse_loss(pred.float(), torch.tensor(y_tr[idx]).cuda())
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
            losses.append(loss.item())
        pv = predict(model, tr.text.values[va_idx])
        print(f"seed {cfg['seed']} fold {fold} ep {ep} "
              f"train_mse {np.mean(losses):.3f} "
              f"val_rmse {np.sqrt(np.mean((pv - y[va_idx]) ** 2)):.4f}", flush=True)
    oof[va_idx] = pv
    test_pred  += predict(model, te.text.values) / cfg["n_splits"]
    del model, opt
    torch.cuda.empty_cache()

pd.DataFrame({
    "split": "train", "filename": tr.filename, "label": y, "pred": oof,
}).to_csv(cfg["oof_out"], index=False)
pd.DataFrame({
    "split": "test", "filename": te.filename, "pred": test_pred,
}).to_csv(cfg["test_out"], index=False)
print(f"seed {cfg['seed']} OOF RMSE {np.sqrt(np.mean((oof - y) ** 2)):.4f}", flush=True)
'''

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
comp  = os.path.dirname(find("Dataset_Final/train.csv"))
train = pd.read_csv(f"{comp}/train.csv").assign(split="train")
test  = pd.read_csv(f"{comp}/test.csv").assign(split="test", label=np.nan)

# Primary transcript (clean Whisper) and secondary (Parakeet-CTC, LM-free).
asr = pd.DataFrame(
    [json.loads(l) for l in open(find(ASR_FILE))]
)[["split", "filename", "text"]]
ctc = pd.DataFrame(
    [json.loads(l) for l in open(find("asr_parakeet_ctc_1.1b.jsonl"))]
)[["split", "filename", "text"]]

asr = asr.merge(ctc.rename(columns={"text": "ctc_text"}), on=["split", "filename"], how="left")
# Concatenate both transcript views with a [SEP] separator.
asr["text"] = (
    asr.text.fillna("").str.strip() + " [SEP] " + asr.ctc_text.fillna("").str.strip()
)
asr = asr.drop(columns=["ctc_text"])

stats = pd.read_csv(find("audio_stats.csv"))[["split", "filename", "dyn_range_db", "peak"]]
data  = pd.concat([train, test]).merge(asr,   on=["split", "filename"], how="left") \
                                .merge(stats, on=["split", "filename"], how="left")
assert data.text.notna().all(), "Missing transcripts"
data["is_noise"] = (data.dyn_range_db < 5) & (data.peak > 0.9)
log(
    "Noise clips:", data.groupby("split").is_noise.sum().to_dict(),
    "| train noise labels:", data[(data.split == "train") & data.is_noise].label.value_counts().to_dict(),
)

# ---------------------------------------------------------------------------
# Pseudo-speaker groups (WavLM layers 3–6, cosine connected components)
# ---------------------------------------------------------------------------
from scipy.sparse import csr_matrix                   # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

zw = np.load(find("emb_wavlm_large.npz"), allow_pickle=True)
V  = np.hstack([
    zw["mean"][:, 3:7].reshape(len(zw["mean"]), -1),
    zw["std"][:, 3:7].reshape(len(zw["std"]), -1),
]).astype(np.float64)
V  = (V - V.mean(0)) / (V.std(0) + 1e-8)
V /= np.linalg.norm(V, axis=1, keepdims=True)
S  = V @ V.T
np.fill_diagonal(S, -1)
thr = float(np.percentile(
    S[np.ix_(zw["split"] == "test", zw["split"] == "train")].max(1), 99.5
))
_, comp_ids = connected_components(csr_matrix(S > thr), directed=False)
data = data.merge(
    pd.DataFrame({"split": zw["split"], "filename": zw["filename"], "speaker": comp_ids}),
    on=["split", "filename"], how="left",
)
log(f"Speaker groups: {data[data.split == 'train'].speaker.nunique()} | threshold: {thr:.3f}")

# Exclude noise clips from training (handled by downstream rule); keep all test rows.
model_data = data[~data.is_noise | (data.split == "test")].copy()
model_data["text"] = model_data.text.fillna("").str.strip().replace("", "(no speech)")
model_data.to_csv(f"{OUT}/model_data.csv", index=False)

# ---------------------------------------------------------------------------
# Launch one worker per GPU (one seed per GPU → averaged OOF)
# ---------------------------------------------------------------------------
n_gpu = sum(
    1 for line in
    subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines()
    if line.startswith("GPU")
)
worker_path = f"{OUT}/deberta_worker.py"
with open(worker_path, "w") as fh:
    fh.write(WORKER_SCRIPT)

procs = []
for g in range(max(n_gpu, 1)):
    cfg = dict(
        seed=g, data=f"{OUT}/model_data.csv", model=MODEL_ID,
        n_splits=N_SPLITS, epochs=EPOCHS, lr=LR, batch=BATCH, max_len=MAX_LEN,
        oof_out=f"{OUT}/oof_seed{g}.csv", test_out=f"{OUT}/test_seed{g}.csv",
    )
    procs.append(subprocess.Popen(
        [sys.executable, worker_path, json.dumps(cfg)],
        env=dict(os.environ,
                 CUDA_VISIBLE_DEVICES=str(g),
                 PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"),
    ))

for p in procs:
    p.wait()
    log("Worker exit code:", p.returncode)

# ---------------------------------------------------------------------------
# Aggregate seeds and write final outputs
# ---------------------------------------------------------------------------
oofs  = [pd.read_csv(f) for f in sorted(glob.glob(f"{OUT}/oof_seed*.csv"))]
tests = [pd.read_csv(f) for f in sorted(glob.glob(f"{OUT}/test_seed*.csv"))]
assert oofs, "No worker output found — check tracebacks above."

oof = oofs[0].copy()
oof["pred"] = np.mean([o.pred.values for o in oofs], axis=0)
tst = tests[0].copy()
tst["pred"] = np.mean([t.pred.values for t in tests], axis=0)

oof.to_csv(f"{OUT}/deberta_oof.csv",  index=False)
tst.to_csv(f"{OUT}/deberta_test.csv", index=False)

rmse = np.sqrt(np.mean((oof.pred - oof.label) ** 2))
pearson = np.corrcoef(oof.pred, oof.label)[0, 1]
log(f"DeBERTa OOF ({len(oofs)} seed(s), scorable clips): RMSE {rmse:.4f} | Pearson r {pearson:.4f}")
