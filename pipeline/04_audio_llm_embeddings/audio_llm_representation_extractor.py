"""SHL Grammar Scoring 2026 - Audio-LLM and HuBERT Frozen Representations.

Two models run in parallel, one per GPU:

  GPU 0 — mistralai/Voxtral-Mini-3B-2507 (Apache-2.0)
    Each clip is fed as [instruction text, audio]. We mean-pool the LM hidden
    states over the audio-token positions for every decoder layer (strongest
    signal was found in LM layers ~9-14), and also record the final-token
    state of every layer.

  GPU 1 — facebook/hubert-large-ll60k (Apache-2.0)
    Layer-wise mean- and std-pooled hidden states — a licence-safe alternative
    to WavLM when the UniSpeech licence is a concern.

Outputs (/kaggle/working):
  emb_voxtral_mini_3b.npz   split, filename, audio_mean[N, L, D], last[N, L, D]
  emb_hubert_large.npz      split, filename, mean[N, L, D], std[N, L, D]
"""
import glob
import json
import os
import subprocess
import sys
import time

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
T0  = time.time()
OUT = "/kaggle/working"
VOXTRAL_REPO = "mistralai/Voxtral-Mini-3B-2507"
HUBERT_REPO  = "facebook/hubert-large-ll60k"
ITEMS_CSV    = "/tmp/items.csv"
GRAMMAR_INSTR = (
    "Listen to this spoken English response and evaluate the speaker's grammatical accuracy."
)


def log(*args):
    """Timestamped console log."""
    print(f"[{time.time() - T0:7.1f}s]", *args, flush=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(
    glob.glob("/kaggle/input/**/Dataset_Final/train.csv", recursive=True)[0]
)
df = pd.concat(
    [
        pd.read_csv(f"{ROOT}/train.csv").assign(split="train"),
        pd.read_csv(f"{ROOT}/test.csv").assign(split="test"),
    ],
    ignore_index=True,
)
df["path"] = [f"{ROOT}/{s}/{f}" for s, f in zip(df.split, df.filename)]
df[["split", "filename", "path"]].to_csv(ITEMS_CSV, index=False)
log(f"Clips: {len(df)}")

# ---------------------------------------------------------------------------
# Install Mistral audio dependency
# ---------------------------------------------------------------------------
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mistral-common[audio]"], check=True)


# ---------------------------------------------------------------------------
# Worker scripts (run as subprocesses, one per GPU)
# ---------------------------------------------------------------------------
VOXTRAL_WORKER = rf'''
import numpy as np, pandas as pd, torch, time
from transformers import AutoProcessor, VoxtralForConditionalGeneration

REPO  = "{VOXTRAL_REPO}"
INSTR = "{GRAMMAR_INSTR}"

items = pd.read_csv("{ITEMS_CSV}")
proc  = AutoProcessor.from_pretrained(REPO)
model = VoxtralForConditionalGeneration.from_pretrained(
    REPO, dtype=torch.float16, device_map="cuda:0"
).eval()
aid   = model.config.audio_token_id

means, lasts = [], []
t0 = time.time()
for i, r in enumerate(items.itertuples()):
    conv   = [{{"role": "user", "content": [{{"type": "text", "text": INSTR}},
                                            {{"type": "audio", "path": r.path}}]}}]
    inputs = proc.apply_chat_template(conv)
    inputs = {{k: (v.to("cuda:0", torch.float16) if v.dtype.is_floating_point else v.to("cuda:0"))
              for k, v in inputs.items()}}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False, logits_to_keep=1)
    pos = inputs["input_ids"][0] == aid
    hs  = torch.stack(out.hidden_states)[:, 0].float()  # (L+1, T, D)
    m   = hs[:, pos].mean(1)
    if not torch.isfinite(m).all():
        raise RuntimeError(f"Non-finite hidden states (fp16 overflow?) at {{r.filename}}")
    means.append(m.cpu().numpy().astype(np.float16))
    lasts.append(hs[:, -1].cpu().numpy().astype(np.float16))
    if i % 100 == 0:
        print(f"voxtral {{i}} audio_tokens={{int(pos.sum())}} {{time.time() - t0:.0f}}s", flush=True)

np.savez_compressed(
    "/kaggle/working/emb_voxtral_mini_3b.npz",
    split=items.split.values, filename=items.filename.values,
    audio_mean=np.stack(means), last=np.stack(lasts),
)
print("voxtral saved", np.stack(means).shape, flush=True)
'''

HUBERT_WORKER = rf'''
import numpy as np, pandas as pd, soundfile as sf, torch, time
from transformers import AutoFeatureExtractor, HubertModel

REPO  = "{HUBERT_REPO}"
items = pd.read_csv("{ITEMS_CSV}")
fe    = AutoFeatureExtractor.from_pretrained(REPO)
model = HubertModel.from_pretrained(REPO, dtype=torch.float16).cuda().eval()

means, stds = [], []
t0 = time.time()
for i, r in enumerate(items.itertuples()):
    x, sr = sf.read(r.path, dtype="float32")
    iv    = fe(x, sampling_rate=16000, return_tensors="pt").input_values.cuda().half()
    with torch.no_grad():
        hs = torch.stack(model(iv, output_hidden_states=True).hidden_states)[:, 0].float()
    means.append(hs.mean(1).cpu().numpy().astype(np.float16))
    stds.append(hs.std(1).cpu().numpy().astype(np.float16))
    if i % 100 == 0:
        print(f"hubert {{i}} {{time.time() - t0:.0f}}s", flush=True)

np.savez_compressed(
    "/kaggle/working/emb_hubert_large.npz",
    split=items.split.values, filename=items.filename.values,
    mean=np.stack(means), std=np.stack(stds),
)
print("hubert saved", np.stack(means).shape, flush=True)
'''


# ---------------------------------------------------------------------------
# Launch workers and wait
# ---------------------------------------------------------------------------
workers = [("voxtral", VOXTRAL_WORKER), ("hubert", HUBERT_WORKER)]
procs   = []
for g, (name, code) in enumerate(workers):
    worker_path = f"/tmp/{name}_worker.py"
    with open(worker_path, "w") as fh:
        fh.write(code)
    procs.append((name, subprocess.Popen(
        [sys.executable, worker_path],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g)),
    )))

for name, p in procs:
    p.wait()
    log(f"{name} worker exit code: {p.returncode}")

log("Done. Output directory:", os.listdir(OUT))
