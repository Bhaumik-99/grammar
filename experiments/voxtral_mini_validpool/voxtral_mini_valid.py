"""SHL Grammar Scoring 2026 - Voxtral-Mini-3B audio-token states, pooled over REAL audio only.

The first Voxtral extraction mean-pooled every audio placeholder token. Voxtral pads
audio to 30-s chunks, so a 60.07 s clip gets a third chunk that is >99% padding
(1125 tokens, ~33% padding) while a 45.06 s clip has ~25% padding: the padding share
depends on clip length, which differs between train (mostly 60 s) and test (mostly
45 s). This version
  * caps audio at 60.0 s (drops the near-empty third chunk), and
  * pools each layer only over tokens that encode real audio (12.5 tokens/s per chunk).
Output: emb_voxtral_mini_3b_valid.npz (split, filename, audio_mean[N, 31, 3072] fp16)
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
REPO = "mistralai/Voxtral-Mini-3B-2507"
INSTR = "Listen to this spoken English response and evaluate the speaker's grammatical accuracy."
CAP, TOK_PER_CHUNK, TOK_PER_SEC = 60 * 16000, 375, 12.5


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mistral-common[audio]"], check=True)
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from transformers import AutoProcessor, VoxtralForConditionalGeneration  # noqa: E402

ROOT = os.path.dirname(sorted(glob.glob("/kaggle/input/**/Dataset_Final/train.csv", recursive=True))[0])
df = pd.concat([pd.read_csv(f"{ROOT}/train.csv").assign(split="train"),
                pd.read_csv(f"{ROOT}/test.csv").assign(split="test")], ignore_index=True)
os.makedirs("/tmp/trim", exist_ok=True)


def capped_path(split, fn):
    path = f"{ROOT}/{split}/{fn}"
    info = sf.info(path)
    if info.frames <= CAP:
        return path, info.frames / info.samplerate
    q = f"/tmp/trim/{split}_{fn}"  # split prefix: train and test reuse file names
    if not os.path.exists(q):
        x, sr = sf.read(path, dtype="int16")
        sf.write(q, x[:CAP], sr, subtype="PCM_16")
    return q, CAP / 16000


def valid_token_mask(n_tok, dur):
    assert n_tok % TOK_PER_CHUNK == 0, n_tok
    mask = np.zeros(n_tok, bool)
    for c in range(n_tok // TOK_PER_CHUNK):
        sec = min(30.0, max(0.0, dur - 30.0 * c))
        mask[c * TOK_PER_CHUNK: c * TOK_PER_CHUNK + min(TOK_PER_CHUNK, int(np.ceil(sec * TOK_PER_SEC)))] = True
    return mask


proc = AutoProcessor.from_pretrained(REPO)
model = VoxtralForConditionalGeneration.from_pretrained(REPO, dtype=torch.float16, device_map="cuda:0").eval()
aid = model.config.audio_token_id
means, counts = [], {}
for i, r in enumerate(df.itertuples()):
    path, dur = capped_path(r.split, r.filename)
    conv = [{"role": "user", "content": [{"type": "text", "text": INSTR}, {"type": "audio", "path": path}]}]
    inputs = proc.apply_chat_template(conv)
    inputs = {k: (v.to("cuda:0", torch.float16) if v.dtype.is_floating_point else v.to("cuda:0"))
              for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False, logits_to_keep=1)
    pos = inputs["input_ids"][0] == aid
    n_tok = int(pos.sum())
    valid = torch.from_numpy(valid_token_mask(n_tok, dur)).to("cuda:0")
    hs = torch.stack(out.hidden_states)[:, 0].float()[:, pos][:, valid]  # (L+1, n_valid, D)
    m = hs.mean(1)
    if not torch.isfinite(m).all():
        raise RuntimeError(f"non-finite states at {r.split}/{r.filename}")
    means.append(m.cpu().numpy().astype(np.float16))
    counts[n_tok] = counts.get(n_tok, 0) + 1
    if i % 100 == 0:
        log(i, "audio tokens", n_tok, "valid", int(valid.sum()))
np.savez_compressed(f"{OUT}/emb_voxtral_mini_3b_valid.npz", split=df.split.values, filename=df.filename.values,
                    audio_mean=np.stack(means))
log("saved", np.stack(means).shape, "| audio-token counts", counts)
