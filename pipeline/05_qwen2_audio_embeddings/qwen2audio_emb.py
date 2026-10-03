"""SHL Grammar Scoring 2026 - Qwen2-Audio-7B-Instruct audio-token representations.

A second audio-LLM view (Apache-2.0). The audio encoder accepts <= 30 s, so each
clip is split into consecutive 30 s windows; for every window we feed
[instruction, audio] and mean-pool the language-model hidden states over the
audio-token positions, then average windows weighted by their duration.
Output: emb_qwen2_audio_7b.npz (split, filename, audio_mean[N, L+1, D] float16)
"""
import glob
import os
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

T0 = time.time()
REPO = "Qwen/Qwen2-Audio-7B-Instruct"
SR, WIN = 16000, 30 * 16000


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


ROOT = os.path.dirname(glob.glob("/kaggle/input/**/Dataset_Final/train.csv", recursive=True)[0])
df = pd.concat([pd.read_csv(f"{ROOT}/train.csv").assign(split="train"),
                pd.read_csv(f"{ROOT}/test.csv").assign(split="test")], ignore_index=True)
proc = AutoProcessor.from_pretrained(REPO)
model = Qwen2AudioForConditionalGeneration.from_pretrained(REPO, dtype=torch.float16, device_map="auto").eval()
audio_id = model.config.audio_token_index
conv = [{"role": "user", "content": [
    {"type": "text", "text": "Listen to this spoken English response and evaluate the speaker's grammatical accuracy."},
    {"type": "audio", "audio_url": "clip.wav"}]}]
prompt = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
log("loaded; audio token id", audio_id)


def encode(chunk):
    try:
        inputs = proc(text=prompt, audio=[chunk], sampling_rate=SR, return_tensors="pt")
    except TypeError:  # older processor signature
        inputs = proc(text=prompt, audios=[chunk], sampling_rate=SR, return_tensors="pt")
    dev = model.get_input_embeddings().weight.device
    return {k: (v.to(dev, torch.float16) if v.dtype.is_floating_point else v.to(dev)) for k, v in inputs.items()}


means = []
for i, r in enumerate(df.itertuples()):
    x, sr = sf.read(f"{ROOT}/{r.split}/{r.filename}", dtype="float32")
    acc, wsum = None, 0.0
    for s in range(0, len(x), WIN):
        chunk = x[s:s + WIN]
        if len(chunk) < SR and s > 0:  # skip sub-1 s tails
            continue
        inputs = encode(chunk)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, use_cache=False)
        pos = (inputs["input_ids"][0] == audio_id).to(out.hidden_states[0].device)
        hs = torch.stack([h[0].to(out.hidden_states[0].device) for h in out.hidden_states])[:, pos].float().mean(1)
        if not torch.isfinite(hs).all():
            raise RuntimeError(f"non-finite states at {r.filename} (fp16 overflow)")
        w = len(chunk) / SR
        acc = hs * w if acc is None else acc + hs * w
        wsum += w
    means.append((acc / wsum).cpu().numpy().astype(np.float16))
    if i % 100 == 0:
        log(i, "audio tokens in last window:", int(pos.sum()))
np.savez_compressed("/kaggle/working/emb_qwen2_audio_7b.npz", split=df.split.values, filename=df.filename.values,
                    audio_mean=np.stack(means))
log("saved", np.stack(means).shape)
