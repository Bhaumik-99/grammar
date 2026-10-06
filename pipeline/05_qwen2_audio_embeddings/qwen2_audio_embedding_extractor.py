"""SHL Grammar Scoring 2026 - Qwen2-Audio-7B-Instruct Audio-Token Representations.

A second audio-LLM view (Apache-2.0). The audio encoder accepts <= 30 s, so
each clip is split into consecutive 30 s windows. For every window we feed
[instruction, audio] and mean-pool the LM hidden states over the audio-token
positions, then average windows weighted by their duration.

Output: /kaggle/working/emb_qwen2_audio_7b.npz
  split       (N,)              str
  filename    (N,)              str
  audio_mean  (N, L+1, D)       float16 — weighted-average of per-window means
"""
import glob
import os
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
T0   = time.time()
REPO = "Qwen/Qwen2-Audio-7B-Instruct"
SR   = 16_000
WIN  = 30 * SR   # 30-second window in samples
MIN_CHUNK_SAMPLES = SR   # skip sub-1 s tails
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
log(f"Clips: {len(df)}")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
proc       = AutoProcessor.from_pretrained(REPO)
model      = Qwen2AudioForConditionalGeneration.from_pretrained(
    REPO, dtype=torch.float16, device_map="auto"
).eval()
audio_id   = model.config.audio_token_index

# Build the prompt once (audio URL placeholder is replaced per clip by the processor).
conv = [{
    "role": "user",
    "content": [
        {"type": "text",  "text": GRAMMAR_INSTR},
        {"type": "audio", "audio_url": "clip.wav"},
    ],
}]
prompt = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
log("Model loaded. Audio token id:", audio_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def encode_chunk(chunk: np.ndarray) -> dict:
    """Tokenise one audio window, handling processor API differences."""
    try:
        inputs = proc(text=prompt, audio=[chunk], sampling_rate=SR, return_tensors="pt")
    except TypeError:
        # Older processor versions use `audios` (plural).
        inputs = proc(text=prompt, audios=[chunk], sampling_rate=SR, return_tensors="pt")
    dev = model.get_input_embeddings().weight.device
    return {
        k: (v.to(dev, torch.float16) if v.dtype.is_floating_point else v.to(dev))
        for k, v in inputs.items()
    }


# ---------------------------------------------------------------------------
# Embedding extraction — duration-weighted average across 30 s windows
# ---------------------------------------------------------------------------
means = []
for i, r in enumerate(df.itertuples()):
    x, sr = sf.read(f"{ROOT}/{r.split}/{r.filename}", dtype="float32")
    acc, wsum = None, 0.0

    for s in range(0, len(x), WIN):
        chunk = x[s:s + WIN]
        if len(chunk) < MIN_CHUNK_SAMPLES and s > 0:
            continue  # skip sub-1 s tails
        inputs = encode_chunk(chunk)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, use_cache=False)

        pos = (inputs["input_ids"][0] == audio_id).to(out.hidden_states[0].device)
        hs  = torch.stack([h[0].to(out.hidden_states[0].device) for h in out.hidden_states])
        hs  = hs[:, pos].float().mean(1)  # (L+1, D)

        if not torch.isfinite(hs).all():
            raise RuntimeError(f"Non-finite states at {r.filename} (fp16 overflow)")

        w    = len(chunk) / SR
        acc  = hs * w if acc is None else acc + hs * w
        wsum += w

    means.append((acc / wsum).cpu().numpy().astype(np.float16))
    if i % 100 == 0:
        log(i, "| audio tokens in last window:", int(pos.sum()))

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
mean_arr = np.stack(means)
np.savez_compressed(
    "/kaggle/working/emb_qwen2_audio_7b.npz",
    split=df.split.values,
    filename=df.filename.values,
    audio_mean=mean_arr,
)
log(f"Saved emb_qwen2_audio_7b.npz  shape={mean_arr.shape}")
