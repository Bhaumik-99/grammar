"""SHL Grammar Scoring 2026 - Frozen Speech-Encoder Embeddings.

For every clip and every encoder layer we store mean- and std-pooled hidden
states, so downstream models can probe which layers carry grammar signal.

Encoders (all encoders-only, run sequentially to avoid OOM):
  whisper_large_v3   openai/whisper-large-v3 encoder    (Apache-2.0)        33 × 1280
  wavlm_large        microsoft/wavlm-large              (UniSpeech licence) 25 × 1024
  w2v_bert_2         facebook/w2v-bert-2.0              (MIT)               25 × 1024

Output: /kaggle/working/emb_<name>.npz with arrays:
  split    (N,)              str   — "train" or "test"
  filename (N,)              str
  mean     (N, L, D)         float16 — layer-wise mean-pooled hidden states
  std      (N, L, D)         float16 — layer-wise std-pooled hidden states
"""
import glob
import os
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
T0     = time.time()
OUT    = "/kaggle/working"
SR     = 16000
DEVICE = "cuda"


def log(*args):
    """Timestamped console log."""
    print(f"[{time.time() - T0:7.1f}s]", *args, flush=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
cands = glob.glob("/kaggle/input/**/Dataset_Final/train.csv", recursive=True)
assert cands, "Competition data not mounted."
ROOT = os.path.dirname(cands[0])

tr = pd.read_csv(f"{ROOT}/train.csv").assign(split="train")
te = pd.read_csv(f"{ROOT}/test.csv").assign(split="test")
df = pd.concat([tr, te], ignore_index=True)
df["path"] = [f"{ROOT}/{s}/{f}" for s, f in zip(df.split, df.filename)]
log(f"Clips: {len(df)}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_audio(path: str) -> np.ndarray:
    """Load audio as 16 kHz mono float32; resample if necessary."""
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    return x


def pool_hidden_states(hidden_states: tuple, n_valid: int | None = None) -> tuple:
    """Mean- and std-pool a tuple of (1, T, D) hidden-state tensors.

    Returns two (L, D) float32 numpy arrays.
    """
    hs = torch.stack([
        h[0, :n_valid] if n_valid is not None else h[0]
        for h in hidden_states
    ]).float()
    return hs.mean(1).cpu().numpy(), hs.std(1).cpu().numpy()


def save_embeddings(name: str, means: list, stds: list) -> None:
    """Compress and write an embedding archive to OUT."""
    mean_arr = np.stack(means).astype(np.float16)
    std_arr  = np.stack(stds).astype(np.float16)
    np.savez_compressed(
        f"{OUT}/emb_{name}.npz",
        split=df.split.values,
        filename=df.filename.values,
        mean=mean_arr,
        std=std_arr,
    )
    log(f"Saved emb_{name}.npz  shape={mean_arr.shape}")


# ---------------------------------------------------------------------------
# 1. Whisper large-v3 encoder
# ---------------------------------------------------------------------------
def run_whisper() -> None:
    """Extract layer-wise states from the Whisper-large-v3 audio encoder."""
    from transformers import WhisperFeatureExtractor, WhisperModel

    repo = "openai/whisper-large-v3"
    fe   = WhisperFeatureExtractor.from_pretrained(repo)
    enc  = WhisperModel.from_pretrained(repo, torch_dtype=torch.float16).encoder
    enc  = enc.to(DEVICE).eval()

    chunk_samples = 30 * SR
    means, stds   = [], []

    for i, path in enumerate(df.path):
        x      = load_audio(path)
        frames = []
        for s in range(0, max(len(x), 1), chunk_samples):
            piece = x[s:s + chunk_samples]
            if len(piece) < SR // 2 and s > 0:  # skip sub-0.5 s tails
                continue
            feats = fe(piece, sampling_rate=SR, return_tensors="pt").input_features
            with torch.no_grad():
                out = enc(feats.to(DEVICE, torch.float16), output_hidden_states=True)
            n_valid = min(1500, int(np.ceil(len(piece) / SR * 50)))  # 50 frames/s
            frames.append(
                torch.stack([h[0, :n_valid] for h in out.hidden_states]).float()
            )
        per_layer = torch.cat(frames, dim=1)  # (L, T_total, D)
        means.append(per_layer.mean(1).cpu().numpy())
        stds.append(per_layer.std(1).cpu().numpy())
        if i % 100 == 0:
            log("Whisper:", i)

    save_embeddings("whisper_large_v3", means, stds)
    del enc
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 2. WavLM large
# ---------------------------------------------------------------------------
def run_wavlm() -> None:
    """Extract layer-wise states from WavLM-large."""
    from transformers import AutoFeatureExtractor, WavLMModel

    repo  = "microsoft/wavlm-large"
    fe    = AutoFeatureExtractor.from_pretrained(repo)
    model = WavLMModel.from_pretrained(repo, torch_dtype=torch.float16).to(DEVICE).eval()

    means, stds = [], []
    for i, path in enumerate(df.path):
        x  = load_audio(path)
        iv = fe(x, sampling_rate=SR, return_tensors="pt").input_values
        with torch.no_grad():
            out = model(iv.to(DEVICE, torch.float16), output_hidden_states=True)
        m, s = pool_hidden_states(out.hidden_states)
        means.append(m)
        stds.append(s)
        if i % 100 == 0:
            log("WavLM:", i)

    save_embeddings("wavlm_large", means, stds)
    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 3. w2v-BERT 2.0
# ---------------------------------------------------------------------------
def run_w2vbert() -> None:
    """Extract layer-wise states from w2v-BERT 2.0."""
    from transformers import AutoFeatureExtractor, Wav2Vec2BertModel

    repo  = "facebook/w2v-bert-2.0"
    fe    = AutoFeatureExtractor.from_pretrained(repo)
    model = Wav2Vec2BertModel.from_pretrained(repo, torch_dtype=torch.float16).to(DEVICE).eval()

    means, stds = [], []
    for i, path in enumerate(df.path):
        x     = load_audio(path)
        feats = fe(x, sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            out = model(
                feats.input_features.to(DEVICE, torch.float16),
                attention_mask=feats.attention_mask.to(DEVICE),
                output_hidden_states=True,
            )
        m, s = pool_hidden_states(out.hidden_states)
        means.append(m)
        stds.append(s)
        if i % 100 == 0:
            log("w2v-BERT:", i)

    save_embeddings("w2v_bert_2", means, stds)
    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Run all encoders; continue if one fails so the rest are not lost
# ---------------------------------------------------------------------------
for encoder_fn in (run_whisper, run_wavlm, run_w2vbert):
    try:
        encoder_fn()
    except Exception as exc:
        import traceback
        log(f"FAILED {encoder_fn.__name__}: {exc!r}")
        traceback.print_exc()

log("Done.")
