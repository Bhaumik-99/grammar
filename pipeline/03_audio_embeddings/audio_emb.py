"""SHL Grammar Scoring 2026 - frozen speech-encoder embeddings.

For every clip and every encoder layer we store mean- and std-pooled hidden
states, so downstream models can probe which layers carry grammar signal.

Encoders:
  whisper_large_v3  openai/whisper-large-v3 encoder   (Apache-2.0)        33 x 1280
  wavlm_large       microsoft/wavlm-large             (UniSpeech licence) 25 x 1024
  w2v_bert_2        facebook/w2v-bert-2.0             (MIT)               25 x 1024

Output: /kaggle/working/emb_<name>.npz with
  split (N,), filename (N,), mean (N, L, D) float16, std (N, L, D) float16
Train and test reuse file names, so rows are keyed by (split, filename).
"""
import glob
import os
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch

T0 = time.time()
OUT = "/kaggle/working"
SR = 16000
DEVICE = "cuda"


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ---------------------------------------------------------------- data
cands = glob.glob("/kaggle/input/**/Dataset_Final/train.csv", recursive=True)
assert cands, "competition data not mounted"
ROOT = os.path.dirname(cands[0])
tr = pd.read_csv(f"{ROOT}/train.csv").assign(split="train")
te = pd.read_csv(f"{ROOT}/test.csv").assign(split="test")
df = pd.concat([tr, te], ignore_index=True)
df["path"] = [f"{ROOT}/{s}/{f}" for s, f in zip(df.split, df.filename)]
log("clips", len(df))


def load_audio(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != SR:
        import librosa

        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    return x


def pool(hidden_states, n_valid=None):
    """hidden_states: tuple of (1, T, D) -> mean/std arrays of shape (L, D)."""
    hs = torch.stack([h[0, :n_valid] if n_valid else h[0] for h in hidden_states]).float()
    return hs.mean(1).cpu().numpy(), hs.std(1).cpu().numpy()


def save(name, means, stds):
    np.savez_compressed(
        f"{OUT}/emb_{name}.npz",
        split=df.split.values, filename=df.filename.values,
        mean=np.stack(means).astype(np.float16), std=np.stack(stds).astype(np.float16))
    log(f"saved emb_{name}.npz", np.stack(means).shape)


# ---------------------------------------------------------------- 1. Whisper large-v3 encoder
def run_whisper():
    from transformers import WhisperFeatureExtractor, WhisperModel

    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
    enc = WhisperModel.from_pretrained("openai/whisper-large-v3", torch_dtype=torch.float16).encoder
    enc = enc.to(DEVICE).eval()
    chunk = 30 * SR
    means, stds = [], []
    for i, p in enumerate(df.path):
        x = load_audio(p)
        per_layer = None
        frames = []
        for s in range(0, max(len(x), 1), chunk):
            piece = x[s:s + chunk]
            if len(piece) < SR // 2 and s > 0:  # ignore sub-0.5 s tails
                continue
            feats = fe(piece, sampling_rate=SR, return_tensors="pt").input_features
            with torch.no_grad():
                out = enc(feats.to(DEVICE, torch.float16), output_hidden_states=True)
            n_valid = min(1500, int(np.ceil(len(piece) / SR * 50)))  # 50 frames / s
            frames.append(torch.stack([h[0, :n_valid] for h in out.hidden_states]).float())
        per_layer = torch.cat(frames, dim=1)  # (L, T, D)
        means.append(per_layer.mean(1).cpu().numpy())
        stds.append(per_layer.std(1).cpu().numpy())
        if i % 100 == 0:
            log("whisper", i)
    save("whisper_large_v3", means, stds)
    del enc
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- 2. WavLM large
def run_wavlm():
    from transformers import AutoFeatureExtractor, WavLMModel

    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")
    model = WavLMModel.from_pretrained("microsoft/wavlm-large", torch_dtype=torch.float16)
    model = model.to(DEVICE).eval()
    means, stds = [], []
    for i, p in enumerate(df.path):
        x = load_audio(p)
        iv = fe(x, sampling_rate=SR, return_tensors="pt").input_values
        with torch.no_grad():
            out = model(iv.to(DEVICE, torch.float16), output_hidden_states=True)
        m, s = pool(out.hidden_states)
        means.append(m)
        stds.append(s)
        if i % 100 == 0:
            log("wavlm", i)
    save("wavlm_large", means, stds)
    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- 3. w2v-BERT 2.0
def run_w2vbert():
    from transformers import AutoFeatureExtractor, Wav2Vec2BertModel

    fe = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0", torch_dtype=torch.float16)
    model = model.to(DEVICE).eval()
    means, stds = [], []
    for i, p in enumerate(df.path):
        x = load_audio(p)
        feats = fe(x, sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            out = model(feats.input_features.to(DEVICE, torch.float16),
                        attention_mask=feats.attention_mask.to(DEVICE),
                        output_hidden_states=True)
        m, s = pool(out.hidden_states)
        means.append(m)
        stds.append(s)
        if i % 100 == 0:
            log("w2v-bert", i)
    save("w2v_bert_2", means, stds)
    del model
    torch.cuda.empty_cache()


for fn in (run_whisper, run_wavlm, run_w2vbert):
    try:
        fn()
    except Exception as e:  # keep the other encoders if one fails
        import traceback

        log(f"FAILED {fn.__name__}: {e!r}")
        traceback.print_exc()
log("done")
