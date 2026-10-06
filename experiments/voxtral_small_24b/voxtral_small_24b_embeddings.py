"""SHL Grammar Scoring 2026 - Voxtral-Small-24B audio-token representations (frozen).

Voxtral-Mini-3B audio-token states are our strongest frozen view. This notebook
extracts the same representation from its 24B sibling (Apache-2.0):
  * only the safetensors shards holding the audio encoder, projector, embeddings
    and the first N language-model layers are downloaded (40 GB instead of 48.5);
  * the language model is truncated to N layers (30 of 40 by default) and
    quantised to 4-bit NF4 on the fly (bitsandbytes), split over 2x T4;
  * each clip is fed as [instruction, audio] and every layer's hidden states are
    mean-pooled over the audio-token positions.
Output: emb_voxtral_small_24b.npz (split, filename, audio_mean[N, 30, 5120] fp16;
        index 0 = embedding output, index k = output of decoder layer k, pre-norm)
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

T0 = time.time()
OUT = "/kaggle/working"
REPO = "mistralai/Voxtral-Small-24B-2507"
INSTR = "Listen to this spoken English response and evaluate the speaker's grammatical accuracy."


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes", "mistral-common[audio]"], check=True)
from huggingface_hub import hf_hub_download  # noqa: E402

# ---------------------------------------------------------------- 1. disk budget -> number of LM layers
cands = [d for d in ("/kaggle/tmp", "/tmp", "/root", "/kaggle/working") if os.path.isdir(d)]
free = {d: shutil.disk_usage(d).free for d in cands}
log("free disk (GB):", {d: round(v / 1e9, 1) for d, v in free.items()})
WD = max(free, key=free.get) + "/voxtral_small"
best = free[max(free, key=free.get)]
idx_path = hf_hub_download(REPO, "model.safetensors.index.json", local_dir=WD)
index = json.load(open(idx_path))
wm = index["weight_map"]


def layer_of(key):
    m = re.match(r"language_model\.model\.layers\.(\d+)\.", key)
    return int(m.group(1)) if m else None


def plan(n_layers):
    keys = {k: f for k, f in wm.items()
            if (layer_of(k) is None or layer_of(k) < n_layers) and k != "language_model.model.norm.weight"}
    files = sorted(set(keys.values()))
    return keys, files


sizes = {}
for n in (30, 24, 20):
    keys, files = plan(n)
    sizes[n] = (keys, files)
# shard sizes from the HF API (bytes); fall back to 5 GB per shard if unavailable
try:
    from huggingface_hub import HfApi

    info = HfApi().model_info(REPO, files_metadata=True)
    fsize = {s.rfilename: (s.size or 5e9) for s in info.siblings}
except Exception as e:  # pragma: no cover
    log("size lookup failed:", e)
    fsize = {}
N_LAYERS = None
for n in (30, 24, 20):
    need = sum(fsize.get(f, 5e9) for f in sizes[n][1]) + 3e9  # + headroom
    log(f"n_layers {n}: needs {need / 1e9:.1f} GB on disk")
    if need < best:
        N_LAYERS = n
        break
assert N_LAYERS == 30, f"bands audioL12-19/L20-29 need 30 LM layers; disk only allows {N_LAYERS}"
keys, files = sizes[N_LAYERS]
log(f"using {N_LAYERS} LM layers in {WD}; shards: {files}")

# ---------------------------------------------------------------- 2. download + truncated config / index
for f in ["config.json", "generation_config.json", "preprocessor_config.json", "tekken.json", "params.json"]:
    try:
        hf_hub_download(REPO, f, local_dir=WD)
    except Exception as e:
        log("optional file missing:", f, e)
for f in files:
    hf_hub_download(REPO, f, local_dir=WD)
    log("downloaded", f)
cfg = json.load(open(f"{WD}/config.json"))
cfg["text_config"]["num_hidden_layers"] = N_LAYERS
json.dump(cfg, open(f"{WD}/config.json", "w"), indent=1)
index["weight_map"] = keys
json.dump(index, open(f"{WD}/model.safetensors.index.json", "w"), indent=1)

# ---------------------------------------------------------------- 3. load in 4-bit across both GPUs
import torch  # noqa: E402
from transformers import AutoProcessor, BitsAndBytesConfig, VoxtralForConditionalGeneration  # noqa: E402

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                         llm_int8_skip_modules=["model.audio_tower", "model.multi_modal_projector", "lm_head"])
n_gpu = torch.cuda.device_count()
max_mem = {i: "13GiB" for i in range(n_gpu)}
max_mem["cpu"] = "20GiB"
model = VoxtralForConditionalGeneration.from_pretrained(WD, quantization_config=bnb, dtype=torch.float16,
                                                        device_map="auto", max_memory=max_mem).eval()
import bitsandbytes as bnb_lib  # noqa: E402

for sub in (model.model.audio_tower, model.model.multi_modal_projector):
    assert not any(isinstance(m, bnb_lib.nn.Linear4bit) for m in sub.modules()), "audio path was quantised"
assert any(isinstance(m, bnb_lib.nn.Linear4bit) for m in model.model.language_model.modules()), "LM not quantised"
proc = AutoProcessor.from_pretrained(REPO)
AUDIO_ID = model.config.audio_token_id
in_dev = model.get_input_embeddings().weight.device
log("loaded; layers", len(model.model.language_model.layers), "| input device", in_dev,
    "| memory (GB):", [round(torch.cuda.memory_allocated(i) / 1e9, 1) for i in range(n_gpu)])
for f in files:  # free disk once weights are on the GPUs
    try:
        os.remove(f"{WD}/{f}")
    except OSError:
        pass

# ---------------------------------------------------------------- 4. extract audio-token layer means
ROOT = os.path.dirname(sorted(glob.glob("/kaggle/input/**/Dataset_Final/train.csv", recursive=True))[0])
df = pd.concat([pd.read_csv(f"{ROOT}/train.csv").assign(split="train"),
                pd.read_csv(f"{ROOT}/test.csv").assign(split="test")], ignore_index=True)
means = []


def dump_partial(n):
    np.savez_compressed(f"{OUT}/emb_voxtral_small_24b_partial.npz", split=df.split.values[:n],
                        filename=df.filename.values[:n], audio_mean=np.stack(means), n_layers=np.array(N_LAYERS))


for i, r in enumerate(df.itertuples()):
    conv = [{"role": "user", "content": [{"type": "text", "text": INSTR},
                                         {"type": "audio", "path": f"{ROOT}/{r.split}/{r.filename}"}]}]
    inputs = proc.apply_chat_template(conv)
    inputs = {k: (v.to(in_dev, torch.float16) if v.dtype.is_floating_point else v.to(in_dev)) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False, logits_to_keep=1)
    pos = inputs["input_ids"][0] == AUDIO_ID
    layer_means = []
    for h in out.hidden_states[:N_LAYERS]:  # embeddings + decoder outputs 1..N-1 (pre-norm; GPUs may differ)
        layer_means.append(h[0][pos.to(h.device)].float().mean(0).cpu())
    m = torch.stack(layer_means)
    if not torch.isfinite(m).all():
        if means:
            dump_partial(len(means))
        raise RuntimeError(f"non-finite states at {r.split}/{r.filename} (clip {i})")
    means.append(m.numpy().astype(np.float16))
    if (i + 1) % 200 == 0:
        dump_partial(len(means))
    if i in (0, 5) or i % 100 == 0:
        el = time.time() - T0
        log(f"{i} audio tokens {int(pos.sum())} | {el / 60:.1f} min elapsed")
np.savez_compressed(f"{OUT}/emb_voxtral_small_24b.npz", split=df.split.values, filename=df.filename.values,
                    audio_mean=np.stack(means), n_layers=np.array(N_LAYERS))
if os.path.exists(f"{OUT}/emb_voxtral_small_24b_partial.npz"):
    os.remove(f"{OUT}/emb_voxtral_small_24b_partial.npz")
log("saved", np.stack(means).shape)
