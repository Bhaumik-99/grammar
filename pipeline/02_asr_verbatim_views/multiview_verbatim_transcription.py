"""SHL Grammar Scoring 2026 - Error-Preserving ASR Views.

Whisper's decoder behaves like a strong language model and silently "repairs"
learner grammar, capping every transcript-based grammar feature. This notebook
adds two transcript views that preserve more of what was actually said:

  1. whisper_large_v3_verbatim   faster-whisper large-v3 conditioned on a
     deliberately disfluent, ungrammatical initial prompt; keeps fillers,
     repetitions and errors. Word timestamps + probabilities.
  2. parakeet_ctc_1.1b           NVIDIA Parakeet-CTC-1.1B (CC-BY-4.0): greedy
     CTC decoding with no language model; word timings + frame confidences.

Outputs (/kaggle/working):
  asr_whisper_large_v3_verbatim.jsonl  one record per (split, filename)
  asr_parakeet_ctc_1.1b.jsonl         one record per (split, filename)
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
T0 = time.time()
OUT = "/kaggle/working"
# Disfluent initial prompt: encourages Whisper to keep learner errors rather
# than silently correct them.
VERBATIM_PROMPT = (
    "Umm, so, uh, I I go to the market yesterday and, and I buyed some, uh, vegetable. "
    "Then I, I am going home and... hmm, like, my mother she cook the food, no, she cooked it."
)
PARAKEET_FRAME_SEC = 0.08  # FastConformer: 10 ms hop × 8× subsampling
LOG_EVERY = 50             # worker progress report interval (clips)


def log(*args):
    """Timestamped console log."""
    print(f"[{time.time() - T0:7.1f}s]", *args, flush=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
cands = glob.glob("/kaggle/input/**/Dataset_Final/train.csv", recursive=True)
ROOT  = os.path.dirname(cands[0])
df = pd.concat(
    [
        pd.read_csv(f"{ROOT}/train.csv").assign(split="train"),
        pd.read_csv(f"{ROOT}/test.csv").assign(split="test"),
    ],
    ignore_index=True,
)
df["path"] = [f"{ROOT}/{s}/{f}" for s, f in zip(df.split, df.filename)]
n_gpu = sum(
    1 for line in
    subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines()
    if line.startswith("GPU")
)
log(f"Clips: {len(df)} | GPUs: {n_gpu}")


# ---------------------------------------------------------------------------
# 1. Whisper large-v3 — verbatim (disfluent) prompt
# ---------------------------------------------------------------------------
WHISPER_WORKER = r'''
import json, sys
import soundfile as sf
from faster_whisper import WhisperModel

items  = [json.loads(l) for l in open(sys.argv[1])]
prompt = sys.argv[3]
model  = WhisperModel("large-v3", device="cuda", compute_type="float16")

with open(sys.argv[2], "w") as out:
    for i, it in enumerate(items):
        audio, sr = sf.read(it["path"], dtype="float32")
        segs, info = model.transcribe(
            audio, language="en", beam_size=5, word_timestamps=True,
            initial_prompt=prompt, condition_on_previous_text=True, vad_filter=False,
        )
        segs = list(segs)
        out.write(json.dumps({
            "split": it["split"], "filename": it["filename"],
            "duration": float(info.duration),
            "text": " ".join(s.text.strip() for s in segs).strip(),
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text,
                 "avg_logprob": s.avg_logprob, "no_speech_prob": s.no_speech_prob,
                 "compression_ratio": s.compression_ratio, "temperature": s.temperature}
                for s in segs
            ],
            "words": [[w.start, w.end, w.word, w.probability]
                      for s in segs for w in (s.words or [])],
        }) + "\n")
        out.flush()
        if i % 50 == 0:
            print("whisper-verbatim", sys.argv[2], i, flush=True)
'''

log("Installing faster-whisper ...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "faster-whisper"], check=True)

import site  # noqa: E402

lib_dirs: list = []
for sp in site.getsitepackages():
    lib_dirs += glob.glob(os.path.join(sp, "nvidia", "*", "lib"))
env = dict(os.environ, LD_LIBRARY_PATH=":".join(lib_dirs + [os.environ.get("LD_LIBRARY_PATH", "")]))

whisper_worker_path = f"{OUT}/whisper_verbatim_worker.py"
with open(whisper_worker_path, "w") as fh:
    fh.write(WHISPER_WORKER)

# Longest clips first, interleaved across GPUs for load balance.
order = df.assign(sz=[os.path.getsize(p) for p in df.path]).sort_values("sz", ascending=False)
procs, part_files = [], []
for g in range(n_gpu):
    items_path  = f"{OUT}/wv_items_{g}.jsonl"
    output_path = f"{OUT}/wv_part_{g}.jsonl"
    with open(items_path, "w") as fh:
        for r in order.iloc[g::n_gpu].itertuples():
            fh.write(json.dumps({"split": r.split, "filename": r.filename, "path": r.path}) + "\n")
    procs.append(subprocess.Popen(
        [sys.executable, whisper_worker_path, items_path, output_path, VERBATIM_PROMPT],
        env=dict(env, CUDA_VISIBLE_DEVICES=str(g)),
    ))
    part_files.append(output_path)

for p in procs:
    p.wait()
    log("Whisper verbatim worker exit:", p.returncode)

recs = [json.loads(l) for pf in part_files if os.path.exists(pf) for l in open(pf)]
with open(f"{OUT}/asr_whisper_large_v3_verbatim.jsonl", "w") as fh:
    for r in recs:
        fh.write(json.dumps(r) + "\n")

# Clean up temporary shard files.
for pattern in ("wv_items_*", "wv_part_*"):
    for p in glob.glob(f"{OUT}/{pattern}"):
        os.remove(p)
log(f"Whisper verbatim records: {len(recs)}")


# ---------------------------------------------------------------------------
# 2. Parakeet-CTC-1.1B — greedy CTC decode, no language model
# ---------------------------------------------------------------------------
import transformers  # noqa: E402

# Ensure a transformers release that supports Parakeet is installed.
if tuple(int(x) for x in transformers.__version__.split(".")[:2]) < (4, 57):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers"], check=True)
    import importlib
    importlib.reload(transformers)

import soundfile as sf  # noqa: E402
import torch            # noqa: E402
from transformers import AutoModelForCTC, AutoProcessor  # noqa: E402

MODEL_ID = "nvidia/parakeet-ctc-1.1b"
log("transformers", transformers.__version__)
processor = AutoProcessor.from_pretrained(MODEL_ID)
ctc_model = AutoModelForCTC.from_pretrained(MODEL_ID, torch_dtype=torch.float16).cuda().eval()
vocab     = processor.tokenizer.convert_ids_to_tokens(list(range(ctc_model.config.vocab_size)))
blank_id  = None


def ctc_decode(logits: torch.Tensor) -> dict:
    """Greedy CTC decode with word timings and confidences.

    '▁' marks a word-boundary token (SentencePiece convention).
    """
    global blank_id
    conf, ids = logits.float().softmax(-1).max(-1)
    ids_np, conf_np = ids.cpu().numpy(), conf.cpu().numpy()

    if blank_id is None:
        # The blank token dominates the argmax path; find it empirically.
        blank_id = int(np.bincount(ids_np).argmax())
        log("CTC blank id:", blank_id)

    words, cur, prev = [], None, -1
    for t, (tok_id, c) in enumerate(zip(ids_np, conf_np)):
        if tok_id == prev or tok_id == blank_id:
            prev = tok_id
            continue
        prev  = tok_id
        piece = vocab[tok_id] if tok_id < len(vocab) else ""
        if piece.startswith("▁") or cur is None:
            if cur:
                words.append(cur)
            cur = {"w": piece.lstrip("▁"), "s": t, "e": t, "c": [float(c)]}
        else:
            cur["w"] += piece
            cur["e"]  = t
            cur["c"].append(float(c))
    if cur:
        words.append(cur)

    nonblank = conf_np[ids_np != blank_id]
    return {
        "text":  " ".join(w["w"] for w in words if w["w"]).strip(),
        "words": [
            [w["s"] * PARAKEET_FRAME_SEC, (w["e"] + 1) * PARAKEET_FRAME_SEC,
             w["w"], float(np.mean(w["c"]))]
            for w in words if w["w"]
        ],
        "frame_conf_mean": float(nonblank.mean())          if len(nonblank) else 0.0,
        "frame_conf_p10":  float(np.percentile(nonblank, 10)) if len(nonblank) else 0.0,
        "blank_frac":      float((ids_np == blank_id).mean()),
    }


with open(f"{OUT}/asr_parakeet_ctc_1.1b.jsonl", "w") as fh:
    for k, r in enumerate(df.itertuples()):
        x, sr = sf.read(r.path, dtype="float32")
        feats  = processor(x, sampling_rate=16000, return_tensors="pt")
        inputs = {
            n: (v.cuda().half() if v.dtype.is_floating_point else v.cuda())
            for n, v in feats.items()
        }
        with torch.no_grad():
            logits = ctc_model(**inputs).logits[0]
        fh.write(json.dumps({
            "split": r.split, "filename": r.filename,
            "duration": len(x) / sr,
            **ctc_decode(logits),
        }) + "\n")
        if k % 100 == 0:
            log("Parakeet:", k)

log("Done.")
