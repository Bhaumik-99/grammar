"""SHL Grammar Scoring 2026 - error-preserving ASR views.

Whisper's decoder behaves like a strong language model and silently "repairs"
learner grammar, which caps every transcript-based grammar feature. This
notebook adds two transcript views that keep more of what was actually said:

  1. whisper_large_v3_verbatim  faster-whisper large-v3 conditioned on a
     deliberately disfluent, ungrammatical initial prompt (keeps fillers,
     repetitions and errors), word timestamps + probabilities.
  2. parakeet_ctc_1.1b          NVIDIA Parakeet-CTC-1.1B (CC-BY-4.0): greedy CTC
     decoding with no language model, word timings + frame confidences.

Outputs (/kaggle/working): asr_whisper_large_v3_verbatim.jsonl,
asr_parakeet_ctc_1.1b.jsonl - one record per (split, filename).
"""
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

T0 = time.time()
OUT = "/kaggle/working"


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


cands = glob.glob("/kaggle/input/**/Dataset_Final/train.csv", recursive=True)
ROOT = os.path.dirname(cands[0])
df = pd.concat([pd.read_csv(f"{ROOT}/train.csv").assign(split="train"),
                pd.read_csv(f"{ROOT}/test.csv").assign(split="test")], ignore_index=True)
df["path"] = [f"{ROOT}/{s}/{f}" for s, f in zip(df.split, df.filename)]
n_gpu = len([l for l in subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines()
             if l.startswith("GPU")])
log("clips", len(df), "gpus", n_gpu)

# ---------------------------------------------------------------- 1. Whisper large-v3, disfluent prompt
VERBATIM_PROMPT = ("Umm, so, uh, I I go to the market yesterday and, and I buyed some, uh, vegetable. "
                   "Then I, I am going home and... hmm, like, my mother she cook the food, no, she cooked it.")

WHISPER_WORKER = r'''
import json, sys
import soundfile as sf
from faster_whisper import WhisperModel
items = [json.loads(l) for l in open(sys.argv[1])]
prompt = sys.argv[3]
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
with open(sys.argv[2], "w") as out:
    for i, it in enumerate(items):
        audio, sr = sf.read(it["path"], dtype="float32")
        segs, info = model.transcribe(audio, language="en", beam_size=5, word_timestamps=True,
                                      initial_prompt=prompt, condition_on_previous_text=True,
                                      vad_filter=False)
        segs = list(segs)
        out.write(json.dumps({
            "split": it["split"], "filename": it["filename"], "duration": float(info.duration),
            "text": " ".join(s.text.strip() for s in segs).strip(),
            "segments": [{"start": s.start, "end": s.end, "text": s.text, "avg_logprob": s.avg_logprob,
                          "no_speech_prob": s.no_speech_prob, "compression_ratio": s.compression_ratio,
                          "temperature": s.temperature} for s in segs],
            "words": [[w.start, w.end, w.word, w.probability] for s in segs for w in (s.words or [])],
        }) + "\n"); out.flush()
        if i % 50 == 0:
            print("whisper-verbatim", sys.argv[2], i, flush=True)
'''

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "faster-whisper"], check=True)
import site  # noqa: E402

lib_dirs = []
for sp in site.getsitepackages():
    lib_dirs += glob.glob(os.path.join(sp, "nvidia", "*", "lib"))
env = dict(os.environ, LD_LIBRARY_PATH=":".join(lib_dirs + [os.environ.get("LD_LIBRARY_PATH", "")]))
with open(f"{OUT}/whisper_worker.py", "w") as f:
    f.write(WHISPER_WORKER)
order = df.assign(sz=[os.path.getsize(p) for p in df.path]).sort_values("sz", ascending=False)
procs, parts = [], []
for g in range(n_gpu):
    pf, of = f"{OUT}/wv_items_{g}.jsonl", f"{OUT}/wv_part_{g}.jsonl"
    with open(pf, "w") as f:
        for r in order.iloc[g::n_gpu].itertuples():
            f.write(json.dumps({"split": r.split, "filename": r.filename, "path": r.path}) + "\n")
    procs.append(subprocess.Popen([sys.executable, f"{OUT}/whisper_worker.py", pf, of, VERBATIM_PROMPT],
                                  env=dict(env, CUDA_VISIBLE_DEVICES=str(g))))
    parts.append(of)
for p in procs:
    p.wait()
    log("whisper worker exit", p.returncode)
recs = [json.loads(l) for of in parts if os.path.exists(of) for l in open(of)]
with open(f"{OUT}/asr_whisper_large_v3_verbatim.jsonl", "w") as f:
    for r in recs:
        f.write(json.dumps(r) + "\n")
for pattern in ("wv_items_*", "wv_part_*"):
    for p in glob.glob(f"{OUT}/{pattern}"):
        os.remove(p)
log("whisper verbatim records", len(recs))

# ---------------------------------------------------------------- 2. Parakeet-CTC-1.1B (no LM)
import transformers  # noqa: E402

if tuple(int(x) for x in transformers.__version__.split(".")[:2]) < (4, 57):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers"], check=True)
    import importlib

    importlib.reload(transformers)
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCTC, AutoProcessor  # noqa: E402

MODEL = "nvidia/parakeet-ctc-1.1b"
FRAME_SEC = 0.08  # FastConformer: 10 ms hop x 8 subsampling
log("transformers", transformers.__version__)
processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForCTC.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()
vocab = processor.tokenizer.convert_ids_to_tokens(list(range(model.config.vocab_size)))
blank_id = None


def ctc_decode(logits):
    """Greedy CTC decode with word timings and confidences ('▁' marks a word start)."""
    global blank_id
    conf, ids = logits.float().softmax(-1).max(-1)
    ids, conf = ids.cpu().numpy(), conf.cpu().numpy()
    if blank_id is None:  # blank dominates the argmax path
        blank_id = int(np.bincount(ids).argmax())
        log("blank id", blank_id)
    words, cur, prev = [], None, -1
    for t, (i, c) in enumerate(zip(ids, conf)):
        if i == prev or i == blank_id:
            prev = i
            continue
        prev = i
        piece = vocab[i] if i < len(vocab) else ""
        if piece.startswith("▁") or cur is None:
            if cur:
                words.append(cur)
            cur = {"w": piece.lstrip("▁"), "s": t, "e": t, "c": [float(c)]}
        else:
            cur["w"] += piece
            cur["e"] = t
            cur["c"].append(float(c))
    if cur:
        words.append(cur)
    nonblank = conf[ids != blank_id]
    return {
        "text": " ".join(w["w"] for w in words if w["w"]).strip(),
        "words": [[w["s"] * FRAME_SEC, (w["e"] + 1) * FRAME_SEC, w["w"], float(np.mean(w["c"]))]
                  for w in words if w["w"]],
        "frame_conf_mean": float(nonblank.mean()) if len(nonblank) else 0.0,
        "frame_conf_p10": float(np.percentile(nonblank, 10)) if len(nonblank) else 0.0,
        "blank_frac": float((ids == blank_id).mean()),
    }


with open(f"{OUT}/asr_parakeet_ctc_1.1b.jsonl", "w") as f:
    for k, r in enumerate(df.itertuples()):
        x, sr = sf.read(r.path, dtype="float32")
        feats = processor(x, sampling_rate=16000, return_tensors="pt")
        inputs = {n: (v.cuda().half() if v.dtype.is_floating_point else v.cuda()) for n, v in feats.items()}
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        f.write(json.dumps({"split": r.split, "filename": r.filename, "duration": len(x) / sr,
                            **ctc_decode(logits)}) + "\n")
        if k % 100 == 0:
            log("parakeet", k)
log("done")
