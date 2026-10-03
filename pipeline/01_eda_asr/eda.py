"""SHL Grammar Scoring 2026 - audio EDA + Whisper large-v3 transcription.

Runs on Kaggle GPU. Produces in /kaggle/working:
  audio_stats.csv            per-clip format / loudness / silence statistics
  asr_whisper_large_v3.jsonl per-clip transcript, segments, word timestamps,
                             language-ID probabilities
  eda_*.png                  diagnostic plots
Train and test reuse the same file names for different recordings, so every
record is keyed by (split, filename).
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


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


OUT = "/kaggle/working"

# ---------------------------------------------------------------- data
cands = glob.glob("/kaggle/input/**/Dataset_Final/train.csv", recursive=True)
assert cands, f"train.csv not found; /kaggle/input has {os.listdir('/kaggle/input')}"
ROOT = os.path.dirname(cands[0])
log("DATA ROOT", ROOT)
tr = pd.read_csv(f"{ROOT}/train.csv").assign(split="train")
te = pd.read_csv(f"{ROOT}/test.csv").assign(split="test", label=np.nan)
df = pd.concat([tr, te], ignore_index=True)
df["path"] = [f"{ROOT}/{s}/{f}" for s, f in zip(df.split, df.filename)]
missing = [p for p in df.path if not os.path.exists(p)]
assert not missing, missing[:5]
log("clips:", df.split.value_counts().to_dict())

# ---------------------------------------------------------------- ASR workers (start first, GPU-bound)
WORKER = r'''
import json, sys, time
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

items = [json.loads(l) for l in open(sys.argv[1])]
out = open(sys.argv[2], "w")
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
t0 = time.time()
for i, it in enumerate(items):
    # All clips are 16 kHz mono PCM_16; read directly (Kaggle's PyAV breaks decode_audio).
    audio, sr = sf.read(it["path"], dtype="float32")
    assert sr == 16000 and audio.ndim == 1, (sr, audio.shape)
    lang = None
    try:
        code, prob, allp = model.detect_language(audio)
        top = sorted(allp, key=lambda x: -x[1])[:5]
        lang = {"lang": code, "prob": float(prob), "top5": [[c, float(p)] for c, p in top]}
    except Exception as e:  # older faster-whisper without detect_language
        lang = {"error": repr(e)}
    segs, info = model.transcribe(
        audio, language="en", beam_size=5, word_timestamps=True,
        condition_on_previous_text=False, vad_filter=False)
    segs = list(segs)
    rec = {
        "split": it["split"], "filename": it["filename"],
        "duration": float(info.duration), "langid": lang,
        "text": " ".join(s.text.strip() for s in segs).strip(),
        "segments": [{"start": s.start, "end": s.end, "text": s.text,
                      "avg_logprob": s.avg_logprob, "no_speech_prob": s.no_speech_prob,
                      "compression_ratio": s.compression_ratio, "temperature": s.temperature}
                     for s in segs],
        "words": [[w.start, w.end, w.word, w.probability] for s in segs for w in (s.words or [])],
    }
    out.write(json.dumps(rec) + "\n"); out.flush()
    if i % 25 == 0:
        print(f"worker {sys.argv[2]}: {i+1}/{len(items)} done, {time.time()-t0:.0f}s", flush=True)
out.close()
'''


def gpu_count():
    try:
        q = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        log(q.stdout.strip())
        return len([l for l in q.stdout.splitlines() if l.startswith("GPU")])
    except FileNotFoundError:
        return 0


n_gpu = gpu_count()
assert n_gpu > 0, "GPU required"
log("installing faster-whisper")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "faster-whisper"], check=True)

# ctranslate2 needs the cuBLAS / cuDNN shared libraries shipped in nvidia-* wheels.
import site  # noqa: E402

lib_dirs = []
for sp in site.getsitepackages():
    lib_dirs += glob.glob(os.path.join(sp, "nvidia", "*", "lib"))
env = dict(os.environ)
env["LD_LIBRARY_PATH"] = ":".join(lib_dirs + [env.get("LD_LIBRARY_PATH", "")])

with open(f"{OUT}/asr_worker.py", "w") as f:
    f.write(WORKER)
# Longest clips first, interleaved across GPUs for balance.
order = df.assign(sz=[os.path.getsize(p) for p in df.path]).sort_values("sz", ascending=False)
procs, parts = [], []
for g in range(n_gpu):
    part = order.iloc[g::n_gpu]
    pf, of = f"{OUT}/asr_items_{g}.jsonl", f"{OUT}/asr_part_{g}.jsonl"
    with open(pf, "w") as f:
        for r in part.itertuples():
            f.write(json.dumps({"split": r.split, "filename": r.filename, "path": r.path}) + "\n")
    e = dict(env, CUDA_VISIBLE_DEVICES=str(g))
    procs.append(subprocess.Popen([sys.executable, f"{OUT}/asr_worker.py", pf, of], env=e))
    parts.append(of)
log(f"launched {n_gpu} ASR workers")

# ---------------------------------------------------------------- audio statistics (CPU, overlaps ASR)
import soundfile as sf  # noqa: E402


def frame_db(x, sr, win_s=0.025, hop_s=0.010):
    win, hop = int(win_s * sr), int(hop_s * sr)
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt((x[idx] ** 2).mean(1) + 1e-12)
    return 20 * np.log10(rms + 1e-12), hop_s


rows = []
for r in df.itertuples():
    info = sf.info(r.path)
    x, sr = sf.read(r.path, dtype="float32", always_2d=True)
    mono = x.mean(1)
    db, hop_s = frame_db(mono, sr)
    p10, p50, p90 = np.percentile(db, [10, 50, 90])
    thr = max(p90 - 25.0, -55.0)  # adaptive energy threshold
    active = db > thr
    act_idx = np.flatnonzero(active)
    lead = act_idx[0] * hop_s if len(act_idx) else len(db) * hop_s
    trail = (len(db) - 1 - act_idx[-1]) * hop_s if len(act_idx) else len(db) * hop_s
    # pauses = inactive runs >= 300 ms between first and last active frame
    pauses = []
    if len(act_idx):
        seg = ~active[act_idx[0]:act_idx[-1] + 1]
        run = 0
        for v in np.append(seg, False):
            if v:
                run += 1
            else:
                if run * hop_s >= 0.3:
                    pauses.append(run * hop_s)
                run = 0
    rows.append({
        "split": r.split, "filename": r.filename, "label": r.label,
        "sr": info.samplerate, "channels": info.channels, "subtype": info.subtype,
        "format": info.format, "duration": info.frames / info.samplerate,
        "rms_dbfs": float(20 * np.log10(np.sqrt((mono ** 2).mean()) + 1e-12)),
        "peak": float(np.abs(mono).max()), "clip_frac": float((np.abs(mono) > 0.999).mean()),
        "noise_floor_db": float(p10), "median_db": float(p50), "speech_level_db": float(p90),
        "dyn_range_db": float(p90 - p10), "active_ratio": float(active.mean()),
        "active_sec": float(active.sum() * hop_s), "lead_sil": float(lead), "trail_sil": float(trail),
        "n_pauses": len(pauses), "pause_total": float(sum(pauses)),
        "pause_max": float(max(pauses) if pauses else 0.0),
    })
stats = pd.DataFrame(rows)
stats.to_csv(f"{OUT}/audio_stats.csv", index=False)
log("audio stats done")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
print(stats.groupby("split")[["sr", "channels"]].agg(lambda s: s.value_counts().to_dict()))
print(stats.groupby("split").subtype.value_counts())
print(stats.groupby("split")[["duration", "rms_dbfs", "active_ratio", "active_sec", "dyn_range_db",
                              "n_pauses", "pause_total", "lead_sil", "trail_sil"]].describe().T)
trs = stats[stats.split == "train"].copy()
print("\nPer-label means (train):")
print(trs.groupby("label")[["duration", "active_sec", "active_ratio", "rms_dbfs", "dyn_range_db",
                            "n_pauses", "pause_total"]].mean().round(2))
num = ["duration", "active_sec", "active_ratio", "rms_dbfs", "dyn_range_db", "noise_floor_db",
       "n_pauses", "pause_total", "pause_max", "lead_sil", "trail_sil", "clip_frac"]
print("\nSpearman corr with label (train):")
print(trs[num + ["label"]].corr(method="spearman")["label"].drop("label").sort_values())

# ---------------------------------------------------------------- wait for ASR
for p in procs:
    p.wait()
    log("worker exit code", p.returncode)
recs = []
for of in parts:
    if os.path.exists(of):
        recs += [json.loads(l) for l in open(of)]
log(f"ASR records: {len(recs)} / {len(df)}")
assert recs, "ASR produced no records - see worker tracebacks above"
with open(f"{OUT}/asr_whisper_large_v3.jsonl", "w") as f:
    for rec in recs:
        f.write(json.dumps(rec) + "\n")

asr = pd.DataFrame([{
    "split": r["split"], "filename": r["filename"], "text": r["text"],
    "n_words": len(r["words"]),
    "lang": (r["langid"] or {}).get("lang"), "lang_prob": (r["langid"] or {}).get("prob"),
    "mean_word_prob": float(np.mean([w[3] for w in r["words"]])) if r["words"] else np.nan,
    "mean_avg_logprob": float(np.mean([s["avg_logprob"] for s in r["segments"]])) if r["segments"] else np.nan,
    "max_no_speech": float(max([s["no_speech_prob"] for s in r["segments"]])) if r["segments"] else np.nan,
    "max_compression": float(max([s["compression_ratio"] for s in r["segments"]])) if r["segments"] else np.nan,
} for r in recs])
m = stats.merge(asr, on=["split", "filename"], how="left")
m["wpm"] = m.n_words / (m.duration / 60)
m["wps_active"] = m.n_words / m.active_sec.clip(lower=1)
m.drop(columns=["text"]).to_csv(f"{OUT}/eda_merged.csv", index=False)

print("\nLanguage ID by split:")
print(m.groupby("split").lang.value_counts().head(20))
mt = m[m.split == "train"]
print("\nPer-label ASR stats (train):")
print(mt.groupby("label")[["n_words", "wpm", "wps_active", "mean_word_prob", "mean_avg_logprob",
                          "max_no_speech", "lang_prob"]].mean().round(3))
print("\nSpearman corr with label (ASR features, train):")
print(mt[["n_words", "wpm", "wps_active", "mean_word_prob", "mean_avg_logprob", "max_no_speech",
          "lang_prob", "label"]].corr(method="spearman")["label"].drop("label").sort_values())
print("\nnon-English language ID (all):")
print(m[m.lang != "en"][["split", "filename", "label", "lang", "lang_prob", "n_words", "duration"]]
      .to_string())

# Transcript samples (private notebook; competition data stays on Kaggle)
txt = m.set_index(["split", "filename"]).text
print("\n=== ALL label-0 train clips ===")
for r in mt[mt.label == 0].itertuples():
    print(f"- {r.filename} dur={r.duration:.1f}s active={r.active_sec:.1f}s words={r.n_words} "
          f"lang={r.lang}:{(r.lang_prob or 0):.2f} :: {str(txt[('train', r.filename)])[:300]}")
print("\n=== 2 samples per label ===")
for lab, g in mt[mt.label > 0].groupby("label"):
    for r in g.sample(min(2, len(g)), random_state=0).itertuples():
        print(f"[{lab}] {r.filename} dur={r.duration:.1f}s words={r.n_words} :: "
              f"{str(txt[('train', r.filename)])[:600]}")
print("\n=== 5 random test clips ===")
for r in m[m.split == "test"].sample(5, random_state=0).itertuples():
    print(f"[test] {r.filename} dur={r.duration:.1f}s words={r.n_words} :: "
          f"{str(txt[('test', r.filename)])[:400]}")
print("\n=== shortest 10 clips (any split) ===")
print(m.nsmallest(10, "duration")[["split", "filename", "label", "duration", "active_sec", "n_words"]]
      .to_string())

# ---------------------------------------------------------------- plots
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(2, 3, figsize=(16, 9))
mt.label.value_counts().sort_index().plot.bar(ax=ax[0, 0], title="Train label distribution")
for s, g in m.groupby("split"):
    ax[0, 1].hist(g.duration, bins=40, alpha=0.6, label=s, density=True)
ax[0, 1].set_title("Duration (s) by split")
ax[0, 1].legend()
for s, g in m.groupby("split"):
    ax[0, 2].hist(g.wpm.dropna(), bins=40, alpha=0.6, label=s, density=True)
ax[0, 2].set_title("Words per minute by split")
ax[0, 2].legend()
for a, col in zip([ax[1, 0], ax[1, 1], ax[1, 2]], ["n_words", "mean_word_prob", "active_sec"]):
    a.scatter(mt[col], mt.label + np.random.uniform(-0.12, 0.12, len(mt)), s=8, alpha=0.5)
    a.set_xlabel(col)
    a.set_ylabel("label (jittered)")
    a.set_title(f"label vs {col}")
plt.tight_layout()
plt.savefig(f"{OUT}/eda_overview.png", dpi=110)
log("done")
