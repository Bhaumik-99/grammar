"""SHL Grammar Scoring 2026 - LLM-derived text features from ASR transcripts.

Model: Qwen/Qwen3-8B (Apache-2.0) in fp16, split over 2x T4 with HF transformers.
For the clean Whisper transcript (asr_whisper_large_v3.jsonl; the two error-preserving
views are processed by shl-2026-text-llm-features-views):
  1. hidden   - mean-pooled hidden states of selected decoder layers
                (frozen-representation probes; middle layers carry most signal)
  2. judge    - grammar score read from the next-token distribution over a
                half-step 1-9 scale (1 = rubric 1.0 ... 9 = rubric 5.0), no
                chain-of-thought: zero-shot, and with 8 human-scored anchors
                retrieved from OTHER folds only (fold-aware, no label leakage)
  3. embed    - Qwen/Qwen3-Embedding-4B (Apache-2.0) sentence embeddings
  4. gec      - minimal-edit grammatical correction of the clean Whisper view
                (edit-rate features are computed in the final notebook)
  5. ppl      - per-token surprisal statistics of each transcript
Outputs (/kaggle/working): llmfeat_<view>.npz, judge_<view>.csv, ppl_<view>.csv,
qwen3emb_<view>.npz, gec_whisper_large_v3.jsonl, anchor_folds.csv
"""
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch

T0 = time.time()
OUT = "/kaggle/working"
LLM = "Qwen/Qwen3-8B"
EMB = "Qwen/Qwen3-Embedding-4B"
LAYERS = [8, 12, 16, 20, 24, 28, 32, 36]
N_ANCHORS_NN, BANDS = 4, [(2.0, 2.5), (3.0, 3.5), (4.0, 4.5), (5.0, 5.0)]


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def find(pattern):
    return sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))


comp = os.path.dirname(find("Dataset_Final/train.csv")[0])
labels = pd.read_csv(f"{comp}/train.csv")
stats = pd.read_csv(find("audio_stats.csv")[0])
noise = set(map(tuple, stats.loc[(stats.dyn_range_db < 5) & (stats.peak > 0.9), ["split", "filename"]].values))
VIEW_FILES = ["asr_whisper_large_v3.jsonl"]
views = {f[4:-6]: find(f)[0] for f in VIEW_FILES if find(f)}  # final transcript views only
log("views:", list(views))


def load_view(path):
    recs = [json.loads(l) for l in open(path)]
    d = pd.DataFrame([{"split": r["split"], "filename": r["filename"],
                       "text": (r.get("text") or "").strip() or "(no speech recognised)"} for r in recs])
    return d.merge(labels, on="filename", how="left").assign(
        label=lambda x: np.where(x.split == "train", x.label, np.nan))


# ---------------------------------------------------------------- fold-aware anchor pools
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

base = load_view(next(iter(views.values())))
is_tr = (base.split == "train") & ~base.apply(lambda r: (r.split, r.filename) in noise, axis=1)
tr_idx = np.flatnonzero(is_tr)
# pseudo-speaker groups (low-layer WavLM statistics) -> anchors never come from the target's own speaker
zw = np.load(find("emb_wavlm_large.npz")[0], allow_pickle=True)
Vw = np.hstack([zw["mean"][:, 3:7].reshape(len(zw["mean"]), -1), zw["std"][:, 3:7].reshape(len(zw["std"]), -1)])
Vw = Vw.astype(np.float64)
Vw = (Vw - Vw.mean(0)) / (Vw.std(0) + 1e-8)
Vw /= np.linalg.norm(Vw, axis=1, keepdims=True)
Sw = Vw @ Vw.T
np.fill_diagonal(Sw, -1)
thr = float(np.percentile(Sw[np.ix_(zw["split"] == "test", zw["split"] == "train")].max(1), 99.5))
_, comp = connected_components(csr_matrix(Sw > thr), directed=False)
spk_map = dict(zip(zip(zw["split"], zw["filename"]), comp))
speaker = np.array([spk_map[(s_, f_)] for s_, f_ in zip(base.split, base.filename)])
fold = np.full(len(base), -1)
strat = np.maximum(np.round(base.label.values[tr_idx] * 2).astype(int), 4)
skf = StratifiedGroupKFold(5, shuffle=True, random_state=42)
for k, (_, v) in enumerate(skf.split(tr_idx, strat, speaker[tr_idx])):
    fold[tr_idx[v]] = k
base.assign(anchor_fold=fold)[["split", "filename", "anchor_fold"]].to_csv(f"{OUT}/anchor_folds.csv", index=False)

tok = None


def chat(user):
    msgs = [{"role": "system", "content": "You are an experienced examiner of spoken English proficiency."},
            {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


# The run on Kaggle used the competition's 1-5 grammar rubric here (one line per level, from the competition's data
# description page). It is not reproduced in this public copy; paste it in to reproduce the judge features exactly.
RUBRIC = """Grammar score rubric (half points allowed):
<levels 1-5 of the competition's grammar rubric>"""
CONTEXT = ("Each transcript is an automatic speech-recognition transcript of a 45-60 second spontaneous spoken "
           "English response. Ignore punctuation, capitalisation and likely transcription noise; hesitations "
           "and fillers are not errors by themselves, self-corrections are a sign of control, and sentences "
           "left incomplete count as errors. Judge only the speaker's grammar.")
SCALE = ("Answer with a single digit on this scale: 1 = 1.0, 2 = 1.5, 3 = 2.0, 4 = 2.5, 5 = 3.0, "
         "6 = 3.5, 7 = 4.0, 8 = 4.5, 9 = 5.0. Answer with the digit only.")


def clip_words(t, n=110):
    w = t.split()
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")


def judge_prompts(view_df):
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(view_df.text)
    X = tfidf.transform(view_df.text)
    zero, few, anchor_log = [], [], []
    for i, r in view_df.iterrows():
        t = clip_words(r.text, 160)
        zero.append(chat(f"{CONTEXT}\n\n{RUBRIC}\n\nTranscript:\n\"\"\"{t}\"\"\"\n\n{SCALE}"))
        pool = tr_idx[fold[tr_idx] != fold[i]] if fold[i] >= 0 else tr_idx
        pool = pool[pool != i]
        sims = (X[pool] @ X[i].T).toarray().ravel()
        chosen = list(pool[np.argsort(-sims)[:N_ANCHORS_NN]])
        rng = np.random.default_rng(i)
        for lo, hi in BANDS:
            cand = [j for j in pool if lo <= view_df.label.values[j] <= hi and j not in chosen]
            if cand:
                chosen.append(int(rng.choice(cand)))
        rng.shuffle(chosen)
        ex = "\n\n".join(f"Example {n + 1} (human grammar score {view_df.label.values[j]:.1f}):\n"
                         f"\"\"\"{clip_words(view_df.text.values[j])}\"\"\"" for n, j in enumerate(chosen))
        few.append(chat(f"{CONTEXT}\n\n{RUBRIC}\n\nScored examples from the same test:\n\n{ex}\n\n"
                        f"Now score this transcript.\nTranscript:\n\"\"\"{t}\"\"\"\n\n{SCALE}"))
        anchor_log.append(";".join(f"{view_df.split.values[j]}/{view_df.filename.values[j]}" for j in chosen))
    return zero, few, anchor_log


@torch.no_grad()
def next_token_dist(model, prompts, digit_ids, bs=4):
    out = []
    for s in range(0, len(prompts), bs):
        enc = tok(prompts[s:s + bs], return_tensors="pt", padding=True).to(model.device)
        logits = model(**enc, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
        full = logits.softmax(-1)
        sub = full[:, digit_ids]
        out.append(torch.cat([sub / sub.sum(-1, keepdim=True), sub.sum(-1, keepdim=True)], -1).cpu().numpy())
    return np.concatenate(out)


def dist_features(P, name):
    scores = 1.0 + 0.5 * np.arange(9)
    p, mass = P[:, :9], P[:, 9]
    e = p @ scores
    sd = np.sqrt(np.clip(p @ scores ** 2 - e ** 2, 0, None))
    f = {f"{name}_exp": e, f"{name}_sd": sd, f"{name}_p_le2": p[:, :3].sum(1),
         f"{name}_p_ge45": p[:, 7:].sum(1), f"{name}_mass": mass, f"{name}_argmax": scores[p.argmax(1)]}
    for k in range(9):
        f[f"{name}_p{k + 1}"] = p[:, k]
    return f


# ---------------------------------------------------------------- 1+2. Qwen3-8B: hidden states + judge
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(LLM)
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(LLM, torch_dtype=torch.float16, device_map="auto",
                                             attn_implementation="sdpa").eval()
digit_ids = [tok.convert_tokens_to_ids(str(d)) for d in range(1, 10)]
assert len(set(digit_ids)) == 9 and None not in digit_ids, digit_ids
log("loaded", LLM, "digit ids", digit_ids)

for vname, vpath in views.items():
    vdf = load_view(vpath)
    assert (vdf.filename.values == base.filename.values).all() and (vdf.split.values == base.split.values).all()
    # hidden states: mean over transcript tokens only
    hs = np.zeros((len(vdf), len(LAYERS), model.config.hidden_size), np.float16)
    ppl_rows = []
    prefix = "Transcript of a spoken English response:\n"
    for s in range(0, len(vdf), 8):
        texts = [prefix + t for t in vdf.text.values[s:s + 8]]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False)
        n_pre = len(tok(prefix).input_ids)
        m = enc.attention_mask.clone()
        for b in range(m.shape[0]):  # drop prefix tokens (left padded)
            first = int((m[b] == 1).nonzero()[0])
            m[b, first:first + n_pre] = 0
        # per-token surprisal of the transcript (scripted / AI-like text is very predictable,
        # learner errors produce surprisal spikes)
        for b in range(m.shape[0]):
            with torch.no_grad():
                lp = torch.log_softmax(out.logits[b, :-1].float(), -1)
                nll = -lp.gather(-1, enc.input_ids[b, 1:, None].to(lp.device)).squeeze(-1)
            v = nll[m[b, 1:].bool().to(nll.device)].cpu().numpy()
            if len(v) == 0:
                v = np.array([0.0])
            ppl_rows.append({"split": vdf.split.values[s + b], "filename": vdf.filename.values[s + b],
                             "nll_mean": float(v.mean()), "nll_sd": float(v.std()),
                             "nll_p90": float(np.percentile(v, 90)), "nll_max": float(v.max()),
                             "nll_frac_low": float((v < 0.5).mean()), "nll_frac_high": float((v > 6).mean())})
        m = m.unsqueeze(-1).float()
        for li, L in enumerate(LAYERS):
            h = out.hidden_states[L].float()
            hs[s:s + 8, li] = ((h * m.to(h.device)).sum(1) / m.to(h.device).sum(1).clamp(min=1)).cpu().numpy()
        del out
    pd.DataFrame(ppl_rows).to_csv(f"{OUT}/ppl_{vname}.csv", index=False)
    log(vname, "hidden states + perplexity done")
    zero, few, anchors = judge_prompts(vdf)
    feats = {**dist_features(next_token_dist(model, zero, digit_ids), "judge0"),
             **dist_features(next_token_dist(model, few, digit_ids, bs=2), "judgefs")}
    jd = pd.DataFrame({"split": vdf.split, "filename": vdf.filename, **feats, "anchors": anchors})
    jd.to_csv(f"{OUT}/judge_{vname}.csv", index=False)
    tr = jd.merge(labels, on="filename").query("split == 'train'")
    tr = tr[~tr.apply(lambda r: (r.split, r.filename) in noise, axis=1)]
    for c in ("judge0_exp", "judgefs_exp"):
        log(vname, c, "pearson", round(float(np.corrcoef(tr[c], tr.label)[0, 1]), 4))
    np.savez_compressed(f"{OUT}/llmfeat_{vname}.npz", split=vdf.split.values, filename=vdf.filename.values,
                        layers=np.array(LAYERS), hidden=hs)

# ---------------------------------------------------------------- 4. minimal-edit GEC (clean view only)
GEC = ("Rewrite the transcript with the MINIMAL edits needed to make each sentence grammatically correct "
       "standard English. Keep the speaker's words, meaning and order; do not paraphrase or improve style; "
       "drop fillers (um, uh) and abandoned false starts. Output only the corrected text.")
gec_view = "whisper_large_v3" if "whisper_large_v3" in views else next(iter(views))
vdf = load_view(views[gec_view])
gec_out = []
for s in range(0, len(vdf), 16):
    prompts = [chat(f"{CONTEXT}\n\nTranscript:\n\"\"\"{t}\"\"\"\n\n{GEC}") for t in vdf.text.values[s:s + 16]]
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=320, do_sample=False, pad_token_id=tok.pad_token_id)
    gec_out += tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    if s % 160 == 0:
        log("gec", s)
with open(f"{OUT}/gec_{gec_view}.jsonl", "w") as f:
    for r, g in zip(vdf.itertuples(), gec_out):
        f.write(json.dumps({"split": r.split, "filename": r.filename, "text": r.text, "gec": g.strip()}) + "\n")
del model
torch.cuda.empty_cache()
log("GEC done")

# ---------------------------------------------------------------- 3. Qwen3-Embedding-4B
etok = AutoTokenizer.from_pretrained(EMB, padding_side="left")
emodel = AutoModel.from_pretrained(EMB, torch_dtype=torch.float16, device_map="auto").eval()
instr = "Instruct: Represent this spoken English response for grading the speaker's grammar\nQuery: "
for vname, vpath in views.items():
    vdf = load_view(vpath)
    embs = []
    for s in range(0, len(vdf), 16):
        enc = etok([instr + t for t in vdf.text.values[s:s + 16]], return_tensors="pt", padding=True,
                   truncation=True, max_length=1024).to(emodel.device)
        with torch.no_grad():
            h = emodel(**enc).last_hidden_state[:, -1].float()  # last-token pooling (left padded)
        embs.append(torch.nn.functional.normalize(h, dim=-1).cpu().numpy())
    np.savez_compressed(f"{OUT}/qwen3emb_{vname}.npz", split=vdf.split.values, filename=vdf.filename.values,
                        emb=np.concatenate(embs).astype(np.float16))
    log(vname, "embedding done")
log("done")
