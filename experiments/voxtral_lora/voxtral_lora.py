"""SHL Grammar Scoring 2026 - LoRA-tuned Voxtral-Mini-3B grammar regressor.

Frozen Voxtral audio-token states are our strongest single view (speaker-grouped
CV RMSE 0.542 with ridge). Here the speech-LLM itself is adapted: LoRA adapters on
the attention projections of the first 16 language-model layers (the band where
the frozen probe peaks) plus a linear head on the audio-token mean, trained with
MSE on standardised grades.

Protocol (no leaderboard information is used anywhere):
  * speaker-grouped StratifiedGroupKFold (5 folds), pseudo-speakers rebuilt with
    the same WavLM-layer-3-6 recipe as the final notebook;
  * fixed hyper-parameters chosen a priori, fixed number of epochs, no early
    stopping on the validation fold (keeps the OOF predictions honest);
  * the 37 synthetic-noise clips (label 0) are excluded, as in all other models.
Speed: the audio encoder is frozen, so its projected embeddings are computed once
and cached; training then runs only the truncated 16-layer language model.
Audio is capped at 60.0 s and the head pools only tokens that encode real audio
(Voxtral pads every 30-s chunk; padding tokens would add a duration-dependent bias).

Inputs (attached outputs): shl-2026-eda-asr (audio_stats.csv),
                           shl-2026-audio-embeddings (emb_wavlm_large.npz)
Outputs: voxtral_lora_oof.csv (train, OOF), voxtral_lora_test.csv (test, fold mean)
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
OUT, CACHE = "/kaggle/working", "/tmp/vox_cache"
REPO = "mistralai/Voxtral-Mini-3B-2507"
CFG = dict(n_layers=16, lora_r=16, lora_alpha=32, lora_dropout=0.05, lr_lora=2e-4, lr_head=1e-3, epochs=3,
           grad_accum=8, warmup=0.1, weight_decay=0.01, head_dropout=0.1, seed=42, n_splits=5)
INSTR = "Listen to this spoken English response and evaluate the speaker's grammatical accuracy."


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def find(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    assert hits, f"{pattern} not found"
    return hits[0]


# peft refuses to inject LoRA layers when an old torchao (< 0.16, preinstalled on Kaggle) is importable; it is unused here
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft>=0.19.1", "mistral-common[audio]"], check=True)
import peft  # noqa: E402
import transformers  # noqa: E402
log("peft", peft.__version__, "| transformers", transformers.__version__)
os.makedirs(CACHE, exist_ok=True)


def smoke_test():
    """One mixed-precision LoRA training step on a tiny Llama, mirroring the fold worker (fails in seconds, not after
    the 9-minute caching stage)."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaModel

    tiny = LlamaModel(LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                                  num_key_value_heads=2, vocab_size=100)).half().cuda()
    for p in tiny.parameters():
        p.requires_grad_(False)
    tiny = get_peft_model(tiny, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    for p in tiny.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    head = torch.nn.Linear(64, 1).cuda()
    params = [p for p in tiny.parameters() if p.requires_grad] + list(head.parameters())
    opt, scaler = torch.optim.AdamW(params, lr=1e-3), torch.amp.GradScaler()
    with torch.autocast("cuda", dtype=torch.float16):
        h = tiny(inputs_embeds=torch.randn(1, 12, 64, device="cuda", dtype=torch.float16),
                 attention_mask=torch.ones(1, 12, dtype=torch.long, device="cuda"), use_cache=False).last_hidden_state
    loss = (head(h[0].float().mean(0)).squeeze() - 1.0) ** 2
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    scaler.step(opt)
    scaler.update()
    assert all(p.grad is not None for p in params), "LoRA parameters received no gradient"
    log("smoke test ok: LoRA injection + fp16 autocast step")


smoke_test()

# ---------------------------------------------------------------- 1. data, zero rule, speaker groups, folds
from scipy.sparse import csr_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

ROOT = os.path.dirname(find("Dataset_Final/train.csv"))
df = pd.concat([pd.read_csv(f"{ROOT}/train.csv").assign(split="train"),
                pd.read_csv(f"{ROOT}/test.csv").assign(split="test", label=np.nan)], ignore_index=True)
df["path"] = [f"{ROOT}/{s}/{f}" for s, f in zip(df.split, df.filename)]
st = pd.read_csv(find("audio_stats.csv"))[["split", "filename", "dyn_range_db", "peak"]]
df = df.merge(st, on=["split", "filename"], how="left")
df["is_noise"] = (df.dyn_range_db < 5) & (df.peak > 0.9)

zw = np.load(find("emb_wavlm_large.npz"), allow_pickle=True)
V = np.hstack([zw["mean"][:, 3:7].reshape(len(zw["mean"]), -1), zw["std"][:, 3:7].reshape(len(zw["std"]), -1)])
V = V.astype(np.float64)
V = (V - V.mean(0)) / (V.std(0) + 1e-8)
V /= np.linalg.norm(V, axis=1, keepdims=True)
S = V @ V.T
np.fill_diagonal(S, -1)
thr = float(np.percentile(S[np.ix_(zw["split"] == "test", zw["split"] == "train")].max(1), 99.5))
_, comp = connected_components(csr_matrix(S > thr), directed=False)
spk = pd.DataFrame({"split": zw["split"], "filename": zw["filename"], "speaker": comp})
df = df.merge(spk, on=["split", "filename"], how="left")
assert df.speaker.notna().all()

df["fold"] = -1
trm = ((df.split == "train") & ~df.is_noise).values
tr_idx = np.flatnonzero(trm)
y_tr = df.label.values[tr_idx]
strata = np.maximum(np.round(y_tr * 2).astype(int), 4)  # merge the 4 clips below 2.0 into 2.0
sgkf = StratifiedGroupKFold(n_splits=CFG["n_splits"], shuffle=True, random_state=CFG["seed"])
for k, (_, va) in enumerate(sgkf.split(tr_idx, strata, df.speaker.values[tr_idx])):
    df.loc[tr_idx[va], "fold"] = k
df["idx"] = np.arange(len(df))
df.to_csv("/tmp/meta.csv", index=False)
log(f"clips {len(df)} | speech train {trm.sum()} | speaker groups {df[trm].speaker.nunique()} | "
    f"fold sizes {df[trm].fold.value_counts().sort_index().to_dict()} | threshold {thr:.3f}")

# ---------------------------------------------------------------- 2. cache audio embeddings (frozen encoder)
import torch  # noqa: E402
from transformers import AutoProcessor, VoxtralForConditionalGeneration  # noqa: E402

import soundfile as sf  # noqa: E402

CAP = 60 * 16000          # 60.0 s: a 60.01-61.04 s clip would otherwise get a 3rd, >99%-padding 30-s chunk
TOK_PER_CHUNK, TOK_PER_SEC = 375, 12.5  # Voxtral: 30-s chunk -> 1500 encoder frames -> 375 projected tokens
os.makedirs("/tmp/trim", exist_ok=True)


def capped_path(split, fn, path):
    info = sf.info(path)
    if info.frames <= CAP:
        return path, info.frames / info.samplerate
    q = f"/tmp/trim/{split}_{fn}"  # split prefix: train and test reuse file names
    if not os.path.exists(q):
        x, sr = sf.read(path, dtype="int16")
        sf.write(q, x[:CAP], sr, subtype="PCM_16")
    return q, CAP / 16000


def valid_token_mask(n_tok, dur):
    """True for audio tokens that encode real audio (the tail of the last 30-s chunk is padding)."""
    assert n_tok % TOK_PER_CHUNK == 0, n_tok
    mask = np.zeros(n_tok, bool)
    for c in range(n_tok // TOK_PER_CHUNK):
        sec = min(30.0, max(0.0, dur - 30.0 * c))
        mask[c * TOK_PER_CHUNK: c * TOK_PER_CHUNK + min(TOK_PER_CHUNK, int(np.ceil(sec * TOK_PER_SEC)))] = True
    return mask


proc = AutoProcessor.from_pretrained(REPO)
model = VoxtralForConditionalGeneration.from_pretrained(REPO, dtype=torch.float16, device_map="cuda:0").eval()
AUDIO_ID = model.config.audio_token_id
n_tok_hist = {}
for r in df.itertuples():
    path, dur = capped_path(r.split, r.filename, r.path)
    conv = [{"role": "user", "content": [{"type": "text", "text": INSTR}, {"type": "audio", "path": path}]}]
    inputs = proc.apply_chat_template(conv)
    ids = inputs["input_ids"][0]
    with torch.no_grad():
        ae = model.model.get_audio_features(inputs["input_features"].to("cuda:0", torch.float16),
                                            return_dict=True).pooler_output
    n_tok = int((ids == AUDIO_ID).sum())
    assert ae.shape[0] == n_tok, (r.filename, ae.shape, n_tok)
    assert torch.isfinite(ae).all(), r.filename
    np.save(f"{CACHE}/{r.idx}_ids.npy", ids.cpu().numpy().astype(np.int64))
    np.save(f"{CACHE}/{r.idx}_ae.npy", ae.float().cpu().numpy().astype(np.float16))
    np.save(f"{CACHE}/{r.idx}_valid.npy", valid_token_mask(n_tok, dur))
    n_tok_hist[n_tok] = n_tok_hist.get(n_tok, 0) + 1
    if r.idx % 200 == 0:
        log("cached", r.idx, "audio tokens", n_tok, "seq len", len(ids))
del model
torch.cuda.empty_cache()
log("audio embeddings cached; audio-token counts:", n_tok_hist)

# ---------------------------------------------------------------- 3. per-GPU fold workers
WORKER = r'''
import json, math, sys, time
import numpy as np, pandas as pd, torch
from torch import nn
from peft import LoraConfig, get_peft_model
from transformers import VoxtralForConditionalGeneration, get_cosine_schedule_with_warmup

cfg = json.loads(sys.argv[1]); folds = cfg["folds"]; CACHE = cfg["cache"]
meta = pd.read_csv("/tmp/meta.csv")
T0 = time.time()
def log(*a): print(f"[gpu{cfg['gpu']} {time.time()-T0:7.1f}s]", *a, flush=True)

def load_sample(i):
    return (torch.from_numpy(np.load(f"{CACHE}/{i}_ids.npy")), torch.from_numpy(np.load(f"{CACHE}/{i}_ae.npy")),
            torch.from_numpy(np.load(f"{CACHE}/{i}_valid.npy")))

def build():
    m = VoxtralForConditionalGeneration.from_pretrained(cfg["repo"], dtype=torch.float16, device_map="cuda:0")
    lm = m.model.language_model
    lm.layers = lm.layers[: cfg["n_layers"]]
    lm.config.num_hidden_layers = cfg["n_layers"]
    lm.config.use_cache = False
    embed = lm.get_input_embeddings()
    del m  # keep only the truncated language model (audio tower / lm_head released)
    torch.cuda.empty_cache()
    for p in lm.parameters():
        p.requires_grad_(False)
    lcfg = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none")
    lm = get_peft_model(lm, lcfg)
    for n, p in lm.named_parameters():  # adapters in fp32 for stable fp16 mixed-precision training
        if p.requires_grad:
            p.data = p.data.float()
    head = nn.Sequential(nn.Dropout(cfg["head_dropout"]), nn.Linear(lm.config.hidden_size, 1)).cuda()
    nn.init.zeros_(head[1].weight); nn.init.zeros_(head[1].bias)
    return lm, embed, head

AUDIO_ID = cfg["audio_id"]

def forward(lm, embed, head, i):
    ids, ae, valid = load_sample(i)
    ids = ids.cuda()
    with torch.autocast("cuda", dtype=torch.float16):
        emb = embed(ids[None])  # frozen token embeddings (fp16)
        mask = (ids == AUDIO_ID)[None, :, None]
        emb = emb.masked_scatter(mask, ae.cuda().to(emb.dtype))
        h = lm(inputs_embeds=emb, attention_mask=torch.ones_like(ids)[None], use_cache=False).last_hidden_state
        audio_h = h[0][ids == AUDIO_ID]
        pooled = audio_h[valid.cuda()].float().mean(0)  # real-audio tokens only (no chunk padding)
    return head(pooled).squeeze(-1)

out_rows = []
for fold in folds:
    torch.manual_seed(cfg["seed"] + fold); np.random.seed(cfg["seed"] + fold)
    tr = meta[(meta.fold >= 0) & (meta.fold != fold)]
    va = meta[meta.fold == fold]
    te = meta[meta.split == "test"]
    mu, sd = float(tr.label.mean()), float(tr.label.std())
    lm, embed, head = build()
    lora_params = [p for p in lm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": lora_params, "lr": cfg["lr_lora"]},
                             {"params": head.parameters(), "lr": cfg["lr_head"]}], weight_decay=cfg["weight_decay"])
    n_updates = math.ceil(len(tr) * cfg["epochs"] / cfg["grad_accum"])
    sch = get_cosine_schedule_with_warmup(opt, int(cfg["warmup"] * n_updates), n_updates)
    scaler = torch.amp.GradScaler()
    step, bad, t_fold = 0, 0, time.time()
    for ep in range(cfg["epochs"]):
        lm.train(); head.train()
        order = np.random.permutation(tr.idx.values)
        losses = []
        for k, i in enumerate(order):
            target = (float(meta.label.values[i]) - mu) / sd
            pred = forward(lm, embed, head, i)
            loss = (pred.float() - target) ** 2
            if not torch.isfinite(loss):
                bad += 1; opt.zero_grad(set_to_none=True); continue
            scaler.scale(loss / cfg["grad_accum"]).backward()
            losses.append(loss.item())
            if (k + 1) % cfg["grad_accum"] == 0 or k == len(order) - 1:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(lora_params + list(head.parameters()), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sch.step(); step += 1
            if fold == folds[0] and ep == 0 and k == 15:
                per = (time.time() - t_fold) / 16
                log(f"self-test ok: 16 samples, {per:.2f}s/sample, peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB "
                    f"-> ~{per * len(tr) * cfg['epochs'] / 60:.0f} min per fold")
        lm.eval(); head.eval()
        with torch.no_grad():
            pv = np.array([forward(lm, embed, head, i).item() for i in va.idx.values]) * sd + mu
        rmse = float(np.sqrt(np.mean((pv - va.label.values) ** 2)))
        log(f"fold {fold} epoch {ep} train_mse(std) {np.mean(losses):.3f} val_rmse {rmse:.4f} non-finite {bad}")
    with torch.no_grad():
        pt = np.array([forward(lm, embed, head, i).item() for i in te.idx.values]) * sd + mu
    out_rows += [{"split": "train", "filename": f, "fold": fold, "label": l, "pred": p}
                 for f, l, p in zip(va.filename, va.label, pv)]
    out_rows += [{"split": "test", "filename": f, "fold": fold, "label": np.nan, "pred": p}
                 for f, p in zip(te.filename, pt)]
    pd.DataFrame(out_rows).to_csv(cfg["out"], index=False)
    log(f"fold {fold} done in {(time.time() - t_fold) / 60:.1f} min")
    del lm, embed, head, opt; torch.cuda.empty_cache()
'''

with open("/tmp/lora_worker.py", "w") as f:
    f.write(WORKER)
n_gpu = len([l for l in subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines()
             if l.startswith("GPU")])
procs = []
for g in range(n_gpu):
    wcfg = dict(CFG, gpu=g, folds=list(range(CFG["n_splits"]))[g::n_gpu], cache=CACHE, repo=REPO,
                audio_id=int(AUDIO_ID), out=f"/tmp/lora_preds_gpu{g}.csv")
    procs.append(subprocess.Popen([sys.executable, "/tmp/lora_worker.py", json.dumps(wcfg)],
                                  env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g),
                                           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")))
deadline, alive = time.time() + 6 * 3600, list(procs)
while alive:
    time.sleep(20)
    for p in list(alive):
        if p.poll() is None:
            continue
        alive.remove(p)
        log("worker exit code", p.returncode)
        if p.returncode != 0:  # one failed fold makes the OOF column unusable: stop the others now
            for q in alive:
                q.kill()
            raise SystemExit(f"fold worker failed (exit {p.returncode}); aborting")
    if time.time() > deadline:
        for q in alive:
            q.kill()
        raise SystemExit("time cap reached; aborting")

# ---------------------------------------------------------------- 4. merge OOF / test predictions
parts = [pd.read_csv(f) for f in sorted(glob.glob("/tmp/lora_preds_gpu*.csv"))]
assert parts, "no worker produced predictions"
pr = pd.concat(parts, ignore_index=True)
oof = pr[pr.split == "train"]  # keeps the fold id so the grouping can be audited downstream
tst = pr[pr.split == "test"].groupby(["split", "filename"], as_index=False).pred.mean()
n_folds_done = pr[pr.split == "test"].fold.nunique()
# fail loudly on partial results (a crashed worker must not yield a silently incomplete OOF column)
expected = set(df.loc[trm, "filename"])
assert set(oof.filename) == expected and len(oof) == len(expected), \
    f"OOF incomplete: {len(oof)} of {len(expected)} speech clips"
assert n_folds_done == CFG["n_splits"] and len(tst) == (df.split == "test").sum(), "test predictions incomplete"
assert np.isfinite(oof.pred).all() and np.isfinite(tst.pred).all(), "non-finite predictions"
oof.to_csv(f"{OUT}/voxtral_lora_oof.csv", index=False)
tst.to_csv(f"{OUT}/voxtral_lora_test.csv", index=False)
r = np.corrcoef(oof.pred, oof.label)[0, 1]
log(f"Voxtral-LoRA OOF ({len(oof)} speech clips, {n_folds_done} folds): RMSE {np.sqrt(np.mean((oof.pred - oof.label) ** 2)):.4f} "
    f"| Pearson {r:.4f}")
json.dump(CFG, open(f"{OUT}/voxtral_lora_config.json", "w"), indent=1)
