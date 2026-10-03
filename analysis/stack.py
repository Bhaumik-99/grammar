"""Level-1 base models on every available feature view + NNLS stack with a second-level grouped CV.

Usage: python analysis/stack.py [--groups pseudo_spk] [--repeats 3] [--tag name]
Reads cached artifacts under outputs/ (whatever exists) and writes
outputs/stack/<tag>/{oof_level1.csv, test_level1.csv, report.json, submission_candidate.csv}

Protocol:
  * synthetic-noise clips (label 0) are handled by a rule -> regressors use speech clips only
  * speaker-grouped, label-stratified folds, identical for every base model
  * embedding layer bands are fixed a priori from the literature / public solutions,
    not selected on this CV (avoids optimistic selection)
  * ridge alpha chosen by speaker-grouped inner CV (default; --ridge-mode loo uses exact LOO)
  * stack = NNLS (+ free intercept) over OOF columns, evaluated by a second-level
    grouped CV on different folds; predictions clipped to [1, 5] for speech clips
  * leaderboard proxy: composite = (RMSE + 1 - Pearson) / 2
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).parent))
from cv import make_folds, nested_blend_score, nnls_blend, report, rmse  # noqa: E402
from fastridge import ridge_fold, ridge_oof, ridge_oof_grouped, standardize  # noqa: E402

ROOT = Path(__file__).parents[1]
OUTS = ROOT / "outputs"
CLIP_LO, CLIP_HI = 1.0, 5.0

# layer bands (hidden-state index; 0 = embedding output)
BANDS = {
    "whisper_large_v3": {"L30-32": [30, 31, 32], "L20-28": list(range(20, 29))},
    "wavlm_large": {"L19-21": [19, 20, 21]},
    "w2v_bert_2": {"L12-22": list(range(12, 23))},
    "hubert_large": {"L18-22": list(range(18, 23))},
    "voxtral_mini_3b": {"audioL9-14": list(range(9, 15)), "audioL15-22": list(range(15, 23))},
    "qwen2_audio_7b": {"audioL10-16": list(range(10, 17))},
    "voxtral_small_24b": {"audioL12-19": list(range(12, 20)), "audioL20-29": list(range(20, 30))},
    "voxtral_mini_3b_valid": {"audioL9-14": list(range(9, 15)), "audioL15-22": list(range(15, 23))},
}


def composite(r):
    return (r["rmse"] + 1 - r["pearson"]) / 2


def load_base():
    tr = pd.read_csv(ROOT / "data/train.csv").assign(split="train")
    te = pd.read_csv(ROOT / "data/test.csv").assign(split="test", label=np.nan)
    df = pd.concat([tr, te], ignore_index=True)
    st = pd.read_csv(OUTS / "eda/audio_stats.csv").drop(columns=["label"])
    df = df.merge(st, on=["split", "filename"], how="left")
    df["is_noise"] = (df.dyn_range_db < 5) & (df.peak > 0.9)
    df["is45"] = (np.abs(df.duration - 45.06) < 0.2).astype(int)
    df["dur_bucket"] = pd.cut(df.duration, [0, 44, 46, 59, 99], labels=["<44", "44-46", "46-59", ">=59"]).astype(str)
    return df


def align(df, keys_split, keys_file, arr):
    idx = pd.MultiIndex.from_arrays([keys_split, keys_file])
    pos = pd.Series(np.arange(len(idx)), index=idx)
    take = pos.reindex(pd.MultiIndex.from_arrays([df.split, df.filename])).values
    assert not np.isnan(take).any(), "missing rows in feature view"
    return arr[take.astype(int)]


def embedding_views(df):
    """Yield (name, X_all_rows) for every cached embedding view and fixed layer band."""
    for f in sorted((OUTS / "audio_emb").glob("emb_*.npz")):
        name = f.stem[4:]
        if name not in BANDS:
            continue
        z = np.load(f, allow_pickle=True)
        arr = z["audio_mean"] if "audio_mean" in z.files else z["mean"]
        arr = arr.astype(np.float32)
        for bname, layers in BANDS[name].items():
            layers = [l for l in layers if l < arr.shape[1]]
            X = arr[:, layers].mean(1)
            if "std" in z.files and name in ("wavlm_large", "hubert_large"):
                X = np.hstack([X, z["std"][:, layers].astype(np.float32).mean(1)])
            yield f"{name}_{bname}", align(df, z["split"], z["filename"], X)
    for f in sorted((OUTS / "llm_feats").glob("llmfeat_*.npz")):
        z = np.load(f, allow_pickle=True)
        hid = z["hidden"].astype(np.float32)
        layers = list(z["layers"])
        band = [layers.index(L) for L in (16, 20, 24) if L in layers]
        yield f"qwen3_8b_L16-24_{f.stem[8:]}", align(df, z["split"], z["filename"], hid[:, band].mean(1))
    for f in sorted((OUTS / "llm_feats").glob("qwen3emb_*.npz")):
        z = np.load(f, allow_pickle=True)
        yield f"qwen3emb_{f.stem[9:]}", align(df, z["split"], z["filename"], z["emb"].astype(np.float32))


def _words(text):
    import re

    return re.findall(r"[a-z0-9']+", (text or "").lower())


def _wer(ref, hyp):
    if not ref:
        return float(len(hyp) > 0)
    prev = list(range(len(hyp) + 1))
    for i, rw in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, hw in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[-1] / len(ref)


def asr_view_features(path):
    """Timing/confidence features of an extra ASR view + disagreement with clean Whisper."""
    from textfeats import timing_features

    name = path.stem[4:]
    clean = {(r["split"], r["filename"]): r for r in
             map(json.loads, open(OUTS / "eda/asr_whisper_large_v3.jsonl", encoding="utf-8"))}
    rows = []
    for r in map(json.loads, open(path, encoding="utf-8")):
        f = {f"{name}_{k}": v for k, v in timing_features(r, r.get("duration", 60.0)).items()}
        f[f"{name}_wer_vs_clean"] = _wer(_words(clean[(r["split"], r["filename"])]["text"]), _words(r["text"]))
        for k in ("frame_conf_mean", "frame_conf_p10", "blank_frac"):
            if k in r:
                f[f"{name}_{k}"] = r[k]
        rows.append({"split": r["split"], "filename": r["filename"], **f})
    return pd.DataFrame(rows).drop(columns=[f"{name}_n_words"], errors="ignore")


def tabular_view(df):
    """Hand-crafted fluency/confidence/linguistic + LLM judge + perplexity + GEC features."""
    parts = []
    hc = OUTS / "eda/features_handcrafted.csv"
    if hc.exists():
        h = pd.read_csv(hc)
        drop = {"label", "is_noise", "sr", "channels", "subtype", "format", "duration", "peak", "n_words",
                "ling_n_words", "n_pauses", "pause_total", "active_sec", "n_sents"}
        keep = [c for c in h.columns if c not in drop and c not in ("split", "filename") and h[c].dtype.kind in "fi"]
        parts.append(h[["split", "filename"] + keep])
    for f in sorted((OUTS / "llm_feats").glob("judge_*.csv")):
        j = pd.read_csv(f)
        view = f.stem[6:]
        cols = [c for c in j.columns if c.endswith(("_exp", "_sd", "_p_le2", "_p_ge45", "_mass"))]
        parts.append(j[["split", "filename"] + cols].rename(columns={c: f"{c}_{view}" for c in cols}))
    for f in sorted((OUTS / "llm_feats").glob("ppl_*.csv")):
        p = pd.read_csv(f)
        view = f.stem[4:]
        cols = [c for c in p.columns if c.startswith("nll_")]
        parts.append(p[["split", "filename"] + cols].rename(columns={c: f"{c}_{view}" for c in cols}))
    for f in sorted((OUTS / "llm_feats").glob("gec_*.jsonl")):
        from gec_features import gec_features

        rows = [json.loads(l) for l in open(f, encoding="utf-8")]
        parts.append(pd.DataFrame([{"split": r["split"], "filename": r["filename"],
                                    **gec_features(r["text"], r["gec"])} for r in rows]))
    for f in sorted((OUTS / "asr_views").glob("asr_*.jsonl")):
        parts.append(asr_view_features(f))
    rep = OUTS / "asr_views/repair_features.csv"
    if rep.exists():
        parts.append(pd.read_csv(rep))
    if not parts:
        return None
    t = df[["split", "filename"]]
    for p in parts:
        t = t.merge(p, on=["split", "filename"], how="left")
    return t.drop(columns=["split", "filename"])


# ---------------------------------------------------------------- heads
def run_ridge(X, y, folds, Xt):
    oof, test, _ = ridge_oof(X, y, folds, X_test=Xt)
    return oof, test


def ridge_full(X, y, Xt):
    """Refit on all speech train clips; return in-sample and test predictions."""
    Z = standardize(np.vstack([X, Xt]).astype(np.float64))
    K = Z @ Z.T / Z.shape[1]
    n = len(y)
    alphas = np.logspace(-1, 6, 36) * X.shape[1] / 1000.0 / Z.shape[1]
    p_tr, _ = ridge_fold(K[:n, :n], K[:n, :n], y, alphas)
    p_te, _ = ridge_fold(K[:n, :n], K[n:, :n], y, alphas)
    return p_tr, p_te


def svr_fp(Xa, ya, Xv, Xte):
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    m = make_pipeline(StandardScaler(), PCA(n_components=min(128, Xa.shape[0] - 1), random_state=0),
                      SVR(C=3.0, epsilon=0.1, gamma="scale")).fit(Xa, ya)
    return m.predict(Xv), m.predict(Xte)


def lgbm_fp(Xa, ya, Xv, Xte):
    pv, pt = [], []
    for seed in range(3):
        m = LGBMRegressor(n_estimators=500, learning_rate=0.02, num_leaves=7, min_child_samples=25,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0,
                          random_state=seed, verbose=-1).fit(Xa, ya)
        pv.append(m.predict(Xv))
        pt.append(m.predict(Xte))
    return np.mean(pv, 0), np.mean(pt, 0)


def cv_generic(fp, X, y, folds, Xt):
    n = len(y)
    s, c, tp = np.zeros(n), np.zeros(n), []
    for tr, va in folds:
        pv, pt = fp(X[tr], y[tr], X[va], Xt)
        s[va] += pv
        c[va] += 1
        tp.append(pt)
    return s / c, np.mean(tp, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="pseudo_spk")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tag", default="latest")
    ap.add_argument("--svr", action="store_true", help="also fit PCA+RBF-SVR heads on embedding views")
    ap.add_argument("--exclude", default="", help="comma-separated substrings of views to drop")
    ap.add_argument("--ridge-mode", default="grouped", choices=["grouped", "loo"])
    args = ap.parse_args()

    df = load_base()
    gfile = "pseudo_speakers.csv" if args.groups == "pseudo_spk" else "speaker_clusters.csv"
    sp = pd.read_csv(OUTS / "speakers" / gfile)
    df = df.merge(sp[["split", "filename", args.groups]], on=["split", "filename"], how="left")
    trm = ((df.split == "train") & ~df.is_noise).values
    tem = (df.split == "test").values
    y = df.label.values[trm]
    groups = df[args.groups].values[trm]
    folds = make_folds(y, n_splits=5, n_repeats=args.repeats, seed=42, groups=groups)
    is45 = df.is45.values[trm].astype(bool)
    print(f"speech train {trm.sum()} | test {tem.sum()} | folds {len(folds)} | groups={args.groups} "
          f"({len(set(groups))} speakers)")

    cols, oofs, tests, fulls = [], [], [], []

    cache_dir = OUTS / "stack" / "cache" / f"{args.groups}_r{args.repeats}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def cached(name, compute):
        """Load OOF/test/full predictions for a base model from cache, or compute and store them."""
        f = cache_dir / (re.sub(r"[^\w.-]+", "_", name) + ".npz")
        if f.exists():
            z = np.load(f, allow_pickle=True)
            return z["oof"], z["test"], (z["full"] if z["full"].ndim else None)
        oof, test, full = compute()
        np.savez(f, oof=oof, test=test, full=full if full is not None else np.array(None))
        return oof, test, full

    def add(name, oof, test, full=None):
        r = report(y, np.clip(oof, CLIP_LO, CLIP_HI))
        r45 = rmse(y[is45], np.clip(oof[is45], CLIP_LO, CLIP_HI))
        print(f"  {name:38s} rmse {r['rmse']:.4f}  r {r['pearson']:.4f}  comp {composite(r):.4f}  rmse45 {r45:.4f}",
              flush=True)
        cols.append(name)
        oofs.append(oof)
        tests.append(test)
        fulls.append(full)

    skip = set(args.exclude.split(",")) if args.exclude else set()
    for name, X in embedding_views(df):
        if any(k and k in name for k in skip):
            continue
        Xtr, Xte = X[trm], X[tem]
        if args.ridge_mode == "grouped":
            oof, test, p_full = cached(f"{name}|gridge", lambda: ridge_oof_grouped(Xtr, y, folds, groups, Xte)[:3])
            add(f"{name}|gridge", oof, test, p_full)
        else:
            oof, test, p_full = cached(f"{name}|ridge", lambda: (*run_ridge(Xtr, y, folds, Xte), ridge_full(Xtr, y, Xte)[0]))
            add(f"{name}|ridge", oof, test, p_full)
        if args.svr:
            oof, test, _ = cached(f"{name}|svr", lambda: (*cv_generic(svr_fp, Xtr, y, folds, Xte), None))
            add(f"{name}|svr", oof, test, None)
    tab = tabular_view(df)
    if tab is not None:
        T = tab.fillna(tab.median(numeric_only=True)).values.astype(np.float32)
        tkey = f"tabular{T.shape[1]}"
        if args.ridge_mode == "grouped":
            oof, test, p_full = cached(f"{tkey}|gridge", lambda: ridge_oof_grouped(T[trm], y, folds, groups, T[tem])[:3])
            add("tabular|gridge", oof, test, p_full)
        else:
            oof, test, p_full = cached(f"{tkey}|ridge", lambda: (*run_ridge(T[trm], y, folds, T[tem]), ridge_full(T[trm], y, T[tem])[0]))
            add("tabular|ridge", oof, test, p_full)
        oof, test, _ = cached(f"{tkey}|lgbm", lambda: (*cv_generic(lgbm_fp, T[trm], y, folds, T[tem]), None))
        add("tabular|lgbm", oof, test, None)
    deb = OUTS / "text_deberta/deberta_oof.csv"
    if deb.exists() and "deberta" not in skip:
        o = pd.read_csv(deb)
        t = pd.read_csv(OUTS / "text_deberta/deberta_test.csv")
        oof = df[trm][["split", "filename"]].merge(o, on=["split", "filename"], how="left").pred.values
        tst = df[tem][["split", "filename"]].merge(t, on=["split", "filename"], how="left").pred.values
        add("deberta_v3_large", oof, tst, None)

    for stem, colname in (("voxtral_lora/voxtral_lora", "voxtral_mini_lora"),):
        fo, ft = OUTS / f"{stem}_oof.csv", OUTS / f"{stem}_test.csv"
        if fo.exists() and colname not in skip:
            o, t = pd.read_csv(fo), pd.read_csv(ft)
            oof = df[trm][["split", "filename"]].merge(o, on=["split", "filename"], how="left").pred.values
            tst = df[tem][["split", "filename"]].merge(t, on=["split", "filename"], how="left").pred.values
            assert not np.isnan(oof).any() and not np.isnan(tst).any(), f"{colname}: missing predictions"
            add(colname, oof, tst, None)
    A, At = np.column_stack(oofs), np.column_stack(tests)
    folds2 = make_folds(y, n_splits=5, n_repeats=args.repeats, seed=2026, groups=groups)
    pred, rep = nested_blend_score(A, y, folds2)
    pred = np.clip(pred, CLIP_LO, CLIP_HI)
    rep = report(y, pred)
    # batch-aware variant: NNLS blend + per-batch (45 s vs other) linear recalibration, second-level CV
    pb_sum, pb_cnt = np.zeros(len(y)), np.zeros(len(y))
    for tr, va in folds2:
        wf, bf = nnls_blend(A[tr], y[tr])
        p_tr, p_va = A[tr] @ wf + bf, A[va] @ wf + bf
        for flag in (0, 1):
            mt, mv = is45[tr] == flag, is45[va] == flag
            if mt.sum() > 20 and mv.any():
                s, c = np.polyfit(p_tr[mt], y[tr][mt], 1)
                p_va[mv] = s * p_va[mv] + c
        pb_sum[va] += p_va
        pb_cnt[va] += 1
    pred_b = np.clip(pb_sum / pb_cnt, CLIP_LO, CLIP_HI)
    rep_b = report(y, pred_b)
    print(f"batch-aware recalibration (second-level CV): rmse {rep_b['rmse']:.4f}  r {rep_b['pearson']:.4f}  "
          f"composite {composite(rep_b):.4f}  rmse45 {rmse(y[is45], pred_b[is45]):.4f} "
          f"(plain rmse45 {rmse(y[is45], pred[is45]):.4f})")
    for lo in (1.5, 2.0):
        r_lo = report(y, np.clip(pred, lo, CLIP_HI))
        print(f"   clip low={lo}: rmse {r_lo['rmse']:.4f} composite {composite(r_lo):.4f}")
    w, b = nnls_blend(A, y)
    print(f"\nNNLS stack (second-level grouped CV): rmse {rep['rmse']:.4f}  r {rep['pearson']:.4f}  "
          f"composite {composite(rep):.4f}")
    print("weights:", {n: round(float(x), 3) for n, x in zip(cols, w) if x > 1e-4}, "intercept", round(float(b), 3))
    for bk in ["44-46", ">=59", "<44", "46-59"]:
        m = df.dur_bucket.values[trm] == bk
        print(f"   bucket {bk:6s} n={m.sum():3d} rmse={rmse(y[m], pred[m]):.4f}")
    tw = pd.Series(df.dur_bucket.values[tem]).value_counts(normalize=True)
    bks = df.dur_bucket.values[trm]
    tmix = float(np.sqrt(sum(tw[k] * np.mean((y[bks == k] - pred[bks == k]) ** 2) for k in tw.index)))
    print(f"   test-mix weighted RMSE {tmix:.4f}")
    # in-sample training RMSE (refit base models on all speech clips, apply stack weights)
    have_full = [i for i, f in enumerate(fulls) if f is not None]
    if have_full:
        Af = np.column_stack([fulls[i] if fulls[i] is not None else oofs[i] for i in range(len(cols))])
        train_fit = np.clip(Af @ w + b, CLIP_LO, CLIP_HI)
        print(f"   in-sample training RMSE (ridge views refit on all train; others OOF): {rmse(y, train_fit):.4f}")

    out = OUTS / "stack" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(A, columns=cols).assign(filename=df.filename.values[trm], label=y).to_csv(out / "oof_level1.csv", index=False)
    pd.DataFrame(At, columns=cols).assign(filename=df.filename.values[tem]).to_csv(out / "test_level1.csv", index=False)
    test_pred = np.clip(At @ w + b, CLIP_LO, CLIP_HI)
    test_pred[df.is_noise.values[tem]] = 0.0
    json.dump({"cols": cols, "weights": w.tolist(), "intercept": float(b), "nested": rep,
               "composite": composite(rep), "test_mix_rmse": tmix},
              open(out / "report.json", "w"), indent=1)
    df[tem][["filename"]].assign(label=test_pred).to_csv(out / "submission_candidate.csv", index=False)
    pd.DataFrame({"filename": df.filename.values[trm], "speaker": groups, "is45": is45.astype(int), "label": y,
                  "stack_oof": pred}).to_csv(out / "stack_oof.csv", index=False)
    # speaker-level bootstrap of the second-level CV metrics (uncertainty of the CV estimate)
    rng = np.random.default_rng(0)
    spk_ids = np.unique(groups)
    by_spk = {g_: np.flatnonzero(groups == g_) for g_ in spk_ids}
    boots = []
    for _ in range(1000):
        idx = np.concatenate([by_spk[g_] for g_ in rng.choice(spk_ids, len(spk_ids))])
        rb = report(y[idx], pred[idx])
        boots.append((rb["rmse"], rb["pearson"], composite(rb)))
    boots = np.array(boots)
    print(f"   speaker-bootstrap SE: rmse {boots[:, 0].std():.4f}  r {boots[:, 1].std():.4f}  composite {boots[:, 2].std():.4f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
