"""Hand-crafted features from ASR output (word timings, confidences, transcript).

Every count is normalised by clip duration or word count: test clips are
shorter on average (median 45 s vs 60 s in train), so raw counts would shift.
"""
from __future__ import annotations

import re

import numpy as np

FILLERS = {"um", "uh", "er", "ah", "erm", "hmm", "mm", "uhm", "umm", "uhh", "eh", "huh", "hm"}
CLAUSE_DEPS = {"advcl", "ccomp", "xcomp", "acl", "relcl", "csubj", "csubjpass"}
POS_KEYS = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "AUX", "CCONJ", "SCONJ", "PROPN", "INTJ"]
VERB_TAGS = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "MD"]

_strip = re.compile(r"^[^\w']+|[^\w']+$")


def _norm(tok: str) -> str:
    return _strip.sub("", tok.strip().lower())


def timing_features(rec: dict, duration: float) -> dict:
    """Fluency and ASR-confidence features from word-level timestamps."""
    words = [w for w in rec.get("words", []) if _norm(w[2])]
    n = len(words)
    minutes = max(duration, 1.0) / 60.0
    f = {"n_words": n, "wpm": n / minutes}
    if n == 0:
        return f
    starts = np.array([w[0] for w in words], float)
    ends = np.array([w[1] for w in words], float)
    probs = np.array([w[3] for w in words], float)
    toks = [_norm(w[2]) for w in words]
    span = max(ends[-1] - starts[0], 1e-3)
    gaps = np.clip(starts[1:] - ends[:-1], 0, None) if n > 1 else np.zeros(0)
    p25, p50, p100 = gaps[gaps >= 0.25], gaps[gaps >= 0.5], gaps[gaps >= 1.0]
    # mean length of run: words between pauses >= 0.25 s
    cuts = np.flatnonzero(gaps >= 0.25)
    runs = np.diff(np.concatenate([[0], cuts + 1, [n]]))
    f.update({
        "wpm_span": n / (span / 60.0),
        "articulation_rate": n / max(float((ends - starts).sum()), 1e-3),
        "speech_span_frac": span / max(duration, 1e-3),
        "lead_time": float(starts[0]),
        "pause25_per_min": len(p25) / minutes,
        "pause50_per_min": len(p50) / minutes,
        "pause100_per_min": len(p100) / minutes,
        "pause_mean": float(p25.mean()) if len(p25) else 0.0,
        "pause_max": float(gaps.max()) if len(gaps) else 0.0,
        "pause_time_frac": float(p25.sum()) / max(duration, 1e-3),
        "mean_run_len": float(runs.mean()),
        "max_run_len": float(runs.max()),
        "word_prob_mean": float(probs.mean()),
        "word_prob_p10": float(np.percentile(probs, 10)),
        "low_prob_frac": float((probs < 0.5).mean()),
        "filler_per100": 100.0 * sum(t in FILLERS for t in toks) / n,
        "repeat_per100": 100.0 * sum(toks[i] == toks[i - 1] for i in range(1, n)) / n,
        "bigram_repeat_per100": 100.0 * sum(
            toks[i - 1:i + 1] == toks[i - 3:i - 1] for i in range(3, n)) / n,
    })
    segs = rec.get("segments", [])
    if segs:
        f["seg_logprob_mean"] = float(np.mean([s["avg_logprob"] for s in segs]))
        f["seg_logprob_min"] = float(np.min([s["avg_logprob"] for s in segs]))
        f["seg_compression_max"] = float(np.max([s["compression_ratio"] for s in segs]))
        f["seg_no_speech_max"] = float(np.max([s["no_speech_prob"] for s in segs]))
    return f


def _mattr(toks: list[str], window: int = 50) -> float:
    if len(toks) < window:
        return len(set(toks)) / max(len(toks), 1)
    return float(np.mean([len(set(toks[i:i + window])) / window for i in range(len(toks) - window + 1)]))


def _depth(tok) -> int:
    d = 0
    while tok.head.i != tok.i and d < 100:  # spaCy creates new Token objects; compare indices
        tok = tok.head
        d += 1
    return d


def linguistic_features(doc) -> dict:
    """Lexical and syntactic complexity from a spaCy Doc."""
    words = [t for t in doc if t.is_alpha]
    n = len(words)
    f = {"ling_n_words": n}
    if n == 0:
        return f
    toks = [t.lower_ for t in words]
    sents = [s for s in doc.sents if any(t.is_alpha for t in s)]
    sent_lens = np.array([sum(t.is_alpha for t in s) for s in sents], float) if sents else np.array([n], float)
    depths = [max((_depth(t) for t in s), default=0) for s in sents] or [0]
    f.update({
        "ttr": len(set(toks)) / n,
        "mattr50": _mattr(toks),
        "mean_word_len": float(np.mean([len(t) for t in toks])),
        "long_word_frac": float(np.mean([len(t) > 6 for t in toks])),
        "n_sents": len(sents),
        "sent_len_mean": float(sent_lens.mean()),
        "sent_len_std": float(sent_lens.std()),
        "sent_len_max": float(sent_lens.max()),
        "dep_depth_mean": float(np.mean(depths)),
        "dep_depth_max": float(np.max(depths)),
        "clauses_per_sent": sum(t.dep_ in CLAUSE_DEPS for t in doc) / max(len(sents), 1),
        "conj_per_sent": sum(t.dep_ == "conj" for t in doc) / max(len(sents), 1),
        "sconj_per_100": 100.0 * sum(t.pos_ == "SCONJ" or t.dep_ == "mark" for t in doc) / n,
        "passive_per_100": 100.0 * sum(t.dep_ in {"nsubjpass", "auxpass"} for t in doc) / n,
        "noun_chunks_per_sent": len(list(doc.noun_chunks)) / max(len(sents), 1),
    })
    for p in POS_KEYS:
        f[f"pos_{p}"] = sum(t.pos_ == p for t in words) / n
    vt = [t.tag_ for t in doc if t.tag_ in VERB_TAGS]
    for tag in VERB_TAGS:
        f[f"vtag_{tag}"] = vt.count(tag) / n
    f["verb_form_diversity"] = len(set(vt))
    return f
