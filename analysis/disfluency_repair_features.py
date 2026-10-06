"""'LM-repair' features: where did Whisper's language model change what the CTC model heard?

Aligning the clean Whisper transcript with the LM-free Parakeet-CTC transcript, we count
edits that touch grammar-bearing words (articles, auxiliaries/copulas, prepositions,
pronouns) and morphological variants of the same stem (go/goes, walk/walked, chair/chairs).
Such edits are a proxy for learner errors that Whisper silently corrected.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

ARTICLES = {"a", "an", "the"}
AUX = {"is", "are", "was", "were", "be", "been", "being", "am", "has", "have", "had", "do", "does", "did",
       "will", "would", "can", "could", "shall", "should", "may", "might", "must"}
PREPS = {"in", "on", "at", "to", "for", "of", "with", "from", "by", "about", "into", "over", "under"}
PRONOUNS = {"i", "me", "my", "he", "him", "his", "she", "her", "they", "them", "their", "we", "us", "our", "it", "its"}
FILLERS = {"um", "uh", "er", "ah", "erm", "hmm", "mm", "uhm", "umm"}
_tok = re.compile(r"[a-z0-9']+")


def _words(t):
    return [w for w in _tok.findall((t or "").lower()) if w not in FILLERS]


def _same_stem(a, b):
    if a == b or min(len(a), len(b)) < 3:
        return False
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return k >= max(3, min(len(a), len(b)) - 2)  # differ only in a short ending


def repair_features(clean: str, ctc: str) -> dict:
    a, b = _words(clean), _words(ctc)
    n = max(len(a), 1)
    c = {"art": 0, "aux": 0, "prep": 0, "pron": 0, "morph": 0}
    for op, i1, i2, j1, j2 in SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        wa, wb = a[i1:i2], b[j1:j2]
        for w in wa + wb:  # insertions/deletions/substitutions of function words
            c["art"] += w in ARTICLES
            c["aux"] += w in AUX
            c["prep"] += w in PREPS
            c["pron"] += w in PRONOUNS
        if op == "replace":  # same stem, different ending (go/goes, walk/walked), any pairing in the span
            c["morph"] += sum(any(_same_stem(x, y) for y in wb) for x in wa)
    out = {f"repair_{k}_per100": 100.0 * v / n for k, v in c.items()}
    out["repair_total_per100"] = sum(out.values())
    return out


if __name__ == "__main__":
    print(repair_features("Yesterday I walked to the station", "yesterday i walk to station"))  # invented example
