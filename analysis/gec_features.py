"""Edit-rate features between an ASR transcript and its minimal grammatical correction.

More (and larger) corrections per word indicate weaker grammatical control.
Fillers are dropped from both sides first so removing "um"/"uh" is not counted.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

import numpy as np

FILLERS = {"um", "uh", "er", "ah", "erm", "hmm", "mm", "uhm", "umm", "uhh", "eh", "hm"}
_tok = re.compile(r"[a-z0-9']+")


def _words(text: str) -> list[str]:
    return [w for w in _tok.findall((text or "").lower()) if w not in FILLERS]


def gec_features(original: str, corrected: str, prefix: str = "gec") -> dict:
    a, b = _words(original), _words(corrected)
    n = max(len(a), 1)
    sm = SequenceMatcher(a=a, b=b, autojunk=False)
    ins = dele = sub = 0
    edit_spans = 0
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        edit_spans += 1
        if op == "insert":
            ins += j2 - j1
        elif op == "delete":
            dele += i2 - i1
        else:  # replace
            sub += max(i2 - i1, j2 - j1)
    return {
        f"{prefix}_edits_per100": 100.0 * edit_spans / n,
        f"{prefix}_ins_per100": 100.0 * ins / n,
        f"{prefix}_del_per100": 100.0 * dele / n,
        f"{prefix}_sub_per100": 100.0 * sub / n,
        f"{prefix}_word_change_rate": (ins + dele + sub) / n,
        f"{prefix}_unchanged_ratio": sm.ratio(),
        f"{prefix}_len_ratio": len(b) / n,
    }


if __name__ == "__main__":
    print(gec_features("last week my sister walk to the station and um miss two train",  # invented example
                       "Last week my sister walked to the station and missed two trains."))
    print(gec_features("I am fine.", "I am fine."))
    assert np.isclose(gec_features("I am fine.", "I am fine.")["gec_word_change_rate"], 0)
