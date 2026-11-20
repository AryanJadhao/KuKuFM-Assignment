"""Grounding check: does a quote the model gave us really occur in the screenplay text?

Token recall in order: the share of the quote's words found in the source text, matched
left-to-right. 1.0 = every word appears in order. Punctuation, case and curly quotes are ignored,
so harmless reformatting passes while invented lines fail.
"""
from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9']+")


def words(s: str) -> list[str]:
    s = (s or "").lower().replace("’", "'").replace("‘", "'")
    return _WORD.findall(s)


def grounded_ratio(quote: str, source: str) -> float:
    q, src = words(quote), words(source)
    if not q:
        return 0.0
    # greedy in-order match: tolerant of dropped words, intolerant of invented ones
    hits, j = 0, 0
    for w in q:
        k = j
        while k < len(src) and src[k] != w:
            k += 1
        if k < len(src):
            hits += 1
            j = k + 1
    return hits / len(q)


GROUNDED = 0.8
