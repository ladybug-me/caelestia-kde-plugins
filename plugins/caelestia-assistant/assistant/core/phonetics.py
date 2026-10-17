"""core.phonetics — deterministic soundex-style keys + char-ngram sets.

Capability-1 (sealed-gap) feature: phonetic matching lets "kwin
activewindow bridge"-style mishearings and typo'd tool atoms
("transparancy") still reach the right tool WITHOUT widening the
lexical channel for everything else. Pure functions, stdlib only.
"""
from __future__ import annotations

import re
from typing import Dict, Set

__all__ = ["soundex", "char_ngrams", "phonetic_hit_rate"]

_LETTER_CODE = {
    **{c: "1" for c in "BFPV"},
    **{c: "2" for c in "CGJKQSXZ"},
    **{c: "3" for c in "DT"},
    "L": "4",
    **{c: "5" for c in "MN"},
    "R": "6",
}
_NON_CODE = re.compile(r"[AEIOUYHW]", re.I)


def soundex(word: str) -> str:
    """Classic soundex, bounded to 4 chars. Deterministic; empty for
    strings with no letters (digits/punct never produce keys)."""
    word = re.sub(r"[^A-Za-z]", "", word).upper()
    if not word:
        return ""
    first = word[0]
    coded = []
    prev = _LETTER_CODE.get(first, "")
    for ch in word[1:]:
        code = _LETTER_CODE.get(ch, "")
        if ch in "HW":
            continue  # H/W do not reset the previous code
        if code and code != prev:
            coded.append(code)
        prev = code
    return (first + "".join(coded) + "000")[:4]


def char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Lowercased padded character n-grams (the same normalization the
    fuzz channel uses, exposed for the mining scripts and tests)."""
    text = re.sub(r"\s+", " ", text.lower()).strip()
    if len(text) < n:
        return {text} if text else set()
    padded = f"  {text}  "
    return {padded[i:i + n] for i in range(len(padded) - n + 1)}


def phonetic_hit_rate(query_tokens, atom_tokens) -> float:
    """Fraction of the query's content tokens whose soundex key matches
    ANY atom token's key. 0.0 when the query has no phonetic content —
    digits and one/two-letter tokens never match, so pure-value requests
    ('bar 34') keep their honest 0.0."""
    q_keys = [soundex(t) for t in query_tokens
              if len(t) > 2 and re.search(r"[A-Za-z]", t)]
    q_keys = [k for k in q_keys if k]
    a_keys = {soundex(a) for a in atom_tokens} - {""}
    if not q_keys or not a_keys:
        return 0.0
    hits = sum(1 for k in q_keys if k in a_keys)
    return hits / len(q_keys)
