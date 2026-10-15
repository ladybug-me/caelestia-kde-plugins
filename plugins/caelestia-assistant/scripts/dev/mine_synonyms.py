#!/usr/bin/env python3
"""Capability-1: PMI synonym mining over the corpus.

For every tool name-atom, find corpus content words that co-occur with
it far more often than chance (PMI with smoothing, min frequency),
bounded deterministically to the top-2 per atom. Output:
assistant/data/mined_synonyms.json — loaded by the lexicon's query
expansion as HINTS (weight 0.3 channel, never replacements).

Dev-side tool: re-run after corpus changes. Deterministic output
(sorted keys, stable tie-breaks).
"""
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from assistant.adapters.caelestia.registry import TOOL_SPECS  # noqa: E402
from assistant.core.corpus import tool_atoms  # noqa: E402

CORPUS_DIR = os.path.join(REPO, "assistant", "data", "corpus")
OUT = os.path.join(REPO, "assistant", "data", "mined_synonyms.json")
WINDOW = 8          # token co-occurrence window
MIN_FREQ = 4        # co-occurrence floor
MIN_PMI = 1.5       # nats
TOP_K = 2
STOP = set("""the a an and or of to for on in with is are be been was were it its
this that these those i you we they my your our not no do does did so if then
than as at by from into when what which how why where can could should would
will shall may might must have has had but about after before over under
again more most some any all every each other another same different new old
just only also very much many few lot lots get got make made making want
wants wanted need needs needed like likes use used using set setting settings
change changing turn turning put keep kept show hide enable disable""".split())

TOKEN_RE = re.compile(r"[a-z][a-z0-9_-]{1,}")


def sentences(text: str):
    text = re.sub(r"`[^`]*`", " ", text)
    for para in re.split(r"\n\s*\n", text):
        para = re.sub(r"^#+ .*$", " ", para, flags=re.M)
        for sent in re.split(r"(?<=[.!?])\s+|[\n;|]+", para):
            toks = [t for t in TOKEN_RE.findall(sent.lower())]
            if toks:
                yield toks


def main():
    atoms = {}
    for spec in TOOL_SPECS:
        atoms[spec.name] = sorted({a.lower() for a in tool_atoms(spec)
                                   if len(a) > 2})
    files = sorted(
        os.path.join(CORPUS_DIR, f) for f in os.listdir(CORPUS_DIR)
        if f.endswith(".md"))
    docs = []
    for path in files:
        with open(path, encoding="utf-8") as fh:
            docs.append(fh.read())

    atom_set = {a for lst in atoms.values() for a in lst}
    cooc = defaultdict(Counter)   # atom -> Counter(word)
    atom_freq = Counter()
    word_freq = Counter()
    total = 0
    for text in docs:
        for toks in sentences(text):
            total += len(toks)
            freqs = Counter(toks)
            for w, c in freqs.items():
                word_freq[w] += c
            present = [t for t in dict.fromkeys(toks) if t in atom_set]
            for a in present:
                atom_freq[a] += 1
            for t_i, tok in enumerate(toks):
                if tok in atom_set:
                    lo, hi = max(0, t_i - WINDOW), min(len(toks), t_i + WINDOW)
                    for w in toks[lo:hi]:
                        if w == tok or w in STOP or len(w) < 4:
                            continue
                        cooc[tok][w] += 1

    out = {}
    for tool in sorted(atoms):
        mined = []
        for a in atoms[tool]:
            if a not in cooc:
                continue
            fa = atom_freq[a]
            cands = []
            for w, c in cooc[a].items():
                if c < MIN_FREQ or word_freq[w] < MIN_FREQ:
                    continue
                pmi = math.log((c * total) / (fa * word_freq[w]))
                if pmi >= MIN_PMI:
                    cands.append((round(pmi, 4), w))
            cands.sort(key=lambda t: (-t[0], t[1]))
            mined.extend(w for _, w in cands[:TOP_K])
        if mined:
            out[tool] = sorted(set(mined))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"how": "PMI-mined synonym hints (this script); hints only, "
                          "never replacements; reload happens at import",
                   "window": WINDOW, "min_freq": MIN_FREQ, "min_pmi": MIN_PMI,
                   "mined": out}, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(f"mined hints for {len(out)} tools -> {OUT}")


if __name__ == "__main__":
    main()
