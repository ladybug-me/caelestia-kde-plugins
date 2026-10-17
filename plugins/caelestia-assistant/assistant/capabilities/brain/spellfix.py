"""Typo correction against the vault's own vocabulary — no external
dictionary, so it works fully offline and never flags project-specific
jargon as wrong just because a dictionary hasn't heard of it.

Builds a SymSpell-style delete-index: every word's 1- and 2-delete variants
map back to it. A rare word that is a near-miss (edit distance <= 2) of a
much more common word in the same vault is very likely a typo of it.
"""
from collections import Counter, defaultdict

from .nlp import words


def _deletes(word, depth=2):
    variants = {word}
    frontier = {word}
    for _ in range(depth):
        nxt = set()
        for w in frontier:
            for i in range(len(w)):
                nxt.add(w[:i] + w[i + 1:])
        variants |= nxt
        frontier = nxt
    return variants


class SpellIndex:
    def __init__(self, min_word_len=3, rare_max=1, common_min=5):
        self.min_word_len = min_word_len
        self.rare_max = rare_max
        self.common_min = common_min
        self.counts = Counter()
        self.index = defaultdict(set)

    def build(self, texts):
        self.counts = Counter(w for t in texts for w in words(t) if len(w) >= self.min_word_len)
        self.index = defaultdict(set)
        for w in self.counts:
            for d in _deletes(w):
                self.index[d].add(w)
        return self

    def suggest(self, top=20):
        """Rare words that are a near-miss of a much more common vault word."""
        out = []
        for w, n in self.counts.items():
            if n > self.rare_max:
                continue
            candidates = set()
            for d in _deletes(w):
                candidates |= self.index.get(d, set())
            candidates.discard(w)
            best = max(
                ((c, self.counts[c]) for c in candidates if self.counts[c] >= self.common_min),
                key=lambda x: x[1], default=None,
            )
            if best:
                out.append({"typo": w, "count": n, "likely": best[0], "likely_count": best[1]})
        return sorted(out, key=lambda r: -r["likely_count"])[:top]
