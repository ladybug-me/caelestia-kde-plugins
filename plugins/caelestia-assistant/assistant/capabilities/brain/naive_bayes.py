"""Multinomial Naive Bayes, one class per tag/category, incremental.

Multi-label by design: a note can be trained into several tag classes. rank()
returns every class with a softmax-normalised probability, so callers pick a
threshold or a top-k.
"""
import math
from collections import defaultdict


class NaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_docs = defaultdict(int)
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.class_totals = defaultdict(int)
        self.vocab = set()
        self.n_docs = 0

    def train(self, tokens, labels):
        self.n_docs += 1
        for label in labels:
            self.class_docs[label] += 1
            for w in tokens:
                self.word_counts[label][w] += 1
                self.class_totals[label] += 1
                self.vocab.add(w)

    def rank(self, tokens):
        if not self.class_docs:
            return []
        V = max(len(self.vocab), 1)
        scores = {}
        for c in self.class_docs:
            s = math.log(self.class_docs[c] / self.n_docs)
            denom = self.class_totals[c] + self.alpha * V
            wc = self.word_counts[c]
            for w in tokens:
                s += math.log((wc.get(w, 0) + self.alpha) / denom)
            scores[c] = s
        m = max(scores.values())
        exps = {c: math.exp(v - m) for c, v in scores.items()}
        z = sum(exps.values())
        return sorted(((c, exps[c] / z) for c in exps), key=lambda x: -x[1])

    def to_dict(self):
        return {
            "alpha": self.alpha, "n_docs": self.n_docs,
            "class_docs": dict(self.class_docs),
            "class_totals": dict(self.class_totals),
            "word_counts": {c: dict(w) for c, w in self.word_counts.items()},
            "vocab": sorted(self.vocab),
        }

    @classmethod
    def from_dict(cls, d):
        nb = cls(alpha=d.get("alpha", 1.0))
        nb.n_docs = d.get("n_docs", 0)
        nb.class_docs.update(d.get("class_docs", {}))
        nb.class_totals.update(d.get("class_totals", {}))
        for c, w in d.get("word_counts", {}).items():
            nb.word_counts[c].update(w)
        nb.vocab = set(d.get("vocab", []))
        return nb
