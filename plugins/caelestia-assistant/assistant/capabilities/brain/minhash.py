"""MinHash + LSH banding for near-duplicate note detection.

Signature length num_perm; LSH splits it into bands so only colliding
documents are compared exactly. Jaccard is estimated from signature agreement
and then verified.
"""
import hashlib
import random

from .nlp import shingles

_P = (1 << 61) - 1
_MAX = (1 << 32) - 1


def _perms(num_perm, seed=42):
    rng = random.Random(seed)
    return [(rng.randrange(1, _P), rng.randrange(0, _P)) for _ in range(num_perm)]


def signature(text, num_perm=64):
    sh = shingles(text)
    if not sh:
        return None
    perms = _perms(num_perm)
    hashes = [int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")
              for s in sh]
    return [min((a * h + b) % _P for h in hashes) & _MAX for a, b in perms]


def jaccard(sig_a, sig_b):
    return sum(x == y for x, y in zip(sig_a, sig_b)) / len(sig_a)


def near_duplicates(docs, threshold=0.8, bands=16):
    """docs: {doc_id: text}. Returns [(id_a, id_b, est_jaccard)] sorted desc."""
    sigs = {k: s for k, s in ((k, signature(t)) for k, t in docs.items()) if s}
    if not sigs:
        return []
    rows = len(next(iter(sigs.values()))) // bands
    buckets = {}
    for doc_id, sig in sigs.items():
        for b in range(bands):
            key = (b, tuple(sig[b * rows:(b + 1) * rows]))
            buckets.setdefault(key, []).append(doc_id)
    candidates = set()
    for ids in buckets.values():
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                candidates.add(tuple(sorted((ids[i], ids[j]))))
    out = []
    for a, b in candidates:
        score = jaccard(sigs[a], sigs[b])
        if score >= threshold:
            out.append((a, b, round(score, 3)))
    return sorted(out, key=lambda x: -x[2])
