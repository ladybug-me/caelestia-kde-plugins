"""Variant sweep for the A2 SVD weighting choice (measurement, not shipped).

Why: the first SVD attempt (positive-part, sqrt scaling) measured WORSE
than the random projection at top-1 (0.8925 vs 0.9678). Standard LSA
practice offers several component weightings; this script measures the
usual suspects on the same eval as measure_svd_delta.py so the shipped
choice is the measured winner, not a guess:

  p=1     v = U Sigma          (row of the rank-k reconstruction)
  p=1/2   v = U Sigma^0.5       (kernel-preserving, classic LSA)
  p=0     v = U                 (directions only)
  drop1   p=1/2 minus the top component (the shared 'tool-ness' direction)

Run: PYTHONPATH=. python3 scripts/dev/measure_svd_variants.py
"""

from __future__ import annotations

import math

from assistant.core.corpus import all_rows, tool_documents
from assistant.core.vectorize import (
    PpmiEmbedder,
    _count_pairs,
    _corpus_rows,
    _gram_schmidt,
    _jacobi_eigh,
    _ppmi_sparse,
    _sparse_matvec,
)
from collections import Counter
import random


def build_components(dim=64, seed=0x20260926, power_iters=2):
    pair_counts = Counter()
    word_counts = Counter()
    total = _count_pairs(_corpus_rows(), pair_counts, word_counts)
    vocab, index, ppmi = _ppmi_sparse(pair_counts, word_counts, total)
    n = len(vocab)
    rng = random.Random(seed)
    omega_cols = [[rng.uniform(-1.0, 1.0) for _r in range(n)] for _c in range(dim)]
    y_cols = [_sparse_matvec(ppmi, x) for x in omega_cols]
    y = [[y_cols[c][r] for c in range(dim)] for r in range(n)]
    q = _gram_schmidt(y)
    for _ in range(power_iters):
        qc = len(q[0])
        y_cols = [_sparse_matvec(ppmi, [q[r][c] for r in range(n)]) for c in range(qc)]
        y = [[y_cols[c][r] for c in range(qc)] for r in range(n)]
        q = _gram_schmidt(y)
    qc = len(q[0])
    aq_cols = [_sparse_matvec(ppmi, [q[r][c] for r in range(n)]) for c in range(qc)]
    aq = [[aq_cols[c][r] for c in range(qc)] for r in range(n)]
    from assistant.capabilities.genius import linalg
    b = linalg.matmul(linalg.transpose(q), aq)
    eigvals, w = _jacobi_eigh(b)
    return vocab, index, q, w, eigvals


def vectors_for(vocab, q, w, eigvals, power, drop_first=False, use_abs=False):
    n = len(vocab)
    order = sorted(range(len(eigvals)), key=lambda j: -abs(eigvals[j]))
    kept = order[: len(q[0]) and 64]
    if drop_first and kept:
        kept = kept[1:]
    vectors = []
    for r in range(n):
        row = []
        for j in kept:
            lam = eigvals[j]
            if use_abs:
                lam = abs(lam)
            if lam <= 0:
                row.append(0.0)
                continue
            u_rj = sum(q[r][t] * w[t][j] for t in range(len(q[0])))
            row.append(u_rj * (lam ** power))
        norm = math.sqrt(sum(x * x for x in row))
        if norm > 0:
            row = [x / norm for x in row]
        vectors.append(row)
    return vectors, [eigvals[j] for j in kept]


def topk(vectors, index, rows, doc_vecs, ks=(1, 3, 5)):
    hits = {k: 0 for k in ks}
    n = 0

    def embed(text):
        from assistant.core.vectorize import tokenize
        from collections import Counter as C
        toks = tokenize(text)
        counts = C(toks)
        if not vectors:
            return []
        dim = len(vectors[0])
        vec = [0.0] * dim
        mass = 0.0
        for word, tf in counts.items():
            i = index.get(word)
            if i is None:
                continue
            wt = 1.0 + math.log(tf)
            wv = vectors[i]
            for col in range(dim):
                vec[col] += wt * wv[col]
            mass += wt
        if mass > 0:
            vec = [v / mass for v in vec]
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def cos(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na > 0 and nb > 0 else 0.0

    for row in rows:
        qv = embed(row.text)
        if not any(qv):
            continue
        scored = sorted(((cos(qv, dv), t) for t, dv in doc_vecs.items()),
                        key=lambda pair: (-pair[0], pair[1]))
        rank = next((i for i, (_, t) in enumerate(scored, start=1)
                     if t == row.surface), None)
        if rank is None:
            continue
        n += 1
        for k in ks:
            if rank <= k:
                hits[k] += 1
    return n, {k: round(hits[k] / max(1, n), 4) for k in ks}


def main() -> None:
    docs = tool_documents()
    rows = [r for r in all_rows() if r.surface in docs]
    rp = PpmiEmbedder()
    rp_vecs = {t: rp.embed(d) for t, d in docs.items()}
    n, rp_acc = topk(rp.vectors, rp.index, rows, rp_vecs)
    print(f"RP(random projection)      top-1 {rp_acc[1]:.4f}  top-3 {rp_acc[3]:.4f}  top-5 {rp_acc[5]:.4f}  (n={n})")

    vocab, index, q, w, eigvals = build_components()
    variants = [
        ("p=1/2 pos-part (current)", 0.5, False, False),
        ("p=1 abs", 1.0, False, True),
        ("p=1/2 abs", 0.5, False, True),
        ("p=0 (directions only)", 0.0, False, False),
        ("p=1/2 abs, drop top-1", 0.5, True, True),
        ("p=0 abs, drop top-1", 0.0, True, True),
    ]
    for name, power, drop_first, use_abs in variants:
        vectors, kept = vectors_for(vocab, q, w, eigvals, power, drop_first, use_abs)

        def embed_doc(doc_text):
            from assistant.core.vectorize import tokenize
            from collections import Counter as C
            toks = tokenize(doc_text)
            counts = C(toks)
            dim = len(vectors[0])
            vec = [0.0] * dim
            mass = 0.0
            for word, tf in counts.items():
                i = index.get(word)
                if i is None:
                    continue
                wt = 1.0 + math.log(tf)
                wv = vectors[i]
                for col in range(dim):
                    vec[col] += wt * wv[col]
                mass += wt
            if mass > 0:
                vec = [v / mass for v in vec]
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            return vec

        doc_vecs = {t: embed_doc(d) for t, d in docs.items()}
        n, acc = topk(vectors, index, rows, doc_vecs)
        print(f"{name:<26} top-1 {acc[1]:.4f}  top-3 {acc[3]:.4f}  top-5 {acc[5]:.4f}  (n={n}, dims={len(kept)})")


if __name__ == "__main__":
    main()
