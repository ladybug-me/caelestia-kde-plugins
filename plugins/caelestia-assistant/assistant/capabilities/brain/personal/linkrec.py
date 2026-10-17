"""Suggest new wiki-links between notes that share meaning but aren't linked.

Two independent signals, combined (phase 3.2):

- Adamic-Adar scores a candidate pair by shared *neighbours* in the existing
  link graph, down-weighting hub neighbours that link to everything (a rare
  shared connection is stronger evidence than a popular one);
- TF-IDF cosine similarity over note TEXT (brain/textmine.py — the shared
  engine the personal README lists for root imports), so pairs with no
  graph history yet still compare by meaning, not just shingles.

The score is a blend: ``(1 - w) * aa_norm + w * cos`` with ``w =
tfidf_weight`` (default 0.5). ``aa_norm`` is the saturating transform
``aa / (aa + 1)`` — the same transform class fusion.py and units.py use to
bring an unbounded score into (0, 1) — so the two signals finally share one
scale. The ``via`` label names the dominant signal honestly:
"shared_neighbours", "text_similarity" (TF-IDF cosine only, gated by
``min_jaccard`` — the same floor the old shingle-Jaccard fallback used) or
"combined" when both fired.
"""
import math

from ..minhash import jaccard, signature
from ..textmine import cosine as tfidf_cosine, tfidf_vectors


def _neighbours(graph):
    out = {n: set(graph.out.get(n, set())) for n in graph.nodes}
    for src, tgts in graph.out.items():
        for t in tgts:
            out.setdefault(t, set()).add(src)
    return out


def adamic_adar(graph, a, b):
    neigh = _neighbours(graph)
    common = neigh.get(a, set()) & neigh.get(b, set())
    if not common:
        return 0.0
    return sum(1.0 / math.log(len(neigh.get(c, set())) + 1.001) for c in common)


def _aa_norm(aa):
    """Saturating transform of the unbounded Adamic-Adar score into
    (0, 1): aa / (aa + 1). Monotone, deterministic, never claims
    certainty — the same shape fusion.py/units.py use."""
    return aa / (aa + 1.0)


def suggest_links(graph, notes, top=10, min_jaccard=0.15,
                  tfidf_weight: float = 0.5):
    """notes: {id: text}. Returns [{"a", "b", "score", "via"}] sorted desc,
    via in {"shared_neighbours", "text_similarity", "combined"}. Skips pairs
    already linked in either direction.

    ``tfidf_weight`` is the TF-IDF cosine's share of the blend (0.0
    reproduces the pre-3.2 graph-only ranking, still via the same
    labels). The minhash shingle-Jaccard fallback is RETAINED for the
    cold-start case where the tokenizer finds nothing to vectorize
    (empty-token notes) — it gates on ``min_jaccard`` exactly as
    before.
    """
    ids = sorted(notes)
    sigs = {i: signature(notes[i]) for i in ids}
    vecs = tfidf_vectors([notes[i] for i in ids])
    vec_by_id = dict(zip(ids, vecs))
    scored = []
    for idx, a in enumerate(ids):
        for b in ids[idx + 1:]:
            if b in graph.out.get(a, set()) or a in graph.out.get(b, set()):
                continue
            aa = adamic_adar(graph, a, b)
            cos = tfidf_cosine(vec_by_id.get(a) or {},
                               vec_by_id.get(b) or {})
            w = float(tfidf_weight)
            aa_part = (1.0 - w) * _aa_norm(aa) if aa > 0 else 0.0
            cos_part = w * cos if cos > 0 else 0.0
            if aa_part > 0 and cos_part > 0:
                score = aa_part + cos_part
                scored.append({"a": a, "b": b, "score": round(score, 3),
                               "via": "combined",
                               "aa": round(aa, 3), "cos": round(cos, 3)})
            elif aa_part > 0:
                scored.append({"a": a, "b": b, "score": round(aa_part, 3),
                               "via": "shared_neighbours"})
            elif cos_part > 0 and cos >= min_jaccard:
                scored.append({"a": a, "b": b, "score": round(cos_part, 3),
                               "via": "text_similarity"})
            elif sigs[a] and sigs[b]:
                j = jaccard(sigs[a], sigs[b])
                if j >= min_jaccard:
                    scored.append({"a": a, "b": b, "score": round(j, 3),
                                   "via": "text_similarity"})
    return sorted(scored, key=lambda r: -r["score"])[:top]
