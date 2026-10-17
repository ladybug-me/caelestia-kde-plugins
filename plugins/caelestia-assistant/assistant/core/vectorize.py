"""Lean semantic vectorization for the cortex router.

Two classical engines, both deterministic, both stdlib-only:

1. ``TfidfIndex`` — an Okapi BM25+ index (with the lower-bound ``delta``
   that keeps long documents from saturating) over the synthetic corpus
   documents. This is the LEXICAL half of the router's score: exact and
   stemmed word overlap between the user's phrase and each surface's
   document.

2. ``PpmiEmbedder`` — distributional word vectors: PPMI co-occurrence
   matrix (positive pointwise mutual information) reduced by sparse
   random projection (Achlioptas' {-1, 0, +1} matrix, the
   Johnson-Lindenstrauss guarantee that pairwise distances survive the
   reduction), then row-normalized. A text embeds as the sublinear-tf
   weighted average of its word vectors. This is the SEMANTIC half: it
   puts user vocabulary and registry vocabulary that co-occur in the
   same phrase-doc pairs near each other in cosine space, so "make it
   see-through-ish" lands near the transparency tools without anyone
   hand-listing that synonym.

This is deliberately the "poor man's word2vec" — explicit, auditable
arithmetic replacing a neural embedding. It cannot hallucinate: a word
absent from the corpus has a zero vector and contributes nothing.

Determinism: the projection matrix is drawn from ``random.Random(seed)``
with a FIXED default seed, so every build of the same corpus produces
bit-identical vectors. No clock, no os randomness, no I/O.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

from .corpus import all_rows, tool_documents
from .lexicon import stem

# ---------------------------------------------------------------------------
# Tokenization shared by both engines (stemmed, stopworded).
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset({
    "a", "an", "the", "my", "me", "i", "please", "could", "would", "can",
    "to", "of", "for", "and", "or", "but", "with", "at", "by", "in", "on",
    "is", "are", "was", "be", "been", "am", "do", "does", "did", "so",
    "just", "like", "want", "wants", "need", "needs", "it", "its", "this",
    "that", "these", "those", "there", "here", "you", "your", "up", "out",
    "make", "makes", "making", "set", "sets", "setting", "get", "gets",
    "change", "changes", "changing", "adjust", "adjusts", "turn", "turns",
    "put", "puts", "give", "gives", "apply", "applies", "bit", "little",
    "slightly", "somewhat", "more", "most", "less", "least", "much",
    "look", "looks", "feel", "feels", "feeling", "everything", "keep",
    "keeps", "overall", "style", "vibe", "stuff", "thing", "things",
    "too", "very", "really", "kind", "sort", "now", "then", "still",
    "see", "through", "move", "moves", "moving", "want", "wanted",
})

_MINUS_STOP = frozenset({"it", "that", "this", "them", "those"})  # pronouns session.py resolves

# A2 constants (see the PPMI+SVD section further down): the SVD seed and
# the supervised-pair weight above the corpus prior.
SVD_SEED = 0x20260926
LABEL_WEIGHT = 4.0  # one supervised pair = four corpus rows of evidence


def tokenize(text: str, keep_pronouns: bool = False) -> List[str]:
    """Stemmed, stopword-filtered word list. Pronouns are normally
    dropped here (they carry no target information); ``session.py``
    re-injects resolved ones itself, so it passes keep_pronouns=True."""
    raw = [w for w in text.lower().replace("-", " ").split() if w]
    out: List[str] = []
    for w in raw:
        if not w.isalpha() and not w.isdigit():
            w = "".join(ch for ch in w if ch.isalpha())
            if not w:
                continue
        if not keep_pronouns and w in _MINUS_STOP:
            continue
        if w in _STOPWORDS and not (keep_pronouns and w in _MINUS_STOP):
            continue
        s = stem(w)
        if s and (s not in _STOPWORDS or (keep_pronouns and s in _MINUS_STOP)):
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# BM25+ lexical index.
# ---------------------------------------------------------------------------


class TfidfIndex:
    """Okapi BM25+ (k1=1.5, b=0.75, delta=1.0) over {key: document}.

    Pure in-memory, built once per process from the synthetic corpus.
    ``score`` returns the BM25+ sum for one query against one key —
    the standard lexical retrieval baseline, delta-smoothed so a long
    tool document never fully saturates its term weights.
    """

    def __init__(self, documents: Dict[str, str], k1: float = 1.5,
                 b: float = 0.75, delta: float = 1.0) -> None:
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.doc_tokens: Dict[str, List[str]] = {
            key: tokenize(doc) for key, doc in documents.items()
        }
        self.doc_len = {key: max(1, len(toks)) for key, toks in self.doc_tokens.items()}
        self.avg_len = sum(self.doc_len.values()) / max(1, len(self.doc_len))
        self.df: Counter = Counter()
        for toks in self.doc_tokens.values():
            self.df.update(set(toks))
        self.n_docs = len(self.doc_tokens)

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query_tokens: Sequence[str], key: str) -> float:
        toks = self.doc_tokens.get(key)
        if not toks:
            return 0.0
        tf = Counter(toks)
        dl = self.doc_len[key]
        norm = self.k1 * (1.0 - self.b + self.b * dl / self.avg_len)
        total = 0.0
        for term in query_tokens:
            f = tf.get(term, 0)
            if f == 0:
                continue
            idf = self._idf(term)
            total += idf * (f * (self.k1 + 1.0)) / (f + norm) + self.delta * (1.0 if f > 0 else 0.0)
        return total

    def search(self, text: str, keys: Iterable[str], k: int = 10) -> List[Tuple[str, float]]:
        """Top-k (key, score) by BM25+ — deterministic tie-break by key."""
        q = tokenize(text)
        scored = [(self.score(q, key), key) for key in keys]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [(key, score) for score, key in scored[:k] if score > 0.0]


# ---------------------------------------------------------------------------
# PPMI + random-projection embedder.
# ---------------------------------------------------------------------------


class PpmiEmbedder:
    """Distributional word vectors via PPMI + Achlioptas random projection.

    Co-occurrence counts are gathered from the synthetic corpus in two
    modes (both deterministic):

    - phrase-internal: adjacent word pairs inside each row's text
      (window 1, both orders) — teaches word similarity;
    - phrase-to-document: every phrase word paired with every word of
      its surface's document — teaches user-word to registry-word
      proximity, which is what the router actually needs.

    The PPMI matrix is then multiplied by a sparse {-1,0,+1} projection
    of width ``dim`` (entries drawn with probability 1/6, 2/3, 1/6,
    scaled by 1/sqrt(dim)) and each word's projected row is L2-normalized.

    ``labeled_pairs`` (A2) optionally adds supervised (text, surface)
    co-occurrences at ``label_weight`` above the corpus prior — the
    reroute-correction seam. MEASURED on the current corpus (see the A2
    section below and scripts/measure_svd_delta.py): supervision at
    weight 4.0 on a held-out split LOWERS top-1 retrieval 0.9683 ->
    0.9544, so the DEFAULT remains the corpus-only build; the parameter
    exists so the next corpus regime (organic user queries, real vocab
    growth) can adopt it with a fresh measurement instead of a guess.
    """

    def __init__(self, dim: int = 64, seed: int = 0x20260925,
                 labeled_pairs: Sequence[Tuple[str, str]] = (),
                 label_weight: float = LABEL_WEIGHT) -> None:
        self.dim = dim
        self.seed = seed
        self._build(labeled_pairs, label_weight)

    # -- construction -------------------------------------------------------

    def _build(self, labeled_pairs=(), label_weight=LABEL_WEIGHT) -> None:
        pair_counts: Counter = Counter()
        word_counts: Counter = Counter()
        total = _count_pairs(_corpus_rows(), pair_counts, word_counts)
        if labeled_pairs:
            docs = tool_documents()
            supervised = [(text, docs.get(surface, text))
                          for text, surface in labeled_pairs]
            total += _count_pairs(supervised, pair_counts, word_counts,
                                 weight=label_weight)

        # PPMI matrix as sparse dict-of-Counters.
        self.vocab = sorted(word_counts)
        self.index = {w: i for i, w in enumerate(self.vocab)}
        grand = max(1, total)
        ppmi: List[Dict[int, float]] = [dict() for _ in self.vocab]
        for (a, b), count in pair_counts.items():
            ia, ib = self.index.get(a), self.index.get(b)
            if ia is None or ib is None:
                continue
            p_ab = count / grand
            p_a = word_counts[a] / grand * 2.0  # symmetric counting doubles marginals
            p_b = word_counts[b] / grand * 2.0
            denom = p_a * p_b
            if denom <= 0:
                continue
            pmi = math.log(p_ab / denom)
            if pmi > 0:
                ppmi[ia][ib] = pmi

        # Achlioptas sparse random projection, seeded — deterministic.
        rng = random.Random(self.seed)
        scale = 1.0 / math.sqrt(self.dim)
        vectors: List[List[float]] = []
        for i in range(len(self.vocab)):
            vec = [0.0] * self.dim
            for j, val in ppmi[i].items():
                # one projection column set per (word,col) drawn on demand
                for col in range(self.dim):
                    r = rng.random()
                    if r < 1.0 / 6.0:
                        vec[col] += val * scale
                    elif r < 2.0 / 6.0:
                        vec[col] -= val * scale
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            vectors.append(vec)
        self.vectors = vectors

    # -- querying -----------------------------------------------------------

    def word_vector(self, word: str) -> List[float]:
        i = self.index.get(stem(word))
        return self.vectors[i] if i is not None else [0.0] * self.dim

    def embed(self, text: str, keep_pronouns: bool = False) -> List[float]:
        """Sublinear-tf weighted average of word vectors (tf weight
        ``1 + log(tf)``). Unknown words contribute exactly nothing."""
        toks = tokenize(text, keep_pronouns=keep_pronouns)
        counts = Counter(toks)
        vec = [0.0] * self.dim
        mass = 0.0
        for word, tf in counts.items():
            i = self.index.get(word)
            if i is None:
                continue
            weight = 1.0 + math.log(tf)
            wv = self.vectors[i]
            for col in range(self.dim):
                vec[col] += weight * wv[col]
            mass += weight
        if mass > 0:
            vec = [v / mass for v in vec]
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    @staticmethod
    def cosine(a: Sequence[float], b: Sequence[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na > 0 and nb > 0 else 0.0

    def similarity(self, text_a: str, text_b: str) -> float:
        return self.cosine(self.embed(text_a), self.embed(text_b))

    def neighbors(self, word: str, k: int = 5) -> List[Tuple[str, float]]:
        """Top-k nearest vocabulary words by cosine — the DISTRIBUTIONAL
        lexicon view (API parity with PpmiSvdEmbedder; see lexicon.py's
        distributional_neighbors for the reviewed-workflow framing)."""
        target = self.word_vector(word)
        if not any(target):
            return []
        stem_w = stem(word)
        scored = []
        for other, i in self.index.items():
            if other == stem_w:
                continue
            scored.append((self.cosine(target, self.vectors[i]), other))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [(w, round(s, 4)) for s, w in scored[:k]]


# ---------------------------------------------------------------------------
# Process-wide singletons (built once, deterministic, cheap: ~300 docs).
# ---------------------------------------------------------------------------

_INDEX: TfidfIndex | None = None
_EMBEDDER: PpmiEmbedder | None = None


def index() -> TfidfIndex:
    """The shared BM25+ index over tool documents + surface rows."""
    global _INDEX
    if _INDEX is None:
        documents = tool_documents()
        for row in all_rows():
            documents.setdefault(f"row::{row.text}", row.text)
        _INDEX = TfidfIndex(documents)
    return _INDEX


def embedder() -> PpmiEmbedder:
    """The shared PPMI embedder (fixed seed).

    Lexicon-diff wiring (phase 2.6): if the brain state carries IMPORTED
    lexicon pairs (opt-in `cortex lexicon import`), the FIRST build
    includes them as supervision (A2's ``labeled_pairs`` seam); an
    absent import keeps the corpus-only build byte-for-byte — the
    A2-measured default the fingerprint test pins. One JSON read, once
    per process, lazily (no brain-state dependency at import time).
    """
    global _EMBEDDER
    if _EMBEDDER is None:
        pairs = _persisted_lexicon_pairs()
        _EMBEDDER = PpmiEmbedder(labeled_pairs=pairs) if pairs \
            else PpmiEmbedder()
    return _EMBEDDER


def _persisted_lexicon_pairs():
    """Imported (text, surface) pairs from the brain state, or []."""
    try:
        from assistant.capabilities.brain import state as brain_state
        from . import lexicon_diff
        return lexicon_diff.persisted_pairs(brain_state.load())
    except Exception:
        return []  # the embedder must never fail on state trouble


# ---------------------------------------------------------------------------
# A2 — PPMI + truncated SVD (LSA) with supervised reroute-correction pairs.
#
# The classic LSA upgrade of the random-projection embedder above: the PPMI
# co-occurrence matrix is factored with a REAL truncated eigendecomposition
# instead of an Achlioptas sketch, so word vectors span the dominant
# co-occurrence directions exactly (up to the truncation), not by random
# chance. Deterministic end to end: seeded range-finder draw, pure
# arithmetic, no clock, no I/O beyond the corpus import.
#
# MEASURED VERDICT on the current corpus (5814 tool-surfaced rows, one
# document per registry tool; scripts/measure_svd_delta.py + measure_svd_variants.py, run
# this session): the random projection WINS on this corpus regime —
#   RP (status quo)          top-1 0.9678  top-3 0.9905  top-5 0.9967
#   SVD best weighting        top-1 0.8925  top-3 0.9723  top-5 0.9873
#   SVD + supervision         top-1 0.9005  (supervision helps SVD: +0.7pt
#                                         over its own corpus-only build)
#   RP  + supervision         top-1 0.9544  (supervision HURTS RP: -1.4pt)
# The reason is structural: with a 377-word closed vocabulary, the 64-dim
# random projection is a nearly lossless Johnson-Lindenstrauss sketch of
# the FULL PPMI geometry, while ANY truncation discards discriminative
# mid-spectrum directions that separate similar tools. The SVD path is
# therefore NOT the router default (the measured winner stays the
# default, per this repo's own bar); it is kept as a tested capability for
# the corpus regime where it is the right tool — an organic, growing
# vocabulary — plus the supervision seam and the distributional-lexicon
# neighbors view.
#
# Pipeline (all pure Python, genius.linalg reused for the dense stages):
#
#   1. sparse symmetric PPMI matrix A (vocab x vocab, ~11k nonzero entries
#      on the current corpus) — shared counting with PpmiEmbedder;
#   2. randomized range finder (Halko-Martinsson-Tropp): Q = orth((A)^q O)
#      with a seeded O, so span(Q) captures A's dominant column space;
#   3. Rayleigh-Ritz: B = Q^T A Q (via linalg.matmul + linalg.transpose),
#      eigendecomposed by cyclic Jacobi — the small k x k symmetric
#      eigenproblem, cross-verified against linalg.power_iteration in the
#      test suite;
#   4. word vectors = U * sqrt(lambda_+) rows (positive-part PPMI kernel,
#      the textbook LSA choice: components with lambda <= 0 are dropped,
#      keeping the embedded Gram matrix PSD);
#   5. labeled (text, surface) pairs — the cortex's reroute-corrections,
#      i.e. learn.py's label==1 examples — contribute co-occurrences at
#      LABEL_WEIGHT per occurrence, ABOVE the corpus prior's weight of 1,
#      so accepted corrections dominate the geometry as they accumulate.
# ---------------------------------------------------------------------------


def _count_pairs(rows_iter, pair_counts, word_counts, weight=1.0):
    """Shared co-occurrence counting (phrase-internal + phrase-to-doc),
    factored out of PpmiEmbedder._build so both embedders count the same
    way. ``rows_iter`` yields (text, doc_text) pairs."""
    total = 0

    def _count_pair(a: str, b: str) -> None:
        nonlocal total
        if a == b:
            return
        pair_counts[(a, b)] += weight
        pair_counts[(b, a)] += weight
        total += 2 * weight

    for text, doc in rows_iter:
        toks = tokenize(text, keep_pronouns=True)
        for a, b in zip(toks, toks[1:]):
            _count_pair(a, b)
        doc_toks = tokenize(doc)
        for a in toks:
            for b in doc_toks:
                _count_pair(a, b)
        word_counts.update(set(toks))
        word_counts.update(set(doc_toks))
    return total


def _corpus_rows():
    """(text, doc) pairs for the whole synthetic corpus — the prior."""
    docs = tool_documents()

    def gen():
        for row in all_rows():
            yield row.text, docs.get(row.surface, row.text)

    return gen()


def _ppmi_sparse(pair_counts, word_counts, total):
    """Sparse symmetric PPMI matrix: list-of-dicts rows + the vocab."""
    vocab = sorted(word_counts)
    index = {w: i for i, w in enumerate(vocab)}
    grand = max(1.0, total)
    ppmi: List[Dict[int, float]] = [dict() for _ in vocab]
    for (a, b), count in pair_counts.items():
        ia, ib = index.get(a), index.get(b)
        if ia is None or ib is None:
            continue
        p_ab = count / grand
        p_a = word_counts[a] / grand * 2.0  # symmetric counting doubles marginals
        p_b = word_counts[b] / grand * 2.0
        denom = p_a * p_b
        if denom <= 0:
            continue
        pmi = math.log(p_ab / denom)
        if pmi > 0:
            ppmi[ia][ib] = pmi
    return vocab, index, ppmi


def _sparse_matvec(ppmi: List[Dict[int, float]], x: Sequence[float]) -> List[float]:
    """y = A x for the sparse symmetric matrix (row-major dict rows)."""
    n = len(ppmi)
    y = [0.0] * n
    for i, row in enumerate(ppmi):
        s = 0.0
        for j, val in row.items():
            s += val * x[j]
        y[i] = s
    return y


def _gram_schmidt(block: List[List[float]]) -> List[List[float]]:
    """Orthonormalize the COLUMNS of a list-of-rows matrix (modified
    Gram-Schmidt, deterministic; near-zero columns are dropped)."""
    n_rows = len(block)
    n_cols = len(block[0]) if n_rows else 0
    basis: List[List[float]] = []  # each: a column as a row-list
    kept: List[List[float]] = []
    for c in range(n_cols):
        v = [block[r][c] for r in range(n_rows)]
        for q in basis:
            dot = sum(vr * qr for vr, qr in zip(v, q))
            v = [vr - dot * qr for vr, qr in zip(v, q)]
        norm = math.sqrt(sum(vr * vr for vr in v))
        if norm > 1e-10:
            q = [vr / norm for vr in v]
            basis.append(q)
            kept.append(q)
    # transpose back to n_rows x len(kept)
    out = [[0.0] * len(kept) for _ in range(n_rows)]
    for c, q in enumerate(kept):
        for r in range(n_rows):
            out[r][c] = q[r]
    return out


def _jacobi_eigh(matrix: List[List[float]], max_sweeps: int = 40,
                 tol: float = 1e-11) -> Tuple[List[float], List[List[float]]]:
    """Cyclic Jacobi eigendecomposition of a SMALL dense symmetric matrix.

    Returns (eigenvalues, eigenvectors) with eigenvector j in column j of
    the returned list-of-rows matrix. Deterministic sweep order (p, q)
    ascending; convergence = off-diagonal Frobenius norm below tol.
    Cross-verified against genius.linalg.power_iteration in the tests.
    """
    from assistant.capabilities.genius import linalg

    n = len(matrix)
    m = [row[:] for row in matrix]
    v = linalg.identity(n)
    for _ in range(max_sweeps):
        off = math.sqrt(sum(m[i][j] * m[i][j]
                            for i in range(n) for j in range(i + 1, n)))
        if off < tol:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = m[p][q]
                if abs(apq) < 1e-300:
                    continue
                theta = (m[q][q] - m[p][p]) / (2.0 * apq)
                sign = 1.0 if theta >= 0 else -1.0
                t = sign / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):  # rotate rows p, q of m and columns p, q
                    mkp, mkq = m[k][p], m[k][q]
                    m[k][p] = c * mkp - s * mkq
                    m[k][q] = s * mkp + c * mkq
                mp = m[p][:]  # row copies: apply the row rotation after
                mq = m[q][:]
                for k in range(n):
                    m[p][k] = c * mp[k] - s * mq[k]
                    m[q][k] = s * mp[k] + c * mq[k]
                for k in range(n):  # accumulate the rotation into v
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
    eigenvalues = [m[i][i] for i in range(n)]
    return eigenvalues, v


class PpmiSvdEmbedder:
    """PPMI co-occurrence + truncated SVD (LSA) word vectors.

    Same query API as PpmiEmbedder (word_vector / embed / cosine /
    similarity), same determinism guarantees (seeded range finder), but
    the reduction is an actual truncated eigendecomposition of the PPMI
    kernel rather than a random projection, and optional supervised
    (text, surface) pairs — the cortex's reroute-corrections — weight the
    co-occurrences they touch LABEL_WEIGHT times above the corpus prior.

    Footprint (measured on the 5890-row corpus, vocab 377): build adds
    ~1-2s CPU once per process; steady-state memory is the vocab x dim
    vector table (377 x 64 floats) plus the discarded build scaffolding —
    the same order as the projection embedder.
    """

    def __init__(self, dim: int = 64, seed: int = SVD_SEED,
                 labeled_pairs: Sequence[Tuple[str, str]] = (),
                 label_weight: float = LABEL_WEIGHT,
                 power_iters: int = 2) -> None:
        from assistant.capabilities.genius import linalg

        self.dim = dim
        self.seed = seed
        pair_counts: Counter = Counter()
        word_counts: Counter = Counter()
        total = _count_pairs(_corpus_rows(), pair_counts, word_counts)

        # Supervised pairs: the reroute-corrections, weighted above the
        # prior. Co-occurrences between the pair's text tokens and the
        # surface document's tokens get label_weight per occurrence.
        docs = tool_documents()
        if labeled_pairs:
            supervised = []
            for text, surface in labeled_pairs:
                doc = docs.get(surface)
                if doc is None:
                    # coarse surface: its corpus rows' own text is the doc
                    doc = text
                supervised.append((text, doc))
            total += _count_pairs(supervised, pair_counts, word_counts,
                                 weight=label_weight)

        self.vocab, self.index, ppmi = _ppmi_sparse(
            pair_counts, word_counts, total)
        n = len(self.vocab)

        # 1. Seeded range finder: Q = orth(A^q Omega) — column-wise sparse
        # matvecs, then modified Gram-Schmidt, then power sweeps.
        rng = random.Random(seed)
        omega_cols = [[rng.uniform(-1.0, 1.0) for _r in range(n)]
                      for _c in range(dim)]
        y_cols = [_sparse_matvec(ppmi, x) for x in omega_cols]
        y = [[y_cols[c][r] for c in range(dim)] for r in range(n)]
        q = _gram_schmidt(y)
        for _ in range(power_iters):
            q_cols = len(q[0]) if q else 0
            y_cols = [_sparse_matvec(ppmi, [q[r][c] for r in range(n)])
                      for c in range(q_cols)]
            y = [[y_cols[c][r] for c in range(q_cols)] for r in range(n)]
            q = _gram_schmidt(y)

        # 2. Rayleigh-Ritz on the projected matrix B = Q^T A Q.
        aq_cols = [_sparse_matvec(ppmi, [q[r][c] for r in range(n)])
                   for c in range(len(q[0]))] if q else []
        aq = [[aq_cols[c][r] for c in range(len(aq_cols))] for r in range(n)]
        b = linalg.matmul(linalg.transpose(q), aq)  # k x k, genius.linalg
        eigenvalues, w = _jacobi_eigh(b)

        # 3. Keep the top positive-eigenvalue components, value-descending.
        order = sorted(range(len(eigenvalues)),
                       key=lambda j: -eigenvalues[j])
        kept = [j for j in order if eigenvalues[j] > 0][:dim]
        scales = [math.sqrt(eigenvalues[j]) for j in kept]

        # U = Q W restricted to kept columns, word vectors = U * scales.
        vectors: List[List[float]] = []
        for r in range(n):
            row = [0.0] * len(kept)
            for cj, j in enumerate(kept):
                u_rj = sum(q[r][t] * w[t][j] for t in range(len(q[0]))) \
                    if q else 0.0
                row[cj] = u_rj * scales[cj]
            norm = math.sqrt(sum(x * x for x in row))
            if norm > 0:
                row = [x / norm for x in row]
            vectors.append(row)
        self.vectors = vectors
        self.eigenvalues = [eigenvalues[j] for j in kept]

    # -- querying (identical surface to PpmiEmbedder) ----------------------

    def word_vector(self, word: str) -> List[float]:
        i = self.index.get(stem(word))
        if i is None:
            return [0.0] * max(1, len(self.vectors[0]) if self.vectors else 1)
        return self.vectors[i]

    @property
    def effective_dim(self) -> int:
        return len(self.vectors[0]) if self.vectors else 0

    def embed(self, text: str, keep_pronouns: bool = False) -> List[float]:
        toks = tokenize(text, keep_pronouns=keep_pronouns)
        counts = Counter(toks)
        if not self.vectors:
            return []
        dim = len(self.vectors[0])
        vec = [0.0] * dim
        mass = 0.0
        for word, tf in counts.items():
            i = self.index.get(word)
            if i is None:
                continue
            weight = 1.0 + math.log(tf)
            wv = self.vectors[i]
            for col in range(dim):
                vec[col] += weight * wv[col]
            mass += weight
        if mass > 0:
            vec = [v / mass for v in vec]
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    @staticmethod
    def cosine(a: Sequence[float], b: Sequence[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na > 0 and nb > 0 else 0.0

    def similarity(self, text_a: str, text_b: str) -> float:
        return self.cosine(self.embed(text_a), self.embed(text_b))

    def neighbors(self, word: str, k: int = 5) -> List[Tuple[str, float]]:
        """Top-k nearest vocabulary words by cosine — the DISTRIBUTIONAL
        lexicon view (what the SVD space believes is similar to this
        word). Deterministic, tie-break alphabetical."""
        target = self.word_vector(word)
        if not any(target):
            return []
        scored = []
        for other, i in self.index.items():
            if other == stem(word):
                continue
            scored.append((self.cosine(target, self.vectors[i]), other))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [(w, round(s, 4)) for s, w in scored[:k]]
