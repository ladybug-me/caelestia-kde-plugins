"""personal.topics — NMF topic extraction over a markdown vault.

Lee & Seung 1999, "Learning the Parts of Objects by Non-Negative Matrix
Factorization", Nature 401, 788-791 — multiplicative-update NMF: a
non-negative docs-by-terms count matrix V is factorized as V ~= W H
(docs-by-k times k-by-terms) with every entry held >= 0, giving
additive parts-based topics ("each note is a non-negative mixture of
k topics") instead of the signed axes a singular-vector decomposition
would produce. The updates are the paper's alternating rules for the
squared-Euclidean objective (H *= (W^T V) / (W^T W H), then
W *= (V H^T) / (W H H^T)), which the paper proves monotone
non-increasing; nothing here is invented beyond the standard eps guard
in the denominators (zero-division protection, stated because the raw
alternative is 0/0).

COMPUTATION NOTES (they matter at vault scale): V is SPARSE (a note
holds few distinct terms), so the two V-touching products accumulate
over nonzeros only, and the reconstruction error is never materialized
densely — it uses the Frobenius identity ||V - WH||^2 = ||V||^2 -
2<V, WH> + sum_{t,t'} (W^T W)[t][t'] * (H H^T)[t][t'], whose pieces
are all k x k or sparse. Per-iteration cost is O(nnz(V) * k +
(n + m) * k^2), independent of the dense n*m matrix size.

HONESTY (what a topic report may and may not claim):

- NMF is a LOCAL optimizer: the factorization is not unique, and the
  topics converged to depend on the init. The init here is positive
  uniform draws from random.Random(seed) with a FIXED seed (default 0)
  — deterministic, stated, and pinned byte-identical by test — but a
  different seed is a different (equally valid) local optimum. The
  report therefore always carries iterations / converged / error, and
  the note says descriptive clusters, not discovered truth;
- converged means the updates became STATIONARY (relative error change
  below tol) — on factorizations whose true factors hold exact zeros
  (a perfectly separable corpus) the multiplicative tail approaches
  zero sublinearly, so the run can honestly report converged=False
  with a small, still-shrinking error; the error field, not the flag,
  carries the quality;
- k is caller-chosen; k < 1 or k > number of docs is refused (a topic
  per note is not a topic model), as are empty corpora and empty
  vocabularies;
- the relative error is ||V - WH||_F / ||V||_F — reported, never
  hidden, so a bad factorization cannot masquerade as structure.

Pure stdlib, no I/O beyond the caller-supplied notes dict; no network;
the only randomness is the documented seeded init.
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

from ..nlp import tokens

EPS = 1e-10
DEFAULT_ITERS = 300
DEFAULT_TOL = 1e-9
DEFAULT_MAX_TERMS = 2000

__all__ = ["term_matrix", "nmf", "extract", "topic_mix",
            "drift_report", "_percentile_ranks", "build_basis"]


def term_matrix(notes: Dict[str, Dict[str, Any]],
                max_terms: int = DEFAULT_MAX_TERMS,
                ) -> Tuple[List[str], List[str], List[List[float]]]:
    """The docs-by-terms count matrix over a scanned vault: notes and
    terms in sorted order (deterministic), counts from the shared
    tokenizer (brain/nlp.tokens). The vocabulary is capped at
    ``max_terms`` by DOCUMENT FREQUENCY then term order (the standard
    IR bound; keeps the factorization cost tied to the corpus's real
    signal, not to pathological vocabularies) — the cap is reported by
    whoever surfaces the matrix, never silent here: ``extract`` puts
    the kept term count in its report."""
    docs = sorted(notes)
    df: Dict[str, int] = {}
    per_doc = []
    for rel in docs:
        counts: Dict[str, int] = {}
        for tok in tokens(notes[rel]["text"]):
            counts[tok] = counts.get(tok, 0) + 1
        per_doc.append(counts)
        for tok in counts:
            df[tok] = df.get(tok, 0) + 1
    terms = sorted(df)
    if len(terms) > max_terms:
        terms = sorted(
            sorted(terms, key=lambda t: (-df[t], t))[:max_terms])
    rows = [[float(counts.get(t, 0)) for t in terms]
            for counts in per_doc]
    return docs, terms, rows


def nmf(v: List[List[float]], k: int, iters: int = DEFAULT_ITERS,
        tol: float = DEFAULT_TOL, seed: int = 0) -> Dict[str, Any]:
    """Factorize the non-negative matrix v (n docs x m terms) as W
    (n x k) times H (k x m) via Lee & Seung 1999 multiplicative
    updates from a seeded positive init. Deterministic for fixed
    (v, k, iters, tol, seed). Returns {"W", "H", "iterations",
    "converged", "error"} where error is the RELATIVE Frobenius error
    ||v - WH||_F / ||v||_F of the final iterate (computed via the
    Frobenius identity, see the module docstring). Refusals: k < 1,
    k > n, empty matrix, empty vocabulary."""
    n = len(v)
    if n == 0:
        raise ValueError("the matrix is empty — nothing to factorize")
    if k < 1:
        raise ValueError(f"k must be >= 1 (got {k!r})")
    if k > n:
        raise ValueError(
            f"k={k} exceeds the {n} document row(s) — a topic per note "
            "is not a topic model; refusing rather than padding")
    m = len(v[0])
    if m == 0:
        raise ValueError("the vocabulary is empty — no terms to form "
                         "topics from")
    # sparse view of V: per doc {term_index: count}
    sparse = [{j: row[j] for j in range(m) if row[j] > 0} for row in v]
    base_sq = sum(c * c for doc in sparse for c in doc.values())

    rng = random.Random(seed)
    w = [[rng.uniform(0.05, 1.0) for _ in range(k)] for _ in range(n)]
    h = [[rng.uniform(0.05, 1.0) for _ in range(m)] for _ in range(k)]

    def _wtw() -> List[List[float]]:
        return [[sum(w[i][a] * w[i][b] for i in range(n))
                 for b in range(k)] for a in range(k)]

    def _hht() -> List[List[float]]:
        return [[sum(h[a][j] * h[b][j] for j in range(m))
                 for b in range(k)] for a in range(k)]

    error = float("inf")
    for it in range(1, iters + 1):
        # ---- H *= (W^T V) / (W^T W H + eps)
        num = [[0.0] * m for _ in range(k)]
        for i, doc in enumerate(sparse):
            wi = w[i]
            for j, c in doc.items():
                for t in range(k):
                    num[t][j] += wi[t] * c
        wtw = _wtw()
        for a in range(k):
            ha, na = h[a], num[a]
            wa = wtw[a]
            for j in range(m):
                den = 0.0
                for b in range(k):
                    den += wa[b] * h[b][j]
                ha[j] = ha[j] * (na[j] / (den + EPS))
        # ---- W *= (V H^T) / (W H H^T + eps)
        num2 = [[0.0] * k for _ in range(n)]
        for i, doc in enumerate(sparse):
            ni = num2[i]
            for j, c in doc.items():
                for t in range(k):
                    ni[t] += c * h[t][j]
        hht = _hht()
        for i in range(n):
            wi, ni = w[i], num2[i]
            den = [sum(wi[b] * hht[b][t] for b in range(k))
                   for t in range(k)]
            for t in range(k):
                wi[t] = wi[t] * (ni[t] / (den[t] + EPS))
        # ---- error via the Frobenius identity (never dense WH)
        wtw, hht = _wtw(), _hht()
        wh_sq = sum(wtw[a][b] * hht[a][b]
                    for a in range(k) for b in range(k))
        cross = 0.0
        for i, doc in enumerate(sparse):
            wi = w[i]
            for j, c in doc.items():
                s = 0.0
                for t in range(k):
                    s += wi[t] * h[t][j]
                cross += c * s
        e2 = base_sq - 2.0 * cross + wh_sq
        prev = error
        error = (max(e2, 0.0) ** 0.5) / ((base_sq ** 0.5) + EPS)
        if abs(prev - error) < tol * max(error, EPS):
            return {"W": w, "H": h, "iterations": it,
                    "converged": True, "error": error}
    return {"W": w, "H": h, "iterations": iters, "converged": False,
            "error": error}


def extract(notes: Dict[str, Dict[str, Any]], k: int = 3,
            terms_per_topic: int = 6,
            max_terms: int = DEFAULT_MAX_TERMS) -> Dict[str, Any]:
    """The topic report over a scanned vault: k NMF topics, each with
    its strongest terms, every note with its mixture and dominant
    topic, and the honesty fields (iterations / converged / error /
    vocabulary size). Refuses (ValueError, with the reason) on empty
    corpora, empty vocabularies, and k outside 1..len(notes)."""
    if not notes:
        raise ValueError("no notes — nothing to extract topics from")
    docs, terms, rows = term_matrix(notes, max_terms=max_terms)
    if not terms:
        raise ValueError("no terms across the vault — the tokenizer "
                         "found nothing to build topics from")
    if k > len(docs):
        raise ValueError(
            f"k={k} exceeds the {len(docs)} note(s) — a topic per note "
            "is not a topic model")
    result = nmf(rows, k)
    w, h = result["W"], result["H"]

    topics = []
    for ti in range(k):
        ranked = sorted(zip(terms, h[ti]), key=lambda p: (-p[1], p[0]))
        top = [(t, round(x, 6)) for t, x in ranked[:terms_per_topic]
               if x > 0]
        member_docs = []
        for di, rel in enumerate(docs):
            row = w[di]
            if sum(row) <= 0:
                continue
            best = max(range(k), key=lambda j: (row[j], -j))
            if best == ti:
                member_docs.append(rel)
        topics.append({"terms": top, "docs": member_docs})

    doc_topics: Dict[str, Dict[str, Any]] = {}
    for di, rel in enumerate(docs):
        row = w[di]
        total = sum(row)
        if total <= 0:
            doc_topics[rel] = {"dominant": None, "mixture": []}
            continue
        mixture = [round(x / total, 6) for x in row]
        best = max(range(k), key=lambda j: (row[j], -j))
        doc_topics[rel] = {"dominant": best, "mixture": mixture}

    return {
        "k": k,
        "docs": len(docs),
        "terms": len(terms),
        "terms_kept": len(terms),
        "topics": topics,
        "doc_topics": doc_topics,
        "iterations": result["iterations"],
        "converged": result["converged"],
        "relative_error": round(result["error"], 8),
        "note": "NMF topics are descriptive co-occurrence clusters from "
                "a seeded local optimum — not unique, not semantic "
                "truth; the error/converged fields say how well the "
                "k-topic mixture explains the counts",
    }


# ---------------------------------------------------------------------------
# Shared percentile helper + windowed topic drift (exponential-build-4 C)
# ---------------------------------------------------------------------------


def _percentile_ranks(values: Dict[str, float]) -> Dict[str, float]:
    """Rank-based percentile in [0, 100] per key (deterministic: ties
    share the mean rank). Shared by the graph bridge view and the
    drift report — ONE definition, never two."""
    if not values:
        return {}
    keys = sorted(values)
    ordered = sorted(keys, key=lambda k: values[k])
    n = len(ordered)
    out: Dict[str, float] = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[ordered[j + 1]] == values[ordered[i]]:
            j += 1
        mean_rank = (i + j) / 2.0
        pct = (mean_rank / (n - 1)) * 100.0 if n > 1 else 0.0
        for k in ordered[i:j + 1]:
            out[k] = pct
        i = j + 1
    return out


def _counts_rows(notes: Dict[str, Dict[str, Any]],
                 vocab: List[str]) -> List[List[float]]:
    """Doc-by-term counts over a FIXED vocabulary (the shared basis's
    own term order) — the projection input."""
    from ..nlp import tokens as _tokens
    vocab_set = set(vocab)
    rows: List[List[float]] = []
    for rel in sorted(notes):
        counts: Dict[str, float] = {}
        for tok in _tokens(notes[rel]["text"]):
            if tok in vocab_set:
                counts[tok] = counts.get(tok, 0.0) + 1.0
        rows.append([counts.get(t, 0.0) for t in vocab])
    return rows


def _project_w(rows: List[List[float]], h: List[List[float]],
               iters: int = 32) -> List[List[float]]:
    """Non-negative least-squares projection of V onto the FIXED basis
    H (solve V ~= W H for W): standard multiplicative updates with H
    frozen (Lee & Seung's W-update half). Deterministic, no fitting
    beyond the fixed-iteration refinement."""
    n, m = len(rows), len(rows[0]) if rows else 0
    k = len(h)
    if n == 0 or m == 0 or k == 0:
        return [[0.0] * k for _ in rows]
    w = [[1.0] * k for _ in range(n)]
    hh = [[sum(h[a][t] * h[b][t] for t in range(m)) for b in range(k)]
          for a in range(k)]
    for _ in range(iters):
        for i in range(n):
            for a in range(k):
                num = sum(rows[i][t] * h[a][t] for t in range(m))
                den = sum(w[i][b] * hh[a][b] for b in range(k))
                if den > 1e-12:
                    w[i][a] = max(1e-12, w[i][a] * num / den)
    return w


def build_basis(all_notes: Dict[str, Dict[str, Any]], k: int = 3
                ) -> Dict[str, Any]:
    """ONE NMF basis over the union of every snapshot's notes. Mixes
    are only comparable across time when they come from the SAME basis
    — per-snapshot NMF would change the basis under the signal, which
    is unsound (the build-4 test suite caught exactly that)."""
    docs, terms, rows = term_matrix(all_notes)
    if not docs or not terms or len(docs) < 2:
        return {"terms": [], "h": [], "k": 0}
    topics_k = max(1, min(k, len(docs) - 1, len(terms)))
    factors = nmf(rows, topics_k)
    return {"terms": terms, "h": factors["H"], "k": topics_k,
            "n_docs": len(docs),
            "relative_error": round(factors.get("error", 0.0), 4)}


def topic_mix(notes: Dict[str, Dict[str, Any]], basis: Dict[str, Any]
              ) -> Dict[str, Any]:
    """One snapshot's topic mix AGAINST the shared basis: projected
    document loadings W, column-summed and L1-normalized. Empty vocab
    or no docs -> an empty mix, honestly (the stream carries the gap)."""
    terms = basis.get("terms") or []
    h = basis.get("h") or []
    if not terms or not h:
        return {"mix": [], "n_docs": 0}
    rows = _counts_rows(notes, terms)
    if not rows or not any(any(r) for r in rows):
        return {"mix": [], "n_docs": len(rows)}
    w = _project_w(rows, h)
    k = len(h)
    col = [sum(row[t] for row in w) for t in range(k)]
    total = sum(col)
    mix = [c / total for c in col] if total > 0 else col
    return {"mix": [round(m, 6) for m in mix], "n_docs": len(rows)}


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or len(a) != len(b):
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    if da == 0.0 or db == 0.0:
        return 0.0
    return max(0.0, min(1.0, num / (da * db)))


def drift_report(snapshots: List[Tuple[str, Dict[str, Dict[str, Any]]]],
                 k: int = 3,
                 consensus: Optional[Any] = None) -> Dict[str, Any]:
    """Windowed topic drift: the EXISTING ADWIN + Page-Hinkley
    consensus (cortex/adwin.py DriftConsensus) reapplied — verbatim,
    unmodified — to a new signal: the cosine similarity between
    consecutive daily topic mixes of the vault, all mixes projected on
    ONE basis built from the union corpus (build_basis). No second
    drift detector is written here; the reapplication of the existing
    machinery to a different stream is the design intent.

    ``snapshots``: [(date_str, {note_id: {text: ...}})] in
    chronological order — the caller's snapshot cadence IS the window.
    The report names the topic gaining the most loading mass in the
    recent half vs the older half (an NMF co-occurrence cluster, per
    the module's honesty note — NOT a semantic topic).

    Fewer than 4 snapshots (or no shift the detectors' own bound can
    see) is an honest non-alarm, never a fabricated trend."""
    from assistant.core.adwin import DriftConsensus  # the existing machinery

    if len(snapshots) < 4:
        return {"drifted": False, "n_snapshots": len(snapshots),
                "note": "fewer than 4 snapshots: no drift verdict is "
                        "possible without inventing one — bring more "
                        "snapshot history"}
    union: Dict[str, Dict[str, Any]] = {}
    for _date, notes in snapshots:
        for nid, note in notes.items():
            union[nid] = note
    basis = build_basis(union, k)
    if not basis["terms"]:
        return {"drifted": False, "n_snapshots": len(snapshots),
                "note": "the union corpus has no vocabulary — nothing "
                        "to build a basis from"}
    consensus = consensus if consensus is not None else DriftConsensus()
    sims: List[Tuple[str, float]] = []
    mixes: List[List[float]] = []
    # SIGNAL CONDITIONING (the one added step between the NMF signal
    # and the detectors, documented because it is a choice): the
    # existing ADWIN/Page-Hinkley machinery consumes a BINARY stream
    # by contract — update(accepted: bool) binarizes its input, so
    # feeding raw similarities would silently destroy them. The
    # boolean fed is "the mix HELD": cosine >= 95% of the running
    # maximum. A sustained mix change is a sustained False run, which
    # is exactly the acceptance-drop shape those detectors were built
    # for. 5% is fixed and stated, not fitted.
    running_max = 0.0
    for i in range(1, len(snapshots)):
        prev = topic_mix(snapshots[i - 1][1], basis)
        curr = topic_mix(snapshots[i][1], basis)
        if not prev["mix"] or not curr["mix"]:
            continue  # empty snapshot: no similarity is invented
        mixes.append(curr["mix"])
        sim = _cosine(prev["mix"], curr["mix"])
        running_max = max(running_max, sim)
        held = sim >= 0.95 * running_max if running_max > 0 else True
        sims.append((snapshots[i][0], sim))
        consensus.update(held)
    status = consensus.status()
    half = max(1, len(mixes) // 2)
    gained: List[Dict[str, Any]] = []
    labels = None
    if mixes and mixes[0]:
        width = len(mixes[0])
        old_mean = [sum(m[t] for m in mixes[:half]) / half for t in range(width)]
        new_mean = [sum(m[t] for m in mixes[half:]) / (len(mixes) - half)
                    for t in range(width)]
        # label each topic by its top terms off the shared basis
        labels = []
        for t in range(width):
            pairs = sorted(zip(basis["terms"], basis["h"][t]),
                           key=lambda p: (-p[1], p[0]))
            labels.append([term for term, _w in pairs[:4]])
        for t in range(width):
            gained.append({"topic": t, "top_terms": labels[t],
                           "old_mean": round(old_mean[t], 4),
                           "new_mean": round(new_mean[t], 4),
                           "delta": round(new_mean[t] - old_mean[t], 4)})
        gained.sort(key=lambda g: -g["delta"])
    return {
        "drifted": bool(status.get("flag_drift")),
        "detector_status": status,
        "similarity_stream": [(d, round(s, 4)) for d, s in sims],
        "gaining_topic": gained[0] if gained and gained[0]["delta"] > 0 else None,
        "topic_deltas": gained,
        "n_snapshots": len(snapshots),
        "note": "ADWIN + Page-Hinkley consensus (binary contract) over "
                "'mix held' booleans conditioned from consecutive-mix "
                "cosine similarity (>= 95% of running max) on one "
                "shared NMF basis; topics are co-occurrence clusters, "
                "not semantic themes",
    }


