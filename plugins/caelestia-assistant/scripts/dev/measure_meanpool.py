"""E3 measurement: static mean-pooling embeddings vs the shipped router
(measurement, not shipped code — pre-registered decision rule below).

The Group E hypothesis: a static-embedding channel (mean pooling of
per-token vectors) could beat or back the router's lexical ranker. This
script measures it on the SAME dev arena the shipped baseline is
recorded on (assistant/eval/baseline.json: routing top1 0.8843 over 121
items), under the same determinism rules, with the budget clocked:

Variants (the honest instantiations of "static embedding, mean pooled"
for a stdlib-only repo with no pretrained vectors):
  onehot  word vector = unit basis vector  -> mean = normalized TF
                                          (the classic vector-space model)
  tfidf   word vector = idf-scaled basis   -> mean = tf-idf centroid
  ppmi    word vector = the word's PPMI context profile (count-based
          static vectors, the only kind trainable in pure stdlib)

Pre-registered decision rule (written before measuring, honored after):
ADOPT a channel only if its top1 on the arena EXCEEDS the shipped
baseline point (0.8843) AND its per-query cost fits the route budget
(p50 <= 15 ms). Anything else is CUT with the numbers recorded.

Run: PYTHONPATH=. python3 scripts/dev/measure_meanpool.py
"""

from __future__ import annotations

import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path

import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from assistant.core.corpus import tool_documents  # noqa: E402
from assistant.capabilities.retrieval.indexer import tokenize  # noqa: E402

ARENA = REPO / "assistant" / "eval" / "sets" / "routing_dev.json"
BASELINE = REPO / "assistant" / "eval" / "baseline.json"
BUDGET_P50_MS = 15.0


def _token_counts(text: str) -> Counter:
    return Counter(tokenize(text))


def build_idf(docs: dict) -> dict:
    df: Counter = Counter()
    for text in docs.values():
        for term in set(_token_counts(text)):
            df[term] += 1
    n = len(docs)
    return {t: math.log(1.0 + (n - d + 0.5) / (d + 0.5))
            for t, d in df.items()}


def build_ppmi_profiles(docs: dict) -> dict:
    """Static word vectors from co-occurrence: each word's profile is its
    PPMI-weighted context distribution over the OTHER words in the same
    tool document (bounded, deterministic)."""
    word_doc_tf = {}
    df: Counter = Counter()
    for name, text in docs.items():
        counts = _token_counts(text)
        word_doc_tf[name] = counts
        df.update(counts.keys())
    n_docs = len(docs)
    profiles: dict = {}
    for name, counts in word_doc_tf.items():
        doc_terms = set(counts)
        # word w's context profile: PPMI(w, t) over co-occurring t
        for w in counts:
            prof = profiles.setdefault(w, {})
            for t in doc_terms:
                if t == w:
                    continue
                pmi = math.log((counts[w] * counts[t]) /
                               (df[w] * df[t] / n_docs) + 1e-9)
                if pmi > 0:
                    prof[t] = pmi
    return profiles


def _l2(vec: dict) -> dict:
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {k: v / norm for k, v in vec.items()}


def embed_onehot(counts: Counter) -> dict:
    norm = math.sqrt(sum(c * c for c in counts.values())) or 1.0
    return {t: c / norm for t, c in counts.items()}


def embed_tfidf(counts: Counter, idf: dict) -> dict:
    vec = {t: c * idf.get(t, 0.0) for t, c in counts.items()}
    return _l2(vec)


def embed_ppmi(counts: Counter, profiles: dict) -> dict:
    """Mean of the words' static PPMI profiles, weighted by tf."""
    acc: dict = {}
    for t, c in counts.items():
        for ct, w in profiles.get(t, {}).items():
            acc[ct] = acc.get(ct, 0.0) + c * w
    return _l2(acc)


def cosine(a: dict, b: dict) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


def main() -> None:
    docs = tool_documents()
    arena = json.loads(ARENA.read_text())["items"]
    baseline = json.loads(BASELINE.read_text())["metrics"]
    shipped_top1 = baseline["routing.top1_rate"]["point"]
    # the arena's own scoring contract: accept-empty rows are the honest
    # abstention rows (OUT_OF_ONTOLOGY / AMBIGUOUS). A pure embedding
    # channel has NO abstention mechanism — it always names a tool — so
    # on those rows it is structurally wrong. Reported both ways:
    # routable-only (the signal) and all-items (shipped-comparable).
    routable = [(it["text"], it["accept"]) for it in arena
                if it.get("accept")]
    abstain_rows = sum(1 for it in arena if not it.get("accept"))

    idf = build_idf(docs)
    t0 = time.perf_counter()
    profiles = build_ppmi_profiles(docs)
    ppmi_build_s = time.perf_counter() - t0

    doc_counts = {name: _token_counts(text) for name, text in docs.items()}
    variants = {
        "onehot": (lambda c: embed_onehot(c), None),
        "tfidf": (lambda c: embed_tfidf(c, idf), None),
        "ppmi": (lambda c: embed_ppmi(c, profiles), ppmi_build_s),
    }

    print(f"arena items: {len(arena)}  tools: {len(docs)}  "
          f"routable: {len(routable)}  abstention-rows: {abstain_rows}")
    print(f"shipped baseline routing.top1: {shipped_top1:.4f} "
          "(fused, recorded 2026-09-30; the shipped router ABSTAINS on "
          "the abstention-rows — an embedding channel cannot)")
    print()

    def run(embedder, rows):
        doc_vecs = {name: embedder(counts)
                    for name, counts in doc_counts.items()}
        hits = 0
        per_query_ms = []
        for text, accept in rows:
            t1 = time.perf_counter()
            qv = embedder(_token_counts(text))
            ranked = sorted(doc_vecs.items(),
                            key=lambda kv: (-cosine(qv, kv[1]), kv[0]))
            per_query_ms.append((time.perf_counter() - t1) * 1000.0)
            if ranked and ranked[0][0] in accept:
                hits += 1
        return hits / len(rows), statistics.median(per_query_ms)

    results = {}
    for vname, (embedder, build_s) in variants.items():
        t0 = time.perf_counter()
        rout_top1, p50 = run(embedder, routable)
        build_s_total = time.perf_counter() - t0 + (build_s or 0.0)
        all_top1, _ = run(embedder, routable +
                          [(it["text"], []) for it in arena
                           if not it.get("accept")])
        results[vname] = {"top1_all": all_top1, "top1_routable": rout_top1,
                          "p50_ms": p50, "build_s": build_s_total}
        print(f"{vname:7s} top1(all)={all_top1:.4f}  "
              f"top1(routable)={rout_top1:.4f}  p50={p50:6.2f} ms  "
              f"build={build_s_total*1000:7.1f} ms")

    print()
    best = max(results.items(), key=lambda kv: kv[1]["top1_all"])
    adopt = best[1]["top1_all"] > shipped_top1 and \
        best[1]["p50_ms"] <= BUDGET_P50_MS
    print(f"best variant: {best[0]} top1(all)={best[1]['top1_all']:.4f} "
          f"(shipped {shipped_top1:.4f})")
    print(f"decision rule: adopt only if top1(all items) > shipped AND "
          f"p50 <= {BUDGET_P50_MS} ms")
    print("DECISION:", "ADOPT" if adopt else
          f"CUT — the deterministic router wins "
          f"({shipped_top1:.4f} vs {best[1]['top1_all']:.4f} on the "
          "arena's own scoring, abstention included); the embedding "
          "channel is a measured no-op for this corpus")


if __name__ == "__main__":
    main()
