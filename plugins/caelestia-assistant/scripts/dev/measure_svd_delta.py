"""A2 measurement: top-k surface retrieval accuracy, pre vs post.

Pre  = PpmiEmbedder       (PPMI + Achlioptas random projection, the status quo)
Post = PpmiSvdEmbedder    (PPMI + truncated SVD / LSA, A2)

Eval set: every corpus row whose surface has a real tool document (the
router's own semantic signal, isolated: embed the row text, rank every
registry tool document by cosine, check the surface's rank).

Supervision ablation: with a deterministic every-4th-row supervised split
(labeled pairs at LABEL_WEIGHT), evaluated on the OTHER rows — shows the
reroute-correction mechanism moves accuracy above the corpus prior.

Run: python3 scripts/dev/measure_svd_delta.py
Prints the numbers; asserts nothing. The committed test
(cortex/tests/test_svd_embedder.py) pins the direction (SVD >= RP at
top-1) so CI catches regressions without pinning floats.
"""

from __future__ import annotations

import time

from assistant.core.corpus import all_rows, tool_documents
from assistant.core.vectorize import (
    LABEL_WEIGHT,
    PpmiEmbedder,
    PpmiSvdEmbedder,
)


def topk_accuracy(embedder, rows, doc_vecs, ks=(1, 3, 5)):
    hits = {k: 0 for k in ks}
    n = 0
    for row in rows:
        q = embedder.embed(row.text)
        if not any(q):
            continue
        scored = sorted(
            ((embedder.cosine(q, dv), tool) for tool, dv in doc_vecs.items()),
            key=lambda pair: (-pair[0], pair[1]),
        )
        rank = next((i for i, (_, tool) in enumerate(scored, start=1)
                     if tool == row.surface), None)
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
    print(f"eval rows (tool-surfaced): {len(rows)} of {len(all_rows())}")
    print(f"documents: {len(docs)}")

    t0 = time.time()
    rp = PpmiEmbedder()
    rp_build = time.time() - t0
    rp_vecs = {t: rp.embed(d) for t, d in docs.items()}
    n, rp_acc = topk_accuracy(rp, rows, rp_vecs)
    print(f"\nPRE  PpmiEmbedder (random projection)  build {rp_build:.2f}s")
    print(f"     top-1 {rp_acc[1]:.4f}  top-3 {rp_acc[3]:.4f}  "
          f"top-5 {rp_acc[5]:.4f}   (n={n})")

    t0 = time.time()
    svd = PpmiSvdEmbedder()
    svd_build = time.time() - t0
    svd_vecs = {t: svd.embed(d) for t, d in docs.items()}
    n, svd_acc = topk_accuracy(svd, rows, svd_vecs)
    print(f"\nPOST PpmiSvdEmbedder (truncated SVD)   build {svd_build:.2f}s")
    print(f"     top-1 {svd_acc[1]:.4f}  top-3 {svd_acc[3]:.4f}  "
          f"top-5 {svd_acc[5]:.4f}   (n={n})")

    print("\ndelta (post - pre):")
    for k in (1, 3, 5):
        print(f"     top-{k}: {svd_acc[k] - rp_acc[k]:+.4f}")

    # Supervision ablation: every 4th row supervised, evaluated on the rest.
    supervised_rows = rows[::4]
    eval_rows = [r for i, r in enumerate(rows) if i % 4 != 0]
    pairs = [(r.text, r.surface) for r in supervised_rows]
    t0 = time.time()
    sup = PpmiSvdEmbedder(labeled_pairs=pairs, label_weight=LABEL_WEIGHT)
    sup_build = time.time() - t0
    sup_vecs = {t: sup.embed(d) for t, d in docs.items()}
    n_s, sup_acc = topk_accuracy(sup, eval_rows, sup_vecs)
    n_b, base_acc = topk_accuracy(svd, eval_rows, svd_vecs)
    print(f"\nABLATION SVD + {len(pairs)} supervised pairs "
          f"(weight {LABEL_WEIGHT}), build {sup_build:.2f}s")
    print(f"     corpus-only  top-1 {base_acc[1]:.4f}  top-3 {base_acc[3]:.4f}  top-5 {base_acc[5]:.4f}  (n={n_b})")
    print(f"     supervised  top-1 {sup_acc[1]:.4f}  top-3 {sup_acc[3]:.4f}  top-5 {sup_acc[5]:.4f}  (n={n_s})")
    print(f"     delta        top-1 {sup_acc[1] - base_acc[1]:+.4f}  "
          f"top-3 {sup_acc[3] - base_acc[3]:+.4f}  top-5 {sup_acc[5] - base_acc[5]:+.4f}")


if __name__ == "__main__":
    main()
