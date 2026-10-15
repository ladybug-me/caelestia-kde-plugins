"""A2 — PPMI + truncated SVD (LSA) embedder: capability, supervision, and
the measured verdict.

What these tests pin:

- BIT-IDENTICAL RP REFACTOR: PpmiEmbedder's outputs are fingerprint-pinned
  (the counting was factored into _count_pairs for the supervision seam;
  the projection path must not have moved a single float).
- SVD CAPABILITY: deterministic across builds, API-parity with the
  projection embedder (unknown words are zero vectors, cosine bounds),
  positive-part spectrum only, effective dim sane for the corpus.
- NUMERICS: the cyclic-Jacobi eigensolver is cross-verified against
  genius.linalg.power_iteration on random symmetric matrices (the
  primitive-reuse discipline: new math verified against the existing one).
- SUPERVISION: labeled (text, surface) pairs at LABEL_WEIGHT measurably
  improve the SVD embedder's held-out top-1 retrieval over its own
  corpus-only build (the reroute-correction seam works).
- THE MEASURED VERDICT: on TODAY's corpus the random projection still
  beats the SVD at top-1 — pinned as a direction test with the explicit
  instruction that a failure means "re-measure and flip the default
  deliberately", never "tune until green".
"""

from __future__ import annotations

import hashlib
import math
import random
import unittest

from assistant.core.corpus import all_rows, tool_documents
from assistant.core.vectorize import (
    LABEL_WEIGHT,
    PpmiEmbedder,
    PpmiSvdEmbedder,
    _jacobi_eigh,
)
from assistant.capabilities.genius import linalg


def _topk_accuracy(embedder, rows, doc_vecs, k=1):
    hits = 0
    n = 0
    for row in rows:
        q = embedder.embed(row.text)
        if not any(q):
            continue
        # Single pass: the target's rank = 1 + docs scoring above it
        # (deterministic tie-break by tool name, matching sorted order).
        target = doc_vecs.get(row.surface)
        if target is None:
            continue
        target_score = embedder.cosine(q, target)
        rank = 1
        for tool, dv in doc_vecs.items():
            if tool == row.surface:
                continue
            score = embedder.cosine(q, dv)
            if score > target_score or (score == target_score and tool < row.surface):
                rank += 1
        n += 1
        if rank <= k:
            hits += 1
    return hits / max(1, n)


class RefactorFingerprintTests(unittest.TestCase):
    def test_ppmi_projection_refactor_is_bit_identical(self) -> None:
        # Captured BEFORE the _count_pairs factoring (this session, on the
        # pre-change code): if this fingerprint moves, the supervision seam
        # changed the default build's arithmetic — a regression, not a
        # feature.
        e = PpmiEmbedder()
        probes = ["bar scale", "make the dock icons bigger",
                  "blur the frosted glass",
                  "notification popups launcher results",
                  "greeter morning start", "make it see-through",
                  "workspace pill overview"]
        blob = repr([[round(x, 12) for x in e.embed(p)] for p in probes])
        self.assertEqual(
            hashlib.sha256(blob.encode()).hexdigest()[:16], "588d52c5fc0ed311")
        # ^ pin spans the generated-registry corpus; it updates on
        # registry resyncs (e.g. the 2026-09-28 upstream tool-prune), never
        # on refactors of the embedder itself


class SvdCapabilityTests(unittest.TestCase):
    def test_deterministic_across_builds(self) -> None:
        e1, e2 = PpmiSvdEmbedder(), PpmiSvdEmbedder()
        self.assertEqual(e1.embed("bar scale"), e2.embed("bar scale"))
        self.assertEqual(e1.eigenvalues, e2.eigenvalues)

    def test_unknown_word_is_zero_vector(self) -> None:
        e = PpmiSvdEmbedder()
        self.assertEqual(e.word_vector("zzzqqqxyz"),
                         [0.0] * e.effective_dim)

    def test_effective_dim_is_positive_and_bounded(self) -> None:
        e = PpmiSvdEmbedder(dim=64)
        self.assertGreater(e.effective_dim, 0)
        self.assertLessEqual(e.effective_dim, 64)

    def test_positive_part_spectrum_only(self) -> None:
        e = PpmiSvdEmbedder()
        for lam in e.eigenvalues:
            self.assertGreater(lam, 0.0)

    def test_cosine_bounds(self) -> None:
        e = PpmiSvdEmbedder()
        value = e.similarity("bar scale", "dock icons")
        self.assertLessEqual(value, 1.0)
        self.assertGreaterEqual(value, -1.0)

    def test_neighbors_are_semantically_sane_and_deterministic(self) -> None:
        e = PpmiSvdEmbedder()
        n1 = e.neighbors("bar", k=5)
        n2 = e.neighbors("bar", k=5)
        self.assertEqual(n1, n2)
        words = [w for w, _ in n1]
        self.assertIn("taskbar", words)   # same registry noun family
        self.assertIn("panel", words)
        trans = e.neighbors("transparency", k=5)
        self.assertIn("opacity", [w for w, _ in trans])

    def test_neighbors_of_unknown_word_is_empty(self) -> None:
        e = PpmiSvdEmbedder()
        self.assertEqual(e.neighbors("zzzqqqxyz"), [])


class JacobiNumericsTests(unittest.TestCase):
    def test_jacobi_matches_power_iteration_on_random_symmetric(self) -> None:
        rng = random.Random(42)
        for trial in range(5):
            n = 12
            a = [[0.0] * n for _ in range(n)]
            for i in range(n):
                for j in range(i, n):
                    v = round(rng.uniform(-1.0, 1.0), 3)
                    a[i][j] = v
                    a[j][i] = v
            # A dominant diagonal boost gives a clear spectral gap, so
            # power iteration converges cleanly (near-degenerate random
            # spectra stall it — a known property of the primitive, not a
            # Jacobi bug; the cross-check needs the separated regime).
            a[0][0] += 6.0
            eigvals, eigvecs = _jacobi_eigh(a)
            # power_iteration converges to the eigenvalue LARGEST IN
            # MAGNITUDE (signed Rayleigh quotient); align Jacobi the same way.
            top = max(range(n), key=lambda i: abs(eigvals[i]))
            lam, vec = linalg.power_iteration(a)
            self.assertAlmostEqual(lam, eigvals[top], places=6)
            # Eigenvector agreement up to sign.
            dot = sum(vec[r] * eigvecs[r][top] for r in range(n))
            sign = 1.0 if dot >= 0 else -1.0
            for r in range(n):
                self.assertAlmostEqual(vec[r], sign * eigvecs[r][top],
                                       places=4)

    def test_jacobi_eigen_equation_holds(self) -> None:
        rng = random.Random(7)
        n = 9
        a = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                v = round(rng.uniform(-2.0, 2.0), 3)
                a[i][j] = v
                a[j][i] = v
        eigvals, vecs = _jacobi_eigh(a)
        for j in range(n):
            for r in range(n):
                lhs = sum(a[r][t] * vecs[t][j] for t in range(n))
                self.assertAlmostEqual(lhs, eigvals[j] * vecs[r][j],
                                       places=6)


class SupervisionTests(unittest.TestCase):
    """The reroute-correction seam: labeled pairs, weighted above the
    corpus prior, measurably improve the SVD embedder's held-out accuracy
    over its own corpus-only build (measured +0.7pt top-1 this session;
    deterministic, so the relation is pinned)."""

    @classmethod
    def setUpClass(cls) -> None:
        docs = tool_documents()
        rows = [r for r in all_rows() if r.surface in docs]
        cls.docs = docs
        # Deterministic split: every 4th row supervised, every 4th of the
        # remainder evaluated (keeps the eval fast; the build dominates).
        cls.pairs = [(r.text, r.surface) for r in rows[::4]]
        cls.eval_rows = [r for i, r in enumerate(rows) if i % 4 != 0][::4]
        cls.base = PpmiSvdEmbedder()
        cls.sup = PpmiSvdEmbedder(labeled_pairs=cls.pairs,
                                  label_weight=LABEL_WEIGHT)

    def test_supervision_improves_held_out_top1(self) -> None:
        base_vecs = {t: self.base.embed(d) for t, d in self.docs.items()}
        sup_vecs = {t: self.sup.embed(d) for t, d in self.docs.items()}
        base_acc = _topk_accuracy(self.base, self.eval_rows, base_vecs, k=1)
        sup_acc = _topk_accuracy(self.sup, self.eval_rows, sup_vecs, k=1)
        self.assertGreaterEqual(
            sup_acc, base_acc,
            "supervised SVD should hold or improve held-out top-1; "
            f"got supervised={sup_acc:.4f} < corpus-only={base_acc:.4f} "
            "(re-measure scripts/measure_svd_delta.py if this flipped)")

    def test_supervision_changes_the_geometry(self) -> None:
        self.assertNotEqual(self.base.embed("bar scale"),
                            self.sup.embed("bar scale"))

    def test_supervised_build_is_deterministic(self) -> None:
        again = PpmiSvdEmbedder(labeled_pairs=self.pairs,
                                label_weight=LABEL_WEIGHT)
        self.assertEqual(again.embed("blur the frosted glass"),
                         self.sup.embed("blur the frosted glass"))

    def test_rp_embedder_accepts_labeled_pairs_too(self) -> None:
        rp = PpmiEmbedder(labeled_pairs=self.pairs[:50])
        plain = PpmiEmbedder()
        self.assertNotEqual(rp.embed("bar scale"), plain.embed("bar scale"))
        # Both remain deterministic.
        rp2 = PpmiEmbedder(labeled_pairs=self.pairs[:50])
        self.assertEqual(rp.embed("bar scale"), rp2.embed("bar scale"))


class MeasuredVerdictTests(unittest.TestCase):
    """The A2 verification, numbers not claims: on TODAY's corpus the
    random projection beats the SVD at top-1. This is a DIRECTION pin on a
    measured, deterministic comparison. If it ever fails, the corpus regime
    changed (organic vocabulary growth) — re-run
    scripts/measure_svd_delta.py, and if SVD now wins, flip the router
    default DELIBERATELY with the fresh numbers recorded; never tune the
    test to get green."""

    def test_random_projection_still_wins_top1_on_current_corpus(self) -> None:
        docs = tool_documents()
        rows = [r for r in all_rows() if r.surface in docs]
        # Deterministic subsample keeps the test fast (build dominates).
        rows = rows[::9]
        rp = PpmiEmbedder()
        svd = PpmiSvdEmbedder()
        rp_vecs = {t: rp.embed(d) for t, d in docs.items()}
        svd_vecs = {t: svd.embed(d) for t, d in docs.items()}
        rp_acc = _topk_accuracy(rp, rows, rp_vecs, k=1)
        svd_acc = _topk_accuracy(svd, rows, svd_vecs, k=1)
        self.assertGreaterEqual(
            rp_acc, svd_acc,
            f"RP top-1 {rp_acc:.4f} < SVD top-1 {svd_acc:.4f}: the corpus "
            "regime changed — re-measure before touching the default")


if __name__ == "__main__":
    unittest.main()
