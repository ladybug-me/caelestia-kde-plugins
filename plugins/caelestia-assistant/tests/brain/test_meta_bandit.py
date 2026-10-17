"""Tests for exponential-build 3 item B3: the hierarchical meta-bandit
over the recommendation engines (brain/meta_bandit.py — Thompson
sampling per Chapelle & Li 2011 over per-context Beta posteriors pooled
upward through brain/pooling.py's Efron-Morris 1975 helpers, contexts
matched via brain/features.py's signed feature hashing).

The contract under test:

- SYNTHETIC RECOVERY: three engines with context-dependent correctness
  (thompson best in context 1, linucb best in context 2, adagrad weak
  everywhere); after a fixed-seed 300-round simulated run
  (random.Random(7), rewards by common random numbers so every policy
  sees the SAME stream), the meta-bandit's per-context empirical choice
  over the last 50 rounds matches the TRUE best engine in BOTH
  contexts, and the globally-weak engine is suppressed — all counts
  pinned exactly;
- REGRET AUDIT: the draw log goes through brain/regret.py's
  audit_from_draws (engines as arms, not forked); the meta-bandit beats
  EVERY fixed engine on the realized common-random-numbers stream;
  the audit's own ESTIMATED regret (vs the estimated best-fixed
  baseline) is pinned — including the honest finding that the estimate
  is context-contaminated and therefore OPTIMISTIC for the baseline
  (pinned direction), which is exactly why regret.py labels it an
  estimate, never a bound;
- COLD-START HONESTY: before any update the report says "no evidence:
  engines indistinguishable"; unknown engines are REJECTED with
  ValueError; the engine callables are NEVER invoked by choose();
- HIERARCHY: evidence in one context shifts the prior used for an
  UNSEEN context (direction pinned, exact values hand-derived);
  disagreeing contexts pool weakly so per-context specialization
  survives;
- DETERMINISM: identical runs produce identical draw logs, and the
  persistence round-trip is byte-exact with mismatched state refused.
"""
import hashlib
import random
import unittest

from assistant.capabilities.brain.meta_bandit import CONTEXT_DIMENSIONS, MetaBandit

# The three engines mirror the live recommendation engines the module
# was designed to sit over (cortex/learn.py's strategy bandit, the
# LinUCB preset bandit, the AdaGrad router) — here they are stubs that
# RECORD every call, so the never-invoked invariant is pinnable.
CALLS = []


def _engine(name):
    def fn(context):
        CALLS.append(name)
        return f"{name}:{context}"
    return fn


ENGINES = {name: _engine(name)
           for name in ("thompson", "linucb", "adagrad")}

CONTEXTS = [
    "settings panel appearance requests",   # context 1
    "weather forecast questions",           # context 2
    "calendar schedule planning",           # context 3 (unseen in the sim)
]
CTX1, CTX2, CTX3 = CONTEXTS

# round texts: two phrasing variants per context — NOT the descriptors
# themselves, so every choose()/update() goes through the hashed
# context-matching path
TEXTS = [
    "change the settings panel appearance",    # -> context 1
    "what is the weather forecast today",      # -> context 2
    "adjust panel appearance settings again",  # -> context 1
    "will it rain weather forecast please",    # -> context 2
]
CTX_OF_TEXT = {TEXTS[0]: CTX1, TEXTS[1]: CTX2, TEXTS[2]: CTX1, TEXTS[3]: CTX2}

# true accept probabilities: thompson wins context 1, linucb wins
# context 2, adagrad is weak everywhere (the "overall-weak" engine)
PROBS = {
    CTX1: {"thompson": 0.85, "linucb": 0.20, "adagrad": 0.10},
    CTX2: {"thompson": 0.20, "linucb": 0.85, "adagrad": 0.10},
}


def crn(t, engine, ctx):
    """Common random numbers: a deterministic Bernoulli per (round,
    engine, context) — blake2b, no RNG state — so every policy (the
    meta-bandit AND each fixed engine) sees the SAME reward stream and
    the fixed-engine counterfactuals are exact, not resampled."""
    payload = f"t4-meta:{t}:{engine}:{ctx}".encode("utf-8")
    u = int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(), "big") / 2.0 ** 64
    return 1 if u < PROBS[ctx][engine] else 0


def _simulate(rounds=300, seed=7):
    """The fixed-seed simulated run: alternate contexts, Thompson-choose
    (rng injected), reward by common random numbers, update."""
    mb = MetaBandit(CONTEXTS, ENGINES)
    rng = random.Random(seed)
    log = []
    for t in range(rounds):
        text = TEXTS[t % 4]
        engine = mb.choose(text, rng)
        ctx = CTX_OF_TEXT[text]
        reward = crn(t, engine, ctx)
        mb.update(text, engine, reward)
        log.append((t, ctx, engine, reward))
    return mb, log


MB, LOG = _simulate()  # deterministic: module-level, computed once


class ContextResolutionTests(unittest.TestCase):
    """Contexts are MATCHED through features.py's signed hashing (the
    ONE hashing implementation — imported, not reimplemented)."""

    def test_exact_descriptor_passes_through(self):
        mb = MetaBandit(CONTEXTS, ENGINES)
        self.assertEqual(mb.resolve_context(CTX2), CTX2)

    def test_variant_texts_resolve_to_their_context(self):
        mb = MetaBandit(CONTEXTS, ENGINES)
        for text, expected in CTX_OF_TEXT.items():
            self.assertEqual(mb.resolve_context(text), expected, text)

    def test_matching_runs_at_the_wide_end_of_the_documented_range(self):
        # d=64 (the top of features.py's documented 8..64 caller range):
        # matching is a nearest-vector lookup where collisions are pure
        # noise — at the shared-state default d=16 the "will it rain
        # weather forecast please" text LOSES its genuine two-token
        # overlap to a coincidental same-bucket collision with the
        # calendar descriptor (verified during the build; pinned here as
        # the reason the matching dimension is not the default 16)
        self.assertEqual(CONTEXT_DIMENSIONS, 64)

    def test_tokenless_input_resolves_to_the_first_descriptor(self):
        mb = MetaBandit(CONTEXTS, ENGINES)
        # "!!!" tokenizes to nothing -> zero vector -> every cosine ties
        # -> the fixed first-in-order tie-break decides, documented
        self.assertEqual(mb.resolve_context("!!!"), CTX1)


class ColdStartTests(unittest.TestCase):
    def test_report_says_no_evidence_before_any_update(self):
        mb = MetaBandit(CONTEXTS, ENGINES)
        report = mb.report()
        self.assertEqual(report["status"],
                         "no evidence: engines indistinguishable")
        self.assertEqual(report["draws"], 0)
        self.assertIn("cold-start pooled prior", report["note"])

    def test_choose_returns_a_valid_engine_and_explores_the_flat_prior(self):
        mb = MetaBandit(CONTEXTS, ENGINES)
        rng = random.Random(7)
        seen = {mb.choose(CTX1, rng) for _ in range(300)}
        # flat Beta(1,1) everywhere: all three engines surface — the
        # bandit is honestly guessing, not silently pinned to one
        self.assertEqual(seen, {"thompson", "linucb", "adagrad"})

    def test_choose_never_invokes_the_engine_callables(self):
        # the meta-bandit returns WHICH engine to trust; the CALLER
        # executes it — 300 simulated rounds, zero engine calls
        self.assertEqual(CALLS, [])


class RecoveryTests(unittest.TestCase):
    """THE synthetic-recovery pin: after 300 fixed-seed rounds the
    per-context empirical choice matches the true best engine in BOTH
    contexts (majority over the last 50 rounds of each context)."""

    def test_majority_over_last_50_rounds_matches_truth_in_both_contexts(self):
        for ctx, true_best, pinned in (
                (CTX1, "thompson", {"thompson": 48, "linucb": 1, "adagrad": 1}),
                (CTX2, "linucb", {"linucb": 50})):
            draws = [e for (_t, c, e, _r) in LOG if c == ctx][-50:]
            counts = {}
            for e in draws:
                counts[e] = counts.get(e, 0) + 1
            self.assertEqual(counts, pinned, ctx)
            self.assertEqual(max(counts, key=counts.get), true_best, ctx)

    def test_globally_weak_engine_is_suppressed(self):
        totals = {}
        for (_t, _c, e, _r) in LOG:
            totals[e] = totals.get(e, 0) + 1
        # pinned: the weak engine barely gets played once the pooled
        # hierarchy (weak in BOTH contexts -> agreeing evidence ->
        # full pooling) has sharpened against it
        self.assertEqual(totals, {"thompson": 152, "linucb": 145,
                                  "adagrad": 3})

    def test_first_eight_rounds_pinned(self):
        # the exact early exploration sequence (flat priors, then the
        # first evidence arriving) — the strongest determinism pin
        self.assertEqual([(c, e) for (_t, c, e, _r) in LOG[:8]],
                         [(CTX1, "linucb"), (CTX2, "adagrad"),
                          (CTX1, "thompson"), (CTX2, "thompson"),
                          (CTX1, "thompson"), (CTX2, "thompson"),
                          (CTX1, "adagrad"), (CTX2, "linucb")])
        self.assertEqual([r for (_t, _c, _e, r) in LOG[:8]],
                         [0, 0, 1, 1, 0, 0, 0, 1])


class RegretAuditTests(unittest.TestCase):
    """The audit runs through brain/regret.py's audit_from_draws with
    engines as arms (the shape fits exactly — the regret math is not
    forked). Fixed-engine counterfactuals use the SAME common-random-
    numbers stream, so the comparison is realized, not resampled."""

    def test_fixed_engine_totals_on_the_same_stream(self):
        ctx_seq = [CTX_OF_TEXT[TEXTS[t % 4]] for t in range(300)]
        fixed = {x: sum(crn(t, x, ctx_seq[t]) for t in range(300))
                 for x in ENGINES}
        self.assertEqual(fixed, {"thompson": 155, "linucb": 161,
                                 "adagrad": 30})

    def test_meta_bandit_beats_every_fixed_engine_on_the_realized_stream(self):
        ctx_seq = [CTX_OF_TEXT[TEXTS[t % 4]] for t in range(300)]
        fixed = {x: sum(crn(t, x, ctx_seq[t]) for t in range(300))
                 for x in ENGINES}
        best_fixed = max(fixed.values())
        meta_total = sum(r for (_t, _c, _e, r) in LOG)
        self.assertEqual(meta_total, 241)  # pinned
        realized_regret = best_fixed - meta_total
        # the meta-bandit's realized regret (-80) is <= EVERY fixed
        # engine's realized regret (best-fixed linucb: 0, thompson: 6,
        # adagrad: 131): specializing per context beats always playing
        # any single engine
        for x, total in fixed.items():
            self.assertLessEqual(realized_regret, best_fixed - total,
                                 f"vs fixed {x}")

    def test_audit_runs_through_regret_py_with_engines_as_arms(self):
        out = MB.audit()
        self.assertEqual(out["status"], "compared")
        self.assertEqual(out["horizon"], 300)
        self.assertEqual(out["cumulative_reward"], 241.0)
        self.assertEqual(out["best_fixed_arm"], "linucb")
        plays = {a["arm"]: a["plays"] for a in out["per_arm"]}
        self.assertEqual(plays, {"adagrad": 3.0, "linucb": 145.0,
                                 "thompson": 152.0})
        self.assertEqual(out["per_context_rounds"], {CTX1: 150, CTX2: 150})
        # the estimate caveat regret.py already carries travels intact
        self.assertIn("estimate", out["method"].lower())
        self.assertIn("not a bound", out["method"])
        # ...plus the meta-level restatement
        self.assertIn("context-blind", out["meta_note"])
        self.assertIn("context-contaminated", out["meta_note"])

    def test_the_estimated_regret_is_pinned_and_honestly_caveated(self):
        # WHAT IS TRUE, pinned: the audit's estimated regret (+9.3449)
        # is well under the WORST fixed engine's realized regret (131),
        # but it EXCEEDS the two better fixed engines' realized regrets
        # (thompson 6, linucb 0) — because the estimated best-fixed
        # baseline is CONTEXT-CONTAMINATED: linucb was played mostly in
        # the context where it wins, so its observed mean (121/145 =
        # 0.8345) times the horizon (250.3449) overestimates what
        # always-linucb actually collects on this stream (161). The
        # audit prints an estimate over unobserved rounds, never a
        # bound; the meta_note says exactly this. No overclaim.
        out = MB.audit()
        self.assertEqual(out["estimated_regret"], 9.3449)
        self.assertEqual(out["best_fixed_estimate"], 250.3449)
        self.assertLessEqual(out["estimated_regret"], 131)  # worst fixed
        self.assertGreater(out["best_fixed_estimate"], 161)  # contamination
        ctx_seq = [CTX_OF_TEXT[TEXTS[t % 4]] for t in range(300)]
        always_linucb = sum(crn(t, "linucb", ctx_seq[t]) for t in range(300))
        self.assertEqual(always_linucb, 161)

    def test_untouched_bandit_audit_abstains(self):
        out = MetaBandit(CONTEXTS, ENGINES).audit()
        self.assertEqual(out["status"], "abstained")


class HierarchyTests(unittest.TestCase):
    """The Efron-Morris pooling (brain/pooling.py, REUSED) is the
    hierarchy: a cold context borrows the population's opinion;
    disagreeing contexts pool weakly so specialization survives."""

    def test_evidence_in_context_1_shifts_the_prior_for_unseen_context_3(self):
        # hand-derived: 9 accepts / 1 reject for thompson in context 1
        # only. pool_arms sees a SINGLE evidence arm -> tau2 = 0 ->
        # FULL pooling, k0 = 10 -> hierarchical prior Beta(9, 1); the
        # unseen context 3 starts there (mean 0.9, pulled UP from the
        # flat 0.5 toward context 1's observed rate — the direction).
        mb = MetaBandit(CONTEXTS, ENGINES)
        for _ in range(9):
            mb.update(CTX1, "thompson", True)
        mb.update(CTX1, "thompson", False)
        p3 = mb.posterior(CTX3, "thompson")
        self.assertEqual((p3["alpha"], p3["beta"]), (9.0, 1.0))
        self.assertEqual(p3["mean"], 0.9)
        # an engine with no evidence ANYWHERE stays flat (pooling's own
        # honesty rule: nothing to borrow from -> the flat prior stands)
        p3b = mb.posterior(CTX3, "linucb")
        self.assertEqual((p3b["alpha"], p3b["beta"]), (1.0, 1.0))
        self.assertEqual(p3b["mean"], 0.5)

    def test_sole_evidence_context_preserves_its_mean_sharpens_confidence(self):
        # the population prior is fit from ALL contexts including the
        # one being scored (the classic Efron-Morris construction): with
        # context 1 the only evidence source, its own rate sets the
        # prior mean, so the posterior MEAN is unchanged (0.9) and only
        # the confidence sharpens — Beta(18, 2) from 10 real events.
        # Stated in the module docstring, pinned here, not hidden.
        mb = MetaBandit(CONTEXTS, ENGINES)
        for _ in range(9):
            mb.update(CTX1, "thompson", True)
        mb.update(CTX1, "thompson", False)
        p1 = mb.posterior(CTX1, "thompson")
        self.assertEqual((p1["alpha"], p1["beta"]), (18.0, 2.0))
        self.assertEqual(p1["mean"], 0.9)

    def test_disagreeing_contexts_pool_weakly_specialization_survives(self):
        # after the full run: context 1's effective posterior still
        # favors thompson and context 2's favors linucb (pinned exact
        # means) — the between-context disagreement keeps tau2 large /
        # k0 small, so the pool never swamps the per-context evidence
        r1 = {e: MB.posterior(CTX1, e)["mean"] for e in ENGINES}
        r2 = {e: MB.posterior(CTX2, e)["mean"] for e in ENGINES}
        self.assertEqual(r1, {"thompson": 0.804184, "linucb": 0.156432,
                              "adagrad": 0.090909})
        self.assertEqual(r2, {"thompson": 0.681408, "linucb": 0.855973,
                              "adagrad": 0.111111})
        self.assertGreater(r1["thompson"], r1["linucb"])
        self.assertGreater(r2["linucb"], r2["thompson"])

    def test_cold_context_inherits_global_structure(self):
        # the UNSEEN context 3 after the full run: the globally-weak
        # engine is suppressed there (agreeing evidence pools fully),
        # while both plausible engines stay above it — the hierarchy
        # transfers population structure to cold contexts
        r3 = {e: MB.posterior(CTX3, e)["mean"] for e in ENGINES}
        self.assertEqual(r3["adagrad"], 0.142857)
        self.assertLess(r3["adagrad"], 0.5)
        self.assertGreater(r3["thompson"], 0.5)
        self.assertGreater(r3["linucb"], 0.5)


class InterfaceTests(unittest.TestCase):
    def test_unknown_engine_update_is_rejected(self):
        mb = MetaBandit(CONTEXTS, ENGINES)
        with self.assertRaises(ValueError) as ctx:
            mb.update(CTX1, "bert", True)
        self.assertIn("thompson", str(ctx.exception))

    def test_unknown_engine_posterior_is_rejected(self):
        mb = MetaBandit(CONTEXTS, ENGINES)
        with self.assertRaises(ValueError):
            mb.posterior(CTX1, "nope")

    def test_bad_construction_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            MetaBandit(CONTEXTS, {})           # no engines
        with self.assertRaises(ValueError):
            MetaBandit(CONTEXTS, {"x": 42})    # not callable
        with self.assertRaises(ValueError):
            MetaBandit([], ENGINES)            # no contexts
        with self.assertRaises(ValueError):
            MetaBandit([CTX1, CTX1], ENGINES)  # duplicate descriptors

    def test_determinism_two_identical_runs(self):
        mb_a, log_a = _simulate()
        mb_b, log_b = _simulate()
        self.assertEqual(log_a, log_b)
        self.assertEqual(mb_a.draws, mb_b.draws)
        self.assertEqual(mb_a.to_dict(), mb_b.to_dict())

    def test_default_rng_is_seeded_and_reproducible(self):
        a = MetaBandit(CONTEXTS, ENGINES)
        b = MetaBandit(CONTEXTS, ENGINES)
        seq_a = [a.choose(CTX1) for _ in range(10)]
        seq_b = [b.choose(CTX1) for _ in range(10)]
        self.assertEqual(seq_a, seq_b)
        self.assertEqual(seq_a[:4], ["thompson", "thompson", "linucb",
                                     "thompson"])

    def test_persistence_round_trip_is_exact(self):
        data = MB.to_dict()
        rebuilt = MetaBandit.from_dict(data, CONTEXTS, ENGINES)
        self.assertEqual(rebuilt.to_dict(), data)
        self.assertEqual(rebuilt.draws, MB.draws)
        self.assertEqual(rebuilt.audit(), MB.audit())

    def test_persistence_refuses_mismatched_state(self):
        data = MB.to_dict()
        bad_ctx = dict(data, state={**data["state"], "mystery ctx": {}})
        with self.assertRaises(ValueError):
            MetaBandit.from_dict(bad_ctx, CONTEXTS, ENGINES)
        bad_engine = dict(data, draws=[["mystery ctx", "thompson", 1]])
        with self.assertRaises(ValueError):
            MetaBandit.from_dict(bad_engine, CONTEXTS, ENGINES)
        # a non-dict payload is a fresh cold start, not a crash
        fresh = MetaBandit.from_dict(None, CONTEXTS, ENGINES)
        self.assertEqual(fresh.report()["draws"], 0)

    def test_learning_report_carries_the_audit(self):
        report = MB.report()
        self.assertEqual(report["status"], "learning")
        self.assertEqual(report["draws"], 300)
        self.assertIn("regret", report)
        self.assertEqual(report["regret"]["best_fixed_arm"], "linucb")


if __name__ == "__main__":
    unittest.main()
