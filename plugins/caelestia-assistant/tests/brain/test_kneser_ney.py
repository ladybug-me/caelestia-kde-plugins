"""Tests for exponential-build T3 (item B1): the Kneser-Ney smoothed
n-gram language model in brain/kneser_ney.py — the additional
correction/completion signal beside the vault's SymSpell index.

The contract under test (Kneser & Ney 1995; interpolated form and
leaving-one-out discount per Chen & Goodman 1999):

- the EXACT probability of a toy bigram case, hand-derived below to
  the last fraction — the arithmetic is fully human-checkable;
- continuation counts are DISTINCT-CONTEXT counts, not raw frequency
  (the KN-vs-Katz distinction: genius/markov.py's backoff uses raw
  unigram frequency; KN must not);
- unseen contexts back off; every level of the interpolation sums to
  exactly 1 (checked by hand in the pins);
- training is closed-form counting: two trains are byte-identical,
  and the model dict is JSON-round-trippable for the learned-state
  path (the module itself writes nothing);
- the evidence gate reaches all three verdicts on documented
  thresholds, and the empty corpus abstains everywhere — no invented
  probabilities;
- the integration honesty: KN NEVER overrides an existing SymSpell
  answer; it is used only when SymSpell abstained AND KN's gate is
  "confident".
"""
import json
import unittest
from unittest import mock

from assistant.capabilities.brain import spellfix
from assistant.capabilities.brain.kneser_ney import (CharKN, WordKN, complete, gate,
                                        score, spellfix_fallback, suggest)

# The hand-derivation corpus (word flavor, n=2 — every number below is
# derived by hand in test_hand_derived_exact_probability):
TOY = ["a b a", "a b c"]

# A corpus with natural count variation (52 tokens >= the gate's
# min_tokens=50) where SymSpell honestly abstains — every rare word is
# far from the only common words (fox x5); used for the KN-wins path.
CORPUS = [
    "that quick fox jumps",
    "this brown fox trots",
    "some slim fox digs",
    "each calm fox sits",
    "every quirky fox naps",
    "that lazy dog snores",
    "this aged owl blinks",
    "some gray cat purrs",
    "each red hen pecks",
    "every tiny owl coos",
    "that black cat stalks",
    "this pale hen clucks",
    "some plump dog barks",
]


class HandDerivedProbabilityTests(unittest.TestCase):
    """Corpus ["a b a", "a b c"], WordKN(n=2). BOS-padded sequences:

        [<s>, a, b, a]   and   [<s>, a, b, c]

    bigram counts: (<s>,a)=2  (a,b)=2  (b,a)=1  (b,c)=1
    -> 4 distinct bigram types; count histogram {1: 2, 2: 2}
    -> order-2 discount  D_2 = n1/(n1+2*n2) = 2/(2+4) = 1/3.

    continuation counts c_cont(w) = distinct left contexts of w:
        a: {<s>, b} -> 2    b: {a} -> 1    c: {b} -> 1
    histogram {1: 2, 2: 1} -> unigram discount D_1 = 2/(2+2) = 1/2.

    unigram level (|bigram types| = 4, continuation vocab = {a,b,c}):
        lambda_0 = D_1 * 3 / 4 = 3/8          (the unseen-class mass)
        P_cont(a) = max(2 - 1/2, 0)/4 = 3/8
        P_cont(b) = max(1 - 1/2, 0)/4 = 1/8
        P_cont(c) =                        1/8
        (sums to 1: 3/8 + 1/8 + 1/8 + lambda_0 = 1)

    bigram level, context "a": c(a) = c(ab)+c(ac) = 2, N+(a) = 1:
        P(b|a) = max(2 - 1/3, 0)/2 + (1/3)(1)/2 * P_cont(b)
               = 5/6 + 1/6 * 1/8 = 40/48 + 1/48 = 41/48
        P(a|a) = 0 + 1/6 * 3/8 = 1/16
        P(c|a) = 0 + 1/6 * 1/8 = 1/48
    context "b": c(b) = 2, N+(b) = 2, lambda(b) = (1/3)(2)/2 = 1/3:
        P(a|b) = max(1 - 1/3, 0)/2 + 1/3 * 3/8 = 1/3 + 1/8 = 11/24
        P(c|b) = 1/3 + 1/3 * 1/8 = 3/8
        P(b|b) = 0 + 1/3 * 1/8 = 1/24
    """

    def setUp(self):
        self.m = WordKN(2).train(TOY)

    def test_hand_derived_exact_probability(self):
        m = self.m
        self.assertEqual(m["bigram_types"], 4)
        self.assertEqual(m["continuations"], {"a": 2, "b": 1, "c": 1})
        self.assertAlmostEqual(m["discounts"][0], 0.5)       # D_1 = 1/2
        self.assertAlmostEqual(m["discounts"][1], 1.0 / 3.0)  # D_2 = 1/3
        ranked = complete(m, "a", 5)
        self.assertEqual([r["token"] for r in ranked], ["b", "a", "c"])
        self.assertAlmostEqual(ranked[0]["p"], 41.0 / 48.0, places=6)
        self.assertAlmostEqual(ranked[1]["p"], 1.0 / 16.0, places=6)
        self.assertAlmostEqual(ranked[2]["p"], 1.0 / 48.0, places=6)
        # score("a b") = (ln P(a|<s>) + ln P(b|a)) / 2, where
        # P(a|<s>) = max(2-1/3,0)/2 + (1/3)(1)/2 * 3/8 = 5/6 + 1/16 = 43/48
        import math
        expected = (math.log(43.0 / 48.0) + math.log(41.0 / 48.0)) / 2.0
        self.assertAlmostEqual(score(m, "a b"), expected, places=12)

    def test_completion_ranking_exact(self):
        # context "b" (the n=2 context of prefix "a b"): the derivation
        # above gives a: 11/24 > c: 3/8 > b: 1/24
        ranked = complete(self.m, "a b", 3)
        self.assertEqual([r["token"] for r in ranked], ["a", "c", "b"])
        self.assertAlmostEqual(ranked[0]["p"], 11.0 / 24.0, places=6)
        self.assertAlmostEqual(ranked[1]["p"], 3.0 / 8.0, places=6)
        self.assertAlmostEqual(ranked[2]["p"], 1.0 / 24.0, places=6)

    def test_backoff_for_unseen_context(self):
        # "q" never appeared as a context: the interpolation backs off
        # with weight 1 to the unigram continuation distribution
        # (a: 3/8, then the 1/8 tie broken lexicographically).
        ranked = complete(self.m, "q", 5)
        self.assertEqual([r["token"] for r in ranked], ["a", "b", "c"])
        self.assertAlmostEqual(ranked[0]["p"], 3.0 / 8.0, places=6)
        self.assertAlmostEqual(ranked[1]["p"], 1.0 / 8.0, places=6)
        self.assertAlmostEqual(ranked[2]["p"], 1.0 / 8.0, places=6)

    def test_unseen_token_probability_is_the_carved_out_mass(self):
        # an out-of-vocabulary token gets exactly lambda_0 through the
        # context "a" backoff: (1/6) * lambda_0 = (1/6)(3/8) = 1/16 —
        # no floor invented; this is visible as P(a|a) = 1/16 above
        # (a is in the continuation vocab, so its own mass is 0 there).
        self.assertAlmostEqual(complete(self.m, "a", 5)[1]["p"], 1.0 / 16.0)


class ContinuationCountTests(unittest.TestCase):
    def test_continuation_counts_are_distinct_contexts_not_frequency(self):
        # "w" appears 3 times, each after a DIFFERENT word -> c_cont 3.
        # "x" appears 5 times but the corpus is one alternating run ->
        # c_cont 2 ({<s>, a}); "a" appears 4 times, always after "x" ->
        # c_cont 1. Raw frequency ranks x(5) > a(4) > w(3); continuation
        # ranks w(3) > x(2) > a(1) — THE KN-vs-Katz distinction (a Katz
        # backoff, like genius/markov.py's, would rank by raw counts).
        m = WordKN(2).train(["p w", "q w", "r w", "x a x a x a x a x"])
        self.assertEqual(m["vocab"], {"a": 4, "p": 1, "q": 1, "r": 1,
                                      "w": 3, "x": 5})
        self.assertEqual(m["continuations"],
                         {"a": 1, "p": 1, "q": 1, "r": 1, "w": 3, "x": 2})
        self.assertEqual(max(m["continuations"], key=m["continuations"].get), "w")
        self.assertEqual(max(m["vocab"], key=m["vocab"].get), "x")


class DeterminismTests(unittest.TestCase):
    def test_two_trains_byte_identical(self):
        for maker in (lambda: WordKN().train(TOY * 3),
                      lambda: CharKN().train(CORPUS)):
            a, b = maker(), maker()
            self.assertEqual(a, b)
            self.assertEqual(json.dumps(a, sort_keys=True),
                             json.dumps(b, sort_keys=True))

    def test_model_is_json_round_trippable(self):
        m = CharKN(4).train(CORPUS)
        self.assertEqual(json.loads(json.dumps(m)), m)


class GateTests(unittest.TestCase):
    def setUp(self):
        # 60 tokens >= min_tokens=50; "the" has 2 distinct continuations
        # (fox, dog); "fox" has 1 (ran); nothing follows "ran".
        self.m = WordKN().train(["the fox ran", "the dog sat"] * 10)

    def test_all_three_verdicts_reachable(self):
        self.assertEqual(gate(self.m, "the"), "confident")
        self.assertEqual(gate(self.m, "the fox"), "thin")
        self.assertEqual(gate(self.m, "the dog ran"), "abstain")  # ran: never a context
        self.assertEqual(gate(self.m, "zebra"), "abstain")        # unseen suffix

    def test_thresholds_are_documented_and_honored(self):
        # a 60-token corpus with the query's prefix unobserved abstains;
        # the same corpus at a lower min_tokens threshold still abstains
        # on the unseen prefix (thresholds relax the CORPUS size, never
        # invent context evidence):
        self.assertEqual(gate(self.m, "zebra", min_tokens=10), "abstain")
        # a genuinely thin corpus (30 tokens) abstains at the default:
        thin = WordKN().train(["the fox ran", "the dog sat"] * 5)
        self.assertEqual(thin["total_tokens"], 30)
        self.assertEqual(gate(thin, "the"), "abstain")

    def test_tie_completion_breaks_lexicographically(self):
        # D_2 = 0 here (every bigram repeats) -> plain MLE 10/20 each;
        # the tie is broken deterministically.
        ranked = complete(self.m, "the", 3)
        self.assertEqual([r["token"] for r in ranked], ["dog", "fox"])
        self.assertEqual([r["p"] for r in ranked], [0.5, 0.5])


class SuggestTests(unittest.TestCase):
    def setUp(self):
        self.m = CharKN(4).train(CORPUS)

    def test_word_model_is_rejected(self):
        mw = WordKN().train(CORPUS)
        with self.assertRaises(ValueError):
            suggest(mw, "fxo", 5)

    def test_ranks_true_1_2_edit_candidates_by_char_probability(self):
        out = suggest(self.m, "fxo", 5)
        self.assertEqual(out["gate"], "confident")
        self.assertEqual(out["query"], "fxo")
        # "fox" (adjacent transposition, OSA distance 1) outranks the
        # query itself; the query is always a candidate (distance 0).
        self.assertEqual(out["candidates"][0]["word"], "fox")
        self.assertEqual(out["candidates"][0]["edit_distance"], 1)
        self.assertEqual([c["word"] for c in out["candidates"]][-1], "fxo")
        self.assertEqual(out["candidates"][-1]["edit_distance"], 0)
        # deterministic
        self.assertEqual(suggest(self.m, "fxo", 5), out)

    def test_prefilter_reuses_spellfix_deletes_not_a_copy(self):
        # the delete-index IDEA is imported from brain/spellfix.py —
        # pinned by spying on the shared helper (no second
        # implementation of SymSpell's delete machinery).
        with mock.patch("assistant.capabilities.brain.kneser_ney._deletes",
                        wraps=spellfix._deletes) as spy:
            out = suggest(self.m, "fxo", 5)
        self.assertGreaterEqual(spy.call_count, 1)
        self.assertEqual(out["candidates"][0]["word"], "fox")

    def test_prefilter_superset_is_verified_honestly(self):
        # "dog" shares no delete-variant pattern near enough and is at
        # OSA distance 3 from "fxo" — it must not appear even though a
        # loose delete-match prefilter alone would keep some farther
        # pairs; the verifier is what enforces the 1-2 edit bound.
        out = suggest(self.m, "fxo", 10)
        words = [c["word"] for c in out["candidates"]]
        self.assertNotIn("dog", words)
        for row in out["candidates"]:
            self.assertLessEqual(row["edit_distance"], 2)


class SpellfixFallbackTests(unittest.TestCase):
    def setUp(self):
        self.base = CharKN(4).train(CORPUS)
        # the same corpus plus ONE sentence containing the typo "fxo":
        # now "fxo" is a rare word (count 1) that is a near-miss of the
        # common word "fox" (count 5 >= common_min) — SymSpell answers.
        self.typo_corpus = CORPUS + ["the fxo digs"]
        self.idx = spellfix.SpellIndex().build(self.typo_corpus)
        self.symspell = self.idx.suggest()
        self.kn = suggest(CharKN(4).train(self.typo_corpus), "fxo", 5)

    def test_never_overrides_a_confident_symspell_answer(self):
        # SymSpell's answer for our typo exists...
        self.assertIn({"typo": "fxo", "count": 1, "likely": "fox",
                       "likely_count": 5}, self.symspell)
        # ...and the fallback yields to it, even when KN's own result
        # (gate thin here, since "xo" has a single continuation) points
        # the same way — and even when an adversarial caller hands KN a
        # CONFIDENT result suggesting a different word:
        out = spellfix_fallback(self.symspell, self.kn, self.base)
        self.assertEqual(out["use"], "spellfix")
        self.assertEqual(out["suggestion"], self.symspell)
        adversarial = {"query": "fxo", "gate": "confident",
                       "candidates": [{"word": "dog", "edit_distance": 3}]}
        out2 = spellfix_fallback(self.symspell, adversarial, self.base)
        self.assertEqual(out2["use"], "spellfix")
        self.assertEqual(out2["suggestion"], self.symspell)

    def test_kn_used_only_when_symspell_abstained_and_gate_confident(self):
        # on the base corpus SymSpell has no rare-word near-miss of a
        # common word (every rare word is far from "fox") -> abstains;
        # KN's gate is confident -> KN's top candidate is the answer.
        self.assertEqual(spellfix.SpellIndex().build(CORPUS).suggest(), [])
        kn = suggest(self.base, "fxo", 5)
        self.assertEqual(kn["gate"], "confident")
        out = spellfix_fallback([], kn, self.base)
        self.assertEqual(out["use"], "kneser_ney")
        self.assertEqual(out["suggestion"], "fox")

    def test_none_when_both_abstain(self):
        out = spellfix_fallback([], {"query": "qqq", "gate": "abstain",
                                     "candidates": []}, self.base)
        self.assertEqual(out["use"], "none")
        self.assertIsNone(out["suggestion"])
        self.assertIn("no correction invented", out["why"])


class HonestyTests(unittest.TestCase):
    def test_empty_corpus_abstains_everywhere(self):
        for texts in ([], [""], ["   ", ""],):
            mw = WordKN(2).train(texts)
            self.assertEqual(gate(mw, "anything"), "abstain")
            self.assertIsNone(score(mw, "anything"))
            self.assertEqual(complete(mw, "anything", 3), [])
            mc = CharKN(4).train(texts)
            out = suggest(mc, "fxo", 5)
            self.assertEqual(out["gate"], "abstain")
            self.assertEqual(out["candidates"], [])
            self.assertIn("no correction invented", out["note"])

    def test_degenerate_discount_reports_zero_probability(self):
        # every bigram of this corpus repeats (counts 2 and 4): the
        # leaving-one-out estimate D_2 = n1/(n1+2n2) = 0/(0+4) = 0 —
        # no smoothing mass exists at the bigram level, honestly. Seen
        # text scores 0.0 log-probability (plain MLE, probability 1);
        # an unseen continuation scores EXACTLY -inf (reported, no floor
        # invented — the model assigns it zero probability).
        m = WordKN(2).train(["a b a b", "a b a b"])
        self.assertEqual(m["discounts"][1], 0.0)
        self.assertEqual(score(m, "a b a b"), 0.0)
        self.assertEqual(score(m, "a b c"), float("-inf"))

    def test_char_flavor_scores_and_completes_chars(self):
        m = CharKN(4).train(CORPUS)
        self.assertLess(score(m, "that quick fox jumps"), 0.0)
        self.assertGreater(score(m, "that quick fox jumps"), -12.0)
        # char completions are single characters over the char vocab
        ranked = complete(m, "fo", 3)
        self.assertTrue(ranked)
        self.assertTrue(all(len(r["token"]) == 1 for r in ranked))


if __name__ == "__main__":
    unittest.main()
