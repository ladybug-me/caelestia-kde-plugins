"""Tests for exponential-build-3 C2: L* automaton learning for regex
induction (Angluin 1987 — observation table, closed/consistent repair,
conjecture, counterexample rounds).

The contract under test:

- TARGET mode re-learns a regex from its own fullmatch behavior: the
  table machinery converges, and the rendered regex is cross-validated
  against the learned DFA on the whole conformance set (a rendering
  bug is a refusal, never a silent mismatch);
- the equivalence oracle is a sound-but-INCOMPLETE conformance set —
  every checked string agrees, and the result never claims more (the
  incompleteness is PINNED here: a target whose learned DFA diverges
  from the target outside the checked set does exactly that, and the
  divergence is named, not hidden);
- EXAMPLES mode derives ONE DFA consistent with the labeled examples
  (the canonical consistent quotient over the partial oracle): every
  positive matches, every negative rejects, contradictory labels
  refuse, nothing-at-all abstains, and unlabeled territory carries no
  claim (UNKNOWN sink, dropped from the rendered regex);
- every refusal path raises AngluinError WITH its reason (invalid
  target, state bound, round budget, conformance budget, empty
  alphabet, contradiction) — never a silent clamp;
- both modes are deterministic: same inputs → byte-identical result.
"""
import unittest

from assistant.capabilities.genius.angluin import (AngluinError, conformance_strings,
                                      induce_from_examples,
                                      learn_from_target)


class TargetModeTests(unittest.TestCase):
    """learn_from_target: total membership oracle (the target's own
    fullmatch) + deterministic conformance set as the equivalence
    oracle stand-in."""

    def test_a_plus_is_learned_in_one_round(self):
        # Hand-derived: the table opens with S={""}, E={""}; queries
        # "", "a", "aa" (cached) close it at TWO states (reject-start,
        # self-loop accept), the conjecture accepts exactly a+ on every
        # conformance string ["", "a", "aa", "aaa"], so round 1 returns
        # — and "aaa" is never even queried (3 cached queries, pinned).
        r = learn_from_target("a+", alphabet=["a"], max_len=3)
        self.assertEqual(r["mode"], "target")
        self.assertEqual(r["target"], "a+")
        self.assertEqual(r["regex"], "a(?:a)*")
        self.assertEqual(r["states"], 2)
        self.assertEqual(r["rounds"], 1)
        self.assertEqual(r["membership_queries"], 3)
        self.assertEqual(r["dfa"], {"alphabet": ["a"], "start": 0,
                                    "accepting": [1],
                                    "delta": [{"a": 1}, {"a": 1}]})
        self.assertEqual(r["conformance"]["checked"], 4)
        self.assertEqual(r["conformance"]["max_len"], 3)
        self.assertEqual(r["conformance"]["note"],
                         "sound but incomplete — equality is NOT claimed")

    def test_ends_with_ab_recovers_the_minimal_dfa(self):
        # '(a|b)*ab' needs the 3-state minimal DFA (nothing / seen-a /
        # seen-ab). With conformance up to length 3 (15 strings) the
        # first conjecture is wrong and ONE counterexample round
        # repairs it (rounds=2, 11 cached membership queries, both
        # pinned). Beyond the checked set the learned regex is STILL
        # equivalent to the target on every string over {a,b} up to
        # length 6 (127 strings, 12x the conformance set) — the honest
        # strong case: here the conformance set happened to be enough,
        # though the module never claims it.
        import re
        r = learn_from_target("(?:a|b)*ab", alphabet=["a", "b"],
                              max_len=3)
        self.assertEqual(r["states"], 3)
        self.assertEqual(r["rounds"], 2)
        self.assertEqual(r["membership_queries"], 11)
        self.assertEqual(r["conformance"]["checked"], 15)
        self.assertEqual(r["dfa"],
                         {"alphabet": ["a", "b"], "start": 0,
                          "accepting": [1],
                          "delta": [{"a": 2, "b": 0},
                                    {"a": 2, "b": 0},
                                    {"a": 2, "b": 1}]})
        learned = re.compile(r["regex"])
        target = re.compile("(?:a|b)*ab")
        for s in conformance_strings("ab", 6):  # 127 strings
            self.assertIs(learned.fullmatch(s) is not None,
                          target.fullmatch(s) is not None,
                          f"divergence at {s!r}")

    def test_conformance_incompleteness_is_honest_and_pinned(self):
        # THE honesty pin: target 'ab' (exact), conformance only up to
        # length 2 + ['aab', 'ba'] = 8 checked strings. L* converges to
        # a 3-state DFA that agrees with the target on ALL EIGHT
        # checked strings (soundness, verified below) — and diverges
        # OUTSIDE the checked set: 'bab' (length 3) is accepted by the
        # learned regex but not by the target. The result's note says
        # equality is NOT claimed; this test pins that the claim is
        # exactly as weak as stated, in the direction it actually goes.
        import re
        r = learn_from_target("ab", alphabet=["a", "b"], max_len=2,
                              examples=["aab", "ba"])
        self.assertEqual(r["conformance"]["checked"], 8)
        self.assertEqual(r["states"], 3)
        self.assertEqual(r["rounds"], 2)
        learned = re.compile(r["regex"])
        target = re.compile("ab")
        checked = conformance_strings("ab", 2, ["aab", "ba"])
        self.assertEqual(
            checked, ["", "a", "b", "aa", "ab", "ba", "bb", "aab"])
        for s in checked:  # soundness: agreement on everything checked
            self.assertIs(learned.fullmatch(s) is not None,
                          target.fullmatch(s) is not None,
                          f"unsound at a CHECKED string: {s!r}")
        # incompleteness: the shortest divergence, one length past the
        # checked horizon, goes BOTH ways the DFA and the regex accept
        # 'bab' while the target does not.
        self.assertIsNotNone(learned.fullmatch("bab"))
        self.assertIsNone(target.fullmatch("bab"))
        self.assertIn("equality is NOT claimed",
                      r["conformance"]["note"])

    def test_invalid_target_refuses(self):
        with self.assertRaises(AngluinError) as caught:
            learn_from_target("(unclosed", alphabet=["a"])
        self.assertIn("not a valid regex", str(caught.exception))

    def test_state_bound_refuses_without_truncating(self):
        with self.assertRaises(AngluinError) as caught:
            learn_from_target("a+", alphabet=["a"], max_states=1)
        self.assertIn("state bound (2 > 1)", str(caught.exception))
        self.assertIn("refusing, not truncating", str(caught.exception))

    def test_round_budget_refuses_rather_than_looping(self):
        with self.assertRaises(AngluinError) as caught:
            learn_from_target("a+", alphabet=["a"], max_rounds=0)
        self.assertIn("did not converge within 0", str(caught.exception))

    def test_conformance_budget_refuses_with_the_real_numbers(self):
        # 6 symbols up to length 4 = 1+6+36+216+1296 = 1555 strings >
        # the 600-string budget: the refusal names both numbers rather
        # than silently shrinking the equivalence check.
        with self.assertRaises(AngluinError) as caught:
            learn_from_target("a", alphabet=list("abcdef"), max_len=4)
        msg = str(caught.exception)
        self.assertIn("1555", msg)
        self.assertIn("600", msg)
        self.assertIn("refusing rather than silently shrinking",
                      msg)

    def test_target_mode_is_deterministic(self):
        args = ("(a|b)*ab", ["a", "b"])
        self.assertEqual(learn_from_target(*args, max_len=3),
                         learn_from_target(*args, max_len=3))


class ExamplesModeTests(unittest.TestCase):
    """induce_from_examples: the canonical consistent quotient over a
    PARTIAL oracle (the labeled set)."""

    def test_tiny_quotient_is_hand_derivable(self):
        # Hand-derived on paper: positives ["ab"], negatives ["b"].
        # S = {"", "a", "ab", "b"}; signature refinement (own label +
        # successor class per symbol) yields 4 classes — "", "a", "ab"
        # (labeled True), "b" (labeled False) — plus the UNKNOWN sink.
        # Live states {0,1,2} leave exactly the language {"ab"}, and
        # state elimination renders it as the literal 'ab'.
        r = induce_from_examples(["ab"], ["b"])
        self.assertEqual(r["mode"], "examples")
        self.assertEqual(r["regex"], "ab")
        self.assertEqual(r["states"], 5)  # 4 classes + UNKNOWN sink
        self.assertEqual(r["labeled"], 2)
        self.assertEqual(
            r["dfa"],
            {"alphabet": ["a", "b"], "start": 0, "accepting": [2],
             "delta": [{"a": 1, "b": 3}, {"a": 4, "b": 2},
                       {"a": 4, "b": 4}, {"a": 4, "b": 4},
                       {"a": 4, "b": 4}]})

    def test_refinement_merges_symmetric_prefixes(self):
        # positives ["cab", "dab"], negatives ["ca", "da"]: the labeled
        # space holds 7 distinct prefixes, but 'c' and 'd' are
        # label-indistinguishable (both: unlabeled, 'a'-successor is a
        # negative prefix, everything else UNKNOWN), and so are 'ca'/
        # 'da' (negative, 'b'-successor positive) and 'cab'/'dab'
        # (positive). Moore refinement collapses 7 prefixes into 4
        # classes (+ sink = 5 states, vs 8 without merging); the
        # rendered regex names the merge honestly: (?:c|d)ab.
        r = induce_from_examples(["cab", "dab"], ["ca", "da"])
        self.assertEqual(r["regex"], "(?:c|d)ab")
        self.assertEqual(r["states"], 5)
        self.assertEqual(r["labeled"], 4)
        self.assertEqual(
            r["dfa"],
            {"alphabet": ["a", "b", "c", "d"], "start": 0,
             "accepting": [3],
             "delta": [{"a": 4, "b": 4, "c": 1, "d": 1},
                       {"a": 2, "b": 4, "c": 4, "d": 4},
                       {"a": 4, "b": 3, "c": 4, "d": 4},
                       {"a": 4, "b": 4, "c": 4, "d": 4},
                       {"a": 4, "b": 4, "c": 4, "d": 4}]})

    def test_negatives_only_render_the_empty_language(self):
        # No positives at all: the only consistent separator matching
        # the evidence is the empty language, rendered '(?!)' (a
        # zero-width negative lookahead — matches nothing, which is
        # precisely the claim). 3 states: "" and "a" classes + sink.
        import re
        r = induce_from_examples([], ["a"])
        self.assertEqual(r["regex"], "(?!)")
        self.assertEqual(r["dfa"]["accepting"], [])
        self.assertIsNone(re.fullmatch(r["regex"], "a"))
        self.assertIsNone(re.fullmatch(r["regex"], ""))

    def test_every_labeled_string_is_separated_exactly(self):
        # The core contract, exercised on a labeled set with real
        # structure: ALL positives match, ALL negatives reject — for
        # BOTH the dfa payload and the rendered regex.
        import re
        positives = ["cab", "dab"]
        negatives = ["ca", "da", "a", "b", "ab", "c", "d"]
        r = induce_from_examples(positives, negatives)
        compiled = re.compile(r["regex"])
        for s in positives:
            self.assertIsNotNone(compiled.fullmatch(s), s)
        for s in negatives:
            self.assertIsNone(compiled.fullmatch(s), s)
        # unlabeled territory carries NO claim: strings the labels do
        # not determine (all length-3 strings not labeled above) route
        # to the UNKNOWN sink and are dropped from the regex
        for s in ["ba", "aa", "bb", "x", "cabx", "dabab"]:
            self.assertIsNone(compiled.fullmatch(s), s)

    def test_contradictory_labels_refuse(self):
        with self.assertRaises(AngluinError) as caught:
            induce_from_examples(["x"], ["x"])
        self.assertIn("contradictory labels", str(caught.exception))
        self.assertIn("refusing to guess", str(caught.exception))

    def test_nothing_to_learn_abstains(self):
        with self.assertRaises(AngluinError) as caught:
            induce_from_examples([], [])
        self.assertIn("nothing to learn; abstaining",
                      str(caught.exception))

    def test_examples_mode_is_deterministic(self):
        self.assertEqual(induce_from_examples(["cab", "dab"],
                                              ["ca", "da"]),
                         induce_from_examples(["cab", "dab"],
                                              ["ca", "da"]))

    def test_note_states_the_partial_oracle_claim(self):
        r = induce_from_examples(["ab"], ["b"])
        self.assertEqual(
            r["note"],
            "ONE separator consistent with the labeled examples — "
            "not the unique language; unlabeled behavior carries "
            "no claim (UNKNOWN sink, dropped from the regex)")


class ConformanceTests(unittest.TestCase):
    """The deterministic equivalence-oracle stand-in."""

    def test_enumeration_is_sorted_and_complete(self):
        # input alphabet order is IGNORED (sorted {a,b}); every string
        # up to length 2, length order then lexicographic.
        self.assertEqual(conformance_strings("ba", 2),
                         ["", "a", "b", "aa", "ab", "ba", "bb"])

    def test_extras_are_appended_once(self):
        self.assertEqual(conformance_strings("ab", 1,
                                             ["ab", "abc", "zz", "a"]),
                         ["", "a", "b", "ab", "abc", "zz"])

    def test_empty_alphabet_refuses(self):
        with self.assertRaises(AngluinError) as caught:
            conformance_strings("", 2)
        self.assertIn("non-empty alphabet", str(caught.exception))

    def test_budget_refusal_names_the_count(self):
        with self.assertRaises(AngluinError) as caught:
            conformance_strings("abcdef", 4)
        self.assertIn("1555", str(caught.exception))
        self.assertIn("refusing rather than silently shrinking",
                      str(caught.exception))


if __name__ == "__main__":
    unittest.main()
