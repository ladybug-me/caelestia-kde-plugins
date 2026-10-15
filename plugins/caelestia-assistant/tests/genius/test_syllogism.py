"""Tests for exponential-build phase 2.4: the first-order resolution
prover for categorical syllogisms (`genius/logic.py::syllogism_check`,
next to the existing SAT/CSP code, wired into the logic domain).

The contract under test:

- premises + NEGATED conclusion are refuted by binary resolution
  (Robinson 1965): the empty clause proves VALID; saturation without it
  proves INVALID — both decided, never guessed;
- valid classics (Barbara, singular syllogisms, negative premises) are
  accepted; the invalid forms (undistributed middle, some->all
  overreach) are rejected;
- statements outside the well-formed fragment are honest LogicErrors
  listing the accepted forms — never silently mis-parsed;
- the meta dispatcher routes 'therefore'-shaped arguments to the
  prover only when EVERY clause parses, falling through to the
  propositional path otherwise.
"""
import unittest

from assistant.capabilities.genius.logic import (LogicError, parse_statement,
                                    syllogism_check)
from assistant.capabilities.genius import meta


class ValidArgumentTests(unittest.TestCase):
    def test_singular_syllogism(self):
        # regular plurals: the fragment's canonicalization merges
        # 'cats'/'cat' (one trailing 's'), so this classic shape works
        r = syllogism_check(["all cats are animals", "tom is a cat"],
                            "tom is an animal")
        self.assertTrue(r["valid"])
        self.assertIn("Robinson", r["method"])

    def test_irregular_plural_stays_distinct_honestly(self):
        # 'men'/'man' is NOT regular: the fragment does not guess
        # morphology, so these are different predicates and the
        # argument does not go through (documented limitation)
        r = syllogism_check(["all men are mortal", "socrates is a man"],
                            "socrates is mortal")
        self.assertFalse(r["valid"])

    def test_barbara_all_all_all(self):
        r = syllogism_check(["all greeks are europeans",
                             "all europeans are people"],
                            "all greeks are people")
        self.assertTrue(r["valid"])

    def test_negative_premise_with_singular(self):
        r = syllogism_check(["no cats are dogs", "tom is a cat"],
                            "tom is not a dog")
        self.assertTrue(r["valid"])

    def test_duplicate_premise_is_still_valid(self):
        r = syllogism_check(["all a are b", "all a are b"], "all a are b")
        self.assertTrue(r["valid"])

    def test_conclusion_repeats_a_premise(self):
        # a premise entails itself: refutation of p + not-p fails
        r = syllogism_check(["all a are b"], "all a are b")
        self.assertTrue(r["valid"])

    def test_some_not_conclusion(self):
        # no A are B |- some A are not B? NO — existential import is NOT
        # assumed (modern reading): this must be INVALID.
        r = syllogism_check(["no a are b"], "some a are not b")
        self.assertFalse(r["valid"])


class InvalidArgumentTests(unittest.TestCase):
    def test_undistributed_middle_rejected(self):
        r = syllogism_check(["all a are b", "all c are b"], "all a are c")
        self.assertFalse(r["valid"])

    def test_some_premise_carry_all_conclusion(self):
        r = syllogism_check(["some greeks are philosophers",
                             "all philosophers are people"],
                            "all greeks are people")
        self.assertFalse(r["valid"])

    def test_unrelated_terms(self):
        r = syllogism_check(["all cats are animals", "tom is a cat"],
                            "tom is a dog")
        self.assertFalse(r["valid"])

    def test_affirming_the_consequent_shape(self):
        # all B are A; some C are A |- some C are B — INVALID
        r = syllogism_check(["all b are a", "some c are a"], "some c are b")
        self.assertFalse(r["valid"])


class HonestContractTests(unittest.TestCase):
    def test_malformed_statement_lists_the_fragment(self):
        with self.assertRaises(LogicError) as ctx:
            syllogism_check(["cats are nice animals"], "x is y")
        self.assertIn("accepted forms", str(ctx.exception))

    def test_three_word_statement_refused(self):
        with self.assertRaises(LogicError):
            parse_statement("all black things are visible")

    def test_result_carries_the_evidence(self):
        r = syllogism_check(["all cats are animals", "tom is a cat"],
                            "tom is an animal")
        self.assertIn("negated_conclusion", r)
        self.assertEqual(r["negated_conclusion"], "is_not tom animal")
        self.assertGreater(r["initial_clauses"], 0)
        self.assertTrue(r["derivation_prefix"])  # the steps are shown
        self.assertIn("note", r)

    def test_deterministic(self):
        args = (["all a are b", "b is a c"], "x is a c")
        self.assertEqual(syllogism_check(["all a are b", "b is a c"],
                                         "x is a c"),
                         syllogism_check(["all a are b", "b is a c"],
                                         "x is a c"))


class MetaWiringTests(unittest.TestCase):
    def test_therefore_argument_routes_to_the_prover(self):
        out = meta._run_domain(
            "logic",
            "all cats are animals, tom is a cat, "
            "therefore tom is an animal")
        self.assertTrue(out["valid"])

    def test_non_fragment_text_falls_through_to_the_formula_path(self):
        # a propositional phrase with 'therefore' in prose must NOT be
        # forced into the syllogism parser: the fall-through keeps the
        # propositional path intact
        out = meta._run_domain("logic", "p and q")
        self.assertNotIn("valid", out)  # not a syllogism answer

    def test_cues_include_syllogism(self):
        self.assertIn("syllogism", meta._DOMAIN_CUES["logic"])


if __name__ == "__main__":
    unittest.main()
