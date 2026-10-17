"""cortex.tree — CART beside the logistic router (§6.3).

Pins: Gini against the formula; a hand-checkable perfect split; the
depth/leaf caps; readable rule rendering; determinism; and — the point of
the module — the AGREEMENT report between the tree and the fitted
logistic model on a labeled corpus built from the router's own features
(top-routed candidate = accepted, a lower-ranked one = rejected), with
the honest floors the measured numbers support.
"""

from __future__ import annotations

import unittest

from assistant.core.router import route
from assistant.core.tree import (agreement_report, build_tree, gini,
                                   tree_classify, tree_from_examples,
                                   tree_to_rules)


def _labeled_corpus(phrases):
    """Route each phrase; the top candidate's feature row is a positive
    example (that surface was the one to act on), and one lower-ranked
    candidate's row is a negative — the accept/reject signal the learner
    itself would observe."""
    examples = []
    for text in phrases:
        result = route(text)
        if result.top is None:
            continue
        top = result.top
        features = result.features.get(top.surface, {})
        if features:
            examples.append({"features": features, "label": 1,
                             "surface": top.surface})
        for cand in result.candidates[1:2]:
            lower = result.features.get(cand.surface, {})
            if lower:
                examples.append({"features": lower, "label": 0,
                                  "surface": cand.surface})
    return examples


PHRASES = [
    "make my bar thinner", "make the bar 25% smaller", "thicker bar please",
    "make the dock bigger", "smaller dock", "increase the blur",
    "disable blur", "turn on the blur", "make animations faster",
    "slower animations", "why is my dock blurry", "what controls the blur",
    "undo the last change", "revert that", "what did i change",
    "change my wallpaper", "new background", "switch the color scheme",
    "my shell crashed", "nothing works after update", "how do i find docs",
    "make everything compact", "a minimal look", "gaming setup",
    "save battery", "more like macos", "set the transparency lower",
    "more see-through panels", "move the dock to the left",
    "bar to the top", "notifications off", "enable the launcher badges",
    "dock on the right side", "reduce corner radius", "bigger osd",
    "darker overview", "sidebar width smaller", "border size up",
    "greeter morning start", "night light color", "audio volume step",
    "calculator in the bar", "weather in the dashboard",
    "make the bar a little bit thinner", "much bigger icons",
    "slightly smaller gaps", "hide the labels", "show seconds in the clock",
    "clock in the bar", "battery percentage on", "media widget compact",
]


class GiniTests(unittest.TestCase):

    def test_matches_formula(self):
        self.assertAlmostEqual(gini([]), 0.0)
        self.assertAlmostEqual(gini([1, 1, 1]), 0.0)
        self.assertAlmostEqual(gini([0, 1]), 0.5)
        self.assertAlmostEqual(gini([1, 0, 1, 0, 1]), 1 - (0.6 ** 2 + 0.4 ** 2))


class BuildTreeTests(unittest.TestCase):

    def test_hand_checkable_perfect_split(self):
        # lex >= 0.5 perfectly separates the classes
        rows = [[0.1], [0.2], [0.8], [0.9]] * 3
        labels = [0, 0, 1, 1] * 3
        tree = build_tree(rows, labels, max_depth=2, min_leaf=2)
        self.assertFalse(tree["leaf"])
        self.assertEqual(tree["name"], "lex")
        self.assertAlmostEqual(tree["threshold"], 0.5)
        self.assertEqual(tree["left"]["prediction"], 0)
        self.assertEqual(tree["right"]["prediction"], 1)

    def test_depth_cap_is_respected(self):
        rows = [[i / 20.0] for i in range(20)]
        labels = [i % 2 for i in range(20)]
        tree = build_tree(rows, labels, max_depth=1, min_leaf=2)
        depth, node = 0, tree
        while not node.get("leaf"):
            node = node["right"]
            depth += 1
        self.assertLessEqual(depth, 1)

    def test_min_leaf_cap_is_respected(self):
        rows = [[i / 100.0] for i in range(100)]
        labels = [0] * 50 + [1] * 50
        tree = build_tree(rows, labels, max_depth=4, min_leaf=10)
        stack = [tree]
        while stack:
            node = stack.pop()
            if node.get("leaf"):
                self.assertGreaterEqual(node["n"], 10)
            else:
                stack.extend((node["left"], node["right"]))

    def test_deterministic(self):
        corpus = _labeled_corpus(PHRASES)
        a = tree_from_examples(corpus)
        b = tree_from_examples(corpus)
        self.assertEqual(a, b)

    def test_too_few_examples_returns_none(self):
        self.assertIsNone(tree_from_examples([]))
        self.assertIsNone(tree_from_examples(
            [{"features": {"lex": 0.5}, "label": 1}] * 4))

    def test_classify_returns_readable_path(self):
        rows = [[0.1], [0.2], [0.8], [0.9]] * 3
        labels = [0, 0, 1, 1] * 3
        tree = build_tree(rows, labels, max_depth=2, min_leaf=2)
        label, path = tree_classify(tree, {"lex": 0.9, "sem": 0.0})
        self.assertEqual(label, 1)
        self.assertTrue(any("lex >=" in step for step in path))
        self.assertTrue(path[-1].startswith("-> accept"))

    def test_rules_are_readable_lines(self):
        corpus = _labeled_corpus(PHRASES)
        tree = tree_from_examples(corpus)
        rules = tree_to_rules(tree)
        self.assertTrue(rules)
        for rule in rules:
            self.assertIn("->", rule)
            self.assertIn("n=", rule)


class AgreementTests(unittest.TestCase):
    """The promotion gate: the tree must substantially agree with the
    fitted logistic router on the same labeled corpus before it earns a
    place as a verdict path. The floors here are the measured behavior,
    not wishes — see the CHANGELOG entry for the recorded numbers."""

    @classmethod
    def setUpClass(cls):
        cls.corpus = _labeled_corpus(PHRASES)
        # the corpus must be big enough for an honest held-out split
        cls.report = agreement_report(cls.corpus)

    def test_corpus_is_nontrivial(self):
        self.assertGreaterEqual(len(self.corpus), 40)

    def test_report_exists_and_reports_both_sides(self):
        self.assertIsNotNone(self.report)
        for key in ("train_agreement", "test_agreement",
                    "tree_test_accuracy", "logistic_test_accuracy",
                    "n_train", "n_test"):
            self.assertIn(key, self.report)

    def test_tree_agrees_with_logistic_on_training_data(self):
        # 0.662 measured on the shipped corpus: the depth-3/min-5 tree
        # stops early by design while the 5-epoch logistic fits training
        # harder — the honest floor is the measured one.
        self.assertGreaterEqual(self.report["train_agreement"], 0.60)

    def test_tree_agrees_with_logistic_on_held_out_data(self):
        # 0.853 measured on the shipped corpus (34 held-out examples):
        # the readable path and the weight vector learned compatible
        # decision boundaries on unseen data
        self.assertGreaterEqual(self.report["test_agreement"], 0.75)

    def test_tree_is_competitive_with_logistic_accuracy(self):
        # the explainability trade must not cost a fortune: the tree's
        # held-out accuracy stays within 0.15 of the logistic model's
        self.assertGreaterEqual(self.report["tree_test_accuracy"],
                                 self.report["logistic_test_accuracy"] - 0.15)

    def test_tree_trains_from_learner_examples_shape(self):
        # the same shape the learner persists (learn.py's example rows)
        corpus = self.corpus[:40]
        tree = tree_from_examples(corpus)
        self.assertIsNotNone(tree)


if __name__ == "__main__":
    unittest.main()
