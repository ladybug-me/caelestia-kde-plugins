"""Tests for exponential-build phase 2.5: the devflow commit-risk score
(McCabe cyclomatic complexity x recent churn, from piped text only).

The contract under test:

- per-function cyclomatic complexity from the stdlib ast counts the
  standard decision points, with nested functions owning their own
  decisions (McCabe 1976);
- ``git log --numstat`` text parses into commits (newest first) with
  the file rows REUSING diffstat.parse_numstat — one numstat parser;
- churn decays by commit recency (fixed half-life, git's own order as
  the signal); risk = peak_cc x log2(1 + churn), tiered at stated
  thresholds;
- unparsable source refuses the complexity half honestly (churn still
  reported, risk stays None) — never a guessed number; no subprocess
  exists anywhere on the path.
"""
import unittest

from assistant.capabilities.devflow.risk import (commit_risk, cyclomatic_complexity,
                                    parse_log_numstat, recent_churn)

SIMPLE = """
def flat(x):
    if x > 0:
        return x
    return -x

def branching(x, y):
    if x and y or x:
        for i in range(y):
            while i:
                i -= 1
    return [i for i in range(x) if i]

def inner_outer(x):
    def nested(y):
        if y:
            return 1
        return 0
    return nested(x)
"""

LOG = """commit aaaaaaa1111111222222223333333344444444
Author: someone
Date:   Mon Sep 21 10:00:00 2026 +0000

    newest change

10\t2\tpkg/hot.py
3\t1\tREADME.md

commit bbbbbbb1111111222222223333333344444444
Author: someone
Date:   Mon Sep 14 10:00:00 2026 +0000

    older change

2\t2\tpkg/hot.py
"""

BAD = "def broken(:\n    pass\n"


class CyclomaticTests(unittest.TestCase):
    def test_counts_standard_decision_points(self):
        out = cyclomatic_complexity(SIMPLE)
        by_name = {f["name"]: f["cc"] for f in out["functions"]}
        # flat: 1 + if = 2
        self.assertEqual(by_name["flat"], 2)
        # branching: 1 + if + (2 extra bool operands) + for + while
        #           + comprehension-for + comprehension-if = 8
        self.assertEqual(by_name["branching"], 8)

    def test_nested_function_owns_its_decisions(self):
        out = cyclomatic_complexity(SIMPLE)
        by_name = {f["name"]: f["cc"] for f in out["functions"]}
        self.assertEqual(by_name["inner_outer"], 1)   # no decisions of its own
        self.assertEqual(by_name["inner_outer.nested"], 2)
        self.assertEqual(out["peak_function"], "branching")
        self.assertEqual(out["peak_cc"], 8)

    def test_unparsable_source_is_an_honest_refusal(self):
        with self.assertRaises(ValueError):
            cyclomatic_complexity(BAD)


class NumstatLogTests(unittest.TestCase):
    def test_parses_commits_newest_first(self):
        commits = parse_log_numstat(LOG)
        self.assertEqual(len(commits), 2)
        self.assertTrue(commits[0]["commit"].startswith("aaaaaaa1"))
        self.assertEqual(commits[0]["added"], 13)
        self.assertEqual(commits[0]["deleted"], 3)
        self.assertTrue(commits[0]["added"] >= commits[1]["added"])

    def test_churn_decays_with_recency(self):
        churn = recent_churn(LOG)
        files = {f["path"]: f["churn"] for f in churn["files"]}
        # newest commit's 12 lines weigh 1.0; the older 4 lines weigh
        # 0.5 ** (1 / halflife_commits) = 0.5 ** 0.2
        older_weight = 0.5 ** (1 / 5)
        self.assertAlmostEqual(files["pkg/hot.py"],
                               12 + 4 * older_weight, places=2)
        self.assertAlmostEqual(files["README.md"], 4, places=2)

    def test_halflife_must_be_positive(self):
        with self.assertRaises(ValueError):
            recent_churn(LOG, halflife_commits=0)


class CommitRiskTests(unittest.TestCase):
    def test_score_is_the_product_with_stated_tiers(self):
        out = commit_risk(LOG, SIMPLE)
        import math
        expected = round(8 * math.log2(1 + out["churn"]["total"]), 2)
        self.assertEqual(out["risk"], expected)
        self.assertIn(out["tier"], ("low", "medium", "high"))
        self.assertIn("McCabe", out["algorithm"])

    def test_high_complexity_high_churn_tiers_high(self):
        big = "def f(x):\n" + "".join(
            f"    if x == {i}:\n        return {i}\n" for i in range(30))
        hot = "".join(
            f"commit {'a' * 40}\n\n5\t5\tpkg/hot.py\n" for _ in range(20))
        out = commit_risk(hot, big)
        self.assertGreaterEqual(out["risk"], 60)
        self.assertEqual(out["tier"], "high")

    def test_before_source_reports_the_cc_delta(self):
        before = "def flat(x):\n    return x\n"
        out = commit_risk(LOG, SIMPLE, source_before=before)
        self.assertEqual(out["cc_delta"], 7)  # peak 8 - peak 1

    def test_unparsable_source_keeps_churn_refuses_score(self):
        out = commit_risk(LOG, BAD)
        self.assertIsNone(out["risk"])
        self.assertIsNone(out["tier"])
        self.assertIn("churn", out)  # the half that could be computed
        self.assertIn("does not parse", out["complexity_refused"])

    def test_deterministic(self):
        self.assertEqual(commit_risk(LOG, SIMPLE), commit_risk(LOG, SIMPLE))


if __name__ == "__main__":
    unittest.main()
