"""Fuzzy tool-name search over the generated tool registry (§6.2).

The bit-parallel Levenshtein automaton (Wu & Manber 1992, full-string
variant) is fuzzed against a reference two-row DP here, and suggest_tools'
operating properties are pinned: name typos, spoken camelCase forms and
case-insensitivity all find the right tool; gibberish finds nothing; the
CLI's unknown-tool error now carries the suggestions.
"""

from __future__ import annotations

import random
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

from assistant.capabilities.settings import cli as settings_cli
from assistant.adapters.caelestia.registry import (
    _bitap_distance,
    _camel_atoms,
    suggest_tools,
)


def _reference(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


class BitapFuzzTests(unittest.TestCase):
    """The automaton must equal the reference DP — always."""

    def test_fuzz_against_reference_dp(self):
        rnd = random.Random(0x120)
        for _ in range(3000):
            a = "".join(rnd.choice("abcdeij") for _ in range(rnd.randrange(0, 12)))
            b = "".join(rnd.choice("abcdeij") for _ in range(rnd.randrange(0, 13)))
            want = _reference(a, b)
            got = _bitap_distance(a, b)
            self.assertEqual(got, want, f"{a!r} vs {b!r}")

    def test_capped_semantics(self):
        rnd = random.Random(0x121)
        for _ in range(1500):
            a = "".join(rnd.choice("abc") for _ in range(rnd.randrange(0, 9)))
            b = "".join(rnd.choice("abc") for _ in range(rnd.randrange(0, 9)))
            want, cap = _reference(a, b), rnd.randrange(0, 4)
            got = _bitap_distance(a, b, cap=cap)
            if want <= cap:
                self.assertEqual(got, want, f"{a!r} vs {b!r} cap={cap}")
            else:
                self.assertIsNone(got, f"{a!r} vs {b!r} cap={cap} got {got}")

    def test_empty_pattern(self):
        self.assertEqual(_bitap_distance("", "abc"), 3)
        self.assertIsNone(_bitap_distance("", "abc", cap=2))
        self.assertEqual(_bitap_distance("a", ""), 1)


class SuggestToolsTests(unittest.TestCase):

    def test_name_typo_finds_the_tool(self):
        got = suggest_tools("setBarPositin", max_distance=2)
        self.assertIn(("setBarPosition", 1), got)

    def test_case_insensitive(self):
        got = suggest_tools("setbarscale")
        self.assertEqual(got[0], ("setBarScale", 0))

    def test_spoken_camel_atoms_form(self):
        # "set bar scale" is 2 insertions away from the atom-joined name
        got = suggest_tools("set bar scale")
        self.assertIn(("setBarScale", 2), got)

    def test_gibberish_finds_nothing(self):
        self.assertEqual(suggest_tools("zzzqqqxx"), [])

    def test_results_sorted_nearest_first_and_limited(self):
        got = suggest_tools("setBarPositin", limit=2)
        self.assertLessEqual(len(got), 2)
        dists = [d for _n, d in got]
        self.assertEqual(dists, sorted(dists))

    def test_exact_name_has_distance_zero(self):
        got = dict(suggest_tools("setDockBadges"))
        self.assertEqual(got.get("setDockBadges"), 0)

    def test_camel_atoms_split(self):
        self.assertEqual(_camel_atoms("setBarScale"), ["set", "bar", "scale"])
        self.assertEqual(_camel_atoms("setDockBadges"), ["set", "dock", "badges"])

    def test_no_duplicate_names(self):
        got = suggest_tools("set bar scale")
        names = [n for n, _d in got]
        self.assertEqual(len(names), len(set(names)))


class CliSuggestionTests(unittest.TestCase):

    def test_unknown_tool_error_carries_suggestions(self):
        err, out = StringIO(), StringIO()
        with redirect_stderr(err), redirect_stdout(out):
            rc = settings_cli.main(["--tool", "setBarPositin"])
        self.assertEqual(rc, 1)
        self.assertIn("unknown tool", err.getvalue())
        self.assertIn("did you mean:", err.getvalue())
        self.assertIn("setBarPosition", err.getvalue())

    def test_known_tool_still_works(self):
        out, err = StringIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = settings_cli.main(["--tool", "setBarScale"])
        self.assertEqual(rc, 0)
        self.assertIn("setBarScale", out.getvalue())

    def test_unmatched_unknown_tool_stays_a_plain_error(self):
        err, out = StringIO(), StringIO()
        with redirect_stderr(err), redirect_stdout(out):
            rc = settings_cli.main(["--tool", "zzzqqqxx"])
        self.assertEqual(rc, 1)
        self.assertIn("unknown tool", err.getvalue())
        self.assertNotIn("did you mean:", err.getvalue())


if __name__ == "__main__":
    unittest.main()
