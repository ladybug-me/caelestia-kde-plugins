"""F7 no-silent-drop invariant (exponential-build-5, D7).

'what is 15% of 2400 plus 18% GST' answered 360 with ok:True — the
second clause vanished. The invariant: every input span is either
CLAIMED by the handler (part of the evaluated expression / matched
reading) or listed under 'unhandled'. A request with unclaimed content
returns AMBIGUOUS with its readings — never a partial answer presented
as complete.
"""
from __future__ import annotations

import random
import unittest

from assistant.capabilities.genius import meta


def run_math(text: str):
    return meta._run_domain("math_eval", text)


class NoSilentDropTests(unittest.TestCase):
    def test_d7_percent_plus_gst_is_ambiguous_not_partial(self):
        out = run_math("what is 15% of 2400 plus 18% GST")
        self.assertTrue(out.get("ambiguous"))
        readings = out.get("readings") or []
        self.assertTrue(readings, "both readings must be shown")
        self.assertIn("15% of 2400", str(readings))
        self.assertTrue(out.get("unhandled"), "the GST clause must be listed")
        # and it is never reported as a complete ok answer
        self.assertNotEqual(out.get("ok"), True)

    def test_simple_percent_of_still_answers(self):
        out = run_math("what is 15% of 2400")
        self.assertEqual(out["value"], 360.0)
        self.assertFalse(out.get("ambiguous"))

    def test_plain_expression_still_answers(self):
        out = run_math("what is 2+2")
        self.assertAlmostEqual(out["value"], 4.0)
        self.assertFalse(out.get("ambiguous"))


class CompoundCoveragePropertyTests(unittest.TestCase):
    """Seeded property: compound requests never silently drop a clause.

    For every generated compound request, the result is either a FULL
    evaluation (no unhandled spans) or an AMBIGUOUS verdict carrying
    readings + the unhandled spans. A value plus ok and leftover text
    is the bug this pins.
    """

    def test_compound_requests_never_drop_clauses(self):
        rng = random.Random(20260928)
        filler_words = ["GST", "VAT", "tip", "tax", "discount", "fee"]
        bad = []
        for i in range(60):
            a = rng.randint(2, 90)
            b = rng.choice([80, 240, 1000, 2400])
            c = rng.randint(2, 30)
            kind = i % 3
            if kind == 0:
                text = f"what is {a}% of {b} plus {c}% {rng.choice(filler_words)}"
            elif kind == 1:
                text = f"calculate {a} + {b} and {c} times {b}"
            else:
                text = f"what is {a}% of {b} minus {c}% {rng.choice(filler_words)}"
            out = run_math(text)
            has_value = "value" in out and out.get("value") is not None
            ambiguous = bool(out.get("ambiguous"))
            unhandled = bool(out.get("unhandled"))
            if has_value and not ambiguous:
                # a full evaluation must have claimed everything
                if unhandled:
                    bad.append((text, "value with unhandled spans"))
                if out.get("ok") is True and unhandled:
                    bad.append((text, "ok with unhandled spans"))
            if not has_value and not ambiguous:
                bad.append((text, f"neither answered nor ambiguous: {out}"))
        self.assertEqual(bad, [], f"silent drops: {bad}")


if __name__ == "__main__":
    unittest.main()
