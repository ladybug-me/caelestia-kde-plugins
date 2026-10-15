"""F5 absence explainer (exponential-build-5, D3).

'Increase the blur' used to plan a no-op (appearance.blur: true -> true).
An all-no-op plan is not a plan — it is the honest answer that the
requested MAGNITUDE does not exist on that setting. The pipeline must
explain the setting's real nature and cite the nearest real magnitude
controls (consequences-table neighbors first — cited edges, not guesses),
never present a no-op change.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from assistant.core.pipeline import process
from assistant.core.router import DEFAULT_STATE


def process_on(text: str):
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "shell.json"
        return process(text, file_path=target, router_state=DEFAULT_STATE)


class AbsenceExplanationTests(unittest.TestCase):
    def test_increase_blur_explains_instead_of_noop(self):
        res = process_on("increase the blur")
        self.assertEqual(res.verdict, "EXPLAIN")
        self.assertIsNotNone(res.explain_answer)
        self.assertTrue(res.explain_answer.get("absence"))
        answer = res.explain_answer["answer"]
        self.assertIn("on/off", answer)
        # the nearest magnitude control is cited by name
        self.assertIn("setTransparencyBase", answer)
        # and grounded in a real file citation
        cites = res.explain_answer.get("cites") or []
        self.assertTrue(cites, "absence explanations must carry citations")
        # and no no-op plan is attached
        self.assertIsNone(res.plan)

    def test_noop_plan_never_rendered(self):
        res = process_on("increase the blur")
        self.assertNotEqual(res.verdict, "PLAN")

    def test_real_changes_still_plan(self):
        res = process_on("turn off blur")
        self.assertEqual(res.verdict, "PLAN")
        self.assertEqual(res.ops[0]["tool"], "setBlurEnabled")


if __name__ == "__main__":
    unittest.main()
