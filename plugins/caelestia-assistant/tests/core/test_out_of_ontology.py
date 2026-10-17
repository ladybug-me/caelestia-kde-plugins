"""F6 out-of-ontology detector (exponential-build-5, D4, D5).

'what's the weather like tomorrow' confidently routed to setShowWeather
and 'what is the capital of France' crashed the genius layer with a
ValueError — there was no out-of-ontology class. Now:

- a QUESTION with no value cue and only thin single-word lexical overlap
  is OUT_OF_ONTOLOGY, never a confident settings route;
- a request whose content words match NOTHING in the registry vocabulary
  is OUT_OF_ONTOLOGY ('order a pizza');
- why-questions reach the explain surface (D4: free text answers with a
  grounded, cited reason instead of mis-routing to a size tool);
- the explain fallback never crashes into genius: only genius-SHAPED
  questions delegate there (D5b);
- real settings questions keep planning ('can you make the bar thinner?').
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from assistant.core.pipeline import process
from assistant.core.router import DEFAULT_STATE, route


def process_on(text: str):
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "shell.json"
        return process(text, file_path=target, router_state=DEFAULT_STATE)


class OutOfOntologyTests(unittest.TestCase):
    def test_weather_question_is_out_of_ontology(self):
        res = route("what's the weather like tomorrow", state=DEFAULT_STATE, k=3)
        self.assertEqual(res.verdict, "OUT_OF_ONTOLOGY")
        # candidates still ride along as evidence
        self.assertTrue(res.candidates)

    def test_capital_of_france_does_not_crash(self):
        res = process_on("what is the capital of France")
        self.assertNotEqual(res.verdict, "DELEGATE")
        self.assertNotEqual(res.verdict, "PLAN")
        self.assertIn(res.verdict, ("OUT_OF_ONTOLOGY", "EXPLAIN"))

    def test_order_a_pizza_is_out_of_ontology(self):
        res = route("order a pizza", state=DEFAULT_STATE, k=3)
        self.assertEqual(res.verdict, "OUT_OF_ONTOLOGY")

    def test_pipeline_surfaces_suggestions_not_a_route(self):
        res = process_on("what's the weather like tomorrow")
        self.assertEqual(res.verdict, "OUT_OF_ONTOLOGY")
        self.assertFalse(res.ops)
        self.assertIsNone(res.plan)


class WhyQuestionsReachExplainTests(unittest.TestCase):
    def test_why_is_my_dock_blurry_explains(self):
        res = process_on("why is my dock blurry")
        self.assertEqual(res.verdict, "EXPLAIN")
        self.assertIsNotNone(res.explain_answer)
        self.assertTrue(res.explain_answer.get("cites"),
                        "the dock-blurry answer must be grounded")

    def test_why_question_never_routes_to_a_size_tool(self):
        res = route("why is my dock blurry", state=DEFAULT_STATE, k=3)
        if res.candidates:
            self.assertNotEqual(res.candidates[0].kind, "tool")


class RealQuestionsKeepWorkingTests(unittest.TestCase):
    def test_math_question_still_reaches_genius(self):
        res = process_on("what is 2+2")
        self.assertEqual(res.verdict, "DELEGATE")
        self.assertEqual(res.delegate, "genius")

    def test_how_do_i_still_reaches_search(self):
        res = process_on("how do I find the lockscreen docs")
        self.assertEqual(res.verdict, "DELEGATE")
        self.assertEqual(res.delegate, "search")

    def test_polite_settings_question_still_plans(self):
        res = process_on("can you make the bar thinner?")
        self.assertEqual(res.verdict, "PLAN")
        self.assertEqual(res.ops[0]["tool"], "setBarScale")


if __name__ == "__main__":
    unittest.main()
