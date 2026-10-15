"""brain.nlg tests — planning, aggregation, morphology, determinism."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from assistant.capabilities.brain import nlg  # noqa: E402

CHANGES = [
    {"tool": "setSpacingScale", "value": 1.1, "group": "bar",
     "field": "spacing"},
    {"tool": "setBarDragThreshold", "value": 20, "group": "bar",
     "field": "spacing"},
    {"tool": "setAccentColor", "value": "red", "group": "bar",
     "field": "color"},
    {"tool": "setBarPosition", "value": "top", "group": "bar"},
    {"tool": "setDockIconSize", "value": 48, "group": "dock",
     "field": "color"},
]


class TestPlanning(unittest.TestCase):
    def test_groups_by_tool_group(self):
        plan = nlg.plan_document(CHANGES)
        self.assertEqual(plan["n_groups"], 2)
        self.assertEqual(plan["groups"][0]["group"], "bar")
        self.assertEqual(plan["groups"][0]["n"], 4)

    def test_within_group_order_is_deterministic(self):
        plan = nlg.plan_document(list(reversed(CHANGES)))
        tools = [c["tool"] for c in plan["groups"][0]["changes"]]
        self.assertEqual(tools, sorted(tools))

    def test_uninformative_tool_groups_as_other(self):
        # 'set' is all stop-word: no content word survives, so the
        # honest group is 'other'
        plan = nlg.plan_document([{"tool": "set"}])
        self.assertEqual(plan["groups"][0]["group"], "other")

    def test_dotted_path_uses_first_segment(self):
        plan = nlg.plan_document([{"tool": "bar.custom.thing"}])
        self.assertEqual(plan["groups"][0]["group"], "bar")


class TestAggregation(unittest.TestCase):
    def test_mass_nouns_never_pluralize_the_noun(self):
        lines = nlg.aggregate(nlg.plan_document(CHANGES))
        self.assertIn("2 spacing changes and 1 color change",
                      lines[0])

    def test_singular_morphology(self):
        lines = nlg.aggregate(nlg.plan_document(CHANGES[4:]))
        self.assertEqual(lines, ["1 change in the dock: 1 color change"])

    def test_no_fields_falls_back_to_tool_names(self):
        lines = nlg.aggregate(nlg.plan_document(
            [{"tool": "setBarPosition", "group": "bar"},
             {"tool": "setBarScale", "group": "bar"}]))
        self.assertIn("setBarPosition", lines[0])
        self.assertIn("setBarScale", lines[0])


class TestSurfaces(unittest.TestCase):
    def test_summary_lead_sentence(self):
        lines = nlg.render_changes_summary(CHANGES)
        self.assertTrue(lines[0].startswith("this plan makes 5 changes "
                                            "across 2 areas"))

    def test_empty_plan(self):
        self.assertEqual(nlg.render_changes_summary([]),
                         ["no changes to make."])
        self.assertEqual(nlg.render_plan_paragraph([]),
                         "Nothing to change.")

    def test_single_change_collapses(self):
        text = nlg.render_plan_paragraph([CHANGES[0]])
        self.assertEqual(text,
                         "This plan will make 1 change to the bar "
                         "(setSpacingScale).")

    def test_paragraph_mentions_each_group_once(self):
        text = nlg.render_plan_paragraph(CHANGES)
        self.assertEqual(text.count("bar"), 1)
        self.assertTrue(text.endswith("."))

    def test_deterministic(self):
        a = nlg.render_changes_summary(CHANGES)
        b = nlg.render_changes_summary(list(CHANGES))
        self.assertEqual(a, b)
        self.assertEqual(nlg.render_plan_paragraph(CHANGES),
                         nlg.render_plan_paragraph(CHANGES))


if __name__ == "__main__":
    unittest.main()
