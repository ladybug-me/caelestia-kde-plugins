"""F15 taught-concept tests: teach (registry-checked, consent-gated),
recall verdicts (RECALL / ABSTAIN / AMBIGUOUS / CONFLICT_ASK), Beta
outcome learning, capped reviewable share diffs, and the dispatch
integration (a weak cortex result with a matching taught concept answers
locally with a gated apply payload)."""

from __future__ import annotations

import json
import unittest
from unittest import mock

from assistant.core import concepts
from assistant.core.dispatch import dispatch
from assistant.capabilities.settings.curations import PRESETS


class TeachTests(unittest.TestCase):
    def test_teach_validates_against_registry(self):
        state = {}
        hit = concepts.teach(state, "glassy",
                             ["setBlurEnabled=true", "setTransparencyBase=0.6"],
                             examples=["make it glassy"])
        self.assertEqual(hit["calls"], [["setBlurEnabled", True],
                                        ["setTransparencyBase", 0.6]])
        with self.assertRaises(concepts.ConceptError):
            concepts.teach(state, "bad", ["setNoSuchTool=1"])

    def test_teach_shape_guards(self):
        state = {}
        with self.assertRaises(concepts.ConceptError):
            concepts.teach(state, "", ["setBlurEnabled=true"])
        with self.assertRaises(concepts.ConceptError):
            concepts.teach(state, "ok name", [])
        with self.assertRaises(concepts.ConceptError):
            concepts.teach(state, "ok name", [f"setBlurEnabled={i}"
                                              for i in range(20)])
        with self.assertRaises(concepts.ConceptError):
            concepts.teach(state, "ok name", ["setBlurEnabled=true"],
                           examples=["x"] * 6)

    def test_store_cap(self):
        state = {}
        with mock.patch.object(concepts, "MAX_CONCEPTS", 2):
            concepts.teach(state, "aa", ["setBlurEnabled=true"])
            concepts.teach(state, "bb", ["setBlurEnabled=true"])
            with self.assertRaises(concepts.ConceptError):
                concepts.teach(state, "cc", ["setBlurEnabled=true"])


class RecallTests(unittest.TestCase):
    def setUp(self):
        self.state = {}
        concepts.teach(self.state, "glassy",
                       ["setBlurEnabled=true", "setTransparencyBase=0.6"],
                       examples=["make it glassy", "frosted glass windows"])
        concepts.teach(self.state, "gaming boost",
                       ["setGameModeAutoEnable=true"],
                       examples=["gaming mode on"])

    def test_recall_by_name_and_example(self):
        for text in ("make it glassy", "i want frosted glass windows",
                     "glassy"):
            hit = concepts.recall(self.state, text, presets=PRESETS)
            self.assertEqual(hit["verdict"], "RECALL", text)
            self.assertEqual(hit["name"], "glassy")
            self.assertEqual(hit["calls"][0]["name"], "setBlurEnabled")

    def test_recall_abstains_honestly(self):
        hit = concepts.recall(self.state, "write a poem about autumn",
                              presets=PRESETS)
        self.assertEqual(hit["verdict"], "ABSTAIN")

    def test_recall_ambiguous_between_concepts(self):
        concepts.teach(self.state, "glassy panel",
                       ["setSidebarEnabled=false"],
                       examples=["glassy panel please"])
        hit = concepts.recall(self.state, "make it glassy panel",
                              presets=PRESETS)
        self.assertIn(hit["verdict"], ("AMBIGUOUS", "RECALL"))
        if hit["verdict"] == "AMBIGUOUS":
            self.assertEqual(len(hit["candidates"]), 2)

    def test_conflict_with_builtin_preset_asks(self):
        concepts.teach(self.state, "minimal look",
                       ["setSpacingScale=0.7"],
                       examples=["make everything minimal"])
        hit = concepts.recall(self.state, "please make everything minimal",
                              presets=PRESETS)
        self.assertEqual(hit["verdict"], "CONFLICT_ASK")
        self.assertIn("preset", hit)

    def test_outcome_learning(self):
        concepts.record_outcome(self.state, "glassy", accepted=True)
        concepts.record_outcome(self.state, "glassy", accepted=True)
        rows = concepts.list_concepts(self.state)
        glassy = [r for r in rows if r["name"] == "glassy"][0]
        for p in glassy["posterior"]:
            self.assertEqual(p["p"], 1.0)
            self.assertEqual(p["n"], 2)
        concepts.record_outcome(self.state, "glassy", accepted=False)
        glassy = [r for r in concepts.list_concepts(self.state)
                  if r["name"] == "glassy"][0]
        self.assertTrue(all(0.0 < p["p"] < 1.0 for p in glassy["posterior"]))
        with self.assertRaises(concepts.ConceptError):
            concepts.record_outcome(self.state, "nope", accepted=True)


class ShareTests(unittest.TestCase):
    def test_export_is_recipe_only_and_import_round_trips(self):
        src = {}
        concepts.teach(src, "glassy", ["setBlurEnabled=true"],
                       examples=["make it glassy"])
        concepts.record_outcome(src, "glassy", accepted=True)
        diff = concepts.export_diff(src)
        blob = json.dumps(diff)
        self.assertNotIn("alpha", blob)
        self.assertNotIn("beta", blob)
        dst = {}
        res = concepts.import_diff(dst, json.loads(blob))
        self.assertEqual(res, {"added": 1, "updated": 0, "skipped": 0})
        res2 = concepts.import_diff(dst, json.loads(blob))
        self.assertEqual(res2["skipped"], 1)
        res3 = concepts.import_diff(dst, json.loads(blob), force=True)
        self.assertEqual(res3["updated"], 1)
        with self.assertRaises(concepts.ConceptError):
            concepts.import_diff(dst, [{"name": "x", "calls": "bad"}])
        with self.assertRaises(concepts.ConceptError):
            concepts.export_diff(src, ["missing"])


class DispatchIntegrationTests(unittest.TestCase):
    def test_weak_route_recalls_taught_concept_locally(self):
        state = {}
        concepts.teach(state, "frosted",
                       ["setBlurEnabled=true"],
                       examples=["make my windows all frosted glass"])
        outcome = dispatch("make my windows all frosted glass", state=state)
        self.assertEqual(outcome["action"], "local")
        self.assertIsNotNone(outcome.get("apply"))
        self.assertEqual(outcome["apply"]["calls"],
                         [{"name": "setBlurEnabled", "value": True}])
        self.assertIn("taught concept", outcome["answer"][0])

    def test_strong_local_plan_is_not_preempted(self):
        state = {}
        concepts.teach(state, "glassy", ["setBlurEnabled=true"],
                       examples=["make it glassy"])
        outcome = dispatch("make my bar thinner", state=state)
        # the cortex owns this confidently; concepts must not interfere
        self.assertEqual(outcome["action"], "local")
        if outcome.get("concept") is not None:
            self.fail("concepts preempted a confident local plan")


if __name__ == "__main__":
    unittest.main()
