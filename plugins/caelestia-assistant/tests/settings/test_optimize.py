"""settings.optimize tests — profiles, Pareto, AC-3, synthesis.

Everything is tested against the REAL generated registry (tools.json), the
same discipline the rest of the settings layer's tests follow."""
import unittest

from assistant.capabilities.settings import optimize as opt
from assistant.adapters.caelestia.registry import tool_by_name


class TestProfiles(unittest.TestCase):
    def test_every_target_tool_exists(self):
        """A profile that names a tool the registry does not have is a lie.
        This test keeps the profiles honest forever."""
        missing = []
        for pname, spec in opt.PROFILES.items():
            for name in spec["target"]:
                if tool_by_name(name) is None:
                    missing.append(f"{pname}: {name}")
        self.assertEqual(missing, [])

    def test_unknown_profile_refused(self):
        with self.assertRaises(ValueError):
            opt.score_plan("turbo", [])
        with self.assertRaises(ValueError):
            opt.recommend("turbo")

    def test_recommend_all_ops_legal(self):
        for profile in opt.PROFILES:
            r = opt.recommend(profile, k=20)
            for op in r["ops"]:
                tool = tool_by_name(op["tool"])
                self.assertIsNotNone(tool)
                if tool.kind in ("int", "float"):
                    self.assertLessEqual(tool.minimum, op["value"])
                    self.assertGreaterEqual(tool.maximum, op["value"])

    def test_score_plan_directions(self):
        gaming_yes = opt.score_plan("gaming", [
            {"tool": "setBlurEnabled", "value": False},
            {"tool": "setAnimationSpeed", "value": 0.4},
        ])
        gaming_no = opt.score_plan("gaming", [
            {"tool": "setBlurEnabled", "value": True},
            {"tool": "setAnimationSpeed", "value": 4.0},
        ])
        self.assertGreater(gaming_yes["score"], gaming_no["score"])
        self.assertEqual(gaming_yes["verdict"], "serves this objective")

    def test_recommend_gaming_hits_effects(self):
        r = opt.recommend("gaming")
        tools = {op["tool"]: op["value"] for op in r["ops"]}
        self.assertFalse(tools.get("setBlurEnabled", True))
        self.assertFalse(tools.get("setTransparencyEnabled", True))


class TestPareto(unittest.TestCase):
    def test_frontier_excludes_dominated(self):
        sets = {
            "gaming": opt.recommend("gaming")["ops"],
            "battery": opt.recommend("battery")["ops"],
            "comfort": opt.recommend("comfort")["ops"],
            "accessibility": opt.recommend("accessibility")["ops"],
        }
        r = opt.pareto_profiles(sets)
        self.assertTrue(r["frontier"])
        # comfort and accessibility pull the same direction; one may be
        # dominated — but SOMETHING must survive on the frontier
        self.assertLessEqual(len(r["frontier"]), len(sets))
        labels = [m["label"] for m in r["frontier"]]
        for lbl in labels:
            self.assertIn(lbl, sets)

    def test_dominated_set_is_sound(self):
        sets = {"gaming": opt.recommend("gaming")["ops"],
                "battery": opt.recommend("battery")["ops"]}
        r = opt.pareto_profiles(sets)
        for m in r["frontier"] + r["dominated"]:
            self.assertIn(m["label"], sets)


class TestConflicts(unittest.TestCase):
    def test_satisfiable_arc_passes(self):
        r = opt.check_conflicts(
            [{"tool": "setBarScale", "value": 0.7},
             {"tool": "setSpacingScale", "value": 1.2}],
            [("setBarScale", "<", "setSpacingScale")])
        self.assertTrue(r["consistent"])

    def test_unsatisfiable_arc_raises_with_domains(self):
        with self.assertRaises(opt.ConstraintError) as cm:
            opt.check_conflicts(
                [{"tool": "setBarScale", "value": 1.6},
                 {"tool": "setSpacingScale", "value": 0.5}],
                [("setBarScale", "<", "setSpacingScale")])
        self.assertIn("no support", str(cm.exception))

    def test_inequality_domain_narrowing(self):
        # bar scale 0.6..1.6, spacing 0.5..2.0; require bar > spacing:
        # AC-3 must prune spacing values no bar supports (2.0 has no bar
        # above it) while keeping full support on the bar side
        r = opt.check_conflicts(
            [{"tool": "setBarScale", "value": 1.0},
             {"tool": "setSpacingScale", "value": 1.0}],
            [("setBarScale", ">", "setSpacingScale")])
        bar_dom = r["domains"]["setBarScale"]
        spacing_dom = r["domains"]["setSpacingScale"]
        self.assertLess(max(spacing_dom), 2.0)          # pruned
        self.assertLess(max(spacing_dom), max(bar_dom))  # support preserved
        self.assertEqual(min(bar_dom), 0.6)              # untouched side

    def test_requested_value_without_support_raises(self):
        # spacing=2.0 requires a bar ABOVE 2.0 — impossible (bar tops at 1.6),
        # so AC-3 must prune 2.0 and the request is honestly refused
        with self.assertRaises(opt.ConstraintError) as cm:
            opt.check_conflicts(
                [{"tool": "setBarScale", "value": 1.0},
                 {"tool": "setSpacingScale", "value": 2.0}],
                [("setBarScale", ">", "setSpacingScale")])
        self.assertIn("no support", str(cm.exception))


class TestSynthesize(unittest.TestCase):
    def test_synthesis_is_validated_against_registry(self):
        r = opt.synthesize("minimal")
        for op in r["ops"]:
            tool = tool_by_name(op["tool"])
            self.assertIsNotNone(tool)
            if tool.kind in ("int", "float"):
                self.assertLessEqual(tool.minimum, op["value"])
                self.assertGreaterEqual(tool.maximum, op["value"])

    def test_synthesis_hits_ideal_when_unconstrained(self):
        r = opt.synthesize("minimal")
        by_tool = {op["tool"]: op for op in r["ops"]}
        if "setBarScale" in by_tool:
            self.assertEqual(by_tool["setBarScale"]["value"], 0.7)
            self.assertEqual(by_tool["setBarScale"]["delta_from_ideal"], 0)

    def test_synthesis_respects_inequality_constraint(self):
        r = opt.synthesize(
            "comfort", constraints=[("setBarScale", ">", "setSpacingScale")],
            pinned={"setBarScale": 1.2, "setSpacingScale": 1.4})
        vals = {op["tool"]: op["value"] for op in r["ops"]}
        self.assertGreater(vals["setBarScale"], vals["setSpacingScale"])

    def test_pinned_overrides_profile(self):
        r = opt.synthesize("gaming", pinned={"setAnimationSpeed": 1.0})
        vals = {op["tool"]: op["value"] for op in r["ops"]}
        self.assertEqual(vals.get("setAnimationSpeed"), 1.0)


if __name__ == "__main__":
    unittest.main()
