"""B2 — config-drift scoring: live shell.json vs the preference posterior.

Contract under test (settings/optimize.py preference_drift + brain/prefs.py
config_drift + brain/brief.py opt-in section + CLI flag):

- PURE SCORING: plain (config, posterior) in, plain dict out; tools at
  default contribute nothing; only ranged numeric tools participate.
- HONEST ABSTAIN: posterior rows with fewer than 3 decisions are reported
  in ``thin`` and never scored — one decision is one decision.
- DRIFT = the config sits in a direction the user's own history rejects:
  a low p_accept on (group, direction) + a live value moved that way.
  Weighting: offset magnitude x rejection depth x capped evidence.
- OPT-IN ONLY: compose() without config_drift renders no drift section;
  with it, flags render; the CLI flag reads the live file read-only via
  the planner's own reader and degrades honestly when unreadable.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from assistant.capabilities.brain import brief as brief_mod
from assistant.capabilities.brain import prefs as prefs_mod
from assistant.capabilities.brain.ledger import Ledger
from assistant.capabilities.settings.optimize import preference_drift


_POSTERIOR = {
    ("bar", "increase"): {"p_accept": 0.15, "n": 8},   # rejected, strong
    ("appearance", "decrease"): {"p_accept": 0.2, "n": 5},  # rejected
    ("bar", "decrease"): {"p_accept": 0.8, "n": 6},    # liked
    ("launcher", "increase"): {"p_accept": 0.3, "n": 1},   # too thin
}


class PreferenceDriftTests(unittest.TestCase):
    def test_at_default_config_scores_zero(self) -> None:
        result = preference_drift({}, _POSTERIOR)
        self.assertEqual(result["score"], 0)
        self.assertEqual(result["flags"], [])
        self.assertEqual(result["evaluated"], 0)

    def test_rejected_direction_flags(self) -> None:
        config = {"bar": {"scale": 1.4}}  # bar.scale up from default 1.0
        result = preference_drift(config, _POSTERIOR)
        self.assertTrue(result["flags"])
        flag = result["flags"][0]
        self.assertEqual(flag["tool"], "setBarScale")
        self.assertEqual(flag["direction"], "increase")
        self.assertEqual(flag["p_accept"], 0.15)
        self.assertGreater(result["score"], 0)

    def test_liked_direction_does_not_flag(self) -> None:
        config = {"bar": {"scale": 0.85}}  # decrease, which the user likes
        result = preference_drift(config, _POSTERIOR)
        tools = [f["tool"] for f in result["flags"]]
        self.assertNotIn("setBarScale", tools)

    def test_thin_evidence_is_reported_not_scored(self) -> None:
        config = {"launcher": {"maxShown": 20}}  # launcher increase, n=1
        result = preference_drift(config, _POSTERIOR)
        thin_tools = [t["tool"] for t in result["thin"]]
        self.assertIn("setLauncherMaxShown", thin_tools)
        self.assertNotIn("setLauncherMaxShown",
                          [f["tool"] for f in result["flags"]])

    def test_score_bounds_and_monotonicity(self) -> None:
        small = preference_drift({"bar": {"scale": 1.1}}, _POSTERIOR)
        big = preference_drift({"bar": {"scale": 1.5}}, _POSTERIOR)
        self.assertGreaterEqual(big["score"], small["score"])
        for result in (small, big):
            self.assertLessEqual(result["score"], 1.0)
            self.assertGreaterEqual(result["score"], 0.0)

    def test_evidence_weight_capped(self) -> None:
        # n=8 (weight 0.8) vs n=100 (capped at 1.0): the cap holds.
        heavy = dict(_POSTERIOR)
        heavy[("bar", "increase")] = {"p_accept": 0.15, "n": 100}
        base = preference_drift({"bar": {"scale": 1.4}}, _POSTERIOR)
        capped = preference_drift({"bar": {"scale": 1.4}}, heavy)
        self.assertGreater(capped["score"], base["score"])
        self.assertLessEqual(capped["score"], 1.0)

    def test_pure_no_mutation(self) -> None:
        config = {"bar": {"scale": 1.3}}
        snapshot = json.dumps(config, sort_keys=True)
        posterior = {k: dict(v) for k, v in _POSTERIOR.items()}
        preference_drift(config, posterior)
        preference_drift(config, posterior)
        self.assertEqual(json.dumps(config, sort_keys=True), snapshot)


class PrefsAdapterTests(unittest.TestCase):
    def test_model_to_drift_end_to_end(self) -> None:
        model = prefs_mod.PreferenceModel()
        # The user rejects bar increases (5 rejections, 1 approval).
        for _ in range(5):
            model.observe("bar", "increase", accepted=False)
        model.observe("bar", "increase", accepted=True)
        config = {"bar": {"scale": 1.3}}
        drift = prefs_mod.config_drift(model, config)
        self.assertTrue(drift["flags"])
        self.assertEqual(drift["flags"][0]["tool"], "setBarScale")

    def test_hour_bucket_filtering(self) -> None:
        model = prefs_mod.PreferenceModel()
        # Night-time rejections only (bucket 0 = hours 0-5).
        for _ in range(5):
            model.observe("bar", "increase", accepted=False, hour=2)
        drift_day = prefs_mod.config_drift(
            model, {"bar": {"scale": 1.3}}, hour=14)  # afternoon: no evidence
        self.assertEqual(drift_day["flags"], [])
        drift_night = prefs_mod.config_drift(
            model, {"bar": {"scale": 1.3}}, hour=2)
        self.assertTrue(drift_night["flags"])


class BriefSectionTests(unittest.TestCase):
    def test_compose_without_drift_renders_no_section(self) -> None:
        b = brief_mod.compose(pending=[])
        self.assertIsNone(b.get("config_drift"))
        self.assertNotIn("config drift", brief_mod.render(b))

    def test_compose_with_drift_renders_flags(self) -> None:
        drift = {"score": 0.12, "evaluated": 9,
                 "flags": [{"path": "bar.scale", "live": 1.4,
                            "default": 1.0, "p_accept": 0.15, "n": 8,
                            "direction": "increase"}],
                 "thin": []}
        b = brief_mod.compose(pending=[], config_drift=drift)
        text = brief_mod.render(b)
        self.assertIn("config drift", text)
        self.assertIn("bar.scale", text)
        self.assertIn("0.15", text)

    def test_compose_with_clean_drift_renders_alignment(self) -> None:
        drift = {"score": 0.0, "evaluated": 9, "flags": [], "thin": []}
        text = brief_mod.render(brief_mod.compose(pending=[], config_drift=drift))
        self.assertIn("config drift: none", text)
        self.assertIn("9 settings checked", text)

    def test_cli_flag_end_to_end(self) -> None:
        import contextlib
        import io

        from assistant.capabilities.brain import cli as brain_cli
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "shell.json"
            target.write_text(json.dumps({"bar": {"scale": 1.4}}))
            ledger_path = str(Path(tmp) / "ledger.json")
            ledger = Ledger(ledger_path)
            for _ in range(6):
                pid = ledger.propose("settings", "bar.scale",
                                     {"old": 1.0, "new": 1.4},
                                     "bigger bar", 0.7)
                ledger.decide(pid, False)  # always reject bar increases
            argv = ["--ledger", ledger_path,
                    "brief", "--config-drift", "--target", str(target)]
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = brain_cli.main(argv)
            self.assertEqual(code, 0)
            text = out.getvalue()
            self.assertIn("config drift", text)
            self.assertIn("bar.scale", text)

    def test_cli_flag_degrades_on_unreadable_target(self) -> None:
        import contextlib
        import io

        from assistant.capabilities.brain import cli as brain_cli
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "shell.json"
            target.write_text("{not json")
            argv = ["--ledger", str(Path(tmp) / "none.json"),
                    "brief", "--config-drift", "--target", str(target)]
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = brain_cli.main(argv)
            self.assertEqual(code, 0)  # honest degradation, not a crash
            self.assertIn("config drift", out.getvalue())


if __name__ == "__main__":
    unittest.main()
