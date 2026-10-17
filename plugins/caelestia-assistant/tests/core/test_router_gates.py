"""F2 type-gated routing + structural floors (exponential-build-5, D1/D2).

The cue-kind agreement used to be a soft nudge (+/-0.55 * w_noun) that noun
hits and lexical scores could outrank — the D1 defect class. These tests
pin the constraint semantics:

- polarity verbs GATE candidate kind (off/hide/mute/disable/stop -> bool;
  direction -> numeric; move/put/switch + position -> enum), with a
  fallback that never gates silently to nothing;
- an Enabled prior for group nouns with on/off verbs;
- full name-atom coverage and adjacent name bigrams are specific
  addressing and floor above generic lexical overlap;
- digit words normalize ("24 hour" -> "twenty four hour") so enum
  vocabulary is reachable and numbers stop hijacking unrelated tools (D2).
"""
from __future__ import annotations

import unittest

from assistant.core.router import DEFAULT_STATE, Router


def top1(text: str):
    res = Router().route(text, state=DEFAULT_STATE, k=5)
    return (res.candidates[0].surface if res.candidates else None), res.verdict


class TypeGateD1Tests(unittest.TestCase):
    """The four documented confident-wrong routes (D1) become correct."""

    def test_disable_the_launcher_routes_enabled(self):
        self.assertEqual(top1("disable the launcher")[0], "setLauncherEnabled")

    def test_turn_off_volume_popup_routes_osd_toggle(self):
        self.assertIn(top1("turn off the volume popup")[0],
                      ("setEnableVolume", "setOsdEnabled"))

    def test_turn_off_music_visualizer_routes_visualiser(self):
        self.assertEqual(top1("turn off the music visualizer")[0],
                         "setVisualiserEnabled")

    def test_lock_screen_on_start_routes_lock_on_startup(self):
        self.assertEqual(top1("lock the screen when the shell starts")[0],
                         "setLockOnStartup")


class TypeGateMoreDefects(unittest.TestCase):
    """p12/p34/p39 and D2 from the build-5 defect list."""

    def test_mute_all_shell_sounds_routes_enabled(self):
        self.assertEqual(top1("mute all shell sounds")[0], "setSoundsEnabled")

    def test_bar_hide_until_hover(self):
        self.assertEqual(top1("let the bar hide until I hover")[0],
                         "setBarShowOnHover")

    def test_show_windows_in_workspace_indicators(self):
        self.assertEqual(top1("show windows in the workspace indicators")[0],
                         "setWorkspacesShowWindows")

    def test_use_24_hour_time_routes_clock_format(self):
        self.assertEqual(top1("use 24 hour time")[0], "setClockFormat")

    def test_hide_the_dashboard_routes_enabled(self):
        self.assertEqual(top1("hide the dashboard")[0], "setDashboardEnabled")


class TypeGateSafetyTests(unittest.TestCase):
    """The gate must never gate silently to nothing and never touch
    presets/surfaces."""

    def test_presets_survive_the_gate(self):
        self.assertEqual(top1("optimize for gaming")[0], "preset:gaming")
        self.assertEqual(top1("battery saving mode")[0], "preset:battery-saver")
        self.assertEqual(top1("make it minimal")[0], "preset:minimal")

    def test_gate_falls_back_when_tools_empty(self):
        # "mute the bar scale" names a float tool with a bool verb: the
        # gate suppresses it; whatever survives (or the fallback when
        # nothing does) must still return a ranked result, not crash and
        # not an empty candidate list.
        res = Router().route("mute the bar scale", state=DEFAULT_STATE, k=5)
        self.assertTrue(res.candidates)
        self.assertIn(res.verdict, ("ROUTED", "AMBIGUOUS", "ABSTAIN"))

    def test_no_regression_sample(self):
        """A sample of currently-correct routes stays correct under the gate."""
        cases = {
            "make the bar taller": "setBarScale",
            "put the bar at the top": "setBarPosition",
            "make everything pitch black": "setPitchBlack",
            "make the dock icons bigger": "setDockIconSize",
            "fewer notification popups at once": "setNotifsMaxPopups",
            "raise the max volume to 120": "setMaxVolume",
            "make the borders thicker": "setBorderThickness",
        }
        for text, expect in cases.items():
            self.assertEqual(top1(text)[0], expect, msg=text)


class EnabledPriorTests(unittest.TestCase):
    def test_enable_the_launcher(self):
        self.assertEqual(top1("enable the launcher")[0], "setLauncherEnabled")

    def test_stop_checking_for_updates(self):
        self.assertEqual(top1("stop checking for updates")[0], "setCheckUpdates")

    def test_stop_the_wallpaper_slideshow(self):
        self.assertEqual(top1("stop the wallpaper slideshow")[0],
                         "setSlideshowEnabled")


class EvidenceDeterminismTests(unittest.TestCase):
    """Stage C1 golden probe caught: the matched-terms evidence loop
    iterated a bare set, so PYTHONHASHSEED order decided WHICH
    matched-term line survived the pipeline's [:4] evidence cap — same
    input, different evidence across runs. The fix emits matched terms
    in sorted order; this pins the invariant."""

    def test_matched_term_evidence_is_sorted_and_complete(self):
        res = Router().route("hide the clock icon", state=DEFAULT_STATE, k=1)
        matched = [e for e in res.candidates[0].evidence
                   if e.startswith("matched '")]
        self.assertEqual(matched, sorted(matched),
                         "matched-term evidence must be emitted sorted")
        # both query atoms that hit the tool's doc tokens are present —
        # the cap may keep only the first lines, but the ROUTER's own
        # evidence list must not lose either
        self.assertIn("matched 'clock'", matched)
        self.assertIn("matched 'icon'", matched)


if __name__ == "__main__":
    unittest.main()
