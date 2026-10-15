"""Parser tests for the settings layer (the settings design notes (see module docstrings) §3, §9).

The parser is a PURE function (no file I/O, no environment, no clock, no
randomness); these tests pin the grammar by executable example:

- issue #120's five example sentences, with the exact verdicts/ops the settings design notes (see module docstrings)
  §2 promised for them;
- absolute values pass through unvalidated (range REJECTION is the
  planner's job — "set the bar scale to 9" is still an INTENT op here);
- relative percents, bare percent (AMBIGUOUS except on the transparency
  base, whose own UI displays percent), bool magnitude pass-through;
- enum words, reset-to-default, animation/transparency inversions (the
  direction SIGN is the contract: "faster" must LOWER durations scale);
- multi-value and targetless requests are AMBIGUOUS, never guessed;
- scheme/wallpaper requests are SUGGESTED with inert
  SUGGESTED_NOT_EXECUTED strings only;
- determinism and the exact result shape; purity is proven by running
  parse() with builtins.open patched to raise.
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, List
from unittest import mock

from assistant.capabilities.settings.parser import parse

# Issue #120's five example sentences (the settings design notes (see module docstrings) §2.1).
SENTENCE_THINNER = "Make my bar thinner."
SENTENCE_DOCK_LEFT = "Move the dock to the left."
SENTENCE_BLUR = "Increase the blur."
SENTENCE_COMPACT = "Make everything feel more compact."
SENTENCE_MINIMAL = "Give the desktop a minimal look."


class Issue120SentenceTests(unittest.TestCase):
    def test_bar_thinner_is_intent_step_down(self) -> None:
        result = parse(SENTENCE_THINNER)
        self.assertEqual(result["verdict"], "INTENT")
        self.assertEqual(
            result["ops"],
            [{"tool": "setBarScale", "action": "step", "value": -1, "raw": "thinner"}],
        )

    def test_dock_left_is_ambiguous_dead_end(self) -> None:
        result = parse(SENTENCE_DOCK_LEFT)
        self.assertEqual(result["verdict"], "AMBIGUOUS")
        self.assertEqual(result["ops"], [])
        self.assertEqual(result["candidates"][0]["tool"], "setBarPosition")
        self.assertIn("bar.entries", result["notes"][0])
        self.assertIn("Nexus", result["notes"][0])
        self.assertIn("move the bar to the left", result["question"])

    def test_increase_blur_is_intent_bool_on_with_note(self) -> None:
        result = parse(SENTENCE_BLUR)
        self.assertEqual(result["verdict"], "INTENT")
        self.assertEqual(len(result["ops"]), 1)
        op = result["ops"][0]
        self.assertEqual(op["tool"], "setBlurEnabled")
        self.assertEqual(op["action"], "set")
        self.assertIs(op["value"], True)
        self.assertIn("on/off only", op["note"])
        self.assertIn("no strength to increase", op["note"])

    def test_compact_preset_is_intent_three_ops(self) -> None:
        result = parse(SENTENCE_COMPACT)
        self.assertEqual(result["verdict"], "INTENT")
        self.assertEqual(
            result["ops"],
            [
                {"tool": "setBarScale", "action": "set", "value": 0.85, "raw": "more compact"},
                {"tool": "setSpacingScale", "action": "set", "value": 0.9, "raw": "more compact"},
                {"tool": "setPaddingScale", "action": "set", "value": 0.9, "raw": "more compact"},
            ],
        )
        self.assertTrue(any("preset 'compact'" in note for note in result["notes"]))

    def test_minimal_preset_is_intent_four_ops(self) -> None:
        result = parse(SENTENCE_MINIMAL)
        self.assertEqual(result["verdict"], "INTENT")
        self.assertEqual(
            [(op["tool"], op["action"], op["value"]) for op in result["ops"]],
            [
                ("setBarScale", "set", 0.8),
                ("setSpacingScale", "set", 0.9),
                ("setPaddingScale", "set", 0.9),
                ("setRoundingScale", "set", 0.9),
            ],
        )
        self.assertTrue(any("preset 'minimal'" in note for note in result["notes"]))


class AbsoluteAndRelativeTests(unittest.TestCase):
    def test_absolute_values_become_set_ops_pass_through(self) -> None:
        # In range: a plain set op. Out of range: STILL a set op — the
        # parser never applies registry ranges (§3); REJECTION is the
        # planner's job (§4.4).
        result = parse("set the bar scale to 1.4")
        self.assertEqual(result["verdict"], "INTENT")
        self.assertEqual(
            result["ops"],
            [{"tool": "setBarScale", "action": "set", "value": 1.4, "raw": "1.4"}],
        )
        out = parse("set the bar scale to 9")
        self.assertEqual(out["verdict"], "INTENT")
        self.assertEqual(len(out["ops"]), 1)
        op = out["ops"][0]
        self.assertEqual((op["tool"], op["action"], op["value"]), ("setBarScale", "set", 9.0))
        self.assertNotIn("error", op)

    def test_percent_comparatives_are_multiply_ops(self) -> None:
        bigger = parse("make the bar 20% bigger")
        self.assertEqual(bigger["verdict"], "INTENT")
        self.assertEqual(
            bigger["ops"],
            [{"tool": "setBarScale", "action": "multiply", "value": 1.2, "raw": "20% bigger"}],
        )
        smaller = parse("make the bar 20% smaller")
        self.assertEqual(
            smaller["ops"],
            [{"tool": "setBarScale", "action": "multiply", "value": 0.8, "raw": "20% smaller"}],
        )

    def test_bare_percent_on_scale_tool_is_ambiguous(self) -> None:
        result = parse("bar scale 20%")
        self.assertEqual(result["verdict"], "AMBIGUOUS")
        self.assertEqual(result["ops"], [])
        self.assertIn("percent of what?", result["question"])

    def test_bare_percent_on_transparency_is_absolute(self) -> None:
        # §3.4: the transparency base's own UI displays percent
        # (AppearancePage.qml:143), so "40%" means an absolute 0.4 there.
        result = parse("transparency 40%")
        self.assertEqual(result["verdict"], "INTENT")
        self.assertEqual(
            result["ops"],
            [{"tool": "setTransparencyBase", "action": "set", "value": 0.4, "raw": "40%"}],
        )


class BoolMagnitudeTests(unittest.TestCase):
    """§3.4/§3.5: a number/percent on a bool tool is never coerced by the
    parser — the op carries the raw magnitude and the PLANNER rejects it
    ("this setting is on/off only; it has no magnitude")."""

    def test_magnitude_on_bool_tool_passes_through_uncoerced(self) -> None:
        for sentence, raw, value in (
            ("blur 50%", "50%", 0.5),
            ("set blur to 5", "5", 5.0),
        ):
            result = parse(sentence)
            self.assertEqual(result["verdict"], "INTENT", msg=sentence)
            self.assertEqual(
                result["ops"],
                [{"tool": "setBlurEnabled", "action": "set", "value": value, "raw": raw}],
                msg=sentence,
            )


class EnumAndResetTests(unittest.TestCase):
    def test_position_words_set_the_enum(self) -> None:
        for sentence, word in (("move the bar to the left", "left"),
                               ("put the bar at the top", "top")):
            result = parse(sentence)
            self.assertEqual(result["verdict"], "INTENT", msg=sentence)
            self.assertEqual(
                result["ops"],
                [{"tool": "setBarPosition", "action": "set", "value": word, "raw": word}],
                msg=sentence,
            )

    def test_reset_returns_registry_default(self) -> None:
        # One value phrase ("reset") applies to every noun-matched tool
        # (§3.3 step 6): both bar tools are named by "bar" here.
        result = parse("reset the bar scale")
        self.assertEqual(result["verdict"], "INTENT")
        self.assertEqual(
            [(op["tool"], op["action"], op["value"]) for op in result["ops"]],
            [("setBarScale", "set", 1.0), ("setBarPosition", "set", "bottom")],
        )

    def test_borderless_is_absolute_zero(self) -> None:
        # §3.4 noun-list parenthetical: "borderless" -> absolute 0 (the
        # shipped UI's own "Set to 0 for a borderless look").
        result = parse("make the shell borderless")
        self.assertEqual(result["verdict"], "INTENT")
        self.assertEqual(
            result["ops"],
            [{"tool": "setBorderThickness", "action": "set", "value": 0.0, "raw": "borderless"}],
        )


class InversionTests(unittest.TestCase):
    def test_animation_inversion_direction_signs(self) -> None:
        # §9: "faster" must LOWER durations scale (step -1 -> planner
        # computes old - 0.25); "slower" must raise it (+1).
        faster = parse("make animations faster")
        self.assertEqual(faster["verdict"], "INTENT")
        op = faster["ops"][0]
        self.assertEqual((op["tool"], op["action"], op["value"]),
                         ("setAnimationSpeed", "step", -1))
        self.assertIn("durations scale: lower = faster", op["note"])
        slower = parse("make animations slower")["ops"][0]
        self.assertEqual((slower["tool"], slower["action"], slower["value"]),
                         ("setAnimationSpeed", "step", 1))

    def test_transparency_inversions(self) -> None:
        # §9: "more transparent" -> base DOWN (step -1), "more opaque" -> UP.
        down = parse("make the transparency more transparent")["ops"][0]
        self.assertEqual((down["tool"], down["value"]), ("setTransparencyBase", -1))
        up = parse("make the transparency more opaque")["ops"][0]
        self.assertEqual((up["tool"], up["value"]), ("setTransparencyBase", 1))


class AmbiguityTests(unittest.TestCase):
    def test_multiple_value_phrases_are_ambiguous(self) -> None:
        result = parse("bar scale 1.2 and blur on")
        self.assertEqual(result["verdict"], "AMBIGUOUS")
        self.assertEqual(result["ops"], [])
        self.assertIn("split that into separate requests", result["question"])

    def test_targetless_direction_lists_numeric_candidates(self) -> None:
        result = parse("make it smaller")
        self.assertEqual(result["verdict"], "AMBIGUOUS")
        self.assertEqual(result["ops"], [])
        # §3.3 step 8: every numeric size tool in registry order —
        # appended the two int counts (setNotifsMaxPopups, setNotifsMaxNotifs).
        self.assertEqual(
            [cand["tool"] for cand in result["candidates"]],
            [
                "setBarScale", "setDockIconSize", "setRoundingScale", "setSpacingScale",
                "setPaddingScale", "setFontScale", "setBorderThickness", "setLauncherMaxShown",
                "setNotifsMaxPopups", "setNotifsMaxNotifs",
            ],
        )
        self.assertIn("which setting did you mean?", result["question"])

    def test_hello_world_is_no_intent_listing_everything(self) -> None:
        result = parse("hello world")
        self.assertEqual(result["verdict"], "NO_INTENT")
        joined = "\n".join(result["notes"])
        for name in (
            "setBarScale", "setBarPosition", "setDockIconSize", "setBarPersistent",
            "setLivePreviews", "setBlurEnabled", "setTransparencyBase", "setRoundingScale",
            "setSpacingScale", "setPaddingScale", "setFontScale", "setAnimationSpeed",
            "setBorderThickness", "setLauncherMaxShown", "setPitchBlack",
            "setNotifsMaxPopups", "setNotifsMaxNotifs", "setDockBadges",
        ):
            self.assertIn(name, joined)
        self.assertIn("--list-tools", result["question"])


class SuggestedTests(unittest.TestCase):
    def test_accent_color_is_suggested_with_inert_strings(self) -> None:
        result = parse("make my accent color blue")
        self.assertEqual(result["verdict"], "SUGGESTED")
        self.assertEqual(result["ops"], [])
        self.assertEqual(len(result["suggestions"]), 2)
        for suggestion in result["suggestions"]:
            self.assertTrue(suggestion.startswith("SUGGESTED_NOT_EXECUTED: "))
        self.assertIn("caelestia scheme list", result["suggestions"][0])
        self.assertIn("caelestia scheme set -n <name>", result["suggestions"][1])

    def test_wallpaper_is_suggested_with_inert_strings(self) -> None:
        result = parse("give me a new wallpaper")
        self.assertEqual(result["verdict"], "SUGGESTED")
        self.assertEqual(result["ops"], [])
        joined = "\n".join(result["suggestions"])
        self.assertIn("SUGGESTED_NOT_EXECUTED: caelestia wallpaper -f <path>", joined)
        self.assertIn("SUGGESTED_NOT_EXECUTED: caelestia wallpaper -r (random)", joined)


class ExtendedCoreToolParserTests(unittest.TestCase):
    """Three tools added to the 18-tool core after the original 14
    (the settings design notes (see module docstrings) §16): setPitchBlack, setNotifsMaxPopups, setNotifsMaxNotifs —
    and the preserved dead-ends around them."""

    def test_pitch_black_bool_wording(self) -> None:
        for sentence, value, raw in (("enable bezel mode", True, "enable"),
                                     ("turn on pitch black mode", True, "turn on"),
                                     ("turn off pitch black", False, "turn off"),
                                     ("disable bezel mode", False, "disable")):
            result = parse(sentence)
            self.assertEqual(result["verdict"], "INTENT", msg=sentence)
            self.assertEqual(
                result["ops"],
                [{"tool": "setPitchBlack", "action": "set", "value": value, "raw": raw}],
                msg=sentence,
            )

    def test_make_shell_pitch_black_is_implicit_bool_on(self) -> None:
        # §3.4: the phrase itself is the on-cue (the mirror of
        # "borderless" -> 0), because "make the shell pitch black" carries
        # no on/off word for the standard §3.4 path to scan.
        result = parse("make the shell pitch black")
        self.assertEqual(result["verdict"], "INTENT")
        self.assertEqual(
            result["ops"],
            [{"tool": "setPitchBlack", "action": "set", "value": True, "raw": "pitch black"}],
        )
        # "make my bar pitch black" names the bar too; the bool phrase
        # applies only to the bool tool, so no bar op appears (§3.3 step 6).
        bar_too = parse("make my bar pitch black")
        self.assertEqual(
            [(o["tool"], o["value"]) for o in bar_too["ops"]],
            [("setPitchBlack", True)],
        )

    def test_pitch_black_comparative_and_reset_use_standard_paths(self) -> None:
        # §3.5: comparatives on bool tools map to on/off WITH the note.
        less = parse("less pitch black")
        self.assertEqual(
            [(o["tool"], o["action"], o["value"]) for o in less["ops"]],
            [("setPitchBlack", "set", False)],
        )
        self.assertIn("on/off only", less["ops"][0]["note"])
        # "reset" keeps its §3.4 reading (registry default = false).
        reset = parse("reset the pitch black mode")
        self.assertEqual(
            [(o["tool"], o["action"], o["value"]) for o in reset["ops"]],
            [("setPitchBlack", "set", False)],
        )

    def test_notifs_max_popups_absolute_and_direction(self) -> None:
        result = parse("set the max notification popups to 5")
        self.assertEqual(result["verdict"], "INTENT")
        self.assertEqual(
            result["ops"],
            [{"tool": "setNotifsMaxPopups", "action": "set", "value": 5.0, "raw": "5"}],
        )
        reduced = parse("reduce notification popups")
        self.assertEqual(
            [(o["tool"], o["action"], o["value"]) for o in reduced["ops"]],
            [("setNotifsMaxPopups", "step", -1)],
        )
        # 0 is in range: the shipped stepper starts at 0
        # (NotificationPreferencesPage.qml:142). The parser never range-
        # checks; the planner does.
        zero = parse("notification popups 0")
        self.assertEqual(
            [(o["tool"], o["action"], o["value"]) for o in zero["ops"]],
            [("setNotifsMaxPopups", "set", 0.0)],
        )

    def test_notifs_max_notifs_absolute_and_reset(self) -> None:
        result = parse("keep at most 200 stored notifications")
        self.assertEqual(result["verdict"], "INTENT")
        self.assertEqual(
            result["ops"],
            [{"tool": "setNotifsMaxNotifs", "action": "set", "value": 200.0, "raw": "200"}],
        )
        reset = parse("reset the stored notifications limit")
        self.assertEqual(
            [(o["tool"], o["action"], o["value"]) for o in reset["ops"]],
            [("setNotifsMaxNotifs", "set", 50)],
        )

    def test_notifications_on_off_dead_end_is_preserved(self) -> None:
        # §3.6: the on/off dead-end keeps firing BEFORE per-tool matching —
        # the clarifying question points at the count tools instead.
        for sentence in ("turn notifications off", "enable notifications",
                         "show only 5 notification popups"):
            result = parse(sentence)
            self.assertEqual(result["verdict"], "NO_INTENT", msg=sentence)
            self.assertEqual(result["ops"], [])
            self.assertIn("no upstream 'notifications enabled' boolean", result["notes"][0])
            self.assertIn("no notifications on/off setting", result["question"])

    def test_transparency_on_off_dead_end_is_preserved(self) -> None:
        result = parse("turn off transparency")
        self.assertEqual(result["verdict"], "NO_INTENT")
        self.assertEqual(result["ops"], [])
        self.assertIn("appearance.transparency.enabled exists upstream", result["notes"][0])
        self.assertIn("transparency LEVEL", result["question"])

    def test_drag_threshold_and_other_rejected_candidates_stay_out(self) -> None:
        # bar.dragThreshold is grounded but NOT a tool (the settings design notes (see module docstrings) §16): a natural
        # sentence naming it must not invent a tool; "bar ... 50" hits the
        # pre-existing bar-scale reading, unaffected by these tools.
        result = parse("set the bar drag threshold to 50")
        self.assertEqual(
            [(o["tool"], o["action"], o["value"]) for o in result["ops"]],
            [("setBarScale", "set", 50.0)],
        )
        # "increase the drag threshold" (no bar word): a targetless size
        # direction — the fixed candidate list, with no drag-threshold tool.
        targetless = parse("increase the drag threshold")
        self.assertEqual(targetless["verdict"], "AMBIGUOUS")
        self.assertNotIn("setBarDragThreshold", [c["tool"] for c in targetless["candidates"]])


class DockBadgesParserTests(unittest.TestCase):
    """The fourth tool added to the 18-tool core (the settings design notes (see module docstrings) §16):
    setDockBadges, plus the bare-"dock" collision traces pinned to their
    honest outcomes. The noun surface "app badges|badges|badge" mirrors the
    shipped control label (BarDock.qml:77) and collides with no other tool
    noun; the word "dock" is only present when the user volunteers it, and
    an on/off value phrase reaches ONLY bool tools (§3.3 step 6 kind filter),
    so setDockIconSize (int) receives no op — never a silent iconSize write."""

    def test_dock_badges_bool_wording(self) -> None:
        for sentence, value, raw in (
            ("show app badges", True, "show"),
            ("enable the app badges", True, "enable"),
            ("turn off app badges", False, "turn off"),
            ("hide badges", False, "hide"),
        ):
            result = parse(sentence)
            self.assertEqual(result["verdict"], "INTENT", msg=sentence)
            self.assertEqual(
                result["ops"],
                [{"tool": "setDockBadges", "action": "set", "value": value, "raw": raw}],
                msg=sentence,
            )

    def test_dock_word_does_not_hijack_badges_on_off_wording(self) -> None:
        # "dock" also noun-matches setDockIconSize, but the bool cue fits
        # only the bool tool: the plan stays a clean single setDockBadges op
        # and bar.dock.iconSize is never written. (Pre-change this phrasing
        # was AMBIGUOUS — "what should I do with setDockIconSize?" — so the
        # extension strictly improves it; see the settings design notes (see module docstrings) §16.)
        for sentence, value in (("hide the dock badges", False),
                                ("show the dock badges", True)):
            result = parse(sentence)
            self.assertEqual(result["verdict"], "INTENT", msg=sentence)
            self.assertEqual(
                [(o["tool"], o["action"], o["value"]) for o in result["ops"]],
                [("setDockBadges", "set", value)],
                msg=sentence,
            )

    def test_turn_off_badges_on_the_dock_is_split_ambiguous(self) -> None:
        # "turn off ... on the dock" carries TWO distinct bool phrases
        # ("turn off" and the bare word "on") — §3.3 step 6's split
        # AMBIGUOUS, the same verdict the 17-tool parser already gave.
        result = parse("turn off badges on the dock")
        self.assertEqual(result["verdict"], "AMBIGUOUS")
        self.assertEqual(result["ops"], [])
        self.assertIn("split that into separate requests", result["question"])

    def test_dock_badges_number_is_never_coerced(self) -> None:
        # §3.4/§3.5: a number on a bool tool passes through uncoerced; the
        # PLANNER rejects it ("on/off only; it has no magnitude") — same
        # contract as blur in BoolMagnitudeTests.
        result = parse("set the badges to 3")
        self.assertEqual(result["verdict"], "INTENT")
        self.assertEqual(
            result["ops"],
            [{"tool": "setDockBadges", "action": "set", "value": 3.0, "raw": "3"}],
        )

    def test_dock_badges_comparative_uses_section_3_5_path(self) -> None:
        # Dock-free comparative: §3.5 maps it to on/off WITH the honest note.
        result = parse("make badges bigger")
        self.assertEqual(result["verdict"], "INTENT")
        self.assertEqual(
            [(o["tool"], o["action"], o["value"]) for o in result["ops"]],
            [("setDockBadges", "set", True)],
        )
        self.assertIn("on/off only", result["ops"][0]["note"])
        self.assertIn("no strength to bigger", result["ops"][0]["note"])
        self.assertIn("app badges", result["ops"][0]["note"])

    def test_dock_badges_comparative_with_dock_is_the_known_residual(self) -> None:
        # KNOWN RESIDUAL, pinned honestly (the settings design notes (see module docstrings) §"Registry scope
        # extension" collision analysis): a size comparative + the
        # user-volunteered word "dock" steps iconSize TOO — the same
        # pre-existing bare-"dock" hazard already documented for the
        # previewScales family ("make the dock popouts bigger" behaves
        # identically on the 17-tool parser). On/off wording never takes
        # this path (see the tests above).
        result = parse("make the dock badges bigger")
        self.assertEqual(result["verdict"], "INTENT")
        self.assertEqual(
            [(o["tool"], o["action"], o["value"]) for o in result["ops"]],
            [("setDockIconSize", "step", 1), ("setDockBadges", "set", True)],
        )

    def test_reset_app_badges_returns_registry_default(self) -> None:
        result = parse("reset the app badges")
        self.assertEqual(result["verdict"], "INTENT")
        self.assertEqual(
            [(o["tool"], o["action"], o["value"]) for o in result["ops"]],
            [("setDockBadges", "set", True)],
        )

    def test_badges_noun_leaves_the_five_dead_ends_untouched(self) -> None:
        # The §3.6 dead-ends are byte-identical: no "badge" word
        # was added to any dead-end pattern, so these keep their prior
        # verdicts (the notifications dead-end is also covered in
        # ExtendedCoreToolParserTests; the dock+position one is #120 sentence 2).
        self.assertEqual(parse("turn notifications off")["verdict"], "NO_INTENT")
        self.assertEqual(parse("turn off transparency")["verdict"], "NO_INTENT")
        self.assertEqual(parse("change the font family")["verdict"], "NO_INTENT")
        self.assertEqual(parse("make the shell pitch black")["verdict"], "INTENT")


class PurityAndShapeTests(unittest.TestCase):
    SENTENCES: List[str] = [
        SENTENCE_THINNER, SENTENCE_DOCK_LEFT, SENTENCE_BLUR,
        SENTENCE_COMPACT, SENTENCE_MINIMAL,
        "set the bar scale to 1.4", "make the bar 20% bigger",
        "hello world", "make my accent color blue",
        "make the shell pitch black", "set the max notification popups to 5",
        "keep at most 200 stored notifications", "turn notifications off",
        "show app badges", "hide the dock badges", "turn off badges on the dock",
    ]

    def test_parse_is_deterministic_with_the_documented_shape(self) -> None:
        for sentence in self.SENTENCES:
            result = parse(sentence)
            self.assertEqual(
                set(result.keys()),
                {"verdict", "ops", "candidates", "question", "notes", "suggestions"},
                msg=sentence,
            )
            self.assertEqual(result, parse(sentence), msg=sentence)

    def test_parse_never_opens_files(self) -> None:
        # §3: parse() is pure — no file I/O of any kind. Proven by running
        # it with builtins.open patched to blow up.
        with mock.patch("builtins.open", side_effect=AssertionError("parser must not open files")):
            for sentence in self.SENTENCES:
                result: Dict[str, Any] = parse(sentence)
                self.assertIn(result["verdict"], ("INTENT", "AMBIGUOUS", "NO_INTENT", "SUGGESTED"))


if __name__ == "__main__":
    unittest.main()
