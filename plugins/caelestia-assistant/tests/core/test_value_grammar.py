"""F4 value & unit grammar (exponential-build-5, D2 + nlplan defect class).

Pins the value-resolution semantics the arena's nlplan suite measures:

- percent words: "20 percent smaller" is a percent cue, not the number 20;
- fraction-range tools: "raise the max volume to 120 percent" stores 1.2
  (the registry range 0.5-2.0 is a fraction; the UI shows percent);
- unit suffixes: "2 seconds" on a millisecond tool (default >= 1000)
  stores 2000;
- word multipliers: "half speed" multiplies by 0.5;
- enum literals: "24 hour" reaches TwentyFourHour via digit-word
  normalization + camel-split enum values;
- bool from strong polarity verbs: "stop the wallpaper slideshow" is
  bool-off even though "stop" is not in the parser's BOOL_OFF vocabulary;
- absolute words on bool tools: "make everything pitch black" is bool-on.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from assistant.core.pipeline import process
from assistant.core.router import DEFAULT_STATE, extract_cues


def plan_of(text: str):
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "shell.json"
        res = process(text, file_path=target, router_state=DEFAULT_STATE)
    return res


class PercentGrammarTests(unittest.TestCase):
    def test_percent_word_is_a_percent_not_a_number(self):
        cues = extract_cues("make the bar 20 percent smaller")
        self.assertEqual(cues.get("percent"), 20.0)
        self.assertNotIn("number", cues)  # the number must not double-count

    def test_bar_20_percent_smaller_multiplies(self):
        res = plan_of("make the bar 20 percent smaller")
        self.assertEqual(res.ops[0]["tool"], "setBarScale")
        self.assertEqual(res.ops[0]["action"], "multiply")
        self.assertEqual(res.ops[0]["value"], 0.8)

    def test_font_10_percent_bigger_multiplies(self):
        res = plan_of("make the font 10 percent bigger")
        self.assertEqual(res.ops[0]["tool"], "setFontScale")
        self.assertEqual(res.ops[0]["value"], 1.1)

    def test_fraction_range_tool_percent_is_stored_as_fraction(self):
        # services.maxVolume: float 0.5-2.0, default 1.0, UI shows percent
        res = plan_of("raise the max volume to 120 percent")
        self.assertEqual(res.ops[0]["tool"], "setMaxVolume")
        self.assertEqual(res.ops[0]["action"], "set")
        self.assertEqual(res.ops[0]["value"], 1.2)


class UnitGrammarTests(unittest.TestCase):
    def test_seconds_on_a_millisecond_tool(self):
        res = plan_of("set the osd hide delay to 2 seconds")
        self.assertEqual(res.ops[0]["tool"], "setHideDelay")
        self.assertEqual(res.ops[0]["value"], 2000)

    def test_seconds_on_the_expire_timeout(self):
        res = plan_of("notifications should time out after 5 seconds")
        self.assertEqual(res.ops[0]["tool"], "setDefaultExpireTimeout")
        self.assertEqual(res.ops[0]["value"], 5000)


class WordMultiplierTests(unittest.TestCase):
    def test_half_speed(self):
        res = plan_of("make animations half speed")
        self.assertEqual(res.ops[0]["tool"], "setAnimationSpeed")
        self.assertEqual(res.ops[0]["action"], "multiply")
        self.assertEqual(res.ops[0]["value"], 0.5)


class EnumLiteralTests(unittest.TestCase):
    def test_24_hour_reaches_twentyfourhour(self):
        res = plan_of("use the 24 hour clock format")
        self.assertEqual(res.ops[0]["tool"], "setClockFormat")
        self.assertEqual(res.ops[0]["value"], "TwentyFourHour")


class BoolPolarityTests(unittest.TestCase):
    def test_stop_is_bool_off(self):
        cues = extract_cues("stop the wallpaper slideshow")
        self.assertIs(cues.get("bool"), False)

    def test_stop_the_slideshow_plans_false(self):
        res = plan_of("stop the wallpaper slideshow")
        self.assertEqual(res.ops[0]["tool"], "setSlideshowEnabled")
        self.assertEqual(res.ops[0]["value"], False)

    def test_pitch_black_is_bool_on(self):
        res = plan_of("make everything pitch black")
        self.assertEqual(res.ops[0]["tool"], "setPitchBlack")
        self.assertEqual(res.ops[0]["value"], True)


if __name__ == "__main__":
    unittest.main()
