"""Paraphrase-equivalence battery for the compositional slot grammar (A1).

Design contract under test (the settings design notes (see module docstrings) §3.8, slots.py):

- FROZEN OUTPUTS UNCHANGED: every canonical phrasing below is resolved by
  parser.py's own frozen grammar, and its route is pinned here so a future
  grammar change cannot drift it silently.
- PARAPHRASES ROUTE IDENTICALLY: each paraphrase — including the ones the
  frozen grammar alone left AMBIGUOUS ("thin out the bar", "speed up the
  animations", "increase the transparency") — must produce the SAME
  (tool, action, value) route as its canonical partner.
- HONEST DECLINES STAY: bare percents, split requests, targetless
  directions, dead-ends and bool cues on numeric tools keep their frozen
  AMBIGUOUS/NO_INTENT verdicts; the slot grammar adds signal, never
  guesses (each decline case is pinned too).
- INTENSIFIERS ARE MAGNITUDE-NEUTRAL: "a bit"/"slightly" (light) and
  "a lot"/"significantly" (strong) never change the step count — the
  documented §3.1 determinism rule, now enforced by tests for both
  intensities.
- PURITY: parse() stays a pure function — identical calls, identical
  results (pinned with a double-call identity check).

Pair curation rule: the CANONICAL side of every pair is a phrasing the
frozen grammar resolves on its own (verified against §3.4's vocabularies
and the registry's noun groups); the PARAPHRASE side exercises the slot
grammar's composition. A pair whose canonical the frozen grammar cannot
resolve is a test-design bug, not a feature request.
"""

from __future__ import annotations

import unittest

from assistant.capabilities.settings.parser import parse


def route(text):
    """The routing identity of a parse result: verdict + (tool, action,
    value) triples. Notes carry the honest caveats and audit trails, not
    the route itself."""
    result = parse(text)
    return (result["verdict"],
            [(op["tool"], op["action"], op["value"]) for op in result["ops"]])


class SizeDimensionPairs(unittest.TestCase):
    """15+ paraphrase pairs, size axis (bar scale, dock icons, rounding,
    spacing, padding, border, font, launcher results)."""

    PAIRS = [
        ("make the bar thinner", "thin out the bar"),
        ("make the bar thinner", "shave the bar down"),
        ("make the bar thinner", "minimize the bar a little"),
        ("make the bar thicker", "fatten up the bar"),
        ("make the bar bigger", "enlarge the bar"),
        ("make the bar bigger", "expand the bar"),
        ("make the bar bigger", "boost the bar"),
        ("make the bar bigger", "bump up the bar"),
        ("make the bar smaller", "narrow the bar"),
        ("make the dock icons bigger", "enlarge the dock icons"),
        ("make the dock icons smaller", "trim the dock icons"),
        ("make the corner rounding bigger", "enlarge the corner rounding"),
        ("make the spacing smaller", "trim the spacing"),
        ("make the spacing roomier", "expand the spacing"),
        ("make the padding smaller", "narrow the padding"),
        ("make the font bigger", "enlarge the font"),
        ("make the font smaller", "shorten the font"),
        ("make the border thicker", "heighten the border"),
        ("make the border thinner", "narrow the border"),
        ("reduce the launcher results", "trim the launcher results"),
    ]

    def test_pairs_route_identically(self) -> None:
        for canonical, paraphrase in self.PAIRS:
            with self.subTest(canonical=canonical, paraphrase=paraphrase):
                self.assertEqual(route(canonical), route(paraphrase))

    def test_canonical_routes_are_pinned(self) -> None:
        self.assertEqual(route("make the bar thinner"),
                         ("INTENT", [("setBarScale", "step", -1)]))
        self.assertEqual(route("make the bar bigger"),
                         ("INTENT", [("setBarScale", "step", 1)]))
        self.assertEqual(route("make the dock icons bigger"),
                         ("INTENT", [("setDockIconSize", "step", 1)]))
        self.assertEqual(route("make the corner rounding bigger"),
                         ("INTENT", [("setRoundingScale", "step", 1)]))
        self.assertEqual(route("make the spacing smaller"),
                         ("INTENT", [("setSpacingScale", "step", -1)]))

    def test_multi_target_single_cue_applies_to_all(self) -> None:
        # The frozen single-phrase pattern, through the recovery path:
        # "enlarge the bar and the dock icons" -> two step ops.
        self.assertEqual(
            route("make the bar and the dock icons bigger"),
            route("enlarge the bar and the dock icons"))


class AnimationDimensionPairs(unittest.TestCase):
    """15+ paraphrase pairs, animation axis — including the inversion
    composition: "speed up" arrives as a pre-joined cue and composes to a
    step DOWN on durations (the documented §2/§3.4 inversion). Split
    verb-particles ("speed the animations up") are deliberately out of
    scope — a bounded recovery lexicon beats an over-broad regex."""

    PAIRS = [
        ("make the animations faster", "speed up the animations"),
        ("make the animations faster", "speed up the animation speed"),
        ("make the animations faster", "speeding up the animations"),
        ("make the animations faster", "speed up the transitions"),
        ("make the animations faster", "hurry up the animations"),
        ("make the animations slower", "slow down the animations"),
        ("make the animations slower", "slow down the transitions"),
        ("make the animations slower", "slowing down the animations"),
        ("make the transitions faster", "speed up the transitions"),
        ("make the motion faster", "speed up the motion"),
        ("make the transitions snappier", "speed up the transitions"),
        ("make the animations smoother", "slow down the animations"),
        ("make the animation speed faster", "speed up the animation speed"),
        ("make the transitions quicker", "speed up the transitions"),
        ("make the transitions slower", "slow down the motion"),
    ]

    def test_pairs_route_identically(self) -> None:
        for canonical, paraphrase in self.PAIRS:
            with self.subTest(canonical=canonical, paraphrase=paraphrase):
                self.assertEqual(route(canonical), route(paraphrase))

    def test_inversion_is_pinned_both_ways(self) -> None:
        self.assertEqual(route("make the animations faster"),
                         ("INTENT", [("setAnimationSpeed", "step", -1)]))
        self.assertEqual(route("make the animations slower"),
                         ("INTENT", [("setAnimationSpeed", "step", 1)]))

    def test_generic_words_decline_on_the_anim_axis(self) -> None:
        # "increase the animations" is ambiguous between more-speed and
        # more-duration: the composition declines (slots.py _compose), and
        # the frozen AMBIGUOUS verdict stands.
        verdict, ops = route("increase the animations")
        self.assertEqual(verdict, "AMBIGUOUS")
        self.assertEqual(ops, [])


class TransparencyDimensionPairs(unittest.TestCase):
    """15+ paraphrase pairs, transparency axis — the compositional core:
    ``base_step = sign(word) * polarity(noun)`` where "transparency" and
    "opacity" carry opposite polarity on the same knob (the opacity base).
    The canonical side uses the frozen trans phrases ("lighter",
    "less transparent", "more solid", "see-through") with a trans noun."""

    PAIRS = [
        ("make the transparency lighter", "increase the transparency"),
        ("make the transparency lighter", "more transparency please"),
        ("make the transparency lighter", "boost the transparency"),
        ("make the transparency lighter", "expand the transparency"),
        ("make the transparency lighter", "raise the transparency"),
        ("make the transparency lighter", "heighten the transparency"),
        ("make the transparency lighter", "reduce the opacity"),
        ("make the transparency lighter", "dial back the opacity"),
        ("make the transparency see-through", "increase the transparency"),
        ("make the transparency less transparent", "reduce the transparency"),
        ("make the transparency less transparent", "less transparency please"),
        ("make the transparency less transparent", "lower the transparency"),
        ("make the opacity more solid", "increase the opacity"),
        ("make the opacity more solid", "more opacity please"),
        ("make the opacity more solid", "boost the base opacity"),
        ("make the opacity more solid", "reduce the transparency"),
    ]

    def test_pairs_route_identically(self) -> None:
        for canonical, paraphrase in self.PAIRS:
            with self.subTest(canonical=canonical, paraphrase=paraphrase):
                self.assertEqual(route(canonical), route(paraphrase))

    def test_polarity_multiplication_is_pinned(self) -> None:
        # +1 (increase) x -1 (transparency polarity) = step DOWN on base.
        self.assertEqual(route("increase the transparency"),
                         ("INTENT", [("setTransparencyBase", "step", -1)]))
        # -1 (reduce) x +1 (opacity polarity) = step DOWN on base.
        self.assertEqual(route("reduce the opacity"),
                         ("INTENT", [("setTransparencyBase", "step", -1)]))
        # +1 (increase) x +1 (opacity polarity) = step UP on base.
        self.assertEqual(route("increase the opacity"),
                         ("INTENT", [("setTransparencyBase", "step", 1)]))
        # -1 (reduce) x -1 (transparency polarity) = step UP on base.
        self.assertEqual(route("reduce the transparency"),
                         ("INTENT", [("setTransparencyBase", "step", 1)]))

    def test_size_verbs_go_through_the_same_multiplication(self) -> None:
        # Pre-joined size cues on the trans knob compose by polarity too:
        # "boost the transparency" is step DOWN, not up.
        self.assertEqual(route("boost the transparency"),
                         ("INTENT", [("setTransparencyBase", "step", -1)]))
        self.assertEqual(route("boost the opacity"),
                         ("INTENT", [("setTransparencyBase", "step", 1)]))

    def test_audit_note_shows_the_composition(self) -> None:
        result = parse("increase the transparency")
        note = result["notes"][0]
        self.assertIn("compositional slot grammar", note)
        self.assertIn("polarity=transparency(-1)", note)
        self.assertIn("down on the opacity base", note)


class BoolDimensionPairs(unittest.TestCase):
    """15+ paraphrase pairs, bool axis — activation verbs and comparatives
    on bool tools map to on/off WITH the honest §3.5 note. Canonical sides
    use the frozen enable/disable/show/hide/turn-on vocabulary; the
    paraphrases use activate/deactivate and recovery comparatives."""

    PAIRS = [
        ("enable the blur", "activate the blur"),
        ("disable the blur", "deactivate the blur"),
        ("turn on the blur", "activate the background blur"),
        ("turn off the blur", "deactivate the background blur"),
        ("enable the blur", "activate the frosted glass"),
        ("enable live previews", "activate live previews"),
        ("disable live previews", "deactivate live previews"),
        ("turn on live previews", "activate the window previews"),
        ("turn off live previews", "deactivate the previews"),
        ("enable the always visible bar", "activate the always visible bar"),
        ("disable the auto-hide bar", "deactivate the auto hide bar"),
        ("enable pitch black", "activate the bezel mode"),
        ("disable pitch black", "deactivate the bezel mode"),
        ("show app badges", "activate the app badges"),
        ("hide app badges", "deactivate the badges"),
        ("increase the blur", "boost the blur"),
        ("increase the blur", "crank up the blur"),
        ("less blur", "dial back the blur"),
        ("less blur", "tone down the blur"),
    ]

    def test_pairs_route_identically(self) -> None:
        for canonical, paraphrase in self.PAIRS:
            with self.subTest(canonical=canonical, paraphrase=paraphrase):
                self.assertEqual(route(canonical), route(paraphrase))

    def test_comparative_on_bool_carries_the_honest_note(self) -> None:
        result = parse("boost the blur")
        self.assertEqual(route("boost the blur"),
                         ("INTENT", [("setBlurEnabled", "set", True)]))
        self.assertIn("on/off only", result["ops"][0]["note"])
        self.assertIn("boost", result["ops"][0]["note"])

    def test_bool_cue_on_numeric_tool_declines(self) -> None:
        # Activation wording on a numeric tool is meaningless: the slot
        # grammar declines and the frozen AMBIGUOUS verdict stands.
        verdict, ops = route("activate the bar scale")
        self.assertEqual(verdict, "AMBIGUOUS")
        self.assertEqual(ops, [])


class PositionDimensionPairs(unittest.TestCase):
    """15+ paraphrase pairs, position/enum axis — verb framings around the
    same position word (the position word itself is the signal; these pin
    the SHARED-grammar property: one position slot routes every verb
    framing identically).

    Known frozen-grammar boundary, pinned deliberately out of scope: a
    preposition that is also a bool word ("put the bar ON the left")
    hits the §3.3 step-6 split rule before any recovery hook — pinned
    below as split-AMBIGUOUS, the honest frozen answer."""

    PAIRS = [
        ("move the bar to the left", "put the bar at the left"),
        ("move the bar to the left", "place the bar at the left side"),
        ("move the bar to the left", "relocate the bar to the left"),
        ("move the bar to the left", "shift the bar to the left side"),
        ("move the bar to the top", "put the bar at the top"),
        ("move the bar to the top", "position the bar at the top edge"),
        ("move the bar to the bottom", "put the bar at the bottom"),
        ("move the bar to the bottom", "place the bar at the bottom side"),
        ("move the bar to the bottom", "put the bar at the bottom edge"),
        ("move the bar to the right", "put the bar at the right"),
        ("move the bar to the right", "relocate the bar to the right edge"),
        ("move the panel to the top", "put the panel at the top"),
        ("move the taskbar to the bottom", "place the taskbar at the bottom"),
        ("move the panel to the left", "position the panel at the left"),
        ("set the bar position to left", "move the bar to the left"),
    ]

    def test_pairs_route_identically(self) -> None:
        for canonical, paraphrase in self.PAIRS:
            with self.subTest(canonical=canonical, paraphrase=paraphrase):
                self.assertEqual(route(canonical), route(paraphrase))

    def test_position_routes_are_pinned(self) -> None:
        self.assertEqual(route("move the bar to the left"),
                         ("INTENT", [("setBarPosition", "set", "left")]))

    def test_bool_preposition_boundary_stays_split(self) -> None:
        # "put the bar on the left": the preposition "on" is a frozen bool
        # word, so the §3.3 step-6 split rule fires before any recovery —
        # the honest frozen answer, deliberately not overridden.
        verdict, ops = route("put the bar on the left")
        self.assertEqual(verdict, "AMBIGUOUS")
        self.assertEqual(ops, [])


class IntensifierSlotTests(unittest.TestCase):
    """The intensifier slot: extracted, reported, magnitude-neutral."""

    INTENSIFIER_PAIRS = [
        ("make the bar thinner", "make the bar a bit thinner"),
        ("make the bar thinner", "make the bar slightly thinner"),
        ("make the bar thinner", "make the bar a little thinner"),
        ("make the bar thinner", "make the bar somewhat thinner"),
        ("make the bar thinner", "make the bar a lot thinner"),
        ("make the bar thinner", "make the bar significantly thinner"),
        ("make the bar thinner", "make the bar much thinner"),
        ("make the bar thinner", "make the bar noticeably thinner"),
        ("thin out the bar", "thin out the bar a bit"),
        ("thin out the bar", "thin out the bar quite a lot"),
        ("speed up the animations", "speed up the animations a little"),
        ("speed up the animations", "significantly speed up the animations"),
        ("increase the transparency", "slightly increase the transparency"),
        ("increase the transparency", "increase the transparency a lot"),
        ("enlarge the bar", "enlarge the bar a tad"),
    ]

    def test_intensifiers_never_change_the_step(self) -> None:
        for plain, intensified in self.INTENSIFIER_PAIRS:
            with self.subTest(plain=plain, intensified=intensified):
                self.assertEqual(route(plain), route(intensified))

    def test_recovery_reports_the_intensifier_slot(self) -> None:
        result = parse("thin out the bar a bit")
        self.assertIn("intensifier=light", result["notes"][0])
        result = parse("thin out the bar a lot")
        self.assertIn("intensifier=strong", result["notes"][0])


class HonestDeclinePins(unittest.TestCase):
    """The recovery layer must NOT resolve what the frozen grammar
    deliberately leaves ambiguous — recovery adds signal, never guesses."""

    def test_bare_percent_stays_ambiguous(self) -> None:
        verdict, ops = route("bar scale 20%")
        self.assertEqual(verdict, "AMBIGUOUS")
        self.assertEqual(ops, [])

    def test_split_requests_stay_ambiguous(self) -> None:
        verdict, ops = route("bar scale 1.2 and blur on")
        self.assertEqual(verdict, "AMBIGUOUS")
        self.assertEqual(ops, [])

    def test_targetless_direction_stays_ambiguous(self) -> None:
        verdict, ops = route("make it smaller")
        self.assertEqual(verdict, "AMBIGUOUS")
        self.assertEqual(ops, [])

    def test_dock_dead_end_stays_ambiguous(self) -> None:
        # §3.6 dead-ends run at precedence 4, before any recovery hook.
        verdict, ops = route("move the dock to the left")
        self.assertEqual(verdict, "AMBIGUOUS")
        self.assertEqual(ops, [])

    def test_conflicting_recovery_cues_decline(self) -> None:
        # "enlarge" (+1) and "narrow" (-1) disagree: conflict, no recovery;
        # the frozen AMBIGUOUS verdict stands.
        verdict, ops = route("enlarge the bar and narrow it")
        self.assertEqual(verdict, "AMBIGUOUS")
        self.assertEqual(ops, [])

    def test_numbers_stay_on_the_frozen_paths(self) -> None:
        # A magnitude present means the frozen value-phrase paths own the
        # request; recovery declines even when its cue words are present.
        self.assertEqual(route("thin out the bar to 1.2"),
                         ("INTENT", [("setBarScale", "set", 1.2)]))
        verdict, ops = route("enlarge the bar by 20%")
        self.assertEqual(verdict, "AMBIGUOUS")  # percent-of-what question stands

    def test_purity_double_call_identity(self) -> None:
        for text in ("thin out the bar", "speed up the animations",
                     "increase the transparency", "activate the blur"):
            with self.subTest(text=text):
                self.assertEqual(parse(text), parse(text))


class FrozenSurfaceUnchangedTests(unittest.TestCase):
    """The frozen grammar's own outputs, re-pinned after the slots hook:
    every one of these is resolved BEFORE the recovery layer can run."""

    def test_frozen_basics_still_byte_identical(self) -> None:
        self.assertEqual(
            parse("make the bar thinner"),
            {"verdict": "INTENT",
             "ops": [{"tool": "setBarScale", "action": "step", "value": -1,
                      "raw": "thinner"}],
             "candidates": [], "question": None, "notes": [],
             "suggestions": []})
        self.assertEqual(
            parse("set the bar scale to 1.4")["ops"],
            [{"tool": "setBarScale", "action": "set", "value": 1.4,
              "raw": "1.4"}])
        self.assertEqual(parse("make my bar compact")["ops"][0]["tool"],
                         "setBarScale")  # preset path untouched
        self.assertEqual(parse("give the desktop a minimal look")["ops"][0]
                         ["tool"], "setBarScale")
        self.assertEqual(parse("set the accent color to blue")["verdict"],
                         "SUGGESTED")  # scheme path untouched
        self.assertEqual(parse("undo nothing here")["verdict"], "NO_INTENT")


if __name__ == "__main__":
    unittest.main()
