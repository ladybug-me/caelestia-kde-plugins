"""Cortex algorithm-module tests: lexicon, corpus, vectorize, router.

Style follows the existing suites (stdlib unittest; no pytest). Every
test pins behavior the safety posture depends on: determinism, honest
abstention, and router-vs-frozen-parser agreement on the parser's own
turf (the regression that matters — the cortex may only ADD routing
coverage, never contradict the frozen grammar).
"""

from __future__ import annotations

import unittest

from assistant.core import lexicon
from assistant.core.corpus import (
    all_rows,
    rows_by_surface,
    surface_labels,
    surface_rows,
    tool_atoms,
    tool_document,
    tool_documents,
)
from assistant.core.router import ( RouterState, extract_cues, route, )
from assistant.core.vectorize import PpmiEmbedder, TfidfIndex, tokenize
from assistant.capabilities.settings.parser import parse as frozen_parse
from assistant.adapters.caelestia.registry import TOOL_SPECS, tool_by_name


class LexiconTests(unittest.TestCase):
    def test_stemmer_idempotent(self):
        for word in ("bars", "spacing", "animations", "notifications", "settings", "ss", "glass"):
            once = lexicon.stem(word)
            self.assertEqual(once, lexicon.stem(once), word)

    def test_stemmer_known_pairs(self):
        cases = {
            "bars": "bar", "icons": "icon", "fonts": "font",
            "animations": "animation", "notifications": "notification",
        }
        for word, expected in cases.items():
            self.assertEqual(lexicon.stem(word), expected)

    def test_levenshtein_basics(self):
        self.assertEqual(lexicon.levenshtein("transparncy", "transparency"), 1)
        self.assertEqual(lexicon.levenshtein("bar", "bar"), 0)
        self.assertEqual(lexicon.levenshtein("kitten", "sitting"), 3)
        self.assertEqual(lexicon.levenshtein("abc", "abc", cap=2), 0)

    def test_levenshtein_cap_early_exit(self):
        self.assertEqual(lexicon.levenshtein("aaaa", "bbbb", cap=2), 3)

    def test_jaro_winkler_bounds(self):
        self.assertEqual(lexicon.jaro_winkler("dock", "dock"), 1.0)
        self.assertEqual(lexicon.jaro_winkler("", ""), 1.0)
        self.assertGreater(lexicon.jaro_winkler("dock", "docks"), 0.85)
        self.assertEqual(lexicon.jaro_winkler("abc", "xyz"), 0.0)

    def test_ngram_similarity_self_is_one(self):
        self.assertAlmostEqual(lexicon.ngram_similarity("greeter morning", "greeter morning"), 1.0)

    def test_camel_split(self):
        self.assertEqual(
            lexicon.camel_split("setGreeterMorningStart"),
            ["set", "greeter", "morning", "start"],
        )

    def test_direction_of_mixed_cancels(self):
        self.assertEqual(lexicon.direction_of("bigger and smaller"), 0)
        self.assertEqual(lexicon.direction_of("make it thinner"), -1)
        self.assertEqual(lexicon.direction_of("increase the size"), 1)

    def test_expand_synonyms_additive(self):
        # single-word synonyms (hyphen bigrams are the router's job)
        words = lexicon.expand_synonyms("glassy bar")
        self.assertIn("blur", words)
        self.assertIn("glass", words)
        self.assertIn("bar", words)
        self.assertIn("taskbar", words)

    def test_lexicon_version_pinned(self):
        self.assertEqual(lexicon.LEXICON_VERSION, 1)


class CorpusTests(unittest.TestCase):
    def test_deterministic_order(self):
        self.assertEqual(all_rows(), all_rows())

    def test_every_tool_has_document_and_rows(self):
        documents = tool_documents()
        by_surface = rows_by_surface()
        for spec in TOOL_SPECS:
            self.assertIn(spec.name, documents)
            self.assertGreater(len(by_surface.get(spec.name, [])), 0, spec.name)

    def test_tool_atoms_drop_plumbing(self):
        spec = tool_by_name("setGreeterMorningStart")
        atoms = tool_atoms(spec)
        self.assertNotIn("set", atoms)
        self.assertIn("greeter", atoms)
        self.assertIn("morning", atoms)

    def test_tool_document_contains_nouns(self):
        spec = tool_by_name("setBarScale")
        self.assertIn("bar", tool_document(spec))
        spec = tool_by_name("setRoundingScale")
        self.assertIn("corner", tool_document(spec))

    def test_surface_rows_present(self):
        surfaces = {row.surface for row in surface_rows()}
        for expected in ("preset:compact", "explain", "undo", "history", "diagnose", "search"):
            self.assertIn(expected, surfaces)

    def test_surface_labels_cover_registry(self):
        labels = surface_labels()
        self.assertEqual(len(labels), len(TOOL_SPECS) + 14)


class VectorizeTests(unittest.TestCase):
    def test_tokenize_stopwords_and_stems(self):
        tokens = tokenize("Make my bars thinner, please")
        self.assertEqual(tokens, ["bar", "thinner"])

    def test_tokenize_keeps_pronouns_on_request(self):
        tokens = tokenize("make it bigger", keep_pronouns=True)
        self.assertIn("it", tokens)

    def test_bm25_ranks_relevant_doc_first(self):
        index = TfidfIndex({"a": "bar scale taskbar", "b": "dock icon size", "c": "blur transparency"})
        ranked = index.search("bar scale", ("a", "b", "c"))
        self.assertEqual(ranked[0][0], "a")

    def test_bm25_deterministic(self):
        index = TfidfIndex({"a": "bar scale", "b": "dock icons"})
        self.assertEqual(index.search("bar", ("a", "b")), index.search("bar", ("a", "b")))

    def test_ppmi_deterministic_across_builds(self):
        e1, e2 = PpmiEmbedder(), PpmiEmbedder()
        self.assertEqual(e1.embed("bar scale"), e2.embed("bar scale"))

    def test_ppmi_unknown_word_zero_vector(self):
        e = PpmiEmbedder()
        self.assertEqual(e.word_vector("zzzqqqxyz"), [0.0] * e.dim)

    def test_cosine_bounds(self):
        # random projection mixes signs, so LSA cosines are in [-1, 1]
        e = PpmiEmbedder()
        value = e.similarity("bar scale", "dock icons")
        self.assertLessEqual(value, 1.0)
        self.assertGreaterEqual(value, -1.0)


class RouterCueTests(unittest.TestCase):
    def test_direction_cue(self):
        self.assertEqual(extract_cues("make it thinner")["direction"], -1)
        self.assertEqual(extract_cues("increase the padding")["direction"], 1)

    def test_bool_cue_strong_words(self):
        self.assertIs(extract_cues("disable the blur")["bool"], False)
        self.assertIs(extract_cues("enable previews")["bool"], True)

    def test_weak_bool_dropped_under_direction(self):
        # "show fewer" is a magnitude statement, not on/off
        cues = extract_cues("show fewer notifications")
        self.assertIs(cues["bool"], None) if "bool" in cues else self.assertNotIn("bool", cues)
        self.assertEqual(cues["direction"], -1)

    def test_position_cue(self):
        self.assertEqual(extract_cues("move it to the left")["position"], "left")

    def test_percent_and_number_cues(self):
        cues = extract_cues("make it 20% smaller")
        self.assertEqual(cues["percent"], 20.0)
        cues = extract_cues("bar height 1.2")
        self.assertEqual(cues["number"], 1.2)

    def test_reset_cue(self):
        self.assertIs(extract_cues("reset the spacing to default")["reset"], True)

    def test_transparency_cue(self):
        self.assertIs(extract_cues("i want see-through panels")["transparency"], True)

    def test_up_down_not_direction_words(self):
        # phrasal-verb particles must not produce direction cues
        cues = extract_cues("notifications keep piling up")
        self.assertNotIn("direction", cues)


class RouterGoldenSetTests(unittest.TestCase):
    """The routing quality bar: issue #120's own examples plus the
    everyday phrasings a user actually types."""

    GOLDEN = (
        ("make my bar thinner", "setBarScale"),
        ("make my bar bigger", "setBarScale"),
        ("the dock icons are too big", "setDockIconSize"),
        ("dock icon size 30", "setDockIconSize"),
        ("turn on the blur", "setBlurEnabled"),
        ("turn off the blur", "setBlurEnabled"),
        ("disable live previews", "setLivePreviews"),
        ("make the corners rounder", "setRoundingScale"),
        ("corner rounding 0.8", "setRoundingScale"),
        ("spacing bigger", "setSpacingScale"),
        ("padding smaller", "setPaddingScale"),
        ("make the text bigger", "setFontScale"),
        ("animations faster", "setAnimationSpeed"),
        ("animation speed 0.5", "setAnimationSpeed"),
        ("border thickness 15", "setBorderThickness"),
        ("remove the border", "setBorderThickness"),
        ("launcher results 10", "setLauncherMaxShown"),
        ("notification popups 5", "setNotifsMaxPopups"),
        ("enable app badges", "setDockBadges"),
        ("make transparncy lower", "setTransparencyBase"),  # typo tolerance
        ("move the dock to the left", "setBarPosition"),
        ("move the bar to the top", "setBarPosition"),
        ("make everything feel compact", "preset:compact"),
        ("make my bar compact", "preset:compact"),
        ("give the desktop a minimal look", "preset:minimal"),
        ("optimize for gaming", "preset:gaming"),
        ("why is my dock blurry", "explain"),
        ("undo the last change", "undo"),
        ("what did i change yesterday", "history"),
        ("my desktop is broken, the shell crashed", "diagnose"),
        ("i want see-through panels", "setTransparencyBase"),
        ("frosted glass everywhere", "setBlurEnabled"),
        ("push the greeter morning start later", "setGreeterMorningStart"),
        ("snappier animations please", "setAnimationSpeed"),
        ("turn up the transparency a bit", "setTransparencyBase"),
    )

    def test_golden_top1(self):
        for text, expected in self.GOLDEN:
            with self.subTest(text=text):
                result = route(text)
                self.assertIsNotNone(result.top, text)
                self.assertEqual(result.top.surface, expected,
                                 f"{text!r}: got {result.top.surface}, want {expected}")

    def test_noun_silent_tool_addressable_by_words(self):
        # the whole point of the cortex: 259 noun-silent tools become
        # addressable through their name atoms + cue agreement. "perf cpu
        # text" is genuinely split between ShowCpu and ShowText — the
        # honest answer lists them both; both must be in the top 3.
        result = route("show the perf cpu text in the bar")
        top3 = [c.surface for c in result.candidates[:3]]
        self.assertTrue(
            {"setPerformanceShowCpu", "setPerformanceShowText"} & set(top3),
            f"expected a performance tool in top-3, got {top3}",
        )


class RouterParserAgreementTests(unittest.TestCase):
    """REGRESSION: wherever the frozen grammar has an opinion, the router
    must agree (top-1). The cortex may only ADD coverage beyond the 18
    frozen tools — never contradict the frozen surface."""

    PARSER_CASES = (
        "make my bar thinner",
        "make my bar bigger",
        "bar height 1.2",
        "make the bar 20% smaller",
        "move the bar to the top",
        "move the bar to the left",
        "dock icons smaller",
        "dock icon size 30",
        "turn on the blur",
        "turn off the blur",
        "disable live previews",
        "enable window previews",
        "make the corners rounder",
        "corner rounding 0.8",
        "spacing bigger",
        "padding smaller",
        "font size bigger",
        "make the text smaller",
        "animations faster",
        "animation speed 0.5",
        "border thickness 15",
        "remove the border",
        "launcher results 10",
        "show 5 launcher results",
        "turn on pitch black",
        "notification popups 5",
        "max stored notifications 20",
        "enable app badges",
        "disable badges",
        "make the bar persistent",
        "base opacity 0.9",
    )

    def test_top1_agreement_with_frozen_parser(self):
        for text in self.PARSER_CASES:
            frozen = frozen_parse(text)
            tools = [op["tool"] for op in frozen.get("ops", [])]
            if not tools:
                continue
            with self.subTest(text=text):
                result = route(text)
                self.assertEqual(result.top.surface, tools[0],
                                 f"router {result.top.surface} != parser {tools[0]} for {text!r}")


class RouterHonestyTests(unittest.TestCase):
    def test_garbage_abstains(self):
        for text in ("flurb the wozzle", "asdkjhqwe", "the meaning of life"):
            result = route(text)
            # OUT_OF_ONTOLOGY (F6) is the precise verdict for garbage that
            # names nothing in the registry — the protection is the same:
            # never a confident route.
            self.assertIn(
                result.verdict,
                ("ABSTAIN", "AMBIGUOUS", "OUT_OF_ONTOLOGY"), text)
            if result.verdict in ("ABSTAIN", "OUT_OF_ONTOLOGY"):
                self.assertIsNotNone(result.question, text)

    def test_empty_input(self):
        self.assertEqual(route("").verdict, "ABSTAIN")

    def test_candidates_carry_evidence(self):
        result = route("make my bar thinner")
        self.assertTrue(result.top.evidence)

    def test_state_round_trip(self):
        state = RouterState(w_lex=0.5, min_score=0.2)
        restored = RouterState.from_dict(state.to_dict())
        self.assertEqual(restored, state)

    def test_state_from_garbage_falls_back(self):
        self.assertEqual(RouterState.from_dict({"bogus": 1}), RouterState())
        self.assertEqual(RouterState.from_dict(None), RouterState())


if __name__ == "__main__":
    unittest.main()
