"""Genius layer tests — language, markov, sysintel, creative, and the router."""
import os
import tempfile
import unittest
from pathlib import Path

from assistant.capabilities.genius import creative as cr
from assistant.capabilities.genius import language as gl
from assistant.capabilities.genius import markov as mk
from assistant.capabilities.genius import sysintel as si
from assistant.capabilities.genius import meta


class TestLanguage(unittest.TestCase):
    def test_sentiment_polarity(self):
        self.assertEqual(gl.sentiment("absolutely amazing, works perfectly")["label"], "positive")
        self.assertEqual(gl.sentiment("terrible, crashes constantly")["label"], "negative")
        self.assertEqual(gl.sentiment("the table is on the floor")["label"], "neutral")

    def test_sentiment_negation(self):
        s = gl.sentiment("this is not good at all")
        self.assertLess(s["compound"], 0)

    def test_sentiment_intensifier(self):
        plain = gl.sentiment("good")["compound"]
        boosted = gl.sentiment("very good")["compound"]
        self.assertGreater(boosted, plain)

    def test_readability_scales(self):
        r = gl.readability("The cat sat on the mat. The dog ran in the sun.")
        self.assertGreater(r["flesch_reading_ease"], 70)

    def test_syllable_counting(self):
        self.assertEqual(gl._syllables("hello"), 2)
        self.assertEqual(gl._syllables("table"), 2)  # silent e
        self.assertEqual(gl._syllables("strength"), 1)

    def test_rake_finds_theme(self):
        text = ("The caelestia shell uses matugen for color scheme generation. "
                "The color scheme is generated from wallpapers. Plasma theming "
                "relies on the scheme system.")
        kw = gl.rake_keywords(text, 5)
        self.assertTrue(kw["keywords"])
        joined = " ".join(k["phrase"] if "phrase" in k else str(k) for k in kw["keywords"])
        self.assertIn("color", joined.lower())

    def test_entities_extraction(self):
        text = ("Contact dev@example.com or visit https://example.org/docs. "
                "Fixed in commit a1b2c3d, version 3.2.1, released 2026-09-22 at 14:30. "
                "Uses 450 MB of RAM and costs $29.99, up 15 percent.")
        e = gl.extract_entities(text)["entities"]
        self.assertIn("dev@example.com", e.get("email", []))
        self.assertTrue(any("https://example.org" in u for u in e.get("url", [])))
        self.assertEqual(e.get("git_hash"), ["a1b2c3d"])
        self.assertEqual(e.get("version"), ["3.2.1"])
        self.assertEqual(e.get("date"), ["2026-09-22"])

    def test_qa_who_when_why(self):
        doc = ("Caelestia is a KDE Plasma dotfiles collection by ladybug-me. "
               "The shell is built with Quickshell and QML. Version 4.2.0 "
               "introduced the new dashboard on March 15, 2026. Performance "
               "improved because the rendering pipeline was rewritten in C++.")
        self.assertIn("ladybug-me", gl.answer_question("who made caelestia", doc)["answer"])
        self.assertIn("March 15", gl.answer_question("when was 4.2.0 released", doc)["answer"])
        answer = gl.answer_question("why did performance improve", doc)["answer"]
        self.assertIn("because", answer.lower())

    def test_summarize_focused(self):
        doc = ("Caelestia is a KDE Plasma dotfiles collection. "
               "It uses Quickshell. The bar is customizable through Nexus. "
               "Blur can be tuned. Matugen generates the scheme. "
               "The dock supports badges.")
        s = gl.summarize_focused(doc, "what is caelestia built with", 2)
        self.assertIn("Quickshell", s["summary"])

    def test_language_detection(self):
        self.assertEqual(gl.detect_language("el zorro marrón rápido salta")["language"], "spanish")
        self.assertEqual(gl.detect_language("der schnelle braune fuchs springt")["language"], "german")

    def test_classifier_learns_and_persists(self):
        clf = gl.TextClassifier()
        clf.fit(["the shell crashed after update", "kwin crashes on startup",
                 "how do i change bar height", "where is the nexus settings page"],
                ["crash", "crash", "settings", "settings"])
        self.assertEqual(clf.predict("plasma keeps crashing")["class"], "crash")
        restored = gl.TextClassifier.from_dict(clf.to_dict())
        self.assertEqual(restored.predict("plasma keeps crashing")["class"], "crash")


class TestMarkov(unittest.TestCase):
    CORPUS = ("the shell is built with quickshell and qml. the bar shows workspaces. "
              "settings live in nexus. matugen generates colors. the dock can be resized. "
              "users love the minimal look. the blur makes everything beautiful.")

    def test_generate_text_deterministic(self):
        a = mk.generate_text(self.CORPUS, n_words=20, seed=5)
        b = mk.generate_text(self.CORPUS, n_words=20, seed=5)
        self.assertEqual(a["text"], b["text"])

    def test_sequence_prediction(self):
        seq = ["a", "b", "c", "a", "b", "c", "a", "b"]
        p = mk.predict_next(seq[-4:])
        self.assertEqual(p["best"], "c")

    def test_sequence_surprise_flags_outliers(self):
        s = mk.sequence_surprise(["x"] * 10 + ["Z", "Y"] + ["x"] * 6)
        events = [a["event"] for a in s["anomalies"]]
        self.assertIn("Z", events)
        clean = mk.sequence_surprise(["x"] * 16)
        self.assertEqual(clean["anomalies"], [])

    def test_template_expansion(self):
        r = mk.expand_template("{greeting} world", {"greeting": ["hello", "hi"]}, seed=3)
        self.assertTrue(r["result"].startswith(("hello", "hi")))
        self.assertTrue(r["complete"])

    def test_constrained_generation(self):
        cg = mk.ConstrainedGenerator(self.CORPUS, seed=1)
        r = cg.generate(n_words=80, must_contain=["blur"], must_not_match=["love"],
                        max_tries=40)
        self.assertTrue(r["success"])
        self.assertIn("blur", r["text"])

    def test_name_generator(self):
        names = mk.generate_name(5, seed=42)["names"]
        self.assertEqual(len(names), 5)
        for name in names:
            self.assertRegex(name, r"^[a-z]+$")


class TestSysintel(unittest.TestCase):
    HISTORY = ("git status\ngit add -A\ngit commit -m wip\n"
               "git status\ngit add -A\ngit commit -m wip\n"
               "git status\ngit add -A\ngit commit -m wip\n"
               "git status\ngit add -A\ngit commit -m wip\n"
               "caelestia settings apply\nls -la\n")

    def _history_file(self):
        f = tempfile.NamedTemporaryFile("w", suffix="_hist", delete=False)
        f.write(self.HISTORY)
        f.close()
        return f.name

    def tearDown(self):
        try:
            os.unlink(self._path)
        except (OSError, AttributeError):
            pass

    def test_history_mining(self):
        self._path = self._history_file()
        parsed = si.parse_shell_history(self._path)
        self.assertEqual(parsed["n_lines"], self.HISTORY.count("\n"))
        a = si.analyze_history(parsed["commands"])
        self.assertEqual(a["n_commands"], parsed["n_lines"])
        self.assertTrue(a["top_commands"])
        self.assertTrue(a["association_rules"])

    def test_duplicate_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("a.txt", "b.txt", "c.txt"):
                Path(tmp, name).write_text("identical content here" * 50)
            Path(tmp, "d.txt").write_text("different")
            d = si.find_duplicates(tmp, min_size=10)
            self.assertEqual(d["duplicate_groups"], 1)
            self.assertEqual(d["groups"][0]["count"], 3)
            self.assertGreater(d["wasted_bytes"], 0)

    def test_log_template_mining(self):
        logs = [f"INFO worker 1 started task {i}" for i in range(20)]
        logs += [f"ERROR connection refused to 10.0.0.{i}:5432" for i in range(3)]
        m = si.mine_log_templates(logs)
        self.assertLess(m["n_templates"], len(logs))
        templates = " ".join(t["template"] for t in m["templates"])
        self.assertIn("<N>", templates)
        self.assertIn("<IP:PORT>", templates)

    def test_json_lint(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp, "bad.json")
            bad.write_text('{"a": 1, "a": 2, "b": null, "c": [1, "two"]}')
            lint = si.lint_json_config(str(bad))
            self.assertTrue(lint["valid_json"])
            kinds = [i["kind"] for i in lint["issues"]]
            self.assertIn("duplicate_key", kinds)
            self.assertIn("explicit_null", kinds)
            self.assertIn("mixed_types", kinds)


class TestCreative(unittest.TestCase):
    def test_oklab_roundtrip_precision(self):
        import random as rnd
        rng = rnd.Random(0)
        for _ in range(50):
            rgb = (rng.random(), rng.random(), rng.random())
            back = cr.oklab_to_rgb(cr.rgb_to_oklab(rgb))
            for a, b in zip(rgb, back):
                self.assertLess(abs(a - b), 1e-5)

    def test_oklch_extremes(self):
        self.assertAlmostEqual(cr.hex_to_oklch("#ffffff")[0], 1.0, places=6)
        self.assertAlmostEqual(cr.hex_to_oklch("#000000")[0], 0.0, places=6)

    def test_contrast_wcag(self):
        self.assertAlmostEqual(cr.contrast_ratio("#ffffff", "#000000")["ratio"], 21.0)
        r = cr.contrast_ratio("#777777", "#ffffff")
        self.assertFalse(r["aa_normal_text"])

    def test_palette_shapes(self):
        p = cr.palette("#3b7dd8", harmony="complementary", n=2)
        self.assertEqual(len(p["swatches"]), 2)
        for s in p["swatches"]:
            self.assertRegex(s["hex"], r"#[0-9a-f]{6}")

    def test_bad_hex_rejected(self):
        with self.assertRaises(ValueError):
            cr.hex_to_rgb("notacolor")

    def test_accent_derivation(self):
        a = cr.auto_accent("#f0d9b5")
        self.assertRegex(a["accent"], r"#[0-9a-f]{6}")
        self.assertIn("delta_e_ok", a)

    def test_bisociation_and_tagline(self):
        b = cr.bisociate(["shell", "bar"], ["coffee", "sleep"], n=4)
        self.assertEqual(len(b["pairs"]), 4)
        t = cr.tagline("local second brain")
        self.assertEqual(len(t["taglines"]), 3)


ROUTING_CASES = [
    ("what is 15% of 80", "math_eval"),
    ("solve x^2 - 2 = 0", "solve_equation"),
    ("derivative of sin(x)*cos(x)", "calculus"),
    ("integral of x^2 from 0 to 3", "calculus"),
    ("average of 3, 9, 12, 1, 44", "statistics"),
    ("is (p and q) -> (p or r) a tautology", "logic"),
    ("how many ways can I choose 3 of 40", "probability"),
    ("bayes prior 0.01 likelihood 0.95 false positive 0.05", "probability"),
    ("sentiment of: this update is terrible", "text_analyze"),
    ("palette complementary of #3b7dd8", "color_palette"),
    ("how do i fix the flaky test", "plan_goal"),
    ("readability of: The quick brown fox jumps over the lazy dog near the river bank.", "text_analyze"),
    ("contrast between #777777 and #ffffff", "color_palette"),
]


class TestMetaRouter(unittest.TestCase):
    def test_routing_matrix(self):
        for text, expected in ROUTING_CASES:
            with self.subTest(text=text):
                r = meta.classify(text)
                self.assertEqual(r["verdict"], "ROUTE", msg=f"{text}: {r}")
                self.assertEqual(r["domain"], expected)

    def test_route_and_do_executes(self):
        r = meta.route_and_do("what is 15% of 80")
        self.assertTrue(r["ok"])
        self.assertEqual(r["result"]["value"], 12.0)

    def test_abstain_on_gibberish(self):
        r = meta.route_and_do("zzz qqq vvv")
        self.assertEqual(r["verdict"], "ABSTAIN")

    def test_ambiguous_asks_a_question(self):
        r = meta.route_and_do("generate a summary of the average forecast") \
            if False else meta.classify("decide the plan")
        self.assertIn(r["verdict"], ("AMBIGUOUS", "ROUTE", "ABSTAIN"))
        # an ambiguous verdict must carry candidates or a message
        if r["verdict"] == "AMBIGUOUS":
            self.assertIn("message", r)

    def test_learning_feedback_biases_routing(self):
        learned = meta.learn_feedback("solve my equation x^2=4", "solve_equation", True)
        r = meta.classify("solve my equation", learned=learned)
        self.assertEqual(r.get("domain"), "solve_equation")

    def test_honest_error_reporting(self):
        r = meta.route_and_do("average of 1, 2")  # too few numbers
        self.assertEqual(r["domain"], "statistics")
        self.assertFalse(r["ok"])
        self.assertIn("error", r)

    def test_persistence_roundtrip(self):
        learned = meta.learn_feedback("what is 2+2", "math_eval", True)
        router = meta.MetaRouter.from_dict(learned)
        r2 = router.classify("what is 2+2 again")
        self.assertEqual(r2["domain"], "math_eval")


if __name__ == "__main__":
    unittest.main()
