"""Explainability tests (issue #120).

Pins the issue's own example shapes against a temp target file:
- "why is my dock blurry" -> the causal blur rule (gated on
  transparency.enabled && blur), citing BlurOffsets.qml:17;
- value readback in "The X is currently set to Y" style with validation,
  default and the C++ declaration citation;
- read-only: no file is ever written by --explain / explain();
- ambiguity does not guess;
- offline honesty: rules needing live session flags (_gamemode/_light)
  do not fire from the CLI.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from assistant.capabilities.settings import explain
from assistant.capabilities.settings.explain import (
    ExplainError,
    explain as explain_fn,
    resolve_spec,
)


def _write(dir_path: Path, doc: dict) -> Path:
    target = dir_path / "shell.json"
    target.write_text(json.dumps(doc), encoding="utf-8")
    return target


class ResolveTests(unittest.TestCase):
    def test_path_and_tool_name_resolve(self) -> None:
        self.assertEqual(resolve_spec("bar.scale").path, "bar.scale")
        self.assertEqual(resolve_spec("setBarScale").path, "bar.scale")

    def test_question_resolution_issue_example(self) -> None:
        spec = resolve_spec("why is my dock blurry")
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(spec.path, "appearance.blur")

    def test_preference_adjectives(self) -> None:
        self.assertEqual(resolve_spec("why is my bar big").path, "bar.scale")
        self.assertEqual(resolve_spec("why does the bar hide").path,
                         "bar.persistent")

    def test_unresolvable_returns_none(self) -> None:
        self.assertIsNone(resolve_spec("why is the sky blue"))


class ExplainTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        add = self.dir / "shell.json"
        add.write_text(json.dumps({
            "bar": {"scale": 1.2, "persistent": False},
            "appearance": {
                "blur": True,
                "transparency": {"enabled": True},
                "anim": {"durations": {"scale": 2.0}},
            },
        }), encoding="utf-8")
        self.target = add

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_issue_example_blur(self) -> None:
        result = explain_fn("why is my dock blurry", self.target)
        self.assertTrue(result["ok"])
        self.assertTrue(result["rule"])
        self.assertIn("Blur is enabled because", result["answer"])
        self.assertIn("transparency is currently active", result["answer"])
        self.assertIn("BlurOffsets.qml:17", result["cites"][0])

    def test_animations_rule_with_alternation(self) -> None:
        result = explain_fn("why are my animations so slow", self.target)
        self.assertTrue(result["rule"])
        # {slower|faster} resolves against the first numeric value (2.0 > 1)
        self.assertIn("slower", result["answer"])
        self.assertIn("2", result["answer"])
        self.assertIn("appearanceconfig.cpp", result["cites"][0])

    def test_value_readback_style(self) -> None:
        result = explain_fn("bar.scale", self.target)
        self.assertTrue(result["ok"])
        self.assertFalse(result["rule"])
        self.assertIn("The bar.scale is currently set to 1.2", result["answer"])
        self.assertIn("range 0.6-1.6", result["answer"])
        self.assertIn("default 1", result["answer"])
        # the citation line is DERIVED from the committed tools.json (the
        # upstream file shifts lines on every re-pin; the test must follow
        # the artifact, not a hand-copied number)
        spec = resolve_spec("bar.scale")
        decl = next(c[0] for c in spec.citations
                    if Path(c[0].rsplit(":", 1)[0]).suffix == ".hpp")
        self.assertIn(decl.rsplit("/", 1)[-1], result["answer"])

    def test_unset_key_uses_registry_default(self) -> None:
        result = explain_fn("border.thickness", self.target)
        self.assertIn("default 10", result["answer"])

    def test_read_only_never_writes(self) -> None:
        before = self.target.read_bytes()
        history = self.dir / "shell.json.assistant-history.json"
        explain_fn("why is my dock blurry", self.target)
        explain_fn("bar.scale", self.target)
        self.assertEqual(self.target.read_bytes(), before)
        self.assertFalse(history.exists())

    def test_ambiguity_raises(self) -> None:
        # "bar" alone noun-matches setBarScale and setBarPosition at the
        # same position — a genuine tie that must not guess.
        with self.assertRaises(ExplainError):
            explain_fn("why is the bar like that", self.target)

    def test_offline_session_flags_do_not_fire(self) -> None:
        # The GameMode transparency rule needs _gamemode (live session
        # state); from the file-only CLI it must NOT fire.
        self.target.write_text(json.dumps({
            "appearance": {"transparency": {"enabled": True}},
            "utilities": {"gameMode": {"disableShellTransparency": True}},
        }), encoding="utf-8")
        result = explain_fn("appearance.transparency.enabled", self.target)
        self.assertFalse(result["rule"])

    def test_rule_respects_when_predicate(self) -> None:
        # blur on but transparency OFF -> the "no effect" rule fires instead
        self.target.write_text(json.dumps({
            "appearance": {"blur": True, "transparency": {"enabled": False}},
        }), encoding="utf-8")
        result = explain_fn("appearance.blur", self.target)
        self.assertTrue(result["rule"])
        self.assertIn("no effect", result["answer"])

    def test_cli_explain_smoke(self) -> None:
        from assistant.capabilities.settings import cli
        rc = cli.main(["--explain", "why is my dock blurry",
                       "--file", str(self.target)])
        self.assertEqual(rc, 0)


class RuleDataTests(unittest.TestCase):
    def test_every_rule_has_citations_and_predicates(self) -> None:
        for rule in explain.EXPLAIN_RULES if hasattr(explain, "EXPLAIN_RULES") \
                else []:
            self.assertTrue(rule["cites"])
            self.assertIn("when", rule)
            self.assertIn("answer", rule)

    def test_cited_rule_files_exist(self) -> None:
        repo = Path(__file__).resolve().parents[3]
        if not (repo / "shell" / "plugin" / "src" / "Caelestia" / "Config").is_dir():
            self.skipTest("shell/ tree absent — citation guards skip")
        from assistant.adapters.caelestia.registry import EXPLAIN_RULES
        for rule in EXPLAIN_RULES:
            for cite in rule["cites"]:
                rel = cite.split(":")[0]
                self.assertTrue((repo / rel).exists(),
                                f"{rel} (cited by {rule['path']}) is missing")


if __name__ == "__main__":
    unittest.main()
