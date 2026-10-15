"""F23 tests — generated community docs (assistant/settings/catalog.py).

The contract: the committed docs/the generated tool catalog (`settings --gen-catalog`) and
scripts/completions/caelestia-assist.bash are OUTPUT of the generator;
a registry change that outdates them fails the build (the gen_adapter
--verify discipline applied to prose and completions).

Under test:

- the generated catalog names every tool exactly once, carries the
  pinned TOOL_COUNT, and marks preset membership;
- the bash completion enumerates the settings flags, preset names and
  tool names, and is deterministic;
- verify() flags MISSING and STALE artifacts (against a synthetic
  root), and generate()/verify() round-trip byte-identically;
- the COMMITTED artifacts in this repo verify clean (the drift guard).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from assistant.capabilities.settings import catalog
from assistant.capabilities.settings.presets import presets
from assistant.adapters.caelestia.registry import TOOL_COUNT, TOOL_SPECS


class GenerationTests(unittest.TestCase):
    def test_catalog_covers_every_tool_once(self) -> None:
        text = catalog.catalog_markdown()
        self.assertIn(f"**{TOOL_COUNT} tools**", text)
        for spec in TOOL_SPECS:
            self.assertIn(f"`{spec.name}`", text, spec.name)
        # determinism
        self.assertEqual(text, catalog.catalog_markdown())

    def test_catalog_marks_preset_membership(self) -> None:
        text = catalog.catalog_markdown()
        gaming = next(p for p in presets() if p["name"] == "gaming")
        first_tool = gaming["calls"][0][0]
        line = next(l for l in text.splitlines() if f"`{first_tool}` " in l)
        self.assertIn("gaming", line)

    def test_completion_deterministic_and_complete(self) -> None:
        a = catalog.completions_bash()
        b = catalog.completions_bash()
        self.assertEqual(a, b)
        for spec in TOOL_SPECS[:5]:
            self.assertIn(spec.name, a)
        for preset in presets():
            self.assertIn(preset["name"], a)
        self.assertIn("complete -F _caelestia_assist_settings", a)


class VerifyTests(unittest.TestCase):
    def test_generate_then_verify_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog.generate(root)
            self.assertEqual(catalog.verify(root), [])

    def test_missing_and_stale_are_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            problems = catalog.verify(root)
            self.assertEqual(len(problems), len(catalog.ARTIFACTS))
            self.assertTrue(all("MISSING" in p for p in problems))
            catalog.generate(root)
            artifact = root / catalog.ARTIFACTS[0][0]
            artifact.write_text("hand-edited\n", encoding="utf-8")
            problems = catalog.verify(root)
            self.assertEqual(len(problems), 1)
            self.assertIn("STALE", problems[0])

    def test_committed_artifacts_verify_clean(self) -> None:
        """The drift guard itself: the repo's committed docs must be the
        generator's exact output."""
        self.assertEqual(catalog.verify(), [])


if __name__ == "__main__":
    unittest.main()
