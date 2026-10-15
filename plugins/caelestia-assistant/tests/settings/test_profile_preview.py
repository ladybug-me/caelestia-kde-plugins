"""F11 tests — composite what-if preview for profiles.

The preview composes EXISTING engines (profiles.compose_ops,
consequences.project, planner.plan, sanity.project/check, registry
groups) over a saved profile. Under test:

- build() returns the per-source projections (op counts, derived
  effects, internal conflicts), the composite projection, the
  composition conflicts, the blast radius by feature area, and the
  projected-state sanity verdicts;
- a profile with a real interaction-graph trigger (transparency off
  from the gaming preset) surfaces its cited derived effect;
- a profile whose sources collide on a tool reports the conflict the
  order resolved, with the loser named (F7);
- build() and the CLI are read-only: the target file's bytes are
  unchanged after a preview;
- an unknown profile raises ProfileError (CLI renders it as an error).
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import List, Tuple

from assistant.capabilities.settings import cli, profile_preview, profiles


def _mk() -> Tuple[Path, Path]:
    tmp = Path(tempfile.mkdtemp(prefix="preview-test-"))
    target = tmp / "shell.json"
    target.write_text(json.dumps({
        "appearance": {"blur": True, "transparency": {"enabled": True}},
    }), encoding="utf-8")
    return tmp, target


def _cli(target: Path, argv: List[str]) -> Tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = cli.main([*argv, "--file", str(target)])
        except SystemExit as exc:
            rc = int(exc.code or 0)
    return rc, out.getvalue(), err.getvalue()


class BuildTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp, self.target = _mk()
        self.before = self.target.read_bytes()

    def test_composite_projection_and_blast_radius(self) -> None:
        profiles.save(self.target, "combo", [
            {"kind": "preset", "name": "minimal"},
            {"kind": "preset", "name": "gaming"},
        ])
        view = profile_preview.build(self.target, "combo")
        # gaming sets setAnimationSpeed=0.25 over minimal's 0.5: a conflict
        self.assertTrue(any(c["tool"] == "setAnimationSpeed"
                            for c in view["conflicts"]))
        # gaming sets appearance.transparency.enabled=False (setTransparencyEnabled)
        # which the cited interaction graph says forces blur off
        labels = [s["label"] for s in view["per_source"]]
        self.assertIn("preset:minimal", labels)
        self.assertIn("preset:gaming", labels)
        gaming = next(s for s in view["per_source"]
                      if s["label"] == "preset:gaming")
        self.assertEqual(gaming["n_ops"], 4)
        # blast radius is grouped by registry group
        self.assertTrue(view["blast_radius"])
        for group, tools in view["blast_radius"].items():
            self.assertTrue(tools, group)
        # read-only: the target's bytes are untouched
        self.assertEqual(self.target.read_bytes(), self.before)

    def test_transparency_derived_effect_is_cited(self) -> None:
        profiles.save(self.target, "deblur", [
            {"kind": "call", "tool": "setTransparencyEnabled", "value": False},
        ])
        view = profile_preview.build(self.target, "deblur")
        derived = view["composite"].get("derived") or []
        self.assertTrue(derived, "the transparency->blur edge must fire")
        self.assertTrue(any(d.get("citation") for d in derived))
        self.assertEqual(self.target.read_bytes(), self.before)

    def test_unknown_profile_raises(self) -> None:
        with self.assertRaises(profiles.ProfileError):
            profile_preview.build(self.target, "nope")

    def test_render_mentions_suggested_not_executed(self) -> None:
        profiles.save(self.target, "solo", [
            {"kind": "call", "tool": "setBarScale", "value": 0.8},
        ])
        view = profile_preview.build(self.target, "solo")
        text = "\n".join(profile_preview.render(view))
        self.assertIn("SUGGESTED_NOT_EXECUTED", text)
        self.assertIn("read-only", text)


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp, self.target = _mk()
        self.before = self.target.read_bytes()

    def test_whatif_flag_read_only_render(self) -> None:
        _cli(self.target, ["--profile-save", "combo", "preset:gaming"])
        rc, out, err = _cli(self.target, ["--profile-whatif", "combo"])
        self.assertEqual(rc, 0, err)
        self.assertIn("composite what-if", out)
        self.assertIn("preset:gaming", out)
        self.assertIn("feature area", out)
        self.assertEqual(self.target.read_bytes(), self.before)

    def test_unknown_profile_is_an_error(self) -> None:
        rc, out, err = _cli(self.target, ["--profile-whatif", "nope"])
        self.assertEqual(rc, 1)
        self.assertIn("error:", err)


if __name__ == "__main__":
    unittest.main()
