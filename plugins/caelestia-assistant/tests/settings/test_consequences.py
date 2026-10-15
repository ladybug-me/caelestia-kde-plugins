"""Tests for the what-if consequence-graph mode (phase 2.7).

Four independent guards (the proposal's own verification plan):

- a CITATION drift guard: every edge's cited file:line is re-verified
  against the caelestia-kde checkout — an edge whose cited line no
  longer matches its ``claimed_content`` is a test failure, never a
  silent stale edge (the registry's citation-guard pattern, extended
  to the edge table). Skips loudly when the shell/ tree is absent;
- a PROJECTION guard: derived effects fire exactly when their ``when``
  conditions hold, live-state ``requires`` conditions are evaluated
  honestly, output is bounded on adversarial op lists;
- a CONFLICT guard: the AC-3 view (candidate values against registry
  domains) and the INDUCED-value extension (an edge that forces a
  value the user contradicted is reported, cited);
- a SURFACE guard: the read-only contract — ``settings --what-if``
  never writes, and the rendering is the pre-consent artifact.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Optional

from assistant.capabilities.settings.consequences import EDGES, project, render


def _find_shell_root(start: Path) -> Optional[Path]:
    # the vendored caelestia QML lives inside the adapter since the
    # restructure; fall back to a sibling upstream checkout
    for cand in [start, *start.parents]:
        inrepo = cand / "assistant" / "adapters" / "caelestia" / "shell"
        if (inrepo / "modules" / "bar" / "BarWrapper.qml").is_file():
            return inrepo
        if (cand / "shell" / "modules" / "bar" / "BarWrapper.qml").is_file():
            return cand
    return None


SHELL_ROOT = _find_shell_root(Path(__file__).resolve().parent)


@unittest.skipIf(SHELL_ROOT is None,
                 "shell/ tree absent — edge citation guards skip loudly")
class EdgeCitationGuardTests(unittest.TestCase):
    """The honesty rule: every edge's cited line still says what the
    edge claims it says (stale edges are failures, not surprises)."""

    def _cited_lines(self, citation: str) -> str:
        rel, _, span = citation.rpartition(":")
        start, _, end = span.partition("-")
        path = SHELL_ROOT / rel  # type: ignore[operator]
        lines = path.read_text(encoding="utf-8",
                               errors="replace").splitlines()
        lo, hi = int(start), int(end or start)
        return "\n".join(lines[lo - 1:hi])

    def test_every_edge_cites_a_resolving_file(self) -> None:
        for edge in EDGES:
            rel = edge["citation"].rpartition(":")[0]
            self.assertTrue(
                (SHELL_ROOT / rel).is_file(),  # type: ignore[operator]
                f"edge {edge['id']}: cited file {rel} does not resolve")

    def test_every_edge_citation_matches_claimed_content(self) -> None:
        for edge in EDGES:
            cited = self._cited_lines(edge["citation"])
            # whitespace-insensitive compare (QML wraps differently in
            # different releases; the SEMANTIC content is the claim)
            def _squash(s: str) -> str:
                return "".join(s.split())
            self.assertIn(
                _squash(edge["claimed_content"]), _squash(cited),
                f"edge {edge['id']}: cited {edge['citation']} no longer "
                f"matches the claimed content")

    def test_edge_table_shape(self) -> None:
        ids = set()
        for edge in EDGES:
            for key in ("id", "trigger_path", "when", "effect_path",
                        "effect", "citation", "claimed_content",
                        "confidence"):
                self.assertIn(key, edge, f"{edge['id']}: missing {key}")
            self.assertNotIn(edge["id"], ids, "duplicate edge id")
            ids.add(edge["id"])
            self.assertIn(edge["confidence"], ("high", "medium", "low"))
            # trigger paths must be real registry paths
            from assistant.adapters.caelestia.registry import tool_by_path
            self.assertIsNotNone(
                tool_by_path(edge["trigger_path"]),
                f"{edge['id']}: trigger {edge['trigger_path']} is not a "
                "registry path — edges describe shell.json keys")

    def test_table_is_bounded_and_reviewable(self) -> None:
        # the proposal's own footprint target: <= 30 edges
        self.assertLessEqual(len(EDGES), 30)


class ProjectionTests(unittest.TestCase):
    def _ops(self, *pairs) -> list:
        return [{"tool": t, "value": v} for t, v in pairs]

    def test_transparency_off_derives_blur_off(self) -> None:
        p = project(self._ops(("setTransparencyEnabled", False)),
                    current={"appearance": {"blur": True}})
        ids = [d["edge"] for d in p["derived"]]
        self.assertIn("transparency-off-forces-blur-off", ids)
        row = next(d for d in p["derived"]
                   if d["edge"] == "transparency-off-forces-blur-off")
        self.assertEqual(row["citation"],
                         "shell/modules/nexus/pages/wallandstyle/"
                         "AppearancePage.qml:132-137")
        self.assertEqual(row["confidence"], "high")

    def test_transparency_on_fires_nothing(self) -> None:
        p = project(self._ops(("setTransparencyEnabled", True)))
        self.assertEqual(p["derived"], [])

    def test_bar_scale_below_floor_fires_clamp_edge(self) -> None:
        p = project(self._ops(("setBarScale", 0.5)))
        ids = [d["edge"] for d in p["derived"]]
        self.assertIn("bar-scale-floor", ids)

    def test_bar_scale_above_floor_fires_nothing(self) -> None:
        p = project(self._ops(("setBarScale", 0.9)))
        self.assertEqual(p["derived"], [])

    def test_requires_condition_uses_live_state(self) -> None:
        # dodge with persistent=False in the live file -> INERT note
        p = project(self._ops(("setDodgeWindows", True)),
                    current={"bar": {"persistent": False}})
        ids = [d["edge"] for d in p["derived"]]
        self.assertIn("dodge-needs-persistent", ids)
        # same op with persistent=True -> the edge stays silent
        p2 = project(self._ops(("setDodgeWindows", True)),
                     current={"bar": {"persistent": True}})
        self.assertNotIn("dodge-needs-persistent",
                         [d["edge"] for d in p2["derived"]])

    def test_requires_condition_with_no_live_state_stays_silent(self) -> None:
        p = project(self._ops(("setDodgeWindows", True)))  # current=None
        self.assertEqual(p["derived"], [])

    def test_ops_annotated_with_downstream_counts(self) -> None:
        p = project(self._ops(("setTransparencyEnabled", False)))
        row = next(o for o in p["annotated_ops"]
                   if o["tool"] == "setTransparencyEnabled")
        self.assertEqual(row["downstream"], len(p["derived"]))

    def test_projection_is_bounded_on_adversarial_lists(self) -> None:
        # dense trigger list: every op fires an edge; output stays bounded
        ops = self._ops(("setTransparencyEnabled", False),
                        ("setBarScale", 0.5),
                        ("setBorderThickness", 1.0),
                        ("setDodgeWindows", True))
        p = project(ops, current={"bar": {"persistent": False}},
                    max_derived=3)
        self.assertLessEqual(len(p["derived"]), 3)

    def test_reversibility_stated_plainly(self) -> None:
        p = project(self._ops(("setBarScale", 0.8)))
        self.assertTrue(p["reversibility"]["all_journaled"])
        self.assertIn("undo", p["reversibility"]["note"])

    def test_effect_value_feeds_the_conflict_view_only_when_present(self) -> None:
        with_value = next(e for e in EDGES if "effect_value" in e)
        without = [e for e in EDGES if "effect_value" not in e]
        self.assertEqual(with_value["id"],
                         "transparency-off-forces-blur-off")
        self.assertTrue(without, "the INERT/CLAMP edges carry no value")


class ConflictViewTests(unittest.TestCase):
    def _ops(self, *pairs) -> list:
        return [{"tool": t, "value": v} for t, v in pairs]

    def test_out_of_domain_request_is_reported(self) -> None:
        p = project(self._ops(("setBarScale", 9.9)))
        self.assertEqual([c["verdict"] for c in p["conflicts"]],
                         ["UNSUPPORTED"])
        self.assertIn("no support", p["conflicts"][0]["reason"])

    def test_valid_ops_report_no_conflict(self) -> None:
        p = project(self._ops(("setBarScale", 0.9)))
        self.assertEqual(p["conflicts"], [])

    def test_induced_conflict_when_user_contradicts_the_edge(self) -> None:
        # transparency off + blur on: the shell's own handler flips blur
        # off — the requested blur value will not hold
        p = project(self._ops(("setTransparencyEnabled", False),
                              ("setBlurEnabled", True)))
        verdicts = [c["verdict"] for c in p["conflicts"]]
        self.assertIn("INDUCED-CONFLICT", verdicts)
        row = next(c for c in p["conflicts"]
                   if c["verdict"] == "INDUCED-CONFLICT")
        self.assertIn("AppearancePage.qml", row["cited"])
        self.assertIn("will not hold", row["reason"])

    def test_no_induced_conflict_when_user_agrees_with_the_edge(self) -> None:
        p = project(self._ops(("setTransparencyEnabled", False),
                              ("setBlurEnabled", False)))
        self.assertEqual(p["conflicts"], [])

    def test_conflict_check_never_breaks_the_view(self) -> None:
        # garbage ops the registry does not know: the view still renders
        p = project([{"tool": "no-such-tool", "value": "x"}])
        self.assertEqual(p["conflicts"], [])
        self.assertEqual(p["n_ops"], 1)


class RenderTests(unittest.TestCase):
    def test_rendering_is_the_pre_consent_artifact(self) -> None:
        p = project([{"tool": "setTransparencyEnabled", "value": False}])
        text = "\n".join(render(p))
        self.assertIn("read-only view", text)
        self.assertIn("nothing is applied", text)
        self.assertIn("derived effects", text)
        self.assertIn("cited:", text)
        self.assertIn("reversibility:", text)
        self.assertIn(f"{len(EDGES)} edges", text)

    def test_rendering_with_no_derived_effects_is_honest(self) -> None:
        p = project([{"tool": "setBarScale", "value": 0.9}])
        text = "\n".join(render(p))
        self.assertIn("derived effects: none", text)
        self.assertIn("no conflict", text)


class WhatIfSurfaceTests(unittest.TestCase):
    """The read-only contract of `settings --what-if` (never writes)."""

    def _run(self, argv: list, workdir: Path) -> int:
        from assistant.capabilities.settings import cli as settings_cli
        import contextlib
        import io
        out = io.StringIO()
        with unittest.mock.patch.dict(
                os.environ, {"XDG_CONFIG_HOME": str(workdir)}):
            with contextlib.redirect_stdout(out), \
                    contextlib.redirect_stderr(io.StringIO()):
                try:
                    rc = settings_cli.main(argv)
                except SystemExit as exc:
                    rc = int(exc.code or 0)
        self.stdout = out.getvalue()
        return rc

    def setUp(self) -> None:
        import unittest.mock
        self.mock = unittest.mock
        self.stdout = ""

    def test_what_if_presets_project_and_write_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "caelestia" / "shell.json"
            rc = self._run(["--what-if", "gaming", "--file", str(target)],
                           Path(tmp))
            self.assertEqual(rc, 0)
            self.assertIn("what-if", self.stdout)
            # the gaming preset flips transparency off -> the cited edge
            # fires and the view SHOWS it (derived effect + citation)
            self.assertIn("appearance.blur", self.stdout)
            self.assertIn("cited:", self.stdout)
            # the read-only contract: the target was never created
            self.assertFalse(target.exists())

    def test_what_if_resolves_step_ops_to_absolute_values(self) -> None:
        # a STEP phrase must project the RESOLVED value (0.9), never the
        # raw delta (-1) — the bug this pins fired the 0.6-floor edge on
        # "make the bar smaller"
        with tempfile.TemporaryDirectory() as tmp:
            rc = self._run(["--what-if", "make the bar smaller"],
                           Path(tmp))
            self.assertEqual(rc, 0)
            self.assertIn("candidate ops: 1", self.stdout)
            self.assertNotIn("CLAMPED at 0.6", self.stdout)

    def test_what_if_unplannable_request_fails_honestly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_rc = self._run(["--what-if", "utter nonsense words"],
                               Path(tmp))
            self.assertNotEqual(out_rc, 0)


if __name__ == "__main__":
    unittest.main()
