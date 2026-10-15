"""Tests for Layer 4 (issue drafting): gate behavior, template fidelity, dedup.

Offline and deterministic: the dedup-warning case injects a fabricated hits
list directly into the compose function (template.render_draft) so the real
retrieval index is never needed for it. All writes go to temporary dirs.

Note: captures use a duck-typed sink so the package imports stay stdlib-only
below are tiny hand-rolled file-like objects instead.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

from assistant.capabilities.diagnostics import schema_lint
from assistant.capabilities.issues import cli, template

BANNER_PREFIX = "DRAFT — NOT SUBMITTED TO ANYWHERE."


class _Capture:
    """Minimal write-capture standing in for stdout/stderr."""

    def __init__(self) -> None:
        self.chunks: list = []

    def write(self, text: str) -> int:
        self.chunks.append(text)
        return len(text)

    def flush(self) -> None:
        pass

    def getvalue(self) -> str:
        return "".join(self.chunks)


def run_main(argv: list, stdin_text: str = "") -> tuple:
    """Run cli.main with swapped stdout/stderr; returns (exit_code, stdout, stderr)."""
    out, err = _Capture(), _Capture()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    try:
        code = cli.main(argv, stdin_text=stdin_text)
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    return code, out.getvalue(), err.getvalue()


class TestPreviewGate(unittest.TestCase):
    """Without --confirm nothing may be written anywhere."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_home = Path(self._tmp.name) / "xdgdata"
        os.environ["XDG_DATA_HOME"] = str(self.data_home)
        self.addCleanup(os.environ.pop, "XDG_DATA_HOME", None)

    def test_no_confirm_means_no_file_and_preview_on_stdout(self) -> None:
        code, out, _err = run_main(
            ["draft", "--title", "Bar widgets disappear after update"],
            stdin_text="After the last update the bar widgets vanish.",
        )
        self.assertEqual(code, 0)
        self.assertIn(BANNER_PREFIX, out)
        self.assertIn("Bar widgets disappear after update", out)
        self.assertIn("preview only", out)
        self.assertFalse(self.data_home.exists(), "preview mode must not create any directories")

    def test_list_similar_writes_nothing_and_needs_no_confirm(self) -> None:
        code, out, _err = run_main(
            ["draft", "--title", "vesktop freezes when I screenshare", "--list-similar"],
            stdin_text="",
        )
        self.assertEqual(code, 0)
        self.assertIn("Similar existing material", out)
        self.assertFalse(self.data_home.exists(), "--list-similar must not write any file")


class TestConfirmGate(unittest.TestCase):
    """With --confirm the draft is written to a LOCAL file and starts with the banner."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_home = Path(self._tmp.name) / "xdgdata"
        os.environ["XDG_DATA_HOME"] = str(self.data_home)
        self.addCleanup(os.environ.pop, "XDG_DATA_HOME", None)

    def test_confirm_writes_complete_draft_to_out(self) -> None:
        out_path = Path(self._tmp.name) / "draft.md"
        code, out, _err = run_main(
            [
                "draft",
                "--title",
                "Bar widgets disappear after update",
                "--out",
                str(out_path),
                "--confirm",
            ],
            stdin_text="After updating Plasma the bar widgets disappear until the shell is restarted.",
        )
        self.assertEqual(code, 0)
        self.assertTrue(out_path.is_file())
        self.assertIn(str(out_path), out)
        self.assertIn("nothing was submitted", out.lower())
        text = out_path.read_text(encoding="utf-8")
        self.assertTrue(text.startswith(BANNER_PREFIX))
        for needle in (
            "Bar widgets disappear after update",
            "Issue type",
            "What happened?",
            "Before filing",
            "Environment",
            "SUGGESTED_NOT_EXECUTED",
            "journalctl --user -u caelestia-shell --no-pager -n 100",
            "plasmashell --version",
            "quickshell --version",
            "Generated:",
        ):
            self.assertIn(needle, text)
        self.assertIn("(run these yourself and paste output", text)
        # No "submitted" language anywhere except inside the banner's NOT SUBMITTED.
        remainder = re.sub(r"not\s+submitted", "", text, flags=re.IGNORECASE)
        self.assertNotIn("submitted", remainder.lower())

    def test_confirm_default_dir_respects_xdg_data_home(self) -> None:
        code, _out, _err = run_main(
            ["draft", "--title", "t", "--confirm"],
            stdin_text="d",
        )
        self.assertEqual(code, 0)
        drafts = list((self.data_home / "caelestia" / "assistant" / "drafts").glob("issue-draft-*.md"))
        self.assertEqual(len(drafts), 1)
        self.assertTrue(drafts[0].name.startswith("issue-draft-"))
        self.assertTrue(drafts[0].name.endswith(".md"))


class TestTemplateHandling(unittest.TestCase):
    def test_fallback_still_renders_complete_draft(self) -> None:
        original = template.parse_issue_template
        template.parse_issue_template = lambda path: ({}, [])  # monkeypoint: reader finds nothing
        try:
            text = template.render_draft(
                "bug", "T", "B", {}, [], None, "2025-01-01 00:00:00"
            )
        finally:
            template.parse_issue_template = original
        self.assertTrue(text.startswith(BANNER_PREFIX))
        for needle in (
            "Issue type",
            "Bug - something is broken or not working correctly",
            "Help wanted - I need assistance with configuration, setup, or usage",
            "What happened?",
            "B",
            "Steps to reproduce",
            "Caelestia version",
            "Distro",
            "Logs or screenshots",
            "Before filing",
            "Environment",
        ):
            self.assertIn(needle, text)

    def test_feature_template_uses_feature_sections(self) -> None:
        text = template.render_draft("feature", "Add X", "Please", {}, [], None, "t")
        self.assertIn("What's the idea?", text)
        self.assertIn("Why would it be useful?", text)
        self.assertNotIn("Issue type", text)  # feature template has no checkboxes section

    def test_rendered_bug_template_mirrors_real_sections_in_order(self) -> None:
        text = template.render_draft("bug", "T", "the body", {}, [], None, "t")
        positions = [text.index(needle) for needle in (
            "Issue type",
            "What happened?",
            "Steps to reproduce",
            "Caelestia version",
            "Distro",
            "Logs or screenshots",
        )]
        self.assertEqual(positions, sorted(positions), "sections must mirror the yml order")
        self.assertIn("the body", text)
        self.assertIn("[x] Bug - something is broken or not working correctly", text)
        self.assertIn("[ ] Help wanted - I need assistance with configuration, setup, or usage", text)


class TestDedupAdvisory(unittest.TestCase):
    """The dedup warning is tested with an injected (fabricated) hits list."""

    BASE = dict(issue_type="bug", title="T", body="B", env_block={}, commit_info=None, generated_at="t")

    def test_strong_top_hit_gains_duplicate_warning(self) -> None:
        strong = [{
            "doc_id": "ISS-528",
            "title": "Quickshell broken after update",
            "score": 12.5,
            "source": "research/corpus-html/issue_528.html",
            "snippet": "reinstall quickshell-git",
        }]
        text = template.render_draft(similar=strong, **self.BASE)
        self.assertIn("possible duplicate — check ISS-528 first", text)
        self.assertIn("ISS-528", text)
        self.assertIn("score 12.5", text)

    def test_weak_top_hit_has_no_duplicate_warning(self) -> None:
        weak = [{
            "doc_id": "ISS-641",
            "title": "Game fullscreen problem",
            "score": 3.0,
            "source": "research/corpus-html/issue_641.html",
            "snippet": "dodge-windows toggle",
        }]
        text = template.render_draft(similar=weak, **self.BASE)
        self.assertNotIn("possible duplicate", text)
        self.assertIn("ISS-641", text)  # still listed in the similar section, advisory only

    def test_no_hits_renders_honest_empty_section(self) -> None:
        text = template.render_draft(similar=[], **self.BASE)
        self.assertNotIn("possible duplicate", text)
        self.assertIn("no similar material found", text)


class TestCliErrors(unittest.TestCase):
    def test_unknown_arg_is_a_clean_argparse_error(self) -> None:
        err = _Capture()
        old_err = sys.stderr
        sys.stderr = err
        try:
            with self.assertRaises(SystemExit) as ctx:
                cli.main(["draft", "--title", "t", "--definitely-not-a-flag"], stdin_text="")
        finally:
            sys.stderr = old_err
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("unrecognized arguments", err.getvalue())

    def test_empty_description_is_a_clean_error(self) -> None:
        code, _out, err = run_main(["draft", "--title", "t"], stdin_text="   ")
        self.assertEqual(code, 2)
        self.assertIn("--from-file", err)


if __name__ == "__main__":
    unittest.main()
