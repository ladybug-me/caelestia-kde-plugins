"""devflow tests — the drafting tools and the partition guarantees.

The last test class is the §7 hard requirement, made structural: the
README carries the #120/upstream disclaimer, and NOTHING in the #120-bound
surfaces (assistant/settings/, assistant/cortex/, shell/) references
this package. Every drafting function is pinned to a solved example.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from assistant.capabilities.devflow.diffstat import (commit_message, dominant_type,
                                        parse_numstat, pr_skeleton)
from assistant.capabilities.devflow.todo import render, triage_tree

NUMSTAT = "\n".join([
    "120\t4\ttests/test_simhash.py",
    "180\t2\ttests/test_scan.py",
    "30\t0\ttests/test_novelty.py",
    "6\t1\tREADME.md",
])


class ParseTests(unittest.TestCase):

    def test_parse_numstat(self):
        rows = parse_numstat(NUMSTAT)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["path"], "tests/test_simhash.py")
        self.assertEqual(rows[0]["added"], 120)
        self.assertEqual(rows[0]["deleted"], 4)

    def test_binary_and_rename_rows(self):
        rows = parse_numstat("-\t-\tassets/logo.png\n"
                            "1\t1\tsrc/{old => new}/thing.py")
        self.assertTrue(rows[0]["binary"])
        self.assertEqual(rows[1]["renamed_from"], "src/old/thing.py")

    def test_empty_and_garbage_lines_are_skipped(self):
        self.assertEqual(parse_numstat(""), [])
        self.assertEqual(parse_numstat("not a numstat line\n\n"), [])


class ClassificationTests(unittest.TestCase):

    def test_classification_table(self):
        cases = [
            ({"path": "tests/test_x.py", "added": 3, "deleted": 1}, "test"),
            ({"path": "assistant/settings/tests/test_a.py", "added": 3,
              "deleted": 1}, "test"),
            ({"path": "docs/guide.md", "added": 10, "deleted": 0}, "docs"),
            ({"path": ".github/workflows/ci.yml", "added": 1, "deleted": 1}, "ci"),
            ({"path": "src/new_thing.py", "added": 40, "deleted": 0}, "feat"),
            ({"path": "src/dead_code.py", "added": 2, "deleted": 90}, "refactor"),
            ({"path": "src/misc.py", "added": 5, "deleted": 5}, "chore"),
        ]
        for row, expected in cases:
            self.assertEqual(dominant_type([row]), expected, row["path"])

    def test_dominant_type_is_deterministic_under_reordering(self):
        rows = parse_numstat(NUMSTAT)
        self.assertEqual(dominant_type(rows), dominant_type(list(reversed(rows))))


class CommitMessageTests(unittest.TestCase):

    def test_skeleton_shape(self):
        message = commit_message(NUMSTAT)
        lines = message.splitlines()
        self.assertTrue(lines[0].startswith("test(tests):"),
                        f"classification+scope header expected: {lines[0]!r}")
        self.assertLessEqual(len(lines[0]), 72)
        self.assertIn("<describe the what and why>", lines[0])
        self.assertIn("[4 file(s), +336/-7]", lines[0])
        self.assertIn("<why this change", message)
        # real file paths appear verbatim
        self.assertIn("tests/test_simhash.py", message)

    def test_empty_input_is_honest(self):
        self.assertIn("no changes", commit_message(""))

    def test_deterministic(self):
        self.assertEqual(commit_message(NUMSTAT), commit_message(NUMSTAT))


class PrSkeletonTests(unittest.TestCase):

    def test_skeleton_follows_the_upstream_template_shape(self):
        skeleton = pr_skeleton(NUMSTAT, base="dev", head="feat/x")
        for section in ("## What does this change?", "## How did you test it?",
                        "## Type", "## Notes for reviewers"):
            self.assertIn(section, skeleton)
        self.assertIn("dev...feat/x", skeleton)
        self.assertIn("## Changes by area", skeleton)
        # the type checkbox is pre-filled from the classification AND
        # tells the human to correct it
        self.assertIn("uncheck and correct if wrong", skeleton)

    def test_deterministic(self):
        self.assertEqual(pr_skeleton(NUMSTAT), pr_skeleton(NUMSTAT))


class TodoTriageTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        (self.root / "src").mkdir()
        (self.root / "src" / "app.py").write_text(
            "def f():\n"
            "    # TODO: implement me\n"
            "    pass\n"
            "    # FIXME broken on wayland\n"
            "    return 1  # XXX hot path\n",
            encoding="utf-8")
        (self.root / "ui").mkdir()
        (self.root / "ui" / "panel.qml").write_text(
            "// HACK: hardcoded height\n"
            "Item { height: 42 }\n",
            encoding="utf-8")
        (self.root / ".git").mkdir()
        (self.root / ".git" / "config").write_text(
            "# TODO inside a skipped directory\n", encoding="utf-8")

    def test_triage_finds_markers_with_lines(self):
        report = triage_tree(str(self.root))
        by_position = {(i["file"], i["line"]): i for i in report["items"]}
        app_todo = by_position[("src/app.py", 2)]
        self.assertEqual(app_todo["marker"], "TODO")
        self.assertEqual(app_todo["text"], "implement me")
        fixme = by_position[("src/app.py", 4)]
        self.assertEqual(fixme["marker"], "FIXME")
        self.assertIn("broken on wayland", fixme["text"])
        self.assertEqual(by_position[("ui/panel.qml", 1)]["marker"], "HACK")

    def test_skipped_directories_are_not_scanned(self):
        report = triage_tree(str(self.root))
        self.assertTrue(all(not i["file"].startswith(".git")
                            for i in report["items"]))

    def test_render_groups_by_marker(self):
        text = render(triage_tree(str(self.root)))
        self.assertIn("TODO", text)
        self.assertIn("src/app.py:2", text)
        self.assertIn("HACK (1):", text)

    def test_single_file_input(self):
        report = triage_tree(str(self.root / "src" / "app.py"))
        self.assertEqual(len(report["items"]), 3)

    def test_empty_tree_is_honest(self):
        empty = self.root / "empty"
        empty.mkdir()
        text = render(triage_tree(str(empty)))
        self.assertIn("no TODO/FIXME markers", text)


class PartitionTests(unittest.TestCase):
    """§7's hard requirement, checked structurally."""

    def test_package_docstring_carries_the_disclaimer(self):
        import inspect

        from assistant.capabilities import devflow

        text = inspect.getdoc(devflow) or ""
        self.assertIn("#120", text)
        self.assertIn("never be included in any upstream-bound pull request",
                      text)
        self.assertIn("KDE shell", text)

    def test_no_120_bound_surface_references_devflow(self):
        repo_root = Path(__file__).resolve().parents[2]
        checked = 0
        for pattern in ("assistant/capabilities/settings/**/*.py", "assistant/core/**/*.py",
                        "assistant/adapters/caelestia/shell/**/*.qml",
                        "assistant/adapters/caelestia/shell/**/*.py"):
            for path in repo_root.glob(pattern):
                checked += 1
                self.assertNotIn(
                    "devflow", path.read_text(encoding="utf-8", errors="replace"),
                    f"{path} must not reference the devflow package "
                    "(the §7 partition is structural)")
        self.assertGreater(checked, 50,
                           "the partition check must actually cover the "
                           "#120-bound surfaces")

    def test_devflow_does_not_import_shell_facing_layers(self):
        repo_root = Path(__file__).resolve().parents[2]
        for path in (repo_root / "assistant" / "devflow").glob("*.py"):
            source = path.read_text(encoding="utf-8")
            for banned in ("from ..settings", "from ..cortex",
                           "import ..settings", "from ..genius.meta"):
                self.assertNotIn(banned, source,
                                 f"{path.name} must stay partitioned "
                                 f"(found {banned!r})")


if __name__ == "__main__":
    unittest.main()
