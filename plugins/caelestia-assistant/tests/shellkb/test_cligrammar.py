"""shellkb.cligrammar tests — hermetic (no upstream checkout needed).

The committed artifact is the runtime's world; these tests pin its
structure and the miner's behavior on EMBEDDED fixture text (mini
usage heredocs + case bodies in the upstream's own style), so a
contributor without a caelestia-kde checkout can run everything.
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from assistant.capabilities.shellkb import cligrammar  # noqa: E402


class TestCommittedArtifact(unittest.TestCase):
    def test_artifact_loads_and_has_bins(self):
        data = cligrammar.load_grammar()
        self.assertIn("bins", data)
        self.assertGreaterEqual(len(data["bins"]), 6,
                                "the pinned upstream ships >= 6 CLIs")
        self.assertIn("generator", data)
        self.assertEqual(data["generator"],
                         "assistant.capabilities.shellkb.cligrammar.mine")

    def test_artifact_records_upstream_pin(self):
        data = cligrammar.load_grammar()
        up = data.get("upstream", {})
        self.assertIn("checkout", up)
        self.assertTrue(up.get("commit"), "the pin must cite a commit")

    def test_caelestia_top_level_surface(self):
        c = cligrammar.load_grammar()["bins"]["caelestia"]
        for sub in ("shell", "install", "update", "wallpaper", "scheme",
                    "screenshot", "record", "version", "help"):
            self.assertIn(sub, c["subcommands"], f"missing sub {sub}")
        self.assertEqual(c["synopsis"], "caelestia <command> [args...]")

    def test_shell_options_typed_flags(self):
        shell = cligrammar.load_grammar()["bins"]["caelestia"] \
            ["subcommands"]["shell"]
        by_long = {o["long"]: o for o in shell["options"]}
        self.assertFalse(by_long["--daemon"]["takes_value"])
        self.assertFalse(by_long["--kill"]["takes_value"])
        self.assertEqual(by_long["--daemon"]["short"], "-d")

    def test_color_value_taking_options(self):
        color = cligrammar.load_grammar()["bins"]["caelestia-color"]
        by_long = {o["long"]: o
                   for o in color["subcommands"]["wallpaper"]["options"]}
        self.assertTrue(by_long["--file"]["takes_value"],
                        "-f PATH takes a value per the case branch")
        self.assertFalse(by_long["--no-smart"]["takes_value"])
        # nested sub-subcommands are induced too
        self.assertIn("scheme list", color["subcommands"])
        self.assertIn("scheme set", color["subcommands"])

    def test_every_example_is_labeled_not_executed(self):
        for line in cligrammar.example_lines():
            self.assertTrue(line.startswith("SUGGESTED_NOT_EXECUTED:"),
                            f"unlabeled example: {line}")

    def test_render_lines_mentions_no_execution(self):
        lines = cligrammar.render_lines()
        self.assertTrue(lines[0].startswith("shell CLI grammar"))
        self.assertIn("Nothing is ever executed", lines[0])


class TestMinerOnFixtureText(unittest.TestCase):
    """The miner against embedded mini-scripts (upstream's own style)."""

    def _write(self, tmp, body):
        path = Path(tmp) / "caelestia-fixture"
        path.write_text(body)
        return path

    def test_heredoc_and_case_merge(self):
        import tempfile
        fixture = r'''#!/usr/bin/env bash
usage() {
    cat <<'EOF'
caelestia-fixture - test bin

Usage: caelestia-fixture <command> [args...]

Commands
  shell [OPTS] [MESSAGE...]    start the shell
  paint [OPTS]                 paint things

Shell options
  -d, --daemon    start detached
  -l, --log       print the log

Paint options
  -f, --file PATH    the file to paint
EOF
}

shell_usage() {
    cat <<'EOF'
Usage: caelestia-fixture shell [OPTS] [MESSAGE...]

Options
  -d, --daemon    start detached
EOF
}

cmd_shell() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -d|--daemon)
                shift
                ;;
            -h|--help)
                shell_usage
                return 0
                ;;
            *)
                shift
                ;;
        esac
    done
}

cmd_paint() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -f|--file) file="${2:-}"; shift 2 ;;
        esac
    done
}

main() {
    case "$1" in
        shell) cmd_shell "$@" ;;
        paint) cmd_paint "$@" ;;
    esac
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            g = cligrammar.parse_bin(self._write(tmp, fixture))
        self.assertEqual(g["synopsis"],
                         "caelestia-fixture <command> [args...]")
        subs = g["subcommands"]
        # Commands-table rows cross-checked against dispatch/branches
        self.assertIn("shell", subs)
        self.assertIn("paint", subs)
        # section retargeting: 'Shell options' rows belong to shell
        shell_opts = {o["long"] for o in subs["shell"]["options"]}
        self.assertIn("--daemon", shell_opts)
        self.assertIn("--log", shell_opts)
        # 'Paint options' rows belong to paint, with takes_value typed
        paint_opts = {o["long"]: o for o in subs["paint"]["options"]}
        self.assertTrue(paint_opts["--file"]["takes_value"])
        # case-branch enforcement merged onto the advertised rows
        self.assertFalse(
            [o for o in subs["shell"]["options"]
             if o["long"] == "--daemon"][0]["takes_value"])

    def test_one_line_usage_with_bracket_groups(self):
        import tempfile
        fixture = r'''#!/usr/bin/env bash
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            echo "Usage: caelestia-fixture [start] [-s|--sound] [--gif]" >&2
            exit 0
            ;;
        start)
            ACTION=start
            ;;
        -s|--sound)
            SOUND=1
            ;;
        --gif)
            GIF=1
            ;;
    esac
    shift
done
'''
        with tempfile.TemporaryDirectory() as tmp:
            g = cligrammar.parse_bin(self._write(tmp, fixture))
        self.assertEqual(
            g["synopsis"],
            "caelestia-fixture [start] [-s|--sound] [--gif]")
        longs = {o["long"] for o in g["options"]}
        self.assertIn("--sound", longs)
        self.assertIn("--gif", longs)
        # 'start' has no Commands row here, so it stays a positional
        # documented by the synopsis — never invented into a subcommand
        self.assertNotIn("start", g["subcommands"])

    def test_miner_without_upstream_is_honest(self):
        # find_upstream in this container may or may not see a checkout;
        # mine() must degrade to an honest error either way
        data = cligrammar.mine(Path("/nonexistent/root"))
        self.assertIn("error", data)

    def test_artifact_is_sorted_deterministic_json(self):
        raw = cligrammar.GRAMMAR_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
        # round-trip: sorted-keys re-dump is byte-identical
        redump = json.dumps(data, indent=1, sort_keys=True,
                            ensure_ascii=False) + "\n"
        self.assertEqual(raw, redump,
                         "the committed artifact must stay canonical")


if __name__ == "__main__":
    unittest.main()
