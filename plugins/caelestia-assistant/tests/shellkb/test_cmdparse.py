"""shellkb.cmdparse tests — token explanations, glob previews, flags."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from assistant.capabilities.shellkb import cmdparse  # noqa: E402


class TestTokenizer(unittest.TestCase):
    def test_quotes_and_operators(self):
        toks = cmdparse.tokenize("echo 'a b' \"c d\" > out.txt")
        kinds = [(t["kind"], t["text"]) for t in toks]
        self.assertIn(("quoted", "'a b'"), kinds)
        self.assertIn(("quoted", '"c d"'), kinds)
        self.assertIn(("redirect", ">"), kinds)
        self.assertIn(("word", "out.txt"), kinds)

    def test_separators_split(self):
        toks = cmdparse.tokenize("a && b | c; d")
        controls = [t["text"] for t in toks if t["kind"] == "control"]
        self.assertEqual(controls, ["&&", "|", ";"])

    def test_backslash_escape_is_one_token(self):
        toks = cmdparse.tokenize(r"caelestia shell hello\ world")
        words = [t["text"] for t in toks if t["kind"] in ("word", "quoted")]
        self.assertEqual(words[-1], r"hello\ world")


class TestExplain(unittest.TestCase):
    def test_verdict_is_always_not_executed(self):
        for line in ("caelestia screenshot -r", "rm -rf /",
                     "shutdown now", "garbage-unknown --flag"):
            data = cmdparse.explain_line(line)
            self.assertEqual(data["verdict"], "SUGGESTED_NOT_EXECUTED")

    def test_known_bin_subcommand_and_option(self):
        data = cmdparse.explain_line("caelestia screenshot -r")
        seg = data["segments"][0]
        roles = {w["token"]: w["role"] for w in seg["tokens"]}
        self.assertEqual(roles["caelestia"], "binary")
        self.assertEqual(roles["screenshot"], "subcommand")
        self.assertEqual(roles["-r"], "option")
        readings = {w["token"]: w["reading"] for w in seg["tokens"]}
        self.assertIn("--region", readings["-r"])
        self.assertEqual(seg["flags"], [])

    def test_unknown_option_is_flagged_not_guessed(self):
        data = cmdparse.explain_line("caelestia screenshot --regionx")
        seg = data["segments"][0]
        reading = [w["reading"] for w in seg["tokens"]
                   if w["token"] == "--regionx"][0]
        self.assertIn("NOT in the induced grammar", reading)

    def test_destructive_patterns_flag_with_reason(self):
        for line, why in (("rm -rf /tmp/x", "removes files"),
                          ("shutdown now", "stops or reboots"),
                          ("dd if=x of=/dev/sda", "raw device write"),
                          ("git push --force origin main",
                           "force-push")):
            data = cmdparse.explain_line(line)
            flags = data["segments"][0]["flags"]
            self.assertTrue(flags, f"{line} must be flagged")
            self.assertIn(why, flags[0]["reason"])

    def test_shell_kill_option_flagged(self):
        data = cmdparse.explain_line("caelestia shell -k")
        flags = data["segments"][0]["flags"]
        self.assertTrue(any("stops the running shell" == f["reason"]
                            for f in flags))
        data = cmdparse.explain_line("caelestia-shell-ipc quit")
        flags = data["segments"][0]["flags"]
        self.assertTrue(any("stops the shell" in f["reason"] for f in flags))

    def test_safe_caelestia_lines_have_no_flags(self):
        for line in ("caelestia screenshot -r",
                     "caelestia shell -s",
                     "caelestia-color scheme list --flat",
                     "caelestia-shell-ipc show"):
            data = cmdparse.explain_line(line)
            for seg in data["segments"]:
                self.assertEqual(seg.get("flags"), [], line)

    def test_multi_segment_pipeline(self):
        data = cmdparse.explain_line("caelestia shell -s && rm -rf x")
        self.assertEqual(data["n_segments"], 2)
        self.assertEqual(data["segments"][0]["segment_kind"], "command")
        self.assertEqual(data["segments"][1]["segment_kind"], "separator")
        self.assertEqual(data["segments"][1]["text"], "&&")
        self.assertEqual(data["segments"][0]["flags"], [])
        self.assertEqual(data["segments"][2]["flags"][0]["severity"],
                         "destructive")

    def test_env_assignment_prefix(self):
        data = cmdparse.explain_line("FOO=bar caelestia version")
        seg = data["segments"][0]
        self.assertEqual(seg["assignments"][0]["var"], "FOO")
        roles = {w["token"]: w["role"] for w in seg["tokens"]}
        self.assertEqual(roles["caelestia"], "binary")

    def test_glob_preview_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("alpha.log", "beta.log", "keep.txt"):
                Path(tmp, name).write_text("x")
            data = cmdparse.explain_line(
                "rm " + os.path.join(tmp, "*.log"))
            seg = data["segments"][0]
            globbed = [w for w in seg["tokens"]
                       if w.get("glob_preview")]
            self.assertEqual(len(globbed), 1)
            g = globbed[0]["glob_preview"]
            self.assertEqual(g["matches"], ["alpha.log", "beta.log"])
            self.assertEqual(g["n_matches"], 2)

    def test_quoted_glob_does_not_expand(self):
        data = cmdparse.explain_line("echo '*.txt'")
        seg = data["segments"][0]
        globbed = [w for w in seg["tokens"] if w.get("glob_preview")]
        self.assertEqual(len(globbed), 1)
        self.assertIn("quoted", globbed[0]["glob_preview"]["note"])

    def test_option_value_pairing(self):
        data = cmdparse.explain_line(
            "caelestia-color wallpaper -f /tmp/pic.png")
        seg = data["segments"][0]
        roles = {w["token"]: w["role"] for w in seg["tokens"]}
        self.assertEqual(roles["-f"], "option")
        self.assertEqual(roles["/tmp/pic.png"], "option-value")


if __name__ == "__main__":
    unittest.main()
