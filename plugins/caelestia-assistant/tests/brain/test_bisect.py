"""F13 config-bisect tests: Bayesian probe selection, ddmin minimality,
noisy-answer robustness, bounded budgets, registry-resolved revert
proposals, state round-trips, and the CLI loop (temp state + temp config
root; the module must never write to the config tree)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from assistant.capabilities.brain import bisect as bs
from assistant.capabilities.brain import cli as brain_cli


def _write_config(root: Path, shell: dict, extra: dict | None = None) -> None:
    (root / "shell.json").write_text(json.dumps(shell), encoding="utf-8")
    if extra is not None:
        (root / "other.json").write_text(json.dumps(extra), encoding="utf-8")


class LeafMapTests(unittest.TestCase):
    def test_flatten_and_changed_keys(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_config(root, {"bar": {"scale": 1.0, "pos": "top"}},
                          extra={"misc": {"flag": True}})
            good = bs.leaf_map(str(root))
            self.assertIn("shell.json::bar.scale", good)
            self.assertEqual(good["shell.json::bar.pos"], "top")
            self.assertEqual(good["other.json::misc.flag"], True)
            _write_config(root, {"bar": {"scale": 1.4, "pos": "top",
                                         "new": 2}},
                          extra={})
            bad = bs.leaf_map(str(root))
            changed = bs.changed_keys(good, bad)
            self.assertEqual(changed, [
                "other.json::misc.flag",        # removed
                "shell.json::bar.new",          # added
                "shell.json::bar.scale",        # changed
            ])


class EngineTests(unittest.TestCase):
    KEYS = [f"k{i}" for i in range(1, 9)]

    def _drive(self, eng: bs.NoisyBisect, probe) -> None:
        for _ in range(bs.MAX_PROBES + 2):
            if eng.phase == "done":
                return
            subset = eng.next_subset()
            eng.observe(probe(subset))

    def test_single_culprit_found(self):
        eng = bs.NoisyBisect(self.KEYS)
        self._drive(eng, lambda s: "k5" in set(s))
        self.assertEqual(eng.phase, "done")
        self.assertEqual(eng.minimal, ["k5"])
        self.assertLess(eng.probe_count, 20)

    def test_ddmin_pair_culprit_minimality(self):
        eng = bs.NoisyBisect(self.KEYS)
        self._drive(eng, lambda s: {"k2", "k6"} <= set(s))
        self.assertEqual(eng.phase, "done")
        self.assertEqual(sorted(eng.minimal), ["k2", "k6"])

    def test_one_wrong_answer_still_converges(self):
        # deterministic noise: the very first probe answer is flipped
        eng = bs.NoisyBisect(self.KEYS)
        calls = {"n": 0}

        def probe(subset):
            calls["n"] += 1
            truth = "k5" in set(subset)
            return (not truth) if calls["n"] == 1 else truth

        self._drive(eng, probe)
        self.assertEqual(eng.phase, "done")
        self.assertEqual(eng.minimal, ["k5"])

    def test_probe_budget_is_honored(self):
        eng = bs.NoisyBisect([f"k{i}" for i in range(1, 13)])
        with mock.patch.object(bs, "MAX_PROBES", 4):
            for _ in range(10):
                if eng.phase in ("done", "inconclusive"):
                    break
                subset = eng.next_subset()
                eng.observe(False)      # user always answers "good"
            self.assertEqual(eng.phase, "inconclusive")
            with self.assertRaises(bs.BisectError):
                eng.proposal()

    def test_state_roundtrip_resumes(self):
        eng = bs.NoisyBisect(self.KEYS)
        for _ in range(2):
            subset = eng.next_subset()
            eng.observe("k5" in set(subset))
        restored = bs.NoisyBisect.from_state(eng.to_state())
        self.assertEqual(restored.ranking(), eng.ranking())
        self.assertEqual(restored.phase, eng.phase)
        self.assertEqual(restored.probe_count, eng.probe_count)
        # finish on the restored engine
        for _ in range(bs.MAX_PROBES + 2):
            if restored.phase == "done":
                break
            subset = restored.next_subset()
            restored.observe("k5" in set(subset))
        self.assertEqual(restored.minimal, ["k5"])

    def test_empty_keys_rejected(self):
        with self.assertRaises(bs.BisectError):
            bs.NoisyBisect([])


class RevertOpsTests(unittest.TestCase):
    def test_registry_resolution_and_manual_review(self):
        good = {"shell.json::bar.scale": 1.0,
                "shell.json::not.a.registry.leaf": 7,
                "other.json::bar.scale": 5}
        rev = bs.revert_ops(good, ["shell.json::bar.scale",
                                   "shell.json::not.a.registry.leaf",
                                   "other.json::bar.scale"])
        self.assertEqual(rev["ops"], [{
            "tool": "setBarScale", "action": "set", "value": 1.0,
            "raw": "setBarScale=1.0",
            "note": "revert to good value (was changed in shell.json)",
        }])
        reasons = {m["key"]: m["reason"] for m in rev["manual"]}
        self.assertIn("no registry tool owns this leaf",
                      reasons["shell.json::not.a.registry.leaf"])
        self.assertIn("outside the managed shell.json",
                      reasons["other.json::bar.scale"])

    def test_key_absent_in_good_goes_manual(self):
        rev = bs.revert_ops({"shell.json::a": 1}, ["shell.json::ghost"])
        self.assertEqual(rev["ops"], [])
        self.assertEqual(len(rev["manual"]), 1)


class CliTests(unittest.TestCase):
    def test_cli_roundtrip_and_no_config_writes(self):
        with tempfile.TemporaryDirectory() as td:
            import io
            root = Path(td) / "config"
            root.mkdir()
            good_shell = {"bar": {"scale": 1.0}}
            _write_config(root, good_shell)
            state = Path(td) / "brain.json"
            out = io.StringIO()

            def run(argv):
                return brain_cli.main(["--state", str(state), "bisect"]
                                      + argv, out)

            self.assertEqual(run(["mark", "good", "--root", str(root)]), 0)
            _write_config(root, {"bar": {"scale": 1.4}})
            self.assertEqual(run(["mark", "bad", "--root", str(root)]), 0)
            self.assertIn("changed keys between marks: 1",
                          out.getvalue())
            self.assertEqual(run(["next"]), 0)
            self.assertIn("bar.scale", out.getvalue())
            # drive next/observe until the engine is done: the probe is
            # answered "bad" (the single changed key does reproduce), and
            # ddmin's confirmation re-asks it once before finishing
            for _ in range(8):
                self.assertEqual(run(["next"]), 0)
                if "search finished" in out.getvalue():
                    break
                self.assertEqual(run(["observe", "bad"]), 0)
                if "minimal failing set found" in out.getvalue():
                    break
            self.assertEqual(run(["proposal"]), 0)
            self.assertIn("setBarScale = 1.0", out.getvalue())
            # the config tree was never written by the bisect flow
            self.assertEqual(
                json.loads((root / "shell.json").read_text()),
                {"bar": {"scale": 1.4}})
            self.assertEqual(run(["reset"]), 0)

    def test_cli_requires_marks(self):
        import io
        with tempfile.TemporaryDirectory() as td:
            out = io.StringIO()
            rc = brain_cli.main(["--state", str(Path(td) / "s.json"),
                                 "bisect", "next"], out)
            self.assertEqual(rc, 2)
            self.assertIn("need both marks", out.getvalue())


if __name__ == "__main__":
    unittest.main()
