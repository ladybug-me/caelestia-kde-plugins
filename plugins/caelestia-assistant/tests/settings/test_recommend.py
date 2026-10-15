"""Tests for assistant.capabilities.settings.recommend (exponential-build-5 F29).

Covers: the conjugate partial-pooling math (prior-only with no history,
shrinkage with history, dominance with a long history), range clamping
with the clamp reported, the Pareto position against profiles, kind
refusals, the inert contract, and the CLI surface.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from assistant.capabilities.settings import recommend


def _ledger(values):
    tmp = tempfile.mkdtemp(prefix="rec29-")
    p = Path(tmp) / "shell.json.assistant-history.json"
    p.write_text(json.dumps({
        "next_id": 2,
        "entries": [{"id": 1, "at": "2026-09-28T00:00:00+00:00",
                     "label": "t", "ops": [{"path": "bar.dock.iconSize",
                                             "old": None, "new": v}
                                            for v in values]}],
        "undo_log": []}), encoding="utf-8")
    return str(p)


class PoolingTests(unittest.TestCase):
    def test_unknown_tool(self) -> None:
        self.assertEqual(recommend.recommend("setNope")["verdict"],
                         "UNKNOWN_TOOL")

    def test_non_numeric_kind_refused(self) -> None:
        r = recommend.recommend("setBlurEnabled")  # bool
        self.assertEqual(r["verdict"], "UNSUPPORTED_KIND")
        r2 = recommend.recommend("setBarPosition")  # enum
        self.assertEqual(r2["verdict"], "UNSUPPORTED_KIND")

    def test_no_history_is_the_pooled_prior_and_says_so(self) -> None:
        r = recommend.recommend("setDockIconSize")
        self.assertEqual(r["verdict"], "OK")
        self.assertEqual(r["evidence"]["n_history"], 0)
        # pool: default 32, compact 24, minimal 20, macos-like 48
        pool = [32.0, 24.0, 20.0, 48.0]
        self.assertAlmostEqual(r["value"],
                               sum(pool) / len(pool), delta=2.0)
        self.assertIn("no approved history", r["note"])
        self.assertIn("starting point, not your preference", r["note"])

    def test_history_shrinks_toward_the_pool(self) -> None:
        # user approved 40 three times: posterior between 40 and the
        # pool mean (~31), strictly pulled toward the pool
        r = recommend.recommend("setDockIconSize",
                                ledger_path=_ledger([40, 40, 40]))
        self.assertEqual(r["verdict"], "OK")
        pool_mean = 31.0
        self.assertLess(r["value"], 40.0)
        self.assertGreater(r["value"], pool_mean)
        self.assertIn("3 approved value(s)", r["note"])

    def test_long_history_dominates(self) -> None:
        # 30 approvals at 44 with tiny spread: the posterior is ~44
        r = recommend.recommend("setDockIconSize",
                                ledger_path=_ledger([44] * 30))
        self.assertGreater(r["value"], 42.0)

    def test_interval_brackets_the_value_and_clamps_to_range(self) -> None:
        r = recommend.recommend("setDockIconSize")
        lo, hi = r["credible_interval_95"]
        self.assertLessEqual(lo, r["value"])
        self.assertGreaterEqual(hi, r["value"])
        self.assertGreaterEqual(lo, 16)   # registry minimum
        self.assertLessEqual(hi, 96)     # registry maximum
        self.assertFalse(r["clamped"])   # inside the pool by construction

    def test_out_of_range_pool_clamps_and_reports(self) -> None:
        # a preset pool pulled beyond the registry maximum must clamp
        # to the legal range and SAY it clamped
        from assistant.capabilities.settings import recommend as rec
        from unittest import mock

        class Fake:
            name, path, kind = "setFake", "fake.x", "int"
            default, minimum, maximum, step = 50, 0, 60, 1
        with mock.patch.object(rec, "tool_by_name",
                               return_value=Fake()), \
             mock.patch.object(rec, "tool_by_path", return_value=Fake()), \
             mock.patch.object(rec, "_preset_pool",
                               return_value=[("huge", 200.0)]):
            r = rec.recommend("setFake")
        self.assertTrue(r["clamped"])
        self.assertEqual(r["value"], 60)
        self.assertIn("clamped", r["note"])

    def test_pareto_position_against_profiles(self) -> None:
        r = recommend.recommend("setDockIconSize")
        positions = {p["profile"]: p["position"]
                     for p in r["pareto_position"]}
        self.assertIn("compact", positions)
        self.assertIn("minimal", positions)
        self.assertIn("macos-like", positions)
        for p in r["pareto_position"]:
            self.assertIn(p["position"], ("matches", "above", "below"))

    def test_rejections_not_modeled_is_stated(self) -> None:
        r = recommend.recommend("setDockIconSize")
        self.assertIn("rejections", r["note"])
        self.assertIn("NOT part of this estimate", r["note"])

    def test_proposal_only_flag(self) -> None:
        self.assertTrue(recommend.recommend("setDockIconSize")
                        ["proposal_only"])


class CliTests(unittest.TestCase):
    def test_recommend_flag_end_to_end(self) -> None:
        from assistant.capabilities.settings import cli as settings_cli
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "shell.json"
            target.write_text("{}", encoding="utf-8")
            out = StringIO()
            with redirect_stdout(out):
                rc = settings_cli.main(
                    ["--recommend", "setDockIconSize",
                     "--file", str(target)])
            self.assertEqual(rc, 0)
            text = out.getvalue()
            self.assertIn("recommend: setDockIconSize", text)
            self.assertIn("95% interval", text)
            self.assertIn("vs profiles:", text)
            # the recommender wrote nothing
            self.assertEqual(json.loads(target.read_text()), {})
            # no history ledger was created either
            self.assertFalse((target.parent / (target.name +
                             ".assistant-history.json")).exists())

    def test_recommend_unknown_tool_nonzero(self) -> None:
        from assistant.capabilities.settings import cli as settings_cli
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "shell.json"
            target.write_text("{}", encoding="utf-8")
            out = StringIO()
            with redirect_stdout(out):
                rc = settings_cli.main(
                    ["--recommend", "setNope", "--file", str(target)])
            self.assertEqual(rc, 1)
            self.assertIn("unknown tool", out.getvalue())

    def test_recommend_uses_target_ledger_when_present(self) -> None:
        from assistant.capabilities.settings import cli as settings_cli
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "shell.json"
            target.write_text("{}", encoding="utf-8")
            hist = target.parent / (target.name +
                                    ".assistant-history.json")
            hist.write_text(json.dumps({
                "next_id": 2,
                "entries": [{"id": 1, "at": "x", "label": "t",
                             "ops": [{"path": "bar.dock.iconSize",
                                      "old": 32, "new": 40}]}],
                "undo_log": []}), encoding="utf-8")
            out = StringIO()
            with redirect_stdout(out):
                rc = settings_cli.main(
                    ["--recommend", "setDockIconSize",
                     "--file", str(target)])
            self.assertEqual(rc, 0)
            self.assertIn("1 value(s)", out.getvalue())


if __name__ == "__main__":
    unittest.main()
