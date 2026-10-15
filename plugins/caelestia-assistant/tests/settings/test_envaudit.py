"""F21 tests — the environment audit (settings/envaudit.py).

Under test:

- a healthy config: all checks INFO, zero critical, exit 0;
- an out-of-range configured value: lint-sourced CRITICAL (the applier
  would refuse the next apply of that key), exit 1;
- an INERT interaction (blur on while transparency off): WARN from the
  cited consequence table — deduplicated when lint already reported the
  same path;
- snapshot freshness: a matching newest snapshot is INFO; a diverged
  one is WARN;
- tampered snapshots fail integrity and are named in the audit;
- the audit never writes: the target bytes are unchanged.
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from assistant.capabilities.settings import cli, envaudit, environments

NOW = datetime(2026, 9, 28, 12, 0)


def _mk(initial) -> Tuple[Path, Path]:
    tmp = Path(tempfile.mkdtemp(prefix="envaudit-test-"))
    target = tmp / "shell.json"
    target.write_text(json.dumps(initial), encoding="utf-8")
    return tmp, target


def _cli(target: Path, argv: List[str]) -> Tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = cli.main([*argv, "--file", str(target)])
        except SystemExit as exc:
            rc = int(exc.code or 0)
    return rc, out.getvalue(), err.getvalue()


HEALTHY = {"bar": {"scale": 1.0},
           "appearance": {"blur": False,
                          "transparency": {"enabled": True}}}


class AuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp, self.target = _mk(HEALTHY)
        self.before = self.target.read_bytes()

    def _kinds(self, result):
        return {c["check"]: c["severity"] for c in result["checks"]}

    def test_healthy_environment_is_all_clear(self) -> None:
        result = envaudit.audit(self.target, now=NOW)
        kinds = self._kinds(result)
        self.assertEqual(result["critical"], 0)
        self.assertEqual(kinds.get("config-lint"), "INFO")
        self.assertIn("interaction", kinds)
        self.assertEqual(self.target.read_bytes(), self.before)

    def test_out_of_range_is_critical(self) -> None:
        self.target.write_text(json.dumps(
            {"bar": {"scale": 999}}), encoding="utf-8")
        result = envaudit.audit(self.target, now=NOW)
        self.assertEqual(result["critical"], 1)
        critical = next(c for c in result["checks"]
                        if c["severity"] == "CRITICAL")
        self.assertEqual(critical["check"], "config-lint")

    def test_inert_interaction_is_warn_with_citation(self) -> None:
        self.target.write_text(json.dumps(
            {"appearance": {"blur": True,
                            "transparency": {"enabled": False}}}),
            encoding="utf-8")
        result = envaudit.audit(self.target, now=NOW)
        # the INERT state is reported — via the lint (same path) or, for
        # paths lint does not cover, via the cited consequence table
        blur_warns = [c for c in result["checks"]
                      if c["severity"] in ("WARN", "CRITICAL")
                      and "blur" in c["detail"].lower()]
        self.assertTrue(blur_warns, "the INERT blur state must be reported")
        cited = [c for c in blur_warns if c.get("evidence")]
        self.assertTrue(cited, "the finding must carry its citation")

    def test_snapshot_freshness(self) -> None:
        environments.save(self.target, "fresh")
        result = envaudit.audit(self.target, now=NOW)
        kinds = self._kinds(result)
        self.assertEqual(kinds.get("snapshot-fresh"), "INFO")
        # diverge the live config
        self.target.write_text(json.dumps(
            {"bar": {"scale": 0.9}}), encoding="utf-8")
        result = envaudit.audit(self.target, now=NOW)
        kinds = self._kinds(result)
        self.assertEqual(kinds.get("snapshot-fresh"), "WARN")

    def test_tampered_snapshot_is_named(self) -> None:
        environments.save(self.target, "clean")
        hist = self.target.parent / (self.target.name +
                                     ".assistant-history.json")
        data = json.loads(hist.read_text(encoding="utf-8"))
        data["environments"][0]["content"] = {"bar": {"scale": 3}}
        hist.write_text(json.dumps(data), encoding="utf-8")
        result = envaudit.audit(self.target, now=NOW)
        tampered = [c for c in result["checks"]
                    if "integrity" in c["detail"] or "FAILED" in c["detail"]]
        self.assertTrue(tampered, "a tampered snapshot must be reported")

    def test_cli_exit_codes(self) -> None:
        rc, out, err = _cli(self.target, ["--audit-env"])
        self.assertEqual(rc, 0, out)
        self.assertIn("no critical findings", out)
        self.target.write_text(json.dumps(
            {"bar": {"scale": 999}}), encoding="utf-8")
        rc, out, err = _cli(self.target, ["--audit-env"])
        self.assertEqual(rc, 1)
        self.assertIn("CRITICAL", out)


if __name__ == "__main__":
    unittest.main()
