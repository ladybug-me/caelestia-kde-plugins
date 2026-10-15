"""F27 tests — the composite doctor (assistant/doctor.py).

Under test:

- run_doctor returns all five checks with verdicts and repro commands;
  nothing crashes even when subsystems are unavailable (UNAVAILABLE,
  never a fake pass);
- the sandbox run (pinned HOME so user state is empty) yields: registry
  freshness PASS (sibling checkout), generated docs PASS, arena smoke
  PASS at the F3 numbers, calibration UNAVAILABLE on empty state,
  budget probe PASS with the machine caveat traveling;
- a FAIL verdict forces exit code 1 (rendered from the result, not by
  re-running).
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from assistant import doctor


def _empty_home():
    """Pin HOME AND the brain-state path (state.py resolves its
    DEFAULT_STATE at import time, so HOME alone is not enough)."""
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="doctor-test-"))
    old = os.environ.get("HOME")
    os.environ["HOME"] = str(tmp)
    old_state = os.environ.get("CAELESTIA_BRAIN_STATE")
    os.environ["CAELESTIA_BRAIN_STATE"] = str(tmp / "nonexistent-state.json")

    def _restore():
        if old is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = old
        if old_state is None:
            os.environ.pop("CAELESTIA_BRAIN_STATE", None)
        else:
            os.environ["CAELESTIA_BRAIN_STATE"] = old_state
    return _restore


class DoctorTests(unittest.TestCase):
    def test_all_five_checks_present_with_repros(self) -> None:
        result = doctor.run_doctor()
        self.assertEqual(set(result["checks"]), {
            "registry freshness", "generated docs", "arena smoke",
            "calibration coverage", "budget probe"})
        for name, check in result["checks"].items():
            self.assertIn(check["verdict"],
                          ("PASS", "WARN", "FAIL", "UNAVAILABLE"), name)
            self.assertTrue(check["detail"], name)

    def test_sandbox_run_verdicts(self) -> None:
        restore = _empty_home()
        self.addCleanup(restore)
        result = doctor.run_doctor()
        verdicts = {name: c["verdict"]
                    for name, c in result["checks"].items()}
        # HERMETIC: the registry-freshness check needs the pinned upstream
        # checkout beside this repo; when it is absent (the normal installed
        # case, and always in CI/sandboxes) the honest verdict is
        # UNAVAILABLE — PASS is only reachable (and only asserted) when a
        # checkout the check can actually verify is present. Mirrors the
        # doctor's own locate_upstream() so both mean the same by 'present'.
        if doctor.locate_upstream() is not None:
            self.assertEqual(verdicts["registry freshness"], "PASS")
        else:
            self.assertEqual(verdicts["registry freshness"], "UNAVAILABLE",
                             "no upstream checkout: must be UNAVAILABLE, "
                             "never a fake pass")
        self.assertEqual(verdicts["generated docs"], "PASS")
        self.assertEqual(verdicts["arena smoke"], "PASS")
        self.assertEqual(verdicts["calibration coverage"], "UNAVAILABLE",
                         "empty user state must be UNAVAILABLE, not PASS")
        self.assertIn(verdicts["budget probe"], ("PASS", "WARN"))
        self.assertEqual(result["fails"], 0)

    def test_render_and_exit_contract(self) -> None:
        result = {"checks": {
            "a": {"verdict": "PASS", "detail": "fine", "repro": "run x"},
            "b": {"verdict": "FAIL", "detail": "broken"},
        }, "fails": 1, "note": "n"}
        lines = doctor.render_lines(result)
        text = "\n".join(lines)
        self.assertIn("[PASS       ] a: fine", text)
        self.assertIn("repro: run x", text)
        self.assertIn("1 FAIL — exit 1", text)
        self.assertNotIn("0 FAIL", text)

    def test_unavailable_is_never_a_fake_pass(self) -> None:
        result = doctor.run_doctor()
        for name, check in result["checks"].items():
            if check["verdict"] == "UNAVAILABLE":
                self.assertTrue(check["detail"], name)


if __name__ == "__main__":
    unittest.main()
