"""Tests for the quarantined DBus surface (phase 2.7's DBus half,
the former proposals/2026-09-26-c-dbus-surface.md, pruned once
implemented — the design lives in the module docstring and history).

The proposal's own verification plan, as unit tests:

1. Unit tests against a FAKE ``subprocess.run`` (recorded calls, no
   process): argument-array shape, allow-list enforcement, value
   validation, undo records.
2. The kill-switch: nothing spawns while the capability manifest keeps
   ``dbus_surface`` off (the default) — reads AND writes both refuse.
3. Irreversibility: the KWin-script command refuses to run without the
   explicit second confirmation; undo records never claim reversibility
   the tool does not have.
4. Lint parity: the quarantine set is exactly {pkgprobe, dbus_surface}
   (pinned in agent/tests — duplicated here so the surface's own suite
   fails loudly if the carve-out regresses).

The integration test the proposal gates behind
``CAELESTIA_ASSIST_DBUS_TESTS=1`` on a real KDE session is NOT here —
CI never sets the variable; a skipped stub documents it.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from assistant.capabilities.settings import dbus_surface


def _enable(tmp: Path) -> Path:
    user = tmp / "caps.json"
    user.write_text(json.dumps({"dbus_surface": True}))
    os.environ["CAELESTIA_ASSIST_CAPABILITIES"] = str(user)
    return user


class _Env:
    """context: enabled/disabled capability manifest + recorded spawns"""

    def __init__(self, enabled: bool = True, spawn_result=None):
        self.enabled = enabled
        self.spawn_result = spawn_result or _ok("")
        self.calls = []

    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self._old = os.environ.get("CAELESTIA_ASSIST_CAPABILITIES")
        if self.enabled:
            _enable(self.tmp)
        else:
            os.environ["CAELESTIA_ASSIST_CAPABILITIES"] = \
                str(self.tmp / "absent.json")
        self._patcher = mock.patch.object(
            dbus_surface, "_spawn", side_effect=self._fake_spawn)
        self._patcher.start()
        return self

    def _fake_spawn(self, argv):
        self.calls.append(list(argv))
        return dict(self.spawn_result)

    def __exit__(self, *exc):
        self._patcher.stop()
        if self._old is None:
            os.environ.pop("CAELESTIA_ASSIST_CAPABILITIES", None)
        else:
            os.environ["CAELESTIA_ASSIST_CAPABILITIES"] = self._old
        self._tmp.cleanup()
        return False


def _ok(stdout: str = "") -> dict:
    return {"rc": 0, "stdout": stdout, "stderr": ""}


class KillSwitchTests(unittest.TestCase):
    def test_disabled_by_default_refuses_everything(self) -> None:
        # no manifest at all: defaults stand, dbus_surface is OFF, and
        # NOTHING spawns — reads or writes
        old = os.environ.pop("CAELESTIA_ASSIST_CAPABILITIES", None)
        try:
            with mock.patch.object(
                    dbus_surface, "_spawn",
                    side_effect=AssertionError("must not spawn")):
                read = dbus_surface.read_kscreen_outputs()
                write = dbus_surface.run_write(
                    "kwinrc.nightcolor.active", False)
        finally:
            if old is not None:
                os.environ["CAELESTIA_ASSIST_CAPABILITIES"] = old
        self.assertIn("error", read)
        self.assertFalse(read.get("enabled", True))
        self.assertIn("error", write)
        self.assertFalse(write.get("enabled", True))

    def test_plan_works_even_when_disabled(self) -> None:
        # the dry-run artifact is a LOOK: you can always see what WOULD
        # run, what risk it carries, and how it would revert
        old = os.environ.pop("CAELESTIA_ASSIST_CAPABILITIES", None)
        try:
            plan = dbus_surface.plan_write("kwinrc.nightcolor.active",
                                           False)
        finally:
            if old is not None:
                os.environ["CAELESTIA_ASSIST_CAPABILITIES"] = old
        self.assertTrue(plan["dry_run"])
        self.assertEqual(plan["argv"][0], "kwriteconfig6")


class FixedArgumentArrayTests(unittest.TestCase):
    """Every spawned command is a fixed array — never a string, never
    shell=True, values fill typed slots only."""

    def test_kwriteconfig6_shape(self) -> None:
        with _Env() as env:
            # two spawns: kreadconfig6 (old value) then kwriteconfig6
            env.spawn_result = _ok("true")
            out = dbus_surface.run_write("kwinrc.nightcolor.active",
                                         False)
        self.assertTrue(out.get("applied"))
        self.assertEqual(env.calls[0],
                         ["kreadconfig6", "--file", "kwinrc", "--group",
                          "NightColor", "--key", "Active"])
        self.assertEqual(env.calls[1],
                         ["kwriteconfig6", "--file", "kwinrc", "--group",
                          "NightColor", "--key", "Active", "false"])
        for argv in env.calls:
            self.assertIsInstance(argv, list)

    def test_lockscreen_timeout_bounds(self) -> None:
        with _Env() as env:
            out = dbus_surface.run_write("kwinrc.lockscreen.timeout", 300)
        self.assertTrue(out.get("applied"))
        self.assertIn("300", env.calls[-1])

    def test_out_of_catalog_bounds_refused(self) -> None:
        with _Env() as env:
            out = dbus_surface.run_write("kwinrc.lockscreen.timeout",
                                         99999)
        self.assertIn("error", out)
        self.assertEqual(env.calls, [])

    def test_wrong_value_type_refused(self) -> None:
        with _Env() as env:
            out = dbus_surface.run_write("kwinrc.nightcolor.active",
                                         "yes please")
        self.assertIn("error", out)
        self.assertEqual(env.calls, [])

    def test_unknown_command_refused(self) -> None:
        with _Env() as env:
            out = dbus_surface.run_write("rm -rf everything", 1)
        self.assertIn("error", out)
        self.assertEqual(env.calls, [])

    def test_powerprofiles_set_uses_observed_value(self) -> None:
        with _Env() as env:
            # read probe first (profile table), then the set
            env.spawn_result = _ok("  performance:\n* balanced:\n")
            out = dbus_surface.run_write("powerprofiles.set",
                                         "power-saver")
        self.assertTrue(out.get("applied"))
        self.assertEqual(env.calls[0], ["powerprofilesctl"])
        self.assertEqual(env.calls[-1],
                         ["powerprofilesctl", "set", "power-saver"])

    def test_kscreen_write_shape(self) -> None:
        with _Env() as env:
            env.spawn_result = _ok(
                "Output: 27\nName: DP-1\n   Mode: 2560x1440@60*\n")
            out = dbus_surface.run_write(
                "kscreen.mode.set",
                {"output": "DP-1", "mode": "1920x1080@60"})
        self.assertTrue(out.get("applied"))
        self.assertEqual(env.calls[0], ["kscreen-doctor", "--outputs"])
        # the write argv is the fixed template with the observed pair
        self.assertEqual(env.calls[-1],
                         ["kscreen-doctor", "output", "DP-1", "mode",
                          "1920x1080@60"])

    def test_kscreen_pair_must_be_observed_shape(self) -> None:
        with _Env() as env:
            out = dbus_surface.run_write("kscreen.mode.set",
                                         "output DP-1 mode 1920x1080@60")
        self.assertIn("error", out)
        self.assertEqual(env.calls, [])

    def test_catalog_ids_are_pinned(self) -> None:
        # growing the catalog is a reviewable diff: the ids are pinned
        self.assertEqual(
            {e["id"] for e in dbus_surface.CATALOG},
            {"kwinrc.nightcolor.active", "kwinrc.lockscreen.timeout",
             "kscreen.mode.set", "powerprofiles.set",
             "kwin.script.unload"})

    def test_no_destructive_risk_in_catalog(self) -> None:
        # DESTRUCTIVE / PRIVILEGED commands are structurally absent
        for spec in dbus_surface.CATALOG:
            self.assertIn(spec["risk"], ("READ_ONLY", "STATE_CHANGING"))


class UndoRecordTests(unittest.TestCase):
    def test_kwrite_undo_record_carries_old_value(self) -> None:
        with _Env() as env:
            env.spawn_result = _ok("true")
            out = dbus_surface.run_write("kwinrc.nightcolor.active",
                                         False)
        record = out["undo_record"]
        self.assertEqual(record["kind"], "kwriteconfig6")
        # the raw 'true' from kreadconfig6 is coerced back to bool so
        # the undo write round-trips through validation
        self.assertIs(record["old_value"], True)

    def test_apply_undo_writes_the_old_value_back(self) -> None:
        with _Env() as env:
            # first run reads old=true, writes false
            env.spawn_result = _ok("true")
            out = dbus_surface.run_write("kwinrc.nightcolor.active",
                                         False)
            record = out["undo_record"]
            # undo: reads current (false), writes true
            env.spawn_result = _ok("false")
            reverted = dbus_surface.apply_undo(record)
        self.assertTrue(reverted.get("applied"))
        self.assertEqual(env.calls[-1],
                         ["kwriteconfig6", "--file", "kwinrc", "--group",
                          "NightColor", "--key", "Active", "true"])

    def test_powerprofile_undo_sets_prior_active(self) -> None:
        with _Env() as env:
            env.spawn_result = _ok("  performance:\n* balanced:\n")
            out = dbus_surface.run_write("powerprofiles.set",
                                         "performance")
            record = out["undo_record"]
            self.assertEqual(record["old_value"], "balanced")
            env.spawn_result = _ok("* performance:\n  balanced:\n")
            reverted = dbus_surface.apply_undo(record)
        self.assertTrue(reverted.get("applied"))
        self.assertEqual(env.calls[-1],
                         ["powerprofilesctl", "set", "balanced"])

    def test_failed_write_produces_no_undo_record(self) -> None:
        with _Env() as env:
            env.spawn_result = {"rc": 1, "stdout": "",
                                "stderr": "refused"}
            out = dbus_surface.run_write("kwinrc.nightcolor.active",
                                         False)
        self.assertIn("error", out)
        self.assertNotIn("undo_record", out)


class IrreversibleTests(unittest.TestCase):
    def test_kwin_script_unload_refuses_without_confirmation(self) -> None:
        with _Env() as env:
            out = dbus_surface.run_write("kwin.script.unload",
                                         "my-script")
        self.assertIn("error", out)
        self.assertIn("confirm_irreversible", out["error"])
        self.assertEqual(env.calls, [])

    def test_kwin_script_unload_runs_with_confirmation(self) -> None:
        with _Env() as env:
            out = dbus_surface.run_write(
                "kwin.script.unload", "my-script",
                confirm_irreversible=True)
        self.assertTrue(out.get("applied"))
        self.assertFalse(out.get("reversible"))
        self.assertNotIn("undo_record", out)
        self.assertEqual(
            env.calls[0],
            ["dbus-send", "--session", "--dest=org.kde.KWin",
             "/Scripting", "org.kde.kwin.Scripting.unloadScript",
             "string:my-script"])

    def test_plan_declares_the_irreversibility(self) -> None:
        plan = dbus_surface.plan_write("kwin.script.unload", "my-script")
        self.assertFalse(plan["reversible"])
        self.assertIn("confirm_irreversible=True", plan["requires"])


class ReadProbeTests(unittest.TestCase):
    def test_kscreen_topology_parses_outputs(self) -> None:
        sample = ("Output: 27\n"
                  "Name: DP-1\n"
                  "   Modes: 3\n"
                  "     Mode: 1920x1080@60*\n"
                  "     Mode: 2560x1440@60\n"
                  "Output: 41\n"
                  "Name: eDP-1\n"
                  "     Mode: 1920x1200@60\n")
        with _Env() as env:
            env.spawn_result = _ok(sample)
            out = dbus_surface.read_kscreen_outputs()
        self.assertEqual(out["n"], 2)
        names = [o.get("name") for o in out["outputs"]]
        self.assertEqual(names, ["DP-1", "eDP-1"])
        self.assertEqual(out["outputs"][0]["current_mode"],
                         "1920x1080@60")

    def test_kscreen_undo_round_trips_the_observed_pair(self) -> None:
        sample = ("Output: 27\n"
                  "Name: DP-1\n"
                  "     Mode: 1920x1080@60*\n")
        with _Env() as env:
            env.spawn_result = _ok(sample)
            out = dbus_surface.run_write(
                "kscreen.mode.set",
                {"output": "DP-1", "mode": "2560x1440@60"})
            record = out["undo_record"]
            self.assertEqual(record["old_value"],
                             {"output": "DP-1", "mode": "1920x1080@60"})
            reverted = dbus_surface.apply_undo(record)
        self.assertTrue(reverted.get("applied"))
        self.assertEqual(env.calls[-1],
                         ["kscreen-doctor", "output", "DP-1", "mode",
                          "1920x1080@60"])

    def test_power_profiles_table_parses(self) -> None:
        with _Env() as env:
            env.spawn_result = _ok("  performance:\n* balanced:\n"
                                   "  power-saver:\n")
            out = dbus_surface.read_power_profiles()
        self.assertEqual(out["active"], "balanced")
        self.assertEqual({p["name"] for p in out["profiles"]},
                         {"performance", "balanced", "power-saver"})

    def test_read_probe_dispatch(self) -> None:
        with _Env() as env:
            env.spawn_result = _ok("  performance:\n* balanced:\n")
            out = dbus_surface.run_read("power")
            self.assertIn("profiles", out)
            bad = dbus_surface.run_read("nonsense")
        self.assertIn("error", bad)

    def test_honest_error_on_probe_failure(self) -> None:
        with _Env(spawn_result={"error": "kscreen-doctor failed to "
                                          "start: OSError"}) as env:
            out = dbus_surface.read_kscreen_outputs()
        self.assertIn("error", out)
        # the spawn WAS attempted (recorded) and failed honestly —
        # the error dict says so, no exception escaped
        self.assertEqual(env.calls, [["kscreen-doctor", "--outputs"]])


class LintParityTests(unittest.TestCase):
    def test_quarantine_constant_pins_the_two_spawn_users(self) -> None:
        # the import allow-list is gone; the quarantine constant survives
        # as documentation of the only two modules allowed to spawn
        from assistant.capabilities.diagnostics.schema_lint import _QUARANTINED_IMPORTS
        self.assertEqual(
            _QUARANTINED_IMPORTS,
            {"executor/runner.py": frozenset({"subprocess"}),
             "dbus_surface.py": frozenset({"subprocess"})})


@unittest.skipUnless(
    os.environ.get("CAELESTIA_ASSIST_DBUS_TESTS") == "1",
    "integration on a real KDE session only (CAELESTIA_ASSIST_DBUS_TESTS=1); "
    "never set in CI — the proposal's own gate")
class RealSessionIntegrationTests(unittest.TestCase):
    def test_read_probes_on_a_live_session(self) -> None:
        with _Env() as env:  # real spawns are NOT faked here? they are —
            pass  # placeholder: run only by a maintainer with a live KDE
            # session; see the proposal's verification plan item 2.


if __name__ == "__main__":
    unittest.main()
