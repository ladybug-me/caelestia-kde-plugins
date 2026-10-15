"""Tests for exponential-build-3 F2 — the synthetic-fixture framework
(assistant/fixtures.py).

The contract under test: a contributor with NO caelestia-kde
installation can exercise every reader the assistant has, because the
builders fabricate byte-deterministic, reader-compatible artifacts:

- shell-config trees: the canonical valid config, plus the five
  canonical fault variants (out-of-range, unknown key, type mismatch,
  malformed JSON, non-object), all parseable by the settings layer's
  own reader (valid) and honestly broken where they claim to be;
- telemetry trees: /proc + /sys snapshot shapes that the read-only
  probes parse to EXACT pinned values (loadavg, meminfo, battery
  capacity, thermal zones) — including multi-battery and multi-zone
  layouts;
- markdown vaults: scan() sees the notes, tags, and wiki-links, and
  pinned mtimes drive the time-decay arithmetic;
- brain state / ledger files: the real loaders read them back.

Determinism: identical builder calls in different directories produce
byte-identical artifacts (pinned).
"""
import json
import tempfile
import unittest
from pathlib import Path

from assistant import fixtures
from assistant.capabilities.brain.ledger import Ledger
from assistant.capabilities.brain import state as brain_state
from assistant.capabilities.brain.personal.graph import decay_weights, links
from assistant.capabilities.brain.personal.vault import scan
from assistant.capabilities.diagnostics import telemetry


class ShellConfigFixtureTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_canonical_config_bytes(self):
        path = fixtures.shell_config(self.dir / "shell.json")
        # the applier's own formatting: 4-space indent + newline
        self.assertEqual(
            path.read_text(),
            '{\n    "bar": {\n        "scale": 1.0,\n'
            '        "persistent": true,\n'
            '        "position": "bottom"\n    },\n'
            '    "unrelated": {\n        "keep": [\n            1,\n'
            '            2\n        ]\n    }\n}\n')

    def test_raw_override_writes_exact_bytes(self):
        path = fixtures.shell_config(self.dir / "bad.json",
                                     raw="{not json")
        self.assertEqual(path.read_text(), "{not json")

    def test_tree_holds_the_canonical_fault_vocabulary(self):
        made = fixtures.shell_tree(self.dir)
        self.assertEqual(sorted(made), [
            "malformed", "non-object", "out-of-range",
            "type-mismatch", "unknown-key", "valid"])
        doc = json.loads(made["valid"].read_text())
        self.assertEqual(doc["bar"]["scale"], 1.0)
        self.assertEqual(
            json.loads(made["out-of-range"].read_text())["bar"]["scale"],
            99.0)
        with self.assertRaises(ValueError):
            json.loads(made["malformed"].read_text())
        self.assertIsInstance(
            json.loads(made["non-object"].read_text()), list)

    def test_deterministic_bytes(self):
        a = fixtures.shell_tree(self.dir / "a")
        b = fixtures.shell_tree(self.dir / "b")
        for name in a:
            self.assertEqual(a[name].read_bytes(), b[name].read_bytes())


class TelemetryFixtureTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_default_tree_parses_to_pinned_values(self):
        t = fixtures.telemetry_tree(self.dir)
        self.assertEqual(
            telemetry.read_loadavg(t["loadavg"]),
            {"available": True, "source": t["loadavg"],
             "load1": 0.4, "load5": 0.35, "load15": 0.3})
        self.assertEqual(
            telemetry.read_meminfo(t["meminfo"]),
            {"available": True, "source": t["meminfo"],
             "total_kb": 16_000_000, "available_kb": 8_000_000,
             "used_ratio": 0.5})
        battery = telemetry.read_battery(t["battery_globs"][0])
        self.assertTrue(battery["available"])
        self.assertEqual(battery["batteries"][0]["capacity_pct"], 74)
        thermal = telemetry.read_thermal(t["thermal_globs"][0])
        self.assertTrue(thermal["available"])
        self.assertEqual(thermal["zones"][0]["temp_c"], 62.0)

    def test_multi_battery_multi_zone_layouts(self):
        t = fixtures.telemetry_tree(
            self.dir, batteries={"BAT0": 74, "BAT1": 30},
            thermal_zones={"thermal_zone0": 62_000,
                           "thermal_zone1": 55_000})
        battery = telemetry.read_battery(t["battery_globs"][0])
        self.assertEqual([b["capacity_pct"] for b in battery["batteries"]],
                         [74, 30])
        thermal = telemetry.read_thermal(t["thermal_globs"][0])
        self.assertEqual([z["temp_c"] for z in thermal["zones"]],
                         [62.0, 55.0])

    def test_snapshot_sees_every_probe(self):
        t = fixtures.telemetry_tree(self.dir)
        snap = telemetry.snapshot(t["proc"], t["sys"])
        for key in ("loadavg", "meminfo", "battery", "thermal"):
            self.assertIn(key, snap)


class VaultFixtureTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_scan_sees_notes_tags_links(self):
        root = fixtures.vault(self.dir / "vault")
        notes = scan(str(root))
        self.assertEqual(sorted(notes),
                         ["cat.md", "dog.md", "mix.md", "orphan.md"])
        self.assertEqual(links(notes["cat.md"]["text"]), {"dog-notes"})
        self.assertIn("orphan.md", notes)

    def test_pinned_mtimes_drive_decay_weights(self):
        now = 1_800_000_000.0
        root = fixtures.vault(
            self.dir / "vault",
            notes={"fresh.md": "see [[x]]", "old.md": "see [[x]]"},
            mtimes={"fresh.md": now, "old.md": now - 90 * 86400.0})
        weights = decay_weights(scan(str(root)), now, half_life_days=90.0)
        self.assertEqual(weights[("fresh.md", "x")], 1.0)
        self.assertAlmostEqual(weights[("old.md", "x")], 0.5, places=12)


class StateAndLedgerFixtureTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_state_file_loads_through_the_real_loader(self):
        path = fixtures.state_file(self.dir / "state.json")
        state = brain_state.load(str(path))
        self.assertEqual(state["preset_comparisons"][0]["winner"],
                         "minimal")

    def test_ledger_file_loads_through_the_real_ledger(self):
        path = fixtures.ledger_file(self.dir / "ledger.json")
        ledger = Ledger(str(path))
        pending = ledger.pending()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["kind"], "launch_pattern")

    def test_deterministic_bytes(self):
        a = fixtures.state_file(self.dir / "a.json")
        b = fixtures.state_file(self.dir / "b.json")
        self.assertEqual(a.read_bytes(), b.read_bytes())


if __name__ == "__main__":
    unittest.main()
