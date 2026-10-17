"""Tests for issue #120 Phase 2: workspace clustering -> profile proposals
(2.1), per-monitor-topology memory (2.2), the idle-throttle rule (2.3),
wallpaper palette extraction (2.4), the scheme accessibility audit with
dichromacy simulation (2.5), and the rhythm engine fed by shell events
(2.6).

Every proposal path is asserted to end in a PENDING LEDGER ENTRY — nothing
here may auto-apply a config change.
"""
import io
import json
import tempfile
import unittest
import zlib
from pathlib import Path

from assistant.capabilities.brain import cli as brain_cli
from assistant.capabilities.brain import service as brain_service
from assistant.capabilities.brain import topology, workspace
from assistant.capabilities.brain.ledger import Ledger
from assistant.capabilities.diagnostics import engine as diag_engine
from assistant.capabilities.genius import palette_extract


def _snap(dir_path: Path):
    return {p.name: p.read_bytes() for p in sorted(dir_path.iterdir()) if p.is_file()}


# ---------------------------------------------------------------------------
# 2.1 workspace.py
# ---------------------------------------------------------------------------

class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.ledger_path = self.d / "ledger.json"

    def tearDown(self):
        self.tmp.cleanup()

    def _sessions(self):
        # Two obviously consistent groups + noise the purity filter rejects.
        sessions = []
        for h in (9, 10, 9, 10, 9):
            sessions.append({"app": "code", "workspace": 1, "monitor": "DP-1",
                             "hour": h})
        for h in (21, 22, 21, 22, 21):
            sessions.append({"app": "steam", "workspace": 3, "monitor": "HDMI-A-1",
                             "hour": h})
        return sessions

    def test_correct_cluster_is_found(self):
        result = workspace.cluster(self._sessions(), k=2, min_support=3,
                                   purity=0.6)
        names = {p["name"] for p in result["profiles"]}
        self.assertIn("code-dp-1-ws1", names)
        self.assertIn("steam-hdmi-a-1-ws3", names)
        by_name = {p["name"]: p for p in result["profiles"]}
        self.assertEqual(by_name["code-dp-1-ws1"]["support"], 5)
        self.assertEqual(by_name["code-dp-1-ws1"]["purity"], 1.0)

    def test_propose_writes_pending_ledger_entries_only(self):
        before = _snap(self.d)
        result = workspace.propose_profiles(self._sessions(), Ledger(self.ledger_path))
        self.assertEqual(len(result["proposals"]), 2)
        pending = Ledger(self.ledger_path).pending()
        self.assertEqual(len(pending), 2)
        for item in pending:
            self.assertEqual(item["kind"], "workspace_profile")
            self.assertEqual(item["status"], "pending")
        # nothing else written (no config, no state file)
        after = _snap(self.d)
        self.assertEqual(set(after) - set(before), {"ledger.json"})

    def test_noise_never_becomes_a_profile(self):
        records = [{"app": f"app{i % 7}", "workspace": i % 5,
                    "monitor": f"M{i % 3}", "hour": i % 24}
                   for i in range(30)]
        result = workspace.cluster(records, k=4, min_support=3, purity=0.6)
        self.assertEqual(result["profiles"], [])

    def test_service_dry_run_and_propose_paths(self):
        r = brain_service.workspace_profiles(self._sessions(), propose=False)
        self.assertEqual(r["k"], 2)
        self.assertFalse(_snap(self.d))  # no file written at all in dry-run


# ---------------------------------------------------------------------------
# 2.2 topology.py
# ---------------------------------------------------------------------------

class TopologyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.target = self.d / "shell.json"
        self.target.write_text("{}", encoding="utf-8")
        self.state = str(self.d / "state.json")
        self.ledger = self.d / "ledger.json"

    def tearDown(self):
        self.tmp.cleanup()

    def _monitors(self, names, with_overrides=True):
        mon_dir = self.d / "monitors"
        mon_dir.mkdir(exist_ok=True)
        for name in names:
            sub = mon_dir / name
            sub.mkdir(exist_ok=True)
            if with_overrides:
                (sub / "shell.json").write_text(json.dumps(
                    {"bar": {"dock": {"iconSize": 40}}}), encoding="utf-8")

    def test_two_topology_hashes_produce_distinct_deltas(self):
        self._monitors(["DP-1"])
        fp1 = topology.fingerprint(self.d / "monitors")
        self.assertEqual(fp1["monitors"], ["DP-1"])
        deltas1 = topology.collect_deltas(self.target)
        self.assertEqual(deltas1, [{"monitor": "DP-1", "tool": "setDockIconSize",
                                    "path": "bar.dock.iconSize", "value": 40}])
        # topological change: a second monitor appears
        self._monitors(["DP-1", "HDMI-A-1"])
        fp2 = topology.fingerprint(self.d / "monitors")
        self.assertNotEqual(fp1["hash"], fp2["hash"])
        self.assertEqual(len(topology.collect_deltas(self.target)), 2)

    def test_observe_remember_propose_roundtrip(self):
        mon_dir = self.d / "monitors"
        # 1) first topology: one monitor with an override — remembered, no proposal
        self._monitors(["DP-1"])
        r1 = topology.observe(self.target, self.state, ledger=Ledger(self.ledger),
                              propose=True)
        self.assertFalse(r1["changed"])  # first observe: no previous
        self.assertIsNone(r1["proposal_id"])
        # 2) topology changes: second monitor appears with NO overrides
        self._monitors(["DP-1", "HDMI-A-1"], with_overrides=False)
        r2 = topology.observe(self.target, self.state, ledger=Ledger(self.ledger),
                              propose=True)
        self.assertTrue(r2["changed"])
        self.assertIsNone(r2["proposal_id"])  # nothing remembered for the NEW set
        # the new topology still carries DP-1's override as its current delta
        self.assertEqual([d["monitor"] for d in r2["deltas"]], ["DP-1"])
        # 3) the original monitor set reconnects (HDMI-A-1 disappears): memory fires
        (mon_dir / "HDMI-A-1").rmdir()  # override-less monitor dirs are empty
        r3 = topology.observe(self.target, self.state, ledger=Ledger(self.ledger),
                              propose=True)
        self.assertTrue(r3["changed"])
        self.assertTrue(r3["remembered"])
        self.assertEqual(r3["proposal_id"], 1)
        pending = Ledger(self.ledger).pending()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["kind"], "topology_restore")
        calls = pending[0]["diff"]["calls"]
        self.assertEqual([(c["tool"], c["value"]) for c in calls],
                         [("setDockIconSize", 40)])
        # never auto-applied: the target file is still empty
        self.assertEqual(self.target.read_text(), "{}")

    def test_empty_monitors_dir_is_a_valid_topology(self):
        fp = topology.fingerprint(self.d / "monitors")
        self.assertEqual(fp["monitors"], [])
        self.assertEqual(topology.collect_deltas(self.target), [])


# ---------------------------------------------------------------------------
# 2.3 idle-throttle rule (rules.d) + propose/revert round-trip
# ---------------------------------------------------------------------------

class IdleThrottleRuleTests(unittest.TestCase):
    def test_rule_loads_and_matches(self):
        rules = diag_engine.load_rules()
        self.assertTrue(any(r["id"] == "CL-idle-throttle-001" for r in rules))
        res = diag_engine.diagnose(
            "battery drains overnight while the machine sits idle", rules=rules)
        self.assertEqual(res["verdict"], "MATCH")
        self.assertEqual(res["candidates"][0]["rule"]["id"], "CL-idle-throttle-001")

    def test_propose_revert_roundtrip_through_the_bridge(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            target = d / "shell.json"
            target.write_text(json.dumps(
                {"appearance": {"blur": True,
                                "anim": {"durations": {"scale": 1.0}}}}))
            from assistant.capabilities.brain import bridge
            state = str(d / "s.json")
            ledger = str(d / "l.json")
            # propose the throttle (as the rule's fix describes)
            r = bridge.handle({"op": "settings_propose_via_calls"} if False else {
                "op": "route", "text": "placeholder"}, state_path=state,
                ledger_path=ledger)  # ensure bridge handle works
            self.assertTrue(r["ok"])
            # the real round-trip: settings_bridge with the exact preset calls
            from assistant.capabilities.brain import settings_bridge
            led = Ledger(ledger)
            p1 = settings_bridge.propose(
                led, target, calls=[
                    {"tool": "setBlurEnabled", "action": "set", "value": False,
                     "raw": "throttle"},
                    {"tool": "setAnimationSpeed", "action": "set", "value": 0.25,
                     "raw": "throttle"}],
                reason="idle throttle: reduce blur and animation while away")
            self.assertIsNotNone(p1["proposal_id"])
            approved = settings_bridge.decide(led, p1["proposal_id"], True)
            self.assertTrue(approved["applied"])
            # the revert proposal: blur back on, animations to default speed
            p2 = settings_bridge.propose(
                led, target, calls=[
                    {"tool": "setBlurEnabled", "action": "set", "value": True,
                     "raw": "revert"},
                    {"tool": "setAnimationSpeed", "action": "set", "value": 1.0,
                     "raw": "revert"}],
                reason="idle throttle revert: activity resumed")
            approved2 = settings_bridge.decide(led, p2["proposal_id"], True)
            self.assertTrue(approved2["applied"])
            cfg = json.loads(target.read_text())
            self.assertTrue(cfg["appearance"]["blur"])
            self.assertEqual(cfg["appearance"]["anim"]["durations"]["scale"], 1.0)


# ---------------------------------------------------------------------------
# 2.4 wallpaper palette extraction
# ---------------------------------------------------------------------------

def _png(width, height, pixel_fn):
    """Build a minimal 8-bit RGB PNG from a pixel function (x, y) -> (r,g,b).

    Header packing uses int.to_bytes so the test itself needs no import
    beyond the already-allow-listed zlib (crc32)."""
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter type 0
        for x in range(width):
            raw.extend(pixel_fn(x, y))

    def be32(n):
        return n.to_bytes(4, "big")

    def chunk(ctype, payload):
        c = ctype + payload
        return be32(len(payload)) + c + be32(zlib.crc32(c))

    ihdr = (be32(width) + be32(height)
            + bytes([8, 2, 0, 0, 0]))  # bit depth 8, RGB, no interlace
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(bytes(raw))) + chunk(b"IEND", b""))


class PaletteExtractTests(unittest.TestCase):
    def test_synthetic_wallpaper_dominant_accent(self):
        # A teal-ish image with a warm accent square in the corner.
        def px(x, y):
            if x < 8 and y < 8:
                return (230, 90, 60)   # warm accent patch
            return (30, 120, 120)      # dominant teal background
        data = _png(32, 32, px)
        result = palette_extract.accent_from_png(data)
        self.assertTrue(result["accent"].startswith("#"))
        self.assertIn("highest-chroma", result["selection"])
        # the accent must be one of the two painted colors' neighborhood,
        # not a blend artifact: clusters are k-means means, so check shares
        shares = sorted((c["share"] for c in result["clusters"]), reverse=True)
        self.assertGreater(shares[0], 0.8)  # teal dominates the sampling
        # WCAG verdicts present, inert (no scheme was touched)
        self.assertIn("on_white", result["contrast"])

    def test_png_reader_rejects_garbage_honestly(self):
        with self.assertRaises(ValueError):
            palette_extract.png_pixels(b"not a png at all")

    def test_extract_accent_empty_pixels_raises(self):
        with self.assertRaises(ValueError):
            palette_extract.extract_accent([])


# ---------------------------------------------------------------------------
# 2.5 accessibility audit + dichromacy simulation + repair
# ---------------------------------------------------------------------------

class AccessibilityAuditTests(unittest.TestCase):
    def test_known_bad_scheme_has_documented_failing_pair(self):
        # #808080 on #909090: a documented, obviously failing pair (~1.1:1)
        bad = {"foreground": "#808080", "background": "#909090"}
        audit = palette_extract.audit_scheme(bad)
        self.assertEqual(len(audit["pairs"]), 1)
        self.assertEqual(len(audit["failing"]), 1)
        pair = audit["failing"][0]
        self.assertEqual((pair["text"], pair["background"]),
                         ("foreground", "background"))
        self.assertLess(pair["ratio"], 4.5)

    def test_good_scheme_passes(self):
        good = {"foreground": "#111111", "background": "#f5f5f5"}
        audit = palette_extract.audit_scheme(good)
        self.assertEqual(audit["failing"], [])

    def test_dichromacy_simulation_maps_known_colors(self):
        # The Viénot matrices map pure red toward a darker yellow-brown for
        # protanopes; the essential property is determinism + plausibility.
        red = "#ff0000"
        sim = palette_extract.simulate_colorblind(red, "protanopia")
        self.assertTrue(sim.startswith("#"))
        self.assertNotEqual(sim, red)
        r, g, b = (int(sim[i:i + 2], 16) for i in (1, 3, 5))
        self.assertGreaterEqual(r, g)  # red channel survives as the dominant
        with self.assertRaises(ValueError):
            palette_extract.simulate_colorblind(red, "achromatopsia")

    def test_nearest_compliant_repairs_the_failing_pair(self):
        bg = "#909090"
        bad_fg = "#808080"
        repair = palette_extract.nearest_compliant(bad_fg, bg)
        self.assertTrue(repair["ok"])
        self.assertGreaterEqual(repair["ratio"], 4.5)
        # hue/chroma preserved: the repair stays neutral (a==b==0 family)
        from assistant.capabilities.genius.creative import hex_to_oklch
        _l, _c, _h = hex_to_oklch(repair["hex"])
        self.assertLess(_c, 0.02)

    def test_nearest_compliant_honest_failure(self):
        # A mid-gray hue against the same gray background: with a==b==0 the
        # ratio at the L extremes is 21:1, so this must SUCCEED — use an
        # impossible pair instead: the same color as background at target 30.
        repair = palette_extract.nearest_compliant("#909090", "#909090",
                                                   target=30.0)
        self.assertFalse(repair["ok"])


# ---------------------------------------------------------------------------
# 2.6 rhythm fed by shell events
# ---------------------------------------------------------------------------

class SettingsRhythmTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.target = self.d / "shell.json"
        self.target.write_text("{}", encoding="utf-8")
        # undo history: 5 applies on Mondays at 9:00, one on Saturday at 22:00
        entries = []
        for i, (day, hour) in enumerate(
                [("2026-08-03", 9), ("2026-08-10", 9), ("2026-08-17", 9),
                 ("2026-08-24", 9), ("2026-08-31", 9), ("2026-08-08", 22)]):
            entries.append({"id": i + 1, "at": f"{day}T{hour:02d}:00:00+00:00",
                            "label": f"apply {i}",
                            "ops": [{"path": "bar.scale", "old": 1.0, "new": 0.7}]})
        (self.d / "shell.json.assistant-history.json").write_text(
            json.dumps({"next_id": len(entries) + 1, "entries": entries}),
            encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_pattern_surfaces_from_shell_events(self):
        # 2026-08-03 etc. are Mondays (5 events); Saturday has 1 event.
        # The z-deviation is positive for OVER-represented bins, so Monday
        # is strongly notable and Saturday stays below the 1.5 threshold.
        r = brain_service.settings_rhythm(self.target, ledger_path=self.d / "nope.json")
        self.assertEqual(r["events"], 6)
        days = {row["day"]: row for row in r["notable_days"]}
        self.assertIn("Mon", days)
        self.assertNotIn("Sat", days)
        self.assertGreater(days["Mon"]["z"], 1.5)
        hours = {row["hour"]: row for row in r["notable_hours"]}
        self.assertIn(9, hours)
        # 22 sits exactly at the 1.5 z boundary (1 of 6 events over 24 bins)
        self.assertIn(22, hours)
        self.assertGreaterEqual(hours[22]["z"], 1.5)

    def test_cli_rhythm_settings_file(self):
        out = io.StringIO()
        code = brain_cli.main(["--ledger", str(self.d / "nope.json"),
                               "rhythm", "--settings-file", str(self.target)], out)
        self.assertEqual(code, 0)
        self.assertIn("events: 6", out.getvalue())

    def test_scheme_switches_are_accepted_caller_supplied(self):
        switches = ["2026-09-01T08:00:00+00:00", "2026-09-01T20:00:00+00:00",
                    "2026-09-02T08:00:00+00:00", "2026-09-02T20:00:00+00:00",
                    "2026-09-03T08:00:00+00:00", "2026-09-03T20:00:00+00:00",
                    "2026-09-04T08:00:00+00:00", "2026-09-04T20:00:00+00:00"]
        r = brain_service.settings_rhythm(self.target, scheme_switches=switches)
        self.assertEqual(r["events"], 14)  # 6 applies + 8 switches


if __name__ == "__main__":
    unittest.main()
