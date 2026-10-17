"""brain.rete tests — alpha tests, beta joins, windows, proposals."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from assistant.capabilities.brain import rete  # noqa: E402

CRASH_LOOP_RULE = {
    "id": "crash-loop-mute",
    "when": [{"field": "kind", "op": "eq", "value": "crash"}],
    "within_seconds": 600,
    "min_events": 3,
    "then": {"tool": "setNotifsMaxPopups", "value": 3},
    "reason": "repeated crashes observed",
}


def loop_events(spacing=1.0, n=4, start=100.0):
    return [{"time": start + i * spacing, "kind": "crash",
             "app": "konsole"} for i in range(n)] + \
        [{"time": start - 50.0, "kind": "log"}]


class TestAlpha(unittest.TestCase):
    def test_unknown_op_rejected_at_compile(self):
        with self.assertRaises(ValueError):
            rete.Rete([{"id": "x", "when": [{"field": "a", "op": "~"}]}])

    def test_missing_field_does_not_match(self):
        rule = {"id": "r", "when": [{"field": "app", "op": "eq",
                                     "value": "konsole"}]}
        r = rete.run_events([rule], [{"time": 1, "kind": "crash"}])
        self.assertEqual(r["n_fired"], 0)


class TestBeta(unittest.TestCase):
    def test_window_respected(self):
        events = [{"time": 1, "kind": "crash"},
                  {"time": 2, "kind": "crash"},
                  {"time": 5000, "kind": "crash"}]
        r = rete.run_events([CRASH_LOOP_RULE], events)
        for f in r["fired"]:
            times = [e["time"] for e in f["evidence"]]
            self.assertLessEqual(max(times) - min(times), 600)

    def test_min_events_gates_firing(self):
        events = [{"time": 1, "kind": "crash"},
                  {"time": 2, "kind": "crash"}]  # min_events 3
        r = rete.run_events([CRASH_LOOP_RULE], events)
        self.assertEqual(r["n_fired"], 0)

    def test_variable_join_across_conditions(self):
        # both conditions bind the SAME variable: the app that appears
        # in 'crash-start' must be the app in 'crash-end'
        rule = {
            "id": "paired",
            "when": [{"field": "kind", "op": "eq",
                      "value": "crash-start", "var": "app"},
                     {"field": "kind", "op": "eq",
                      "value": "crash-end", "var": "app"}],
            "within_seconds": 60,
            "then": {"note": "one app crashed twice"},
        }
        events = [{"time": 1, "kind": "crash-start", "app": "a"},
                  {"time": 2, "kind": "crash-start", "app": "b"},
                  {"time": 3, "kind": "crash-end", "app": "a"},
                  {"time": 4, "kind": "crash-end", "app": "b"}]
        r = rete.run_events([rule], events)
        apps = [e["app"] for f in r["fired"] for e in f["evidence"]]
        # every token's evidence pairs one app only
        for f in r["fired"]:
            pair_apps = {e["app"] for e in f["evidence"]}
            self.assertEqual(len(pair_apps), 1, f["evidence"])


class TestProposals(unittest.TestCase):
    def test_all_fired_are_proposals(self):
        r = rete.run_events([CRASH_LOOP_RULE], loop_events())
        self.assertGreater(r["n_fired"], 0)
        for f in r["fired"]:
            self.assertEqual(f["verdict"], "SUGGESTED_NOT_EXECUTED")
        self.assertIn("applies nothing", r["note"])

    def test_deterministic(self):
        ev = loop_events()
        a = rete.run_events([CRASH_LOOP_RULE], ev)
        b = rete.run_events([CRASH_LOOP_RULE], ev)
        self.assertEqual(a, b)

    def test_cap_abstains(self):
        r = rete.run_events([CRASH_LOOP_RULE],
                            [{"time": i, "kind": "x"}
                             for i in range(6000)])
        self.assertEqual(r["verdict"], "ABSTAIN")


class TestRulesFile(unittest.TestCase):
    def test_missing_file_is_an_honest_error(self):
        # Phase Z hostile-input finding: a silent empty list on an
        # unreadable file hid the user's own opt-in from them. The
        # contract is now a ValueError the CLI reports; the CLI's
        # own "no rules file yet" notice handles the common case
        # BEFORE calling this (cmd_rules checks path.exists()).
        with self.assertRaises(ValueError):
            rete.load_rules(Path("/nonexistent/rules.json"))

    def test_load_rules_reads_file(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp, "rules.json")
            p.write_text(json.dumps({"rules": [CRASH_LOOP_RULE]}))
            rules = rete.load_rules(p)
            self.assertEqual(len(rules), 1)
            r = rete.run_events(rules, loop_events())
            self.assertGreater(r["n_fired"], 0)


if __name__ == "__main__":
    unittest.main()
