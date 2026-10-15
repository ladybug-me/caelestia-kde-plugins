"""Genius integration tests — hub wiring, bridge ops, cortex delegation, policy.

In-process by design: the stdlib-only discipline applies to tests as well, so these
drive the hub exactly the way brain/tests/test_wiring.py does — by calling
hub.main() with captured stdout, never by spawning processes.
"""
import contextlib
import io
import json
import unittest
from pathlib import Path


def _hub(args) -> tuple:
    """Run the hub in-process; returns (exit_code, stdout, stderr)."""
    from assistant import hub
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = hub.main(list(args))
        except SystemExit as exc:  # argparse --help path
            code = int(exc.code or 0)
    return code, out.getvalue(), err.getvalue()


def _bridge(request) -> dict:
    from assistant.capabilities.brain import bridge
    return bridge.handle(request)


class TestHubWiring(unittest.TestCase):
    def test_do_routes_math(self):
        code, out, _ = _hub(["do", "what is 15% of 80", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["domain"], "math_eval")
        self.assertEqual(payload["result"]["value"], 12.0)

    def test_genius_direct_subcommand(self):
        code, out, _ = _hub(["genius", "math", "2^10 + sqrt(144)", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["value"], 1036.0)

    def test_hub_help_lists_do_and_genius(self):
        code, out, _ = _hub(["--help"])
        self.assertEqual(code, 0)
        self.assertIn("do", out)
        self.assertIn("genius", out)

    def test_solve_via_hub(self):
        code, out, _ = _hub(["do", "solve x^2 - 2 = 0", "--json"])
        payload = json.loads(out)
        self.assertEqual(payload["domain"], "solve_equation")
        # F8 (D6): polynomial equations solve for ALL roots — both ±√2,
        # exact surd forms, Sturm count, exhaustive.
        roots = payload["result"]["roots"]
        values = sorted(round(r["value"], 5) for r in roots)
        self.assertEqual(len(values), 2)
        self.assertAlmostEqual(abs(values[0]), 1.4142135, places=4)
        self.assertAlmostEqual(abs(values[1]), 1.4142135, places=4)
        self.assertEqual(payload["result"]["real_roots_count"], 2)
        self.assertTrue(payload["result"]["exhaustive"])


class TestBridgeOps(unittest.TestCase):
    def test_genius_do(self):
        r = _bridge({"op": "genius_do", "text": "what is 10% of 50"})
        self.assertTrue(r["ok"])
        self.assertEqual(r["result"]["result"]["value"], 5.0)

    def test_genius_sentiment(self):
        r = _bridge({"op": "genius_sentiment", "text": "terrible awful crash"})
        self.assertEqual(r["result"]["label"], "negative")

    def test_genius_logic(self):
        r = _bridge({"op": "genius_logic", "formula": "p or not p"})
        self.assertEqual(r["result"]["classification"], "tautology")

    def test_genius_plan(self):
        r = _bridge({"op": "genius_plan", "goal": "fix the failing test"})
        self.assertEqual(r["result"].get("archetype", ""), "fix")

    def test_genius_stats(self):
        r = _bridge({"op": "genius_stats", "numbers": [1, 2, 3, 4]})
        self.assertEqual(r["result"]["n"], 4)

    def test_genius_palette(self):
        r = _bridge({"op": "genius_palette", "hex": "#3b7dd8", "harmony": "triadic"})
        self.assertEqual(len(r["result"]["swatches"]), 5)

    def test_genius_summarize(self):
        r = _bridge({"op": "genius_summarize",
                     "text": "Caelestia is a shell. It uses QML. The bar is nice.",
                     "query": "what is caelestia"})
        self.assertTrue(r["result"]["summary"])

    def test_genius_math_error_is_caught(self):
        r = _bridge({"op": "genius_math", "expr": "2 +"})
        self.assertIn("error", r["result"])

    def test_unknown_op_still_rejected(self):
        r = _bridge({"op": "genius_arbitrary", "fn": "os.system"})
        self.assertFalse(r["ok"])

    def test_bridge_genius_ops_cannot_execute_arbitrary_calls(self):
        # _genius_safe is keyed by an allow-list; an unknown dotted path fails
        out = _bridge({})  # sanity: bridge itself works
        self.assertFalse(out["ok"])
        from assistant.capabilities.brain import bridge
        result = bridge._genius_safe({}, "os.system", "rm -rf /")
        self.assertIn("error", result)


class TestCortexDelegation(unittest.TestCase):
    def test_route_delegates_math_to_genius(self):
        code, out, err = _hub(["route", "what is 15% of 80", "--json"])
        payload = json.loads(out)
        self.assertEqual(payload.get("delegate"), "genius")
        self.assertEqual(payload["genius"]["domain"], "math_eval")

    def test_settings_routing_unchanged(self):
        code, out, _ = _hub(["route", "make my bar thinner", "--json"])
        payload = json.loads(out)
        self.assertNotEqual(payload.get("delegate"), "genius")
        self.assertTrue(any("setBarScale" in json.dumps(c)
                            for c in payload.get("candidates", [])))

    def test_non_settings_why_question_no_longer_crashes(self):
        code, out, err = _hub(["route", "why do birds fly"])
        self.assertEqual(code, 0)
        self.assertNotIn("Traceback", out + err)

    def test_genius_cannot_be_smuggled_into_a_write(self):
        # a genius delegation is read-only: no plan, no ops, ever
        code, out, _ = _hub(["route", "what is 15% of 80", "--json"])
        payload = json.loads(out)
        genius = payload.get("genius", {})
        self.assertNotIn("plan", genius)
        self.assertNotIn("ops", genius)


class TestSafetyPolicy(unittest.TestCase):
    def test_genius_sources_have_no_executor_or_network(self):
        genius_dir = Path(__file__).resolve().parent.parent
        for py in sorted(genius_dir.glob("*.py")):
            src = py.read_text()
            for banned_import in ("subprocess", "socket", "urllib", "requests",
                                  "ctypes", "pty", "asyncio", "shutil"):
                self.assertNotIn(f"import {banned_import}", src,
                                 msg=f"{py.name} imports {banned_import}")
            for banned_attr in ("os.system", "os.popen", "os.exec", "os.spawn", "os.fork"):
                self.assertNotIn(banned_attr, src,
                                 msg=f"{py.name} references {banned_attr}")

    def test_bridge_ops_list_has_no_write_op(self):
        from assistant.capabilities.brain.bridge import OPS
        for name in OPS:
            if name.startswith("genius_"):
                self.assertIn(name, ("genius_do", "genius_learn", "genius_report",
                                     "genius_math", "genius_stats", "genius_logic",
                                     "genius_decide", "genius_palette",
                                     "genius_plan", "genius_graphs",
                                     "genius_units", "genius_fsbrain_summary",
                                     "genius_sentiment", "genius_summarize"))


class TestExistingSuitesStillGreen(unittest.TestCase):
    """The genius layer must not disturb any existing layer (sampled)."""

    def test_brain_wiring_still_passes(self):
        code, out, err = _hub(["brain", "ledger", "list"])
        self.assertEqual(code, 0)
        self.assertIn("no pending proposals", out)

    def test_settings_parser_still_parses(self):
        from assistant.capabilities.settings import parser
        result = parser.parse("make my bar thinner")
        verdict = result["verdict"] if isinstance(result, dict) else result.verdict
        self.assertEqual(verdict, "INTENT")

    def test_retrieval_search_still_works(self):
        from assistant.capabilities.retrieval import search
        hits = search.search("workspace pills not loading", k=2)
        self.assertTrue(hits)


if __name__ == "__main__":
    unittest.main()
