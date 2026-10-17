"""Hub routing and JSON bridge: the assistant's public surface."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from assistant import hub
from assistant.capabilities.brain import bridge


class HubTests(unittest.TestCase):
    def test_help_lists_every_route(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(hub.main(["--help"]), 0)
        for name in ("diagnose", "ask", "search", "issue", "settings", "brain", "api"):
            self.assertIn(name, out.getvalue())

    def test_unknown_command_is_free_text_not_an_error(self):
        # Phase 1 routing fix: a first token that matches no verb is no
        # longer a hard exit-2 error — the whole argv routes as free text
        # through the cortex pipeline (one-shot, read-only).
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(hub.main(["make", "my", "bar", "thinner"]), 0)
        self.assertIn("[cortex]", out.getvalue())

    def test_typo_verb_gets_did_you_mean_not_a_guess(self):
        # A token within edit distance 2 of exactly one verb gets the
        # settings layer's correction prompt on stderr — never executed.
        err = io.StringIO()
        out = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
            self.assertEqual(hub.main(["chatt"]), 1)
        self.assertIn("did you mean: chat (distance 1)", err.getvalue())
        self.assertNotIn("[cortex]", out.getvalue())

    def test_ambiguous_typo_does_not_guess(self):
        # "brif" is below the suggestion min length (the router's own
        # guard): short tokens never get suggestions, so it falls through
        # to the free-text path rather than a guess.
        out = io.StringIO()
        with contextlib.redirect_stderr(io.StringIO()), \
                contextlib.redirect_stdout(out):
            self.assertEqual(hub.main(["brif"]), 0)
        self.assertIn("[cortex]", out.getvalue())

    def test_suggest_verb_tie_abstains(self):
        # Two verbs equally close to the token: the suggestion abstains
        # (the router's unique-correction rule) — never a coin flip.
        saved = dict(hub.ROUTES)
        try:
            hub.ROUTES["alpha"] = (lambda _argv=None: 0, False)
            hub.ROUTES["alpaa"] = (lambda _argv=None: 0, False)
            self.assertIsNone(hub.suggest_verb("alpja"))  # distance 1 to both
        finally:
            hub.ROUTES.clear()
            hub.ROUTES.update(saved)

    def test_bare_help_still_prints_usage(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(hub.main(["help"]), 0)
        self.assertIn("single entry point", out.getvalue())

    def test_help_with_more_words_is_free_text(self):
        # "help me make my bar thinner" is a request, not a usage request.
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(hub.main(["help", "me", "make", "my", "bar",
                                       "thinner"]), 0)
        self.assertIn("[cortex]", out.getvalue())

    def test_capabilities_card_lists_manifest(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(hub.main(["capabilities"]), 0)
        card = out.getvalue()
        self.assertIn("capability manifest", card)
        self.assertIn("package_audit", card)
        self.assertIn("edit it to flip", card)

    def test_selfcheck_routes_to_diagnostics(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(hub.main(["selfcheck"]), 0)
        self.assertIn("selfcheck OK", out.getvalue())

    def test_brain_routes_to_cli(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            with contextlib.redirect_stdout(io.StringIO()) as out:
                code = hub.main(["brain", "--state", str(d / "s.json"),
                                 "--ledger", str(d / "l.json"), "ledger", "list"])
            self.assertEqual(code, 0)
            self.assertIn("no pending proposals", out.getvalue())


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.state = str(self.d / "s.json")
        self.ledger = str(self.d / "l.json")

    def tearDown(self):
        self.tmp.cleanup()

    def call(self, request):
        return bridge.handle(request, state_path=self.state, ledger_path=self.ledger)

    def test_unknown_op_and_bad_shape(self):
        self.assertFalse(self.call({"op": "explode"})["ok"])
        self.assertFalse(self.call(["not", "a", "dict"])["ok"])

    def test_missing_field_is_an_error_not_a_crash(self):
        r = self.call({"op": "forecast"})
        self.assertFalse(r["ok"])
        self.assertIn("KeyError", r["error"])

    def test_propose_then_list_then_decide_roundtrip(self):
        registry = [{"id": "bar.thickness", "description": "bar height thickness"}]
        r = self.call({"op": "placement_propose", "text": "make bar thinner",
                       "registry": registry, "propose": True})
        self.assertTrue(r["ok"])
        pending = self.call({"op": "ledger_list"})["result"]
        self.assertEqual(len(pending), 1)
        self.assertTrue(self.call({"op": "ledger_decide", "id": pending[0]["id"],
                                   "approve": True})["ok"])
        self.assertEqual(self.call({"op": "ledger_list"})["result"], [])

    def test_genius_decide_op_ranks_options(self):
        r = self.call({"op": "genius_decide",
                       "matrix": [[8, 256], [6, 512]],
                       "labels": ["air", "pro"],
                       "criteria": ["battery", "storage"],
                       "weights": [0.5, 0.5], "method": "topsis"})
        self.assertTrue(r["ok"])
        self.assertEqual(r["result"]["winner"], "pro")
        self.assertEqual(r["result"]["ranking"], ["pro", "air"])

    def test_genius_decide_op_defaults_and_errors(self):
        # omitted weights -> equal weighting; omitted method -> wsm
        r = self.call({"op": "genius_decide", "matrix": [[8], [6]],
                       "labels": ["air", "pro"], "criteria": ["battery"]})
        self.assertTrue(r["ok"])
        self.assertEqual(r["result"]["winner"], "air")
        # unknown method -> honest error inside the result, never a crash
        r = self.call({"op": "genius_decide", "matrix": [[1]], "labels": ["a"],
                       "criteria": ["x"], "method": "vibes"})
        self.assertTrue(r["ok"])
        self.assertIn("unknown method", r["result"]["error"])
        # missing required field -> ok:false error envelope
        self.assertFalse(self.call({"op": "genius_decide"})["ok"])

    def test_stdin_main_returns_exit_codes(self):
        out = io.StringIO()
        code = bridge.main(["--state", self.state, "--ledger", self.ledger],
                           stdin_text=json.dumps({"op": "ledger_list"}), out=out)
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(out.getvalue())["ok"])
        out = io.StringIO()
        self.assertEqual(bridge.main([], stdin_text="not json", out=out), 1)


if __name__ == "__main__":
    unittest.main()
