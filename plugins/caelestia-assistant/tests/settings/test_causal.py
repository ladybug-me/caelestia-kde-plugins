"""Tests for assistant.capabilities.settings.causal (exponential-build-5 F14).

Covers: the backward why-chain over ARMED curated edges (with citations,
confidence, arming conditions and honest terminals), the forward
counterfactual (consequences.project reuse + the unevaluated-edge
honesty layer), every abstention verdict, the CLI surfaces, and the
`why` engine's causal lines.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from io import StringIO
from contextlib import redirect_stdout
from pathlib import Path

from assistant.capabilities.settings import causal


FORCED = {"appearance": {"transparency": {"enabled": False},
                         "blur": False}}
UNFORCED = {"appearance": {"transparency": {"enabled": True},
                           "blur": True}}
FULL = {"appearance": {"transparency": {"enabled": False}, "blur": False},
        "bar": {"persistent": True, "dodgeWindows": True, "scale": 0.7},
        "general": {"borderThickness": 6}}


class WhyChainTests(unittest.TestCase):
    def test_armed_chain_with_citation(self) -> None:
        r = causal.why_chain("appearance.blur", FORCED)
        self.assertEqual(r["verdict"], "OK")
        self.assertEqual(len(r["chains"]), 1)
        chain = r["chains"][0]
        hop = chain["hops"][0]
        self.assertEqual(hop["from"], "appearance.transparency.enabled")
        self.assertEqual(hop["confidence"], "high")
        self.assertIn("AppearancePage.qml", hop["citation"])
        self.assertTrue(hop["claimed_content"])
        self.assertEqual(chain["terminal"], "DIRECT")
        self.assertIn("set directly", chain["terminal_detail"])

    def test_not_armed_is_no_curated_cause(self) -> None:
        r = causal.why_chain("appearance.blur", UNFORCED)
        self.assertEqual(r["verdict"], "NO_CURATED_CAUSE")
        self.assertNotIn("chains", r)

    def test_unknown_live_values_abstain_not_guess(self) -> None:
        r = causal.why_chain("appearance.blur", {})
        self.assertEqual(r["verdict"], "NO_CURATED_CAUSE")
        # the edge that WOULD explain it is listed as unevaluable with
        # the missing value named — never silently defaulted
        reasons = {u.get("reason") for u in r["unevaluated"]}
        self.assertTrue(any("transparency.enabled" in x
                            for x in reasons))

    def test_unknown_path(self) -> None:
        self.assertEqual(causal.why_chain("no.such.path")["verdict"],
                         "UNKNOWN_PATH")

    def test_registry_key_outside_the_dag(self) -> None:
        r = causal.why_chain("services.clockFormat")
        self.assertEqual(r["verdict"], "NO_CURATED_CAUSE")
        self.assertIn("no curated interaction touches it", r["note"])

    def test_tool_name_resolves_to_path(self) -> None:
        r = causal.why_chain("setBlurEnabled", FORCED)
        self.assertEqual(r["path"], "appearance.blur")
        self.assertEqual(r["verdict"], "OK")

    def test_effect_value_mismatch_is_not_an_explanation(self) -> None:
        # transparency off but blur ON: the forcing edge explains a PAST
        # state, not this one — must not be reported as armed
        state = {"appearance": {"transparency": {"enabled": False},
                                "blur": True}}
        r = causal.why_chain("appearance.blur", state)
        self.assertEqual(r["verdict"], "NO_CURATED_CAUSE")

    def test_bounded_depth_refuses_not_truncates(self) -> None:
        r = causal.why_chain("appearance.blur", FORCED, max_depth=0)
        # depth 0 cannot even start: no chain, honest verdict
        self.assertEqual(r["verdict"], "NO_CURATED_CAUSE")

    def test_chains_note_says_cited_code_not_nature(self) -> None:
        r = causal.why_chain("appearance.blur", FORCED)
        self.assertIn("not causes in nature", r["note"])


class CounterfactualTests(unittest.TestCase):
    def test_forward_projection_with_citation(self) -> None:
        r = causal.counterfactual("setTransparencyEnabled", False, FULL)
        self.assertEqual(r["verdict"], "OK")
        self.assertEqual(len(r["derived"]), 1)
        d = r["derived"][0]
        self.assertEqual(d["effect_path"], "appearance.blur")
        self.assertIn("AppearancePage.qml", d["citation"])

    def test_unevaluated_edges_listed_not_dropped(self) -> None:
        # FULL state knows every requires path: nothing unevaluated
        self.assertEqual(causal.counterfactual(
            "setTransparencyEnabled", False, FULL)["unevaluated"], [])
        # empty state: both requires-bearing edges abstain, loudly
        r = causal.counterfactual("setDodgeWindows", True, {})
        edges = {u["edge"] for u in r["unevaluated"]}
        self.assertIn("dodge-needs-persistent", edges)
        self.assertIn("blur-inert-without-transparency", edges)
        for u in r["unevaluated"]:
            self.assertIn("abstaining", u["reason"])

    def test_unknown_tool(self) -> None:
        self.assertEqual(causal.counterfactual("setNope", 1)["verdict"],
                         "UNKNOWN_TOOL")

    def test_unevaluated_edges_helper(self) -> None:
        self.assertEqual(causal.unevaluated_edges(FULL), [])
        unknown = causal.unevaluated_edges({})
        self.assertTrue(unknown)

    def test_renderers(self) -> None:
        lines = causal.render_chain(causal.why_chain("appearance.blur",
                                                     FORCED))
        self.assertTrue(any(l.startswith("causal: appearance.blur <-")
                            for l in lines))
        cf = causal.render_counterfactual(
            causal.counterfactual("setDodgeWindows", True, {}))
        self.assertTrue(any("unevaluated" in l for l in cf))


class CliTests(unittest.TestCase):
    def _run(self, argv, state=FORCED):
        from assistant.capabilities.settings import cli as settings_cli
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "shell.json"
            target.write_text(json.dumps(state), encoding="utf-8")
            out = StringIO()
            with redirect_stdout(out):
                rc = settings_cli.main([*argv, "--file", str(target)])
            return rc, out.getvalue()

    def test_why_chain_cli(self) -> None:
        rc, text = self._run(["--why-chain", "appearance.blur"])
        self.assertEqual(rc, 0)
        self.assertIn("causal: appearance.blur <-", text)
        self.assertIn("AppearancePage.qml", text)

    def test_why_chain_cli_abstains_nonzero(self) -> None:
        rc, text = self._run(["--why-chain", "appearance.blur"],
                             state=UNFORCED)
        self.assertEqual(rc, 1)
        self.assertIn("no armed curated interaction", text)

    def test_counterfactual_cli(self) -> None:
        rc, text = self._run(["--counterfactual",
                              "setTransparencyEnabled", "false"])
        self.assertEqual(rc, 0)
        self.assertIn("counterfactual: if setTransparencyEnabled", text)
        self.assertIn("unevaluated", text)  # FULL state? no: FORCED lacks
        # bar.persistent, so the dodge edge must abstain in the output
        self.assertIn("abstaining", text)

    def test_what_if_lists_unevaluated(self) -> None:
        rc, text = self._run(["--what-if", "gaming"])
        self.assertEqual(rc, 0)
        self.assertIn("unevaluated: edge dodge-needs-persistent", text)

    def test_what_if_write_nothing_contract_holds(self) -> None:
        from assistant.capabilities.settings import cli as settings_cli
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "shell.json"
            out = StringIO()
            with redirect_stdout(out):
                settings_cli.main(["--what-if", "gaming",
                                   "--file", str(target)])
            self.assertFalse(target.exists())  # never created


class WhyEngineTests(unittest.TestCase):
    def test_explain_settings_gains_causal_lines(self) -> None:
        from assistant.core.explain_unified import explain_settings
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "shell.json"
            target.write_text(json.dumps(FORCED), encoding="utf-8")
            result = explain_settings("setBlurEnabled", target)
        causal_lines = [l for l in result["lines"]
                        if l.startswith("causal:")]
        self.assertTrue(causal_lines)
        self.assertIn("appearance.transparency.enabled",
                      causal_lines[0])

    def test_explain_settings_honest_when_no_chain(self) -> None:
        from assistant.core.explain_unified import explain_settings
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "shell.json"
            target.write_text(json.dumps(UNFORCED), encoding="utf-8")
            result = explain_settings("setBlurEnabled", target)
        causal_lines = [l for l in result["lines"]
                        if l.startswith("causal:")]
        # NO chain -> the honest one-liner, never silence
        self.assertEqual(len(causal_lines), 1)
        self.assertIn("no armed curated interaction", causal_lines[0])


if __name__ == "__main__":
    unittest.main()
