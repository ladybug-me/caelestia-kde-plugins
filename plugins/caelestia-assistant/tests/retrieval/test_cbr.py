"""Tests for retrieval/cbr.py — the Aamodt & Plaza CBR cycle.

The contract under test (Aamodt & Plaza 1994, "Case-Based Reasoning:
Foundational Issues, Methodological Variations, and System Approaches",
AI Communications 7(1), 39-59):
- RETRIEVE is the SHIPPED BM25 (search.py's Searcher over an in-memory
  index in indexer.build_index's shape) — the pinned scores below are
  hand-derived from the Okapi BM25 formula, not observed-then-pinned;
- REUSE adapts only through the explicit ADAPTATION_RULES table — every
  fired rule reports its id and its changes, nothing adapts silently,
  and an unregistered, unrepaired tool in the resolution ABSTAINS;
- REVISE marks failing steps with the INJECTED validator's own reason
  and applies nothing;
- RETAIN files EXACTLY ONE ledger proposal (kind="cbr_case") through a
  tmp Ledger and writes nothing outside the ledger's own path;
- the cycle abstains on empty retrieval (no invented cases) and is
  byte-deterministic across runs.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from assistant.capabilities.retrieval import cbr


def _case(cid, problem, resolution, accepted=True, source="issue #1"):
    return {"id": cid, "problem": problem, "resolution": resolution,
            "accepted": accepted, "source": source}


# A five-case base with deliberately DISJOINT vocabularies, so a query
# in one vocabulary can only ever retrieve that one case (the winner is
# unambiguous by construction, not by score arithmetic).
CASE_BASE = [
    _case("case-greeter",
          "greeter shows a black screen after boot and no sddm login prompt appears",
          "reinstall the sddm theme and re-enable the greeter service",
          source="issue #12"),
    _case("case-pacman",
          "pacman refuses to sync because a mirror returned an invalid package signature",
          "refresh the arch keyring then resynchronize the databases",
          source="issue #6"),
    _case("case-qml",
          "shell build fails with missing qml module metadata after a partial upgrade",
          "rebuild the shell plugin from a clean tree",
          source="issue #14"),
    _case("case-fprint",
          "lockscreen fingerprint reader rejects every finger after resume from suspend",
          "re-enroll the fingerprint profile in the settings daemon",
          accepted=False, source="issue #18"),
    _case("case-slideshow",
          "wallpaper slideshow skips images when the interval is set below one second",
          "raise the slideshow interval to at least two seconds",
          source="docs/TROUBLESHOOTING.md"),
]

# A tiny injected registry: only these tool names are "registered".
_FAKE_REGISTRY = {"setBlurEnabled", "setAnimationSpeed"}


def _fake_registry(name):
    return name in _FAKE_REGISTRY


class RetrieveTests(unittest.TestCase):
    def test_retrieve_picks_the_right_case(self) -> None:
        # Query vocabulary (wallpaper/slideshow/skips/interval) overlaps
        # ONLY case-slideshow's problem text; every other case scores 0.
        hits = cbr.retrieve(CASE_BASE, "the wallpaper slideshow skips "
                                       "pictures at small interval values", k=3)
        self.assertEqual([h["case_id"] for h in hits], ["case-slideshow"])
        self.assertTrue(hits[0]["score"] > 0.0)
        self.assertEqual(hits[0]["case"]["source"], "docs/TROUBLESHOOTING.md")

    def test_retrieve_bm25_scores_are_hand_derived(self) -> None:
        # Hand-derived from Okapi BM25 with k1=1.5, b=0.75 (indexer.K1/B):
        #   base: A problem "alpha beta" (dl=2), B problem "alpha" (dl=1)
        #   n_docs=2, avgdl=1.5, df(alpha)=2, df(beta)=1
        #   idf(alpha)=ln(1+(2-2+0.5)/(2+0.5))=ln(1.2)
        #   idf(beta) =ln(1+(2-1+0.5)/(1+0.5))=ln(2)
        #   norm(A)=1-b+b*(2/1.5)=1.25, tf-term(A)=2.5/(1+1.5*1.25)=0.869565
        #   score(A)=(ln1.2+ln2)*0.869565=0.761277 -> 0.7613
        #   norm(B)=1-b+b*(1/1.5)=0.75,  tf-term(B)=2.5/(1+1.5*0.75)=1.176471
        #   score(B)=ln(1.2)*1.176471=0.214496     -> 0.2145
        base = [_case("A", "alpha beta", "fix alpha"),
                _case("B", "alpha", "fix beta")]
        hits = cbr.retrieve(base, "alpha beta", k=2)
        self.assertEqual([h["case_id"] for h in hits], ["A", "B"])
        self.assertEqual([h["score"] for h in hits], [0.7613, 0.2145])
        # Deterministic: two runs agree to the byte.
        again = cbr.retrieve(base, "alpha beta", k=2)
        self.assertEqual(json.dumps(hits, sort_keys=True),
                         json.dumps(again, sort_keys=True))

    def test_empty_case_base_is_an_honest_empty_answer(self) -> None:
        self.assertEqual(cbr.retrieve([], "anything"), [])
        self.assertEqual(cbr.retrieve(CASE_BASE, "zzzqqq blorptastic"), [])

    def test_malformed_cases_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            cbr.retrieve([{"id": "x"}], "query")  # missing keys
        with self.assertRaises(ValueError):
            cbr.retrieve([_case("x", "p", "r", accepted="yes")], "query")
        with self.assertRaises(ValueError):
            cbr.retrieve(["not a case"], "query")

    def test_duplicate_case_ids_are_rejected(self) -> None:
        base = [_case("dup", "alpha", "r1"), _case("dup", "beta", "r2")]
        with self.assertRaises(ValueError):
            cbr.retrieve(base, "query")


class ReuseTests(unittest.TestCase):
    def test_verbatim_when_no_rule_applies(self) -> None:
        case = _case("c1", "problem text", "step one\nstep two")
        result = cbr.reuse(case, "query", tool_registry=_fake_registry)
        self.assertEqual(result["status"], "VERBATIM")
        self.assertEqual(result["resolution"], "step one\nstep two")
        self.assertEqual(result["rules_fired"], [])
        self.assertIsNone(result["abstain_reason"])

    def test_numeric_rescale_fires_with_rule_id_and_changes(self) -> None:
        case = _case("c2", "problem",
                     "set the threshold to 100 and the timeout to 200 seconds")
        result = cbr.reuse(case, "query", numeric_baseline=(100, 125),
                           tool_registry=_fake_registry)
        self.assertEqual(result["status"], "ADAPTED")
        self.assertEqual(result["resolution"],
                         "set the threshold to 125 and the timeout to 250 seconds")
        self.assertEqual(len(result["rules_fired"]), 1)
        rule = result["rules_fired"][0]
        self.assertEqual(rule["id"], "adapt-numeric-rescale")
        self.assertEqual(rule["ratio"], 1.25)
        self.assertEqual(rule["changes"],
                         [{"before": "100", "after": "125"},
                          {"before": "200", "after": "250"}])
        # the rule table's own documentation travels with the firing
        self.assertIn("parameter adjustment", rule["rationale"])

    def test_identity_baseline_does_not_fire(self) -> None:
        case = _case("c2", "problem", "raise the value to 100")
        result = cbr.reuse(case, "query", numeric_baseline=(100, 100),
                           tool_registry=_fake_registry)
        self.assertEqual(result["status"], "VERBATIM")  # a no-op is not an adaptation
        self.assertEqual(result["rules_fired"], [])

    def test_numeric_rescale_leaves_glued_numbers_alone(self) -> None:
        # Deliberately literal (documented): EVERY bare numeric literal
        # scales — including the space-separated version "6.9", which
        # the module has no way to recognize as a version without
        # guessing semantics (and every replacement is listed loudly,
        # which is the honest remedy). Letter-glued ("mpv4") and
        # comma-grouped ("1,000") digits are not bare literals and stay.
        case = _case("c3", "problem",
                     "use Qt 6.9 and pass 1,000 files through mpv4, set it to 20")
        result = cbr.reuse(case, "query", numeric_baseline=(20, 40),
                           tool_registry=_fake_registry)
        self.assertEqual(result["resolution"],
                         "use Qt 13.8 and pass 1,000 files through mpv4, "
                         "set it to 40")
        rule = result["rules_fired"][0]
        self.assertEqual(rule["changes"],
                         [{"before": "6.9", "after": "13.8"},
                          {"before": "20", "after": "40"}])

    def test_drop_stale_step_fires_with_rule_id(self) -> None:
        case = _case("c4", "problem",
                     "refresh the pacman keyring first\nthen rerun the installer")
        result = cbr.reuse(case, "query", stale_premises=["pacman"],
                           tool_registry=_fake_registry)
        self.assertEqual(result["status"], "ADAPTED")
        self.assertEqual(result["resolution"], "then rerun the installer")
        rule = result["rules_fired"][0]
        self.assertEqual(rule["id"], "adapt-drop-stale-step")
        self.assertEqual(rule["changes"],
                         [{"line": 0, "premise": "pacman",
                           "text": "refresh the pacman keyring first"}])

    def test_tool_substitution_mechanism_fires_with_rule_id(self) -> None:
        case = _case("c5", "problem", "call setOldTool to disable blur")
        result = cbr.reuse(case, "query",
                           tool_renames={"setOldTool": "setBlurEnabled"},
                           tool_registry=_fake_registry)
        self.assertEqual(result["status"], "ADAPTED")
        self.assertEqual(result["resolution"], "call setBlurEnabled to disable blur")
        rule = result["rules_fired"][0]
        self.assertEqual(rule["id"], "adapt-tool-substitution")
        self.assertEqual(rule["changes"],
                         [{"line": 0, "before": "setOldTool",
                           "after": "setBlurEnabled"}])

    def test_shipped_tool_renames_table_is_intentionally_empty(self) -> None:
        # Honesty pin (the T1 CONTRADICTIONS pattern): a full read of
        # assistant/settings/tools.json found NO rename-shaped pair —
        # the registry carries only current names with citations and no
        # history of old ones. The mechanism ships; the table stays
        # empty rather than inventing a mapping.
        self.assertEqual(cbr.TOOL_RENAMES, ())
        self.assertEqual({rule["id"] for rule in cbr.ADAPTATION_RULES},
                         {"adapt-tool-substitution", "adapt-drop-stale-step",
                          "adapt-numeric-rescale"})

    def test_abstains_on_unregistered_tool(self) -> None:
        case = _case("c6", "problem", "call setNoSuchTool and reboot")
        result = cbr.reuse(case, "query", tool_registry=_fake_registry)
        self.assertEqual(result["status"], "ABSTAINED")
        self.assertIsNone(result["resolution"])
        self.assertEqual(result["rules_fired"], [])
        self.assertIn("setNoSuchTool", result["abstain_reason"])
        self.assertIn("cannot transfer", result["abstain_reason"])

    def test_substitution_repairs_an_unregistered_name(self) -> None:
        # The old name is unregistered, but a rename maps it onto a
        # registered one: the transfer goes through.
        case = _case("c7", "problem", "call setOldTool to disable blur")
        result = cbr.reuse(case, "query",
                           tool_renames={"setOldTool": "setBlurEnabled"},
                           tool_registry=_fake_registry)
        self.assertEqual(result["status"], "ADAPTED")
        self.assertIn("setBlurEnabled", result["resolution"])

    def test_abstains_when_every_step_is_stale(self) -> None:
        case = _case("c8", "problem",
                     "refresh the pacman keyring first\nthen rerun the installer")
        result = cbr.reuse(case, "query",
                           stale_premises=["pacman", "installer"],
                           tool_registry=_fake_registry)
        self.assertEqual(result["status"], "ABSTAINED")
        self.assertIsNone(result["resolution"])
        self.assertIn("stale", result["abstain_reason"])

    def test_unaccepted_outcome_is_flagged_not_hidden(self) -> None:
        case = _case("c9", "problem", "try re-enrolling the fingerprint",
                     accepted=False)
        result = cbr.reuse(case, "query", tool_registry=_fake_registry)
        self.assertEqual(result["status"], "VERBATIM")
        self.assertIn("NOT accepted", result["note"])

    def test_default_registry_is_the_real_settings_registry(self) -> None:
        # No injected registry: the lazy default resolves through the
        # settings layer's own loader over the committed tools.json.
        ok = cbr.reuse(_case("c10", "p", "call setBlurEnabled to disable blur"),
                       "query")
        self.assertEqual(ok["status"], "VERBATIM")
        bad = cbr.reuse(_case("c11", "p", "call setZzzNotARealToolEver"), "query")
        self.assertEqual(bad["status"], "ABSTAINED")

    def test_bad_adaptation_context_is_rejected(self) -> None:
        case = _case("c1", "problem", "resolution")
        with self.assertRaises(ValueError):
            cbr.reuse(case, "query", numeric_baseline=(0, 10))  # zero baseline
        with self.assertRaises(ValueError):
            cbr.reuse(case, "query", stale_premises=[""])  # empty premise
        with self.assertRaises(ValueError):
            cbr.reuse(case, "query", tool_renames={"setX": "setX"})  # no-op rename


class ReviseTests(unittest.TestCase):
    ADAPTED = {"status": "VERBATIM", "case_id": "c1",
               "resolution": "refresh the keyring first\nthen rerun the installer"}

    def test_marks_failing_steps_with_the_validators_own_reason(self) -> None:
        def validator(step):
            if "installer" in step:
                return (False, "the installer script was renamed upstream")
            return (True, None)

        result = cbr.revise(self.ADAPTED, validator)
        self.assertEqual(result["status"], "REVISE")
        self.assertEqual(result["failed_steps"], [2])
        self.assertEqual(result["steps"][0]["passed"], True)
        self.assertEqual(result["steps"][1]["passed"], False)
        self.assertEqual(result["steps"][1]["reason"],
                         "the installer script was renamed upstream")
        self.assertFalse(result["applied"])  # the module never applies anything

    def test_all_passing_steps_are_confirmed(self) -> None:
        result = cbr.revise(self.ADAPTED, lambda step: (True, None))
        self.assertEqual(result["status"], "CONFIRMED")
        self.assertEqual(result["failed_steps"], [])
        self.assertEqual(result["n_steps"], 2)

    def test_abstained_reuse_has_nothing_to_revise(self) -> None:
        with self.assertRaises(ValueError):
            cbr.revise({"status": "ABSTAINED", "resolution": None}, lambda s: (True, None))

    def test_validator_protocol_is_enforced(self) -> None:
        with self.assertRaises(ValueError):
            cbr.revise(self.ADAPTED, lambda step: True)  # not a (bool, reason) tuple
        with self.assertRaises(ValueError):
            cbr.revise(self.ADAPTED, lambda step: ("yes", "why"))  # not a bool
        with self.assertRaises(ValueError):
            cbr.revise(self.ADAPTED, "not callable")


class RetainTests(unittest.TestCase):
    def test_files_exactly_one_ledger_proposal_and_nothing_else(self) -> None:
        new_case = _case("case-new", "new problem", "new resolution",
                         source="session 2026-09-26")
        with tempfile.TemporaryDirectory() as tmp:
            from assistant.capabilities.brain.ledger import Ledger

            ledger_path = Path(tmp) / "ledger.json"
            ledger = Ledger(ledger_path)
            result = cbr.retain(CASE_BASE, new_case, ledger)

            self.assertEqual(result["status"], "PROPOSED")
            self.assertEqual(result["kind"], "cbr_case")
            self.assertEqual(result["target"], "case-new")
            self.assertFalse(result["appended"])
            pending = ledger.pending()
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0]["kind"], "cbr_case")
            self.assertEqual(pending[0]["target"], "case-new")
            self.assertEqual(pending[0]["status"], "pending")
            # the diff is the CANONICAL case text (json.dumps, sort_keys)
            self.assertEqual(pending[0]["diff"],
                             json.dumps(new_case, sort_keys=True, ensure_ascii=False))
            # the case base itself is untouched — retain never appends
            self.assertNotIn(new_case, CASE_BASE)
            # nothing was written outside the ledger's own path
            self.assertEqual(sorted(os.listdir(tmp)), ["ledger.json"])

    def test_duplicate_case_id_is_rejected(self) -> None:
        from assistant.capabilities.brain.ledger import Ledger

        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(Path(tmp) / "ledger.json")
            with self.assertRaises(ValueError):
                cbr.retain(CASE_BASE, CASE_BASE[0], ledger)  # id already present
            self.assertEqual(ledger.pending(), [])  # nothing was filed

    def test_confidence_defaults_are_conservative_and_overrideable(self) -> None:
        from assistant.capabilities.brain.ledger import Ledger

        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(Path(tmp) / "ledger.json")
            accepted = cbr.retain([], _case("a", "p", "r", accepted=True), ledger)
            self.assertEqual(accepted["confidence"], 0.8)
            rejected = cbr.retain([], _case("b", "p", "r", accepted=False), ledger)
            self.assertEqual(rejected["confidence"], 0.3)
            explicit = cbr.retain([], _case("c", "p", "r"), ledger, confidence=0.55)
            self.assertEqual(explicit["confidence"], 0.55)
            with self.assertRaises(ValueError):
                cbr.retain([], _case("d", "p", "r"), ledger, confidence=1.5)
            with self.assertRaises(ValueError):
                cbr.retain([], _case("e", "p", "r"), ledger, confidence="high")

    def test_retain_needs_a_real_ledger(self) -> None:
        with self.assertRaises(ValueError):
            cbr.retain([], _case("a", "p", "r"), ledger=None)


class CycleTests(unittest.TestCase):
    def _validator(self, step):
        return (False, "always fails in this test") if "keyring" in step else (True, None)

    def test_cycle_runs_retrieve_reuse_revise(self) -> None:
        result = cbr.cbr_cycle(CASE_BASE, "pacman refuses to sync, invalid "
                                          "signature from a mirror",
                               validate_fn=self._validator,
                               tool_registry=_fake_registry)
        self.assertEqual(result["status"], "VERBATIM")
        self.assertEqual(result["case_id"], "case-pacman")
        self.assertEqual(result["reuse"]["case_id"], "case-pacman")
        self.assertEqual(result["revise"]["status"], "REVISE")
        self.assertEqual(result["revise"]["failed_steps"], [1])
        self.assertEqual(result["retain"]["status"], "NOT_RUN")
        self.assertIn("ledger proposal", result["retain"]["reason"])

    def test_cycle_without_validator_skips_revise_honestly(self) -> None:
        result = cbr.cbr_cycle(CASE_BASE, "pacman invalid signature",
                               tool_registry=_fake_registry)
        self.assertEqual(result["revise"]["status"], "NOT_RUN")
        self.assertIn("validator", result["revise"]["reason"])

    def test_cycle_abstains_on_empty_retrieval(self) -> None:
        result = cbr.cbr_cycle(CASE_BASE, "zzzqqq blorptastic frobnicate",
                               tool_registry=_fake_registry)
        self.assertEqual(result["status"], "ABSTAINED")
        self.assertEqual(result["retrieve"], [])
        self.assertIsNone(result["reuse"])
        self.assertIsNone(result["revise"])
        self.assertIsNone(result["retain"])
        self.assertIn("no case", result["abstain_reason"])

    def test_cycle_retains_only_when_given_case_and_ledger(self) -> None:
        from assistant.capabilities.brain.ledger import Ledger

        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(Path(tmp) / "ledger.json")
            new_case = _case("fresh-case", "fresh problem", "fresh resolution")
            result = cbr.cbr_cycle(CASE_BASE, "wallpaper slideshow skips images",
                                   tool_registry=_fake_registry,
                                   new_case=new_case, ledger=ledger)
            self.assertEqual(result["retain"]["status"], "PROPOSED")
            self.assertEqual(len(ledger.pending()), 1)
            # one-sided retain is a caller error, not a silent skip
            with self.assertRaises(ValueError):
                cbr.cbr_cycle(CASE_BASE, "wallpaper slideshow skips images",
                              tool_registry=_fake_registry, new_case=new_case)

    def test_cycle_is_byte_deterministic(self) -> None:
        first = json.dumps(cbr.cbr_cycle(CASE_BASE, "pacman invalid signature",
                                         validate_fn=self._validator,
                                         tool_registry=_fake_registry),
                           sort_keys=True)
        second = json.dumps(cbr.cbr_cycle(CASE_BASE, "pacman invalid signature",
                                          validate_fn=self._validator,
                                          tool_registry=_fake_registry),
                            sort_keys=True)
        self.assertEqual(first, second)


class CliSurfaceTests(unittest.TestCase):
    """The read-only CLI surfaces (minimal wiring; no ledger is touched)."""

    def test_cbr_cli_over_a_case_file(self) -> None:
        import contextlib
        import io

        from assistant.capabilities.retrieval import cli

        with tempfile.TemporaryDirectory() as tmp:
            cases_path = Path(tmp) / "cases.json"
            cases_path.write_text(json.dumps(CASE_BASE), encoding="utf-8")
            for argv, expect_status in (
                (["cbr", "--cases", str(cases_path),
                  "the wallpaper slideshow skips pictures"], "VERBATIM"),
                (["cbr", "--cases", str(cases_path), "--json",
                  "the wallpaper slideshow skips pictures"], "VERBATIM"),
                (["cbr", "--cases", str(cases_path), "zzzqqq blorptastic"],
                 "ABSTAINED"),
            ):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    code = cli.main(argv)
                self.assertEqual(code, 0)
                output = stdout.getvalue()
                if "--json" in argv:
                    self.assertEqual(json.loads(output)["status"], expect_status)
                else:
                    self.assertIn(expect_status, output)
            # a broken case file is a clean error, not a traceback
            bad = Path(tmp) / "bad.json"
            bad.write_text('{"not": "a list"}', encoding="utf-8")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = cli.main(["cbr", "--cases", str(bad), "anything"])
            self.assertEqual(code, 2)
            self.assertIn("error:", stderr.getvalue())

    def test_search_cli_still_runs(self) -> None:
        # Regression pin: the search/stats path referenced an undefined
        # index_path (latent NameError, unexercised by any test) — fixed
        # in the same change that added the cbr/analog surfaces.
        import contextlib
        import io

        from assistant.capabilities.retrieval import cli

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = cli.main(["search", "build fails cmake", "-k", "2"])
        self.assertEqual(code, 0)
        self.assertIn("score", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
