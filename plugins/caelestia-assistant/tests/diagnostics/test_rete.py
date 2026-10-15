"""Tests for diagnostics/rete.py — the forward-chaining Rete network.

The contract under test (Forgy 1982, Artificial Intelligence 19, adapted
as the module docstring states):
- a hand-built 3-rule fixture derives a 2-level chain (derived facts feed
  further rules) with EXACT expected facts in EXACT derivation order and
  citation chains back to the given facts;
- termination on a deliberately cyclic rule pair (re-deriving an
  already-present fact is a no-op — the fixpoint guarantee);
- alpha-memory reuse: a duplicate assert does not re-fire any terminal,
  and a pattern shared by two rules is fed once (terminal log pinned);
- wildcard premise patterns match any value for their key — and only
  their key;
- explain() returns the full derivation tree;
- dormant-rule honesty: rules whose premises no producer supplies are
  listed with their first unsatisfied premise, never hidden;
- the shipped COMPOSITE_RULES reference only REAL rules.d rule ids and
  fire over a real engine diagnose() result (the CLI enrichment path);
- determinism: two independent runs are byte-identical;
- strict validation: malformed facts/rules are REJECTED with ValueError,
  never repaired.

All fixtures are self-contained (synthetic ids marked T-*); the two
real-corpus tests read the actual rules.d via engine.load_rules().
"""

from __future__ import annotations

import json
import unittest

from assistant.capabilities.diagnostics import engine
from assistant.capabilities.diagnostics.rete import (
    COMPOSITE_RULES,
    ReteNetwork,
    derive,
    derived_evidence,
    facts_from_diagnosis,
    render_derived_lines,
)


def _rule(rule_id, premises, conclusion, rationale=None, citation=None):
    return {
        "id": rule_id,
        "premises": premises,
        "conclusion": conclusion,
        "rationale": rationale or f"{rule_id} rationale (test fixture)",
        "citation": citation or f"{rule_id} citation (test fixture)",
    }


def _fact(type_, key, value):
    return {"type": type_, "key": key, "value": value}


# Three rules forming a 2-level chain: givens -> T-chain-1 -> (with a
# second given) T-chain-2 -> T-chain-3. Every citation below is the
# fixture's own marker; no claim about the shell is made.
CHAIN_RULES = [
    _rule(
        "T-chain-1",
        [{"type": "symptom", "key": "boot", "value": "fails"}],
        {"type": "derived", "key": "stage", "value": "boot-broken"}),
    _rule(
        "T-chain-2",
        [{"type": "derived", "key": "stage", "value": "boot-broken"},
         {"type": "symptom", "key": "unit", "value": "caelestia-shell"}],
        {"type": "derived", "key": "stage", "value": "shell-init-broken"}),
    _rule(
        "T-chain-3",
        [{"type": "derived", "key": "stage", "value": "shell-init-broken"}],
        {"type": "derived", "key": "plan", "value": "restart-unit"}),
]

GIVEN_BOOT = _fact("symptom", "boot", "fails")
GIVEN_UNIT = _fact("symptom", "unit", "caelestia-shell")


class ChainTests(unittest.TestCase):
    def test_two_level_chain_exact_facts_in_derivation_order(self) -> None:
        derived = derive([GIVEN_BOOT, GIVEN_UNIT], CHAIN_RULES)
        # Level 1: boot-broken (from the given symptom). Level 2:
        # shell-init-broken (from the level-1 fact + the second given).
        # Level 3: restart-unit (from the level-2 fact alone).
        self.assertEqual(derived, [
            {"type": "derived", "key": "stage", "value": "boot-broken"},
            {"type": "derived", "key": "stage", "value": "shell-init-broken"},
            {"type": "derived", "key": "plan", "value": "restart-unit"},
        ])

    def test_explain_tree_chains_rule_ids_back_to_given_facts(self) -> None:
        network = ReteNetwork(CHAIN_RULES)
        network.assert_fact(GIVEN_BOOT)
        network.assert_fact(GIVEN_UNIT)
        network.run()
        tree = network.explain({"type": "derived", "key": "plan",
                                "value": "restart-unit"})
        # The exact tree: rule chain T-chain-3 -> T-chain-2 -> T-chain-1,
        # leaves are the two given facts.
        self.assertEqual(tree["rule"], "T-chain-3")
        self.assertEqual(tree["citation"], "T-chain-3 citation (test fixture)")
        self.assertEqual(tree["rationale"], "T-chain-3 rationale (test fixture)")
        self.assertEqual(tree["source"], "derived")
        self.assertEqual(len(tree["premises"]), 1)
        mid = tree["premises"][0]
        self.assertEqual(mid["rule"], "T-chain-2")
        self.assertEqual(mid["fact"], {"type": "derived", "key": "stage",
                                       "value": "shell-init-broken"})
        # T-chain-2 consumed the level-1 derived fact AND the given unit
        # symptom (premise order preserved).
        self.assertEqual(len(mid["premises"]), 2)
        top, unit_leaf = mid["premises"]
        self.assertEqual(top["rule"], "T-chain-1")
        self.assertEqual(top["source"], "derived")
        self.assertEqual(top["premises"], [{
            "fact": GIVEN_BOOT, "source": "given", "rule": None,
            "rationale": None, "citation": None, "premises": []}])
        self.assertEqual(unit_leaf, {
            "fact": GIVEN_UNIT, "source": "given", "rule": None,
            "rationale": None, "citation": None, "premises": []})

    def test_explain_unknown_fact_is_rejected(self) -> None:
        network = ReteNetwork(CHAIN_RULES)
        with self.assertRaises(ValueError):
            network.explain(_fact("symptom", "never", "asserted"))

    def test_missing_middle_fact_makes_downstream_rules_dormant(self) -> None:
        # Only the boot symptom: level 1 fires, levels 2/3 stay dormant.
        network = ReteNetwork(CHAIN_RULES)
        network.assert_fact(GIVEN_BOOT)
        derived = network.run()
        self.assertEqual(derived, [
            {"type": "derived", "key": "stage", "value": "boot-broken"}])
        self.assertEqual(network.fired_rules(), ["T-chain-1"])
        dormant = network.dormant_rules()
        self.assertEqual([entry["rule"] for entry in dormant],
                         ["T-chain-2", "T-chain-3"])
        # Each dormant rule names the first premise nobody supplied.
        self.assertEqual(dormant[0]["unsatisfied_premise"],
                         {"type": "symptom", "key": "unit",
                          "value": "caelestia-shell"})
        self.assertEqual(dormant[1]["unsatisfied_premise"],
                         {"type": "derived", "key": "stage",
                          "value": "shell-init-broken"})


class TerminationTests(unittest.TestCase):
    def test_cyclic_rule_pair_reaches_fixpoint(self) -> None:
        # R1: A -> B; R2: B -> A. Assert A (given): R1 derives B; R2 then
        # re-derives A, which is already present -> no-op -> agenda drains.
        rules = [
            _rule("T-cyc-1", [{"type": "s", "key": "k", "value": "A"}],
                  {"type": "s", "key": "k", "value": "B"}),
            _rule("T-cyc-2", [{"type": "s", "key": "k", "value": "B"}],
                  {"type": "s", "key": "k", "value": "A"}),
        ]
        network = ReteNetwork(rules)
        self.assertTrue(network.assert_fact(_fact("s", "k", "A")))
        derived = network.run()
        self.assertEqual(derived, [_fact("s", "k", "B")])
        # The full working memory holds exactly the two facts.
        self.assertEqual(network.facts(), [_fact("s", "k", "A"),
                                           _fact("s", "k", "B")])
        # Both terminals fired exactly once; the cycle closed as a no-op.
        log = network.terminal_log()
        self.assertEqual([(entry["rule"], entry["outcome"]) for entry in log],
                         [("T-cyc-1", "derived"),
                          ("T-cyc-2", "already-present")])
        # Running again at fixpoint derives nothing new.
        self.assertEqual(network.run(), [])

    def test_diamond_derives_each_fact_once(self) -> None:
        # Two rules concluding the SAME fact: first derivation wins, the
        # second firing is recorded as already-present.
        rules = [
            _rule("T-dia-1", [{"type": "s", "key": "k", "value": "A"}],
                  {"type": "d", "key": "k", "value": "C"}),
            _rule("T-dia-2", [{"type": "s", "key": "k", "value": "A"},
                              {"type": "s", "key": "k", "value": "B"}],
                  {"type": "d", "key": "k", "value": "C"}),
        ]
        network = ReteNetwork(rules)
        network.assert_fact(_fact("s", "k", "A"))
        network.assert_fact(_fact("s", "k", "B"))
        self.assertEqual(network.run(), [_fact("d", "k", "C")])
        outcomes = sorted(entry["outcome"]
                          for entry in network.terminal_log())
        self.assertEqual(outcomes, ["already-present", "derived"])
        # explain() carries the FIRST derivation (T-dia-1), deterministic.
        self.assertEqual(
            network.explain(_fact("d", "k", "C"))["rule"], "T-dia-1")

    def test_given_fact_wins_over_later_derivation(self) -> None:
        rules = [
            _rule("T-pre-1", [{"type": "s", "key": "k", "value": "A"}],
                  {"type": "d", "key": "k", "value": "pre-existing"}),
        ]
        network = ReteNetwork(rules)
        network.assert_fact(_fact("d", "k", "pre-existing"))
        network.assert_fact(_fact("s", "k", "A"))
        self.assertEqual(network.run(), [])
        self.assertEqual(
            network.explain(_fact("d", "k", "pre-existing"))["source"],
            "given")


class AlphaMemoryTests(unittest.TestCase):
    def test_duplicate_assert_does_not_refire(self) -> None:
        rules = [
            _rule("T-alpha-1", [{"type": "symptom", "key": "boot",
                                 "value": "fails"}],
                  {"type": "derived", "key": "stage", "value": "broken"}),
        ]
        network = ReteNetwork(rules)
        self.assertTrue(network.assert_fact(GIVEN_BOOT))
        self.assertFalse(network.assert_fact(GIVEN_BOOT))  # no-op
        derived = network.run()
        self.assertEqual(derived, [
            {"type": "derived", "key": "stage", "value": "broken"}])
        # Pinned by the terminal log: exactly ONE firing, not two.
        self.assertEqual(len(network.terminal_log()), 1)
        self.assertEqual(network.fired_rules(), ["T-alpha-1"])

    def test_shared_pattern_fed_once_across_rules(self) -> None:
        # Two rules share the premise pattern (symptom, boot, fails); one
        # assert feeds the SHARED alpha node, so each rule fires exactly
        # once (the alpha-memory reuse property, observable via the log).
        shared = {"type": "symptom", "key": "boot", "value": "fails"}
        rules = [
            _rule("T-shared-1", [shared,
                                 {"type": "symptom", "key": "unit",
                                  "value": "caelestia-shell"}],
                  {"type": "derived", "key": "plan", "value": "p1"}),
            _rule("T-shared-2", [shared,
                                 {"type": "symptom", "key": "unit",
                                  "value": "other"}],
                  {"type": "derived", "key": "plan", "value": "p2"}),
        ]
        network = ReteNetwork(rules)
        network.assert_fact(GIVEN_BOOT)
        network.assert_fact(GIVEN_UNIT)
        network.assert_fact(_fact("symptom", "unit", "other"))
        derived = network.run()
        self.assertEqual(derived, [
            {"type": "derived", "key": "plan", "value": "p1"},
            {"type": "derived", "key": "plan", "value": "p2"}])
        self.assertEqual([(entry["rule"], entry["outcome"])
                          for entry in network.terminal_log()],
                         [("T-shared-1", "derived"),
                          ("T-shared-2", "derived")])
        # The shared pattern's alpha memory holds the fact exactly once.
        self.assertEqual(network.alpha_memory(shared), [GIVEN_BOOT])


class WildcardTests(unittest.TestCase):
    def test_wildcard_matches_any_value_for_that_key(self) -> None:
        rules = [
            _rule("T-wild-1",
                  [{"type": "symptom", "key": "unit", "value": "*"}],
                  {"type": "derived", "key": "plan", "value": "any-unit"}),
        ]
        for value in ("caelestia-shell", "plasma-kglobalaccel", "zzz"):
            network = ReteNetwork(rules)
            network.assert_fact(_fact("symptom", "unit", value))
            self.assertEqual(
                network.run(),
                [{"type": "derived", "key": "plan", "value": "any-unit"}],
                msg=f"wildcard must match value {value!r}")

    def test_wildcard_does_not_match_other_keys_or_types(self) -> None:
        rules = [
            _rule("T-wild-2",
                  [{"type": "symptom", "key": "unit", "value": "*"}],
                  {"type": "derived", "key": "plan", "value": "any-unit"}),
        ]
        for fact in (_fact("symptom", "service", "caelestia-shell"),
                     _fact("match", "unit", "caelestia-shell")):
            network = ReteNetwork(rules)
            network.assert_fact(fact)
            self.assertEqual(network.run(), [])

    def test_wildcard_premise_can_be_satisfied_by_a_derived_fact(self) -> None:
        rules = [
            _rule("T-wild-3",
                  [{"type": "s", "key": "k", "value": "A"}],
                  {"type": "d", "key": "k", "value": "X"}),
            _rule("T-wild-4",
                  [{"type": "d", "key": "k", "value": "*"}],
                  {"type": "d", "key": "plan", "value": "saw-a-derived"}),
        ]
        network = ReteNetwork(rules)
        network.assert_fact(_fact("s", "k", "A"))
        self.assertEqual(network.run(), [
            {"type": "d", "key": "k", "value": "X"},
            {"type": "d", "key": "plan", "value": "saw-a-derived"},
        ])


class HonestyTests(unittest.TestCase):
    def test_no_facts_means_no_derivations(self) -> None:
        self.assertEqual(derive([], CHAIN_RULES), [])
        network = ReteNetwork(CHAIN_RULES)
        self.assertEqual(network.run(), [])
        self.assertEqual([entry["rule"] for entry in network.dormant_rules()],
                         ["T-chain-1", "T-chain-2", "T-chain-3"])

    def test_unsatisfiable_premise_never_fires_and_is_listed(self) -> None:
        # A rule referencing a fact no producer supplies is dormant, and
        # the module says so instead of hiding the rule.
        orphan = _rule("T-orphan-1",
                       [{"type": "symptom", "key": "nobody",
                         "value": "supplies-this"}],
                       {"type": "derived", "key": "plan", "value": "never"})
        network = ReteNetwork(CHAIN_RULES + [orphan])
        network.assert_fact(GIVEN_BOOT)
        network.assert_fact(GIVEN_UNIT)
        network.run()
        self.assertNotIn("T-orphan-1", network.fired_rules())
        dormant = {entry["rule"]: entry
                   for entry in network.dormant_rules()}
        self.assertIn("T-orphan-1", dormant)
        self.assertEqual(dormant["T-orphan-1"]["unsatisfied_premise"],
                         {"type": "symptom", "key": "nobody",
                          "value": "supplies-this"})

    def test_empty_premise_list_is_rejected(self) -> None:
        # An unconditional "conclusion" is an invention, not a derivation.
        with self.assertRaises(ValueError):
            ReteNetwork([_rule("T-empty", [],
                               {"type": "d", "key": "k", "value": "v"})])


class ValidationTests(unittest.TestCase):
    def test_malformed_facts_rejected(self) -> None:
        network = ReteNetwork(CHAIN_RULES)
        for bad in (
            {"type": "s"},                                   # missing keys
            {"type": "s", "key": "k", "value": "v", "x": 1},  # extra key
            {"type": "s", "key": "k", "value": 5},            # non-string
            {"type": "", "key": "k", "value": "v"},           # empty string
            {"type": "s", "key": "k", "value": "*"},          # wildcard fact
            ["s", "k", "v"],                                  # not a dict
        ):
            with self.assertRaises(ValueError, msg=repr(bad)):
                network.assert_fact(bad)

    def test_malformed_rules_rejected(self) -> None:
        good_conclusion = {"type": "d", "key": "k", "value": "v"}
        good_premise = {"type": "s", "key": "k", "value": "v"}
        bad_rules = (
            {"id": "x", "premises": [good_premise], "conclusion": good_conclusion},  # missing rationale/citation
            _rule("x", [good_premise], good_conclusion) | {"extra": 1},              # unknown key
            _rule("x", [good_premise], {"type": "d", "key": "k", "value": "*"}),     # wildcard conclusion
            _rule("x", [good_premise], good_conclusion) | {"id": ""},                # empty id
        )
        for bad in bad_rules:
            with self.assertRaises(ValueError, msg=repr(bad)):
                ReteNetwork([bad])
        with self.assertRaises(ValueError):
            # wildcarded TYPE in a premise (only value may be wildcarded)
            ReteNetwork([_rule("x", [{"type": "*", "key": "k", "value": "v"}],
                               good_conclusion)])
        with self.assertRaises(ValueError):
            # duplicate rule ids
            ReteNetwork([_rule("dup", [good_premise], good_conclusion),
                         _rule("dup", [good_premise], good_conclusion)])


class DeterminismTests(unittest.TestCase):
    def test_two_runs_are_byte_identical(self) -> None:
        def snapshot() -> str:
            network = ReteNetwork(CHAIN_RULES)
            network.assert_fact(GIVEN_BOOT)
            network.assert_fact(GIVEN_UNIT)
            derived = network.run()
            return json.dumps({
                "derived": derived,
                "explain": [network.explain(f) for f in derived],
                "dormant": network.dormant_rules(),
                "log": network.terminal_log(),
            }, sort_keys=True)

        self.assertEqual(snapshot(), snapshot())


class CompositeRulesTests(unittest.TestCase):
    """The shipped rules must be grounded in the REAL rules.d corpus."""

    def setUp(self) -> None:
        self.real_ids = {rule["id"] for rule in engine.load_rules()}

    def test_every_match_premise_references_a_real_rule_id(self) -> None:
        for rule in COMPOSITE_RULES:
            for premise in rule["premises"]:
                if premise["type"] == "match" and premise["key"] == "rule":
                    self.assertIn(
                        premise["value"], self.real_ids,
                        msg=f"{rule['id']} premise references unknown rule "
                            f"{premise['value']!r}")
            self.assertEqual(rule["conclusion"]["type"], "derived")
            self.assertTrue(rule["rationale"])
            self.assertTrue(rule["citation"])

    def test_real_color_report_fires_the_two_level_chain(self) -> None:
        # Real engine input matching all three color rules at once.
        text = ("After every reboot the color variant falls back to Tonal "
                "Spot, and kde-material-you-colors keeps flashing the "
                "screen every second.")
        diagnosis = engine.diagnose(text)
        facts = facts_from_diagnosis(diagnosis)
        self.assertIn(_fact("match", "rule", "CL-config-kmycolors-001"),
                      facts)
        self.assertIn(_fact("match", "rule", "CL-config-schemereset-001"),
                      facts)
        derived = derive(facts)
        self.assertIn(
            {"type": "derived", "key": "combined-diagnosis",
             "value": "legacy-color-service-plus-missing-scheme-handoff"},
            derived)
        # The SECOND level: the derived diagnosis feeds the migration plan.
        self.assertIn(
            {"type": "derived", "key": "migration-plan",
             "value": "full-legacy-material-you-exit"},
            derived)

    def test_single_rule_report_derives_nothing(self) -> None:
        text = "my bluetooth adapter turns on then turns off and nothing pairs"
        diagnosis = engine.diagnose(text)
        self.assertIsNone(derived_evidence(diagnosis))


class CliEnrichmentTests(unittest.TestCase):
    """The read-only CLI wiring in engine.render_report / engine.main."""

    COLOR_TEXT = ("After every reboot the color variant falls back to Tonal "
                  "Spot, and kde-material-you-colors keeps flashing the "
                  "screen every second.")

    def test_render_report_shows_derived_evidence_when_it_fires(self) -> None:
        report = engine.render_report(engine.diagnose(self.COLOR_TEXT))
        self.assertIn("Derived evidence (Rete composite rules", report)
        self.assertIn("CR-color-war-001", report)
        self.assertIn("full-legacy-material-you-exit", report)
        # Dormant rules are listed, not hidden.
        self.assertIn("Dormant composite rules", report)
        self.assertIn("CR-shell-unit-convergence-001", report)

    def test_render_report_shows_nothing_when_nothing_derives(self) -> None:
        report = engine.render_report(
            engine.diagnose("bluetooth turns on then off, nothing pairs"))
        self.assertNotIn("Derived evidence", report)
        self.assertNotIn("Conflict resolution", report)
        no_match = engine.render_report(
            engine.diagnose("my cat sat on the keyboard"))
        self.assertNotIn("Derived evidence", no_match)

    def test_json_payload_carries_derived_evidence_only_when_fired(self) -> None:
        import io
        import tempfile
        from contextlib import redirect_stdout
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.txt"
            input_path.write_text(self.COLOR_TEXT, encoding="utf-8")
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = engine.main(["diagnose", str(input_path), "--json"])
            self.assertEqual(code, 0)
            payload = json.loads(buffer.getvalue())
        self.assertIn("derived_evidence", payload)
        derived = payload["derived_evidence"]["derived"]
        self.assertEqual([node["rule"] for node in derived],
                         ["CR-color-war-001", "CR-legacy-color-cleanup-001",
                          "CR-color-war-002"])

        # A single-rule report adds no enrichment keys at all.
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.txt"
            input_path.write_text(
                "bluetooth turns on then off, nothing pairs",
                encoding="utf-8")
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                engine.main(["diagnose", str(input_path), "--json"])
            payload = json.loads(buffer.getvalue())
        self.assertNotIn("derived_evidence", payload)
        self.assertNotIn("conflict_resolution", payload)

    def test_rendered_lines_mention_no_new_shell_claims(self) -> None:
        # The section renders derivations FROM the matched rules; the
        # citations it prints are the underlying rules' own references.
        payload = derived_evidence(engine.diagnose(self.COLOR_TEXT))
        self.assertIsNotNone(payload)
        lines = render_derived_lines(payload)
        text = "\n".join(lines)
        for rule_id in ("CL-config-kmycolors-001",
                        "CL-config-schemereset-001",
                        "CL-folder-cleanup-legacy-001"):
            self.assertIn(rule_id, text)
        self.assertIn("docs/TROUBLESHOOTING.md", text)
        self.assertIn("issue #14", text)
        self.assertIn("issue #763", text)


if __name__ == "__main__":
    unittest.main()
