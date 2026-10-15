"""Tests for diagnostics/argumentation.py — Dung abstract argumentation.

The contract under test (Dung 1995, "On the Acceptability of Arguments and
its Fundamental Role in Nonmonotonic Reasoning, Logic Programming and
n-Person Games", Artificial Intelligence 77(2), 321-357):
- the classic examples, pinned to their exact Dung values (each hand-
  verified against the definitions; the directed 3-cycle note below
  explains why its unique preferred extension is the EMPTY set);
- admissible(S) is exactly conflict-free + every attacker attacked by S;
- grounded = least fixed point of the characteristic function from ∅;
- preferred = maximal admissible sets by enumeration, refused above 20
  arguments with the stated reason (honesty over completeness);
- structural attack detection: enable vs disable on the SAME unit fires;
  unrelated commands and opposing verbs on DIFFERENT units do not; a
  rule's own reinstall never attacks itself; the REAL rules.d corpus is
  conflict-free under the detector (pinned — the shipped
  CONTRADICTIONS table is empty for exactly this reason);
- resolve(): the winner's defense set is rendered; a symmetric 2-rule
  fight and an odd directed cycle are UNDECIDED and the module ABSTAINS
  ("undecided between N admissible positions", never a silent pick);
- determinism: two runs are byte-identical.
"""

from __future__ import annotations

import json
import unittest

from assistant.capabilities.diagnostics import argumentation as arg
from assistant.capabilities.diagnostics import engine


def _fake_rule(rule_id, command, title=None):
    """A minimal engine-shaped rule with exactly one command fix step."""
    return {
        "id": rule_id,
        "title": title or f"fake rule {rule_id}",
        "category": "runtime-shell",
        "severity": "low",
        "confidence": "probable",
        "matchers": {},
        "fix": [{"text": f"step of {rule_id}", "command": command,
                 "command_risk": "STATE_CHANGING"}],
        "references": [],
    }


def _report(*rules):
    return {"verdict": "AMBIGUOUS", "margin": 0,
            "candidates": [{"rule": rule, "score": 5} for rule in rules]}


class DungClassicTests(unittest.TestCase):
    """The textbook frameworks, values verified against the definitions."""

    def test_two_cycle_grounded_empty_two_preferred(self) -> None:
        # a <-> b: neither defends itself; grounded is empty; the two
        # maximal admissible sets are the singletons.
        af = arg.ArgumentationFramework(["a", "b"], [("a", "b"), ("b", "a")])
        self.assertEqual(af.grounded_extension(), [])
        self.assertEqual(af.preferred_extensions(), [["a"], ["b"]])
        self.assertTrue(af.admissible(["a"]))
        self.assertTrue(af.admissible(["b"]))
        self.assertFalse(af.admissible(["a", "b"]))  # not conflict-free

    def test_two_cycle_plus_unattacked_c_grounded_c(self) -> None:
        # a <-> b plus unattacked c: c is accepted; both {a,c} and {b,c}
        # are maximal admissible (each singleton side is defended by
        # itself against the other's attack... no: each is defended only
        # together with nothing else needed — {a,c}: a's attacker b is
        # attacked by a. Hand-verified.)
        af = arg.ArgumentationFramework(
            ["a", "b", "c"], [("a", "b"), ("b", "a")])
        self.assertEqual(af.grounded_extension(), ["c"])
        self.assertEqual(af.preferred_extensions(), [["a", "c"], ["b", "c"]])

    def test_directed_three_cycle(self) -> None:
        # a -> b -> c -> a (attacks). Every argument is attacked, so
        # F(∅) = ∅ and the grounded extension is empty. The preferred
        # extensions: NO non-empty subset is admissible — {a}'s attacker
        # is c, and a attacks b, not c, so a does not defend itself; by
        # symmetry no singleton works, and every pair contains an attack.
        # Hence the unique maximal admissible set is the EMPTY set (this
        # is the classic AF with no stable extension; the three
        # singletons are its maximal CONFLICT-FREE sets, not admissible
        # ones — pinned here explicitly so the distinction is recorded).
        af = arg.ArgumentationFramework(
            ["a", "b", "c"], [("a", "b"), ("b", "c"), ("c", "a")])
        self.assertEqual(af.grounded_extension(), [])
        self.assertEqual(af.preferred_extensions(), [[]])
        for single in ("a", "b", "c"):
            self.assertFalse(af.admissible([single]),
                             msg=f"{{{single}}} must not be admissible: its "
                                 "attacker is not attacked by itself")
        self.assertTrue(af.is_conflict_free(["a"]))
        self.assertTrue(af.is_conflict_free(["b"]))
        self.assertTrue(af.is_conflict_free(["c"]))

    def test_mutual_triangle_three_preferred_extensions(self) -> None:
        # The classic "three preferred extensions" framework: every pair
        # conflicts mutually (a<->b, b<->c, c<->a). Each singleton defends
        # itself (it attacks both of its attackers), so the preferred
        # extensions are exactly the three singletons; all arguments are
        # attacked, so the grounded extension is empty.
        attacks = [("a", "b"), ("b", "a"), ("b", "c"), ("c", "b"),
                   ("c", "a"), ("a", "c")]
        af = arg.ArgumentationFramework(["a", "b", "c"], attacks)
        self.assertEqual(af.grounded_extension(), [])
        self.assertEqual(af.preferred_extensions(), [["a"], ["b"], ["c"]])

    def test_self_attacker_never_in_any_extension(self) -> None:
        af = arg.ArgumentationFramework(["a", "b"], [("a", "a")])
        self.assertFalse(af.is_conflict_free(["a"]))  # self-attack counts
        self.assertFalse(af.admissible(["a"]))
        self.assertFalse(af.admissible(["a", "b"]))
        self.assertEqual(af.grounded_extension(), ["b"])
        self.assertEqual(af.preferred_extensions(), [["b"]])

    def test_one_way_attack_accepts_attacker(self) -> None:
        af = arg.ArgumentationFramework(["a", "b"], [("a", "b")])
        self.assertEqual(af.grounded_extension(), ["a"])
        self.assertEqual(af.preferred_extensions(), [["a"]])
        self.assertTrue(af.admissible(["a"]))
        self.assertFalse(af.admissible(["b"]))  # b's attacker is unopposed

    def test_characteristic_function_steps(self) -> None:
        # The pure attack CHAIN a->b->c->d->e: the grounded extension grows
        # by two links per round (an argument joins only when its attacker
        # is attacked by a member, so membership propagates along the
        # even-distance parity class):
        #   F(empty)  = {a}          (a is the only unattacked argument)
        #   F({a})    = {a, c}       (a defends c by attacking c's attacker b)
        #   F({a, c}) = {a, c, e}    (c defends e by attacking e's attacker d)
        #   F stable at {a, c, e}    (b and d never join: their attackers a
        #                              and c are never attacked by anyone)
        # Hand-derived per Dung 1995's Definition 5; pinned exactly.
        af = arg.ArgumentationFramework(
            ["a", "b", "c", "d", "e"],
            [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")])
        self.assertEqual(af.characteristic([]), ["a"])
        self.assertEqual(af.characteristic(["a"]), ["a", "c"])
        self.assertEqual(af.characteristic(["a", "c"]), ["a", "c", "e"])
        self.assertEqual(af.grounded_extension(), ["a", "c", "e"])
        # The never-joining odd-parity members are pinned too.
        self.assertFalse(af.admissible(["a", "b"]))
        self.assertFalse(af.admissible(["a", "c", "d"]))

    def test_validation_rejects_unknown_and_duplicate(self) -> None:
        with self.assertRaises(ValueError):
            arg.ArgumentationFramework(["a"], [("a", "z")])
        with self.assertRaises(ValueError):
            arg.ArgumentationFramework(["a", "a"], [])
        with self.assertRaises(ValueError):
            arg.ArgumentationFramework(["a"], []).admissible(["z"])
        with self.assertRaises(ValueError):
            arg.ArgumentationFramework(["a"], [("a", "a"), ("a", "a")])  # ok to dupe? no: pair shape only


class StructuralParserTests(unittest.TestCase):
    def test_systemctl_opposing_verbs_same_unit(self) -> None:
        self.assertEqual(
            arg.parse_command_ops(
                "systemctl --user disable --now kde-material-you-colors"),
            [{"verb": "disable", "unit": "kde-material-you-colors",
              "direction": "down"}])
        self.assertEqual(
            arg.parse_command_ops(
                "sudo systemctl enable --now bluetooth.service"),
            [{"verb": "enable", "unit": "bluetooth", "direction": "up"}])
        self.assertEqual(
            arg.parse_command_ops(
                "systemctl --user restart caelestia-shell.service"),
            [{"verb": "restart", "unit": "caelestia-shell",
              "direction": "up"}])

    def test_queries_and_probes_produce_no_ops(self) -> None:
        for command in (
            "systemctl --user status caelestia-shell.service",
            "systemctl --user is-enabled caelestia-shell.service",
            "systemctl show user@$(id -u).service -p MemoryPeak",
            "pgrep -af 'setup.sh|update.sh'",
            "bash scripts/10-autostart.sh",
            "cat /tmp/caelestia_build.log | tail -n 100",
            "QML2_IMPORT_PATH=$HOME/.local/lib/qt6/qml quickshell -d -n -p "
            "~/.config/quickshell/caelestia/shell.qml",
        ):
            self.assertEqual(arg.parse_command_ops(command), [],
                             msg=command)

    def test_package_managers_install_remove(self) -> None:
        self.assertEqual(
            arg.parse_command_ops("sudo pacman -S --needed base-devel cmake"),
            [{"verb": "install", "unit": "base-devel", "direction": "up"},
             {"verb": "install", "unit": "cmake", "direction": "up"}])
        # -Ss is a SEARCH, not an install; -Qs is a query.
        self.assertEqual(arg.parse_command_ops("pacman -Qs quickshell"), [])
        self.assertEqual(arg.parse_command_ops("pacman -Ss quickshell"), [])
        # A reinstall is two segments: remove then install.
        self.assertEqual(
            arg.parse_command_ops(
                "sudo pacman -R quickshell-git && sudo pacman -S quickshell-git"),
            [{"verb": "remove", "unit": "quickshell-git", "direction": "down"},
             {"verb": "install", "unit": "quickshell-git", "direction": "up"}])
        self.assertEqual(
            arg.parse_command_ops("sudo dnf install gcc-c++ cmake make"),
            [{"verb": "install", "unit": "gcc-c++", "direction": "up"},
             {"verb": "install", "unit": "cmake", "direction": "up"},
             {"verb": "install", "unit": "make", "direction": "up"}])

    def test_rm_paths_are_remove_ops(self) -> None:
        self.assertEqual(
            arg.parse_command_ops("rm -rf ~/.cache/caelestia/imagecache"),
            [{"verb": "remove", "unit": "~/.cache/caelestia/imagecache",
              "direction": "down"}])


class StructuralAttackTests(unittest.TestCase):
    def test_enable_vs_disable_same_unit_fires_both_ways(self) -> None:
        arguments = [
            {"id": "T-up", "title": None, "fix_summary": "",
             "fix_commands": ["systemctl --user enable --now demo.service"]},
            {"id": "T-down", "title": None, "fix_summary": "",
             "fix_commands": ["systemctl --user disable --now demo"]},
        ]
        attacks = arg.structural_attacks(arguments)
        self.assertEqual([(a["attacker"], a["defender"]) for a in attacks],
                         [("T-up", "T-down"), ("T-down", "T-up")])
        self.assertEqual(attacks[0]["source"], "structural")
        self.assertEqual(attacks[0]["unit"], "demo")  # .service normalized
        self.assertIn("enable", attacks[0]["verbs"])
        self.assertIn("disable", attacks[0]["verbs"])
        self.assertIn("'demo'", attacks[0]["reason"])

    def test_opposing_verbs_on_different_units_do_not_fire(self) -> None:
        arguments = [
            {"id": "T-up", "title": None, "fix_summary": "",
             "fix_commands": ["systemctl --user enable --now alpha.service"]},
            {"id": "T-down", "title": None, "fix_summary": "",
             "fix_commands": ["systemctl --user disable --now beta.service"]},
        ]
        self.assertEqual(arg.structural_attacks(arguments), [])

    def test_unrelated_commands_do_not_fire(self) -> None:
        arguments = [
            {"id": "T-a", "title": None, "fix_summary": "",
             "fix_commands": ["rm -rf ~/.cache/caelestia/imagecache"]},
            {"id": "T-b", "title": None, "fix_summary": "",
             "fix_commands": ["systemctl --user restart caelestia-shell.service"]},
        ]
        self.assertEqual(arg.structural_attacks(arguments), [])

    def test_same_direction_verbs_do_not_fire(self) -> None:
        # restart and start both aim the unit UP: compatible, no attack.
        arguments = [
            {"id": "T-restart", "title": None, "fix_summary": "",
             "fix_commands": ["systemctl --user restart demo.service"]},
            {"id": "T-start", "title": None, "fix_summary": "",
             "fix_commands": ["systemctl --user start demo.service"]},
        ]
        self.assertEqual(arg.structural_attacks(arguments), [])

    def test_install_vs_remove_same_package_fires(self) -> None:
        arguments = [
            {"id": "T-inst", "title": None, "fix_summary": "",
             "fix_commands": ["sudo pacman -S matugen"]},
            {"id": "T-rem", "title": None, "fix_summary": "",
             "fix_commands": ["sudo pacman -R matugen"]},
        ]
        attacks = arg.structural_attacks(arguments)
        self.assertEqual(len(attacks), 2)
        self.assertEqual(attacks[0]["unit"], "matugen")

    def test_reinstall_inside_one_rule_never_self_attacks(self) -> None:
        # Attacks are cross-rule: a rule's own remove+install pair is a
        # reinstall, not a self-contradiction.
        arguments = [
            {"id": "T-reinstall", "title": None, "fix_summary": "",
             "fix_commands": [
                 "sudo pacman -R quickshell-git && sudo pacman -S quickshell-git"]},
        ]
        self.assertEqual(arg.structural_attacks(arguments), [])

    def test_real_rules_corpus_is_structurally_conflict_free(self) -> None:
        # The honesty pin behind the empty CONTRADICTIONS table: NO pair
        # of shipped rules has opposing-verb commands on the same unit.
        # If this ever fails, a real conflict shipped — register the pair
        # in CONTRADICTIONS with a reason + citation (module docstring).
        arguments = [
            {"id": rule["id"], "title": rule["title"], "fix_summary": "",
             "fix_commands": [step["command"] for step in rule["fix"]
                              if isinstance(step, dict) and "command" in step]}
            for rule in engine.load_rules()
        ]
        self.assertGreaterEqual(len(arguments), 20)
        self.assertEqual(arg.structural_attacks(arguments), [])
        self.assertEqual(arg.CONTRADICTIONS, ())


class ResolveTests(unittest.TestCase):
    def test_no_conflict_report_has_no_verdict_to_render(self) -> None:
        report = _report(_fake_rule("T-a", "cat /tmp/log"),
                         _fake_rule("T-b", "ls /tmp"))
        verdict = arg.resolve(report)
        self.assertEqual(verdict["status"], "no-conflict")
        self.assertEqual(verdict["grounded"], ["T-a", "T-b"])
        self.assertEqual(verdict["winners"], [])
        self.assertEqual(verdict["losers"], [])
        self.assertIsNone(arg.conflict_resolution(report))

    def test_two_rule_symmetric_conflict_abstains(self) -> None:
        # A pure enable-vs-disable fight: each side rebuts the other, so
        # NOTHING defends either — the grounded extension is empty and
        # the honest verdict is UNDECIDED between the two admissible
        # positions. The module abstains; it never silently picks.
        report = _report(
            _fake_rule("T-enable", "systemctl --user enable --now demo.service"),
            _fake_rule("T-disable", "systemctl --user disable --now demo.service"))
        verdict = arg.resolve(report)
        self.assertEqual(verdict["status"], "undecided")
        self.assertTrue(verdict["abstained"])
        self.assertEqual(verdict["grounded"], [])
        self.assertEqual(verdict["winners"], [])
        self.assertEqual(verdict["losers"], [])
        self.assertEqual(verdict["undecided"]["positions"],
                         [["T-enable"], ["T-disable"]])
        self.assertIn("undecided between 2 admissible positions",
                      verdict["undecided"]["note"])
        self.assertIn("abstaining", verdict["undecided"]["note"])
        statuses = {a["id"]: a["status"] for a in verdict["arguments"]}
        self.assertEqual(statuses, {"T-enable": "undecided",
                                    "T-disable": "undecided"})

    def test_winner_with_defense_set_rendered(self) -> None:
        # A fake 2-rule conflicting report where one rule's fix DEFEATS
        # the other's one-way (a table-registered directed defeat — the
        # only shape that can produce a Dung winner, since structural
        # opposition is symmetric and can only end undecided). The winner
        # carries its defense set: which attacker it defeats and how.
        r1 = _fake_rule("T-handoff",
                        "kwriteconfig6 --file kwinrc --key autoHandoff true",
                        title="Enable the automatic scheme hand-off")
        r2 = _fake_rule("T-legacy", "bash scripts/10-autostart.sh",
                        title="Re-run the legacy autostart step")
        table = [{"attacker": "T-handoff", "defender": "T-legacy",
                  "reason": "the documented hand-off supersedes the legacy "
                            "autostart path (test fixture)",
                  "citation": "test fixture (synthetic)"}]
        report = _report(r1, r2)
        verdict = arg.resolve(report, contradictions=table)
        self.assertEqual(verdict["status"], "resolved")
        self.assertFalse(verdict["abstained"])
        self.assertEqual(verdict["grounded"], ["T-handoff"])
        self.assertEqual(verdict["winners"], [{
            "id": "T-handoff",
            "title": "Enable the automatic scheme hand-off",
            "fix_summary": "step of T-handoff",
            "defense": [{"defeated": "T-legacy",
                         "how": "the documented hand-off supersedes the "
                                "legacy autostart path (test fixture)"}],
        }])
        self.assertEqual(verdict["losers"], [{
            "id": "T-legacy",
            "title": "Re-run the legacy autostart step",
            "fix_summary": "step of T-legacy",
            "undefeated_attackers": [{
                "id": "T-handoff",
                "how": "the documented hand-off supersedes the legacy "
                       "autostart path (test fixture)"}],
        }])
        rendered = "\n".join(arg.render_conflict_lines(verdict))
        self.assertIn("Accepted (winner) T-handoff", rendered)
        self.assertIn("defeats T-legacy", rendered)
        self.assertIn("Rejected (loser) T-legacy", rendered)
        self.assertIn("defeated by T-handoff", rendered)

    def test_mutual_table_pair_is_undecided_between_two(self) -> None:
        rules = [_fake_rule("T-x", "cat /tmp/a"), _fake_rule("T-y", "cat /tmp/b")]
        table = [{"rules": ["T-x", "T-y"],
                  "reason": "synthetic mutual contradiction",
                  "citation": "test fixture (synthetic)"}]
        verdict = arg.resolve(_report(*rules), contradictions=table)
        self.assertEqual(verdict["status"], "undecided")
        self.assertTrue(verdict["abstained"])
        self.assertEqual(verdict["undecided"]["positions"],
                         [["T-x"], ["T-y"]])

    def test_odd_directed_cycle_abstains(self) -> None:
        # a -> b -> c -> a via table entries: grounded is empty, the only
        # admissible position is the empty set, and the module says so.
        rules = [_fake_rule("T-ca", "cat /tmp/a"),
                 _fake_rule("T-cb", "cat /tmp/b"),
                 _fake_rule("T-cc", "cat /tmp/c")]
        table = [
            {"attacker": "T-ca", "defender": "T-cb",
             "reason": "synthetic cycle edge", "citation": "test fixture"},
            {"attacker": "T-cb", "defender": "T-cc",
             "reason": "synthetic cycle edge", "citation": "test fixture"},
            {"attacker": "T-cc", "defender": "T-ca",
             "reason": "synthetic cycle edge", "citation": "test fixture"},
        ]
        verdict = arg.resolve(_report(*rules), contradictions=table)
        self.assertEqual(verdict["status"], "undecided")
        self.assertTrue(verdict["abstained"])
        self.assertEqual(verdict["grounded"], [])
        # The unique maximal admissible set is the empty set (see the
        # directed-3-cycle Dung test for the math); the note must say it.
        self.assertEqual(verdict["undecided"]["positions"], [[]])
        self.assertIn("undecided between 1 admissible position",
                      verdict["undecided"]["note"])
        self.assertIn("the empty set", verdict["undecided"]["note"])

    def test_gt_20_arguments_refuses_enumeration(self) -> None:
        rules = [_fake_rule(f"T-r{i:02d}", "cat /tmp/x") for i in range(19)]
        rules += [
            _fake_rule("T-ca", "systemctl --user enable --now demo.service"),
            _fake_rule("T-cb", "systemctl --user disable --now demo.service"),
        ]
        af = arg.ArgumentationFramework([r["id"] for r in rules], [])
        with self.assertRaises(ValueError) as caught:
            af.preferred_extensions()
        self.assertIn("refused", str(caught.exception))
        self.assertIn("exponential", str(caught.exception))

        # resolve() with the same 21-argument report: still a verdict, the
        # grounded part standing, the enumeration refused with a reason.
        verdict = arg.resolve(_report(*rules))
        self.assertEqual(verdict["status"], "undecided")
        self.assertTrue(verdict["abstained"])
        self.assertIsNone(verdict["undecided"]["positions"])
        self.assertIn("enumeration was refused", verdict["undecided"]["note"])
        self.assertIn("grounded verdict", verdict["undecided"]["note"])

    def test_preferred_enumeration_at_the_limit_still_works(self) -> None:
        # Exactly 20 arguments: enumeration allowed (and fast — no
        # attacks means the single preferred extension is everything).
        rules = [_fake_rule(f"T-r{i:02d}", "cat /tmp/x") for i in range(20)]
        verdict = arg.resolve(_report(*rules))
        self.assertEqual(verdict["status"], "no-conflict")
        af = arg.ArgumentationFramework([r["id"] for r in rules], [])
        self.assertEqual(af.preferred_extensions(),
                         [[r["id"] for r in rules]])

    def test_arguments_from_report_carries_title_and_fix(self) -> None:
        rule = _fake_rule("T-carrier", "systemctl --user start demo.service",
                          title="Start the demo unit")
        rule["fix"].append({"text": "second step", "command": "cat /tmp/y"})
        arguments = arg.arguments_from_report(_report(rule))
        self.assertEqual(len(arguments), 1)
        argument = arguments[0]
        self.assertEqual(argument["id"], "T-carrier")
        self.assertEqual(argument["title"], "Start the demo unit")
        self.assertEqual(argument["fix_summary"],
                         "step of T-carrier | second step")
        self.assertEqual(argument["fix_commands"],
                         ["systemctl --user start demo.service",
                          "cat /tmp/y"])

    def test_determinism_two_runs_byte_identical(self) -> None:
        rules = [
            _fake_rule("T-enable", "systemctl --user enable --now demo.service"),
            _fake_rule("T-disable", "systemctl --user disable --now demo.service"),
            _fake_rule("T-pacify", "cat /tmp/innocent"),
        ]
        report = _report(*rules)

        def snapshot() -> str:
            return json.dumps(arg.resolve(report), sort_keys=True)

        self.assertEqual(snapshot(), snapshot())

    def test_table_entry_validation(self) -> None:
        rules = [_fake_rule("T-x", "cat /tmp/a"), _fake_rule("T-y", "cat /tmp/b")]
        for bad in (
            {"rules": ["T-x"], "reason": "r", "citation": "c"},        # not a pair
            {"rules": ["T-x", "T-x"], "reason": "r", "citation": "c"},  # same id twice
            {"rules": ["T-x", "T-y"], "reason": "", "citation": "c"},   # empty reason
            {"rules": ["T-x", "T-y"], "reason": "r"},                   # no citation
            {"rules": ["T-x", "T-y"], "reason": "r", "citation": "c",
             "extra": 1},                                               # unknown key
            {"attacker": "T-x", "defender": "T-x",
             "reason": "r", "citation": "c"},                           # self-pair
        ):
            with self.assertRaises(ValueError, msg=repr(bad)):
                arg.resolve(_report(*rules), contradictions=[bad])


class CliEnrichmentTests(unittest.TestCase):
    """The read-only conflict-resolution wiring in engine.render_report
    (fake reports: the real corpus has no conflicts, so the section fires
    only through a registered table or a future conflicting pair)."""

    def test_render_report_shows_conflict_section_for_table_pair(self) -> None:
        r1 = _fake_rule("T-handoff", "kwriteconfig6 --file kwinrc --key a true",
                        title="Enable the automatic scheme hand-off")
        r2 = _fake_rule("T-legacy", "bash scripts/10-autostart.sh",
                        title="Re-run the legacy autostart step")
        report = _report(r1, r2)
        table = [{"attacker": "T-handoff", "defender": "T-legacy",
                  "reason": "supersession (test fixture)",
                  "citation": "test fixture (synthetic)"}]
        # render_report calls conflict_resolution with the SHIPPED (empty)
        # table; the wiring is exercised by patching the module constant
        # the way a future real table would be consulted.
        from unittest import mock
        with mock.patch("assistant.capabilities.diagnostics.argumentation.CONTRADICTIONS",
                        table):
            rendered = engine.render_report(report)
        self.assertIn("Conflict resolution (Dung 1995", rendered)
        self.assertIn("Accepted (winner) T-handoff", rendered)
        self.assertIn("defeats T-legacy", rendered)
        self.assertIn("Rejected (loser) T-legacy", rendered)
        # The engine's own matching output above the section is untouched.
        self.assertIn("Verdict: AMBIGUOUS", rendered)
        self.assertIn("[1] Rule T-handoff", rendered)
        self.assertIn("[2] Rule T-legacy", rendered)

    def test_render_report_shows_nothing_without_conflicts(self) -> None:
        rendered = engine.render_report(engine.diagnose(
            "bluetooth adapter turns on then turns off; nothing pairs"))
        self.assertNotIn("Conflict resolution", rendered)
        # And a NO_MATCH report renders neither enrichment section.
        no_match = engine.render_report(
            engine.diagnose("my cat sat on the keyboard"))
        self.assertNotIn("Conflict resolution", no_match)
        self.assertNotIn("Derived evidence", no_match)


if __name__ == "__main__":
    unittest.main()
