"""Tests for retrieval/structure_mapping.py — Gentner structure-mapping.

The contract under test (Gentner 1983, "Structure-Mapping: A
Theoretical Framework for Analogy", Cognitive Science 7(2), 155-170;
the greedy search is a deliberately tiny relative of the
Structure-Mapping Engine — Falkenhainer, Forbus & Gentner 1989):
- THE CORE, pinned: two problems with ZERO token overlap but identical
  relational structure (5.0) outscore two problems with high token
  overlap but different structure (0.0) — surface features are
  discarded, entity/attribute matches contribute nothing by
  themselves;
- systematicity: a matched connected second-order AND/OR system beats
  the SAME first-order matches isolated (5.0 > 3.0);
- connectedness: matched relations sharing a corresponding entity
  score higher (2.5 > 2.0);
- the 64-relation cap rejects rather than degrades;
- rule_match_graph builds honest predicates from a REAL rules.d rule
  (nested allOf/anyOf included) and its self-match score is pinned;
- surface_candidates abstains (empty + reason) on thin structural
  overlap and carries the alongside-never-instead note;
- determinism: two runs byte-identical.
"""

from __future__ import annotations

import json
import unittest

from assistant.capabilities.retrieval import structure_mapping as sm


def _problem(name, entities, relations):
    return {"id": name, "entities": entities, "relations": relations}


# Zero token overlap between G1 and G2 (every entity, literal and
# pattern string disjoint), IDENTICAL relational structure.
G1 = _problem("g1", ["input-one", "file-one"], [
    {"name": "text_regex", "args": ["pattern-one", "input-one"]},
    {"name": "file_present", "args": ["file-one"]},
    {"name": "AND", "args": [0, 1]},
])
G2 = _problem("g2", ["input-two", "file-two"], [
    {"name": "text_regex", "args": ["pattern-two", "input-two"]},
    {"name": "file_present", "args": ["file-two"]},
    {"name": "AND", "args": [0, 1]},
])
# G3 shares G1's entities AND literal text ("pattern-one") — high token
# overlap — but its relational structure is different (different
# relation names, no second-order node).
G3 = _problem("g3", ["input-one", "file-one"], [
    {"name": "text_substring", "args": ["pattern-one", "input-one"]},
    {"name": "cmd_output", "args": ["pattern-one", "file-one"]},
])


class GentnerCoreTests(unittest.TestCase):
    """The theory's whole point, pinned to hand-derived scores."""

    def test_zero_overlap_same_structure_beats_high_overlap_different(self) -> None:
        same = sm.match(G1, G2)
        other = sm.match(G1, G3)
        # Hand-derived: 3 matched relations (text_regex, file_present,
        # AND) + 2 systematicity (both AND children matched) = 5.0.
        self.assertEqual(same["structural_score"], 5.0)
        # No relation name is shared -> no candidates -> 0.0, despite
        # identical entities and shared literal text.
        self.assertEqual(other["structural_score"], 0.0)
        self.assertGreater(same["structural_score"], other["structural_score"])
        self.assertEqual(same["entity_correspondence"],
                         {"input-one": "input-two", "file-one": "file-two"})
        self.assertEqual(len(same["matched_relations"]), 3)
        self.assertIn("no relational overlap", other["why"])

    def test_entity_identity_alone_scores_zero(self) -> None:
        # Identical entity names on both sides, disjoint relation names:
        # the correspondence never even starts.
        a = _problem("a", ["x", "y"], [
            {"name": "uses", "args": ["x", "y"]},
            {"name": "makes", "args": ["x", "y"]},
        ])
        b = _problem("b", ["x", "y"], [
            {"name": "gives", "args": ["x", "y"]},
            {"name": "takes", "args": ["x", "y"]},
        ])
        result = sm.match(a, b)
        self.assertEqual(result["structural_score"], 0.0)
        self.assertEqual(result["entity_correspondence"], {})
        # the contrast: ONE shared relation name IS a structural match
        c = _problem("c", ["u", "v"], [{"name": "uses", "args": ["u", "v"]}])
        self.assertEqual(sm.match(a, c)["structural_score"], 1.0)

    def test_second_order_match_requires_its_children(self) -> None:
        # AND matches AND only as a connected system: strip the
        # children's name match and the combinator never matches hollow.
        a = _problem("a", ["e1", "e2"], [
            {"name": "text_regex", "args": ["p", "e1"]},
            {"name": "file_present", "args": ["e2"]},
            {"name": "AND", "args": [0, 1]},
        ])
        b = _problem("b", ["f1", "f2"], [
            {"name": "text_regex", "args": ["q", "f1"]},
            {"name": "cmd_output", "args": ["r", "f2"]},  # file_present is gone
            {"name": "AND", "args": [0, 1]},
        ])
        result = sm.match(a, b)
        matched_names = {r["name"] for r in result["matched_relations"]}
        self.assertNotIn("AND", matched_names)
        self.assertIn("text_regex", matched_names)  # the isolated match survives
        self.assertEqual(result["structural_score"], 1.0)


class SystematicityTests(unittest.TestCase):
    def test_connected_second_order_beats_isolated_first_order(self) -> None:
        # Same NUMBER of matched relations (3 vs 3); the systematic
        # version wraps two of them in a matched AND system.
        sys_a = _problem("sys-a", ["e1", "e2"], [
            {"name": "text_regex", "args": ["p", "e1"]},
            {"name": "file_present", "args": ["e2"]},
            {"name": "AND", "args": [0, 1]},
        ])
        sys_b = _problem("sys-b", ["f1", "f2"], [
            {"name": "text_regex", "args": ["q", "f1"]},
            {"name": "file_present", "args": ["f2"]},
            {"name": "AND", "args": [0, 1]},
        ])
        iso_a = _problem("iso-a", ["g1", "g2", "g3"], [
            {"name": "text_regex", "args": ["q", "g1"]},
            {"name": "file_present", "args": ["g2"]},
            {"name": "cmd_output", "args": ["r", "g3"]},
        ])
        iso_b = _problem("iso-b", ["h1", "h2", "h3"], [
            {"name": "text_regex", "args": ["q2", "h1"]},
            {"name": "file_present", "args": ["h2"]},
            {"name": "cmd_output", "args": ["r2", "h3"]},
        ])
        systematic = sm.match(sys_a, sys_b)["structural_score"]
        isolated = sm.match(iso_a, iso_b)["structural_score"]
        # Hand-derived: 3 + 2 systematicity = 5.0 vs 3 flat (the three
        # isolated relations share no entities -> no connectedness).
        self.assertEqual(systematic, 5.0)
        self.assertEqual(isolated, 3.0)
        self.assertGreater(systematic, isolated)

    def test_connectedness_shared_entities_score_higher(self) -> None:
        # feeds(a,b) + feeds(b,c): one shared corresponding entity (b)
        # bridges the two matched relations -> 2.0 + 0.5.
        chained_a = _problem("cha", ["a", "b", "c"], [
            {"name": "feeds", "args": ["a", "b"]},
            {"name": "feeds", "args": ["b", "c"]},
        ])
        chained_b = _problem("chb", ["p", "q", "r"], [
            {"name": "feeds", "args": ["p", "q"]},
            {"name": "feeds", "args": ["q", "r"]},
        ])
        # The greedy tie-break pins WHICH correspondence: smallest name
        # (equal), then argument signature -> (0,0) then (1,1).
        result = sm.match(chained_a, chained_b)
        self.assertEqual(result["structural_score"], 2.5)
        self.assertEqual([(r["a"], r["b"]) for r in result["matched_relations"]],
                         [(0, 0), (1, 1)])
        self.assertEqual(result["entity_correspondence"],
                         {"a": "p", "b": "q", "c": "r"})
        # The contrast: same two matches over four disjoint entities.
        disjoint_a = _problem("dia", ["a", "b", "c", "d"], [
            {"name": "feeds", "args": ["a", "b"]},
            {"name": "feeds", "args": ["c", "d"]},
        ])
        disjoint_b = _problem("dib", ["p", "q", "r", "s"], [
            {"name": "feeds", "args": ["p", "q"]},
            {"name": "feeds", "args": ["r", "s"]},
        ])
        self.assertEqual(sm.match(disjoint_a, disjoint_b)["structural_score"], 2.0)


class ValidationAndBoundsTests(unittest.TestCase):
    def test_relation_cap_rejects_above_64(self) -> None:
        def big(n):
            return _problem("big", ["e"], [
                {"name": f"rel{i}", "args": ["e"]} for i in range(n)
            ])

        self.assertEqual(len(big(sm.MAX_RELATIONS)["relations"]), 64)  # at the cap: fine
        with self.assertRaises(ValueError):  # above: refused, not degraded
            sm.match(big(sm.MAX_RELATIONS + 1), big(sm.MAX_RELATIONS + 1))

    def test_malformed_graphs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            sm.match({"entities": "x", "relations": []}, G1)  # not a list
        with self.assertRaises(ValueError):
            sm.match({"entities": ["x", "x"], "relations": []}, G1)  # dup entity
        with self.assertRaises(ValueError):
            sm.match({"entities": [], "relations": [{"name": "r", "args": [True]}]}, G1)
        with self.assertRaises(ValueError):
            # relation reference outside the graph
            sm.match({"entities": [], "relations": [{"name": "AND", "args": [7]}]}, G1)
        with self.assertRaises(ValueError):
            sm.match({"entities": []}, G1)  # missing relations

    def test_argument_roles_must_align(self) -> None:
        # text_regex(literal, entity) vs text_regex(entity, literal):
        # same name and arity, but the structural roles differ by
        # position — the pair is not a candidate.
        a = _problem("a", ["input"], [
            {"name": "text_regex", "args": ["pat", "input"]}])
        b = _problem("b", ["pat"], [
            {"name": "text_regex", "args": ["pat", "other"]}])
        # in b, "pat" is an entity and "other" a literal — roles flip
        self.assertEqual(sm.match(a, b)["structural_score"], 0.0)


class RuleGraphTests(unittest.TestCase):
    def _rules(self):
        from assistant.capabilities.diagnostics.engine import load_rules
        return {rule["id"]: rule for rule in load_rules()}

    def test_builds_honest_predicates_from_a_real_rule(self) -> None:
        rules = self._rules()
        # CL-update-updatercache-001 nests anyOf inside allOf — the
        # deepest shape rules.d ships — over one text_regex and two
        # file_exists leaves.
        graph = sm.rule_match_graph(rules["CL-update-updatercache-001"])
        self.assertEqual(graph["entities"], sorted([
            "input", "~/.config/caelestia-update/repo",
            "~/.cache/caelestia-update-repo"]))
        self.assertEqual([r["name"] for r in graph["relations"]],
                         ["text_regex", "file_present", "file_present", "OR", "AND"])
        # post-order: leaves first, OR over its two leaves, AND over
        # the text_regex and the OR.
        self.assertEqual(graph["relations"][3], {"name": "OR", "args": [1, 2]})
        self.assertEqual(graph["relations"][4], {"name": "AND", "args": [0, 3]})
        # the text_regex predicate carries the pattern as a literal
        self.assertEqual(graph["relations"][0]["args"][1], "input")
        self.assertIn("updater", graph["relations"][0]["args"][0])

    def test_real_rule_self_match_is_pinned(self) -> None:
        graph = sm.rule_match_graph(self._rules()["CL-update-updatercache-001"])
        result = sm.match(graph, graph)
        # Hand-derived: 5 matched relations + 4 systematicity (both OR
        # children and both AND children matched) = 9.0; no two leaf
        # relations share an entity, so connectedness adds nothing.
        self.assertEqual(result["structural_score"], 9.0)
        self.assertEqual(len(result["matched_relations"]), 5)

    def test_file_expected_flag_decides_the_predicate_name(self) -> None:
        rule = {
            "id": "synthetic-rule",
            "matchers": {"allOf": [
                {"type": "file_exists", "path": "~/.cache/stale",
                 "expected": "absent"},
            ]},
        }
        graph = sm.rule_match_graph(rule)
        # The builder is faithful to the tree, not clever about it: a
        # single-member allOf is still a combinator NODE in the rule, so
        # it still becomes an AND over its one child (flattening it
        # would silently erase the difference between a bare leaf and a
        # gated one).
        self.assertEqual(graph["relations"], [
            {"name": "file_absent", "args": ["~/.cache/stale"]},
            {"name": "AND", "args": [0]},
        ])
        # A bare-leaf matcher (schema-valid per engine._walk_matchers)
        # builds to just the leaf, no combinator.
        bare = sm.rule_match_graph({
            "id": "bare-rule",
            "matchers": {"type": "text_substring", "value": "boom"},
        })
        self.assertEqual(bare["relations"],
                         [{"name": "text_substring", "args": ["boom", "input"]}])
        self.assertEqual(bare["entities"], ["input"])

    def test_bad_rule_shapes_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            sm.rule_match_graph({"id": "x"})  # no matchers
        with self.assertRaises(ValueError):
            sm.rule_match_graph({"id": "x", "matchers": {"allOf": [
                {"type": "no_such_matcher"}]}})
        with self.assertRaises(ValueError):
            sm.rule_match_graph({"id": "x", "matchers": {"anyOf": []}})  # asserts nothing


class SurfaceCandidatesTests(unittest.TestCase):
    def test_ranks_candidates_with_full_payload(self) -> None:
        problems = [
            {"id": "structurally-same", "problem": G2},
            {"id": "surface-same", "problem": G3},
            {"id": "unrelated", "problem": _problem("u", ["z"], [
                {"name": "unrelated_rel", "args": ["z"]}])},
        ]
        result = sm.surface_candidates(problems, G1, top=3)
        self.assertEqual(result["status"], "ok")
        self.assertEqual([r["id"] for r in result["results"]],
                         ["structurally-same"])
        hit = result["results"][0]
        self.assertEqual(hit["structural_score"], 5.0)
        self.assertEqual(len(hit["matched_relations"]), 3)
        self.assertIn("systematicity", hit["why"])
        # the signal rides ALONGSIDE lexical retrieval, never instead of it
        self.assertIn("ALONGSIDE BM25/NCD", result["note"])

    def test_abstains_on_no_relational_overlap(self) -> None:
        problems = [{"id": "only-surface", "problem": G3}]
        result = sm.surface_candidates(problems, G1, top=3)
        self.assertEqual(result["status"], "abstained")
        self.assertEqual(result["results"], [])
        self.assertIn("thin", result["reason"])

    def test_single_isolated_match_is_below_the_bar(self) -> None:
        # One lone first-order match (1.0) is surface-level noise by the
        # systematicity principle — abstained, not ranked.
        single = _problem("single", ["u", "v"], [
            {"name": "text_regex", "args": ["p", "u"]}])
        other = _problem("other", ["x", "y"], [
            {"name": "text_regex", "args": ["q", "x"]},
            {"name": "unrelated", "args": ["y"]},
        ])
        result = sm.surface_candidates([{"id": "o", "problem": other}], single)
        self.assertEqual(result["status"], "abstained")

    def test_malformed_entries_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            sm.surface_candidates([{"id": "x"}], G1)  # missing problem
        with self.assertRaises(ValueError):
            sm.surface_candidates([{"id": "x", "problem": G1},
                                   {"id": "x", "problem": G2}], G1)  # dup id
        with self.assertRaises(ValueError):
            sm.surface_candidates([], G1, top=0)

    def test_deterministic_two_runs_identical(self) -> None:
        problems = [{"id": "a", "problem": G2}, {"id": "b", "problem": G3}]
        first = json.dumps(sm.surface_candidates(problems, G1), sort_keys=True)
        second = json.dumps(sm.surface_candidates(problems, G1), sort_keys=True)
        self.assertEqual(first, second)
        m1 = json.dumps(sm.match(G1, G2), sort_keys=True)
        m2 = json.dumps(sm.match(G1, G2), sort_keys=True)
        self.assertEqual(m1, m2)


class CliSurfaceTests(unittest.TestCase):
    def test_analog_cli_over_the_real_rules_corpus(self) -> None:
        import contextlib
        import io

        from assistant.capabilities.retrieval import cli

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = cli.main(["analog", "--rule", "CL-update-updatercache-001",
                             "--top", "3", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertIn(payload["status"], ("ok", "abstained"))
        self.assertIn("ALONGSIDE", payload["note"])
        if payload["results"]:
            self.assertNotIn("CL-update-updatercache-001",
                             [hit["id"] for hit in payload["results"]])
        # an unknown rule id is a clean error, not a traceback
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = cli.main(["analog", "--rule", "CL-no-such-rule"])
        self.assertEqual(code, 2)
        self.assertIn("error:", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
