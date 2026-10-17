"""Tests for the personal-PKM engines (moved from brain/tests during the
issue #120 scope split). Nothing here touches the shell: these modules
answer vault/task questions only.
"""
import unittest

from assistant.capabilities.brain.personal import duration, graph, planner, priority, srs, survival, vault
from assistant.capabilities.brain.personal import ghost, health, journal, linkrec
from assistant.capabilities.brain.personal.graph import Graph


class GraphTests(unittest.TestCase):
    def test_pagerank_sums_to_one_and_favors_hub(self):
        g = graph.Graph()
        g.add("a", {"hub"})
        g.add("b", {"hub"})
        g.add("c", {"hub"})
        g.add("hub", set())
        pr = g.pagerank()
        self.assertAlmostEqual(sum(pr.values()), 1.0, places=6)
        self.assertEqual(max(pr, key=pr.get), "hub")

    def test_orphans(self):
        g = graph.Graph()
        g.add("a", {"b"})
        g.add("lonely", set())
        self.assertEqual(g.orphans(), ["lonely"])

    def test_communities_split_disconnected_components(self):
        g = graph.Graph()
        g.add("a", {"b"})
        g.add("b", {"c"})
        g.add("x", {"y"})
        comms = g.communities()
        self.assertEqual(len(comms), 2)
        self.assertIn(["a", "b", "c"], comms)

    def test_links_parse_aliases_and_anchors(self):
        self.assertEqual(graph.links("see [[Alpha|a]] and [[beta#h]]"), {"alpha", "beta"})


class PriorityTests(unittest.TestCase):
    def test_overdue_beats_far_deadline(self):
        m = priority.PriorityModel()
        overdue = m.score(priority.features({"deadline_days": -1, "importance": 0.5, "effort_min": 30}))
        far = m.score(priority.features({"deadline_days": 60, "importance": 0.5, "effort_min": 30}))
        self.assertGreater(overdue, far)

    def test_score_is_probability(self):
        m = priority.PriorityModel()
        for d in (None, -5, 0, 3, 100):
            s = m.score(priority.features({"deadline_days": d, "importance": 1.0, "effort_min": 1}))
            self.assertTrue(0.0 < s < 1.0)

    def test_fit_moves_weights_toward_labels(self):
        # Importance alone predicts approval: the learned importance weight should grow.
        examples = []
        for imp in (0.0, 0.1, 0.9, 1.0) * 10:
            examples.append(([0.5, imp, 0.5], 1 if imp > 0.5 else 0))
        before = priority.PriorityModel().w[1]
        m = priority.PriorityModel().fit(examples)
        self.assertGreater(m.w[1], before)


class DurationTests(unittest.TestCase):
    def test_prior_when_unseen(self):
        est = duration.DurationModel().estimate("writing")
        self.assertEqual(est["n"], 0)
        self.assertAlmostEqual(est["median_min"], 30.0, places=0)

    def test_observations_pull_estimate(self):
        m = duration.DurationModel()
        for _ in range(20):
            m.observe("review", 90)
        self.assertGreater(m.estimate("review")["median_min"], 70)
        self.assertGreater(m.estimate("review")["p80_min"], m.estimate("review")["median_min"])


class PlannerTests(unittest.TestCase):
    def test_knapsack_optimal_small_case(self):
        items = [
            {"id": "a", "minutes": 60, "value": 6.0},
            {"id": "b", "minutes": 60, "value": 5.0},
            {"id": "c", "minutes": 30, "value": 3.0},
        ]
        chosen, value, used = planner.knapsack(items, 90)
        self.assertEqual(sorted(chosen), ["a", "c"])
        self.assertAlmostEqual(value, 9.0)
        self.assertLessEqual(used, 90)

    def test_knapsack_empty_budget(self):
        self.assertEqual(planner.knapsack([{"id": "a", "minutes": 10, "value": 1}], 0), ([], 0.0, 0))

    def test_cpm_critical_path(self):
        tasks = {
            "design": {"minutes": 60, "deps": []},
            "build": {"minutes": 120, "deps": ["design"]},
            "docs": {"minutes": 30, "deps": ["design"]},
            "ship": {"minutes": 10, "deps": ["build", "docs"]},
        }
        s = planner.cpm(tasks)
        self.assertEqual(s["ship"]["ef"], 190)
        self.assertTrue(s["build"]["critical"])
        self.assertFalse(s["docs"]["critical"])
        self.assertEqual(s["docs"]["slack"], 90)

    def test_cpm_detects_cycle(self):
        with self.assertRaises(ValueError):
            planner.cpm({"a": {"minutes": 1, "deps": ["b"]}, "b": {"minutes": 1, "deps": ["a"]}})


class SurvivalTests(unittest.TestCase):
    def test_kaplan_meier_known_values(self):
        # 4 tasks; times 1, 2, 3, 4; completed at 1 and 3, censored at 2 and 4.
        curve = survival.kaplan_meier([1, 2, 3, 4], [True, False, True, False])
        self.assertAlmostEqual(curve[0][1], 3 / 4)
        self.assertAlmostEqual(curve[1][1], 3 / 4 * 1 / 2)

    def test_completion_prob_zero_after_all_events_when_no_survivors(self):
        curve = survival.kaplan_meier([1, 2], [True, True])
        self.assertEqual(survival.completion_prob(curve, 5, horizon=1), 0.0)


class SrsTests(unittest.TestCase):
    def test_good_grows_stability_and_again_shrinks_it(self):
        c = srs.new_card()
        good = srs.review(c, 3, elapsed_days=1.0)
        self.assertGreater(good["S"], c["S"])
        again = srs.review(good, 1, elapsed_days=1.0)
        self.assertLess(again["S"], good["S"])

    def test_retrievability_at_stability_is_ninety_percent(self):
        self.assertAlmostEqual(srs.retrievability(5.0, 5.0), 0.9)

    def test_due_when_retrievability_drops(self):
        self.assertTrue(srs.due({"S": 2.0, "D": 5.0}, now_elapsed=10.0))
        self.assertFalse(srs.due({"S": 30.0, "D": 5.0}, now_elapsed=1.0))

    def test_rejects_bad_rating(self):
        with self.assertRaises(ValueError):
            srs.review(srs.new_card(), 7)


class VaultTests(unittest.TestCase):
    def test_parse_front_matter_and_inline_tags(self):
        text = "---\ntags: [Python, ml-basics]\n---\nnotes about #tooling and a#notatag"
        self.assertEqual(vault.parse_note(text), {"python", "ml-basics", "tooling"})


class LinkRecTests(unittest.TestCase):
    def test_shared_neighbour_pair_outranks_unrelated_pair(self):
        g = Graph()
        g.add("a", {"hub"})
        g.add("b", {"hub"})
        g.add("c", set())
        notes = {"a": "unrelated text one", "b": "unrelated text two", "c": "totally different"}
        suggestions = linkrec.suggest_links(g, notes, top=5)
        pair_ids = [{s["a"], s["b"]} for s in suggestions]
        self.assertIn({"a", "b"}, pair_ids)

    def test_cold_start_falls_back_to_text_similarity(self):
        g = Graph()
        base = "recipe for sourdough bread with flour and water and a long proofing time"
        notes = {"x": base, "y": base + " plus a pinch of salt at the end"}
        suggestions = linkrec.suggest_links(g, notes, top=5, min_jaccard=0.1)
        self.assertTrue(any(s["via"] == "text_similarity" for s in suggestions))

    def test_already_linked_pair_is_skipped(self):
        g = Graph()
        g.add("a", {"b"})
        notes = {"a": "same same same", "b": "same same same"}
        suggestions = linkrec.suggest_links(g, notes)
        self.assertNotIn({"a", "b"}, [{s["a"], s["b"]} for s in suggestions])


class LinkRecBlendTests(unittest.TestCase):
    """Phase 3.2: the Adamic-Adar recommender combined with TF-IDF
    cosine from the shared textmine engine (the personal README's
    stated shared-utility import direction)."""

    def test_combined_via_when_both_signals_fire(self):
        g = Graph()
        g.add("a", {"hub"})
        g.add("b", {"hub"})
        notes = {"a": "sourdough starter recipe flour water",
                 "b": "sourdough starter feeding flour ratios"}
        suggestions = linkrec.suggest_links(g, notes, top=5)
        pair = next(s for s in suggestions if {s["a"], s["b"]} == {"a", "b"})
        self.assertEqual(pair["via"], "combined")
        self.assertIn("aa", pair)
        self.assertIn("cos", pair)
        # the blend respects the weight: w=1 must equal the pure cosine,
        # w=0 must equal the pure saturated Adamic-Adar
        pure_cos = linkrec.suggest_links(g, notes, top=5, tfidf_weight=1.0)
        pc = next(s for s in pure_cos if {s["a"], s["b"]} == {"a", "b"})
        self.assertAlmostEqual(pc["score"], round(pair["cos"], 3), places=3)

    def test_tfidf_similar_pair_suggested_without_graph(self):
        g = Graph()  # cold start: no links at all
        base = "recipe for sourdough bread with flour and water and a long proofing time"
        notes = {"x": base, "y": base + " plus a pinch of salt at the end",
                 "z": "quarterly budget spreadsheet numbers and totals"}
        suggestions = linkrec.suggest_links(g, notes, top=5, min_jaccard=0.1)
        x_y = next(s for s in suggestions if {s["a"], s["b"]} == {"x", "y"})
        self.assertEqual(x_y["via"], "text_similarity")
        # TF-IDF must rank the sourdough pair above the budget pair
        x_z = [s for s in suggestions if {s["a"], s["b"]} == {"x", "z"}]
        if x_z:
            self.assertGreater(x_y["score"], x_z[0]["score"])

    def test_weight_zero_reproduces_graph_only_ranking(self):
        g = Graph()
        g.add("a", {"hub"})
        g.add("b", {"hub"})
        notes = {"a": "alpha beta gamma delta", "b": "alpha beta epsilon zeta"}
        suggestions = linkrec.suggest_links(g, notes, tfidf_weight=0.0)
        pair = next(s for s in suggestions if {s["a"], s["b"]} == {"a", "b"})
        self.assertEqual(pair["via"], "shared_neighbours")
        # aa_norm for a single shared neighbour of degree 2:
        aa = linkrec.adamic_adar(g, "a", "b")
        self.assertAlmostEqual(pair["score"],
                               round((aa / (aa + 1.0)), 3), places=3)

    def test_empty_notes_never_suggest(self):
        g = Graph()
        notes = {"a": "", "b": ""}
        suggestions = linkrec.suggest_links(g, notes)
        self.assertEqual(suggestions, [])


class GhostTests(unittest.TestCase):
    def test_finds_unchecked_box_and_action_line(self):
        notes = {"todo.md": "- [ ] ship the release\n- buy milk"}
        rows = ghost.find_ghosts(notes, known_task_titles=[])
        self.assertTrue(any("ship the release" in r["line"] for r in rows))

    def test_known_task_is_not_flagged_again(self):
        notes = {"todo.md": "- [ ] ship the release"}
        rows = ghost.find_ghosts(notes, known_task_titles=["ship the release"])
        self.assertEqual(rows, [])


class JournalTests(unittest.TestCase):
    def test_brier_score_zero_for_perfect_confidence(self):
        entries = {}
        journal.record(entries, "d1", "it works", 1.0)
        journal.resolve(entries, "d1", True)
        self.assertAlmostEqual(journal.brier_score(entries), 0.0)

    def test_resolve_unknown_decision_raises(self):
        with self.assertRaises(KeyError):
            journal.resolve({}, "nope", True)

    def test_calibration_curve_buckets_by_confidence(self):
        entries = {}
        journal.record(entries, "a", "s", 0.9)
        journal.resolve(entries, "a", True)
        journal.record(entries, "b", "s", 0.2)
        journal.resolve(entries, "b", False)
        curve = journal.calibration_curve(entries, bins=2)
        self.assertEqual(len(curve), 2)


class HealthTests(unittest.TestCase):
    def test_orphan_and_stale_score_lower_than_healthy(self):
        healthy = health.score("h", age_days=1, orphan_ids=set(), dup_ids=set())
        unhealthy = health.score("u", age_days=200, orphan_ids={"u"}, dup_ids={"u"})
        self.assertGreater(healthy["score"], unhealthy["score"])
        self.assertTrue(unhealthy["stale"])
        self.assertTrue(unhealthy["orphan"])

    def test_rank_notes_worst_first(self):
        rows = [{"id": "a", "score": 90}, {"id": "b", "score": 10}]
        self.assertEqual([r["id"] for r in health.rank_notes(rows)], ["b", "a"])


if __name__ == "__main__":
    unittest.main()
