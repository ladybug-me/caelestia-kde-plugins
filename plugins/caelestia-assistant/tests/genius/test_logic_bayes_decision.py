"""Genius layer tests — logic, Bayes nets, HMM, decisions, tasks, metacog."""
import unittest

from assistant.capabilities.genius import baysnet as bn
from assistant.capabilities.genius import decision as dc
from assistant.capabilities.genius import logic as lg
from assistant.capabilities.genius import metacog as mc
from assistant.capabilities.genius import tasks as tk


class TestLogic(unittest.TestCase):
    def test_classification(self):
        self.assertEqual(lg.classify_formula("p or not p")["classification"], "tautology")
        self.assertEqual(lg.classify_formula("p and not p")["classification"], "contradiction")
        self.assertEqual(lg.classify_formula("p or q")["classification"], "contingency")

    def test_de_morgan_equivalence(self):
        self.assertTrue(lg.equivalent("not (p and q)", "not p or not q")["equivalent"])

    def test_equivalence_counterexample(self):
        r = lg.equivalent("p and q", "p or q")
        self.assertFalse(r["equivalent"])
        self.assertIsNotNone(r["counterexample"])

    def test_entails(self):
        self.assertTrue(lg.entails("p and q", "p")["entails"])
        self.assertFalse(lg.entails("p or q", "p")["entails"])

    def test_sat_dpll(self):
        s = lg.sat_solve("(a or b) and (not a or c) and (not b or not c) and (not a or not b)")
        self.assertTrue(s["satisfiable"])
        self.assertFalse(lg.sat_solve("p and not p")["satisfiable"])
        self.assertFalse(lg.sat_solve("(a <-> b) and (b <-> not a)")["satisfiable"])

    def test_rule_infer_with_proof(self):
        rules = [{"if": ["hot", "humid"], "then": "storm"},
                 {"if": ["storm"], "then": "stay_home"}]
        r = lg.rule_infer(rules, ["hot", "humid"])
        self.assertIn("storm", r["derived"])
        self.assertIn("stay_home", r["derived"])
        self.assertEqual(r["proofs"]["storm"]["from"], ["hot", "humid"])

    def test_csp_map_coloring(self):
        regions = ["wa", "nt", "sa", "q"]
        csp = lg.CSP(regions, {r: ["r", "g", "b"] for r in regions},
                     all_different=[["wa", "nt"], ["wa", "sa"], ["sa", "nt"],
                                    ["sa", "q"]])
        sol = lg.solve_csp(csp)
        self.assertTrue(sol["satisfiable"])
        assignment = sol["first"]
        for group in [["wa", "nt"], ["wa", "sa"], ["sa", "nt"], ["sa", "q"]]:
            self.assertNotEqual(assignment[group[0]], assignment[group[1]])

    def test_csp_arithmetic_counts(self):
        csp = lg.CSP(["x", "y", "z"], {v: [1, 2, 3, 4, 5] for v in "xyz"},
                     constraints=[("x", "y", "<"), ("y", "z", "<")],
                     all_different=[["x", "y", "z"]])
        sol = lg.solve_csp(csp, max_solutions=100)
        self.assertEqual(sol["n_solutions"], 10)


class TestBayes(unittest.TestCase):
    def _net(self):
        net = bn.BayesNet()
        net.add("rain", cpt={"*": 0.2})
        net.add("sprinkler", parents=["rain"], cpt={"rain=T": 0.01, "rain=F": 0.4})
        net.add("grass_wet", parents=["rain", "sprinkler"], cpt={
            "rain=F,sprinkler=F": 0.0, "rain=F,sprinkler=T": 0.9,
            "rain=T,sprinkler=F": 0.8, "rain=T,sprinkler=T": 0.99})
        return net

    def test_rain_posterior_textbook(self):
        q = self._net().query("rain", {"grass_wet": True})
        self.assertAlmostEqual(q["p"], 0.3577, places=4)

    def test_prior(self):
        self.assertAlmostEqual(self._net().query("rain")["p"], 0.2, places=9)

    def test_diagnose_ranks(self):
        d = self._net().diagnose(["rain", "sprinkler"], {"grass_wet": True})
        self.assertEqual(len(d["ranked_causes"]), 2)

    def test_hmm_classic(self):
        hmm = bn.HMM(["rainy", "sunny"], ["walk", "shop", "clean"],
                     [0.6, 0.4], [[0.7, 0.3], [0.4, 0.6]],
                     [[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])
        r = hmm.observe(["walk", "shop", "clean"])
        self.assertEqual(r["decoded_states"], ["sunny", "rainy", "rainy"])
        self.assertAlmostEqual(r["forward"]["likelihood"], 0.033612, places=6)

    def test_hmm_next_prediction(self):
        nxt = bn.hmm_predict_next([0.6, 0.4], [[0.7, 0.3], [0.4, 0.6]],
                                  [[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]],
                                  [0, 0])
        self.assertAlmostEqual(sum(nxt["state_distribution"]), 1.0, places=6)

    def test_naive_bayes(self):
        rows = [{"outlook": "sunny", "wind": "strong"},
                {"outlook": "sunny", "wind": "weak"},
                {"outlook": "rain", "wind": "strong"},
                {"outlook": "rain", "wind": "strong"},
                {"outlook": "rain", "wind": "strong"},
                {"outlook": "overcast", "wind": "weak"}]
        labels = ["no", "no", "yes", "yes", "yes", "yes"]
        nb = bn.NaiveBayes().fit(rows, labels)
        self.assertEqual(nb.predict({"outlook": "rain", "wind": "strong"})["class"], "yes")
        # persistence round-trip
        nb2 = bn.NaiveBayes.from_dict(nb.to_dict())
        self.assertEqual(nb2.predict({"outlook": "rain", "wind": "strong"})["class"], "yes")


class TestDecision(unittest.TestCase):
    MATRIX = [[8, 256, 1.5], [6, 512, 2.0], [9, 128, 1.2], [5, 1024, 3.0]]
    LABELS = ["air", "pro14", "cheap", "pro16"]
    CRITERIA = ["battery", "storage", "weight"]
    WEIGHTS = [0.4, 0.4, 0.2]
    BENEFITS = [True, True, False]

    def test_ahp_consistent(self):
        a = dc.ahp([[1, 1 / 3, 5], [3, 1, 7], [1 / 5, 1 / 7, 1]], ["a", "b", "c"])
        self.assertLess(a["consistency_ratio"], 0.10)
        self.assertTrue(a["consistent"])

    def test_ahp_inconsistent_flagged(self):
        b = dc.ahp([[1, 3, 1 / 5], [1 / 3, 1, 5], [5, 1 / 5, 1]], ["a", "b", "c"])
        self.assertGreater(b["consistency_ratio"], 0.10)
        self.assertFalse(b["consistent"])

    def test_ahp_reciprocal_check(self):
        with self.assertRaises(ValueError):
            dc.ahp([[1, 2], [2, 1]], ["a", "b"])

    def test_topsis_ranks(self):
        t = dc.topsis(self.MATRIX, self.LABELS, self.WEIGHTS, self.CRITERIA,
                      benefits=self.BENEFITS)
        self.assertEqual(t["winner"], "pro16")

    def test_wsm_wpm_agree_on_winner(self):
        w = dc.weighted_sum(self.MATRIX, self.LABELS, self.WEIGHTS, self.CRITERIA,
                            benefits=self.BENEFITS)
        p = dc.weighted_product(self.MATRIX, self.LABELS, self.WEIGHTS, self.CRITERIA,
                                benefits=self.BENEFITS)
        self.assertEqual(w["winner"], p["winner"])

    def test_decision_tree_ev(self):
        tree = {"type": "choice", "children": [
            {"label": "launch", "node": {"type": "chance", "children": [
                {"p": 0.6, "label": "ok", "node": {"type": "leaf", "payoff": 500}},
                {"p": 0.4, "label": "flop", "node": {"type": "leaf", "payoff": -200}}]}},
            {"label": "wait", "node": {"type": "leaf", "payoff": 50}}]}
        self.assertEqual(dc.decision_tree(tree)["expected_value"], 220.0)

    def test_evpi(self):
        r = dc.evpi({"boom": {"invest": 200, "save": 50},
                     "bust": {"invest": -100, "save": 50}},
                    {"boom": 0.5, "bust": 0.5})
        self.assertEqual(r["evpi"], 75.0)

    def test_minimax_regret(self):
        r = dc.minimax_regret([[100, -50], [0, 0]], ["agg", "safe"], ["boom", "bust"])
        self.assertEqual(r["choice"], "agg")

    def test_pareto(self):
        p = dc.pareto_frontier([[10, 5], [8, 6], [12, 4], [5, 10]], ["a", "b", "c", "d"])
        self.assertEqual(sorted(p["frontier"]), ["a", "b", "c", "d"])

    def test_pareto_dominated_detected(self):
        p = dc.pareto_frontier([[10, 10], [5, 5]], ["good", "bad"])
        self.assertEqual(p["frontier"], ["good"])
        self.assertIn("bad", p["dominated"])

    def test_minimax_alpha_beta(self):
        game = {"type": "max", "children": [
            {"type": "min", "children": [{"type": "leaf", "value": 3},
                                         {"type": "leaf", "value": 12},
                                         {"type": "leaf", "value": 8}]},
            {"type": "min", "children": [{"type": "leaf", "value": 2},
                                         {"type": "leaf", "value": 4},
                                         {"type": "leaf", "value": 6}]},
            {"type": "min", "children": [{"type": "leaf", "value": 14},
                                         {"type": "leaf", "value": 5},
                                         {"type": "leaf", "value": 2}]}]}
        g = dc.minimax_game(game)
        self.assertEqual(g["value"], 3)
        self.assertEqual(g["optimal_child"], 0)
        self.assertGreater(g["branches_pruned"], 0)


class TestDecisionCompare(unittest.TestCase):
    """decision.compare — the JSON bridge's single decide entry point."""

    MATRIX = [[8, 256], [6, 512]]
    LABELS = ["air", "pro"]
    CRITERIA = ["battery", "storage"]

    def test_wsm_default_matches_weighted_sum(self):
        r = dc.compare(self.MATRIX, self.LABELS, self.CRITERIA, [0.5, 0.5])
        self.assertEqual(r, dc.weighted_sum(self.MATRIX, self.LABELS,
                                            [0.5, 0.5], self.CRITERIA))

    def test_equal_weights_when_omitted(self):
        r = dc.compare(self.MATRIX, self.LABELS, self.CRITERIA)
        self.assertEqual(r["weights"],
                         {"battery": 0.5, "storage": 0.5})

    def test_method_dispatch_topsis(self):
        r = dc.compare(self.MATRIX, self.LABELS, self.CRITERIA, [0.5, 0.5],
                       method="topsis")
        self.assertEqual(r["winner"],
                         dc.topsis(self.MATRIX, self.LABELS, [0.5, 0.5],
                                   self.CRITERIA)["winner"])

    def test_method_dispatch_pareto_and_regret(self):
        p = dc.compare([[10, 10], [5, 5]], ["good", "bad"], ["x", "y"],
                       method="pareto")
        self.assertEqual(p["frontier"], ["good"])
        g = dc.compare([[3, 1], [2, 4]], ["a", "b"], ["s1", "s2"],
                       method="regret")
        self.assertIn(g["choice"], ("a", "b"))

    def test_method_dispatch_wpm(self):
        r = dc.compare(self.MATRIX, self.LABELS, self.CRITERIA, [0.5, 0.5],
                       method="wpm")
        self.assertEqual(r["method"], "WPM (weighted product)")

    def test_method_is_case_and_space_tolerant(self):
        r = dc.compare(self.MATRIX, self.LABELS, self.CRITERIA, [0.5, 0.5],
                       method=" TOPSIS ")
        self.assertEqual(r["winner"],
                         dc.topsis(self.MATRIX, self.LABELS, [0.5, 0.5],
                                   self.CRITERIA)["winner"])

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            dc.compare(self.MATRIX, self.LABELS, self.CRITERIA,
                       method="vibes")

    def test_bad_shape_raises(self):
        with self.assertRaises(ValueError):
            dc.compare([[8, 256]], ["air", "pro"], self.CRITERIA)


class TestTasks(unittest.TestCase):
    def test_decompose_matches_archetype(self):
        d = tk.decompose("fix the flaky bar widget test")
        self.assertEqual(d["archetype"], "fix")
        self.assertGreater(len(d["steps"]), 4)
        self.assertTrue(all("why" in s for s in d["steps"]))

    def test_decompose_honest_no_match(self):
        d = tk.decompose("buy a new phone")
        self.assertEqual(d["verdict"], "NO_DECOMPOSITION")
        self.assertIn("question", d)

    def test_schedule_topological(self):
        d = tk.decompose("build a new dashboard")
        s = tk.schedule(d["steps"])
        idx = {name: i for i, name in enumerate(s["order"])}
        for step in d["steps"]:
            for dep in step["depends_on"]:
                self.assertLess(idx[dep], idx[step["id"]])

    def test_critical_path_is_longest_chain(self):
        d = tk.decompose("fix the flaky test")
        cp = tk.critical_path(d["steps"])
        self.assertEqual(cp["critical_path_min"], cp["makespan_min"])

    def test_fit_today_respects_dependencies(self):
        d = tk.decompose("fix the flaky test")
        fit = tk.fit_today(d["steps"], 100)
        done = set()
        by_id = {s["id"]: s for s in d["steps"]}
        for chosen in fit["chosen"]:
            self.assertTrue(all(dep in done for dep in by_id[chosen]["depends_on"]))
            done.add(chosen)
        self.assertLessEqual(fit["chosen_effort_min"], 100)

    def test_progress_report_blocked_and_stale(self):
        r = tk.progress_report([
            {"id": "a", "depends_on": [], "created_day": 0, "done": True},
            {"id": "b", "depends_on": ["a"], "created_day": 0, "done": False},
            {"id": "c", "depends_on": ["b"], "created_day": 0, "done": False},
            {"id": "old", "depends_on": [], "created_day": -10, "done": False},
        ], now_day=0)
        self.assertEqual(r["blocked"][0]["id"], "c")
        self.assertEqual(r["stale"][0]["id"], "old")
        self.assertEqual(r["next_action"], "old")


class TestMetacog(unittest.TestCase):
    def test_rule_induction_learns_pattern(self):
        history = []
        for _ in range(12):
            history.append({"features": {"domain": "settings", "hour": "14"},
                           "outcome": "approve"})
            history.append({"features": {"domain": "settings", "hour": "2"},
                           "outcome": "reject"})
        r = mc.induce_rules(history)
        self.assertEqual(r["verdict"], "LEARNED")
        self.assertTrue(any("reject" in rule["prediction"] or
                            "reject" in rule["reads"] for rule in r["rules"]))

    def test_thin_history_is_honest(self):
        r = mc.induce_rules([{"features": {"a": "b"}, "outcome": "approve"}])
        self.assertEqual(r["verdict"], "INSUFFICIENT_HISTORY")

    def test_active_learning_picks_max_entropy(self):
        q = mc.suggest_question([
            {"id": "p1", "features": {}, "p": 0.51},
            {"id": "p2", "features": {}, "p": 0.95},
            {"id": "p3", "features": {}, "p": 0.48},
        ])
        self.assertEqual(q["verdict"], "ASK")
        self.assertEqual(q["question"]["item_id"], "p1")

    def test_coverage_map(self):
        usage = ([{"domain": "settings", "accepted": True, "day": 1}] * 8 +
                 [{"domain": "cleanup", "accepted": False, "day": 1}] * 4)
        c = mc.coverage_map(usage)
        self.assertEqual(c["most_used"], "settings")
        self.assertIn("mismatch", c["domains"]["cleanup"]["verdict"])

    def test_cluster_requests_groups_themes(self):
        reqs = ["make the bar thinner", "bar is too thick", "thinner dock",
                "solve x^2 - 4", "solve 2x + 3 = 7", "find duplicate files"]
        cl = mc.cluster_requests(reqs)
        names = " ".join(c["name"] for c in cl["clusters"])
        self.assertIn("bar", names)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Phase 2.2: the general resource-contention scheduling primitive.
# ---------------------------------------------------------------------------

class TestScheduleResources(unittest.TestCase):
    def test_conflict_free_per_resource(self):
        r = lg.schedule_resources(
            [("compile", "cpu"), ("test", "cpu"), ("package", "cpu")], 3)
        self.assertTrue(r["satisfiable"])
        slots = list(r["schedule"].values())
        self.assertEqual(sorted(slots), [0, 1, 2])  # all distinct

    def test_precedence_orders_the_plan(self):
        r = lg.schedule_resources(
            [("compile", "cpu"), ("test", "cpu"), ("package", "cpu")], 3,
            precedence=[("compile", "test"), ("test", "package")])
        self.assertTrue(r["satisfiable"])
        self.assertEqual(r["schedule"],
                         {"compile": 0, "test": 1, "package": 2})

    def test_resources_run_in_parallel(self):
        # different resources may share a slot
        r = lg.schedule_resources(
            [("a", "cpu"), ("b", "gpu"), ("c", "net")], 1)
        self.assertTrue(r["satisfiable"])
        self.assertEqual(r["slot_load"], {"0": ["a", "b", "c"]})

    def test_infeasible_reports_pruning_not_silence(self):
        r = lg.schedule_resources([("a", "gpu"), ("b", "gpu"), ("c", "gpu")], 2)
        self.assertFalse(r["satisfiable"])
        self.assertIsNone(r["schedule"])
        self.assertIn("reason", r)

    def test_windows_bound_the_domain(self):
        r = lg.schedule_resources([("x", "net"), ("y", "net")], 4,
                                  windows={"x": (0, 1), "y": (0, 1)})
        self.assertTrue(r["satisfiable"])
        self.assertTrue(r["schedule"]["x"] <= 1)
        self.assertTrue(r["schedule"]["y"] <= 1)

    def test_dict_tasks_and_unknown_precedence_rejected(self):
        r = lg.schedule_resources([{"id": "a", "resource": "cpu"},
                                   {"id": "b", "resource": "cpu"}], 2)
        self.assertTrue(r["satisfiable"])
        with self.assertRaises(ValueError):
            lg.schedule_resources([("a", "cpu")], 2,
                                  precedence=[("nope", "a")])

    def test_ac3_leaves_evidence(self):
        r = lg.schedule_resources(
            [("a", "cpu"), ("b", "cpu"), ("c", "cpu")], 3,
            precedence=[("a", "b"), ("b", "c")])
        self.assertIsInstance(r["ac3_pruned"], list)
        self.assertTrue(r["ac3_pruned"])  # domains narrowed before search
