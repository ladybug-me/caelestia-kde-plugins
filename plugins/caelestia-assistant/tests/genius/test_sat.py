"""tests for genius.sat — DPLL/CDCL-style solver + bounded model checking.

Pinned: SAT/UNSAT verdicts against hand-checkable CNFs (including
pigeonhole PHP(5,4), a classic small UNSAT that unit propagation must
derive, not guess), learned-clause accumulation, the node-budget
ABSTAIN (loud, never a silent guess), the reachability encoding's
witness paths, exact deadlock detection, and both integration audits
(settings preset graph CLEAN, HTN goal DAG terminal-nodes-are-goals
interpretation).
"""
import unittest

from assistant.capabilities.genius import sat


def php(n_pigeons: int, n_holes: int):
    """Pigeonhole principle CNF: n pigeons, m < n holes — UNSAT."""
    cnf = []
    for p in range(n_pigeons):
        cnf.append([p * n_holes + h + 1 for h in range(n_holes)])
    for h in range(n_holes):
        for p in range(n_pigeons):
            for q in range(p + 1, n_pigeons):
                cnf.append([-(p * n_holes + h + 1), -(q * n_holes + h + 1)])
    return cnf


class SolverTests(unittest.TestCase):
    def test_satisfiable_model_satisfies_every_clause(self):
        result = sat.solve([[1, 2], [-2, 3], [1, -3]])
        self.assertTrue(result["satisfiable"])
        model = result["model"]
        for clause in ([1, 2], [-2, 3], [1, -3]):
            self.assertTrue(any(
                model.get(abs(l), False) == (l > 0) for l in clause))

    def test_trivial_unsat(self):
        result = sat.solve([[1], [-1]])
        self.assertFalse(result["satisfiable"])
        self.assertEqual(result["learned_clauses"], 1)

    def test_pigeonhole_5_4_unsat_proved(self):
        result = sat.solve(php(5, 4), node_budget=100_000)
        self.assertFalse(result["satisfiable"])
        self.assertGreater(result["learned_clauses"], 0)

    def test_budget_abstains_loudly(self):
        with self.assertRaises(sat.SatBudgetError):
            sat.solve(php(7, 6), node_budget=300)

    def test_budget_must_be_positive(self):
        with self.assertRaises(ValueError):
            sat.SATSolver(node_budget=0)

    def test_literal_zero_refused(self):
        with self.assertRaises(ValueError):
            sat.solve([[1, 0]])

    def test_empty_formula_trivially_satisfiable(self):
        result = sat.solve([])
        self.assertTrue(result["satisfiable"])

    def test_determinism(self):
        cnf = php(4, 3)
        a = sat.solve(cnf)
        b = sat.solve(cnf)
        self.assertEqual(a, b)


class GraphTests(unittest.TestCase):
    GRAPH = {
        "boot": ["a", "b"],
        "a": ["c"],
        "b": ["c"],
        "c": [],
        "orphan": ["boot"],
    }

    def test_deadlocks_exact(self):
        self.assertEqual(sat.deadlock_states(self.GRAPH), ["c"])

    def test_unknown_successor_refused(self):
        with self.assertRaises(ValueError):
            sat.deadlock_states({"a": ["ghost"]})

    def test_witness_path_found(self):
        w = sat.witness_path(self.GRAPH, "boot", "c", 3)
        self.assertTrue(w["reachable_within_k"])
        self.assertEqual(w["steps"], 2)
        self.assertEqual(w["path"][0], "boot")
        self.assertEqual(w["path"][-1], "c")

    def test_unreachable_within_k_is_provable(self):
        # boot can NEVER reach orphan (orphan's only edge points back
        # to boot) — UNSAT at k=1, and still UNSAT at k=5, which is the
        # soundness property the encoding must have
        w = sat.witness_path(self.GRAPH, "boot", "orphan", 1)
        self.assertFalse(w["reachable_within_k"])
        w2 = sat.witness_path(self.GRAPH, "boot", "orphan", 5)
        self.assertFalse(w2["reachable_within_k"])
        # ...while b IS reachable, one hop
        wb = sat.witness_path(self.GRAPH, "boot", "b", 1)
        self.assertTrue(wb["reachable_within_k"])
        self.assertEqual(wb["path"], ["boot", "b"])

    def test_audit_reports_findings(self):
        audit = sat.graph_audit(self.GRAPH, "boot", k=3)
        self.assertEqual(audit["status"], "FINDINGS")
        self.assertEqual(audit["deadlock_states"], ["c"])
        self.assertIn("orphan", audit["unreachable_within_k"])

    def test_audit_clean_graph(self):
        # a cycle: no deadlocks, everything reachable -> CLEAN
        audit = sat.graph_audit({"x": ["y"], "y": ["x"]}, "x", k=2)
        self.assertEqual(audit["status"], "CLEAN")

    def test_audit_abstain_recorded_not_raised(self):
        # a cyclic 24-state chain (no deadlocks, everything reachable):
        # the ONLY possible non-CLEAN outcome is a budget ABSTAIN, and
        # node_budget=1 forces it on every target — recorded in the
        # report, never raised past the audit, never a guessed verdict
        big = {f"s{i}": [f"s{i + 1}"] for i in range(23)}
        big["s23"] = ["s0"]
        audit = sat.graph_audit(big, "s0", k=23, node_budget=1)
        self.assertEqual(audit["status"], "ABSTAINED_BUDGET")
        self.assertTrue(audit["abstained"])


class IntegrationTests(unittest.TestCase):
    def test_preset_transition_graph_audit_clean(self):
        from assistant.capabilities.settings import presets
        graph = presets.transition_graph()
        names = sorted(graph)
        self.assertIn("custom", names)
        audit = presets.audit()
        self.assertEqual(audit["status"], "CLEAN")
        self.assertEqual(audit["unreachable_within_k"], [])

    def test_goal_decompose_carries_audit(self):
        from assistant.capabilities.agent import goals
        d = goals.decompose("tune the bar, then make the dock smaller")
        audit = d.get("reachability_audit") or {}
        self.assertEqual(audit.get("status"), "CLEAN")
        # terminal nodes are goals, not faults
        self.assertEqual(audit.get("dead_ends_with_dependents"), [])
        self.assertEqual(audit.get("unreachable_within_k"), [])
        self.assertTrue(audit.get("terminal_nodes"))


if __name__ == "__main__":
    unittest.main()
