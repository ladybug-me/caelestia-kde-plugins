"""genius.graphs + genius.optimize tests — every algorithm pinned or cross-checked."""
import itertools
import math
import unittest

from assistant.capabilities.genius import graphs as g
from assistant.capabilities.genius import optimize as op


class TestDijkstra(unittest.TestCase):
    GRAPH = {
        "a": {"b": 4, "c": 1},
        "b": {"d": 1},
        "c": {"b": 2, "d": 5},
        "d": {},
    }

    def test_known_distances(self):
        r = g.dijkstra(self.GRAPH, "a")
        self.assertEqual(r["dist"], {"a": 0.0, "c": 1.0, "b": 3.0, "d": 4.0})

    def test_path_reconstruction(self):
        r = g.dijkstra(self.GRAPH, "a", target="d")
        self.assertEqual(r["path"], ["a", "c", "b", "d"])
        self.assertEqual(r["cost"], 4.0)

    def test_unreachable(self):
        r = g.dijkstra({"a": {}, "z": {"a": 1}}, "a", target="z")
        self.assertIsNone(r["path"])
        self.assertIsNone(r["cost"])

    def test_negative_rejected(self):
        with self.assertRaises(ValueError):
            g.dijkstra({"a": {"b": -1}}, "a")

    def test_matches_floyd_warshall_bruteforce(self):
        # cross-check on a small random-ish fixed graph against brute force
        gr = {
            "s": {"a": 2, "b": 5},
            "a": {"c": 1, "b": 2},
            "b": {"c": 4, "d": 1},
            "c": {"d": 3, "t": 7},
            "d": {"t": 2},
            "t": {},
        }
        nodes = list(gr)
        for src, dst in itertools.product(nodes, nodes):
            if src == dst:
                continue
            r = g.dijkstra(gr, src, target=dst)
            self.assertEqual(r["cost"], self._bf(gr, src, dst), (src, dst))

    @staticmethod
    def _bf(gr, src, dst):
        dist = {src: 0.0}
        edges = [(u, v, w) for u, nb in gr.items() for v, w in nb.items()]
        for _ in range(len(gr)):
            for u, v, w in edges:
                if u in dist and dist[u] + w < dist.get(v, math.inf):
                    dist[v] = dist[u] + w
        return dist.get(dst)


class TestAStar(unittest.TestCase):
    def test_grid_manhattan_optimal(self):
        gr = {f"{x},{y}": {} for x in range(5) for y in range(5)}
        for x in range(5):
            for y in range(5):
                for dx, dy in ((1, 0), (0, 1)):
                    if x + dx < 5 and y + dy < 5:
                        gr[f"{x},{y}"][f"{x+dx},{y+dy}"] = 1.0
                        gr[f"{x+dx},{y+dy}"].setdefault(f"{x},{y}", 1.0)

        def h(node):
            x, y = map(int, node.split(","))
            return abs(x - 4) + abs(y - 4)

        r = g.astar(gr, "0,0", "4,4", h)
        self.assertEqual(r["cost"], 8.0)
        self.assertEqual(len(r["path"]), 9)


class TestToposort(unittest.TestCase):
    def test_order(self):
        r = g.toposort([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")])
        self.assertTrue(r["is_dag"])
        self.assertEqual(r["order"], ["a", "b", "c", "d"])

    def test_cycle_detected(self):
        r = g.toposort([("a", "b"), ("b", "a")])
        self.assertFalse(r["is_dag"])
        self.assertEqual(set(r["cycle"]), {"a", "b"})

    def test_deterministic(self):
        edges = [("b", "a"), ("c", "a"), ("b", "c")]
        r1 = g.toposort(edges)
        r2 = g.toposort(list(reversed(edges)))
        self.assertEqual(r1["order"], r2["order"])

    def test_longest_path_critical_chain(self):
        r = g.longest_path_dag([("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")],
                               weights={("s", "a"): 3, ("s", "b"): 1, ("a", "c"): 4,
                                        ("b", "c"): 2, ("c", "t"): 5})
        self.assertEqual(r["length"], 12.0)
        self.assertEqual(r["critical_path"], ["s", "a", "c", "t"])


class TestUnionFindKruskal(unittest.TestCase):
    def test_msn_known(self):
        r = g.min_spanning_tree(
            ["a", "b", "c", "d"],
            [("a", "b", 1.0), ("b", "c", 2.0), ("a", "c", 3.0), ("c", "d", 4.0),
             ("a", "d", 10.0)])
        self.assertAlmostEqual(r["total"], 7.0)
        self.assertEqual(len(r["edges"]), 3)

    def test_components_forest(self):
        r = g.min_spanning_tree(["a", "b", "x", "y"],
                                [("a", "b", 1.0), ("x", "y", 1.0)])
        self.assertEqual(len(r["components"]), 2)

    def test_union_find_connectedness(self):
        uf = g.UnionFind(["1", "2", "3"])
        uf.union("1", "2")
        self.assertTrue(uf.connected("1", "2"))
        self.assertFalse(uf.connected("1", "3"))


class TestBellmanFord(unittest.TestCase):
    def test_negative_weights_ok(self):
        r = g.bellman_ford({"s": {"a": 4, "b": 1}, "b": {"a": -2}, "a": {}}, "s")
        self.assertAlmostEqual(r["dist"]["a"], -1.0)

    def test_negative_cycle_reported(self):
        r = g.bellman_ford({"s": {"a": 1}, "a": {"b": -1}, "b": {"a": -1}, "b2": {}}, "s")
        self.assertIsNotNone(r["negative_cycle"])


class TestArticulation(unittest.TestCase):
    def test_bridge_node_found(self):
        # path graph a-b-c: b is an articulation point
        self.assertEqual(g.articulation_points({"a": ["b"], "b": ["a", "c"],
                                                "c": ["b"]}), ["b"])

    def test_cycle_has_none(self):
        self.assertEqual(g.articulation_points(
            {"a": ["b", "c"], "b": ["a", "c"], "c": ["a", "b"]}), [])


class TestHungarian(unittest.TestCase):
    def test_classic_matrix(self):
        cost = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]
        r = g.hungarian(cost)
        self.assertAlmostEqual(r["total"], 5.0)  # 1+0+4? no: known optimum 5
        for i, j in enumerate(r["assignment"]):
            self.assertLess(j, 3)

    def test_matches_bruteforce(self):
        import random
        rng = random.Random(11)
        for _ in range(30):
            n = 4
            cost = [[rng.randint(0, 9) for _ in range(n)] for _ in range(n)]
            best = min(
                sum(cost[i][p[i]] for i in range(n))
                for p in itertools.permutations(range(n))
            )
            self.assertEqual(g.hungarian(cost)["total"], best)

    def test_rectangular(self):
        cost = [[10, 19, 8, 15], [10, 18, 7, 17]]
        r = g.hungarian(cost)
        self.assertAlmostEqual(r["total"], 17.0)

    def test_rejects_wide_rows(self):
        with self.assertRaises(ValueError):
            g.hungarian([[1, 2], [1]])


class TestOptimize(unittest.TestCase):
    def test_anneal_finds_bowl_minimum(self):
        r = op.anneal(lambda x: (x[0] - 3) ** 2 + (x[1] + 1) ** 2, [10.0, 10.0])
        self.assertAlmostEqual(r["energy"], 0.0, places=2)
        self.assertAlmostEqual(r["x"][0], 3.0, places=1)
        self.assertAlmostEqual(r["x"][1], -1.0, places=1)

    def test_anneal_deterministic_with_seed(self):
        f = lambda x: x[0] ** 2  # noqa: E731
        a = op.anneal(f, [5.0], seed=3)
        b = op.anneal(f, [5.0], seed=3)
        self.assertEqual(a["x"], b["x"])

    def test_anneal_respects_bounds(self):
        r = op.anneal(lambda x: -(x[0] ** 2), [0.0], bounds=[(-1.0, 1.0)], steps=800)
        self.assertLessEqual(r["x"][0], 1.0 + 1e-9)

    def test_hill_climb(self):
        r = op.hill_climb(lambda x: (x[0] - 2) ** 2 + 1, [0.0])
        self.assertAlmostEqual(r["energy"], 1.0, places=2)

    def test_genetic_minimizes(self):
        r = op.genetic(lambda x: sum((v - 2) ** 2 for v in x),
                       bounds=[(-5.0, 5.0)] * 3, pop_size=30, generations=80)
        self.assertLess(r["energy"], 0.05)

    def test_ternary_and_golden_agree(self):
        f = lambda x: (x - 1.234) ** 2  # noqa: E731
        t = op.ternary_min(f, -10.0, 10.0)
        gs = op.golden_section(f, -10.0, 10.0)
        self.assertAlmostEqual(t["x"], 1.234, places=4)
        self.assertAlmostEqual(gs["x"], 1.234, places=4)

    def test_pareto_frontier_known(self):
        pts = [{"cpu": 5, "fps": 60}, {"cpu": 2, "fps": 30}, {"cpu": 9, "fps": 55}]
        r = op.pareto_frontier(pts, ["cpu", "fps"], ["min", "max"])
        self.assertEqual(len(r["front"]), 2)
        self.assertIn({"cpu": 5, "fps": 60}, r["front"])
        self.assertIn({"cpu": 2, "fps": 30}, r["front"])

    def test_pareto_tiers(self):
        pts = [{"a": 1}, {"a": 2}, {"a": 3}]
        r = op.pareto_sort(pts, ["a"], ["min"])
        self.assertEqual(r["tiers"], 3)
        self.assertEqual(r["fronts"][0], [{"a": 1}])

    def test_pareto_missing_axis_honest(self):
        with self.assertRaises(ValueError):
            op.pareto_frontier([{"cpu": 1}], ["gpu"])


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Phase 2.2: max-flow, communities, branch-and-bound, tabu search.
# ---------------------------------------------------------------------------

class TestEdmondsKarp(unittest.TestCase):
    # the classic CLRS-style flow network: answer 15
    NET = {"s": {"a": 10, "b": 5},
           "a": {"b": 15, "t": 10},
           "b": {"t": 10}}

    def test_known_max_flow(self):
        r = g.edmonds_karp(self.NET, "s", "t")
        self.assertEqual(r["max_flow"], 15.0)

    def test_min_cut_matches_flow_value(self):
        # max-flow/min-cut: the cut crossing capacity equals the flow
        r = g.edmonds_karp(self.NET, "s", "t")
        cut_value = sum(self.NET[u][v]
                        for u, v in r["min_cut_edges"])
        self.assertEqual(cut_value, r["max_flow"])

    def test_zero_capacity_edge_is_skipped(self):
        r = g.edmonds_karp({"s": {"t": 0}}, "s", "t")
        self.assertEqual(r["max_flow"], 0.0)

    def test_asymmetric_capacities(self):
        r = g.edmonds_karp({"s": {"t": 7}, "t": {"s": 3}}, "s", "t")
        self.assertEqual(r["max_flow"], 7.0)

    def test_source_equals_sink_rejected(self):
        with self.assertRaises(ValueError):
            g.edmonds_karp(self.NET, "s", "s")

    def test_negative_capacity_rejected(self):
        with self.assertRaises(ValueError):
            g.edmonds_karp({"s": {"t": -1}}, "s", "t")


class TestCommunities(unittest.TestCase):
    def test_two_clusters_separate(self):
        adjacency = {"a": ["b", "c"], "b": ["a"], "c": ["a"],
                     "d": ["e"], "e": ["d"]}
        r = g.communities(adjacency)
        self.assertEqual(r["n_communities"], 2)
        self.assertEqual(sorted(map(sorted, r["communities"])),
                         [["a", "b", "c"], ["d", "e"]])

    def test_dict_neighbors_accepted(self):
        r = g.communities({"a": {"b": 1}, "b": {"a": 1}})
        self.assertEqual(r["communities"], [["a", "b"]])

    def test_isolated_node_is_its_own_group(self):
        r = g.communities({"x": [], "y": ["z"], "z": ["y"]})
        self.assertEqual(sorted(map(sorted, r["communities"])),
                         [["x"], ["y", "z"]])


class TestBranchAndBound(unittest.TestCase):
    def test_classic_knapsack(self):
        # (w,v): (10,60),(20,100),(30,120), cap 50 -> take items 1,2 = 220
        r = op.branch_and_bound([(10, 60), (20, 100), (30, 120)], 50)
        self.assertEqual(r["chosen"], [1, 2])
        self.assertEqual(r["total_value"], 220.0)
        self.assertEqual(r["total_weight"], 50.0)

    def test_optimal_on_exhaustive_cross_check(self):
        import random as _random
        rng = _random.Random(2026)
        for _ in range(25):
            n = rng.randint(4, 10)
            items = [(rng.randint(1, 20), rng.randint(1, 30))
                     for _ in range(n)]
            cap = rng.randint(5, 60)
            best = 0
            for mask in range(1 << n):
                w = sum(items[i][0] for i in range(n) if mask >> i & 1)
                v = sum(items[i][1] for i in range(n) if mask >> i & 1)
                if w <= cap and v > best:
                    best = v
            r = op.branch_and_bound(items, cap)
            self.assertEqual(r["total_value"], float(best),
                             f"items={items} cap={cap}")

    def test_empty_choice_on_tight_capacity(self):
        r = op.branch_and_bound([(10, 5)], 1)
        self.assertEqual(r["chosen"], [])
        self.assertEqual(r["total_value"], 0.0)

    def test_negative_inputs_rejected(self):
        with self.assertRaises(ValueError):
            op.branch_and_bound([(-1, 5)], 10)


class TestTabuSearch(unittest.TestCase):
    def test_sorts_a_switch_cost_sequence(self):
        # minimizing sum |x_i - x_{i+1}|: the sorted order is optimal
        def cost(order):
            return sum(abs(a - b) for a, b in zip(order, order[1:]))

        def neighbors(order):
            out = []
            for i in range(len(order) - 1):
                for j in range(i + 1, len(order)):
                    cand = list(order)
                    cand[i], cand[j] = cand[j], cand[i]
                    out.append(cand)
            return out

        r = op.tabu_search(cost, [4, 1, 3, 2, 5], neighbors, rounds=40)
        self.assertEqual([int(x) for x in r["best"]], [1, 2, 3, 4, 5])
        self.assertEqual(r["score"], 4.0)

    def test_deterministic(self):
        def cost(order):
            return sum(abs(a - b) for a, b in zip(order, order[1:]))

        def neighbors(order):
            return [order[1:] + order[:1], order[:-1]]

        a = op.tabu_search(cost, [3, 1, 2], neighbors, rounds=10)
        b = op.tabu_search(cost, [3, 1, 2], neighbors, rounds=10)
        self.assertEqual(a, b)

    def test_maximize_mode(self):
        def score(order):
            return order[0]

        r = op.tabu_search(score, [1, 2, 3], lambda o: [[o[0] + 1, *o[1:]]],
                           rounds=5, minimize=False)
        self.assertEqual(r["score"], 6.0)
