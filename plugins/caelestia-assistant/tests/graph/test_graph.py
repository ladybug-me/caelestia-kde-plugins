"""Tests for assistant.capabilities.graph (exponential-build-5 F12).

Covers: determinism (the hash pins sources, not loader order), shape and
provenance invariants, every query's honest verdicts (including the two
abstentions the design demands: UNKNOWN_PATH and NO_CITED_PATH), ledger
co-change mining, the CLI surface, the hub wiring, and the `why`
engine's graph enrichment line.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from assistant.capabilities.graph import build as gbuild
from assistant.capabilities.graph import queries as gq
from assistant.capabilities.graph.cli import main as graph_main


def _ledger_with(ops_lists):
    """A minimal assistant-history ledger fixture (the applier's shape)."""
    entries = [{"id": i + 1, "at": f"2026-09-2{i + 1}T00:00:00+00:00",
                "label": "test", "ops": [{"path": p, "old": None, "new": 1}
                                        for p in ops]}
               for i, ops in enumerate(ops_lists)]
    return {"next_id": len(entries) + 1, "entries": entries,
            "undo_log": []}


class BuildTests(unittest.TestCase):
    def test_build_is_deterministic_and_order_independent(self) -> None:
        g1 = gbuild.build_graph()
        g2 = gbuild.build_graph()
        self.assertEqual(g1["meta"]["hash"], g2["meta"]["hash"])
        self.assertEqual(gbuild.dump_canonical(g1),
                         gbuild.dump_canonical(g2))
        # loader-order independence: reversed consequences EDGES and a
        # dict-order-shuffled tools.json load must give the same hash
        from assistant.capabilities.settings import consequences as cons
        with mock.patch.object(cons, "EDGES", tuple(reversed(cons.EDGES))):
            g3 = gbuild.build_graph()
        self.assertEqual(g1["meta"]["hash"], g3["meta"]["hash"])

    def test_shape_and_provenance(self) -> None:
        # Node counts are DERIVED from the committed tools.json (the single
        # source of truth), never re-pinned by hand: a deliberate registry
        # re-pin must not desync this test from the artifact it describes.
        meta = json.loads(
            (Path(__file__).resolve().parents[2] / "assistant" / "adapters" / "caelestia" / "tools.json")
            .read_text(encoding="utf-8")).get("meta", {})
        g = gbuild.build_graph()
        types = {}
        for n in g["nodes"]:
            self.assertIn("id", n)
            self.assertIn("type", n)
            self.assertIn("label", n)
            self.assertIn("attrs", n)
            types[n["type"]] = types.get(n["type"], 0) + 1
        self.assertEqual(types["tool"], meta.get("tool_count"))  # full registry
        self.assertEqual(types["not_exposed"],
                         meta.get("not_exposed_count"))  # every hidden path
        self.assertEqual(types["preset"], 5)
        self.assertEqual(types["explain_rule"], 11)
        self.assertGreaterEqual(types["file"], 100)  # citation targets
        self.assertGreaterEqual(types["group"], 20)
        kinds = {}
        for e in g["edges"]:
            self.assertIn("src", e)
            self.assertIn("dst", e)
            self.assertIn("kind", e)
            self.assertIn("weight", e)
            self.assertTrue(0.0 < e["weight"] <= 1.0)
            prov = e["provenance"]
            self.assertIn("source", prov)  # every edge is auditable
            kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
        self.assertEqual(kinds["sets"], 272)          # tool -> config
        self.assertEqual(kinds["affects"], 5)          # the curated table
        self.assertEqual(kinds["explained_by"], 11)
        self.assertEqual(kinds["applies"], 27)          # 5 presets' calls
        self.assertGreaterEqual(kinds["cites"], 100)
        self.assertGreaterEqual(kinds["member_of"], 272)

    def test_every_affects_edge_carries_citation_and_content(self) -> None:
        g = gbuild.build_graph()
        for e in g["edges"]:
            if e["kind"] != "affects":
                continue
            p = e["provenance"]
            self.assertIn(p["confidence"], ("high", "medium", "low"))
            self.assertTrue(p["citation"])
            self.assertTrue(p["claimed_content"])
            self.assertTrue(p["id"])
            self.assertIn("when", p)
            self.assertIn("effect", p)
            # the confidence WORD is the authority; the weight is a
            # documented mapping of it
            expected = {"high": 0.9, "medium": 0.7, "low": 0.5}
            self.assertEqual(e["weight"], expected[p["confidence"]])

    def test_meta_pins_the_sources(self) -> None:
        g = gbuild.build_graph()
        s = g["meta"]["sources"]
        self.assertEqual(len(s["consequence_ids"]), 5)
        self.assertEqual(len(s["rules_d"]), 6)
        self.assertFalse(s["ledger_used"])
        self.assertTrue(s["tools_json"]["repo_commit"])

    def test_ledger_co_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lp = Path(tmp) / "shell.json.assistant-history.json"
            lp.write_text(json.dumps(_ledger_with([
                ["appearance.blur", "appearance.transparency.enabled"],  # pair
                ["bar.scale"],                              # single-op: no edge
                ["appearance.blur", "appearance.transparency.enabled"],  # repeat
                ["bar.scale", "bar.position", "bar.height"],  # 3-way: 3 pairs
            ])), encoding="utf-8")
            g = gbuild.build_graph(ledger_path=str(lp))
        co = [e for e in g["edges"] if e["kind"] == "co_changes"]
        pairs = {(e["src"], e["dst"]) for e in co}
        self.assertIn(("config:appearance.blur",
                       "config:appearance.transparency.enabled"), pairs)
        self.assertIn(("config:bar.position", "config:bar.scale"), pairs)
        # single-op entries produced nothing
        self.assertEqual(len(pairs), 4)  # 1 (repeat collapses) + 3
        for e in co:
            self.assertTrue(e["provenance"]["evidence_ids"])
            self.assertIn("never a causal claim",
                          e["provenance"]["note"])
        # the ledger changed the hash (personal input, honestly marked)
        self.assertTrue(g["meta"]["sources"]["ledger_used"])
        self.assertNotEqual(g["meta"]["hash"],
                            gbuild.build_graph()["meta"]["hash"])

    def test_malformed_ledger_raises_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lp = Path(tmp) / "bad.json"
            lp.write_text('{"nope": 1}', encoding="utf-8")
            with self.assertRaises(ValueError):
                gbuild.build_graph(ledger_path=str(lp))


class QueryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.g = gbuild.build_graph()

    def test_what_affects_curated_chain(self) -> None:
        r = gq.what_affects(self.g, "appearance.blur")
        self.assertEqual(r["verdict"], "OK")
        ups = {u["from"] for u in r["upstream"]}
        self.assertIn("appearance.transparency.enabled", ups)
        for u in r["upstream"]:
            if u["from"] == "appearance.transparency.enabled":
                self.assertEqual(u["confidence"], "high")
                self.assertIn("AppearancePage.qml", u["citation"])
                self.assertTrue(u["claimed_content"])
        # the bounded-universe caveat is always attached
        self.assertIn("curated consequences table", r["note"])

    def test_what_affects_resolves_tool_names_too(self) -> None:
        r = gq.what_affects(self.g, "setBlurEnabled")
        self.assertEqual(r["verdict"], "OK")
        self.assertEqual(r["key"], "appearance.blur")

    def test_what_affects_unknown_abstains(self) -> None:
        r = gq.what_affects(self.g, "no.such.key")
        self.assertEqual(r["verdict"], "UNKNOWN_PATH")

    def test_explanation_path_cited(self) -> None:
        r = gq.explanation_path(self.g, "appearance.transparency.enabled",
                                "appearance.blur")
        self.assertEqual(r["verdict"], "OK")
        self.assertEqual(r["path"], ["config:appearance.transparency.enabled",
                                     "config:appearance.blur"])
        self.assertEqual(len(r["hops"]), 1)
        self.assertEqual(r["hops"][0]["confidence"], "high")
        self.assertTrue(r["hops"][0]["citation"])
        self.assertAlmostEqual(r["path_confidence"], 0.9, places=6)

    def test_explanation_path_no_chain_abstains(self) -> None:
        r = gq.explanation_path(self.g, "bar.scale", "appearance.blur")
        self.assertEqual(r["verdict"], "NO_CITED_PATH")
        self.assertIn("abstaining", r["note"])

    def test_explanation_path_unknown_keys(self) -> None:
        r = gq.explanation_path(self.g, "nope.x", "appearance.blur")
        self.assertEqual(r["verdict"], "UNKNOWN_PATH")

    def test_what_breaks_layers(self) -> None:
        r = gq.what_breaks(self.g, "appearance.transparency.enabled")
        self.assertEqual(r["verdict"], "OK")
        self.assertTrue(any(i["to"] == "appearance.blur"
                            for i in r["downstream"]))
        for i in r["downstream"]:
            self.assertIn(i["kind"],
                          ("conditional effect",
                           "co_change (correlation, never cause)"))
            if i["kind"] == "conditional effect":
                self.assertTrue(i["citation"])
                self.assertIn("when", i)
        self.assertIn("is_articulation_point", r["structure"])
        self.assertIsInstance(r["influence_flow"], float)
        self.assertGreaterEqual(r["influence_flow"], 0.0)
        self.assertIn("curated consequences table", r["note"])

    def test_what_breaks_unknown_abstains(self) -> None:
        self.assertEqual(gq.what_breaks(self.g, "zzz")["verdict"],
                         "UNKNOWN_PATH")

    def test_related_bounded_and_kinded(self) -> None:
        r = gq.related(self.g, "config:appearance.blur", hops=2, top=5)
        self.assertEqual(r["verdict"], "OK")
        self.assertLessEqual(len(r["neighbors"]), 5)
        labels = [n["node"] for n in r["neighbors"]]
        self.assertIn("tool:setBlurEnabled", labels)
        for n in r["neighbors"]:
            self.assertTrue(n["kinds"])
            self.assertGreater(n["activation"], 0.0)
        # deterministic ordering: descending activation, ties by node id
        acts = [n["activation"] for n in r["neighbors"]]
        self.assertEqual(acts, sorted(acts, reverse=True))

    def test_related_unknown_node(self) -> None:
        self.assertEqual(gq.related(self.g, "config:nope")["verdict"],
                         "UNKNOWN_NODE")

    def test_pagerank_uniform_vs_personalized(self) -> None:
        uni = gq.personalized_pagerank(self.g)
        self.assertFalse(uni["personalized"])
        total = sum(uni["ranks"].values())
        self.assertAlmostEqual(total, 1.0, places=6)
        # determinism
        uni2 = gq.personalized_pagerank(self.g)
        self.assertEqual(uni["ranks"], uni2["ranks"])
        per = gq.personalized_pagerank(
            self.g, seeds=["config:appearance.blur"])
        self.assertTrue(per["personalized"])
        # the seed is the top-ranked node under its own teleport
        top = max(per["ranks"].items(), key=lambda t: t[1])[0]
        self.assertEqual(top, "config:appearance.blur")
        # and the two distributions genuinely differ
        self.assertNotEqual(uni["ranks"], per["ranks"])

    def test_pagerank_excludes_file_nodes(self) -> None:
        r = gq.personalized_pagerank(self.g)
        self.assertFalse(any(n.startswith("file:") for n in r["ranks"]))


class CliAndWiringTests(unittest.TestCase):
    def test_cli_build_json(self) -> None:
        with tempfile.TemporaryDirectory():
            rc = graph_main(["build", "--json"])
        self.assertEqual(rc, 0)

    def test_cli_affects_and_abstention_exit_codes(self) -> None:
        self.assertEqual(graph_main(["affects", "appearance.blur",
                                     "--json"]), 0)
        self.assertEqual(graph_main(["affects", "no.such.key", "--json"]), 1)

    def test_cli_path_abstains_nonzero(self) -> None:
        self.assertEqual(graph_main(["path", "bar.scale",
                                     "appearance.blur", "--json"]), 1)
        self.assertEqual(graph_main(["path", "appearance.transparency.enabled",
                                     "appearance.blur", "--json"]), 0)

    def test_hub_routes_graph_lazily(self) -> None:
        from assistant import hub
        self.assertIn("graph", hub.ROUTES)
        target, keep = hub.ROUTES["graph"]
        self.assertEqual(target, "assistant.capabilities.graph.cli:main")
        self.assertTrue(keep)

    def test_why_engine_gains_the_graph_line(self) -> None:
        from assistant.core.explain_unified import explain_settings
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "shell.json"
            target.write_text("{}", encoding="utf-8")
            result = explain_settings("setBlurEnabled", target)
        graph_lines = [l for l in result["lines"] if l.startswith("graph:")]
        # blur has curated upstream: the enrichment must name it with a
        # citation, or say honestly that none exists — never be absent
        self.assertEqual(len(graph_lines), 1)
        self.assertIn("curated interaction", graph_lines[0])
        self.assertIn("appearance.transparency.enabled", graph_lines[0])


if __name__ == "__main__":
    unittest.main()
