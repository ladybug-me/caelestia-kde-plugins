"""assistant.capabilities.graph.cli — the graph verb (exponential-build-5 F12).

``caelestia-assist graph <subcommand>`` — read-only queries over the
shell knowledge graph. Every subcommand accepts ``--json``; verdicts are
honest (``OK`` / ``UNKNOWN_PATH`` / ``NO_CITED_PATH`` / ``UNKNOWN_NODE``),
and an abstaining verdict exits 1 with the reason on stdout, like a
search that found nothing — never a fabricated answer with exit 0.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from . import build as _build
from . import queries as _q

__all__ = ["main"]


def _emit(result: dict, as_json: bool) -> int:
    if as_json:
        print(json.dumps(result, sort_keys=True, indent=1))
        return 0 if result.get("verdict", "OK") == "OK" else 1
    verdict = result.get("verdict", "OK")
    if verdict != "OK":
        print(f"verdict: {verdict}")
        for k in ("key", "src", "dst", "node"):
            if k in result:
                print(f"{k}: {result[k]}")
        note = result.get("note")
        if note:
            print(f"note: {note}")
        return 1
    print(json.dumps(result, sort_keys=True, indent=1))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "graph":  # hub keep_name passes the verb token
        argv = argv[1:]
    ap = argparse.ArgumentParser(
        prog="caelestia-assist graph",
        description="the shell knowledge graph: rebuildable from "
                    "tools.json, the curated consequences table, the "
                    "diagnostic rules and (optionally) your approved "
                    "apply ledger; every edge carries provenance")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser(
        "build", help="rebuild the graph and print its hash + counts")
    p_build.add_argument("--ledger", default=None,
                         help="optional approved-history ledger to mine "
                              "co-changes from (read-only)")
    p_build.add_argument("--json", action="store_true")
    p_build.add_argument("--dump", action="store_true",
                         help="print the full canonical graph (large)")

    p_aff = sub.add_parser(
        "affects", help="what curated interactions act ON this key?")
    p_aff.add_argument("key", help="config path (appearance.blur) or tool "
                                   "name (setBlurEnabled)")
    p_aff.add_argument("--json", action="store_true")

    p_brk = sub.add_parser(
        "breaks", help="what changes downstream if this key moves?")
    p_brk.add_argument("key")
    p_brk.add_argument("--json", action="store_true")

    p_path = sub.add_parser(
        "path", help="shortest CITED explanation path between two keys "
                     "(Dijkstra on -log confidence; abstains without one)")
    p_path.add_argument("src")
    p_path.add_argument("dst")
    p_path.add_argument("--json", action="store_true")

    p_rel = sub.add_parser(
        "related", help="spreading-activation neighborhood of a node")
    p_rel.add_argument("node", help="node id (tool:..., config:..., "
                                    "group:..., preset:...)")
    p_rel.add_argument("--hops", type=int, default=2)
    p_rel.add_argument("--top", type=int, default=12)
    p_rel.add_argument("--json", action="store_true")

    p_rank = sub.add_parser(
        "rank", help="PageRank over the semantic graph (personalized "
                     "when --seeds are given)")
    p_rank.add_argument("--seeds", default=None,
                        help="comma-separated node ids to personalize "
                             "the teleport with")
    p_rank.add_argument("--top", type=int, default=15)
    p_rank.add_argument("--json", action="store_true")

    args = ap.parse_args(argv)
    if args.cmd == "build":
        g = _build.build_graph(ledger_path=args.ledger)
        counts: dict = {}
        for n in g["nodes"]:
            counts[n["type"]] = counts.get(n["type"], 0) + 1
        kinds: dict = {}
        for e in g["edges"]:
            kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
        summary = {"verdict": "OK", "hash": g["meta"]["hash"],
                   "sources": g["meta"]["sources"],
                   "node_counts": counts, "edge_counts": kinds}
        if args.dump:
            print(_build.dump_canonical(g))
            return 0
        return _emit(summary, args.json)
    g = _build.build_graph()
    if args.cmd == "affects":
        return _emit(_q.what_affects(g, args.key), args.json)
    if args.cmd == "breaks":
        return _emit(_q.what_breaks(g, args.key), args.json)
    if args.cmd == "path":
        return _emit(_q.explanation_path(g, args.src, args.dst), args.json)
    if args.cmd == "related":
        return _emit(_q.related(g, args.node, hops=args.hops,
                                top=args.top), args.json)
    if args.cmd == "rank":
        seeds = [s.strip() for s in args.seeds.split(",")] if args.seeds else None
        res = _q.personalized_pagerank(g, seeds=seeds)
        res["verdict"] = "OK"
        res["ranks"] = dict(list(res["ranks"].items())[:args.top])
        return _emit(res, args.json)
    ap.error(f"unknown subcommand {args.cmd!r}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
