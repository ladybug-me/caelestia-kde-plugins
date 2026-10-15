"""CLI for the retrieval layer: search the offline corpus index.

Usage:
    python3 -m assistant.capabilities.retrieval.cli search "query text" [-k N] [--json] [--index PATH]
    python3 -m assistant.capabilities.retrieval.cli stats [--index PATH]
    python3 -m assistant.capabilities.retrieval.cli cbr --cases cases.json "problem text" [-k N] [--json]
    python3 -m assistant.capabilities.retrieval.cli analog --rule RULE_ID [--top N] [--json]

Reads only the prebuilt JSON index (search/stats), a caller-supplied
case file (cbr), or the shipped rules.d corpus (analog); fully offline,
fully read-only — the cbr surface never retains (no ledger is touched).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from . import DEFAULT_INDEX_PATH
from .search import Searcher


def _format_cycle(result: dict) -> str:
    """Human-readable rendering of one cbr_cycle result (text mode)."""
    lines: List[str] = [f"CBR cycle: {result['status']}"]
    if result["status"] == "ABSTAINED":
        lines.append(f"  {result['abstain_reason']}")
        return "\n".join(lines)
    for hit in result["retrieve"]:
        lines.append(f"  retrieved: {hit['case_id']} (BM25 score {hit['score']})")
    reuse = result["reuse"]
    fired = ", ".join(rule["id"] for rule in reuse["rules_fired"]) or "none"
    lines.append(f"  reuse: {reuse['status']} (adaptation rules fired: {fired})")
    if reuse.get("note"):
        lines.append(f"  note: {reuse['note']}")
    for line in (reuse["resolution"] or "").split("\n"):
        if line.strip():
            lines.append(f"    - {line.strip()}")
    revise = result.get("revise")
    if isinstance(revise, dict) and revise.get("status") in ("CONFIRMED", "REVISE"):
        failing = f" (failing steps: {revise['failed_steps']})" if revise["failed_steps"] else ""
        lines.append(f"  revise: {revise['status']}{failing}")
        for step in revise["steps"]:
            mark = "ok" if step["passed"] else f"REVISE: {step['reason']}"
            lines.append(f"    {step['i']}. {step['text']}  [{mark}]")
    return "\n".join(lines)


def _format_results(results: List[dict]) -> str:
    if not results:
        return "No corpus document matched. The honest answer is: nothing in the local docs/issues fits."
    lines: List[str] = []
    for i, hit in enumerate(results, start=1):
        lines.append(f"[{i}] {hit['doc_id']} — {hit['title']}  (score {hit['score']})")
        lines.append(f"    source: {hit['source']}")
        lines.append(f"    snippet: {hit['snippet']}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m assistant.capabilities.retrieval.cli",
        description="Offline BM25 search over the caelestia docs/resolved-issues corpus.",
    )
    parser.add_argument("--index", default=str(DEFAULT_INDEX_PATH), help="path to bm25.json")
    sub = parser.add_subparsers(dest="cmd", required=True)

    search_cmd = sub.add_parser("search", help="rank corpus docs for a query")
    search_cmd.add_argument("query", help="query text")
    search_cmd.add_argument("-k", type=int, default=5, help="max results (default 5)")
    search_cmd.add_argument("--json", action="store_true", help="emit JSON instead of text")

    sub.add_parser("stats", help="print index statistics")

    # exponential-build 4.1: the disk-backed bounded-memory index
    di = sub.add_parser("disk-index", help="index a folder tree under a RAM "
                                           "ceiling (external merge sort, "
                                           "measured footprint)")
    di.add_argument("root", help="the folder tree to index")
    di.add_argument("--out", required=True, help="output index directory")
    di.add_argument("--ceiling", type=float, default=16,
                    help="RAM ceiling in MB for index structures (default 16)")
    di.add_argument("--max-files", type=int, default=20000)

    ds = sub.add_parser("disk-search", help="search a disk index "
                                            "(metadata-only in RAM)")
    ds.add_argument("query", help="query text")
    ds.add_argument("-k", type=int, default=5)
    ds.add_argument("--json", action="store_true")
    ds.add_argument("--index-dir", required=True)

    # exponential-build 3, A3: the CBR cycle over a caller-supplied case
    # file — read-only (retrieve/reuse/revise only; no ledger is touched,
    # so retain never runs from the CLI).
    cbr_cmd = sub.add_parser("cbr", help="run the Case-Based Reasoning "
                                         "cycle over a caller-supplied "
                                         "case file (read-only)")
    cbr_cmd.add_argument("query", help="the problem text to solve from past cases")
    cbr_cmd.add_argument("--cases", required=True,
                         help="JSON file holding the case base (a list of "
                              "cases; schema in retrieval/cbr.py)")
    cbr_cmd.add_argument("-k", type=int, default=1,
                         help="max cases retrieved (default 1: only the best is reused)")
    cbr_cmd.add_argument("--json", action="store_true", help="emit JSON instead of text")

    # exponential-build 3, A4: structure-mapping analogical lookup over
    # the shipped rules.d corpus — a signal to read alongside BM25.
    analog = sub.add_parser("analog", help="structure-mapping analogical "
                                           "lookup over the diagnostics "
                                           "rules (read-only)")
    analog.add_argument("--rule", required=True,
                        help="rule id whose matcher graph is the query problem")
    analog.add_argument("--top", type=int, default=3)
    analog.add_argument("--rules-dir", default=None,
                        help="rules.d directory (default: the shipped one)")
    analog.add_argument("--json", action="store_true", help="emit JSON instead of text")

    args = parser.parse_args(argv)

    if args.cmd == "disk-index":
        from . import diskindex
        report = diskindex.build_disk_index(args.root, args.out,
                                            ram_ceiling_mb=args.ceiling,
                                            max_files=args.max_files)
        fp = report["footprint"]
        print(f"disk index: {report['n_docs']} docs, {report['n_terms']} "
              f"terms, {fp['runs_spilled']} run(s) spilled")
        print(f"footprint (measured): budget {fp['run_budget_chars']} chars "
              f"({fp['ram_ceiling_mb']} MB ceiling); budget respected: "
              f"{fp['budget_respected_by_construction']}")
        if fp.get("hwm_growth_kb") is not None:
            print(f"process peak RSS growth (VmHWM): {fp['hwm_growth_kb']} kB "
                  f"(within ceiling: {fp['hwm_growth_within_ceiling']})")
        print(fp.get("note", ""))
        return 0
    if args.cmd == "disk-search":
        from . import diskindex
        searcher = diskindex.DiskSearcher(args.index_dir)
        results = searcher.search(args.query, k=args.k)
        if args.json:
            print(json.dumps({"query": args.query, "results": results},
                             ensure_ascii=False, indent=2))
        else:
            if not results:
                print("no match (honest empty answer)")
            for i, hit in enumerate(results, start=1):
                print(f"[{i}] {hit['path']}  (score {hit['score']})")
        return 0

    if args.cmd == "cbr":
        from . import cbr
        try:
            cases = json.loads(Path(args.cases).read_bytes().decode("utf-8"))
        except (OSError, ValueError) as exc:
            print(f"error: cannot load case file {args.cases}: {exc}", file=sys.stderr)
            return 2
        try:
            result = cbr.cbr_cycle(cases, args.query, k=args.k)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(_format_cycle(result))
        return 0

    if args.cmd == "analog":
        from assistant.capabilities.diagnostics.engine import load_rules
        from . import structure_mapping as sm
        try:
            rules = load_rules(Path(args.rules_dir) if args.rules_dir else None)
        except (OSError, ValueError) as exc:
            print(f"error: cannot load rules: {exc}", file=sys.stderr)
            return 2
        rules_by_id = {rule["id"]: rule for rule in rules}
        if args.rule not in rules_by_id:
            print(f"error: no rule {args.rule!r} in the rules corpus", file=sys.stderr)
            return 2
        query_graph = sm.rule_match_graph(rules_by_id[args.rule])
        problems = [{"id": rule["id"], "problem": sm.rule_match_graph(rule)}
                    for rule in rules if rule["id"] != args.rule]
        result = sm.surface_candidates(problems, query_graph, top=args.top)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            if result["status"] == "abstained":
                print(f"analog: abstained — {result['reason']}")
            else:
                print("analogical candidates (read ALONGSIDE BM25/NCD, never instead):")
                for i, hit in enumerate(result["results"], start=1):
                    print(f"[{i}] {hit['id']}  (structural score {hit['structural_score']})")
                    print(f"    {hit['why']}")
        return 0

    index_path = args.index
    if not Path(index_path).is_file():
        print(f"error: index not found at {index_path}; run python3 -m assistant.capabilities.retrieval.build", file=sys.stderr)
        return 2

    searcher = Searcher.from_file(index_path)

    if args.cmd == "stats":
        terms = len(searcher.df)
        tokens = sum(doc.get("dl", 0) for doc in searcher.docs)
        print(f"index: {index_path}")
        print(f"docs: {searcher.n_docs}, unique terms: {terms}, tokens: {tokens}, avgdl: {searcher.avgdl:.1f}")
        print(f"params: k1={searcher.k1}, b={searcher.b}")
        for doc in searcher.docs:
            print(f"  {doc['id']:16s} {doc.get('title', '')}")
        return 0

    results = searcher.search(args.query, k=args.k)
    if args.json:
        print(json.dumps({"query": args.query, "results": results}, ensure_ascii=False, indent=2))
    else:
        print(_format_results(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
