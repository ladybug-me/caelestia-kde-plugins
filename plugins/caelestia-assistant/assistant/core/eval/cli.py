"""caelestia-assist eval — the measurement arena (issue #120 discipline).

Usage:
  caelestia-assist eval [suite] [--json] [--sealed]

Suites: routing | nlplan | abstention | diagnosis | calibration |
footprint | all (default: all dev suites present).

--sealed runs the sealed split. The sealed set exists to be looked at ONLY
at stage boundaries; every --sealed invocation prints a reminder to record
the look in the build state file, and the operator is trusted to do so
(the alternative — network/authors' logs — is worse).
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from .engine import SUITES, run_suite


def main(argv: Optional[List[str]] = None) -> int:
    if argv and argv[0] == "grow":
        return _cmd_grow(argv[1:])
    ap = argparse.ArgumentParser(
        prog="caelestia-assist eval",
        description="the eval arena: seeded, interval-reporting measurement suites",
    )
    ap.add_argument("suite", nargs="?", default="all",
                    choices=list(SUITES) + ["all"],
                    help="which suite to run (default: all present)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--sealed", action="store_true",
                    help="run the SEALED split (record the look in the build state)")
    args = ap.parse_args(argv)

    split = "sealed" if args.sealed else "dev"
    if args.sealed:
        print("SEALED SET RUN — record this look in the build state file "
              "(limit 6 before re-authoring half the set)", file=sys.stderr)

    try:
        report = run_suite(args.suite, split=split)
    except FileNotFoundError as exc:
        print(f"eval: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=1, default=str))
    else:
        _render(report)
    return 0


def _cmd_grow(argv: List[str]) -> int:
    """F26: the arena grows from experience, UNDER REVIEW. mine/list
    work on the user-state quarantine (the only writes are to user
    state); promote appends ONE reviewed candidate to a dev-set file
    the caller names — sealed sets refuse by name."""
    import argparse as _ap
    from assistant.capabilities.brain import state as brain_state
    from . import grow

    ap = _ap.ArgumentParser(prog="caelestia-assist eval grow")
    sub = ap.add_subparsers(dest="action", required=True)
    sub.add_parser("mine", help="mine near-threshold/approved phrases "
                                "into the quarantine (writes user state "
                                "only)")
    sub.add_parser("list", help="read-only: the quarantined candidates")
    prom = sub.add_parser("promote", help="append ONE reviewed candidate "
                                          "to a dev set (sealed sets "
                                          "refuse)")
    prom.add_argument("id", help="the candidate id (see list)")
    prom.add_argument("--dev-set", required=True, metavar="PATH",
                      help="the dev-set JSON file to append to")
    args = ap.parse_args(argv)

    state = brain_state.load()
    if args.action == "list":
        print("\n".join(grow.render_lines(
            state.get(grow.QUARANTINE_KEY) or [])))
        return 0
    if args.action == "mine":
        # the learner's examples live under cortex_cli's LEARN_KEY
        # ("cortex_learn"); mine_candidates reads a normalized shape
        mined = grow.mine_candidates(
            {"cortex_learner": state.get("cortex_learn") or {},
             "cortex_review": state.get("cortex_review") or []})
        stats = grow.save_quarantine(state, mined)
        brain_state.save(state)
        print(f"mined {stats['added']} new candidate(s) "
              f"({stats['total']} quarantined)")
        print("review with: eval grow list; promote explicitly")
        return 0
    try:
        result = grow.promote(state, args.id, args.dev_set_path)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    brain_state.save(state)
    print(f"promoted {result['promoted']} as {result['as']} in "
          f"{result['dev_set']}")
    print("the arena ratchet now includes your reviewed phrase; run "
          "'eval routing' to measure")
    return 0


def _render(report: dict) -> None:
    suites = report.get("suites", {report.get("suite", "?"): report})
    for name, rep in suites.items():
        print(f"== {name} (split {rep.get('split', '?')}, n={rep.get('n', '?')}) ==")
        for metric, val in rep.get("metrics", {}).items():
            if isinstance(val, tuple):
                try:
                    print(f"  {metric}: {val[0]:.4f}  "
                          f"[{val[1]:.4f}, {val[2]:.4f}]")
                except (ValueError, TypeError):
                    # ("n/a", "n/a", "n/a") — a metric with no observations
                    print(f"  {metric}: n/a")
            else:
                print(f"  {metric}: {val}")
        mb = rep.get("n_breaks")
        if mb is not None:
            print(f"  metamorphic breaks: {mb}")
        cw = rep.get("confident_wrong")
        if cw:
            print(f"  confident-wrong: {cw['count']}/{cw['of_routed']} routed"
                  + (f" ({cw['rate']:.1%})" if cw['rate'] is not None else ""))
        wb = rep.get("within_budget")
        if wb:
            print(f"  within budget: {wb}")
        failures = rep.get("failures")
        if failures:
            print(f"  failures ({len(failures)}):")
            for f in failures[:8]:
                print(f"    - {f.get('id')}: {f.get('text')!r} -> "
                      f"{f.get('top1') or f.get('got') or f.get('got_tool')} "
                      f"(want {f.get('accept') or f.get('expect') or f.get('expect_tool')})")
            if len(failures) > 8:
                print(f"    ... and {len(failures) - 8} more")
        if rep.get("note"):
            print(f"  note: {rep['note']}")
        print()


if __name__ == "__main__":
    raise SystemExit(main())
