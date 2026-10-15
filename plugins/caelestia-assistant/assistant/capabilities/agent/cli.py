"""agent.cli — the `caelestia-assist agent` command surface.

    caelestia-assist agent "why does my dock blur" --simulate
    caelestia-assist agent "clean my downloads"              # prompts y/N
    caelestia-assist agent "..." --json                      # machine output
    echo '{"op": "agent_plan", "text": "..."}' | caelestia-assist api

Consent policy: every STATE_CHANGING node prints what it will do and asks
y/N on stdin. --yes-once is deliberately NOT provided: one approval per
graph, per run, is the honest minimum.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from . import goals
from .clarify import EntropyPicker, Question
from .engine import Agent

RISK_ORDER = {"READ_ONLY": 0, "STATE_CHANGING": 1, "PRIVILEGED": 2,
              "DESTRUCTIVE": 3}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="assistant.capabilities.agent",
        description="Goal-decomposing agent over every assistant layer — "
                    "plan, simulate, consent, act, learn")
    p.add_argument("goal", nargs="*", help="the request in plain words")
    p.add_argument("--simulate", action="store_true",
                   help="project the task graph without executing anything")
    p.add_argument("--graph", action="store_true",
                   help="print the raw task graph (nodes + dependencies)")
    p.add_argument("--questions", action="store_true",
                   help="propose the highest-information clarifying questions")
    p.add_argument("--tidy-root", default="~/Downloads",
                   help="root for clean_files goals (default ~/Downloads)")
    p.add_argument("--json", action="store_true", help="machine output")
    return p


def _stdin_consent(node: Dict[str, Any]) -> bool:
    print(f"\nAbout to: {node['title']}")
    print(f"  action: {node['action']}   risk: {node['risk']}")
    if not sys.stdin.isatty():
        print("  non-interactive stdin: refusing (use --simulate or the JSON "
              "bridge with an explicit consent round-trip)")
        return False
    try:
        answer = input("  approve? [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")


def _print_graph(graph: Dict[str, Any]) -> None:
    print("task graph:")
    for node in graph["nodes"]:
        deps = f"  <- {', '.join(node['depends'])}" if node["depends"] else ""
        gate = " [CONSENT]" if node["consent_required"] else ""
        print(f"  {node['id']:<4} ({node['risk']:<14}) {node['action']}"
              f"{gate}{deps}")
        print(f"       {node['title']}")
    cp = graph.get("critical_path") or []
    if cp:
        print(f"critical path: {' -> '.join(cp)}")


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    text = " ".join(args.goal).strip()
    if not text:
        build_parser().print_help()
        return 2

    if args.questions:
        cls = goals.classify_goal(text)
        picker = EntropyPicker(
            {cls["goal"]: max(1.0, cls["score"]),
             **{g: 1.0 for g in cls["runners_up"]}})
        questions = _built_in_questions()
        best = picker.best_question(questions)
        if best is None:
            print("no question worth asking — the goal is already determined")
        else:
            print(f"{best.text}")
            for label in best.answers:
                print(f"  - {label}")
        return 0

    if args.simulate:
        agent = Agent()
        result = agent.simulate(text)
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            if "error" in result:
                print(f"agent: {result['error']}")
                return 1
            print("simulation (nothing executed):")
            for step in result["projection"]:
                gate = " [would ASK you]" if step["consent_required"] else ""
                print(f"  {step['id']:<4} {step['action']:<16} {step['would']}{gate}")
            cp = result.get("critical_path") or []
            if cp:
                print(f"critical path: {' -> '.join(cp)}")
            if result["ambiguous"]:
                print("\nnote: the goal split was ambiguous — run "
                      "`caelestia-assist agent \"...\" --questions`")
        return 0

    if args.graph:
        graph = goals.decompose(text)
        if args.json:
            print(json.dumps(graph, indent=2, default=str))
        else:
            if "error" in graph:
                print(f"agent: {graph['error']}")
                return 1
            _print_graph(graph)
        return 0

    agent = Agent(
        consent_fn=_stdin_consent,
        clarify_questions=[],
        base_ctx={"tidy_root": args.tidy_root},
    )
    result = agent.execute(text)
    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return 0
    if "error" in result:
        print(f"agent: {result['error']}")
        return 1
    print("execution report:")
    for node in result["nodes"]:
        status = node["status"].upper()
        print(f"  [{status:<8}] {node['action']}: {node['title']}")
        res = node.get("result") or {}
        if isinstance(res, dict):
            for key in ("plan", "explanation", "rendered", "note"):
                if key in res and res[key]:
                    body = res[key]
                    if isinstance(body, list):
                        for line in body[:6]:
                            print(f"      {line}")
                    else:
                        print(f"      {str(body)[:400]}")
            if "error" in res:
                print(f"      error: {res['error']}")
            if node["status"] == "skipped":
                print(f"      skipped: {node.get('skip_reason')}")
    print(f"\ntotals: {result['summary']}")
    return 0


def _built_in_questions() -> List[Question]:
    return [
        Question(
            "is_broken", "Is something on your desktop misbehaving, or do "
            "you want a change to how it looks?",
            {"yes": {"diagnose_issue": 1.0, "research_topic": 0.6,
                     "change_settings": 0.0, "clean_files": 0.0},
             "no": {"diagnose_issue": 0.0, "research_topic": 0.2,
                    "change_settings": 1.0, "clean_files": 1.0}},
        ),
        Question(
            "settings_vs_files", "Should the shell's settings change, or do "
            "you want files organised?",
            {"settings": {"change_settings": 1.0, "clean_files": 0.0},
             "files": {"change_settings": 0.0, "clean_files": 1.0}},
        ),
        Question(
            "which_log", "Do you have a log or error text to scan?",
            {"yes": {"diagnose_issue": 1.0}, "no": {"research_topic": 1.0}},
        ),
    ]


if __name__ == "__main__":
    raise SystemExit(main())
