"""settings.gen_adapter — the registry-generation adapter interface
(phase 2.6).

The contract this formalizes (previously implicit in build_registry's
CLI + the drift test):

1. **ONE canonical serialization**: every adapter's dict output renders
   through ``canonical_bytes`` — ``json.dumps(data, indent=2,
   ensure_ascii=False) + "\\n"`` — and that rendering is THE artifact.
   The committed tools.json, the regeneration CLI, and the drift guard
   all use this one function, so "byte-identical output" is a property
   of the pipeline, not a convention each caller re-implements.
2. **Determinism**: the same (adapter, checkout) pair MUST produce
   byte-identical output — no clock, no randomness, no environment
   dependence. ``verify()`` proves it: build, render, compare against
   a committed artifact, and report drift as a structured diff summary
   (never a silent pass).
3. **Registration**: adapters register in ``ADAPTERS`` (name -> build
   callable ``repo_root -> dict``). The shipped canonical adapter is
   ``cpp-headers`` (build_registry.build — the ONLY settings-layer
   piece that reads the caelestia checkout). A third-party adapter
   (a fork's generator, a future schema source) registers the same
   way; its output must still satisfy the registry loader's schema,
   which ``verify_output_schema`` checks structurally.

CLI: ``python3 -m assistant.capabilities.settings gen_adapter [--adapter NAME]
[--repo-root PATH] [--output PATH] [--verify]`` — generate or verify;
``--verify`` never writes, it compares and reports.

Safety: generation is a build-time, developer-run act — the runtime
registry loader (registry.py) is untouched, reads the committed
tools.json, and never executes generator code.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import build_registry

__all__ = [
    "ADAPTERS", "CANONICAL_ADAPTER", "canonical_bytes", "generate",
    "verify", "verify_output_schema", "drift_summary", "main",
]

# The registered adapters: name -> build(repo_root: str) -> dict.
# Registering a new adapter is a reviewable diff (this dict is pinned
# by test to exactly the canonical name today).
ADAPTERS: Dict[str, Callable[[str], Dict[str, Any]]] = {
    "cpp-headers": build_registry.build,
}

CANONICAL_ADAPTER = "cpp-headers"


def canonical_bytes(data: Dict[str, Any]) -> bytes:
    """The ONE rendering of a registry build. Every producer and every
    comparison goes through here (byte-identity lives in the pipeline,
    not in per-caller conventions)."""
    return (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8")


def generate(adapter: str, repo_root: str) -> bytes:
    """Build + canonically render one adapter's output."""
    if adapter not in ADAPTERS:
        raise ValueError(f"unknown adapter {adapter!r} "
                         f"(have: {sorted(ADAPTERS)})")
    return canonical_bytes(ADAPTERS[adapter](repo_root))


def verify_output_schema(data: Dict[str, Any]) -> List[str]:
    """Structural sanity of an adapter's output — the registry loader's
    own expectations: meta/tools/not_exposed/presets/explain_rules with
    the fields the loader reads. Third-party adapters must satisfy the
    same shape (this check says WHY, field by field)."""
    problems: List[str] = []
    for key in ("meta", "tools", "not_exposed", "presets",
                "explain_rules"):
        if key not in data:
            problems.append(f"missing top-level key {key!r}")
    tools = data.get("tools")
    if not isinstance(tools, list) or not tools:
        problems.append("tools must be a non-empty list")
        tools = []
    for i, tool in enumerate(tools):
        if not isinstance(tool, dict):
            problems.append(f"tool[{i}] is not an object")
            continue
        for field in ("name", "path", "kind", "group", "citations"):
            if field not in tool:
                problems.append(f"tool[{i}] ({tool.get('name', '?')}) "
                                f"missing {field!r}")
        if tool.get("kind") not in ("bool", "int", "float", "enum",
                                    "string"):
            problems.append(f"tool {tool.get('name', '?')}: unknown kind "
                            f"{tool.get('kind')!r}")
    meta = data.get("meta")
    if isinstance(meta, dict):
        for field in ("generator", "repo_commit", "tool_count"):
            if field not in meta:
                problems.append(f"meta missing {field!r}")
    return problems


def drift_summary(expected: bytes, actual: bytes) -> Dict[str, Any]:
    """A structured, bounded drift report: counts + the first differing
    line numbers (never a wall of text)."""
    exp_lines = expected.decode("utf-8", errors="replace").splitlines()
    act_lines = actual.decode("utf-8", errors="replace").splitlines()
    diffs: List[Dict[str, int]] = []
    for i in range(max(len(exp_lines), len(act_lines))):
        e = exp_lines[i] if i < len(exp_lines) else None
        a = act_lines[i] if i < len(act_lines) else None
        if e != a:
            diffs.append({"line": i + 1})
            if len(diffs) >= 10:
                break
    return {"identical": expected == actual,
            "n_expected_lines": len(exp_lines),
            "n_actual_lines": len(act_lines),
            "first_diffs": diffs}


def verify(adapter: str, repo_root: str,
           committed_path: Path) -> Dict[str, Any]:
    """The byte-identity guard as a function: build, render, compare
    against the committed artifact, report drift (and schema problems)
    structurally. Read-only — never writes."""
    if adapter not in ADAPTERS:
        return {"error": f"unknown adapter {adapter!r} "
                         f"(have: {sorted(ADAPTERS)})"}
    if not committed_path.exists():
        return {"error": f"no committed artifact at {committed_path}"}
    data = ADAPTERS[adapter](repo_root)
    schema_problems = verify_output_schema(data)
    rendered = canonical_bytes(data)
    committed = committed_path.read_bytes()
    report = drift_summary(committed, rendered)
    report.update({"adapter": adapter, "repo_root": str(repo_root),
                   "committed_path": str(committed_path),
                   "schema_problems": schema_problems})
    return report


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        prog="caelestia-assist settings gen_adapter",
        description="the registry-generation adapter interface: generate "
                    "or verify the committed tools.json (byte-identity "
                    "guard; --verify never writes)")
    ap.add_argument("--adapter", default=CANONICAL_ADAPTER,
                    choices=sorted(ADAPTERS),
                    help="the registered generator (default: the shipped "
                         "cpp-headers walker)")
    ap.add_argument("--repo-root", default=None,
                    help="caelestia checkout root (default: auto-locate)")
    ap.add_argument("--output", default=None,
                    help="output path (default: tools.json beside this "
                         "module)")
    ap.add_argument("--verify", action="store_true",
                    help="read-only: build, render, and compare against "
                         "the committed artifact; report drift and exit "
                         "non-zero on any")
    args = ap.parse_args(argv)

    root = Path(args.repo_root) if args.repo_root else \
        build_registry.find_repo_root(here)
    if root is None or not root.exists():
        print("error: cannot locate a caelestia checkout "
              "(need shell/plugin/src/Caelestia/Config under the root)",
              file=sys.stderr)
        return 2

    out = Path(args.output) if args.output else here / "tools.json"

    if args.verify:
        report = verify(args.adapter, str(root), out)
        if "error" in report:
            print(f"error: {report['error']}", file=sys.stderr)
            return 2
        if report["schema_problems"]:
            for problem in report["schema_problems"]:
                print(f"schema: {problem}", file=sys.stderr)
        if report["identical"] and not report["schema_problems"]:
            print(f"OK: {out.name} is byte-identical to the "
                  f"{args.adapter} adapter's output for {root}")
            return 0
        print(f"DRIFT: {report['n_expected_lines']} committed lines vs "
              f"{report['n_actual_lines']} generated; first diff at line "
              f"{report['first_diffs'][0]['line'] if report['first_diffs'] else '?'}",
              file=sys.stderr)
        print("regenerate with: python3 -m assistant.capabilities.settings gen_adapter "
              "--adapter {a} --repo-root {r}".format(a=args.adapter,
                                                     r=root),
              file=sys.stderr)
        return 1

    data = ADAPTERS[args.adapter](str(root))
    problems = verify_output_schema(data)
    if problems:
        print("error: adapter output fails the schema check — refusing "
              "to write:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    blob = canonical_bytes(data)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_bytes(blob)
    import os
    os.replace(tmp, out)
    meta = data.get("meta", {})
    print(f"wrote {out}")
    print(f"  tools: {meta.get('tool_count', '?')} by group: "
          f"{meta.get('group_counts', {})}")
    print(f"  not exposed: {meta.get('not_exposed_count', '?')} "
          f"({meta.get('excluded_by_rule', '?')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
