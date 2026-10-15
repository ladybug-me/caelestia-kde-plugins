#!/usr/bin/env python3
"""Phase Z4 — dead-code scan over the Group B/C/D/E modules.

AST-based, same method as the Stage B sweep: for each target module,
every top-level def/class and every import name must be referenced
somewhere in the package tree (tests count as references; the module
__all__ and dunder names are exports, not dead code). Duplicated
function BODIES are also checked across the new modules (exact
3+-line duplicates only — layer isolation deliberately repeats small
shapes, and those stay).

Run: PYTHONPATH=. python3 scripts/dev/z_deadcode.py
"""
from __future__ import annotations

import ast
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "assistant"

# everything the groups added, plus their tests
TARGETS = [
    "brain/qa.py", "brain/nlg.py", "brain/sketch.py", "brain/seasonal.py",
    "brain/bursts.py", "brain/gp_prefs.py", "brain/rete.py",
    "brain/tests/test_qa.py", "brain/tests/test_nlg.py",
    "brain/tests/test_sketch.py", "brain/tests/test_seasonal.py",
    "brain/tests/test_bursts.py", "brain/tests/test_gp_prefs.py",
    "brain/tests/test_rete.py",
    "generative/reranker.py", "generative/tests/test_reranker.py",
    "shellkb/cligrammar.py", "shellkb/cmdparse.py", "shellkb/howto.py",
    "shellkb/jsonmerge.py", "shellkb/pubgrub.py",
    "shellkb/tests/test_cmdparse.py", "shellkb/tests/test_cligrammar.py",
    "shellkb/tests/test_howto.py", "shellkb/tests/test_jsonmerge.py",
    "shellkb/tests/test_pubgrub.py",
    "eval/metamorphic.py", "eval/grow.py",
    "cortex/abers.py", "cortex/label_fusion.py", "cortex/confusables.py",
]


def _all_source() -> dict:
    src = {}
    for path in PKG.rglob("*.py"):
        rel = str(path.relative_to(PKG))
        try:
            src[rel] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
    return src


def _defs_and_imports(tree):
    defs, imports = [], []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            defs.append(node.name)
        elif isinstance(node, ast.Import):
            imports += [(a.asname or a.name.split(".")[0])
                        for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            imports += [(a.asname or a.name) for a in node.names]
    return defs, imports


def main() -> int:
    src = _all_source()
    fails = 0
    for target in TARGETS:
        if target not in src:
            print(f"MISS  {target} (not present — skipped)")
            continue
        tree = ast.parse(src[target])
        defs, imports = _defs_and_imports(tree)
        # the reference universe: EVERY module, this one included
        # (tests calling a helper count as real references)
        universe = "\n".join(src.values())
        exported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if getattr(t, "id", "") == "__all__":
                        try:
                            exported |= {e.value for e in node.value.elts}
                        except AttributeError:
                            pass
        for name in defs:
            if name.startswith("__") or name in exported:
                continue
            if name.startswith("Test"):
                # unittest discovery references TestCase subclasses
                # without naming them — never "dead" by name counting
                continue
            uses = universe.count(name)
            if uses <= 1:
                print(f"DEAD def {target}::{name} ({uses} refs)")
                fails += 1
        for name in imports:
            # count as a bare token; a name used only at import time is dead
            uses = universe.count(name)
            if uses <= 1:
                print(f"DEAD import {target}::{name} ({uses} refs)")
                fails += 1
    # exact-duplicate bodies among the new modules
    bodies: Counter = Counter()
    for target in TARGETS:
        if target not in src or "tests/" in target:
            continue
        tree = ast.parse(src[target])
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body = ast.dump(ast.Module(body=node.body, type_ignores=[]))
                if body.count("\n") >= 3 or len(body) > 300:
                    key = (target, node.name, body)
                    bodies[body] += 1
                    bodies._dupes = getattr(bodies, "_dupes", []) + \
                        [(target, node.name, body)]
    dupes = [d for d in getattr(bodies, "_dupes", [])
             if bodies[d[2]] > 1 and "tests/" not in d[0]]
    for target, name, _ in sorted(set((t, n) for t, n, _ in dupes)):
        print(f"DUP  {target}::{name} (exact body duplicated)")
        fails += 1
    print("---")
    print("dead-code scan:", "PASS" if fails == 0 else f"FAIL ({fails})")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
