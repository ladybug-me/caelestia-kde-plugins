"""shellkb.pubgrub — PubGrub-style dependency conflict explanations (B10).

When the shell cannot run, the question is never "which package is
broken" but "WHY can these not coexist". PubGrub (Natalie Weizenbaum's
version solver, pub.dev) became famous for one thing: its explanations
read like reasoning — "because X requires Y which conflicts with Z" —
instead of a dump. This module produces exactly that shape, over a
small dependency universe, with a BOUNDED DPLL-style search (unit
propagation by domain-narrowing, chronological backtracking, a node
budget) doing the actual work.

Where the universe comes from (honestly):
  - the CALLER may pass one (the CLI's --universe FILE, the bridge's
    'universe' object) — the analyst's path;
  - the committed fixture (``dep_universe.json``) is an ILLUSTRATIVE
    universe built from upstream's own installer lists (the session
    packages its scripts install are cited inside); its version bounds
    are plausible but explicitly not authoritative;
  - an INSTALLED overlay may come from agent.pkgprobe (pacman -Q /
    dpkg-query, the existing read-only probe) — only when the
    package_audit capability is on, and only as name->version facts.

The solver never suggests running an installer. Output is an
explanation and, when satisfiable, the version assignment it proved —
SUGGESTED_NOT_EXECUTED like everything else the shellkb says.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["UNIVERSE_PATH", "parse_spec", "spec_ok", "solve",
           "check_installed", "render_solution", "render_conflict"]

UNIVERSE_PATH = Path(__file__).resolve().parent / "dep_universe.json"

_MAX_NODES = 4000  # the bound in "bounded SAT": more than this, ABSTAIN


# ---------------------------------------------------------------------------
# version constraints
# ---------------------------------------------------------------------------

_SPEC_RE = re.compile(r"^\s*(>=|<=|==|!=|>|<|=|\^|~)\s*(\d[0-9A-Za-z.\-+]*)\s*$")


def _vtuple(version: str) -> Tuple[int, ...]:
    """'1.2.10' -> (1, 2, 10) — numeric segments only, longer wins."""
    parts = re.findall(r"\d+", version)
    return tuple(int(p) for p in parts) if parts else (0,)


def _vkey(version: str) -> Tuple[int, Tuple[int, ...], str]:
    """Sort key: numeric-major (descending use), then raw string."""
    return (0, _vtuple(version), version)


def parse_spec(text: str) -> List[Tuple[str, Tuple[int, ...]]]:
    """'>=1,<3' -> [('>=', (1,)), ('<', (3,))] — comma is AND.
    '^1.2' means >=1.2,<2; '~1.2' means >=1.2,<1.3 (pub/cargo style)."""
    specs: List[Tuple[str, Tuple[int, ...]]] = []
    for raw in str(text).split(","):
        raw = raw.strip()
        if not raw or raw == "*":
            continue
        if re.fullmatch(r"\d[0-9A-Za-z.\-+]*", raw):
            raw = "== " + raw  # a bare version pins it exactly
        m = _SPEC_RE.match(raw)
        if not m:
            raise ValueError(f"unparseable version spec: {raw!r}")
        op, ver = m.group(1), m.group(2)
        vt = _vtuple(ver)
        if op == "^":
            specs.append((">=", vt))
            specs.append(("<", (vt[0] + 1,)))
        elif op == "~":
            specs.append((">=", vt))
            hi = list(vt) + [0]
            if len(hi) >= 2:
                hi[1] += 1
            specs.append(("<", tuple(hi[:2])))
        else:
            if op == "=":
                op = "=="
            specs.append((op, vt))
    return specs


def spec_ok(version: str, specs: List[Tuple[str, Tuple[int, ...]]]) -> bool:
    vt = _vtuple(version)
    for op, want in specs:
        if op == ">=" and not vt >= want:
            return False
        if op == "<=" and not vt <= want:
            return False
        if op == ">" and not vt > want:
            return False
        if op == "<" and not vt < want:
            return False
        if op == "==" and vt != want:
            return False
        if op == "!=" and vt == want:
            return False
    return True


class _Universe:
    def __init__(self, data: Dict[str, Any]) -> None:
        self.packages: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for name, versions in (data.get("packages") or {}).items():
            self.packages[name] = {}
            for ver, meta in versions.items():
                depends = meta.get("depends") or {}
                conflicts = meta.get("conflicts") or {}
                self.packages[name][ver] = {
                    "depends": dict(depends),
                    "conflicts": dict(conflicts),
                }

    def versions_desc(self, name: str) -> List[str]:
        vers = list(self.packages.get(name, {}))
        vers.sort(key=_vkey, reverse=True)
        return vers

    def meta(self, name: str, version: str) -> Dict[str, Any]:
        return self.packages.get(name, {}).get(version, {})


# ---------------------------------------------------------------------------
# the bounded search with a because-trail
# ---------------------------------------------------------------------------

class _Trail:
    """The because-trail: for every constraint narrowed onto a package,
    WHO required it (with which chosen version) — the raw material of
    the derivation chain. Dead ends are recorded the moment the search
    hits them (with the exact constraint set that did it), because the
    branch-local copies die on backtrack but the EVIDENCE must not."""

    def __init__(self) -> None:
        self.edges: Dict[str, List[Tuple[str, str, str]]] = {}
        self.conflicts: List[Tuple[str, str, str, str]] = []
        self.deadends: List[Tuple[str, List[Tuple[str, str, str]]]] = []

    def require(self, target: str, requirer: str, requirer_ver: str,
                spec: str) -> None:
        self.edges.setdefault(target, []).append(
            (requirer, requirer_ver, spec))

    def conflict(self, holder: str, holder_ver: str, target: str,
                 spec: str) -> None:
        self.conflicts.append((holder, holder_ver, target, spec))

    def record_deadend(self, target: str,
                       cons: List[Tuple[str, str, str]]) -> None:
        self.deadends.append((target, list(cons)))


def _domain(u: _Universe, name: str,
            specs: List[Tuple[str, str, str]]) -> List[str]:
    """Versions of `name` satisfying every (requirer, version, spec)."""
    viable = u.versions_desc(name)
    for _req, _rver, spec in specs:
        parsed = parse_spec(spec)
        viable = [v for v in viable if spec_ok(v, parsed)]
    return viable


def _walk_chain(trail: _Trail, target: str, seen: Optional[set] = None) \
        -> List[Tuple[str, str, str]]:
    """One deterministic because-chain from the root to `target`,
    following the FIRST recorded requirer edge at each hop (the search
    tried highest-versions-first, so the first edge is the path it
    actually walked). Bounded by `seen` against cycles."""
    seen = seen if seen is not None else set()
    chain: List[Tuple[str, str, str]] = []
    node = target
    while node not in seen:
        seen.add(node)
        edges = trail.edges.get(node)
        if not edges:
            break
        requirer, requirer_ver, spec = edges[0]
        chain.append((requirer, requirer_ver, f"requires {node} {spec}"))
        node = requirer
    chain.reverse()
    return chain


def solve(universe: Dict[str, Any], root_name: str, root_spec: str = "*",
          max_nodes: int = _MAX_NODES) -> Dict[str, Any]:
    """Find a version assignment for the root's dependency closure, or
    produce the because-chain that proves there is none."""
    u = _Universe(universe)
    if root_name not in u.packages:
        return {"ok": False, "verdict": "UNKNOWN_PACKAGE",
                "because": [f"'{root_name}' is not in the dependency "
                            f"universe (known: "
                            f"{', '.join(sorted(u.packages))})"]}
    trail = _Trail()
    nodes = [0]

    def candidates(name: str,
                   cons: Dict[str, List[Tuple[str, str, str]]]) \
            -> List[str]:
        return _domain(u, name, cons.get(name, []))

    def propagate(assigned: Dict[str, str],
                  cons: Dict[str, List[Tuple[str, str, str]]],
                  expanded: set) -> Tuple[bool, Optional[str]]:
        """Expand the chosen versions' own requirements ONCE per
        (package, version); False + a package name when some domain
        emptied. Deterministic queue order. Mutates cons/expanded —
        callers pass private copies."""
        queue = [(n, assigned[n]) for n in sorted(assigned)
                 if (n, assigned[n]) not in expanded]
        while queue:
            name, ver = queue.pop(0)
            expanded.add((name, ver))
            for dep, spec in u.meta(name, ver).get("depends", {}).items():
                edge = (name, ver, spec)
                if edge not in cons.get(dep, []):
                    trail.require(dep, name, ver, spec)
                    cons.setdefault(dep, []).append(edge)
                    if not u.packages.get(dep):
                        trail.record_deadend(dep, cons.get(dep, []))
                        return False, dep
                    if not candidates(dep, cons):
                        trail.record_deadend(dep, cons.get(dep, []))
                        return False, dep
            for bad, spec in u.meta(name, ver).get(
                    "conflicts", {}).items():
                trail.conflict(name, ver, bad, spec)
        # pairwise conflict check over everything currently assigned —
        # a conflict recorded earlier must kill a LATER assignment too
        for holder, hver, bad, spec in trail.conflicts:
            if holder in assigned and bad in assigned and \
                    spec_ok(assigned[bad], parse_spec(spec)):
                return False, bad
        return True, None

    def backtrack(assigned: Dict[str, str],
                  cons: Dict[str, List[Tuple[str, str, str]]],
                  expanded: set) -> Optional[Dict[str, str]]:
        nodes[0] += 1
        if nodes[0] > max_nodes:
            return None
        ok, dead = propagate(assigned, cons, expanded)
        if not ok:
            return None
        unassigned = [n for n in sorted(cons) if n not in assigned]
        if not unassigned:
            return dict(assigned)
        # MRV with deterministic name tie-break
        pick = min(unassigned, key=lambda n: (len(candidates(n, cons)), n))
        if not candidates(pick, cons):
            trail.record_deadend(pick, cons.get(pick, []))
            return None
        for ver in candidates(pick, cons):
            assigned[pick] = ver
            found = backtrack(
                assigned,
                {k: list(v) for k, v in cons.items()},
                set(expanded))
            if found is not None:
                return found
            del assigned[pick]
        return None

    # the root demand is the first constraint in the trail
    cons0: Dict[str, List[Tuple[str, str, str]]] = {
        root_name: [("the request", "", root_spec)]}
    found = backtrack({}, cons0, set())
    if found is None and nodes[0] > max_nodes:
        return {"ok": False, "verdict": "ABSTAIN",
                "because": [f"the search exceeded {max_nodes} nodes: "
                            f"no answer, honestly, rather than a slow "
                            f"or wrong one"]}
    if found is not None:
        return {"ok": True, "verdict": "SATISFIABLE",
                "choices": dict(sorted(found.items())),
                "n_search_nodes": nodes[0],
                "note": "a proven assignment — installing it is your "
                        "move, never ours (SUGGESTED_NOT_EXECUTED)"}

    # UNSATISFIABLE: the derivation walks to the FIRST recorded dead
    # end — the search's own evidence, not a post-hoc guess
    if trail.deadends:
        dead_pkg, dead_cons = trail.deadends[0]
    else:
        dead_pkg, dead_cons = None, []
    chain = _walk_chain(trail, dead_pkg) if dead_pkg else []
    reasons = []
    for requirer, requirer_ver, line in chain:
        reasons.append(f"because {requirer}"
                       + (f" {requirer_ver}" if requirer_ver else "")
                       + f" {line}")
    if dead_pkg:
        demanded = "; ".join(
            f"{req}{' ' + rver if rver else ''} needs {spec}"
            for req, rver, spec in dead_cons[-3:]) or "no version fits"
        reasons.append(f"but {dead_pkg} has no version satisfying the "
                       f"demand ({demanded}); available: "
                       f"{', '.join(u.versions_desc(dead_pkg)) or 'none'}")
    for holder, holder_ver, target, spec in trail.conflicts[-3:]:
        reasons.append(f"because {holder} {holder_ver} conflicts with "
                       f"{target} {spec}")
    return {"ok": False, "verdict": "UNSATISFIABLE",
            "because": reasons,
            "n_search_nodes": nodes[0],
            "note": "a derivation, not a dump: every line is a fact the "
                    "search used"}


def check_installed(universe: Dict[str, Any], installed: Dict[str, str],
                    root_name: str, root_spec: str = "*") -> Dict[str, Any]:
    """No search at all: does the INSTALLED set satisfy the closure?
    Reports missing packages and violated constraints with the same
    because-voice."""
    u = _Universe(universe)
    if root_name not in u.packages:
        return {"ok": False, "verdict": "UNKNOWN_PACKAGE",
                "because": [f"'{root_name}' is not in the universe"]}
    problems: List[str] = []
    seen_edges: set = set()
    queue = [(root_name, "the request", root_spec)]
    while queue:
        name, requirer, spec = queue.pop(0)
        edge = (name, requirer, spec)
        if edge in seen_edges:
            continue  # dependency cycles (kwin <-> plasma-workspace)
        seen_edges.add(edge)
        ver = installed.get(name)
        if ver is None:
            problems.append(f"{requirer} requires {name} {spec}, but "
                            f"{name} is not installed (per the read-only "
                            f"probe)")
            continue
        if not spec_ok(ver, parse_spec(spec)):
            problems.append(f"{requirer} requires {name} {spec}, but "
                            f"{ver} is installed")
        for dep, dep_spec in u.meta(name, ver).get("depends", {}).items():
            queue.append((dep, f"{name} {ver}", dep_spec))
    return {
        "ok": not problems,
        "verdict": "INSTALLED_SATISFIES" if not problems else
        "INSTALLED_CONFLICTS",
        "problems": problems,
        "checked": sorted({e[0] for e in seen_edges}),
    }


def render_solution(data: Dict[str, Any]) -> List[str]:
    lines = [f"dependency check — {data['verdict']}"]
    if data.get("choices"):
        for name, ver in data["choices"].items():
            lines.append(f"  SUGGESTED_NOT_EXECUTED: {name} {ver}")
        lines.append(f"  ({data.get('note')})")
        if "n_search_nodes" in data:
            lines.append(f"  search: {data['n_search_nodes']} nodes "
                         f"(bounded)")
    for line in data.get("because", []):
        lines.append(f"  {line}")
    return lines


def render_conflict(data: Dict[str, Any]) -> List[str]:
    return render_solution(data)


def main(argv=None) -> int:  # pragma: no cover - thin CLI
    import argparse
    import sys
    ap = argparse.ArgumentParser(
        prog="caelestia-assist shellkb conflicts",
        description="PubGrub-style dependency explanations over a "
                    "bounded search (read-only; never installs)")
    ap.add_argument("--universe", default=None,
                    help="universe JSON (default: the committed fixture)")
    ap.add_argument("--root", required=True)
    ap.add_argument("--spec", default="*")
    ap.add_argument("--installed", default=None,
                    help="JSON {name: version} from a read-only probe")
    args = ap.parse_args(list(sys.argv[1:] if argv is None else argv))
    universe = (json.loads(Path(args.universe).read_text())
                if args.universe
                else json.loads(UNIVERSE_PATH.read_text()))
    if args.installed:
        installed = json.loads(Path(args.installed).read_text())
        data = check_installed(universe, installed, args.root, args.spec)
    else:
        data = solve(universe, args.root, args.spec)
    for line in render_solution(data):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
