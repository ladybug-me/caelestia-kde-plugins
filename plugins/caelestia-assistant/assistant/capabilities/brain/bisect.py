"""Config bisect — "what change broke my look?" (exp-build-5 F13).

The user marks one config state good and a later one bad. The suspect set
is the changed leaf keys between the two snapshots. A noisy-answer
Bayesian bisect (per-key Beta posteriors, single-failing-subset
assumption with noise epsilon) chooses which subset to probe next; when
the blame concentrates, classic ddmin (Delta Debugging, Zeller &
Hildebrandt 2002) shrinks to a 1-minimal failing subset under the SAME
probe — ddmin itself makes no probabilistic assumption.

The output is a revert PROPOSAL: planner-shaped ops restoring the good
values (registry tool resolved per key; keys no tool owns are listed for
manual review). Nothing here applies anything — the gated applier does
that only behind the user's explicit consent, and undo stays available.

Probing is INJECTED (a callable subset -> bool). The CLI drives the
stepped engine over several user-paced invocations (brain bisect
next / observe); the assistant never mutates config to manufacture a
probe. All probe answers come from the user's own good/bad marks.

Budgets: probe count is hard-capped; posteriors are O(keys); the module
imports nothing beyond stdlib and the registry/merkle/planner surfaces.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from assistant.capabilities.brain import merkle
from assistant.adapters.caelestia.registry import tool_by_path

__all__ = [
    "leaf_map", "changed_keys", "NoisyBisect", "BisectError",
    "MAX_PROBES", "revert_ops",
]

MAX_PROBES = 64          # hard cap across the whole search
BAYES_TO_DDMIN_PROBES = 8
DDMIN_MAX_PROBES = 48
_LEAF_CAP = 2000         # per-snapshot stored leaf bound (bounded state)


class BisectError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Snapshot flattening: config root -> {key: value} leaf map
# ---------------------------------------------------------------------------

def _flatten(obj: object, prefix: str, out: Dict[str, object], depth: int = 0) -> None:
    if depth > 24:
        raise BisectError(f"leaf depth cap exceeded under {prefix!r}")
    if isinstance(obj, dict):
        for k in sorted(obj):
            _flatten(obj[k], f"{prefix}{k}." if prefix else f"{k}.", out, depth + 1)
        if not obj and prefix:
            out[prefix.rstrip(".")] = {}
    elif isinstance(obj, list):
        out[prefix.rstrip(".")] = obj  # lists are leaves (registry enums etc.)
    else:
        out[prefix.rstrip(".")] = obj


def leaf_map(root: str, cap: int = _LEAF_CAP) -> Dict[str, object]:
    """Flatten every JSON file under the config root into
    ``{"<relfile>::<dotted.leaf>": value}``. Deterministic order; a hard
    cap keeps the stored state bounded (BisectError past it)."""
    out: Dict[str, object] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in sorted(filenames):
            if not fn.endswith(".json"):
                continue
            p = Path(dirpath) / fn
            rel = str(p.relative_to(root)).replace(os.sep, "/")
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (ValueError, OSError) as e:
                raise BisectError(f"cannot parse {rel}: {e}") from e
            leaves: Dict[str, object] = {}
            _flatten(data, "", leaves)
            for k in sorted(leaves):
                key = f"{rel}::{k}"
                if len(out) >= cap:
                    raise BisectError(f"leaf cap {cap} exceeded (at {key})")
                out[key] = leaves[k]
    return out


def changed_keys(good: Dict[str, object], bad: Dict[str, object]) -> List[str]:
    """Keys whose value differs (includes added/removed). Deterministic."""
    return sorted(k for k in set(good) | set(bad)
                  if good.get(k, _MISSING) != bad.get(k, _MISSING))


class _Missing:
    def __repr__(self) -> str:  # pragma: no cover
        return "<missing>"


_MISSING = _Missing()


# ---------------------------------------------------------------------------
# Stepped engine: Bayesian probe selection -> ddmin finish
# ---------------------------------------------------------------------------

def _split(seq: Sequence[str], n: int) -> List[List[str]]:
    """n roughly-equal chunks (ddmin's partitioning)."""
    k = max(1, math.ceil(len(seq) / n))
    return [seq[i:i + k] for i in range(0, len(seq), k)]


def _ddmin_round(candidates: List[str], n: int) -> List[List[str]]:
    """One ddmin round: the n chunks and their complements, deduped,
    empty and full-set probes dropped. Complements are what make the
    final result 1-minimal, not merely minimal."""
    full = frozenset(candidates)
    seen: set = set()
    out: List[List[str]] = []
    for part in _split(candidates, n):
        for s in (part, [k for k in candidates if k not in set(part)]):
            fs = frozenset(s)
            if s and fs != full and fs not in seen:
                seen.add(fs)
                out.append(s)
    return out


class NoisyBisect:
    """Stepped bisect over an injected probe.

    probe(subset) -> True means "BAD" (the break reproduces with exactly
    these keys changed toward the bad values). In the CLI the probe is
    the user answering after applying the printed candidate; in tests it
    is scripted. ``observed`` maps frozenset -> bool for replay.
    """

    def __init__(self, keys: Sequence[str], noise: float = 0.1,
                 prior_strength: float = 1.0) -> None:
        if not keys:
            raise BisectError("no changed keys between good and bad marks")
        if not 0.0 < noise < 0.5:
            raise BisectError(f"noise must be in (0, 0.5), got {noise}")
        self.keys = sorted(keys)
        self.noise = noise
        self.alpha: Dict[str, float] = {k: prior_strength for k in self.keys}
        self.beta: Dict[str, float] = {k: prior_strength for k in self.keys}
        self.observed: Dict[frozenset, bool] = {}
        self.probe_count = 0
        self.phase = "bayes"          # bayes | ddmin | done | inconclusive
        self.ddmin_state: Optional[Dict[str, object]] = None
        self.minimal: Optional[List[str]] = None
        self._pending: List[List[str]] = []
        self.last_subset: Optional[List[str]] = None
        # noisy-answer guard: before presenting a minimal set, ddmin re-asks
        # it once; a contradiction (noise) restarts the blame phase — at
        # most once per set, so a lying probe cannot loop the engine.
        self.confirming: Optional[List[str]] = None
        self.confirm_used: Dict[frozenset, int] = {}

    # -- posteriors --------------------------------------------------------
    def mean(self, key: str) -> float:
        a, b = self.alpha[key], self.beta[key]
        return a / (a + b)

    def ranking(self) -> List[Tuple[str, float]]:
        return sorted(((k, self.mean(k)) for k in self.keys),
                      key=lambda kv: (-kv[1], kv[0]))

    # -- stepping ----------------------------------------------------------
    def next_subset(self) -> List[str]:
        """Choose the next subset to probe (and remember it)."""
        if self.phase == "done":
            raise BisectError("search already finished")
        if self.phase == "inconclusive":
            raise BisectError("search is inconclusive; probe cap reached")
        if self._pending:
            subset = self._pending.pop(0)
        elif self.phase == "bayes":
            subset = self._bayes_next()
        else:
            raise BisectError("ddmin phase without pending subsets")
        self.last_subset = subset
        return subset

    def _bayes_next(self) -> List[str]:
        # information-gain order: probe the top half by posterior mean
        # (classic blame-mass split); if every split variant is already
        # probed, fall back to complements, then singletons in rank
        # order — the probe budget bounds all of this.
        ranked = [k for k, _m in self.ranking()]
        ladder: List[List[str]] = []
        half = max(1, len(ranked) // 2)
        ladder.append(ranked[:half])
        ladder.append(ranked[half:])
        ladder.extend([k] for k in ranked)
        for subset in ladder:
            if subset and frozenset(subset) not in self.observed:
                return subset
        raise BisectError("no unprobed subset left (bayes phase)")

    def observe(self, is_bad: bool) -> None:
        """Record the answer for the last subset and advance the search."""
        if self.last_subset is None:
            raise BisectError("observe() before next_subset()")
        subset = self.last_subset
        key = frozenset(subset)
        self.observed[key] = is_bad
        self.probe_count += 1
        inside = set(subset)
        for k in self.keys:
            if (k in inside) == is_bad:
                self.alpha[k] += 1.0
            else:
                self.beta[k] += 1.0
        self.last_subset = None
        if self.confirming is not None and key == frozenset(self.confirming):
            confirmed = self.confirming
            self.confirming = None
            if is_bad:
                self.minimal = sorted(confirmed)
                self.phase = "done"
            else:
                # noise: the "minimal" set did not reproduce — never hand
                # out a confidently wrong answer; go back to blaming
                self.confirm_used[key] = self.confirm_used.get(key, 0) + 1
                self.phase = "bayes"
                self.ddmin_state = None
                self.minimal = None
        elif self.phase == "ddmin":
            self._ddmin_step(is_bad, subset)
        else:
            self._advance(is_bad, subset)
        if (self.probe_count >= MAX_PROBES
                and self.phase not in ("done", "inconclusive")):
            self.phase = "inconclusive"

    def _advance(self, is_bad: bool, subset: List[str]) -> None:
        if self.phase == "bayes":
            top_key, top_mean = self.ranking()[0]
            cand = [k for k, m in self.ranking() if m > 0.5] or [top_key]
            fs = frozenset(cand)
            if fs in self.observed and self.observed[fs]:
                # the blame-leader set is KNOWN bad: hand it to ddmin
                self._start_ddmin(cand)
            elif (self.probe_count >= BAYES_TO_DDMIN_PROBES
                  or top_mean >= 0.85) and fs not in self.observed:
                self._pending.append(cand)   # must see it BAD to start ddmin
        elif self.phase == "ddmin":
            self._ddmin_step(is_bad, subset)

    # -- ddmin -------------------------------------------------------------
    def _start_ddmin(self, failing: List[str]) -> None:
        """Classic ddmin (Zeller & Hildebrandt 2002), stepped: one round
        tests the n chunks of the current failing set and, when n == 2,
        their complements; a failing test shrinks the set and restarts at
        n = 2; an exhausted round with no failure doubles n. Guaranteed
        1-minimal output; every probe the user answers is one they chose
        to run."""
        if len(failing) <= 1:
            # still confirm: a single blame-leader may be a noise artifact
            self.phase = "ddmin"
            self.ddmin_state = {"candidates": sorted(failing), "n": 2,
                                "index": 0, "round": []}
            self.confirming = sorted(failing)
            self._pending.append(sorted(failing))
            return
        self.phase = "ddmin"
        self.ddmin_state = {
            "candidates": sorted(failing),
            "n": 2,
            "index": 0,
            "round": _ddmin_round(sorted(failing), 2),
        }
        self._pending.append(self.ddmin_state["round"][0])   # type: ignore[index]

    def _ddmin_step(self, is_bad: bool, subset: List[str]) -> None:
        st = self.ddmin_state
        assert st is not None
        candidates: List[str] = st["candidates"]        # type: ignore[assignment]
        if is_bad and len(subset) < len(candidates):
            st["candidates"] = sorted(subset)            # type: ignore[index]
            st["n"] = 2                                  # type: ignore[index]
            st["index"] = 0                              # type: ignore[index]
            st["round"] = _ddmin_round(sorted(subset), 2)  # type: ignore[index]
        else:
            st["index"] = st["index"] + 1                # type: ignore[operator]
        candidates = st["candidates"]                    # type: ignore[assignment]
        index = st["index"]                              # type: ignore[assignment]
        n = st["n"]                                      # type: ignore[assignment]
        round_tests: List[List[str]] = st["round"]       # type: ignore[assignment]
        if index < len(round_tests):
            self._pending.append(round_tests[index])
            return
        # round exhausted with no failing test
        if len(candidates) <= 1 or n >= len(candidates):
            if self.confirm_used.get(frozenset(candidates), 0) < 1:
                # re-ask the whole minimal candidate once (noisy answers)
                self.confirming = list(candidates)
                self._pending.append(list(candidates))
                return
            self.minimal = candidates
            self.phase = "done"
            return
        st["n"] = min(n * 2, len(candidates))            # type: ignore[index]
        st["index"] = 0                                  # type: ignore[index]
        st["round"] = _ddmin_round(candidates, st["n"])  # type: ignore[index]
        self._pending.append(st["round"][0])             # type: ignore[index]

    # -- outcomes ----------------------------------------------------------
    def proposal(self) -> Dict[str, object]:
        if self.phase != "done" or not self.minimal:
            raise BisectError(f"no proposal yet (phase={self.phase})")
        return {"minimal": list(self.minimal),
                "probes": self.probe_count,
                "confidence": {k: round(self.mean(k), 3) for k in self.minimal}}

    # -- persistence (brain state; bounded JSON) ---------------------------
    def to_state(self) -> Dict[str, object]:
        return {
            "keys": list(self.keys),
            "noise": self.noise,
            "alpha": dict(self.alpha),
            "beta": dict(self.beta),
            "observed": [[sorted(fs), v] for fs, v in sorted(
                self.observed.items(), key=lambda kv: sorted(kv[0]))],
            "probe_count": self.probe_count,
            "phase": self.phase,
            "ddmin_state": self.ddmin_state,
            "minimal": self.minimal,
            "pending": [list(s) for s in self._pending],
            "last_subset": self.last_subset,
            "confirming": self.confirming,
            "confirm_used": [[sorted(fs), n] for fs, n in
                             sorted(self.confirm_used.items(),
                                    key=lambda kv: sorted(kv[0]))],
        }

    @classmethod
    def from_state(cls, st: Dict[str, object]) -> "NoisyBisect":
        eng = cls(st["keys"], noise=st["noise"])          # type: ignore[arg-type]
        eng.alpha = dict(st["alpha"])                     # type: ignore[arg-type]
        eng.beta = dict(st["beta"])                       # type: ignore[arg-type]
        eng.observed = {frozenset(k): v for k, v in st["observed"]}  # type: ignore
        eng.probe_count = st["probe_count"]               # type: ignore[arg-type]
        eng.phase = st["phase"]                           # type: ignore[arg-type]
        eng.ddmin_state = st["ddmin_state"]               # type: ignore[arg-type]
        eng.minimal = st["minimal"]                       # type: ignore[arg-type]
        eng._pending = [list(s) for s in st["pending"]]   # type: ignore[arg-type]
        eng.last_subset = st["last_subset"]               # type: ignore[arg-type]
        eng.confirming = st["confirming"]                 # type: ignore[arg-type]
        eng.confirm_used = {frozenset(k): v for k, v in st["confirm_used"]}  # type: ignore
        return eng


# ---------------------------------------------------------------------------
# Revert proposal: good values -> planner-shaped ops
# ---------------------------------------------------------------------------

def revert_ops(good_leaves: Dict[str, object],
               keys: Iterable[str],
               settings_file: str = "shell.json") -> Dict[str, object]:
    """Planner-shaped revert ops for the failing keys, resolving each
    settings leaf to its registry tool. Keys nobody owns — or keys that
    live outside the managed shell.json — are listed for manual review;
    nothing is invented into ops. Dry-run only."""
    ops: List[Dict[str, object]] = []
    manual: List[Dict[str, str]] = []
    for key in sorted(keys):
        rel_file, _, leaf = key.partition("::")
        if good_leaves is not None and key not in good_leaves:
            manual.append({"key": key, "reason": "absent in the good snapshot"})
            continue
        value = good_leaves.get(key) if good_leaves is not None else None
        if rel_file != settings_file:
            manual.append({"key": key,
                           "reason": f"outside the managed {settings_file}"})
            continue
        spec = tool_by_path(leaf)
        if spec is None:
            manual.append({"key": key, "reason": "no registry tool owns this leaf"})
            continue
        ops.append({"tool": spec.name, "action": "set", "value": value,
                    "raw": f"{spec.name}={json.dumps(value)}",
                    "note": f"revert to good value (was changed in {rel_file})"})
    return {"ops": ops, "manual": manual,
            "note": ("dry-run proposal only — applying goes through the "
                     "settings consent gate with a backup and bounded undo")}
