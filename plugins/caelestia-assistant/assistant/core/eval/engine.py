"""The eval arena — measurement suites over the frozen system state.

Six suites, each honest about what it measures:

- ``routing``     request -> expected surface. Top-1 / top-3 / precision
                  among ROUTED / confident-wrong, with seeded bootstrap
                  intervals. Cold start: ``DEFAULT_STATE`` router weights,
                  no learned state, no session — the deterministic floor.
- ``nlplan``      request -> exact planned tool + value + validity, through
                  the real cortex pipeline (router + planner) against a
                  temp target file. Exact-match is the headline.
- ``abstention``  out-of-domain negatives must NOT be confidently routed
                  to a settings tool or preset (a coarse surface like
                  ``search``/``genius``/``explain`` is honest delegation,
                  not a settings mistake).
- ``diagnosis``   log line -> expected rule id, through the diagnostics
                  engine with the shipped rule pack.
- ``calibration`` ECE + Brier over every confident settings route the
                  routing/abstention suites produced (top-1 softmax
                  probability vs correctness). Cold-start numbers: the
                  router's confidence is NOT a calibrated posterior until
                  the learner has history; suite output says so.
- ``footprint``   in-process timing/RSS: first-route index build, steady
                  route latency p50/p95, process peak RSS. Cold-process
                  CLI latency is measured by ``scripts/measure_cli.py``
                  at the repo root — spawning processes from inside
                  ``assistant/`` is forbidden by the behavioral safety tests, and
                  the arena does not get an exception.

Determinism: routing/abstention/calibration are pure functions of the
committed set files (DEFAULT_STATE, fixed bootstrap seeds). nlplan writes
only to a throwaway temp target. diagnosis reads only the shipped rules.
Latency numbers are machine-dependent by nature and are reported as
measurements, not gates; the ratchet (tests/test_eval_ratchet.py) pins the
accuracy suites only.
"""
from __future__ import annotations

import json
import math
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

from .stats import brier, ece, proportion_ci

SET_DIR = Path(__file__).parent / "sets"

SUITES = ("routing", "nlplan", "abstention", "diagnosis", "calibration",
          "footprint", "metamorphic")
SPLITS = ("dev", "sealed")

BOOT_SEED = 120


# ---------------------------------------------------------------------------
# Set loading.
# ---------------------------------------------------------------------------


def load_set(suite: str, split: str) -> Dict[str, Any]:
    # Stage B finding: the sealed files were committed as
    # `sealed_{suite}.json` while this loader demanded
    # `{suite}_sealed.json` — the sealed CLI could never run as shipped
    # (birth defect, F1). The loader now accepts the committed names;
    # set CONTENT is untouched and the sealed manifest still pins it.
    candidates = [SET_DIR / f"{suite}_{split}.json"]
    if split == "sealed":
        candidates.append(SET_DIR / f"sealed_{suite}.json")
    path = next((c for c in candidates if c.exists()), candidates[0])
    if not path.exists():
        raise FileNotFoundError(f"no such set: {candidates[0].name}")
    data = json.loads(path.read_text())
    for item in data.get("items", []):
        if "id" not in item or "text" not in item:
            raise ValueError(f"malformed item in {path.name}: {item!r}")
    return data


def available(split: str) -> List[str]:
    out = []
    for s in SUITES:
        if s == "metamorphic":
            # derived over the DEV routing set only — it has no sealed
            # split by construction (variant generators are pinned by the
            # ratchet test, not by a sealed file)
            if split == "dev":
                out.append(s)
        elif (SET_DIR / f"{s}_{split}.json").exists():
            out.append(s)
        elif split == "sealed" and (SET_DIR / f"sealed_{s}.json").exists():
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# Timing (datetime deltas: stdlib-only discipline keeps `time` off the eval path; wall
# clock has microsecond resolution, which is enough for ms-scale medians).
# ---------------------------------------------------------------------------


def _now_ms() -> float:
    return datetime.now().timestamp() * 1000.0


def _median(xs: Sequence[float]) -> float:
    ys = sorted(xs)
    n = len(ys)
    if not n:
        return float("nan")
    mid = n // 2
    return ys[mid] if n % 2 else (ys[mid - 1] + ys[mid]) / 2.0


def _percentile(xs: Sequence[float], q: float) -> float:
    ys = sorted(xs)
    if not ys:
        return float("nan")
    idx = min(len(ys) - 1, max(0, int(math.floor(q * len(ys)))))
    return ys[idx]


def _peak_rss_kb() -> int:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    except OSError:
        pass
    return -1


# ---------------------------------------------------------------------------
# Suite: routing (cold start — DEFAULT_STATE, pure router ranking).
# ---------------------------------------------------------------------------


def run_routing(split: str = "dev",
                state: "Any | None" = None) -> Dict[str, Any]:
    """``state`` (F28): inject a RouterState to measure ALTERNATIVE
    weights (the refit ratchet measures candidate vs current). None
    means DEFAULT_STATE — the cold-start measurement, unchanged."""
    from assistant.core.router import DEFAULT_STATE, Router
    from assistant.core.label_fusion import refine, vote_router_signals, \
        load_model as load_fusion_model

    data = load_set("routing", split)
    router = Router()
    fusion_model = load_fusion_model()
    eff_state = state if state is not None else DEFAULT_STATE
    results: List[Dict[str, Any]] = []
    for item in data["items"]:
        res = router.route(item["text"],
                           state=state if state is not None else DEFAULT_STATE,
                           k=5)
        cands = [(c.surface, c.kind) for c in res.candidates[:3]]
        expect_verdict = item.get("expect_verdict")
        if expect_verdict:
            # Pre-declared rule (2026-09-28 registry resync): an item whose
            # accepted tool was REMOVED upstream expects the honest
            # out-of-ontology verdict, never a confident route to a
            # coincidental tool. A confident ROUTE here counts as a
            # confident-wrong, which is exactly the signal we want.
            # A5 generalization: expect_verdict may also be AMBIGUOUS —
            # items mined from genuinely confusable classes assert the
            # ASK itself (invariant 2: verdicts stay honest, never a
            # guess) instead of picking a side the reviewer cannot
            # defend. verdict items count as top1-ok/top3-ok when the
            # honest verdict is delivered.
            top1_ok = res.verdict == expect_verdict
            top3_ok = top1_ok
            accept: list = []
        else:
            accept = item["accept"]
            top1_ok = bool(cands and cands[0][0] in accept)
            top3_ok = any(s in accept for s, _k in cands)
        # A2 fused verdict (the pipeline's SHIPPED demote-only behavior):
        # a contested confident route becomes an honest ask and leaves
        # the fused confident-wrong count.
        fused_verdict = res.verdict
        if res.verdict == "ROUTED":
            ballots = vote_router_signals(router, item["text"], eff_state)
            fused_verdict, _note, _post = refine(
                res, ballots["votes"], fusion_model, eff_state)
        results.append({
            "id": item["id"], "text": item["text"],
            "verdict": res.verdict, "top3": cands,
            "top1_ok": top1_ok,
            "top3_ok": top3_ok,
            "accept": accept,
            "fused_verdict": fused_verdict,
        })
    report = _routing_report(split, results)
    fused_routed = [r for r in results if r["fused_verdict"] == "ROUTED"]
    fused_wrong = [r for r in fused_routed if not r["top1_ok"]]
    report["confident_wrong_fused"] = {
        "count": len(fused_wrong), "of_routed": len(fused_routed),
        "rate": (round(len(fused_wrong) / len(fused_routed), 4)
                 if fused_routed else None),
        "note": ("after Dawid-Skene demote-only fusion — the pipeline's "
                 "shipped behavior; contested confident routes become "
                 "honest asks and leave this count"),
    }
    return report


def _routing_report(split: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    top1 = [int(r["top1_ok"]) for r in results]
    top3 = [int(r["top3_ok"]) for r in results]
    routed = [r for r in results if r["verdict"] == "ROUTED"]
    routed_ok = [int(r["top1_ok"]) for r in routed] if routed else []
    wrong = [r for r in routed if not r["top1_ok"]]
    return {
        "suite": "routing", "split": split, "n": n, "routed": len(routed),
        "metrics": {
            "top1_rate": _ci(top1),
            "top3_rate": _ci(top3),
            "precision_among_routed": _ci(routed_ok) if routed_ok else ("n/a", "n/a", "n/a"),
        },
        "confident_wrong": {
            "count": len(wrong), "of_routed": len(routed),
            "rate": (round(len(wrong) / len(routed), 4) if routed else None),
            "items": [
                {"id": r["id"], "text": r["text"], "top1": r["top3"][0][0] if r["top3"] else None,
                 "accept": r["accept"]} for r in wrong],
        },
        "failures": [
            {"id": r["id"], "text": r["text"], "top1": r["top3"][0][0] if r["top3"] else None,
             "accept": r["accept"]} for r in results if not r["top1_ok"]],
    }


def _ci(outcomes: Sequence[int]) -> Tuple[float, float, float]:
    point, lo, hi = proportion_ci(list(outcomes), seed=BOOT_SEED)
    return (round(point, 4), round(lo, 4), round(hi, 4))


# ---------------------------------------------------------------------------
# Suite: nlplan (request -> exact planned tool + value, real pipeline).
# ---------------------------------------------------------------------------


def run_nlplan(split: str = "dev") -> Dict[str, Any]:
    from assistant.core.pipeline import process
    from assistant.core.router import DEFAULT_STATE

    data = load_set("nlplan", split)
    results: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="eval-nlplan-") as tmp:
        target = Path(tmp) / "shell.json"
        for item in data["items"]:
            res = process(item["text"], file_path=target, router_state=DEFAULT_STATE)
            ops = res.ops or []
            tool = ops[0]["tool"] if ops else None
            value = ops[0].get("value") if ops else None
            tool_ok = tool == item["expect_tool"]
            value_ok = _values_equal(value, item.get("expect_value"))
            valid_ok = (res.plan is None) or (not res.plan.get("errors"))
            exact = tool_ok and value_ok and valid_ok and len(ops) >= 1
            results.append({
                "id": item["id"], "text": item["text"],
                "got_tool": tool, "got_value": value,
                "expect_tool": item["expect_tool"], "expect_value": item.get("expect_value"),
                "verdict": res.verdict, "n_ops": len(ops),
                "tool_ok": bool(tool_ok), "value_ok": bool(value_ok),
                "valid_ok": bool(valid_ok), "exact": bool(exact),
            })
    exact_v = [int(r["exact"]) for r in results]
    tool_v = [int(r["tool_ok"]) for r in results]
    return {
        "suite": "nlplan", "split": split, "n": len(results),
        "metrics": {
            "exact_rate": _ci(exact_v),
            "tool_rate": _ci(tool_v),
        },
        "failures": [r for r in results if not r["exact"]],
    }


def _values_equal(got: Any, expect: Any) -> bool:
    if expect is None:
        return True  # item does not pin a value
    if isinstance(got, bool) or isinstance(expect, bool):
        return got is expect
    if isinstance(got, (int, float)) and isinstance(expect, (int, float)):
        return math.isclose(float(got), float(expect), rel_tol=1e-6, abs_tol=1e-9)
    if isinstance(got, str) and isinstance(expect, str):
        return got.strip().lower() == expect.strip().lower()
    return got == expect


# ---------------------------------------------------------------------------
# Suite: abstention (negatives must not confidently route to settings).
# ---------------------------------------------------------------------------


def run_abstention(split: str = "dev") -> Dict[str, Any]:
    from assistant.core.router import DEFAULT_STATE, Router

    data = load_set("abstention", split)
    router = Router()
    results: List[Dict[str, Any]] = []
    for item in data["items"]:
        res = router.route(item["text"], state=DEFAULT_STATE, k=3)
        top = res.candidates[0] if res.candidates else None
        # A settings mistake = confident (ROUTED) route whose top-1 is a
        # settings TOOL or PRESET. A coarse surface (search/genius/explain/
        # diagnose/brain/...) is honest delegation, not a settings error.
        bad = (
            res.verdict == "ROUTED" and top is not None
            and top.kind in ("tool", "preset")
        )
        results.append({
            "id": item["id"], "text": item["text"],
            "verdict": res.verdict,
            "top1": (top.surface, top.kind) if top else None,
            "correctly_not_confident": not bad,
        })
    v = [int(r["correctly_not_confident"]) for r in results]
    return {
        "suite": "abstention", "split": split, "n": len(results),
        "metrics": {"abstain_rate": _ci(v)},
        "failures": [r for r in results if not r["correctly_not_confident"]],
    }


# ---------------------------------------------------------------------------
# Suite: diagnosis (log line -> rule id).
# ---------------------------------------------------------------------------


def run_diagnosis(split: str = "dev") -> Dict[str, Any]:
    from assistant.capabilities.diagnostics.engine import diagnose, load_rules

    data = load_set("diagnosis", split)
    rules = load_rules()
    results: List[Dict[str, Any]] = []
    for item in data["items"]:
        d = diagnose(item["text"], rules=rules, top=3)
        top = d.get("top") or {}
        got = (top.get("rule") or {}).get("id")
        ok = got == item["expect_rule"]
        results.append({
            "id": item["id"], "text": item["text"],
            "got": got, "expect": item["expect_rule"], "ok": bool(ok),
            "verdict": d.get("verdict"),
        })
    v = [int(r["ok"]) for r in results]
    return {
        "suite": "diagnosis", "split": split, "n": len(results),
        "metrics": {"rule_hit_rate": _ci(v)},
        "failures": [r for r in results if not r["ok"]],
    }


# ---------------------------------------------------------------------------
# Suite: calibration (ECE + Brier over confident settings routes).
# ---------------------------------------------------------------------------


def run_calibration(split: str = "dev") -> Dict[str, Any]:
    from assistant.core.router import DEFAULT_STATE, Router

    routing = load_set("routing", split)
    abst = load_set("abstention", split)
    router = Router()
    probs: List[float] = []
    outcomes: List[int] = []
    calib_rows: List[Tuple[float, int]] = []
    for item in routing["items"]:
        if not item.get("accept"):
            # expect_verdict items (removed-tool and A5
            # expect-AMBIGUOUS classes) assert a VERDICT, not a surface:
            # they have no correctness label for a confident route and
            # are excluded from the calibration histogram, not silently
            # scored as wrong.
            continue
        res = router.route(item["text"], state=DEFAULT_STATE, k=1)
        if res.verdict == "ROUTED" and res.candidates:
            probs.append(res.candidates[0].p)
            outcome = int(res.candidates[0].surface in item["accept"])
            outcomes.append(outcome)
            calib_rows.append((res.candidates[0].p, outcome))
    for item in abst["items"]:
        res = router.route(item["text"], state=DEFAULT_STATE, k=1)
        if res.verdict == "ROUTED" and res.candidates:
            top = res.candidates[0]
            if top.kind in ("tool", "preset"):
                probs.append(top.p)
                outcomes.append(0)  # a negative confidently routed is wrong
    metrics = {
        "ece_5bin": round(ece(probs, outcomes), 4),
        "brier": round(brier(probs, outcomes), 4),
    }
    note = ("cold-start: DEFAULT_STATE softmax probabilities, no learned "
            "calibration history — treat as uncalibrated until the learner "
            "has approvals (D8 labeling applies at the CLI surface)")
    out = {
        "suite": "calibration", "split": split, "n": len(probs),
        "metrics": metrics,
        "note": note,
        "thin": len(probs) < 30,
    }
    # A3: ECE before/after honest calibrators (isotonic PAVA,
    # Venn-Abers intervals), out-of-sample over deterministic folds.
    # These are the COLD-START numbers by construction; they move once
    # a real calibration stream exists in user state.
    if len(calib_rows) >= 20:
        try:
            from assistant.core.abers import ece_report
            out["calibration_report"] = ece_report(calib_rows)
        except Exception as exc:  # noqa: BLE001 — the arena reports, never crashes
            out["calibration_report_error"] = str(exc)
    return out


# ---------------------------------------------------------------------------
# Suite: footprint (in-process only; see module docstring).
# ---------------------------------------------------------------------------

FOOTPRINT_BUDGETS = {
    "peak_rss_kb": 45 * 1024,
    "first_route_ms": 400.0,
    "route_p50_ms": 15.0,
    "cold_cli_ms": 250.0,   # measured by scripts/measure_cli.py, reported here
}


def run_footprint() -> Dict[str, Any]:
    from assistant.core.router import DEFAULT_STATE, Router

    t0 = _now_ms()
    router = Router()
    fresh_router_ms = _now_ms() - t0
    probes = [
        "make the bar taller", "turn off blur", "show seconds on the clock",
        "fewer notification popups", "put the bar at the top",
        "make the dock icons bigger", "disable the launcher",
        "more spacing between things", "use 24 hour time",
        "optimize for gaming",
    ]
    lat: List[float] = []
    for text in probes:
        t1 = _now_ms()
        router.route(text, state=DEFAULT_STATE, k=5)
        lat.append(_now_ms() - t1)
    # repeat 5x for a stabler median
    for _ in range(4):
        for text in probes:
            t1 = _now_ms()
            router.route(text, state=DEFAULT_STATE, k=5)
            lat.append(_now_ms() - t1)
    rss = _peak_rss_kb()
    return {
        "suite": "footprint", "split": "n/a",
        "metrics": {
            "fresh_router_ms": round(fresh_router_ms, 1),
            "route_p50_ms": round(_median(lat), 2),
            "route_p95_ms": round(_percentile(lat, 0.95), 2),
            "peak_rss_kb": rss,
        },
        "budgets": FOOTPRINT_BUDGETS,
        "within_budget": {
            "route_p50_ms": _median(lat) <= FOOTPRINT_BUDGETS["route_p50_ms"],
            "peak_rss_kb": rss <= FOOTPRINT_BUDGETS["peak_rss_kb"],
        },
        "note": ("in-process measurements. fresh_router_ms is a new Router() over "
                 "already-built process singletons — the true COLD first-route "
                 "and cold-CLI numbers are subprocess measurements taken by "
                 "scripts/measure_cli.py (the behavioral tests forbid subprocess inside "
                 "assistant/; no exception was requested)"),
    }


# ---------------------------------------------------------------------------
# Dispatch.
# ---------------------------------------------------------------------------


RUNNERS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "routing": run_routing,
    "nlplan": run_nlplan,
    "abstention": run_abstention,
    "diagnosis": run_diagnosis,
    "calibration": run_calibration,
    "footprint": lambda **_kw: run_footprint(),
    # derived suite: variants generated from registry metadata, no set file
    "metamorphic": lambda split="dev": _run_metamorphic(split),
}


def _run_metamorphic(split: str) -> Dict[str, Any]:
    from .metamorphic import run_metamorphic
    return run_metamorphic(split)


def run_suite(suite: str, split: str = "dev") -> Dict[str, Any]:
    if suite == "all":
        out: Dict[str, Any] = {"suites": {}}
        for name in SUITES:
            if (name in ("footprint", "metamorphic")
                    or (SET_DIR / f"{name}_{split}.json").exists()):
                out["suites"][name] = RUNNERS[name](split=split)
        return out
    if suite not in RUNNERS:
        raise ValueError(f"unknown suite {suite!r} (choose from {', '.join(SUITES)})")
    return RUNNERS[suite](split=split)
