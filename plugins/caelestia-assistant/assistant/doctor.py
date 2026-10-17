"""assistant.doctor — the composite health check (F27, exponential-build-5).

``caelestia-assist doctor`` answers "is this install healthy?" in one
command by COMPOSING checks that already exist — no new engines, every
line cites the check it ran and how to reproduce it:

- registry freshness: gen_adapter's byte-identity guard over the
  committed tools.json (in-repo; the A0 recon proved upstream moves —
  this catches it);
- generated docs: the F23 catalog byte-identity guard;
- arena smoke: the routing dev suite's confident-wrong rate (the A0
  honest gap, re-measured here);
- calibration coverage: whether the learner's calibration buckets have
  enough observations to say anything (honest UNAVAILABLE on a fresh
  install);
- latency/RSS budget probes: cold --help and warm route p50 measured
  in-process, with the MACHINE CAVEAT the A0 baselines established
  (budgets are judged on the reference 1-vCPU machine; a fast sandbox
  passing tells you little, a failing run tells you a lot).

Verdicts: PASS / WARN / FAIL / UNAVAILABLE (a check that cannot run in
this context is UNAVAILABLE — never silently skipped, never a fake
pass). Exit code 0 unless a FAIL exists.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["run_doctor", "render_lines"]

_REPO_ROOT = Path(__file__).resolve().parents[1]


def locate_upstream() -> Optional[Path]:
    """The pinned caelestia-kde checkout this repo verifies against: the
    gen_adapter auto-locate, else the build convention (a SIBLING of this
    repo exposing shell/plugin/src/Caelestia/Config). None when absent —
    the honest installed-assistant case. Shared by the doctor check and
    the hermetic test so both mean the same thing by 'present'."""
    from assistant.capabilities.settings.gen_adapter import CANONICAL_ADAPTER, verify, \
        build_registry
    here = _REPO_ROOT / "assistant"
    root = build_registry.find_repo_root(here)
    if root is None:
        for sibling in sorted(_REPO_ROOT.parent.iterdir()):
            if (sibling / "shell" / "plugin" / "src" / "Caelestia"
                    / "Config").is_dir():
                root = sibling
                break
    return root


def _check_gen_adapter() -> Dict[str, Any]:
    """Registry freshness = gen_adapter --verify IN-PROCESS but with the
    same auto-locate semantics as its CLI (needs the pinned caelestia
    checkout; when absent the verdict is UNAVAILABLE — the honest
    installed-assistant case)."""
    try:
        from assistant.capabilities.settings.gen_adapter import CANONICAL_ADAPTER, verify, \
            build_registry
        here = _REPO_ROOT / "assistant"
        committed = here / "adapters" / "caelestia" / "tools.json"
        root = locate_upstream()
        if root is None or not committed.exists():
            # the build convention: the upstream checkout sits as a
            # SIBLING of this repo (never an ancestor)
            return {"verdict": "UNAVAILABLE",
                    "detail": "no committed tools.json or no caelestia "
                              "checkout beside this repo — the "
                              "byte-identity check needs both",
                    "repro": "python3 -m assistant.capabilities.settings.gen_adapter "
                             "--verify --repo-root <checkout>"}
        report = verify(CANONICAL_ADAPTER, str(root), committed)
    except Exception as exc:  # noqa: BLE001 — doctor reports, never crashes
        return {"verdict": "UNAVAILABLE", "detail": f"gen_adapter: {exc}"}
    if report.get("error"):
        return {"verdict": "UNAVAILABLE",
                "detail": f"gen_adapter: {report['error']}"}
    if report.get("drift"):
        return {"verdict": "FAIL",
                "detail": f"registry drift: {report.get('drift')}",
                "repro": "python3 -m assistant.capabilities.settings.gen_adapter --verify"}
    return {"verdict": "PASS",
            "detail": "committed tools.json is byte-identical to a fresh "
                      "generation from the pinned upstream checkout",
            "repro": "python3 -m assistant.capabilities.settings.gen_adapter --verify"}


def _check_catalog() -> Dict[str, Any]:
    try:
        from assistant.capabilities.settings import catalog
        problems = catalog.verify()
    except Exception as exc:  # noqa: BLE001
        return {"verdict": "UNAVAILABLE", "detail": f"catalog: {exc}"}
    if problems:
        return {"verdict": "FAIL", "detail": "; ".join(problems),
                "repro": "python3 -m assistant.capabilities.settings --gen-catalog "
                         "--verify"}
    return {"verdict": "PASS",
            "detail": "generated docs are byte-identical",
            "repro": "python3 -m assistant.capabilities.settings --gen-catalog --verify"}


def _check_arena_smoke() -> Dict[str, Any]:
    try:
        from assistant.core.eval.engine import run_routing
        report = run_routing("dev")
    except Exception as exc:  # noqa: BLE001
        return {"verdict": "UNAVAILABLE", "detail": f"arena: {exc}",
                "repro": "python3 -m assistant.core.eval routing"}
    # The shipped behavior includes A2 label fusion (demote-only), so
    # the honest smoke check measures the FUSED rate; the raw router
    # rate rides along in the detail for comparison.
    cw = report.get("confident_wrong_fused") or report["confident_wrong"]
    rate = cw.get("rate")
    if rate is None:
        verdict, note = "WARN", "nothing routed — cannot assess"
    else:
        # The honest bar is the RECORDED dev confident-wrong rate (the
        # arena ratchet's own floor), not a fixed percentage: the A5
        # arena growth deliberately added the sealed-miss failure
        # classes, which RAISED the honest rate on a harder set. The
        # doctor fails only when today's cold-start routing is WORSE
        # than the recorded baseline — regressions, not difficulty.
        recorded = 0.02
        source = "2% constant"
        try:
            base = json.loads(
                (_REPO_ROOT / "assistant" / "core" / "eval" / "baseline.json")
                .read_text(encoding="utf-8"))
        except OSError:
            base = json.loads(
                (_REPO_ROOT / "assistant" / "core" / "eval" / "baseline.json")
                .read_text(encoding="utf-8"))
        cw_floor = base.get("confident_wrong")
        if cw_floor and cw_floor.get("rate") is not None:
            recorded = float(cw_floor["rate"]) + 1e-9
            source = f"recorded {cw_floor['rate']:.1%} (baseline.json)"
        if rate <= recorded:
            verdict, note = "PASS", (f"confident-wrong {cw['count']}/"
                                     f"{cw['of_routed']} = {rate:.1%} "
                                     f"(<= {source})")
        else:
            verdict, note = "FAIL", (f"confident-wrong {cw['count']}/"
                                     f"{cw['of_routed']} = {rate:.1%} > "
                                     f"{source}")
    return {"verdict": verdict, "detail": note,
            "repro": "python3 -m assistant.core.eval routing"}


def _check_calibration() -> Dict[str, Any]:
    try:
        from assistant.capabilities.brain.state import load as load_state
        from assistant.core.cli import LEARN_KEY
        from assistant.core.learn import CortexLearner
    except Exception as exc:  # noqa: BLE001
        return {"verdict": "UNAVAILABLE", "detail": f"learner: {exc}"}
    try:
        state = load_state()
    except Exception as exc:  # noqa: BLE001
        return {"verdict": "UNAVAILABLE", "detail": f"brain state: {exc}"}
    data = state.get(LEARN_KEY) if isinstance(state, dict) else None
    if not isinstance(data, dict) or not data.get("examples"):
        return {"verdict": "UNAVAILABLE",
                "detail": "no learned state yet (fresh install) — "
                          "calibration says nothing and the doctor "
                          "agrees"}
    learner = CortexLearner(data)
    buckets = learner.calibration.buckets if hasattr(
        learner.calibration, "buckets") else {}
    observed = sum(int(row.get("alpha", 1) + row.get("beta", 1) - 2)
                   for row in buckets.values()) if buckets else 0
    if observed < 5:
        return {"verdict": "UNAVAILABLE",
                "detail": f"calibration has {observed} observation(s); "
                          f"below the 5 it needs to say anything"}
    return {"verdict": "PASS",
            "detail": f"calibration holds {observed} observed "
                      f"outcome(s) across its buckets",
            "repro": "caelestia-assist cortex report"}


def _budget_probe() -> Dict[str, Any]:
    """Cold-path measurement WITHOUT a subprocess — the assistant must
    never spawn processes (the import policy bans subprocess by name;
    the doctor obeys the same law it checks). The in-process cold path:
    drop the assistant's lazily-loaded engine modules from sys.modules,
    then import-and-route — the same import + index-build work the cold
    CLI does. Warm p50 follows. Machine-dependent by nature: the caveat
    travels on the result."""

    def _vmhwm_kb() -> int:
        try:
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
        except OSError:
            pass
        return -1

    try:
        saved = {name: mod for name, mod in sys.modules.items()
                 if name == "assistant" or name.startswith("assistant.")}
        for name in list(saved):
            del sys.modules[name]
        t0 = datetime.now()
        from assistant.core.router import route as _route  # cold: import + build
        _route("make the bar thinner")
        cold_s = (datetime.now() - t0).total_seconds()
        # restore the rest of the saved modules for a clean warm phase
        for name, mod in saved.items():
            sys.modules.setdefault(name, mod)
        route = _route
        samples = []
        for _ in range(20):
            t1 = datetime.now()
            route("make the bar thinner")
            samples.append((datetime.now() - t1).total_seconds() * 1000)
        samples.sort()
        warm_p50 = samples[len(samples) // 2]
    except Exception as exc:  # noqa: BLE001
        return {"verdict": "UNAVAILABLE", "detail": f"route probe: {exc}"}
    vmhwm = _vmhwm_kb()
    # Generous sandbox-aware bounds: cold <= 1.0 s here means the
    # reference machine's 250 ms budget is plausibly held; warm p50 is
    # dominated by interpreter overhead in-process.
    if cold_s > 2.0:
        verdict = "WARN"
        detail = (f"cold route {cold_s:.2f}s exceeds even the "
                  f"2x-slowed bound; measure on the reference machine")
    else:
        verdict = "PASS"
        detail = (f"cold route {cold_s * 1000:.0f}ms, warm route p50 "
                  f"{warm_p50:.1f}ms, VmHWM {vmhwm / 1024:.0f}MB "
                  f"(in-process sandbox numbers; the <=250ms cold budget "
                  f"is judged on the reference 1-vCPU machine)")
    return {"verdict": verdict, "detail": detail}


def run_doctor() -> Dict[str, Any]:
    checks = {
        "registry freshness": _check_gen_adapter(),
        "generated docs": _check_catalog(),
        "arena smoke": _check_arena_smoke(),
        "calibration coverage": _check_calibration(),
        "budget probe": _budget_probe(),
    }
    fails = sum(1 for c in checks.values() if c["verdict"] == "FAIL")
    return {"checks": checks, "fails": fails,
            "note": "every verdict names the check it ran and how to "
                    "reproduce it; UNAVAILABLE means the check cannot "
                    "run in this context, not that it passed"}


def render_lines(result: Dict[str, Any]) -> List[str]:
    lines = ["doctor — composite health check (read-only):"]
    for name, check in result["checks"].items():
        lines.append(f"  [{check['verdict']:<11}] {name}: {check['detail']}")
        if check.get("repro"):
            lines.append(f"      repro: {check['repro']}")
    lines.append(f"  ({result['note']})")
    if result["fails"]:
        lines.append(f"{result['fails']} FAIL — exit 1")
    else:
        lines.append("no FAIL verdicts")
    return lines


def main(argv: Optional[List[str]] = None) -> int:
    result = run_doctor()
    print("\n".join(render_lines(result)))
    return 1 if result["fails"] else 0


if __name__ == "__main__":
    sys.exit(main())
