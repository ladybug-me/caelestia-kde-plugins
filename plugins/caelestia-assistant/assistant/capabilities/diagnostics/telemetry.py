"""Read-only telemetry probes (issue #120 Phase 3.1) — plain file reads.

The ONLY telemetry channel this assistant has: /proc and /sys files read
on demand via open()/pathlib — the same "file probes" precedent Layer 1
already uses for file_exists matchers. No daemon, no poller, no inotify:
every function here runs when a diagnostic verb, a report, or a dreamtime
batch job calls it, and costs one file read.

Probes (each takes an explicit path/glob parameter so tests can point it
at fixture files; the defaults are the real kernel interfaces):

- loadavg      /proc/loadavg                     -> 1/5/15-minute load
- meminfo      /proc/meminfo                     -> total/available/used ratio
- battery      /sys/class/power_supply/*/capacity -> per-battery percent
- thermal      /sys/class/thermal/*/temp         -> per-zone millidegrees C

Missing files are reported as {"available": False} — honestly unavailable,
never guessed, never defaulted to a fake number. Import policy clean:
pathlib, glob, json — no executor, no network.

The drain-rate helper exists for Phase 3.2: a battery-aware secondary
reward term needs a rate (%/hour) from two capacity samples, not just a
level. The mapping to the bandit's [0, 1] secondary reward lives next to
it (reward_from_drain) so the measurement→signal contract is stated once.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

PathLike = Union[str, Path]

# Battery drain-rate mapping (Phase 3.2 contract).
# A drain-rate DELTA of +/-DRAIN_FULL_DELTA_PCT_PER_H (%/hour faster or
# slower than before the preset was active) maps to the full [0, 1]
# secondary reward swing; beyond that it clamps.
DRAIN_FULL_DELTA_PCT_PER_H = 5.0
# The secondary term can never outweigh one real approve/reject decision.
SECONDARY_REWARD_WEIGHT = 0.25


def _read_text(path: PathLike) -> Optional[str]:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return None


def read_loadavg(path: PathLike = "/proc/loadavg") -> Dict[str, Any]:
    text = _read_text(path)
    if text is None:
        return {"available": False, "source": str(path)}
    parts = text.split()
    if len(parts) < 3:
        return {"available": False, "source": str(path)}
    try:
        return {"available": True, "source": str(path),
                "load1": float(parts[0]), "load5": float(parts[1]),
                "load15": float(parts[2])}
    except ValueError:
        return {"available": False, "source": str(path)}


def read_meminfo(path: PathLike = "/proc/meminfo") -> Dict[str, Any]:
    text = _read_text(path)
    if text is None:
        return {"available": False, "source": str(path)}
    fields: Dict[str, int] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        value = rest.strip().split()
        if value and value[0].isdigit():
            fields[key.strip()] = int(value[0])
    total = fields.get("MemTotal")
    if total is None:
        return {"available": False, "source": str(path)}
    available = fields.get("MemAvailable", fields.get("MemFree"))
    result: Dict[str, Any] = {"available": True, "source": str(path),
                              "total_kb": total}
    if available is not None:
        result["available_kb"] = available
        result["used_ratio"] = round(1.0 - available / total, 4) if total else None
    return result


def read_battery(pattern: str = "/sys/class/power_supply/*/capacity") -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(pattern)):
        text = _read_text(path)
        if text is None:
            continue
        try:
            pct = int(text.strip())
        except ValueError:
            continue
        if 0 <= pct <= 100:
            rows.append({"path": path, "capacity_pct": pct})
    return {"available": bool(rows), "batteries": rows}


def read_thermal(pattern: str = "/sys/class/thermal/*/temp") -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(pattern)):
        text = _read_text(path)
        if text is None:
            continue
        try:
            milli = int(text.strip())
        except ValueError:
            continue
        rows.append({"path": path, "temp_c": round(milli / 1000.0, 1)})
    return {"available": bool(rows), "zones": rows}


def snapshot(proc_dir: str = "/proc", sys_dir: str = "/sys") -> Dict[str, Any]:
    """One on-demand snapshot of every probe that is available.

    Never raises for missing interfaces: each probe reports its own
    availability, so a container or a desktop without a battery yields an
    honest partial snapshot.
    """
    return {
        "loadavg": read_loadavg(Path(proc_dir) / "loadavg"),
        "meminfo": read_meminfo(Path(proc_dir) / "meminfo"),
        "battery": read_battery(str(Path(sys_dir) / "class/power_supply/*/capacity")),
        "thermal": read_thermal(str(Path(sys_dir) / "class/thermal/*/temp")),
    }


# ---------------------------------------------------------------------------
# Phase 3.2 helpers: battery drain rate -> the bandit's secondary reward.
# ---------------------------------------------------------------------------

def drain_rate_percent_per_hour(capacity_before: float, capacity_after: float,
                                hours: float) -> float:
    """Positive = the battery DRAINED at that rate (%/hour).

    Raises ValueError for a non-positive window or capacities outside
    [0, 100] — measurement garbage is refused, not averaged in.
    """
    if hours <= 0:
        raise ValueError("hours must be > 0")
    for value in (capacity_before, capacity_after):
        if not 0 <= value <= 100:
            raise ValueError("capacities must be percentages in [0, 100]")
    return (capacity_before - capacity_after) / hours


def reward_from_drain_delta(delta_pct_per_hour: float,
                            full_delta: float = DRAIN_FULL_DELTA_PCT_PER_H,
                            ) -> float:
    """A drain-rate DELTA (%/hour; negative = battery drains SLOWER than
    before, i.e. the preset helped) -> secondary reward in [0, 1].

    0.5 is the neutral midpoint. Mapping: 0.5 - 0.5 * clamp(delta /
    full_delta) — a 5%/h improvement with the default full_delta gives 1.0,
    a 5%/h regression gives 0.0. The bandit weights this by
    SECONDARY_REWARD_WEIGHT, so it can never outweigh one real
    approve/reject decision.
    """
    if full_delta <= 0:
        raise ValueError("full_delta must be > 0")
    clamped = max(-full_delta, min(full_delta, delta_pct_per_hour))
    return round(0.5 - 0.5 * (clamped / full_delta), 4)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="telemetry",
        description="Read-only /proc and /sys snapshot (on-demand; runs "
                    "once, prints, exits — never a daemon).")
    parser.add_argument("--proc", default="/proc", help="proc base dir override")
    parser.add_argument("--sys", default="/sys", help="sys base dir override")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)
    snap = snapshot(proc_dir=args.proc, sys_dir=args.sys)
    if args.json:
        print(json.dumps(snap, indent=2))
        return 0
    load = snap["loadavg"]
    if load["available"]:
        print(f"load: {load['load1']:.2f} {load['load5']:.2f} {load['load15']:.2f}")
    mem = snap["meminfo"]
    if mem["available"]:
        used = mem.get("used_ratio")
        print(f"memory: {mem.get('available_kb', '?')} kB available of "
              f"{mem['total_kb']} kB" + (f" (used {used:.0%})" if used is not None else ""))
    bat = snap["battery"]
    for row in bat["batteries"]:
        print(f"battery: {row['capacity_pct']}%  ({row['path']})")
    for row in snap["thermal"]["zones"]:
        print(f"thermal: {row['temp_c']}C  ({row['path']})")
    missing = [name for name, probe in snap.items() if not probe["available"]]
    if missing:
        print("unavailable: " + ", ".join(missing))
    return 0


# ---------------------------------------------------------------------------
# Time-series accumulation (exponential-build-4 E) — the bridge between
# the one-shot snapshot above and diagnostics/forecast.py's Kalman trend.
# The series lives in the caller's learned-state JSON (the enumerated
# write path); this module still performs no writes itself.
# ---------------------------------------------------------------------------

SERIES_STATE_KEY = "telemetry_series"
SERIES_MAX_POINTS = 240


def _flatten(snapshot: Dict[str, Any]) -> Dict[str, float]:
    """The numeric leaves of one snapshot as a flat {metric: value} —
    the same flattening convention robust_baseline.flatten_snapshot
    uses, so both consumers agree on metric names."""
    from . import robust_baseline
    return robust_baseline.flatten_snapshot(snapshot)


def record_series(state: Dict[str, Any], snapshot: Dict[str, Any],
                  when: str, key: str = SERIES_STATE_KEY) -> Dict[str, Any]:
    """Append one flattened snapshot to the persisted series. ``when``
    is the caller-supplied ISO timestamp (determinism: nothing here
    reads the clock). Bounded at SERIES_MAX_POINTS (oldest dropped —
    a series that grows without bound is a leak, not a feature)."""
    flat = _flatten(snapshot)
    series = state.setdefault(key, [])
    series.append({"when": when, "metrics": flat})
    if len(series) > SERIES_MAX_POINTS:
        del series[:len(series) - SERIES_MAX_POINTS]
    return series


def series_for(state: Dict[str, Any], metric: str,
               key: str = SERIES_STATE_KEY) -> List[Tuple[str, float]]:
    """One metric's [(when, value)] series in arrival order — the input
    shape diagnostics.forecast.steady_state_kalman consumes (values
    only) with the dates carried for the human report. Missing
    readings are SKIPPED, not interpolated (a gap is honest; a
    fabricated point is not)."""
    out: List[Tuple[str, float]] = []
    for row in state.get(key, []):
        metrics = row.get("metrics") or {}
        if metric in metrics and isinstance(metrics[metric], (int, float)):
            out.append((row.get("when", ""), float(metrics[metric])))
    return out
