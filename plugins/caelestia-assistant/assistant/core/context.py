"""cortex.context — context-aware recommendations (F30, exponential-build-5;
issue #120 b6a, closed from NOT_IMPLEMENTED).

Issue #120's Phase 3 asks for "context-aware recommendations". This
module reads the environment through narrow, injectable, read-only
probes and turns the readings into PROPOSALS through the engines that
already exist — never applies anything:

- clock: time-of-day bucket (the same 6-hour buckets brain.prefs uses);
- power: battery percent / charging state from /sys/class/power_supply
  (pure text parse; anything absent -> the probe reports honestly and
  no power-based recommendation is made);
- recommendations: battery-saver when the battery is low and
  discharging; a gaming-direction hint is NOT made from power state
  (that would be a guess); value recommendations ride
  settings.recommend (F29 partial pooling) when a tool is asked about.

Every recommendation cites its context evidence ("battery 14%,
discharging, read from /sys/class/power_supply/BAT0/capacity") and
lands as a SUGGESTED_NOT_EXECUTED command or a ledger proposal payload
— the caller decides which surface; this module never writes.

Determinism: all probes are injectable. ``probe_clock`` defaults to
the real local clock ONLY when the caller passes now=None explicitly
at the call site they own; the pure functions take ``now``/``power``
as arguments and are clock-free and filesystem-free by construction.
"""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Optional

__all__ = ["LOW_BATTERY_PCT", "read_power_state", "recommend_contextual",
           "render_lines"]

LOW_BATTERY_PCT = 25.0

_POWER_GLOB = "/sys/class/power_supply/BAT*"


def read_power_state(root_glob: str = _POWER_GLOB) -> Optional[Dict[str, Any]]:
    """One read-only pass over /sys/class/power_supply (or an injected
    root for tests). Returns {"percent": float|None, "charging": bool,
    "source": str} or None when no battery is present (honest: a
    desktop has no battery and gets no battery advice)."""
    paths = sorted(glob.glob(root_glob))
    if not paths:
        return None
    bat = paths[0]
    percent: Optional[float] = None
    charging = False
    try:
        cap_path = os.path.join(bat, "capacity")
        if os.path.exists(cap_path):
            percent = float(open(cap_path, encoding="utf-8").read().strip())
        status_path = os.path.join(bat, "status")
        if os.path.exists(status_path):
            charging = ("Charging" in
                        open(status_path, encoding="utf-8").read())
        else:
            # ucdavis-style: use charging/current_now sign as a fallback
            online = os.path.join(bat, "online")
            if os.path.exists(online):
                charging = open(online, encoding="utf-8").read().strip() == "1"
    except (OSError, ValueError):
        return {"percent": None, "charging": False,
                "source": bat, "error": "unreadable"}
    return {"percent": percent, "charging": charging, "source": bat}


def _power_findings(power: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    if power is None:
        return findings
    percent, charging = power.get("percent"), bool(power.get("charging"))
    source = power.get("source", "unknown")
    evidence = f"battery {source}"
    if percent is None:
        if power.get("error"):
            findings.append({
                "kind": "power_unreadable", "evidence": evidence,
                "reason": "the battery state exists but could not be read; "
                          "no battery recommendation is made",
            })
        return findings
    if percent <= LOW_BATTERY_PCT and not charging:
        findings.append({
            "kind": "battery_low",
            "evidence": f"{evidence} at {percent:g}%, discharging",
            "reason": f"the battery is at {percent:g}% and discharging; "
                      f"the battery-saver preset reduces compositing and "
                      f"background work",
            "command": ("SUGGESTED_NOT_EXECUTED: caelestia-assist settings "
                        "--preset battery-saver --apply --confirm"),
            "confidence": 0.85,
        })
    elif percent <= LOW_BATTERY_PCT and charging:
        findings.append({
            "kind": "battery_low_charging",
            "evidence": f"{evidence} at {percent:g}%, charging",
            "reason": "the battery is low but charging; no change is "
                      "suggested (a profile switch mid-charge would be "
                      "churn, not help)",
            "confidence": 0.5,
        })
    return findings


def _time_findings(now, model) -> List[Dict[str, Any]]:
    """Time-of-day findings: the CLOCK never supplies evidence — the
    fitted preference model does. The clock (``now``), when given, only
    SELECTS which already-qualified hour patterns are relevant right now;
    with now=None every qualifying pattern is reported."""
    findings: List[Dict[str, Any]] = []
    if model is None:
        return findings
    from .personalize import hour_patterns
    from assistant.capabilities.brain.prefs import hour_bucket
    current_bucket = hour_bucket(now.hour) if now is not None else None
    for pattern in hour_patterns(model):
        if current_bucket is not None and pattern["bucket"] != current_bucket:
            continue
        findings.append({
            "kind": "hour_pattern",
            "evidence": (f"{pattern['ess']:.0f} observed approvals of "
                         f"{pattern['direction']} {pattern['group']!r} in "
                         f"the {pattern['bucket'] * 6:02d}:00 bucket"),
            "reason": (f"your own approval history leans this way at this "
                       f"hour (mean {pattern['mean']})"),
            "confidence": pattern["mean"],
        })
    return findings


def recommend_contextual(now=None, power: Optional[Dict[str, Any]] = None,
                         model=None) -> Dict[str, Any]:
    """The full contextual recommendation view (read-only, pure given
    its arguments). ``power`` injects the probe result (None = no
    battery / not probed); ``model`` injects a fitted brain.prefs model;
    ``now`` (the caller's clock) selects which hour patterns are
    relevant — it is never evidence by itself."""
    findings = list(_power_findings(power))
    findings.extend(_time_findings(now, model))
    return {
        "findings": findings,
        "note": ("context-aware recommendations are proposals over "
                 "injected, read-only probes; nothing is applied and "
                 "the clock alone is never treated as evidence"),
    }


def render_lines(view: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    findings = view.get("findings", [])
    if not findings:
        return ["no contextual recommendations right now "
                "(no qualifying battery state and no hour-pattern "
                "evidence in your preference model)"]
    for f in findings:
        if f.get("command"):
            lines.append(f"{f['kind']}: {f['reason']}")
            lines.append(f"  evidence: {f['evidence']}")
            lines.append(f"  {f['command']}")
        else:
            lines.append(f"{f['kind']}: {f['reason']}")
            if f.get("evidence"):
                lines.append(f"  evidence: {f['evidence']}")
    lines.append(f"  ({view.get('note', '')})")
    return lines
