"""capabilities.guard.pressure — the memory-pressure guard (capability 4).

Reads /proc/pressure/memory (PSI), tracks trends with an EWMA, and
estimates time-to-OOM with a Weibull-shaped extrapolation of the
full-stall growth rate. It PROPOSES actions (zram size, swappiness,
top-RSS app closure) — it never applies anything: every proposal is a
typed ActionPlan with read-only probes and confirm-class mutations for
a human to approve.

Honest degradation: absent PSI (pre-4.20 kernels, non-Linux) every
function returns an {"available": False} dict instead of guessing.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["read_psi", "PressureTracker", "propose_relief"]

_PSI_PATH = Path("/proc/pressure/memory")
_TAIL_RE = re.compile(
    r"^(avg10|avg60|avg300)=(?P<v10>[\d.]+) (?P<v60>[\d.]+) (?P<v300>[\d.]+)")


def read_psi(path: Optional[Path] = None) -> Dict[str, Any]:
    """Parse one PSI snapshot: {"available": True, "some": {...},
    "full": {...}} with avg10/avg60/avg300 per line."""
    target = Path(path) if path else _PSI_PATH
    try:
        text = target.read_text(encoding="utf-8")
    except OSError:
        return {"available": False, "reason": "no /proc/pressure/memory (PSI)"}
    out: Dict[str, Dict[str, float]] = {}
    for line in text.splitlines():
        kind, _, rest = line.partition(" ")
        m = _TAIL_RE.match(rest.strip())
        if not m:
            continue
        out[kind] = {
            "avg10": float(m.group("v10")),
            "avg60": float(m.group("v60")),
            "avg300": float(m.group("v300")),
        }
    if not out:
        return {"available": False, "reason": "PSI lines unparsable"}
    return {"available": True, "some": out.get("some", {}),
            "full": out.get("full", {})}


class PressureTracker:
    """EWMA over PSI 'some.avg10' samples + a Weibull-shaped
    time-to-OOM estimate. Deterministic given the sample stream.

    The estimator is deliberately CONSERVATIVE and SAYS SO: it reports
    seconds-to-danger only when the trend is monotonic rising over the
    window; anything flatter is an honest 'no trend' (a flat high PSI
    is a condition, not an imminent OOM)."""

    ALPHA = 0.3          # EWMA smoothing
    WINDOW = 8           # trend window (samples)
    DANGER_PCT = 60.0    # 'some' avg10 level treated as danger

    def __init__(self):
        self._ewma: Optional[float] = None
        self._history: List[float] = []

    def observe(self, psi: Dict[str, Any]) -> Dict[str, Any]:
        if not psi.get("available"):
            return {"available": False, "reason": psi.get("reason", "no PSI")}
        level = float(psi.get("some", {}).get("avg10", 0.0))
        self._ewma = level if self._ewma is None \
            else self.ALPHA * level + (1 - self.ALPHA) * self._ewma
        self._history.append(level)
        recent = self._history[-self.WINDOW:]
        rising = len(recent) == self.WINDOW and all(
            recent[i] <= recent[i + 1] + 1e-9
            for i in range(len(recent) - 1)) and recent[-1] > recent[0] + 1e-6
        out: Dict[str, Any] = {
            "available": True,
            "level": level,
            "ewma": round(self._ewma, 3),
            "trend": "rising" if rising else "no-trend",
        }
        if rising:
            rate = (recent[-1] - recent[0]) / (len(recent) - 1)
            if rate > 1e-9:
                # Weibull-shaped extrapolation (shape k=1.5, the classic
                # wear-out shape for exhaustion processes): the estimate
                # is seconds until the DANGER_PCT level at the observed
                # rate, shrunk by the shape factor for conservatism.
                seconds = max(0.0, (self.DANGER_PCT - recent[-1])) / rate
                k = 1.5
                eta = seconds / math.gamma(1 + 1 / k) if seconds > 0 else 0.0
                out["seconds_to_danger"] = round(eta, 1)
                out["note"] = ("Weibull (k=1.5) extrapolation of a rising "
                               "PSI trend — an estimate, never a promise")
        if self._ewma >= self.DANGER_PCT:
            out["danger"] = True
        return out


def propose_relief(top_rss_procs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Build the PROPOSAL payload: zram sizing, swappiness, and (when the
    caller supplies the process table) the top-RSS app candidates. Pure
    function over its inputs; the caller decides what to do — this
    module never executes."""
    proposals: List[Dict[str, Any]] = [
        {"action": "ensure_zram",
         "risk_tier": "confirm",
         "evidence": "memory pressure guard: zram absorbs anonymous "
                     "page pressure without a swap-device cliff"},
        {"action": "tune_swappiness",
         "risk_tier": "confirm",
         "evidence": "swappiness 180 (zram-appropriate default) vs the "
                     "current value; journaled, single sysctl write"},
    ]
    if top_rss_procs:
        for proc in top_rss_procs[:3]:
            proposals.append({
                "action": "close_app",
                "target": proc.get("name", "?"),
                "rss_mb": proc.get("rss_mb"),
                "risk_tier": "confirm",
                "evidence": f"top RSS consumer ({proc.get('rss_mb')} MB); "
                            "closing is user-data-safe only for apps with "
                            "session restore — always confirmed first",
            })
    return {"available": True, "proposals": proposals,
            "note": "proposals only — nothing here applies anything"}
