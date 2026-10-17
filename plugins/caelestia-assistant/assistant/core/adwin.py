"""cortex.adwin — ADWIN, the SECOND drift detector beside Page-Hinkley
over the acceptance stream.

Bifet & Gavaldà 2007, "Learning from Time-Changing Data with Adaptive
Windowing", SIAM Data Mining (SDM) 2007 — the compressed bucket-list
variant (the paper's Algorithm 1): an exponential histogram whose
buckets each summarize a run of 2^i consecutive elements as
(total, variance), at most ``max_buckets`` (M=5, the paper's value)
buckets per row. On every element the window is re-examined at every
bucket-boundary split: if the two subwindows' means differ by more
than the Hoeffding bound

    epsilon_cut = sqrt( ln(4/delta) / (2m) ),
    m = 1 / (1/n0 + 1/n1)        (the paper's effective sample size),

the older subwindow is CUT — the window keeps only the recent,
apparently-homogeneous part. The 2009 Knowledge and Information Systems
journal version adds a variance-completion term to epsilon_cut; this
module implements the SDM 2007 variant (the plain Hoeffding bound) —
the buckets still carry (total, variance) per the paper's structure,
but the cut criterion uses only the means. Adapted, not invented: the
exponential-histogram compression is exactly the paper's.

The API mirrors ``PageHinkleyDrift`` in cortex/conformal.py (same
update / to_dict / from_dict / reset shape) so a caller can hold one
of each; ``DriftConsensus`` below holds exactly that pair.

THE CONSENSUS GATE (the integration requirement): drift is FLAGGED TO
THE USER only when BOTH detectors have alarmed — dual agreement is the
point, suppressing single-detector false alarms — while each
detector's own state is always reported individually.

Honesty rules: warmup (width < 32: no alarm — the Hoeffding bound is
vacuous on tiny subwindows, so the check is simply not made and the
report says so); the estimate and width are always reported; alarms
LATCH (like Page-HinkleyDrift) until reset() — the alarm_at index is
the element count at which the first cut happened. Determinism: pure
arithmetic, no RNG anywhere. This module writes NOTHING: state moves
only through plain dicts (the caller's learned-state JSON path
persists them). A brute-force sliding-window cross-check of the cut
behaviour is pinned in tests/test_adwin.py.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .conformal import PageHinkleyDrift

__all__ = ["ADWIN", "DriftConsensus", "consensus"]


# ---------------------------------------------------------------------------
# The detector
# ---------------------------------------------------------------------------

class ADWIN:
    """Adaptive Windowing over the accept/reject stream.

    ``update(accepted)`` returns ``{"drift": bool, "width": n,
    "estimate": mean, "alarm_at": Optional[int]}`` — ``drift`` is the
    EVENT (a cut happened during this update); ``alarm_at`` is the
    latched index of the FIRST alarm (reset() clears it). ``width`` is
    the current window size; ``estimate`` its mean.
    """

    def __init__(self, delta: float = 0.002, max_buckets: int = 5,
                 min_width: int = 32) -> None:
        """``delta``: the paper's confidence parameter (default 0.002,
        within the SDM paper's suggested range); ``max_buckets``: M=5
        rows of compression per level; ``min_width``: warmup — no cut
        is even checked below this width (the Hoeffding bound is
        vacuous on tiny subwindows)."""
        if not (0.0 < delta < 1.0):
            raise ValueError("delta in (0, 1)")
        if max_buckets < 2:
            raise ValueError("max_buckets >= 2")
        if min_width < 2:
            raise ValueError("min_width >= 2")
        self.delta = float(delta)
        self.max_buckets = int(max_buckets)
        self.min_width = int(min_width)
        # rows[i] holds buckets of size 2^i, oldest -> newest; each
        # bucket is [total, variance]. width/mean are derived from the
        # buckets (single source of truth, no bookkeeping to drift).
        self.rows: List[List[List[float]]] = [[]]
        self.n = 0                     # elements added since reset
        self.alarm_at: Optional[int] = None

    # -- bucket mechanics ------------------------------------------------

    def _flat(self) -> List[Tuple[int, float, float]]:
        """The bucket list as ONE chronological sequence oldest->newest:
        [(size, total, variance), ...]. Invariant (kept by the
        merge-oldest-first discipline, verified by tests): sizes are
        non-increasing powers of two, so higher rows are always older."""
        out: List[Tuple[int, float, float]] = []
        for i in range(len(self.rows) - 1, -1, -1):
            size = 1 << i
            for total, variance in self.rows[i]:
                out.append((size, total, variance))
        return out

    def _compress(self) -> None:
        """Merge the two OLDEST buckets of any overflowing row into the
        next row (the paper's compress step, cascading outward)."""
        i = 0
        while i < len(self.rows):
            while len(self.rows[i]) > self.max_buckets:
                b1 = self.rows[i].pop(0)
                b2 = self.rows[i].pop(0)
                if i + 1 >= len(self.rows):
                    self.rows.append([])
                # pooled variance of the merged run (exact identity)
                n1 = n2 = float(1 << i)
                total = b1[0] + b2[0]
                m1 = b1[0] / n1
                m2 = b2[0] / n2
                mean = total / (n1 + n2)
                variance = ((n1 * (b1[1] + m1 * m1)
                             + n2 * (b2[1] + m2 * m2)) / (n1 + n2)
                            - mean * mean)
                self.rows[i + 1].append([total, max(variance, 0.0)])
            i += 1

    def _rebuild(self, flat: Sequence[Tuple[int, float, float]]) -> None:
        """Reinstall the window from a flat bucket list (after a cut —
        only bucket removal ever happens there, so every row stays
        within the per-row cap)."""
        self.rows = [[]]
        for size, total, variance in flat:
            row = size.bit_length() - 1
            while row + 1 > len(self.rows):
                self.rows.append([])
            self.rows[row].append([total, variance])

    def _cut_once(self) -> bool:
        """The paper's cut check over every bucket-boundary split point:
        W0 = the oldest prefix, W1 = the rest. Cut (drop W0) at the
        first split where |mu0 - mu1| >= epsilon_cut. Returns whether a
        cut happened. Below min_width nothing is checked (warmup)."""
        flat = self._flat()
        width = sum(size for size, _t, _v in flat)
        if width < self.min_width:
            return False
        total = sum(t for _s, t, _v in flat)
        n0 = 0
        t0 = 0.0
        for k in range(len(flat) - 1):          # W0 = flat[:k+1]
            size, bucket_total, _v = flat[k]
            n0 += size
            t0 += bucket_total
            n1 = width - n0
            if n1 <= 0:
                break
            mu0 = t0 / n0
            mu1 = (total - t0) / n1
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            epsilon = math.sqrt(math.log(4.0 / self.delta) / (2.0 * m))
            if abs(mu0 - mu1) >= epsilon:
                self._rebuild(flat[k + 1:])
                return True
        return False

    # -- the PH-shaped API ------------------------------------------------

    def update(self, accepted: bool) -> Dict[str, Any]:
        """Feed one accept (True) / reject (False); returns the report."""
        self.rows[0].append([1.0 if accepted else 0.0, 0.0])
        self.n += 1
        self._compress()
        drift = False
        while self._cut_once():
            drift = True
            if self.alarm_at is None:
                self.alarm_at = self.n
        flat = self._flat()
        width = sum(size for size, _t, _v in flat)
        total = sum(t for _s, t, _v in flat)
        return {"drift": drift,
                "width": width,
                "estimate": round(total / width, 6) if width else 0.0,
                "alarm_at": self.alarm_at}

    def status(self) -> Dict[str, Any]:
        flat = self._flat()
        width = sum(size for size, _t, _v in flat)
        total = sum(t for _s, t, _v in flat)
        return {
            "n": self.n,
            "width": width,
            "estimate": round(total / width, 6) if width else None,
            "alarmed": self.alarm_at is not None,
            "alarm_at": self.alarm_at,
            "delta": self.delta,
            "max_buckets": self.max_buckets,
            "min_width": self.min_width,
            "buckets": len(flat),
            "warmup": width < self.min_width,
            "message": (f"acceptance-rate mean shift detected — the window "
                        f"was cut at element {self.alarm_at}")
                        if self.alarm_at is not None else
                        ("no drift detected" if width >= self.min_width
                         else f"warmup: width {width} < {self.min_width}, "
                              f"no cut is checked yet"),
        }

    def reset(self) -> None:
        """Clear the window and the latch after acting."""
        self.rows = [[]]
        self.n = 0
        self.alarm_at = None

    def to_dict(self) -> Dict[str, Any]:
        flat = self._flat()
        width = sum(size for size, _t, _v in flat)
        total = sum(t for _s, t, _v in flat)
        return {"delta": self.delta, "max_buckets": self.max_buckets,
                "min_width": self.min_width, "n": self.n,
                "alarm_at": self.alarm_at, "width": width,
                "estimate": round(total / width, 6) if width else None,
                "buckets": [[size, total, variance]
                            for size, total, variance in flat]}

    def from_dict(self, d: Dict[str, Any]) -> "ADWIN":
        d = d if isinstance(d, dict) else {}
        self.delta = float(d.get("delta", self.delta))
        self.max_buckets = int(d.get("max_buckets", self.max_buckets))
        self.min_width = int(d.get("min_width", self.min_width))
        self.n = int(d.get("n", 0))
        alarm = d.get("alarm_at")
        self.alarm_at = int(alarm) if alarm is not None else None
        self.rows = [[]]
        for bucket in d.get("buckets", []) or []:
            size, total, variance = bucket
            row = int(size).bit_length() - 1
            while row + 1 > len(self.rows):
                self.rows.append([])
            self.rows[row].append([float(total), float(variance)])
        return self


# ---------------------------------------------------------------------------
# The consensus gate
# ---------------------------------------------------------------------------

def consensus(ph_state: Dict[str, Any],
              adwin_state: Dict[str, Any]) -> Dict[str, Any]:
    """Pure consensus over the two PERSISTED state dicts (for callers
    that hold one detector of each without a DriftConsensus). The rule:
    drift is FLAGGED TO THE USER only when BOTH detectors have alarmed
    (dual agreement suppresses single-detector false alarms); each
    detector's own state is always reported individually."""
    ph_alarm = ph_state.get("alarm_at") is not None
    ad_alarm = adwin_state.get("alarm_at") is not None
    flagged = ph_alarm and ad_alarm
    return {
        "page_hinkley": {
            "alarmed": ph_alarm,
            "alarm_at": ph_state.get("alarm_at"),
            "n": ph_state.get("n"),
            "rejection_ema": ph_state.get("mean"),
        },
        "adwin": {
            "alarmed": ad_alarm,
            "alarm_at": adwin_state.get("alarm_at"),
            "width": adwin_state.get("width"),
            "estimate": adwin_state.get("estimate"),
        },
        "flag_drift": flagged,
        "rule": ("drift is flagged to the user only when BOTH detectors "
                 "have alarmed (dual agreement suppresses single-detector "
                 "false alarms); each detector's own state is always "
                 "reported individually"),
        "summary": ("FLAGGED — both detectors alarmed" if flagged else
                    "not flagged (page-hinkley "
                    f"{'alarmed' if ph_alarm else 'quiet'}, adwin "
                    f"{'alarmed' if ad_alarm else 'quiet'})"),
    }


class DriftConsensus:
    """One PageHinkleyDrift + one ADWIN over the SAME acceptance
    stream, with the consensus report on top. ``update(accepted)``
    feeds both and returns the combined report; ``status()`` is the
    read-only view (both individual states + the joint flag)."""

    def __init__(self, ph: Optional[PageHinkleyDrift] = None,
                 adwin: Optional[ADWIN] = None) -> None:
        self.ph = ph if ph is not None else PageHinkleyDrift()
        self.adwin = adwin if adwin is not None else ADWIN()

    def update(self, accepted: bool) -> Dict[str, Any]:
        ph_row = self.ph.update(accepted)
        ad_row = self.adwin.update(accepted)
        out = consensus(self.ph.to_dict(), self.adwin.to_dict())
        out["last_update"] = {"page_hinkley": ph_row, "adwin": ad_row}
        return out

    def status(self) -> Dict[str, Any]:
        return consensus(self.ph.to_dict(), self.adwin.to_dict())

    def reset(self) -> None:
        self.ph.reset()
        self.adwin.reset()

    def to_dict(self) -> Dict[str, Any]:
        return {"page_hinkley": self.ph.to_dict(),
                "adwin": self.adwin.to_dict()}

    def from_dict(self, d: Dict[str, Any]) -> "DriftConsensus":
        d = d if isinstance(d, dict) else {}
        ph_data = d.get("page_hinkley") or {}
        ad_data = d.get("adwin") or {}
        self.ph = PageHinkleyDrift().from_dict(
            ph_data if isinstance(ph_data, dict) else {})
        self.adwin = ADWIN().from_dict(
            ad_data if isinstance(ad_data, dict) else {})
        return self
