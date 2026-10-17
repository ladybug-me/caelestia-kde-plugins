"""Decide whether now is a good time to run heavier batch jobs (organize,
drift check, health scan) and which ones fit in the window — orchestration
only. This module never spawns anything itself (the behavioral tests ban
subprocess for the whole assistant); the caller outside this layer is
responsible for actually invoking whichever jobs get chosen.

Phase 2.1 adds the closed loop around that decision: a CadenceController
that reads diagnostics/telemetry.py's read-only /proc+/sys snapshots
DURING a run and adjusts the scan/brain batch cadence so the machine
stays under a configurable CPU ceiling. It is a PI controller with a
bounded integral (anti-windup) and explicit actuator limits — the
standard loop shape (Astrom & Hagglund 1995, "PID Controllers: Theory,
Design, and Tuning"); no new write path, no scheduler, no thread: the
runner calls observe() between job steps and honors the returned
cadence.
"""
import os
from typing import Any, Callable, Dict, Optional

# The 0/1 knapsack engine moved to brain/personal/ with the day-planning
# tools it was built for; dreamtime (the assistant's OWN batch-window
# scheduler, shell-native) still reuses the algorithm. This is the only
# root->personal import in the brain layer and it imports an engine, not
# a personal-surface function.
from .personal.planner import knapsack

__all__ = ["eligible", "schedule", "CadenceController",
           "DEFAULT_CEILING_PERCENT", "BASE_CADENCE_MIN",
           "MIN_CADENCE_MIN", "MAX_CADENCE_MIN", "load_percent_from_snapshot"]


def eligible(idle_minutes, on_ac_power, cpu_load_percent, min_idle=5, max_load=30):
    return idle_minutes >= min_idle and on_ac_power and cpu_load_percent <= max_load


def schedule(jobs, budget_min):
    """jobs: [{"id", "minutes", "value"}], value = how much you'd want it run
    now (e.g. staleness of that job's last run). Reuses the 0/1 knapsack
    planner. Returns the chosen job ids, highest-value first.
    """
    chosen, _, _ = knapsack(jobs, budget_min)
    by_id = {j["id"]: j for j in jobs}
    return sorted(chosen, key=lambda jid: -by_id[jid]["value"])


# ---------------------------------------------------------------------------
# Closed-loop cadence throttle (phase 2.1).
#
# eligible() decides IF a batch window may open at all; the controller
# decides HOW OFTEN the scan/brain cadence may repeat while jobs run, so
# the window itself stays under the ceiling instead of hogging the box
# the moment idle-time ends. Conservative defaults throughout.
# ---------------------------------------------------------------------------

# Default ceiling: BELOW eligible()'s own max_load=30 gate, so the loop
# starts braking before the window would lose its eligibility.
DEFAULT_CEILING_PERCENT = 25.0
BASE_CADENCE_MIN = 15.0    # minutes between batch runs when load is fine
MIN_CADENCE_MIN = 5.0      # never hammer: even a cold, idle box waits
MAX_CADENCE_MIN = 240.0    # actuator limit: four hours, then it is a
                           # scheduling decision, not a throttle decision
KP = 1.5                   # minutes of extra cadence per percent over ceiling
KI = 0.25                  # minutes per accumulated percent-observation


def load_percent_from_snapshot(snapshot: Dict[str, Any],
                               cores: Optional[int] = None
                               ) -> Optional[float]:
    """telemetry snapshot (or a bare loadavg dict) -> CPU load percent.

    load1 counts whole cores (1.0 = one core busy); the honest percent
    normalization needs the core count (os.cpu_count, a pure read).
    Returns None when the probe reports itself unavailable — the caller
    keeps its previous cadence instead of guessing."""
    loadavg = snapshot.get("loadavg", snapshot)  # bare loadavg dicts work
    if not loadavg.get("available"):
        return None
    load1 = loadavg.get("load1")
    if load1 is None:
        return None
    n = cores if cores is not None else (os.cpu_count() or 1)
    return float(load1) * 100.0 / max(1, int(n))


class CadenceController:
    """PI closed loop: observe load DURING batch runs, set the cadence.

    error = load_percent - ceiling (percent over the ceiling);
    integral accumulates the error across observations, bounded by
    anti-windup (never beyond what the actuator limits can unwind);
    cadence = base + Kp*error + Ki*integral, saturated to
    [min_cadence, max_cadence] — those are actuator limits of the
    controller spec (Astrom & Hagglund 1995), not input clamping.

    ``source`` is the injectable telemetry read (a zero-arg callable
    returning a telemetry snapshot or loadavg dict) — tests pass a
    fixture, the CLI passes diagnostics.telemetry.snapshot; NOTHING here
    touches /proc itself. An unavailable probe is an honest no-op: the
    cadence holds where it was."""

    def __init__(self,
                 ceiling_percent: float = DEFAULT_CEILING_PERCENT,
                 base_cadence_min: float = BASE_CADENCE_MIN,
                 min_cadence_min: float = MIN_CADENCE_MIN,
                 max_cadence_min: float = MAX_CADENCE_MIN,
                 kp: float = KP, ki: float = KI,
                 integral_limit: Optional[float] = None,
                 source: Optional[Callable[[], Dict[str, Any]]] = None,
                 cores: Optional[int] = None):
        if not (0.0 < ceiling_percent <= 100.0):
            raise ValueError(
                f"ceiling_percent must be inside (0, 100], got "
                f"{ceiling_percent}")
        if not (0.0 < min_cadence_min <= base_cadence_min <= max_cadence_min):
            raise ValueError(
                "cadence limits must satisfy "
                "0 < min <= base <= max, got "
                f"{min_cadence_min}/{base_cadence_min}/{max_cadence_min}")
        self.ceiling_percent = float(ceiling_percent)
        self.base_cadence_min = float(base_cadence_min)
        self.min_cadence_min = float(min_cadence_min)
        self.max_cadence_min = float(max_cadence_min)
        self.kp = float(kp)
        self.ki = float(ki)
        # anti-windup bound: the integral may never demand more than the
        # actuator limits can deliver (the standard conditional-integration
        # rule); an explicit limit overrides it
        self.integral_limit = (float(integral_limit) if integral_limit
                               is not None else (max_cadence_min - base_cadence_min))
        self.integral = 0.0
        self.cadence_min = float(base_cadence_min)
        self.source = source
        self.cores = cores
        self.observations = 0

    # -- the loop -------------------------------------------------------------

    def observe(self, reading: Optional[Dict[str, Any]] = None) -> float:
        """One loop step: take a load reading (explicit fixture value or
        the source), update the integral, return the new cadence in
        minutes. Missing/unavailable telemetry is a no-op (the cadence
        holds) — braking on a broken sensor would be a guess."""
        if reading is None and self.source is not None:
            try:
                reading = self.source()
            except OSError:
                reading = None  # a failed read is not a load reading
        load_percent = None
        if reading is not None:
            load_percent = load_percent_from_snapshot(reading, self.cores)
        if load_percent is not None:
            error = load_percent - self.ceiling_percent
            self.integral += error
            limit = self.integral_limit
            if limit is not None and self.integral > limit:
                self.integral = limit  # anti-windup (documented above)
            if limit is not None and self.integral < -limit:
                self.integral = -limit
            self.cadence_min = (
                self.base_cadence_min
                + self.kp * error
                + self.ki * self.integral)
            self.cadence_min = min(self.max_cadence_min,
                                   max(self.min_cadence_min,
                                       self.cadence_min))
            self.observations += 1
        return self.cadence_min

    # -- runner-facing helpers -----------------------------------------------

    def over_ceiling(self, reading: Optional[Dict[str, Any]] = None
                     ) -> bool:
        """True when the current reading is over the ceiling (the runner
        should yield between job steps). No reading -> False (never brake
        on a broken sensor)."""
        if reading is None and self.source is not None:
            try:
                reading = self.source()
            except OSError:
                reading = None
        if reading is None:
            return False
        load_percent = load_percent_from_snapshot(reading, self.cores)
        return load_percent is not None and load_percent > self.ceiling_percent

    def report(self) -> Dict[str, Any]:
        """The controller's own state, for the run report."""
        return {"ceiling_percent": self.ceiling_percent,
                "cadence_min": round(self.cadence_min, 3),
                "base_cadence_min": self.base_cadence_min,
                "integral": round(self.integral, 3),
                "observations": self.observations}
