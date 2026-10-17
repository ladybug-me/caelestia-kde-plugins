"""The eval arena: seeded measurement suites for the whole assistant.

Public API:
    run_suite(suite, split="dev") -> report dict
    load_set(suite, split) -> set dict

CLI: ``caelestia-assist eval [suite] [--json] [--sealed]``

Discipline (the arena is a measurement instrument, not a scoreboard):
- dev sets are for tuning; sealed sets are run only at stage boundaries.
- every rate carries a seeded bootstrap interval.
- ratchets live in ``eval/baseline.json`` and are enforced by
  ``tests/test_eval_ratchet.py`` (accuracy suites only — latency is a
  measurement, never a gate inside the package).
"""
from .engine import SUITES, load_set, run_suite  # noqa: F401

__all__ = ["SUITES", "load_set", "run_suite"]
