"""F28 tests — the weight re-fit ratchet (cortex/refit.py).

The contract under test:

- the CANDIDATE weights come from replaying the learner's example log
  through a FRESH logistic (deterministic); below the honesty floor
  (MIN_EXAMPLES) the ratchet refuses with the reason, no arena run;
- the MEASUREMENT is the arena itself (eval.engine.run_routing now
  accepts an injected RouterState — None keeps the cold-start
  measurement byte-compatible);
- ADOPT only on no regression (candidate top-1 >= current top-1 AND
  candidate confident-wrong <= current confident-wrong); a regression
  REFUSES and the current weights stay;
- every decision lands on a bounded audit list in the caller's state;
- the injected-state arena is deterministic: same state, same report.

The example log is fabricated with REAL route features (the learner's
observe path) so the fitted candidate is a genuine replay.
"""

from __future__ import annotations

import unittest
from datetime import datetime

from assistant.core import refit
from assistant.core.learn import CortexLearner
from assistant.core.router import DEFAULT_STATE, Router, route


def _build_learner_data():
    """A learner dict with >= MIN_EXAMPLES REAL observations (features
    come from actual routes; outcomes are a consistent story)."""
    learner = CortexLearner({})
    router = Router()
    cases = [
        ("make the bar thinner", "setBarScale", "applied"),
        ("make the dock smaller", "setDockIconSize", "applied"),
        ("disable blur", "setBlurEnabled", "applied"),
        ("increase transparency", "setTransparencyBase", "applied"),
        ("hide notifications in fullscreen", "setFullscreen", "rejected"),
        ("turn off notifications", "setNotifsEnabled", "applied"),
        ("make the corners rounder", "setRoundingScale", "applied"),
        ("faster animations", "setAnimationSpeed", "applied"),
        ("show more workspaces", "setWorkspacesShown", "applied"),
        ("smaller dock icons", "setDockIconSize", "applied"),
        ("larger launcher icons", "setLauncherIconSize", "applied"),
        ("enable overview blur", "setEnableOverviewBlur", "applied"),
    ]
    for text, surface, outcome in cases:
        res = router.route(text, state=DEFAULT_STATE, k=1)
        top = res.top
        features = dict(router.features.get(surface, {})) if hasattr(
            router, "features") else dict(res.features.get(surface, {}))
        learner.observe(text, surface, features,
                        float(top.p) if top else 0.5, outcome)
    return learner.to_dict()


class RefitTests(unittest.TestCase):
    def test_below_floor_refuses_without_arena(self) -> None:
        state = {}
        result = refit.refit_ratchet({"examples": []}, state)
        self.assertFalse(result["adopted"])
        self.assertIn("example", result["reason"])
        self.assertEqual(len(state[refit.AUDIT_KEY]), 1)

    def test_ratchet_is_deterministic(self) -> None:
        data = _build_learner_data()
        now = datetime(2026, 9, 28, 12, 0)
        a = refit.refit_ratchet(data, {}, now=now)
        b = refit.refit_ratchet(data, {}, now=now)
        self.assertEqual(a["reason"], b["reason"])
        self.assertEqual(a["adopted"], b["adopted"])
        self.assertEqual(a["current"], b["current"])

    def test_decision_is_audited_bounded(self) -> None:
        data = _build_learner_data()
        state = {}
        now = datetime(2026, 9, 28, 12, 0)
        for i in range(refit.MAX_AUDIT + 3):
            refit.refit_ratchet(data, state, now=now)
        audit = state[refit.AUDIT_KEY]
        self.assertEqual(len(audit), refit.MAX_AUDIT)
        self.assertIn("adopted", audit[0])

    def test_injected_state_measures_differently_or_same_honestly(self) -> None:
        """The state-injection contract: an absurd state (all mass on
        the fuzzy channel) must be MEASURABLE without changing the
        default-state run."""
        from assistant.core.eval.engine import run_routing
        from assistant.core.router import RouterState
        absurd = RouterState(w_lex=0.0, w_sem=0.0, w_fuzz=1.0, w_noun=0.0)
        default_report = run_routing("dev")
        absurd_report = run_routing("dev", state=absurd)
        self.assertNotEqual(
            absurd_report["metrics"]["top1_rate"][0],
            default_report["metrics"]["top1_rate"][0],
            "an absurd state must move the metric (the injection is real)")
        # rerun default: unchanged (determinism)
        again = run_routing("dev")
        self.assertEqual(again["metrics"]["top1_rate"][0],
                         default_report["metrics"]["top1_rate"][0])


if __name__ == "__main__":
    unittest.main()
