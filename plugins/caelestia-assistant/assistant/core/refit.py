"""cortex.refit — the weight re-fit ratchet (F28, exponential-build-5).

The self-learning loop (cortex.learn) updates the router's weights
ONLINE, one decision at a time. Online updates are honest per-decision
but can drift: a run of unlucky decisions nudges weights that then rank
slightly worse overall. This module closes that loop with a GUARDED
re-fit:

1. CANDIDATE: replay the learner's full example log through a FRESH
   online logistic (deterministic batch-equivalent of the online
   updates), map it onto RouterState exactly like
   ``CortexLearner.router_state`` does;
2. MEASURE: run the routing dev arena with the CURRENT weights and
   with the candidate (the arena is the same measurement the A0
   baseline and every ratchet floor come from — eval.engine.run_routing
   now accepts an injected state);
3. ADOPT ONLY IF NO REGRESSION: candidate top-1 point >= current top-1
   point AND candidate confident-wrong count <= current
   confident-wrong count. Otherwise the current weights stay and the
   refusal is recorded;
4. AUDIT: every decision (adopted or refused) lands on a bounded audit
   list in the caller's state — who re-fit, when, what moved, why the
   call went the way it did.

No secrets about the measurement: the dev arena is the TRAINING-
adjacent split (sealed stays sealed); the ratchet's claim is exactly
"no regression on the measured split", stated on every report.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

__all__ = ["AUDIT_KEY", "MAX_AUDIT", "MIN_EXAMPLES", "refit_ratchet",
           "render_lines"]

AUDIT_KEY = "cortex_refit_audit"
MAX_AUDIT = 12
MIN_EXAMPLES = 12  # the learner's own honesty floor for fitted weights


def _candidate_state(learner_data: Dict[str, Any]):
    """Replay the example log through a fresh logistic and map to
    RouterState (the CortexLearner.router_state mapping, applied to a
    from-scratch model)."""
    from .learn import CortexLearner

    fresh = CortexLearner({"model": {}, "examples": []})
    rows = learner_data.get("examples") or []
    for row in rows:
        fresh.model.update(dict(row.get("features") or {}),
                           int(row.get("label") or 0))
    fresh.model.examples = len(rows)
    if len(rows) < MIN_EXAMPLES:
        return None, (f"only {len(rows)} example(s); the refit needs "
                      f">= {MIN_EXAMPLES} to mean anything")
    return fresh.router_state(), None


def _replayed_model(learner_data: Dict[str, Any]) -> Dict[str, Any]:
    """The fresh logistic's serializable dict — what an ADOPTED refit
    persists into the learner's state (weights fitted by replaying the
    whole example log in order)."""
    from .learn import CortexLearner
    fresh = CortexLearner({"model": {}, "examples": []})
    for row in learner_data.get("examples") or []:
        fresh.model.update(dict(row.get("features") or {}),
                           int(row.get("label") or 0))
    fresh.model.examples = len(learner_data.get("examples") or [])
    return fresh.model.to_dict()


def refit_ratchet(learner_data: Dict[str, Any], state: Dict[str, Any],
                  now: Optional[datetime] = None) -> Dict[str, Any]:
    """Measure current vs candidate on the dev arena; adopt only on no
    regression. Mutates ONLY ``state`` (the audit list); the caller
    applies an adopted state to their live learner — the returned
    ``candidate_state`` dict is provided for that, nothing is written
    here. ``now`` anchors the audit stamp (deterministic for tests)."""
    now = now or datetime(2026, 1, 1, 12, 0, 0)
    from .learn import CortexLearner
    from .router import DEFAULT_STATE
    from assistant.core.eval.engine import run_routing

    current_state = CortexLearner(learner_data).router_state()
    current_state = current_state or DEFAULT_STATE
    candidate_state, why_not = _candidate_state(learner_data)

    current_report = run_routing("dev", state=current_state)
    current_top1 = current_report["metrics"]["top1_rate"][0]
    current_cw = current_report["confident_wrong"]["count"]

    record: Dict[str, Any] = {
        "at": now.isoformat(timespec="seconds"),
        "examples": len(learner_data.get("examples") or []),
        "current": {"top1": current_top1, "confident_wrong": current_cw},
    }
    if candidate_state is None:
        record.update({"adopted": False, "reason": why_not})
        _audit(state, record)
        return {"adopted": False, "reason": why_not,
                "current": record["current"], "candidate": None,
                "record": record}

    candidate_report = run_routing("dev", state=candidate_state)
    cand_top1 = candidate_report["metrics"]["top1_rate"][0]
    cand_cw = candidate_report["confident_wrong"]["count"]
    record["candidate"] = {"top1": cand_top1,
                           "confident_wrong": cand_cw}

    if cand_top1 >= current_top1 and cand_cw <= current_cw:
        fresh_model = _replayed_model(learner_data)
        record.update({"adopted": True,
                       "reason": f"no regression: top-1 {current_top1:.4f}"
                                 f"->{cand_top1:.4f}, confident-wrong "
                                 f"{current_cw}->{cand_cw}"})
        _audit(state, record)
        return {"adopted": True, "reason": record["reason"],
                "current": record["current"],
                "candidate": record["candidate"],
                "candidate_state": candidate_state,
                "candidate_model": fresh_model, "record": record}
    record.update({"adopted": False,
                   "reason": f"regression: top-1 {current_top1:.4f}->"
                             f"{cand_top1:.4f}, confident-wrong "
                             f"{current_cw}->{cand_cw}; current weights "
                             f"stay"})
    _audit(state, record)
    return {"adopted": False, "reason": record["reason"],
            "current": record["current"], "candidate": record["candidate"],
            "record": record}


def _audit(state: Dict[str, Any], record: Dict[str, Any]) -> None:
    audit = list(state.get(AUDIT_KEY) or [])
    audit.insert(0, record)
    state[AUDIT_KEY] = audit[:MAX_AUDIT]


def render_lines(result: Dict[str, Any]) -> List[str]:
    lines = ["refit ratchet (measured on the DEV split; sealed stays "
             "sealed):"]
    lines.append(f"  current: top-1 {result['current']['top1']:.4f}, "
                 f"confident-wrong "
                 f"{result['current']['confident_wrong']}")
    if result.get("candidate"):
        lines.append(f"  candidate: top-1 "
                     f"{result['candidate']['top1']:.4f}, confident-wrong "
                     f"{result['candidate']['confident_wrong']}")
    verdict = "ADOPTED" if result["adopted"] else "REFUSED"
    lines.append(f"  {verdict}: {result['reason']}")
    if result["adopted"]:
        lines.append("  apply the candidate by persisting candidate_state "
                     "into your learner (the verb does it when it adopts)")
    return lines
