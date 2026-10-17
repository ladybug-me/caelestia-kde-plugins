"""cortex.coldstart — DP-noised community priors for a fresh install
(exponential-build-4 G).

The gap: a new install's bandits start flat (Beta(1,1) everywhere) and
learn the user's tool-usage shape from zero. The community fix: users
who already have history EXPORT a noised summary of their bandit
evidence; a fresh install IMPORTS it as its starting priors instead of
flat ones.

THE PRIVACY MECHANISM IS NOT NEW — deliberately. cortex/dp.py's
Laplace mechanism (Dwork, McSherry, Nissim & Smith 2006) was built for
the lexicon-diff export; this module REUSES it (dp.laplace_noise, the
same inverse-CDF sampling, the same sensitivity accounting, the same
epsilon semantics) pointed at bandit evidence instead of phrase
counts. Per-arm sensitivities: alpha and beta are per-user COUNTS in
the neighboring relation "this user's history in / out", so
N_SENSITIVITY = 1.0 each — the same accounting dp.py already
documents. Composition across arms: the export noises each arm
independently at the caller's epsilon (the artifact records the
epsilon it was made at; importing does not re-noise).

What is exported: the Beta POSTERIOR parameters (alpha, beta) per
strategy/tool arm — aggregates, never rows. Post-processing is the
standard DP kind: noised parameters are CLIPPED to the flat-prior
floor (1.0) on import, and the clip is recorded in the import report;
noising first then clipping keeps the DP guarantee (post-processing
of DP output is DP).

Honesty: the imported priors are EVIDENCE WITH PROVENANCE, not
truth — the import report carries the artifact's epsilon, the marker
string, and per-arm clip status; a user can inspect the artifact
(canonical JSON) before importing; and nothing forces the import
(flat priors remain the default). State stays in the caller's
learned-state JSON; nothing here writes.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List

from .dp import N_SENSITIVITY, laplace_noise

__all__ = ["SCHEMA", "STATE_KEY", "DEFAULT_EPSILON", "export_priors",
           "import_priors", "bootstrap_learner"]

SCHEMA = "caelestia-coldstart/1"
STATE_KEY = "community_priors"
DEFAULT_EPSILON = 1.0
_FLAT_FLOOR = 1.0
#: the prior CEILING per arm (pseudo-observations). An imported prior is
#: EVIDENCE, not a verdict: past this mass the "prior" would dominate
#: any amount of the user's own lived evidence, which is the hostile-
#: artifact failure mode the build-4 adversarial sweep caught (an
#: unbounded beta could permanently silence an arm). Clipped down and
#: reported — standard DP post-processing either way.
_MAX_PRIOR_MASS = 1000.0


def export_priors(arms: Dict[str, List[float]], epsilon: float,
                  rng: random.Random) -> Dict[str, Any]:
    """One learner's Beta arms -> the DP-noised community artifact.

    ``arms``: {name: [alpha, beta]} exactly as NamedBandit persists
    them. Each parameter is noised independently at
    Laplace(0, N_SENSITIVITY / epsilon). Negative or sub-floor noised
    values are LEFT IN THE ARTIFACT (the importer clips — clipping is
    post-processing either way, and the artifact should show the true
    noised draw rather than a pre-chewed one)."""
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0 (a noise scale is not "
                         "definable from a non-positive epsilon)")
    noised: Dict[str, Dict[str, float]] = {}
    for name in sorted(arms):
        alpha, beta = float(arms[name][0]), float(arms[name][1])
        noised[name] = {
            "alpha": round(laplace_noise(alpha, N_SENSITIVITY, epsilon, rng), 4),
            "beta": round(laplace_noise(beta, N_SENSITIVITY, epsilon, rng), 4),
        }
    return {
        "schema": SCHEMA,
        "epsilon": epsilon,
        "sensitivity": N_SENSITIVITY,
        "mechanism": "Laplace (cortex/dp.py), per-arm independent "
                     "draws; aggregates only, never rows",
        "arms": noised,
    }


def import_priors(artifact: Dict[str, Any],
                  target_arms: Dict[str, List[float]]
                  ) -> Dict[str, Any]:
    """Clip-and-merge one artifact into a learner's arm state.

    ``target_arms`` is the learner's CURRENT {name: [alpha, beta]}
    dict, mutated in place (the caller persists through the usual
    state path). Arms not present in the artifact keep their current
    posterior (an import ADDS evidence, it never deletes it).
    Returns the per-arm report: prior, noised, floored."""
    if not isinstance(artifact, dict) or artifact.get("schema") != SCHEMA:
        raise ValueError(
            f"artifact schema must be {SCHEMA!r} — refusing an "
            "unversioned or foreign artifact")
    epsilon = artifact.get("epsilon")
    if not isinstance(epsilon, (int, float)) or epsilon <= 0:
        raise ValueError("artifact epsilon must be a positive number")
    report: List[Dict[str, Any]] = []
    for name, pair in sorted((artifact.get("arms") or {}).items()):
        try:
            alpha = float(pair["alpha"])
            beta = float(pair["beta"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"arm {name!r} lacks numeric alpha/beta: {exc}") from exc
        a0, b0 = target_arms.get(name, [_FLAT_FLOOR, _FLAT_FLOOR])
        clipped = False
        capped = False
        if alpha < _FLAT_FLOOR:
            alpha, clipped = _FLAT_FLOOR, True
        if beta < _FLAT_FLOOR:
            beta, clipped = _FLAT_FLOOR, True
        if alpha > _MAX_PRIOR_MASS:
            alpha, capped = _MAX_PRIOR_MASS, True
        if beta > _MAX_PRIOR_MASS:
            beta, capped = _MAX_PRIOR_MASS, True
        target_arms[name] = [round(alpha, 4), round(beta, 4)]
        report.append({"arm": name,
                       "prior": [round(float(a0), 4), round(float(b0), 4)],
                       "imported": [round(alpha, 4), round(beta, 4)],
                       "clipped_to_flat_floor": clipped,
                       "capped_at_prior_ceiling": capped})
    return {"epsilon": epsilon, "arms": report,
            "note": "evidence with provenance: the artifact's epsilon "
                    "and clipping are recorded; flat priors remain the "
                    "default and nothing forces an import"}


def bootstrap_learner(learner: Any, artifact: Dict[str, Any]) -> Dict[str, Any]:
    """Wire the import into the learner's strategy bandit: the noised
    community evidence becomes the bandit's starting Beta state (the
    cold-start the module exists for). The learner's other state
    (model, calibration, examples) is untouched."""
    report = import_priors(artifact, learner.bandit.arms)
    return report
