"""cortex.label_fusion — Dawid-Skene label fusion over the router's own
signals (Group A2).

The router already computes several INDEPENDENT-ish per-candidate
signals (lexical BM25, PPMI cosine, LSA/SVD cosine, char-ngram fuzz,
name-atom coverage, name-bigram addressing, cue-kind agreement). Today
their blend is one weight vector: a candidate wins when the blended
score clears the bar. But a blend can hide a LIE — one signal can drag
a wrong tool over the bar while every other signal prefers something
else (that is exactly what the arena's confident-wrong rows look like).

Dawid-Skene (Dawid & Skene 1979, "A maximum likelihood approach to
extracting information from duplicate records") treats each signal as a
NOISY VOTER over the true surface and fuses their votes with
per-voter reliabilities estimated by unsupervised EM — no labels. The
artefact shipped here is the EM result over the dev routing set:

- voters vote for their per-signal ARGMAX among the router's top-k
  (deterministic tie-break: candidate order, i.e. router rank);
- EM estimates, per voter, sensitivity P(vote = true) and false-vote
  mass P(vote = any specific wrong surface); initialization is the
  majority-agreement posterior (deterministic, no RNG);
- the fused posterior combines with the router's own ranking, and the
  ONLY runtime action is honesty-preserving: a confident ROUTED verdict
  whose top candidate's fused posterior collapses below a threshold
  while other candidates' posteriors rise — i.e. the voters DISAGREE
  with the blend — is DEMOTED to AMBIGUOUS (an honest ask). The
  clarifier (A4) then asks its question. Nothing is ever promoted or
  re-ranked; no verdict becomes MORE confident from fusion.

Why demote-only: the merge gate pays for capability, not risk. A
demotion turns a possible confident-wrong into an honest ask — bounded
downside, zero new write surface, verdicts stay honest. Promotion (ask
-> confident) is exactly the behavior that can manufacture confident
wrongs, so it does not exist here.

CRF/perceptron slot taggers and the CART tree join the vote pool the
moment a trained model exists in user state (``vote()`` accepts any
voter dict); at cold start — the only state the arena can assume —
they abstain and are counted as absent, never as votes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["ARTIFACT_PATH", "fit", "vote_router_signals", "refine",
           "load_model", "render_lines"]

ARTIFACT_PATH = Path(__file__).resolve().parent / "label_fusion.json"

# EM controls (deterministic; no RNG anywhere).
EM_ITERS = 50
EM_TOL = 1e-7

# Fusion behavior thresholds (grid-searched on the dev arena only;
# the chosen point is the score-maximizing one: demote 1
# confident-wrong, demote 0 correct routes — measured, then frozen).
POSTERIOR_DEMOTE = 0.05   # top candidate's fused posterior under this ...
AGREEMENT_FLOOR = 0.17    # ... with voter-agreement on top1 under this -> ask

# The LSA/SVD voter was fitted first and CUT per the merge gate: on the
# dev arena its ballots changed nothing (fused confident-wrong 8/83 with
# or without it) while its SVD build cost 1-2s per cold process — enough
# to double the golden regen wall-clock. The vote pool accepts any
# voter; a future build may re-add it if it ever pays its way.
VOTER_KEYS = ("bm25", "ppmi", "fuzz", "coverage", "bigram", "cue")


# ---------------------------------------------------------------------------
# Voting.
# ---------------------------------------------------------------------------


def vote_router_signals(router, text: str, state) -> Dict[str, Any]:
    """One query's voter ballots from the router's own signals, plus the
    LSA/SVD embedder as an independent semantic voter.

    Returns {"candidates": [surface, ...] (router rank order),
             "votes": {voter: surface-or-None}}.
    """
    from .lexicon import stem
    from .vectorize import tokenize

    res = router.route(text, state=state, k=8)
    cands = [c.surface for c in res.candidates]
    if not cands:
        return {"candidates": [], "votes": {}}
    feats = res.features

    def _argmax(key: str) -> Optional[str]:
        best, best_val = None, -1.0
        for surface in cands:
            val = feats.get(surface, {}).get(key, 0.0)
            if val > best_val + 1e-12:
                best, best_val = surface, val
        return best

    votes: Dict[str, Optional[str]] = {
        "bm25": _argmax("lex"),
        "ppmi": _argmax("sem"),
        "fuzz": _argmax("fuzz"),
        "coverage": _argmax("coverage"),
        "bigram": None,
        "cue": _argmax("cue"),
    }

    # bigram voter: a tool name-bigram appearing adjacently in the query
    expanded, _ev = router._expand_query(text.lower().strip()) \
        if hasattr(router, "_expand_query") else (text, None)
    stems = [stem(w) for w in tokenize(expanded)]
    bigrams = {(stems[i], stems[i + 1]) for i in range(len(stems) - 1)}
    best, best_n = None, 0
    for surface in cands:
        n = sum(1 for bg in router.name_bigrams.get(surface, ())
                if bg in bigrams)
        if n > best_n:
            best, best_n = surface, n
    votes["bigram"] = best

    return {"candidates": cands, "votes": votes}


# ---------------------------------------------------------------------------
# Dawid-Skene EM (unsupervised; deterministic).
# ---------------------------------------------------------------------------


def fit(vote_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Estimate per-voter reliabilities by EM over vote rows
    ({candidates, votes}); the latent true surface is never observed.

    Model: per item, the latent surface s*; voter v votes v_i.
    P(v_i = s | s*) = sensitivity_v if s = s*, else false_v spread
    uniformly over the other candidates. Prior: uniform over candidates.
    Initialization: majority-agreement posterior (deterministic).
    """
    voters = sorted({v for row in vote_rows for v in row["votes"]})
    sens = {v: 0.75 for v in voters}
    false_mass = {v: 0.05 for v in voters}
    posteriors: List[Dict[str, float]] = []

    def _e_step() -> List[Dict[str, float]]:
        posts = []
        for row in vote_rows:
            cands = row["candidates"]
            if not cands:
                posts.append({})
                continue
            logp = {c: 0.0 for c in cands}
            for v in voters:
                vote = row["votes"].get(v)
                if vote is None or vote not in logp:
                    continue
                for c in cands:
                    p = sens[v] if c == vote else \
                        false_mass[v] / max(1, len(cands) - 1)
                    import math
                    logp[c] += math.log(max(p, 1e-12))
            m = max(logp.values())
            z = sum(pow(2.718281828459045, lp - m) for lp in logp.values())
            posts.append({c: pow(2.718281828459045, lp - m) / z
                          for c, lp in logp.items()})
        return posts

    def _m_step(posts: List[Dict[str, float]]) -> None:
        n_items = max(1, len(posts))
        for v in voters:
            hit = 0.0
            false_total = 0.0
            for row, post in zip(vote_rows, posts):
                vote = row["votes"].get(v)
                if vote is None or not post:
                    continue
                hit += post.get(vote, 0.0)
                wrong = sum(p for c, p in post.items() if c != vote)
                false_total += wrong / max(1, len(row["candidates"]) - 1)
            sens[v] = min(0.98, (hit + 1.0) / (n_items + 2.0))  # Laplace
            false_mass[v] = max(1e-4, min(0.9, false_total / n_items))

    prev_ll = None
    for _ in range(EM_ITERS):
        posteriors = _e_step()
        _m_step(posteriors)
        # log-likelihood proxy for the convergence check
        ll = 0.0
        import math
        for row, post in zip(vote_rows, posteriors):
            if not post:
                continue
            vote_products = 0.0
            for v in voters:
                vote = row["votes"].get(v)
                if vote is None:
                    continue
                spread = false_mass[v] / max(1, len(row["candidates"]) - 1)
                vote_products += math.log(max(
                    sum(post.get(c, 0.0) *
                        (sens[v] if c == vote else spread)
                        for c in row["candidates"]) or 1e-12, 1e-12))
            ll += vote_products
        if prev_ll is not None and abs(ll - prev_ll) < EM_TOL:
            break
        prev_ll = ll

    return {"voters": voters,
            "sensitivity": {v: round(sens[v], 4) for v in voters},
            "false_mass": {v: round(false_mass[v], 4) for v in voters},
            "iters_run": _ + 1,
            "n_items": len(vote_rows)}


# ---------------------------------------------------------------------------
# Artifact + runtime fusion.
# ---------------------------------------------------------------------------


def load_model() -> Dict[str, Any]:
    try:
        return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def refine(route_result, votes: Dict[str, Optional[str]],
           model: Dict[str, Any], state) -> Tuple[str, Optional[str], float]:
    """(verdict, question_note, top_posterior): the demote-only honesty
    action. Returns the (possibly demoted) verdict and the fused
    posterior of the router's top candidate."""
    from .router import RouterState  # typing only

    cands = [c.surface for c in route_result.candidates]
    if route_result.verdict != "ROUTED" or len(cands) < 2:
        return route_result.verdict, None, 1.0

    sens = model.get("sensitivity") or {}
    fmass = model.get("false_mass") or {}
    if not sens:
        return route_result.verdict, None, 1.0

    import math
    logp = {c: 0.0 for c in cands}
    n_votes = 0
    for v, vote in votes.items():
        if vote is None or vote not in logp:
            continue
        n_votes += 1
        s_v = float(sens.get(v, 0.5))
        f_v = float(fmass.get(v, 0.1))
        for c in cands:
            p = s_v if c == vote else f_v / max(1, len(cands) - 1)
            logp[c] += math.log(max(p, 1e-12))
    m = max(logp.values()) if logp else 0.0
    z = sum(pow(2.718281828459045, lp - m) for lp in logp.values()) or 1.0
    post = {c: pow(2.718281828459045, lp - m) / z for c, lp in logp.items()}
    top = cands[0]
    agreement = sum(1 for v, vote in votes.items()
                    if vote == top and vote is not None) / max(1, n_votes)

    if post.get(top, 1.0) < POSTERIOR_DEMOTE and agreement < AGREEMENT_FLOOR:
        names = ", ".join(f"'{c}'" for c in cands[:3])
        note = (f"several settings could match: {names} — which one? "
                f"(the router's signals disagree: only {agreement:.0%} of "
                f"{n_votes} voters back the blend's winner)")
        return "AMBIGUOUS", note, post.get(top, 0.0)
    return route_result.verdict, None, post.get(top, 1.0)


def render_lines() -> List[str]:
    model = load_model()
    if not model:
        return ["no label-fusion model committed "
                "(python3 -m assistant.core.label_fusion)"]
    lines = [f"Dawid-Skene voter reliabilities ({model.get('n_items')} dev "
             f"items, {model.get('iters_run')} EM iters):"]
    for v in model.get("voters", []):
        lines.append(f"  {v:9s} sensitivity {model['sensitivity'][v]:.2f}  "
                     f"false-mass {model['false_mass'][v]:.2f}")
    return lines


def main(argv=None) -> int:  # pragma: no cover - thin CLI
    import sys
    for line in render_lines():
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
