"""brain.meta_bandit — a hierarchical meta-bandit over the recommendation
engines (exponential-build 3, item B3): PER CONTEXT, which engine's
recommendation should this user actually trust?

The three engines it was designed to sit over (all already shipped,
untouched by this module): (a) the Thompson-sampling strategy bandit
(cortex/learn.py's CortexLearner over brain/preset_bandit.py's
NamedBandit), (b) the LinUCB contextual bandit (brain/preset_bandit.py::
LinUCBBandit, Li, Chu, Langford & Schapire 2010, "A contextual-bandit
approach to personalized news article recommendation", WWW), and (c) the
AdaGrad online-logistic router (cortex/learn.py::OnlineLogistic, Duchi
et al. 2011). The meta level treats each ENGINE as an arm and
Thompson-samples which one to trust in a given context.

Thompson sampling: Chapelle & Li 2011, "An Empirical Evaluation of
Thompson Sampling", NeurIPS — the modern empirical treatment; the
sampling step itself is the same Beta-draw step every bandit in this
repo already uses (bandit.py / preset_bandit.py). Adapted here to a
hierarchy: the arms being sampled are not flat Beta(1,1) posteriors but
per-context posteriors SEEDED from a pooled population prior.

The hierarchy (Efron & Morris 1975, "Data analysis using Stein's
estimator and its simple generalizations", JASA — the partial-pooling
idea) is REUSED, not reimplemented: brain/pooling.py's pool_arms +
hierarchical_prior fit the population hyperparameters per engine from
ALL contexts' evidence for that engine (the classic Efron-Morris
construction: every unit shrinks toward the population fit from all
units), and the effective posterior for (context, engine) is that
hierarchical prior plus the context's own evidence. Consequences worth
stating: contexts that DISAGREE about an engine pool weakly (large
between-context variance -> small k0 -> own evidence rules, which is
what lets per-context specialization emerge); contexts that AGREE pool
fully (a cold context inherits the population's sharp opinion); a
context that is the ONLY one with evidence for an engine sets its own
prior mean (posterior mean unchanged, confidence sharpened — pinned by
test, not hidden); an engine nobody has evidence for anywhere stays
flat Beta(1,1), and the report says "no evidence: engines
indistinguishable".

Contexts are MATCHED, not enumerated: the caller supplies context
DESCRIPTORS (ordered), and an incoming context — an exact descriptor or
any raw text — is resolved to the descriptor whose hashed vector is
closest by cosine. The hashing is brain/features.py's signed feature
hashing (Weinberger et al. 2009), imported and reused: there is no
second hashing implementation anywhere in this module. Descriptor and
incoming vectors are hashed at d=64 — the top of features.py's
documented 8..64 caller range — because context MATCHING (unlike the
shared-state feature spaces, which stay at the default d=16) is a
nearest-vector lookup where hash collisions are pure noise: at d=16 a
genuine two-token overlap can lose to a coincidental same-bucket
same-sign collision with an unrelated descriptor.

Regret audit: the draw log [(context, engine, reward)] accumulates one
entry per OBSERVED round (update()); audit() feeds the engines-as-arms
draw pairs through brain/regret.py's audit_from_draws VERBATIM — the
regret math is not forked — and adds only the meta-level note that the
best-single-fixed-engine baseline is context-blind while the per-engine
means it rests on are context-contaminated when the meta-bandit
specializes engines to contexts (a specialization bonus shows up as an
optimistic best-fixed ESTIMATE — the underlying caveat, never a bound).

Library-first (the slot_tagger/reranker precedent): nothing in the live
routing path calls this module. The caller owns the state (plain dict,
to_dict/from_dict — persisted through whatever learned-state path the
caller already has), injects the rng, executes the chosen engine
itself, and reports the reward back. The engine callables supplied at
construction are RETAINED but never invoked here. Deterministic given
the rng (default: random.Random(0xC0E7E), cortex's own seed choice) and
fixed internal ordering (sorted context/engine iteration everywhere).
"""
from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .features import text_features
from .pooling import hierarchical_prior, pool_arms
from .regret import audit_from_draws

__all__ = ["MetaBandit"]

# Context descriptors + incoming contexts are hashed at d=64 (the top
# of features.py's documented 8..64 caller range) — see module docstring:
# matching is a nearest-vector lookup, and collisions are pure noise
# there (unlike the shared d=16 feature spaces the other learners use).
CONTEXT_DIMENSIONS = 64

_FLAT = (1.0, 1.0)


class MetaBandit:
    """Thompson-sampling meta level over caller-supplied engines.

    ``contexts``: ordered sequence of unique context descriptor strings.
    ``engines``: ordered mapping ``{engine_name: callable(context) ->
    recommendation}`` — retained for the caller's convenience, NEVER
    called by this class (choose() returns which engine to trust; the
    caller executes it and reports the reward via update()).
    """

    def __init__(self, contexts: Sequence[str], engines: Mapping[str, Callable]) -> None:
        try:
            self.engines: Dict[str, Callable] = dict(engines)
        except (TypeError, ValueError):
            raise ValueError("engines must be an ordered {name: callable} mapping")
        if not self.engines:
            raise ValueError("no engines to choose between")
        for name, fn in self.engines.items():
            if not callable(fn):
                raise ValueError(f"engine {name!r} is not callable")
        self._engine_order: List[str] = list(self.engines)
        self._engine_set = set(self._engine_order)

        keys = [str(c) for c in contexts]
        if not keys:
            raise ValueError("no contexts supplied")
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate context descriptors are refused")
        self.contexts: List[str] = keys
        self._context_set = set(keys)
        # each descriptor's signed-hash vector, computed ONCE with the
        # shared feature hashing (brain/features.py — the only hashing)
        self._context_vecs: Dict[str, List[float]] = {
            c: text_features(c, d=CONTEXT_DIMENSIONS) for c in self.contexts
        }

        # per-context per-engine Beta posteriors {context: {engine:
        # [alpha, beta]}} — only contexts/engines with evidence appear
        self.state: Dict[str, Dict[str, List[float]]] = {}
        # the draw log: [(resolved context, engine, reward)] — one entry
        # per OBSERVED round (update()); un-rewarded draws are
        # unobserved rounds and never enter the regret audit
        self.draws: List[Tuple[str, str, int]] = []
        self._rng = random.Random(0xC0E7E)  # seeded default, injectable per call

    # -- context resolution ------------------------------------------------

    def resolve_context(self, context: str) -> str:
        """Map an incoming context to one of the known descriptors.

        An exact descriptor passes through unchanged; any other text is
        hashed with the shared signed feature hashing and matched to
        the descriptor with the highest cosine (both sides are
        L2-normalized by hash_features, so the dot product IS the
        cosine). Ties resolve to the FIRST descriptor in the fixed
        supplied order; a token-less input (zero vector) ties
        everything and therefore also resolves to the first
        descriptor. Deterministic, pure."""
        key = str(context)
        if key in self._context_set:
            return key
        vec = text_features(key, d=CONTEXT_DIMENSIONS)
        best, best_dot = self.contexts[0], None
        for cand in self.contexts:
            dot = sum(a * b for a, b in zip(vec, self._context_vecs[cand]))
            if best_dot is None or dot > best_dot:
                best, best_dot = cand, dot
        return best

    # -- the hierarchy -----------------------------------------------------

    def _own_evidence(self, context_key: str, engine_name: str) -> Tuple[float, float]:
        row = self.state.get(context_key) or {}
        arm = row.get(engine_name) or list(_FLAT)
        # successes / failures beyond the flat prior (pooling.py's own
        # subtraction convention: the prior's mass is never evidence)
        return max(0.0, float(arm[0]) - 1.0), max(0.0, float(arm[1]) - 1.0)

    def _pooled_prior(self, engine_name: str) -> Dict[str, Any]:
        """The population prior for one engine: pool that engine's
        per-context posteriors across ALL contexts (flat defaults for
        contexts without evidence — pool_arms lists and excludes them
        itself) through brain/pooling.py, then take the hierarchical
        prior. No evidence anywhere -> the flat prior stands
        (hierarchical_prior says so itself)."""
        arms = {c: (self.state.get(c) or {}).get(engine_name) or list(_FLAT)
                for c in self.contexts}
        return hierarchical_prior(pool_arms(arms))

    def posterior(self, context: str, engine_name: str) -> Dict[str, Any]:
        """The EFFECTIVE posterior for (context, engine): the pooled
        hierarchical prior for that engine PLUS the context's own
        evidence. This is the distribution choose() samples; exposing
        it makes the hierarchy visible to the caller and the tests."""
        if engine_name not in self._engine_set:
            raise ValueError(f"unknown engine {engine_name!r} "
                             f"(have: {', '.join(self._engine_order)})")
        key = self.resolve_context(context)
        prior = self._pooled_prior(engine_name)
        s, f = self._own_evidence(key, engine_name)
        alpha, beta = prior["alpha"] + s, prior["beta"] + f
        return {
            "context": key,
            "engine": engine_name,
            "alpha": round(alpha, 6),
            "beta": round(beta, 6),
            "mean": round(alpha / (alpha + beta), 6),
            "own_evidence": round(s + f, 6),
            "pooled_prior": prior,
        }

    # -- the bandit interface ----------------------------------------------

    def choose(self, context: str, rng: Optional[random.Random] = None) -> str:
        """Thompson-sample WHICH ENGINE to trust in this context: one
        Beta draw per engine over the effective per-context posteriors,
        highest draw wins (ties: first engine in the fixed supplied
        order). The engines themselves are NOT called — the caller
        executes the returned engine and reports back via update()."""
        rng = rng if rng is not None else self._rng
        key = self.resolve_context(context)
        best_name, best_draw = None, None
        for name in self._engine_order:
            prior = self._pooled_prior(name)
            s, f = self._own_evidence(key, name)
            draw = rng.betavariate(prior["alpha"] + s, prior["beta"] + f)
            if best_draw is None or draw > best_draw:
                best_name, best_draw = name, draw
        return best_name

    def update(self, context: str, engine_name: str, accepted: bool) -> None:
        """Report one observed decision: the caller executed
        ``engine_name`` (ideally the one choose() returned — the caller
        owns that trust) and the recommendation was accepted or not.
        Updates the per-context posterior AND appends the round to the
        draw log the regret audit reads. Unknown engines are REJECTED."""
        if engine_name not in self._engine_set:
            raise ValueError(f"unknown engine {engine_name!r} "
                             f"(have: {', '.join(self._engine_order)})")
        key = self.resolve_context(context)
        arm = self.state.setdefault(key, {}).setdefault(
            engine_name, list(_FLAT))
        if accepted:
            arm[0] += 1.0
        else:
            arm[1] += 1.0
        self.draws.append((key, engine_name, 1 if accepted else 0))

    # -- the regret audit ----------------------------------------------------

    def audit(self) -> Dict[str, Any]:
        """The meta-bandit's draw log audited against the BEST SINGLE
        FIXED ENGINE in hindsight, through brain/regret.py's
        audit_from_draws VERBATIM (engines as arms — the shape fits
        exactly; the regret math is not forked). The per-engine observed
        means the estimate rests on are CONTEXT-CONTAMINATED when this
        bandit specializes engines to contexts (an engine only ever
        played where it wins shows an optimistic mean), so the audit's
        best-fixed baseline can be optimistic — that is the estimate
        caveat regret.py already carries, restated for the meta level;
        the figure is printed for the human, never acted on."""
        out = audit_from_draws([(engine, reward)
                                for (_ctx, engine, reward) in self.draws])
        per_context: Dict[str, int] = {}
        for ctx, _engine, _reward in self.draws:
            per_context[ctx] = per_context.get(ctx, 0) + 1
        out["engines"] = list(self._engine_order)
        out["per_context_rounds"] = dict(sorted(per_context.items()))
        out["meta_note"] = (
            "baseline = best single FIXED engine across ALL contexts "
            "(context-blind); per-engine observed means are context-"
            "contaminated when the meta level specializes engines to "
            "contexts — an estimate over unobserved rounds, printed "
            "for the human, nothing acts on it")
        return out

    # -- reporting -----------------------------------------------------------

    def report(self) -> Dict[str, Any]:
        """The read-only report: per-context effective posteriors for
        every engine (the hierarchy made visible), the audit, and the
        cold-start honesty — with no observed rounds at all, the status
        is exactly 'no evidence: engines indistinguishable'."""
        contexts_out = []
        for c in self.contexts:
            engines_out = {}
            for name in self._engine_order:
                row = self.posterior(c, name)
                engines_out[name] = {
                    "mean": row["mean"],
                    "own_evidence": row["own_evidence"],
                    "alpha": row["alpha"],
                    "beta": row["beta"],
                }
            contexts_out.append({"context": c, "engines": engines_out})
        if not self.draws:
            return {
                "status": "no evidence: engines indistinguishable",
                "draws": 0,
                "contexts": contexts_out,
                "note": ("no round has been observed yet — every posterior "
                         "above is the cold-start pooled prior (flat when "
                         "nothing is pooled anywhere); choose() samples "
                         "that same prior, i.e. it is guessing"),
            }
        return {
            "status": "learning",
            "draws": len(self.draws),
            "contexts": contexts_out,
            "regret": self.audit(),
        }

    # -- persistence (caller-owned state, plain dict) ------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contexts": list(self.contexts),
            "engines": list(self._engine_order),
            "state": {c: {e: [v[0], v[1]] for e, v in sorted(row.items())}
                      for c, row in sorted(self.state.items())},
            "draws": [[c, e, r] for c, e, r in self.draws],
        }

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]], contexts: Sequence[str],
                  engines: Mapping[str, Callable]) -> "MetaBandit":
        """Rebuild from a persisted dict under the SAME contexts and
        engines the state was built with. State keys (contexts, engine
        names) unknown to the supplied sets are REJECTED — a mismatched
        state cannot be interpreted, and the repo rule is refuse, never
        guess. A non-dict payload yields a fresh cold-start bandit."""
        mb = cls(contexts, engines)
        if not isinstance(data, dict):
            return mb
        state = data.get("state") or {}
        if isinstance(state, dict):
            for c, row in state.items():
                if c not in mb._context_set:
                    raise ValueError(f"persisted context {c!r} is not one of "
                                     "the supplied contexts")
                if not isinstance(row, dict):
                    continue
                clean: Dict[str, List[float]] = {}
                for e, arm in row.items():
                    if e not in mb._engine_set:
                        raise ValueError(f"persisted engine {e!r} is not one "
                                         "of the supplied engines")
                    clean[e] = [float(arm[0]), float(arm[1])]
                mb.state[str(c)] = clean
        draws = data.get("draws") or []
        for entry in draws:
            c, e, r = entry[0], entry[1], int(entry[2])
            if c not in mb._context_set:
                raise ValueError(f"draw-log context {c!r} is not one of the "
                                 "supplied contexts")
            if e not in mb._engine_set:
                raise ValueError(f"draw-log engine {e!r} is not one of the "
                                 "supplied engines")
            mb.draws.append((str(c), str(e), r))
        return mb
