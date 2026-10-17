"""generative.reranker — the optional sub-1B candidate reorder (E2,
Group E).

The deterministic router already ranks candidates; a sub-1B model will
not out-rank it on a good day. The ONE place a tiny model can help is
the router's own low-confidence margin, where the difference between
candidate #1 and #2 is noise the router itself flags. This module
re-orders the ROUTER'S OWN candidate list with one loopback generate()
call — and is built so the model can do exactly nothing else:

- the model never sees the registry, only the candidate list the
  router already produced, so it cannot invent a tool;
- the output is parsed under a constrained format: a line that is
  EXACTLY one candidate name (first exact match wins). Anything else
  the model emits — code fences, commentary, hallucinated names — is
  dropped and counted, never parsed loosely;
- the result is a REORDER of candidates, marked MODEL_SUGGESTED on
  every surfaced element, and the caller still routes any resulting
  change through the ordinary plan/apply gates (invariant 1: the only
  write path in the product). A model suggestion here can change the
  ORDER of an ask, never produce an answer or an apply;
- OFF by default twice: the layer's own --generative gate AND the
  model_reranker capability (a file edit — a request cannot flip it);
- one attempt, loopback-only, through the EXISTING client transport
  (client.generate); an unavailable server degrades to UNCHANGED —
  the deterministic order always stands.

Verified candidates (Ollama library, fetched live 2026-09-30; sizes are
the library's own download numbers):

  - smollm2:135m  (271 MB)  Apache 2.0   HuggingFaceTB SmolLM2
  - smollm2:360m  (726 MB)  Apache 2.0   HuggingFaceTB SmolLM2
  - qwen2.5:0.5b  (398 MB)  Apache 2.0*  *per the library page: all
                            Qwen2.5 models except the 3B and 72B are
                            Apache 2.0

All three are IN the Ollama model library (unlike the apertus family
the old the model guidance (see the generative package docs) recommended, which had no listing at all). Serving
one is a user-side step; the assistant downloads nothing, ever.

Determinism, stated plainly: model output is NOT deterministic, so this
module must never sit on a path where its output is trusted without the
deterministic layers' validation. The tests therefore run it against a
fake connection factory (the established pattern from test_assisted.py)
— no test touches a network, and no production path trusts the model
without a registry-validated plan behind it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import client

__all__ = ["rerank", "render_lines", "VERIFIED_CANDIDATES"]

# the only role this project recommends a model for, and the only
# candidates named in the model guidance (see the generative package docs) — each verified against the Ollama
# library before being written down (see module docstring)
VERIFIED_CANDIDATES = [
    {"name": "smollm2:135m", "download": "271 MB", "license": "Apache 2.0"},
    {"name": "smollm2:360m", "download": "726 MB", "license": "Apache 2.0"},
    {"name": "qwen2.5:0.5b", "download": "398 MB", "license": "Apache 2.0"},
]

_PROMPT_TEMPLATE = """You are a ranker. Pick the single best candidate for the request.
Answer with EXACTLY one candidate name from the list and nothing else.

Request: {query}

Candidates:
{candidates}

Answer with one candidate name only:"""


def _build_prompt(query: str, candidates: List[Dict[str, Any]]) -> str:
    lines = []
    for c in candidates:
        desc = str(c.get("description") or "").strip()
        lines.append(f"- {c['name']}" + (f" — {desc}" if desc else ""))
    return _PROMPT_TEMPLATE.format(query=query, candidates="\n".join(lines))


def _parse_choice(raw: str, allowed_names: List[str]) -> Optional[str]:
    """The constrained parse: a line that IS a candidate name. First
    exact match wins; near-misses and hallucinations are not guesses
    waiting to happen, they are drops."""
    allowed = set(allowed_names)
    for line in (raw or "").splitlines():
        token = line.strip().strip("`*\"' ")
        if token in allowed:
            return token
    return None


def rerank(query: str,
           candidates: List[Dict[str, Any]],
           model: Optional[str] = None,
           url: Optional[str] = None,
           conn_factory: Optional[Any] = None,
           timeout: float = client.DEFAULT_TIMEOUT_S,
           capability_enabled: Optional[bool] = None) -> Dict[str, Any]:
    """Re-order the router's candidates with one loopback model call.

    candidates: [{"name", "description"}] — the router's own shortlist,
    registry names, in the router's own order. Returns a JSON-shaped
    report:
      verdict MODEL_SUGGESTED  the model's choice moved to the front
                               (order = [choice] + the rest, stable)
      verdict UNCHANGED        no usable model answer, or the gate is
                               off, or the server is unavailable — the
                               caller's order stands untouched
    The deterministic candidate list is never mutated in place; the
    reordered list is a copy.
    """
    base = {"query": query,
            "model": client.resolve_model(model),
            "n_candidates": len(candidates)}
    names = [str(c.get("name") or "") for c in candidates]
    if not names or not query.strip():
        return {**base, "verdict": "UNCHANGED",
                "reason": "nothing to rerank"}

    # the capability gate (defense in depth: the caller gates too, and
    # the flag lives in the file, not in any request)
    if capability_enabled is None:
        from assistant.core.features import enabled
        capability_enabled = enabled("model_reranker")
    if not capability_enabled:
        return {**base, "verdict": "UNCHANGED",
                "reason": "model_reranker capability is off on this "
                          "install"}

    prompt = _build_prompt(query, candidates)
    try:
        raw = client.generate(prompt, model=model, url=url,
                              conn_factory=conn_factory, timeout=timeout)
    except Exception:  # noqa: BLE001 — degraded service is the contract:
        return {**base, "verdict": "UNCHANGED",     # no retry, no raise
                "reason": "loopback model unavailable"}

    choice = _parse_choice(raw, names)
    if choice is None:
        return {**base, "verdict": "UNCHANGED",
                "reason": "model named no candidate",
                "dropped": len((raw or "").splitlines())}
    if choice == names[0]:
        return {**base, "verdict": "UNCHANGED",
                "reason": "model agreed with the router's first choice"}
    reordered = [c for c in candidates if str(c.get("name")) == choice]
    reordered += [c for c in candidates if str(c.get("name")) != choice]
    return {**base, "verdict": "MODEL_SUGGESTED",
            "suggested": choice,
            "order": [str(c.get("name")) for c in reordered],
            "candidates": reordered}


def render_lines(r: Dict[str, Any]) -> List[str]:
    """The human card. MODEL_SUGGESTED is the loudest thing on it —
    a model touched this output, and the reader must know."""
    head = f"[{r['verdict']}] rerank over {r['n_candidates']} candidates " \
           f"(model: {r['model']})"
    lines = [head]
    if r["verdict"] == "MODEL_SUGGESTED":
        lines.append(f"  model suggests: {r['suggested']}")
        lines.append("  order after suggestion: " +
                     " > ".join(r["order"]))
        lines.append("  every element above is MODEL_SUGGESTED; the "
                     "router's own ranking is next, and nothing here "
                     "bypasses the plan/apply gates")
    else:
        lines.append(f"  deterministic order stands ({r.get('reason', '')})")
    return lines
