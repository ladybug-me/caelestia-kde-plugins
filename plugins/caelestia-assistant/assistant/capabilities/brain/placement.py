"""Suggest which existing target (a config key, a preset, a tag, a project)
a free-text preference most resembles — without ever writing anything.

This is the ISS-120-safe slice of "natural-language settings": issue #120
proposes letting the assistant turn "make my bar thinner" into a config
write, and the maintainer's own condition on that proposal is that the
assistant must never touch config files directly, only request changes
through a controlled API. This module does the retrieval half only: it
ranks a caller-supplied registry of known targets by text similarity and
hands back candidates for a human — or a separate, explicit --apply layer
that is not this module — to act on.
"""
from .nlp import tokens


def _score(query_tokens, target_tokens):
    q, t = set(query_tokens), set(target_tokens)
    if not q or not t:
        return 0.0
    return len(q & t) / len(q | t)


def suggest(text, registry, top=5):
    """registry: [{"id", "description", "tags": [...] (optional)}, ...].

    Returns [{"id", "score", "description"}] ranked by token-Jaccard between
    the query text and each target's description + tags.
    """
    q = tokens(text)
    scored = []
    for target in registry:
        t_text = target.get("description", "") + " " + " ".join(target.get("tags", []))
        s = _score(q, tokens(t_text))
        if s > 0:
            scored.append({"id": target["id"], "score": round(s, 3),
                          "description": target.get("description", "")})
    return sorted(scored, key=lambda r: -r["score"])[:top]
