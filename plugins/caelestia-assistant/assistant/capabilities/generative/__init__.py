"""Layer 3 (OPTIONAL, OFF by default) of the caelestia assistant: generative
suggestions over a LOCAL Ollama server, strictly retrieval-grounded.

Guarantees shared by every module in this package:

- OFF by default: activates only when the caller passes enable=True / the CLI
  flag --generative AND a loopback Ollama server answers. Absent either, the
  result states that the generative layer is disabled/unavailable and carries
  the unchanged Layer 2 retrieval hits. Single attempt, ~10s timeout: never a
  crash, never a retry storm.
- Loopback only: the only network surface in the whole assistant is one
  http.client POST to 127.0.0.1:11434 (port configurable via
  CAELESTIA_ASSISTANT_OLLAMA_URL); generative.client hard-rejects any host
  that is not localhost / 127.0.0.1 / [::1] before connecting.
- Grounded only: prompts are built exclusively from Layer 2 retrieval hits
  plus the user's problem text; the system preamble forbids invented commands
  and execution claims. Zero retrieval hits => the model is not contacted.
- Suggestion-only: output is post-processed (SUGGESTED_NOT_EXECUTED labels +
  risk tiers; DESTRUCTIVE suggestions withheld). Nothing is ever executed,
  auto-run, trained on, or written to disk; no subprocess/socket imports.
- Model guidance: which model to serve locally — see the model guidance (see the generative package docs).
"""

from .rag import build_rag_prompt, sanitize_suggestion, suggest

__all__ = ["build_rag_prompt", "sanitize_suggestion", "suggest"]
