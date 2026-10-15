"""caelestia assistant — single entry point routing to every module.

  caelestia-assist diagnose FILE | selfcheck        troubleshooting rules (Layer 1)
  caelestia-assist ask "text" [--generative]        rules, then retrieval, then optional LLM
  caelestia-assist search "query" [-k N]            offline retrieval over repo docs
  caelestia-assist scan FILE [--pattern P]          one-pass bounded-memory log scanning
  caelestia-assist issue draft|list-similar ...     issue-report drafting (never submits)
  caelestia-assist settings "request" [--apply]     natural-language shell.json editing
  caelestia-assist brain <command> ...              lean intelligence layer (proposals)
  caelestia-assist brief                            the daily brief (ledger+plans+forecast)
  caelestia-assist chat | route "text"              learned cortex routing (confirmation-gated)
  caelestia-assist cortex report|recall|...         learning/memory dashboard
  caelestia-assist inbox list|approve|reject        unified pending-decisions inbox (four sources)
  caelestia-assist why [<id>|--engine ...]          one explainer over every engine's own output
  caelestia-assist do "anything"                    genius: universal intelligence layer
  caelestia-assist genius <command> ...             direct domain access (math/stats/...)
  caelestia-assist agent "goal" [--simulate]        agentic orchestrator (consent-gated)
  caelestia-assist shellkb <verb> ...              shell employee: CLI grammar, explain, how-to
  caelestia-assist api < request.json               JSON bridge for the brain (QML/IPC)
  caelestia-assist eval [suite] [--json] [--sealed] the measurement arena (dev/sealed sets)

Verbless front door: a first token that matches no
verb above is no longer a hard error. A token within edit distance 2 of
exactly one verb gets a "did you mean" prompt (the settings layer's own
correction style — never a silent guess); anything else is treated as
free text and routed through the cortex pipeline one-shot, so

  caelestia-assist "make my bar thinner"

works end to end: an answer, a pending plan, or a simulated agent plan.

Every module keeps its own safety rules; this file only routes.
"""
import sys
from importlib import import_module
from typing import List, Optional, Tuple

USAGE = __doc__

# R4 (exponential-build-5): every module CLI is imported LAZILY at
# dispatch time. The previous eager imports put the agent (135ms of
# import time) and the genius stack on the cold path of EVERY verb —
# `--help` alone pulled the whole engine tree. Values are either
# "module:func" specs (resolved on first use) or direct callables
# (tests inject these); the second element is keep_name (the module
# parses its own subcommand token).
ROUTES = {
    "diagnose": ("assistant.capabilities.diagnostics.cli:main", True),
    "selfcheck": ("assistant.capabilities.diagnostics.cli:main", True),
    "doctor": ("assistant.doctor:main", True),
    # rulepack: the signed rule-pack surface (export/import/list/
    # render — exponential-build-3 F1; same parser as diagnose)
    "rulepack": ("assistant.capabilities.diagnostics.cli:main", True),
    "search": ("assistant.capabilities.retrieval.cli:main", True),
    "scan": ("assistant.capabilities.scan.cli:main", False),
    "ask": ("assistant.core.ask_pipeline:main", False),
    "issue": ("assistant.capabilities.issues.cli:main", False),
    "settings": ("assistant.capabilities.settings.cli:main", False),
    "brain": ("assistant.capabilities.brain.cli:main", False),
    # brief/tidy are brain subcommands surfaced at the top level too
    "brief": (lambda _argv=None: import_module(
        "assistant.capabilities.brain.cli").main(["brief"]), False),
    "tidy": ("assistant.capabilities.brain.cli:main", False),
    "api": ("assistant.capabilities.brain.bridge:main", False),
    # cortex: chat/route keep their own subcommand token (argparse owns it)
    "chat": ("assistant.core.cli:main", True),
    "route": ("assistant.core.cli:main", True),
    "cortex": ("assistant.core.cli:main", False),
    # the unified pending-decisions inbox: one ranked view over the
    # ledger, gap clusters, the pending plan and agent consents
    "inbox": ("assistant.core.inbox:main", False),
    # one `why` over every engine's own explanation output (walks back
    # through whichever engine produced the last surfaced item)
    "why": ("assistant.core.explain_unified:main", False),
    # genius: the universal intelligence layer (its own subcommands;
    # `do` is the one-word front door)
    "do": ("assistant.capabilities.genius.cli:main", False),
    "genius": ("assistant.capabilities.genius.cli:main", False),
    # agent: the consent-gated orchestrator over every layer
    "agent": ("assistant.capabilities.agent.cli:main", False),
    # shellkb: the shell employee (read-only knowledge of the shell's
    # own CLI/configs/deps; everything it prints is SUGGESTED_NOT_EXECUTED)
    "shellkb": ("assistant.capabilities.shellkb.cli:main", False),
    # capabilities: the per-install manifest card (read-only listing)
    "capabilities": (lambda _argv=None: (_print_card(), 0)[1], False),
    # eval: the measurement arena (exponential-build-5 F1). Lazily:
    # the arena pulls the full router/diagnostics stack and must never
    # sit on the cold-start path of other verbs.
    "eval": ("assistant.core.eval.cli:main", False),
    # graph: the shell knowledge graph (exponential-build-5 F12) —
    # rebuildable from tools.json + the curated consequences table +
    # the diagnostic rules; read-only queries with provenance. Lazy for
    # the same reason as eval: it pulls the settings + genius stacks.
    "graph": ("assistant.capabilities.graph.cli:main", True),
}


def _resolve(verb: str):
    """(callable, keep_name) for a verb, importing its CLI lazily.
    Direct callables (tests, brief/capabilities) pass through."""
    entry = ROUTES.get(verb)
    if entry is None:
        return None, None
    target, keep = entry
    if callable(target):
        return target, keep
    module_name, func_name = target.split(":", 1)
    return getattr(import_module(module_name), func_name), keep


def _print_card() -> None:
    from .core import features as capabilities_mod  # lazy (R4)
    print(capabilities_mod.render())


# Verb suggestion bounds: the same discipline the rest of the assistant
# already applies. Distance 2 is the settings CLI's forgiveness bound
# (``settings --tool setBarPositin`` -> "did you mean setBarPosition
# (distance 1)"); tokens shorter than 5 characters are never suggested
# against (the router's own min-length guard — at distance 2 short garbage
# starts matching real verbs, and a wrong suggestion is worse than none).
_SUGGEST_MAX_DISTANCE = 2
_SUGGEST_MIN_LEN = 5


def suggest_verb(cmd: str) -> Optional[Tuple[str, int]]:
    """The unique ROUTES verb within edit distance 2 of ``cmd``.

    Reuses ``cortex/lexicon.py``'s ``levenshtein`` (the single edit-distance
    implementation the router's query-side correction already uses) — this
    is a caller, not a second implementation. Returns ``(verb, distance)``
    only when exactly one verb is closest (ties are abstentions, mirroring
    the router's unique-correction rule); ``None`` otherwise.
    """
    from assistant.core.lexicon import levenshtein  # lazy: cold path stays cold
    if len(cmd) < _SUGGEST_MIN_LEN or not cmd.isalpha():
        return None
    lowered = cmd.lower()
    ranked: List[Tuple[int, str]] = []
    for verb in ROUTES:
        distance = levenshtein(lowered, verb, cap=_SUGGEST_MAX_DISTANCE)
        if distance <= _SUGGEST_MAX_DISTANCE:
            ranked.append((distance, verb))
    if not ranked:
        return None
    ranked.sort()
    if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
        return None  # ambiguous: two verbs equally close — do not guess
    return ranked[0][1], ranked[0][0]


def _route_free_text(argv: List[str]) -> int:
    """The verbless front door: whole argv as one request, through the
    cortex pipeline one-shot (read-only routing; writes stay behind the
    chat confirmation gate). A leading ``--`` separator is dropped so
    `caelestia-assist -- make my bar thinner` reads naturally too."""
    words = argv[1:] if argv and argv[0] == "--" else argv
    text = " ".join(words).strip()
    if not text:
        print(USAGE)
        return 0
    # Guard argparse from a free-text phrase that happens to start with
    # a dash: everything after `--` is the positional, never an option.
    guard = ["--"] if text.startswith("-") else []
    return _resolve("route")[0](["route", *guard, text])


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or (argv[0] in ("-h", "--help", "help") and len(argv) == 1):
        print(USAGE)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd not in ROUTES:
        near = suggest_verb(cmd)
        if near is not None:
            verb, distance = near
            # The settings layer's own correction voice: the suggestion is
            # printed, never executed — the user re-runs the right verb.
            print(f"caelestia-assist: unknown command {cmd!r}", file=sys.stderr)
            print(f"did you mean: {verb} (distance {distance})", file=sys.stderr)
            return 1
        return _route_free_text(argv)
    fn, keep_name = _resolve(cmd)
    return fn(argv if keep_name else rest)


def _run() -> int:
    try:
        return main()
    except BrokenPipeError:  # output piped into head/less that closed early
        return 0


if __name__ == "__main__":
    raise SystemExit(_run())
