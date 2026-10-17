"""cortex.delegate — inline execution of DELEGATE verdicts.

When the router hands a request to another layer (genius / diagnose /
search / brain / issue / agent), the conversation does not stop at a
"try: <command>" hint: the target layer's own existing entry function
runs inline, read-only, and its answer renders in the same turn. This is
the routing fix's core: every verb's worth of request — verbless or not
— gets an answer, never a pointer somewhere else.

Runners (each calls an EXISTING entry point; nothing here is a second
implementation of anything):

- ``genius``   : ``genius.meta.route_and_do`` (the one the chat REPL
                 already used inline — generalized, not replaced).
- ``diagnose`` : ``diagnostics.engine.diagnose`` — signature matching of
                 the routed text, rendered by ``render_report``.
- ``search``   : ``retrieval.search.search`` — the offline BM25 index
                 (the same function ``issues`` uses for similar issues).
- ``brain``    : ``brain.cli.main(["brief"])`` — the brain's synthesized
                 answer to plan/focus-shaped requests (pending proposals,
                 forecasts, rhythm), composed from the ledger.
- ``issue``    : ``issues.cli.main(["draft", ...])`` — a PREVIEW draft
                 composed from the routed text; without ``--confirm`` it
                 writes nothing, ever (the module's own contract).
- ``agent``    : ``agent.cli.main(["--simulate", ...])`` — the projected
                 task graph, nothing executed. The agent's execute path
                 (with its stdin consent gate) is deliberately NOT
                 reachable from here: consent stays a manual step.

Safety spine (unchanged, inherited — this module adds zero write paths):

- every runner is read-only or preview-only (see per-runner notes);
- a runner failure degrades to the caller's hint path (``run_delegate``
  returns None) — the turn shows the "try:" fallback, never a crash;
- stdlib + existing assistant modules only; the behavioral safety tests
  untouched; no subprocess, no network, no clock dependence beyond what
  the called modules already do.
"""

from __future__ import annotations

import contextlib
import io
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = ["RUNNERS", "runner_names", "run_delegate"]

# A runner: (routed text, brain state dict) -> (JSON-safe payload, lines).
# The payload is what --json / the QML bridge emit; the lines are the
# terminal rendering for the same turn.
DelegateRunner = Callable[[str, Dict[str, Any]], Tuple[Dict[str, Any], List[str]]]


# ---------------------------------------------------------------------------
# Formatters.
# ---------------------------------------------------------------------------


def _format_genius(payload: Dict[str, Any]) -> List[str]:
    """Compact formatter for an inline genius answer (the chat card body
    the REPL has always printed, factored out so both surfaces share it)."""
    lines: List[str] = []
    domain = payload.get("domain") or payload.get("verdict")
    confidence = payload.get("confidence")
    lines.append(f"  genius -> {domain}"
                 + (f" (confidence {confidence})" if confidence else ""))
    result = payload.get("result")
    if result is None:
        if payload.get("error"):
            lines.append(f"  error: {payload['error']}")
        elif payload.get("message"):
            lines.append(f"  {payload['message']}")
        return lines
    for key, value in list(result.items())[:14]:
        if key == "note":
            continue  # rendered separately below
        if isinstance(value, (int, float, str, bool)) or value is None:
            lines.append(f"  {key.replace('_', ' ')}: {value}")
        elif isinstance(value, list) and value and isinstance(value[0], (int, float, str)):
            lines.append(f"  {key.replace('_', ' ')}: {value[:8]}")
    note = result.get("note") if isinstance(result, dict) else None
    if note:
        lines.append(f"  note: {note}")
    return lines


def _format_search(payload: Dict[str, Any]) -> List[str]:
    results = payload.get("results") or []
    if not results:
        return ["  no corpus document matched the local docs/issues index"]
    lines = ["  retrieval hits (offline BM25):"]
    for i, hit in enumerate(results, start=1):
        lines.append(f"    [{i}] {hit.get('doc_id', '?')} — {hit.get('title', '')}"
                     f"  (score {hit.get('score', 0)})")
        if hit.get("source"):
            lines.append(f"        source: {hit['source']}")
        if hit.get("snippet"):
            snippet = str(hit["snippet"]).replace("\n", " ")
            lines.append(f"        {snippet[:160]}")
    return lines


# ---------------------------------------------------------------------------
# Runners (one per DELEGATE category; each reuses an existing entry).
# ---------------------------------------------------------------------------


def _delegate_genius(text: str, state: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    from assistant.capabilities.genius import meta as genius_meta

    payload = genius_meta.route_and_do(text, state.get("genius_learn"))
    return payload, _format_genius(payload)


def _delegate_diagnose(text: str, state: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    from assistant.capabilities.diagnostics.engine import diagnose as diagnose_text
    from assistant.capabilities.diagnostics.engine import render_report

    payload = diagnose_text(text)
    # render_report is the diagnose CLI's own renderer; reuse verbatim.
    lines = [f"  {line}" for line in render_report(payload).splitlines()]
    return payload, lines


def _delegate_search(text: str, state: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    from assistant.capabilities.retrieval import search as retrieval_search

    results = retrieval_search.search(text, k=5)
    payload = {"query": text, "results": results}
    return payload, _format_search(payload)


def _delegate_brain(text: str, state: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """The brain's answer to plan/focus-shaped requests: the daily brief
    (pending proposals + forecast + rhythm), composed exactly the way
    ``caelestia-assist brief`` composes it — same entry function, same
    ledger, read-only."""
    from assistant.capabilities.brain import cli as brain_cli

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = brain_cli.main(["brief"])
    if code != 0:
        raise RuntimeError(f"brain brief exited {code}")
    body = buffer.getvalue()
    payload = {"brief": body}
    lines = [f"  {line}" for line in body.splitlines() if line.strip()]
    return payload, lines


def _delegate_issue(text: str, state: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Compose an issue-draft PREVIEW from the routed text (title seed +
    description). Without --confirm the issues layer writes nothing —
    its own docstring contract, relied on here verbatim."""
    from assistant.capabilities.issues import cli as issues_cli

    title = " ".join(text.split())[:72]
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = issues_cli.main(["draft", "--title", title], stdin_text=text)
    if code != 0:
        raise RuntimeError(f"issue draft exited {code}")
    body = buffer.getvalue()
    payload = {"draft": body}
    lines = [f"  {line}" for line in body.splitlines()]
    return payload, lines


def _delegate_agent(text: str, state: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Simulate the multi-step goal and render the projected task graph.
    Calls ``agent.cli.main`` with ``--simulate`` — the agent's execute
    path (and its consent gate) is never reached from the conversation:
    turning a simulation into action stays a manual, explicit step."""
    from assistant.capabilities.agent import cli as agent_cli

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = agent_cli.main(["--simulate", *text.split()])
    if code != 0:
        raise RuntimeError(f"agent simulate exited {code}")
    body = buffer.getvalue()
    payload = {"simulated": True, "output": body}
    lines = [f"  {line}" for line in body.splitlines() if line.strip()]
    return payload, lines


RUNNERS: Dict[str, DelegateRunner] = {
    "genius": _delegate_genius,
    "diagnose": _delegate_diagnose,
    "search": _delegate_search,
    "brain": _delegate_brain,
    "issue": _delegate_issue,
    "agent": _delegate_agent,
}


def runner_names() -> List[str]:
    """Delegate categories that can answer inline this turn."""
    return sorted(RUNNERS)


def run_delegate(name: str, text: str,
                 state: Optional[Dict[str, Any]] = None
                 ) -> Optional[Tuple[Dict[str, Any], List[str]]]:
    """Run one delegate inline. Returns ``(payload, lines)`` on success,
    or ``None`` when the category has no runner or the run failed — the
    caller keeps its hint path in that case (the safety net stays)."""
    runner = RUNNERS.get(name)
    if runner is None:
        return None
    try:
        return runner(text, state or {})
    except Exception:  # honest degradation: the caller's hint path stands
        return None
