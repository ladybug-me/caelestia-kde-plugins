"""Assistant pipeline: Layer 1 deterministic rules, then Layer 2 retrieval.

Usage:
    python3 -m assistant.core.ask_pipeline "text or pasted log"
    ... | python3 -m assistant.core.ask_pipeline        (stdin)
    python3 -m assistant.core.ask_pipeline "text" --json
    python3 -m assistant.core.ask_pipeline "text" --generative
    python3 -m assistant.core.ask_pipeline "text" --generative --json

assist(text) -> dict:
- Runs the Layer 1 deterministic engine (assistant.capabilities.diagnostics.engine).
- On MATCH/AMBIGUOUS: returns {"layer": "rules", ...} — verified signatures.
- On NO_MATCH: runs the offline retrieval search (assistant.capabilities.retrieval.search)
  and returns {"layer": "retrieval", ...} with a banner stating these are the
  closest past resolutions from the repo's docs and resolved issues, NOT
  verified diagnoses.

--generative (Layer 3, opt-in, only reached on a Layer 1 NO_MATCH): the
pipeline delegates to assistant.capabilities.generative.rag.suggest(text, k, enable=True)
and returns that result dict — {"enabled", "available", "suggestion",
"retrieval_hits", "reason"} — with one added "layer": "generative" marker
(--json emits exactly this dict; the human renderer is rag.render_result).
Layer 1 always wins when it matches: the generative layer exists only for
problems the rules cannot name. Every Layer 3 guarantee (opt-in only,
loopback-only single connection attempt, retrieval-grounded prompt,
SUGGESTED_NOT_EXECUTED labeling, graceful unavailability) lives in
assistant/generative/rag.py; this module routes to it and reimplements
none of it. Without --generative the generative code path is never taken
and the output is identical to the two-layer pipeline.

Import surface: assistant.capabilities.diagnostics.engine + assistant.capabilities.retrieval +
assistant.capabilities.generative.rag (intra-package relative imports) + stdlib only.
Nothing here executes commands or touches the network.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from assistant.capabilities.diagnostics import engine
from assistant.capabilities.generative import rag
from assistant.capabilities.retrieval import search

RETRIEVAL_BANNER = (
    "Layer 2 — closest past resolutions from this repo's docs and resolved issues. "
    "These are retrieval hits, NOT verified diagnoses: the assistant may be wrong. "
    "Check the cited source before acting on anything below."
)

NO_RESULTS_NOTE = (
    "Nothing in the local docs or resolved issues matched either. The honest answer is: "
    "enable Debug Mode (Nexus -> About -> Advanced), capture logs with "
    "journalctl --user -u caelestia-shell, and draft an issue report."
)


def assist(text: str, k: int = 5, generative: bool = False) -> Dict[str, Any]:
    """Run Layer 1, fall back to Layer 2 retrieval on NO_MATCH.

    With generative=True, a NO_MATCH is delegated to Layer 3
    (rag.suggest(enable=True)) instead: its result dict — which carries the
    Layer 2 hits, the availability state, and the safety-labeled suggestion —
    is returned with a "layer": "generative" marker. A Layer 1 match always
    wins regardless of the flag.
    """
    diagnosis = engine.diagnose(text)

    if diagnosis["verdict"] != "NO_MATCH":
        return {
            "layer": "rules",
            "verdict": diagnosis["verdict"],
            "margin": diagnosis.get("margin"),
            "candidates": [
                {
                    "id": result["rule"]["id"],
                    "title": result["rule"].get("title", ""),
                    "score": result["score"],
                    "confidence": result["rule"].get("confidence", "probable"),
                    "evidence": result.get("evidence", [])[:6],
                }
                for result in diagnosis["candidates"]
            ],
            "report": engine.render_report(diagnosis),
        }

    if generative:
        # Layer 3 owns this path end to end — grounding, loopback-only
        # transport, availability, labeling — so delegate and add only the
        # layer marker the renderer/JSON emit; suggest() already ran the
        # Layer 2 search that fills its retrieval_hits.
        return {"layer": "generative", **rag.suggest(text, k=k, enable=True)}

    results = search.search(text, k=k)
    payload: Dict[str, Any] = {
        "layer": "retrieval",
        "verdict": "NO_MATCH",
        "banner": RETRIEVAL_BANNER,
        "results": results,
    }
    if not results:
        payload["note"] = NO_RESULTS_NOTE
    return payload


def render(payload: Dict[str, Any]) -> str:
    """Human-readable rendering of an assist() payload."""
    if payload["layer"] == "rules":
        return payload["report"]
    if payload["layer"] == "generative":
        return rag.render_result(payload)
    lines: List[str] = []
    lines.append("caelestia assistant — layer 2 retrieval (closest past resolutions)")
    lines.append(payload["banner"])
    lines.append("")
    results = payload.get("results", [])
    if not results:
        lines.append(payload.get("note", "No results."))
        return "\n".join(lines)
    for i, hit in enumerate(results, start=1):
        lines.append(f"[{i}] {hit['doc_id']} — {hit['title']}  (score {hit['score']})")
        lines.append(f"    source: {hit['source']}")
        lines.append(f"    snippet: {hit['snippet']}")
    lines.append("")
    lines.append("Reminder: these are past resolutions, not verified answers. Nothing was executed.")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m assistant.core.ask_pipeline",
        description="caelestia assistant: deterministic rules first, retrieval fallback (offline, runs nothing).",
    )
    parser.add_argument("text", nargs="?", default="-", help="problem text or pasted log; '-' reads stdin")
    parser.add_argument("--json", action="store_true", help="emit the raw payload as JSON")
    parser.add_argument(
        "--generative",
        action="store_true",
        help=(
            "opt in to the Layer 3 generative suggestion layer when the rules layer has no match "
            "(requires a local Ollama server on loopback; suggestion-only, nothing is executed)"
        ),
    )
    parser.add_argument("-k", type=int, default=5, help="max retrieval results (default 5)")
    args = parser.parse_args(argv)

    if args.text == "-":
        text = sys.stdin.read()
    else:
        text = args.text

    payload = assist(text, k=args.k, generative=args.generative)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(render(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
