"""assistant.capabilities.generative.assisted — model-assisted plan drafting for
out-of-ontology requests (A8, exponential-build-5). DEFAULT OFF.

The local ontology answers most requests deterministically. For the
rest — genuinely out-of-ontology phrasings — A8 offers an OPTIONAL
drafting tier: a LOCAL model (the existing loopback-only Ollama
transport, generative.client) is asked to map the phrasing onto
REGISTRY TOOLS, and every draft is validated by the same planner the
user's own requests go through. The hard guarantees, unchanged from
the Layer-3 contract this package already pins:

- DEFAULT OFF: enabled ONLY by an explicit caller decision
  (``enable=True`` — the CLI's ``--local-model``); when off, the
  function returns {"verdict": "DISABLED"} and the pipeline behaves
  EXACTLY as it would without this module (test-pinned);
- LOOPBACK ONLY: the transport already hard-rejects any non-loopback
  host before connecting (client.parse_loopback) — unchanged;
- SINGLE ATTEMPT: one request, no streaming, no retries — a timeout is
  an honest "model unavailable", never a retry loop;
- VALIDATED, NEVER EXECUTED: every drafted (tool, value) pair rides
  the ordinary planner (type/range validation against the live
  registry). Invalid drafts are REPORTED, not repaired; valid ones are
  returned as a DRY-RUN plan with the MODEL_SUGGESTED label and the
  standard SUGGESTED_NOT_EXECUTED discipline. The applier is never
  reachable from this module;
- PROMPT IS SCHEMA-CONSTRAINED: the prompt carries the registry tool
  list (names + kinds + ranges, generated from tools.json) and
  demands JSON [{tool, value}] — a format the model can satisfy but
  cannot use to invent write paths.

Pure module apart from the injected transport: no RNG, no clock, the
caller injects ``conn_factory`` for tests.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

__all__ = ["MODEL_SUGGESTED_LABEL", "build_prompt", "parse_drafts",
           "draft_plan", "render_result"]

MODEL_SUGGESTED_LABEL = "MODEL_SUGGESTED"
_TOOL_NAME_RE = re.compile(r"^set[A-Za-z0-9_]{1,64}$")


def build_prompt(text: str, max_tools: int = 272) -> str:
    """The schema-constrained prompt: the registry's tool list (name,
    kind, range) plus the user's phrasing, demanding JSON output. Pure
    and deterministic."""
    from assistant.adapters.caelestia.registry import TOOL_SPECS
    lines = [
        "You map a user's desktop-customization request onto EXACTLY the",
        "tools below. Reply with ONLY a JSON array like",
        '[{"tool": "setBarScale", "value": 0.8}] — no prose. Use a tool',
        "only for what it actually controls; use [] when nothing fits.",
        "",
        "TOOLS:",
    ]
    count = 0
    for spec in TOOL_SPECS:
        if count >= max_tools:
            break
        if spec.enum:
            detail = "enum: " + "|".join(str(v) for v in spec.enum)
        elif spec.minimum is not None or spec.maximum is not None:
            detail = f"range {spec.minimum}..{spec.maximum}"
        else:
            detail = spec.kind
        lines.append(f"- {spec.name} ({spec.kind}; {detail})")
        count += 1
    lines += ["", "REQUEST:", text.strip()]
    return "\n".join(lines)


def parse_drafts(raw: str) -> Dict[str, Any]:
    """Extract the JSON array from the model's output. Returns
    {"drafts": [{tool, value}], "parse_error": optional}. Tolerates
    code fences and prose around the array (single scan, no retries —
    the model's output is DATA here, not instructions)."""
    text = str(raw or "")
    match = re.search(r"\[.*\]", text, re.S)
    if not match:
        return {"drafts": [], "parse_error": "no JSON array in output"}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return {"drafts": [], "parse_error": f"invalid JSON: {exc}"}
    drafts = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool", ""))
        if not _TOOL_NAME_RE.match(tool):
            continue
        drafts.append({"tool": tool, "value": item.get("value")})
    return {"drafts": drafts[:8], "parse_error": None}


def draft_plan(text: str, *, enable: bool = False,
               model: Optional[str] = None, url: Optional[str] = None,
               conn_factory=None, target=None) -> Dict[str, Any]:
    """One model-assisted draft for an out-of-ontology request.

    OFF unless ``enable`` — the caller's explicit decision. Returns a
    dict whose verdict is one of DISABLED / UNAVAILABLE / PARSE_EMPTY /
    DRAFTED, carrying only validated, dry-run, MODEL_SUGGESTED-labeled
    plans."""
    if not enable:
        return {"verdict": "DISABLED",
                "note": "the local model tier is off by default; enable "
                        "it with an explicit --local-model (loopback "
                        "Ollama only)"}
    from . import client as gen_client
    try:
        available = gen_client.is_available(url, conn_factory=conn_factory)
    except gen_client.LoopbackViolation as exc:
        return {"verdict": "DISABLED", "note": f"rejected URL: {exc}"}
    if not available:
        return {"verdict": "UNAVAILABLE",
                "note": "no local model answered on the loopback "
                        "endpoint; the request stays out-of-ontology"}
    prompt = build_prompt(text)
    try:
        raw = gen_client.generate(prompt, model=model, url=url,
                                  conn_factory=conn_factory)
    except (gen_client.GenerativeError, OSError) as exc:
        return {"verdict": "UNAVAILABLE",
                "note": f"the model call failed once and is not "
                        f"retried: {exc}"}
    parsed = parse_drafts(raw)
    if parsed["parse_error"] or not parsed["drafts"]:
        return {"verdict": "PARSE_EMPTY",
                "note": parsed["parse_error"] or
                        "the model proposed no tools ([])"}

    # Validation rides the ordinary planner: same registry, same gates.
    from assistant.capabilities.settings import planner
    from assistant.capabilities.settings.cli import default_target
    ops = [{"tool": d["tool"], "action": "set", "value": d["value"],
            "raw": f"{MODEL_SUGGESTED_LABEL}: {d['tool']}="
                   f"{d['value']!r}"}
           for d in parsed["drafts"]]
    plan = planner.plan(ops, target if target is not None
                        else default_target())
    valid = [e for e in plan.get("entries", [])
             if not e.get("error") and not e.get("no_op")]
    rejected = [e for e in plan.get("entries", [])
                if e.get("error") or e.get("no_op")]
    return {
        "verdict": "DRAFTED",
        "label": MODEL_SUGGESTED_LABEL,
        "valid": valid,
        "rejected": rejected,
        "plan": plan,
        "note": ("model drafts are validated by the same planner as "
                 "your own requests and are NEVER executed; applying "
                 "rides the ordinary preview/confirm gates"),
    }


def render_result(result: Dict[str, Any]) -> List[str]:
    label = result.get("verdict", "?")
    if label == "DRAFTED":
        lines = [f"{MODEL_SUGGESTED_LABEL} — {len(result['valid'])} "
                 f"validated draft(s), {len(result['rejected'])} "
                 f"rejected by the planner:"]
        for entry in result["valid"]:
            lines.append(f"  {entry['tool']} = {entry.get('new')!r}")
        for entry in result["rejected"]:
            lines.append(f"  rejected: {entry.get('tool')}: "
                         f"{entry.get('error') or 'no-op'}")
        lines.append(f"  ({result['note']})")
        lines.append("  SUGGESTED_NOT_EXECUTED: review the plan above; "
                     "apply through the normal --apply --confirm path")
        return lines
    return [f"{label}: {result.get('note', '')}"]
