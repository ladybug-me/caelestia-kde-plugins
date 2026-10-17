"""Explainability for the settings layer (issue #120), read-only.

Read-only: this module never writes anything. It answers "why does it look
like this" questions against the live config state (the target shell.json,
with registry defaults for unset keys — the same effective-value semantics
ConfigObject's fallback gives the QML side), in the exact style the issue
itself shows:

    "Blur is enabled because Glass Mode is currently active."
    "The launcher scale is currently set to 1.25x."

Grounding is the same registry + citations as every other tool: causal
answers come from registry.EXPLAIN_RULES (each rule carries its own
file:line citations); with no rule firing, the
answer is the live value readback with validation, default and the C++
declaration citation.

Honesty note (offline vs in-shell): the QML service evaluates the same
rules with LIVE session flags (GameMode.enabled, Colours.light). This CLI
only sees the file, so rules referencing _gamemode/_light do not fire
here — they fall through to the value readback instead of guessing.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .parser import _NOUN_RES, _noun_matched, _normalize
from assistant.adapters.caelestia.registry import EXPLAIN_RULES, ToolSpec, tool_by_name, tool_by_path

FileTarget = Union[str, Path]

# Read-only adjective forms the \b-bounded write-path nouns intentionally
# do not match (the write path stays byte-frozen; only the read
# path learns these). Keep this list small and obvious.
EXPLAIN_SYNONYMS: Dict[str, str] = {
    "blurry": "blur",
    "fuzzy": "blur",
    "laggy": "animations",
    "sluggish": "animations",
    "hide": "auto hide",
    "hides": "auto hide",
    "hidden": "auto hide",
    "disappear": "auto hide",
    "disappears": "auto hide",
}

# Phenomenon adjectives that should pick a specific candidate when that
# candidate is already noun-matched ("why is my bar big" -> the SIZE tool,
# not the position tool). Read-only; the write path is unchanged.
EXPLAIN_PREFERENCES: Dict[str, str] = {
    "big": "setBarScale",
    "large": "setBarScale",
    "huge": "setBarScale",
    "small": "setBarScale",
    "tiny": "setBarScale",
    "giant": "setDockIconSize",
    "slow": "setAnimationSpeed",
    "fast": "setAnimationSpeed",
    "speedy": "setAnimationSpeed",
    "snappy": "setAnimationSpeed",
}


class ExplainError(RuntimeError):
    """Raised when a question cannot be answered."""


def _spec_json(spec: ToolSpec) -> Dict[str, Any]:
    """Explicit JSON shape for the ``spec`` attached to an explain answer.

    The raw ``ToolSpec`` dataclass is not JSON-serializable, and the
    explain result rides through ``cortex`` ``--json`` output and the
    brain bridge's ``json.dumps`` — so the spec is flattened field by
    field here, mirroring the tools.json row (the same explicit-dict
    discipline the PLAN verdict's candidates use). Never a bare object.
    """
    return {
        "name": spec.name,
        "path": spec.path,
        "kind": spec.kind,
        "default": spec.default,
        "minimum": spec.minimum,
        "maximum": spec.maximum,
        "enum": list(spec.enum) if spec.enum is not None else None,
        "global_only": spec.global_only,
        "step": spec.step,
        "nouns": list(spec.nouns),
        "group": spec.group,
        "citations": [list(citation) for citation in spec.citations],
        "string_max_len": spec.string_max_len,
    }


def _read_state(target: FileTarget, specs: List[ToolSpec]) -> Dict[str, Any]:
    """Effective state for the given specs: file value, else default."""
    import json

    path = Path(target)
    data: Dict[str, Any] = {}
    if path.exists():
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                data = parsed
        except (OSError, ValueError):
            raise ExplainError(
                f"target file {path} cannot be parsed; nothing was written"
            )
    state: Dict[str, Any] = {}
    for spec in specs:
        node: Any = data
        for segment in spec.path.split("."):
            if not isinstance(node, dict) or segment not in node:
                node = None
                break
            node = node[segment]
        state[spec.path] = spec.default if node is None else node
    return state


def _paths_in(expr: str) -> List[str]:
    """Dotted paths referenced by a rule's `when` predicate."""
    out: List[str] = []
    for token in re.findall(r"[A-Za-z_][\w.]*", expr):
        if token not in ("true", "false") and token not in out:
            out.append(token)
    return out


def _values_equal(have: Any, want: Any) -> bool:
    if have == want:
        return True
    if isinstance(have, (int, float)) and isinstance(want, (int, float)) \
            and not isinstance(have, bool) and not isinstance(want, bool):
        return abs(float(have) - float(want)) < 1e-9
    if isinstance(have, bool) and isinstance(want, str):
        return str(have).lower() == want.lower()
    return False


def _eval_when(rule: Dict[str, Any], state: Dict[str, Any]) -> bool:
    """The restricted predicate grammar (same spec as the QML service):
    `path == value` / `!=` / `>` / `<` joined by ` and `. Unknown state
    keys (live-session flags the file cannot know) make the rule NOT fire."""
    clauses = str(rule["when"]).split(" and ")
    for clause in clauses:
        m = re.match(r"^\s*([\w.]+)\s*(==|!=|>|<)\s*(.+?)\s*$", clause)
        if not m:
            return False
        key, op, raw = m.group(1), m.group(2), m.group(3)
        have = state.get(key)
        if have is None:
            return False
        if raw == "true":
            want: Any = True
        elif raw == "false":
            want = False
        elif re.fullmatch(r"-?\d+(\.\d+)?", raw):
            want = float(raw) if "." in raw else int(raw)
        else:
            want = raw.strip("\"'")
        if op == "==":
            ok = _values_equal(have, want)
        elif op == "!=":
            ok = not _values_equal(have, want)
        elif op == ">":
            ok = isinstance(have, (int, float)) and have > want
        else:
            ok = isinstance(have, (int, float)) and have < want
        if not ok:
            return False
    return True


def _fmt(value: Any) -> str:
    if value is True:
        return "on"
    if value is False:
        return "off"
    if isinstance(value, float):
        return f"{round(value, 2):g}"
    return str(value)


def _render(template: str, state: Dict[str, Any], cites: Tuple[str, ...]) -> str:
    """Render an answer template. Placeholders:
    {dotted.path} -> formatted value; {cite}/{citeN} -> citations;
    {a|b} -> alternation resolved by the first numeric value placeholder
    in the template (> 1 picks the first option). Same spec as QML."""
    out = template
    first_numeric: Optional[float] = None
    m = re.search(r"\{([A-Za-z_][\w.]*)\}", out)
    if m:
        v = state.get(m.group(1))
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            first_numeric = float(v)
    out = re.sub(
        r"\{([^{}|]+)\|([^{}]+)\}",
        lambda mm: mm.group(1) if (first_numeric is not None and first_numeric > 1)
        else mm.group(2),
        out,
    )
    out = re.sub(
        r"\{cite(\d*)\}",
        lambda mm: cites[int(mm.group(1)) - 1] if mm.group(1)
        else (cites[0] if cites else ""),
        out,
    )
    out = re.sub(
        r"\{([A-Za-z_][\w.]*)\}",
        lambda mm: _fmt(state[mm.group(1)]) if mm.group(1) in state else mm.group(0),
        out,
    )
    return out


def resolve_spec(query: str) -> Optional[ToolSpec]:
    """A path, a tool name, or a noun-matched question -> one ToolSpec.

    Noun matching reuses the parser's own matcher (so the write and read
    surfaces share one noun grammar), with two READ-ONLY refinements the
    frozen write path deliberately does not get:

    - EXPLAIN_SYNONYMS: adjective forms ("blurry" -> "blur") that the
      \\b-bounded write-path nouns intentionally do not match;
    - phenomenon preference: for "why is my X so Y" questions, candidates
      whose noun matches in the predicate tail (after is/looks/seems/so)
      win over candidates that only matched the subject ("dock" is
      incidental in "why is my dock blurry" — the phenomenon is blur).

    Ambiguity that survives both refinements does not guess: None."""
    q = query.strip()
    spec = tool_by_path(q) or tool_by_name(q)
    if spec is not None:
        return spec
    text = _normalize(q)
    for adj, noun in EXPLAIN_SYNONYMS.items():
        if re.search(rf"\b{re.escape(adj)}\b", text):
            text = text.replace(adj, noun)
    matched = _noun_matched(text)
    if len(matched) > 1:
        # Preference adjectives first: an explicit phenomenon word picks
        # its tool when that tool is among the candidates.
        for word, tool_name in EXPLAIN_PREFERENCES.items():
            if re.search(rf"\b{re.escape(word)}\b", text):
                preferred = next((s for s in matched if s.name == tool_name), None)
                if preferred is not None:
                    return preferred
        # Phenomenon preference: in "why is my X so Y", Y (the predicate)
        # comes LAST — the candidate whose noun matches latest in the
        # sentence is the phenomenon being asked about; the subject ("dock"
        # in "why is my dock blurry") is incidental. Deterministic and
        # position-based; ties stay ambiguous (no guessing).
        def last_position(spec: ToolSpec) -> int:
            best = -1
            for regex in _NOUN_RES[spec.name]:
                for m in regex.finditer(text):
                    best = max(best, m.end())
            return best

        positions = {spec.name: last_position(spec) for spec in matched}
        latest = max(positions.values())
        winners = [s for s in matched if positions[s.name] == latest]
        if len(winners) == 1:
            return winners[0]
        matched = winners
    if len(matched) == 1:
        return matched[0]
    return None


def explain(query: str, target: FileTarget) -> Dict[str, Any]:
    """Answer one explainability question against the target file.

    Returns {ok, answer, cites, rule, spec} — ``spec`` flattened to a
    JSON-serializable dict (see ``_spec_json``), never the raw dataclass.
    Never writes.
    """
    spec = resolve_spec(query)
    if spec is None:
        candidates = _noun_matched(_normalize(query))
        if candidates:
            names = ", ".join(s.name for s in candidates)
            raise ExplainError(
                f"the question's noun matches are ambiguous ({names}); "
                "name the setting explicitly (a path or a tool name)"
            )
        raise ExplainError(
            f"cannot resolve {query!r} to a registry setting "
            f"(see --list-tools for paths)"
        )

    needed = [spec.path]
    for rule in EXPLAIN_RULES:
        if rule["path"] != spec.path:
            continue
        for p in _paths_in(str(rule["when"])):
            if p not in needed:
                needed.append(p)
    state = _read_state(target, [tool_by_path(p) for p in needed if tool_by_path(p)])

    for rule in EXPLAIN_RULES:
        if rule["path"] != spec.path:
            continue
        if _eval_when(rule, state):
            cites = tuple(str(c) for c in rule["cites"])  # type: ignore[arg-type]
            return {
                "ok": True,
                "answer": _render(str(rule["answer"]), state, cites),
                "cites": cites,
                "rule": True,
                "spec": _spec_json(spec),
            }

    # Value readback fallback, in the issue's own style.
    cite = spec.citations[0][0] if spec.citations else ""
    if spec.kind == "enum":
        validation = "one of " + "|".join(str(v) for v in (spec.enum or ()))
    elif spec.kind == "bool":
        validation = "on/off"
    elif spec.kind == "string":
        validation = "string"
    else:
        validation = f"range {spec.minimum}-{spec.maximum}"
    value = state.get(spec.path, spec.default)
    answer = (
        f"The {spec.path} is currently set to {_fmt(value)} "
        f"({validation}, default {_fmt(spec.default)}"
        + (f"; {cite}" if cite else "") + ")."
    )
    return {
        "ok": True,
        "answer": answer,
        "cites": tuple(c[0] for c in spec.citations[:2]),
        "rule": False,
        "spec": _spec_json(spec),
    }


# ---------------------------------------------------------------------------
# Multi-hop provenance (issue #120 Phase 1.3) — read-only backward walk:
#   current value <- which apply set it (undo history)
#                 <- which preset/tool call produced that apply (ledger)
#                 <- that preset's approval rate for this user (bandit)
# Stop at the first ledger entry found or MAX_PROVENANCE_HOPS, whichever
# comes first. Every source is read; nothing is written, ever.
# ---------------------------------------------------------------------------

MAX_PROVENANCE_HOPS = 5


def _call_paths(ops: List[Any]) -> List[str]:
    """Registry paths touched by a ledger diff's call list."""
    from assistant.adapters.caelestia.registry import tool_by_name

    paths: List[str] = []
    for call in ops or []:
        if not isinstance(call, dict):
            continue
        tool = str(call.get("tool", ""))
        spec = tool_by_name(tool)
        if spec is not None:
            paths.append(spec.path)
    return paths


def _ledger_settings_items(ledger_path: Optional[FileTarget]) -> List[Dict[str, Any]]:
    """The ledger's decided settings proposals, newest decision last-walked
    first. Missing file -> []. Unreadable -> [] (provenance is best-effort
    and read-only; it must never break the explain answer)."""
    import json as _json

    if not ledger_path:
        return []
    path = Path(ledger_path)
    if not path.exists():
        return []
    try:
        data = _json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    items = data.get("proposals", []) if isinstance(data, dict) else []
    return [i for i in items
            if isinstance(i, dict) and i.get("kind") == "settings"]


def provenance(query: str, target: FileTarget,
               ledger_path: Optional[FileTarget] = None,
               bandit_state: Optional[Dict[str, Any]] = None,
               max_hops: int = MAX_PROVENANCE_HOPS) -> Dict[str, Any]:
    """Backward provenance chain for one setting, fully read-only.

    Returns {ok, path, value, hops, chain, stopped_by} where chain is a
    list of rendered hop strings (newest first) and stopped_by names the
    stop rule ("no more sources", "ledger entry", or "hop limit").
    """
    from . import history as history_mod

    spec = resolve_spec(query)
    if spec is None:
        raise ExplainError(
            f"cannot resolve {query!r} to a registry setting "
            f"(see --list-tools for paths)"
        )

    chain: List[Dict[str, Any]] = []
    stopped_by = "no more sources"
    hops = 0

    # Hop 1: the undo history — which apply set the value most recently.
    apply_entry = None
    try:
        for entry in history_mod.entries(target):  # newest first
            if any(op.get("path") == spec.path for op in entry.get("ops", [])):
                apply_entry = entry
                break
    except history_mod.HistoryError:
        apply_entry = None
    if apply_entry is not None and hops < max_hops:
        hops += 1
        label = str(apply_entry.get("label") or "(unlabelled)")
        chain.append({
            "hop": hops,
            "source": "apply history",
            "detail": (f"apply #{apply_entry.get('id')} at {apply_entry.get('at')} "
                       f"set {spec.path} (label: {label})"),
        })
    elif apply_entry is not None:
        stopped_by = "hop limit"

    # Hop 2: the ledger — which proposal/tool call produced that apply
    # (matched by target file + touched path; the ledger file stores
    # proposals oldest-first, so walk reversed() = newest first).
    ledger_hit = None
    items = _ledger_settings_items(ledger_path)
    for item in reversed(items):
        diff = item.get("diff") or {}
        if str(diff.get("file", "")) != str(Path(target)):
            continue
        if spec.path in _call_paths(diff.get("calls")):
            ledger_hit = item
            break
    if ledger_hit is not None:
        if hops < max_hops:
            hops += 1
            status = str(ledger_hit.get("status", "?"))
            decided = ledger_hit.get("decided_at") or "?"
            reason = str(ledger_hit.get("reason") or "")
            preset = (ledger_hit.get("diff") or {}).get("preset")
            detail = (f"ledger proposal #{ledger_hit.get('id')} ({status} at "
                      f"{decided}): {reason or 'settings change'}")
            if preset:
                detail += f" — via preset '{preset}'"
            chain.append({
                "hop": hops,
                "source": "ledger",
                "detail": detail,
                "preset": preset,
                "status": status,
            })
            stopped_by = "ledger entry"
        else:
            stopped_by = "hop limit"

    # Hop 3: the preset bandit — that preset's approval rate for this user
    # (Beta-Binomial posterior mean; preset_bandit.py's NamedBandit arms).
    preset_name = None
    for hop in chain:
        if hop.get("source") == "ledger" and hop.get("preset"):
            preset_name = hop["preset"]
            break
    if preset_name and bandit_state and hops < max_hops:
        arms = bandit_state.get("arms") or {}
        arm = arms.get(str(preset_name))
        if isinstance(arm, (list, tuple)) and len(arm) == 2:
            alpha, beta = float(arm[0]), float(arm[1])
            if alpha + beta > 2.0:  # more than the prior (1, 1): real data
                hops += 1
                mean = alpha / (alpha + beta)
                chain.append({
                    "hop": hops,
                    "source": "preset bandit",
                    "detail": (f"preset '{preset_name}' approval estimate for "
                               f"this user: {mean:.2f} "
                               f"(Beta({alpha:g}, {beta:g}) posterior mean, "
                               "Thompson-sampling arm)"),
                })

    if len(chain) >= max_hops and stopped_by == "no more sources":
        stopped_by = "hop limit"

    return {
        "ok": True,
        "path": spec.path,
        "value": None,  # filled by explain(); kept explicit here
        "hops": hops,
        "chain": chain,
        "stopped_by": stopped_by,
    }


def render_provenance(result: Dict[str, Any]) -> List[str]:
    """Plain-text provenance chain for the --explain output."""
    lines = [f"provenance for {result['path']} ({result['hops']} hop(s), "
             f"stopped by: {result['stopped_by']})"]
    for hop in result["chain"]:
        lines.append(f"  <- hop {hop['hop']} [{hop['source']}]: {hop['detail']}")
    return lines
