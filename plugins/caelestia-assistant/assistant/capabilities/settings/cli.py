"""CLI for the settings layer (the settings design notes (see module docstrings) §5):

    python3 -m assistant.capabilities.settings "<text>" [--apply] [--file PATH] [--json]
    python3 -m assistant.capabilities.settings --call NAME=VALUE [--call ...] [--apply] [--file PATH] [--json]
    python3 -m assistant.capabilities.settings --list-tools [--group SLUG]
    python3 -m assistant.capabilities.settings --tool NAME
    python3 -m assistant.capabilities.settings --threeway "REQUEST|PRESET" [--file PATH]
        # read-only: current value vs proposal vs undo-restore value
    python3 -m assistant.capabilities.settings --restore [--file PATH]

Guarantees:

- Dry-run by default: without --apply nothing is written — not the target,
  not a backup, not a tmp file — and the printed plan ends with the exact
  re-run command including --apply, so the confirmation step is a
  copy-paste (§5.2). This is the CLI equivalent of #120's "confirmation
  for larger changes".
- Exit codes (§5.1): 0 for every honest verdict (INTENT / AMBIGUOUS /
  NO_INTENT / SUGGESTED, including dry-run plans containing REJECTED
  entries) and for a successful apply/restore/list; 1 for operational
  failures (target file invalid JSON or non-object, unreadable file,
  --apply on a blocked plan, write failure, --restore with no backup);
  2 for usage errors (argparse's default).
- --call (direct tool addressing): NAME is a registry tool name,
  VALUE is parsed as JSON (true/false, numbers, quoted strings; a bare
  unquoted token that fails JSON parsing is treated as a plain string).
  Each --call becomes one {"tool", "action": "set", "value", "raw"} op
  and is routed through the SAME planner/render/--apply path as NL
  requests — this is how presets and model-driven callers address tools
  without the NL parser. A --call run whose plan contains any error entry
  (unknown tool, out-of-range value, bad enum, string-valued tool) exits 1
  even in dry-run: a direct invocation with a bad name or value is a
  caller error, not an honest NL verdict. --call and the positional
  request sentence are mutually exclusive (usage error, exit 2).
- Default target: $XDG_CONFIG_HOME/caelestia/shell.json falling back to
  $HOME/.config/caelestia/shell.json (the same two env vars Layer 1
  already allowlists); never any other path unless --file says so.
- No execution, no network: suggested commands are inert
  SUGGESTED_NOT_EXECUTED strings; the only writes are the ones the
  applier performs behind --apply.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import applier, parser, planner

from assistant.adapters.caelestia import registry
from assistant.adapters.caelestia.registry import ToolSpec, format_value, list_tools_lines, tool_by_name

PROG = "python3 -m assistant.capabilities.settings"

# The string-kind pre-check message: the planner validates
# bool/enum/number values only, so string-valued tools are rejected here,
# before the planner is asked to resolve the op.
CALL_STRING_TOOLS_MESSAGE = (
    "string-valued tools are not settable through --call yet "
    "(registry-only until the planner gains string support)"
)


def default_target() -> Path:
    """§5.1: $XDG_CONFIG_HOME/caelestia/shell.json, falling back to
    $HOME/.config/caelestia/shell.json (QStandardPaths' GenericConfigLocation
    resolves exactly these two; [C7][C29])."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "caelestia" / "shell.json"


def _shell_quote(text: str) -> str:
    """Minimal double-quoting for the re-run command line (stdlib-only)."""
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


# ---------------------------------------------------------------------------
# Rendering (§5.5). --json emits the same structure, machine-readable.
# ---------------------------------------------------------------------------


def _header(apply_mode: bool) -> List[str]:
    return [
        "caelestia assistant — settings layer (NL -> validated tool calls; dry-run by default)",
        "Nothing is written without --apply; this run writes nothing.",
    ] if not apply_mode else [
        "caelestia assistant — settings layer (NL -> validated tool calls; dry-run by default)",
        "Writes are gated: this run writes only the target file and its .assistant-backup sibling.",
    ]


def _rerun(text: str, file_arg: Optional[str], extra: str,
           call_args: Optional[List[str]] = None) -> str:
    """The exact re-run command (§5.2): confirmation as a copy-paste."""
    parts = [PROG]
    if call_args:
        for call in call_args:
            parts.append(f"--call {_shell_quote(call)}")
    if text:
        parts.append(_shell_quote(text))
    if file_arg:
        parts.append(f"--file {file_arg}")
    parts.append(extra)
    return " ".join(parts)


def _render_entry(index: int, entry: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    arrow = f"{entry['path']}: {format_value(entry['old'])} -> {format_value(entry['new'])}"
    action = entry.get("action") or ""
    raw = entry.get("raw")
    paren = f"({entry['tool']}, {action} {raw})" if raw else f"({entry['tool']}, {action})"
    if entry.get("error"):
        lines.append(f"  {index}. {arrow}  {paren} — REJECTED: {entry['error']}")
    elif entry.get("no_op"):
        lines.append(f"  {index}. {arrow}  {paren} [no change needed]")
    else:
        lines.append(f"  {index}. {arrow}  {paren}")
    if entry.get("note"):
        lines.append(f"     note: {entry['note']}")
    return lines


def render_plan(plan: Dict[str, Any], parse_notes: List[str], text: str,
                file_arg: Optional[str], apply_mode: bool,
                apply_result: Optional[Dict[str, Any]],
                call_args: Optional[List[str]] = None) -> List[str]:
    lines = ["caelestia assistant — settings layer (NL -> validated tool calls; dry-run by default)"]
    if apply_result is not None and apply_result.get("written"):
        lines.append(f"Writes are gated: this run wrote {plan['file']} and its .assistant-backup sibling.")
    elif apply_result is not None and apply_result.get("no_changes"):
        lines.append("Nothing is written without --apply; this run wrote nothing (no changes were needed).")
    elif apply_mode:
        lines.append("Writes are gated: this run writes only the target file and its .assistant-backup sibling.")
    else:
        lines.append("Nothing is written without --apply; this run writes nothing.")
    lines.append("")

    entries = plan.get("entries", [])
    notes = list(parse_notes) + list(plan.get("notes", []))
    if apply_result is not None and apply_result.get("written"):
        lines.append(f"Verdict: INTENT — {len(entries)} change(s) applied on {plan['file']}")
    else:
        lines.append(f"Verdict: INTENT — {len(entries)} change(s) planned on {plan['file']}")
    for idx, entry in enumerate(entries, start=1):
        lines.extend(_render_entry(idx, entry))
    if plan.get("apply_blocked"):
        lines.append(
            "apply_blocked: this plan contains rejected or type-mismatched entries; "
            "--apply will refuse to write anything"
        )
    if notes:
        lines.append("notes:")
        lines.extend(f"  - {note}" for note in notes)
    if apply_result is not None:
        lines.append(apply_result["message"])
        if apply_result.get("written"):
            lines.append(f"undo with: {_rerun('', file_arg, '--restore')}")
    else:
        # §5.2: the printed plan ends with the exact re-run command.
        lines.append(f"Apply with: {_rerun(text, file_arg, '--apply', call_args)}")
    return lines


def render_verdict(result: Dict[str, Any]) -> List[str]:
    """Render AMBIGUOUS / NO_INTENT / SUGGESTED verdicts (§5.5)."""
    lines = _header(False)
    lines.append("")
    verdict = result["verdict"]
    if verdict == "AMBIGUOUS":
        if result.get("candidates"):
            lines.append("Verdict: AMBIGUOUS — several settings fit; name one:")
            for idx, cand in enumerate(result["candidates"], start=1):
                lines.append(f"  {idx}. {cand['description']}")
        else:
            lines.append("Verdict: AMBIGUOUS — the request is not specific enough to act on safely.")
        if result.get("notes"):
            for note in result["notes"]:
                lines.append(f"  note: {note}")
        if result.get("question"):
            lines.append(f"Clarify: {result['question']}")
    elif verdict == "NO_INTENT":
        lines.append("Verdict: NO supported setting matched.")
        for note in result.get("notes", []):
            lines.append(f"  {note}")
        if result.get("question"):
            lines.append(f"Clarify: {result['question']}")
    elif verdict == "SUGGESTED":
        lines.append(
            "Verdict: SUGGESTED — this request is real, but it lives outside shell.json "
            "(the scheme/wallpaper system); nothing was executed and nothing was written."
        )
        for suggestion in result.get("suggestions", []):
            lines.append(f"  {suggestion}")
        for note in result.get("notes", []):
            lines.append(f"  note: {note}")
    return lines


# ---------------------------------------------------------------------------
# --list-tools / --tool (registry inspection; these never write anything).
# ---------------------------------------------------------------------------


def _list_tools_lines(group: Optional[str]) -> List[str]:
    """The --list-tools table: group-filtered when --group is given (the
    registry itself renders per-group subtotal lines when unfiltered)."""
    return list_tools_lines(group)


def _tool_card(spec: ToolSpec) -> List[str]:
    """The --tool NAME detail card: every registry fact we hold about one
    tool, including every citation line ("file:line — what it evidences").
    Fields the current registry build does not carry (group, citations on
    the 18-tool core) are rendered as explicit absences, never guessed."""
    if spec.kind == "enum":
        enum_values = "|".join(spec.enum or ()) or "(none)"
        validation = f"enum {enum_values}"
    elif spec.kind == "bool":
        validation = "on/off"
    elif spec.kind == "string":
        validation = "string"
    else:
        low = "(unbounded)" if spec.minimum is None else spec.minimum
        high = "(unbounded)" if spec.maximum is None else spec.maximum
        validation = f"range {low}-{high}"
    group = getattr(spec, "group", None)
    citations = getattr(spec, "citations", None) or ()
    lines = [
        "caelestia assistant — settings layer: tool detail",
        "",
        f"  name:        {spec.name}",
        f"  path:        {spec.path}",
        f"  group:       {group if group else '(no group metadata in this registry build)'}",
        f"  kind:        {spec.kind}",
        f"  validation:  {validation}",
        f"  default:     {format_value(spec.default)}",
        f"  global-only: {'yes' if spec.global_only else 'no'}",
        f"  step:        {spec.step if spec.step else '(not steppable)'}",
        f"  nouns:       {'; '.join(spec.nouns) if spec.nouns else '(none)'}",
        "  citations:",
    ]
    if not citations:
        lines.append("    (none recorded in this registry build)")
        return lines
    for citation in citations:
        if isinstance(citation, (tuple, list)) and len(citation) >= 2:
            lines.append(f"    {citation[0]} — {citation[1]}")
        else:
            lines.append(f"    {citation}")
    return lines


# ---------------------------------------------------------------------------
# --call NAME=VALUE (direct tool addressing): parse into planner
# ops, with two CLI-side pre-checks that produce error entries WITHOUT
# calling the planner for that op (the planner only knows bool/enum/number).
# ---------------------------------------------------------------------------


def _parse_call_value(value_text: str) -> Any:
    """VALUE is parsed as JSON (true/false, numbers, quoted strings); a
    bare unquoted token that fails JSON parsing is a plain string."""
    try:
        return json.loads(value_text)
    except ValueError:
        return value_text


def _call_error_entry(spec: Optional[ToolSpec], name: str, raw: str,
                      error: str) -> Dict[str, Any]:
    """A planner-shaped error entry synthesized by the CLI (same keys as
    planner._error_entry) for ops the planner must not be asked to resolve."""
    return {
        "tool": spec.name if spec is not None else name,
        "path": spec.path if spec is not None else "",
        "action": "set",
        "raw": raw,
        "old": None,
        "new": None,  # an errored entry is never applied
        "clamped": False,
        "no_op": False,
        "note": None,
        "error": error,
    }


def _compose_call_ops(calls: List[str]) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """Turn --call NAME=VALUE flags into planner ops.

    Returns (ops, cli_errors): ``ops`` holds the ops the planner should
    resolve (in flag order); ``cli_errors`` maps each rejected flag's
    position to a ready-made error entry. Raises ValueError for malformed
    flags (no '=' / empty name) — a usage error, exit 2.
    """
    ops: List[Dict[str, Any]] = []
    cli_errors: Dict[int, Dict[str, Any]] = {}
    for position, call in enumerate(calls):
        name, sep, value_text = call.partition("=")
        if not sep or not name:
            raise ValueError(f"--call expects NAME=VALUE (got {call!r})")
        value = _parse_call_value(value_text)
        op = {"tool": name, "action": "set", "value": value, "raw": call}
        spec = tool_by_name(name)
        if spec is None:
            # Unknown tool: the planner's existing unknown-tool error entry
            # (it renders through the same REJECTED path).
            ops.append(op)
            continue
        if getattr(spec, "kind", "") == "string":
            # The planner validates bool/enum/number values only; string
            # tools are rejected here, before the planner sees the op.
            cli_errors[position] = _call_error_entry(
                spec, name, call, CALL_STRING_TOOLS_MESSAGE)
            continue
        if isinstance(value, float) and not math.isfinite(value):
            # json.loads accepts NaN/Infinity; the planner's range checks
            # cannot reject them (all comparisons with NaN are False), so
            # they are stopped here — they can never be written as JSON.
            cli_errors[position] = _call_error_entry(
                spec, name, call,
                f"value {value_text!r} is not a finite number; "
                "NaN/Infinity are not valid JSON values",
            )
            continue
        ops.append(op)
    return ops, cli_errors


def _splice_call_errors(plan: Dict[str, Any],
                        cli_errors: Dict[int, Dict[str, Any]], total: int) -> None:
    """Merge the CLI-synthesized error entries back into the plan at their
    original --call positions, extend the plan's error list and mark it
    apply_blocked (all-or-nothing, §4.5)."""
    if not cli_errors:
        return
    merged: List[Dict[str, Any]] = []
    planner_entries = plan.get("entries", [])
    planner_index = 0
    for position in range(total):
        if position in cli_errors:
            merged.append(cli_errors[position])
        else:
            merged.append(planner_entries[planner_index])
            planner_index += 1
    plan["entries"] = merged
    errors = plan.setdefault("errors", [])
    for entry in cli_errors.values():
        errors.append(
            {"tool": entry["tool"], "path": entry["path"], "error": entry["error"]}
        )
    plan["apply_blocked"] = True


# ---------------------------------------------------------------------------
# Argument parsing and dispatch.
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "caelestia assistant settings layer (#120): natural-language request "
            "-> validated tool calls on the shell's shell.json. Dry-run by default; the only "
            "writes are the target file behind --apply plus its .assistant-backup sibling."
        ),
        epilog="Nothing is ever executed; suggested commands are inert strings.",
    )
    arg_parser.add_argument(
        "text", nargs="?", default=None,
        help='the request, e.g. "make the bar thinner" (required unless '
             "--list-tools/--tool/--restore/--call)",
    )
    gate = arg_parser.add_mutually_exclusive_group()
    gate.add_argument(
        "--apply", action="store_true",
        help="actually write the planned changes (default: dry-run, nothing written)",
    )
    gate.add_argument(
        "--restore", action="store_true",
        help="undo: put the single .assistant-backup slot back (one level of undo)",
    )
    arg_parser.add_argument(
        "--file", metavar="PATH", default=None,
        help=f"target shell.json override (default: {default_target()})",
    )
    arg_parser.add_argument(
        "--json", action="store_true", help="emit the machine-readable plan/verdict as JSON"
    )
    arg_parser.add_argument(
        "--list-tools", action="store_true",
        help="print the tool registry table (every tool with path, type, range) and exit",
    )
    arg_parser.add_argument(
        "--group", metavar="SLUG", default=None,
        help="feature-area filter for --list-tools (default: all groups)",
    )
    arg_parser.add_argument(
        "--tool", metavar="NAME", default=None,
        help="print a detail card for one tool (path, validation, citations) and exit",
    )
    arg_parser.add_argument(
        "--call", action="append", dest="calls", metavar="NAME=VALUE",
        help="direct tool call: NAME is a registry tool name, VALUE is JSON "
             "(true/false, number, quoted string; a bare token that fails JSON "
             "parsing is a plain string); repeatable — multiple --call flags "
             "compose one multi-op plan",
    )
    arg_parser.add_argument(
        "--explain", metavar="QUERY", default=None,
        help="read-only: explain a setting's current state (a dotted path, a "
             "tool name, or a plain-words question like 'why is my dock "
             "blurry'); never writes",
    )
    arg_parser.add_argument(
        "--history", action="store_true",
        help="read-only: list the bounded undo history (newest first) and exit",
    )
    arg_parser.add_argument(
        "--prefer", nargs=2, metavar=("PRESET_A", "PRESET_B"), default=None,
        help="opt-in pairwise comparison: dry-run previews of both presets "
             "side by side, then record which you pick (feeds the Elo/"
             "Bradley-Terry ladder; writes one comparison row to the "
             "assistant state, never to shell.json)",
    )
    arg_parser.add_argument(
        "--threeway", metavar="TEXT", default=None,
        help="read-only three-way diff view: for each setting the "
             "request (or preset) would touch — the file's current "
             "value, the proposal's value, and the value the undo "
             "history would restore — so 'apply then undo' is never a "
             "surprise; never writes",
    )
    arg_parser.add_argument(
        "--what-if", metavar="TEXT", default=None,
        help="read-only consequence view: plan the request (or a preset "
             "name) as a DRY RUN, then project it through the cited "
             "interaction table (derived effects, conflicts, "
             "reversibility); never writes, never applies",
    )
    arg_parser.add_argument(
        "--why-chain", metavar="KEY", default=None,
        help="read-only causal attribution (F14): the backward chain "
             "of ARMED, cited interactions explaining the current "
             "value of a registry path or tool — with the live state "
             "read from the target file; unevaluable edges are listed, "
             "never guessed",
    )
    arg_parser.add_argument(
        "--counterfactual", nargs=2, metavar=("TOOL", "VALUE"),
        default=None,
        help="read-only forward counterfactual (F14): 'if TOOL were "
             "VALUE' — the cited immediate projection plus every "
             "curated edge that could not be evaluated against the "
             "live state (abstained, not dropped)",
    )
    arg_parser.add_argument(
        "--accessibility-check", nargs=2, metavar=("FG", "BG"),
        default=None,
        help="read-only WCAG 2.x contrast verdicts (F22) for an "
             "fg/bg pair of #rrggbb colors, with the Machado (2009) "
             "CVD simulation of each color under --cvd/--severity; "
             "pure math, never writes",
    )
    arg_parser.add_argument(
        "--cvd", default=None,
        choices=["protanopia", "deuteranopia", "tritanopia"],
        help="accessibility-check: simulate this color-vision deficiency",
    )
    arg_parser.add_argument(
        "--severity", type=float, default=1.0,
        help="accessibility-check: CVD severity in [0, 1] (default 1.0; "
             "partial severities are the documented linear blend)",
    )
    arg_parser.add_argument(
        "--recommend", metavar="TOOL", default=None,
        help="read-only value recommendation (F29) for a numeric tool: "
             "hierarchical partial pooling over the registry default, "
             "the preset pool and your approved history, with a "
             "95-percent "
             "credible interval and the Pareto position against the "
             "optimization profiles; a proposal, never applied",
    )
    arg_parser.add_argument(
        "--rank", action="store_true",
        help="read-only: print the learned pairwise preference ladder for "
             "presets (Elo + Bradley-Terry, with comparison counts)",
    )
    arg_parser.add_argument(
        "--undo", nargs="?", const=1, default=None, type=int, metavar="STEPS",
        help="undo the newest STEPS applies (default 1) from the bounded "
             "history; writes only the reverted values",
    )
    arg_parser.add_argument(
        "--undo-id", type=int, default=None, metavar="ID",
        help="revert one specific history entry (see --history) — the "
             "'restore the theme I had yesterday' operation",
    )
    arg_parser.add_argument(
        "--preset", metavar="NAME", default=None,
        help="apply a named preset (compact, minimal, gaming, battery-saver, "
             "macos-like) — a bundle of validated tool calls; multi-change, "
             "so it always shows the preview and needs --confirm/--apply",
    )
    arg_parser.add_argument(
        "--list-presets", action="store_true",
        help="print the preset table (name, description, bundled calls) and exit",
    )
    arg_parser.add_argument(
        "--confirm", action="store_true",
        help="second consent for multi-change applies: without it (and without "
             "an interactive yes), a plan touching more than one setting is "
             "previewed but NOT written",
    )
    arg_parser.add_argument(
        "--lint", action="store_true",
        help="read-only config health lint of the target shell.json (unknown "
             "keys, out-of-range or mistyped values, silent no-ops, inert "
             "customizations); never writes",
    )
    arg_parser.add_argument(
        "--audit-env", action="store_true",
        help="read-only environment audit (F21): config lint, cited "
             "interaction INERT states, snapshot freshness (F17), "
             "per-screen overrides (F20) and backup coverage in one "
             "report; exit 1 on CRITICAL findings",
    )
    arg_parser.add_argument(
        "--wallpaper-palette", metavar="PATH", default=None,
        help="read-only: derive a WCAG-checked accent candidate from a "
             "wallpaper PNG (k-means in OKLab) and surface it through the "
             "same inert scheme-suggestion path as accent-color requests",
    )
    arg_parser.add_argument(
        "--wizard", action="store_true",
        help="first-run setup wizard: pairwise tradeoff questions -> AHP "
             "weights -> TOPSIS over the shipped presets -> a starting-preset "
             "recommendation (writes nothing; apply via --preset yourself)",
    )
    arg_parser.add_argument(
        "--answers", metavar="N,N,N,N,N,N", default=None,
        help="non-interactive wizard answers, one 1-5 intensity per "
             "question, in the order --wizard prints them",
    )
    gate2 = arg_parser.add_mutually_exclusive_group()
    gate2.add_argument(
        "--macro", metavar="NAME", default=None,
        help="replay a saved macro (an APPROVED proposal sequence "
             "captured earlier): dry-run plan by default; with --apply "
             "it STILL needs --confirm or the interactive prompt — "
             "every replay is re-approved, single-change macros "
             "included",
    )
    gate2.add_argument(
        "--macro-save", metavar="NAME", default=None,
        help="capture the newest applied change (or --from-id ID, see "
             "--history) as a named macro — a record of something you "
             "already approved, stored in the history file's bounded "
             "macro list (never the target file)",
    )
    gate2.add_argument(
        "--macro-list", action="store_true",
        help="read-only: list the saved macros and exit",
    )
    gate2.add_argument(
        "--macro-delete", metavar="NAME", default=None,
        help="delete one saved macro by name (an explicit command; "
             "never a silent overwrite)",
    )
    gate3 = arg_parser.add_mutually_exclusive_group()
    gate3.add_argument(
        "--profile", metavar="NAME", default=None,
        help="apply a saved profile (F9: a named composition of "
             "presets, macros and direct calls, later source wins per "
             "tool with conflicts reported): dry-run plan by default; "
             "with --apply it needs --confirm like any multi-change "
             "bundle",
    )
    gate3.add_argument(
        "--profile-save", nargs=2, metavar=("NAME", "SRC"), default=None,
        help="save a profile: NAME plus the first source; repeat "
             "--also SRC for more. A source is preset:NAME, macro:NAME, "
             "or TOOL=VALUE; sources compose in order, later wins per "
             "tool (conflicts reported, never dropped)",
    )
    gate3.add_argument(
        "--profile-list", action="store_true",
        help="read-only: list the saved profiles and exit",
    )
    gate3.add_argument(
        "--profile-show", metavar="NAME", default=None,
        help="read-only: one profile's sources, composed ops and the "
             "conflicts their order resolved; never writes",
    )
    gate3.add_argument(
        "--profile-diff", nargs=2, metavar=("A", "B"), default=None,
        help="read-only: effective-value diff of two saved profiles "
             "(only A / only B / changed per tool)",
    )
    gate3.add_argument(
        "--profile-delete", metavar="NAME", default=None,
        help="delete one saved profile by name (an explicit command)",
    )
    gate3.add_argument(
        "--profile-whatif", metavar="NAME", default=None,
        help="read-only composite preview (F11): per-source consequences, "
             "the conflicts the source order resolved, the blast radius by "
             "feature area, and the projected-state sanity verdicts the "
             "applier itself would enforce — before any consent",
    )
    arg_parser.add_argument(
        "--also", action="append", dest="also_sources",
        metavar="SRC", default=None,
        help="with --profile-save: an additional source (see "
             "--profile-save); repeatable, order matters",
    )
    gate4 = arg_parser.add_mutually_exclusive_group()
    gate4.add_argument(
        "--env-save", metavar="NAME", default=None,
        help="snapshot the current environment (shell.json content, "
             "hash-pinned; macros/profiles as metadata) into the "
             "history file's bounded environment list",
    )
    gate4.add_argument(
        "--env-list", action="store_true",
        help="read-only: list the saved environments and exit",
    )
    gate4.add_argument(
        "--env-show", metavar="NAME", default=None,
        help="read-only: one environment's metadata + integrity "
             "verification (with --env-verify, default on) and per-area "
             "summary",
    )
    gate4.add_argument(
        "--env-diff", metavar="NAME", default=None,
        help="read-only: what RESTORING NAME would change — registry "
             "paths set back to snapshot values / reset to defaults, "
             "unknown keys REPORTED and never touched",
    )
    gate4.add_argument(
        "--env-restore", metavar="NAME", default=None,
        help="restore environment NAME: the diff becomes a normal "
             "planned multi-change apply (dry-run by default; --apply "
             "still needs --confirm)",
    )
    gate4.add_argument(
        "--env-delete", metavar="NAME", default=None,
        help="delete one saved environment by name (an explicit command)",
    )
    gate4.add_argument(
        "--env-export", metavar="NAME", default=None,
        help="export a verified environment as a portable bundle (with "
             "--to PATH); nothing else is written",
    )
    gate4.add_argument(
        "--env-import", metavar="FROM_JSON", default=None,
        help="import a bundle as a NEW snapshot, REVIEW-ONLY (with --as "
             "NAME): schema/hash validated, drift measured, nothing "
             "applied — restore stays the explicit F17 path",
    )
    arg_parser.add_argument(
        "--monitor", metavar="SCREEN", default=None,
        help="F20: target ONE screen's override file "
             "(monitors/SCREEN/shell.json, the upstream forScreen layer — "
             "same schema, same validation, same gates) instead of the "
             "global config; combine with a request or --call; cannot be "
             "combined with --file",
    )
    arg_parser.add_argument(
        "--monitors", action="store_true",
        help="read-only monitor report (F20): the discovered per-screen "
             "override files and the global tools that select monitor "
             "behavior, with citations; never writes",
    )
    gate5 = arg_parser.add_mutually_exclusive_group()
    gate5.add_argument(
        "--gen-catalog", action="store_true",
        help="community docs generator (F23): (re)write the generated "
             "the generated tool catalog (`settings --gen-catalog`) and bash completion from the frozen "
             "registry; with --verify it only CHECKS byte-identity (CI "
             "safe)",
    )
    arg_parser.add_argument(
        "--verify", dest="gen_verify", action="store_true",
        help="with --gen-catalog: verify the committed artifacts match a "
             "fresh generation instead of rewriting them",
    )
    arg_parser.add_argument(
        "--plain", action="store_true",
        help="accessible output (F24): one fact per line, no tables or "
             "decorations — also CAELESTIA_ASSISTANT_PLAIN=1; the "
             "contract is test-enforced (no ANSI, no box-drawing, "
             "stable order)",
    )
    arg_parser.add_argument(
        "--a11y-recipe", action="store_true",
        help="read-only reduced-motion recipe (F24): registry-validated "
             "knobs composed into an inert one-line --profile-save "
             "command; nothing is saved or applied by printing it",
    )
    arg_parser.add_argument(
        "--to", metavar="PATH", default=None,
        help="with --env-export: the bundle output path",
    )
    arg_parser.add_argument(
        "--as", dest="as_name", metavar="NAME", default=None,
        help="with --env-import: the local name for the imported "
             "environment",
    )
    arg_parser.add_argument(
        "--from-id", type=int, default=None, metavar="ID",
        help="with --macro-save: capture this history entry id instead "
             "of the newest (see --history)",
    )
    return arg_parser


def _preview_block(plan: Dict[str, Any], label: str) -> List[str]:
    """The issue #120 preview, verbatim in shape:
    "I found several changes that match your request:" + one line per
    change + "Apply these changes?"."""
    lines = ["I found several changes that match your request"
             + (f" ({label})" if label else "") + ":"]
    for entry in plan.get("entries", []):
        if entry.get("error") or entry.get("no_op"):
            continue
        lines.append(f"- {entry.get('path')}: "
                     f"{entry.get('old', '(unset)')} -> {entry.get('new')}")
    lines.append("Apply these changes?")
    return lines


def _confirm_multi(plan: Dict[str, Any], args: Any) -> Tuple[bool, str]:
    """The multi-change confirmation gate (issue #120).

    --confirm: explicit second consent. Interactive tty: the preview plus a
    y/N prompt. Otherwise: refusal with the re-run hint. Never guesses."""
    label = getattr(args, "text", None) or ""
    if getattr(args, "confirm", False):
        return True, ""
    if sys.stdin is not None and sys.stdin.isatty():
        print("\n".join(_preview_block(plan, label)))
        try:
            answer = input("Apply these changes? [y/N] ").strip().lower()
        except EOFError:
            answer = ""
        if answer in ("y", "yes"):
            return True, ""
        return False, "not confirmed; nothing was written"
    return False, (
        "multi-change plans need confirmation: re-run with --confirm "
        "(or answer the prompt interactively); nothing was written"
    )


def main(argv: Optional[List[str]] = None) -> int:
    arg_parser = _build_arg_parser()
    args = arg_parser.parse_args(argv)

    if args.calls is not None and args.text is not None:
        arg_parser.error(
            "--call and a request sentence are mutually exclusive: address tools "
            "directly with --call, or describe the change in plain words"
        )
    if args.group is not None and not args.list_tools:
        arg_parser.error("--group is only valid together with --list-tools")
    if args.undo is not None and args.undo_id is not None:
        arg_parser.error("--undo and --undo-id are mutually exclusive")
    if args.from_id is not None and args.macro_save is None:
        arg_parser.error("--from-id is only valid together with --macro-save")

    if args.list_presets:
        from . import presets as presets_mod
        print("\n".join(presets_mod.describe_lines()))
        return 0

    if args.preset is not None:
        # A preset is a bundle of validated tool calls riding the ordinary
        # plan path — never a bespoke writer.
        from . import presets as presets_mod
        try:
            ops = presets_mod.preset_ops(args.preset)
        except presets_mod.PresetError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        target = Path(args.file) if args.file else default_target()
        try:
            plan = planner.plan(ops, target)
        except planner.PlannerError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if not args.apply:
            print("\n".join(render_plan(plan, [], f"preset: {args.preset}",
                                       args.file, False, None, None)))
            return 0
        confirmed, why = _confirm_multi(plan, args)
        if not confirmed:
            print(why, file=sys.stderr)
            return 1
        try:
            apply_result = applier.apply(
                plan, target, write=True, label=f"preset: {args.preset}")
        except applier.ApplierError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print("\n".join(render_plan(plan, [], f"preset: {args.preset}",
                                   args.file, True, apply_result, None)))
        return 0

    if args.threeway is not None:
        # exponential-build-3 G1: the three-way diff view — current
        # file value vs the current proposal vs the undo-log restore
        # value, per touched key. Read-only: the parser/planner spine
        # runs dry (plan only) and the history is only READ.
        from . import history as history_mod

        request = args.threeway
        try:
            parse = parser.parse(request)
            ops = parse.get("ops") or []
        except Exception:
            ops = []
        if not ops:
            from . import presets as presets_mod
            try:
                ops = presets_mod.preset_ops(request)
            except Exception:
                ops = []
        if not ops:
            print(f"error: threeway could not plan {request!r} (use a "
                  "request or a preset name)", file=sys.stderr)
            return 1
        target = Path(args.file) if args.file else default_target()
        try:
            plan = planner.plan(ops, target)
        except planner.PlannerError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        entries = history_mod.entries(target)  # newest first
        print("\n".join(_header(False)))
        print("")
        print(f"three-way view for {request!r} (read-only; nothing "
              "is written):")
        shown = 0
        for entry in plan.get("entries", []):
            if entry.get("error") or entry.get("no_op"):
                continue
            path = str(entry.get("path"))
            current = entry.get("old", "(unset)")
            proposed = entry.get("new")
            record = next((e for e in entries
                           if any(op.get("path") == path
                                  for op in e.get("ops", []))), None)
            print(f"  {path}:")
            print(f"    now (file):          {current!r}")
            print(f"    proposal:            {proposed!r}")
            if record is not None:
                op = next(op for op in record.get("ops", [])
                          if op.get("path") == path)
                restore = op.get("old")
                label = record.get("label") or "(unlabelled)"
                print(f"    undo would restore:  {restore!r}   "
                      f"(history #{record.get('id')} {label!r}, "
                      f"{record.get('at', '?')})")
                if restore != current:
                    print("    note: apply-then-undo lands on the "
                          "restored value, NOT back on the current "
                          "one")
                else:
                    print("    note: apply-then-undo returns exactly "
                          "to the current value")
            else:
                print("    undo record:         none for this key "
                      "(first change would create one)")
            shown += 1
        if not shown:
            print("  the request resolves to no applicable changes")
        return 0

    if args.why_chain is not None:
        # F14: backward causal attribution — read-only, cited, live-state
        from . import causal as causal_mod
        from . import planner as chain_planner

        target = Path(args.file) if args.file else default_target()
        current: Dict[str, Any] = {}
        if target.exists():
            try:
                current, _notes = chain_planner._read_current(target)
            except Exception:
                current = {}
        result = causal_mod.why_chain(args.why_chain, current)
        print("\n".join(_header(False)))
        print("")
        print("\n".join(causal_mod.render_chain(result)))
        if args.json:
            print(json.dumps(result, sort_keys=True, indent=1))
        return 0 if result.get("verdict") == "OK" else 1

    if args.counterfactual is not None:
        # F14: forward counterfactual — the what-if projection for one
        # explicit tool/value pair, with the unevaluated-edge honesty
        from . import causal as causal_mod
        from . import planner as cf_planner

        tool_name, raw_value = args.counterfactual
        parsed_value: Any
        if raw_value == "true":
            parsed_value = True
        elif raw_value == "false":
            parsed_value = False
        elif re.fullmatch(r"-?\d+", raw_value):
            parsed_value = int(raw_value)
        elif re.fullmatch(r"-?\d+\.\d+", raw_value):
            parsed_value = float(raw_value)
        else:
            parsed_value = raw_value
        target = Path(args.file) if args.file else default_target()
        current: Dict[str, Any] = {}
        if target.exists():
            try:
                current, _notes = cf_planner._read_current(target)
            except Exception:
                current = {}
        result = causal_mod.counterfactual(tool_name, parsed_value, current)
        print("\n".join(_header(False)))
        print("")
        print("\n".join(causal_mod.render_counterfactual(result)))
        if args.json:
            print(json.dumps(result, sort_keys=True, indent=1))
        return 0 if result.get("verdict") == "OK" else 1

    if args.recommend is not None:
        # F29: the value recommender — read-only, pooled, honest
        from . import recommend as recommend_mod

        ledger = None
        target = Path(args.file) if args.file else default_target()
        hist = target.parent / (target.name + ".assistant-history.json")
        if hist.exists():
            ledger = str(hist)
        result = recommend_mod.recommend(args.recommend, ledger_path=ledger)
        print("\n".join(_header(False)))
        print("")
        print("\n".join(recommend_mod.render_recommendation(result)))
        if args.json:
            print(json.dumps(result, sort_keys=True, indent=1))
        return 0 if result.get("verdict") == "OK" else 1

    if args.accessibility_check is not None:
        # F22: WCAG 2.x contrast verdicts + optional Machado CVD
        # simulation — pure published math, read-only.
        from . import accessibility as ax

        fg_hex, bg_hex = args.accessibility_check
        try:
            fg, bg = ax.hex_to_rgb(fg_hex), ax.hex_to_rgb(bg_hex)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print("\n".join(_header(False)))
        print("")
        print(f"contrast {fg_hex} on {bg_hex}: "
              f"{ax.contrast_ratio(fg, bg):.2f}:1")
        for f in ax.wcag_findings(fg, bg):
            mark = "pass" if f["pass"] else "FAIL"
            print(f"  [{mark}] {f['level']}: {f['ratio']}:1 "
                  f"(threshold {f['threshold']})")
        if args.cvd:
            sev = args.severity
            if not 0.0 <= sev <= 1.0:
                print("error: --severity must be in [0, 1]",
                      file=sys.stderr)
                return 2
            sim_fg = ax.rgb_to_hex(ax.simulate_cvd(fg, args.cvd, sev))
            sim_bg = ax.rgb_to_hex(ax.simulate_cvd(bg, args.cvd, sev))
            ratio = ax.contrast_ratio(
                ax.simulate_cvd(fg, args.cvd, sev),
                ax.simulate_cvd(bg, args.cvd, sev))
            print(f"  under {args.cvd} (severity {sev}): {sim_fg} on "
                  f"{sim_bg}, contrast {ratio:.2f}:1")
        print(f"  cite: {ax.WCAG_CITATION}")
        if args.cvd:
            print(f"  cite: {ax.SIMULATE_CITATION}")
        if args.json:
            import json as _json
            print(_json.dumps({
                "fg": fg_hex, "bg": bg_hex,
                "ratio": round(ax.contrast_ratio(fg, bg), 4),
                "findings": ax.wcag_findings(fg, bg),
                "cvd": ({"kind": args.cvd, "severity": args.severity,
                         "fg": ax.rgb_to_hex(ax.simulate_cvd(fg, args.cvd, args.severity)),
                         "bg": ax.rgb_to_hex(ax.simulate_cvd(bg, args.cvd, args.severity))}
                        if args.cvd else None),
            }, sort_keys=True, indent=1))
        return 0

    if args.what_if is not None:
        # Phase 2.7: the what-if consequence view — a DRY-RUN plan plus
        # the projection through the cited interaction table. Read-only:
        # the parser/planner/applier spine is untouched and this surface
        # can never apply anything.
        from . import consequences as consequences_mod
        from . import parser as whatif_parser

        request = args.what_if
        try:
            parse = whatif_parser.parse(request)
            ops = parse.get("ops") or []
        except Exception:
            ops = []
        if not ops:
            # preset name? plan its bundle as the candidate list
            from . import presets as presets_mod

            try:
                ops = presets_mod.preset_ops(request)
            except Exception:
                ops = []
        if not ops:
            print(f"error: what-if could not plan {request!r} (use a "
                  "request or a preset name)", file=sys.stderr)
            return 1
        target = Path(args.file) if args.file else default_target()
        current: Dict[str, Any] = {}
        if target.exists():
            try:
                current, _notes = planner._read_current(target)
            except Exception:
                current = {}
        # Resolve the ops through the standard planner ONCE: step and
        # multiply ops become the absolute values they produce. The
        # projection runs over the RESOLVED entries — projecting a raw
        # step delta (-1) as an absolute value fires false edges.
        resolved_ops: List[Dict[str, Any]] = []
        try:
            plan = planner.plan(ops, target)
            resolved_ops = [{"tool": e.get("tool"), "value": e.get("new")}
                            for e in plan.get("entries", [])
                            if not e.get("error")]
        except planner.PlannerError as exc:
            plan = None
            print(f"note: planner refused: {exc} (projection below is "
                  "the honest view of the requested ops)")
        projection = consequences_mod.project(
            resolved_ops or ops, current=current)
        print("\n".join(_header(False)))
        print("")
        print("\n".join(consequences_mod.render(projection)))
        # F14 honesty extension: curated edges that could not be
        # evaluated against the live state are listed, not dropped
        from . import causal as whatif_causal

        for u in whatif_causal.unevaluated_edges(current):
            print(f"  unevaluated: edge {u['edge']}: {u['reason']}")
        print("")
        if plan is not None:
            print("\n".join(render_plan(plan, [],
                                         f"what-if: {request}", args.file,
                                         False, None, None)))
        return 0

    if args.rank:
        # Phase 2.4: the learned pairwise ladder — read-only, over the
        # comparison rows --prefer records. The shared ranking primitive
        # (brain/ranking.py) is the same one agent plan comparisons use.
        from assistant.capabilities.brain import ranking as ranking_mod
        from assistant.capabilities.brain import state as brain_state

        state = brain_state.load()
        rows = state.get("preset_comparisons") or []
        # exponential-build-3 G2: surface a pending idle-time invitation
        # BEFORE the empty-log early return — a fresh install's first
        # invitation must be visible even with zero comparisons
        from assistant.capabilities.brain import preference_prompts
        invite = preference_prompts.pending(state)
        if not rows:
            if invite:
                print("no pairwise comparisons recorded yet — but an "
                      "idle-time invitation is waiting:")
                print(f"  settings --prefer {invite.get('a')} "
                      f"{invite.get('b')}  ({invite.get('reason')}; "
                      f"invited {invite.get('at')})")
                return 0
            print("no pairwise comparisons recorded yet; record one with "
                  "settings --prefer PRESET_A PRESET_B")
            return 0
        pairs = [(row.get("winner"), row.get("loser")) for row in rows
                 if row.get("winner") and row.get("loser")]
        report = ranking_mod.ladder_report(pairs)
        print("preset preference ladder (pairwise, learned from your "
              "explicit comparisons):")
        for row in report["ladder"]:
            conf = (f"{row['comparisons']} comparisons"
                    if row["enough_data"] else "not enough data")
            print(f"  {row['rating']:8.1f}  {row['item']:<16} ({conf})")
        tau = report["agreement_kendall_tau"]
        if tau is not None:
            print(f"elo/bradley-terry order agreement (kendall tau): {tau}")
        # exponential-build-3 G2: surface a pending idle-time invitation
        if invite:
            print(f"idle-time invitation pending: settings --prefer "
                  f"{invite.get('a')} {invite.get('b')}  "
                  f"({invite.get('reason')}; invited {invite.get('at')})")
        print("proposals only — nothing is applied; the planner/applier "
              "gates are untouched")
        return 0

    if args.prefer:
        # Phase 2.4: ONE explicit pairwise comparison (the proposal's
        # opt-in design — each pair is a deliberate choice, no batch).
        from assistant.capabilities.brain import state as brain_state
        from . import presets as presets_mod

        name_a, name_b = args.prefer
        known = [str(p["name"]) for p in presets_mod.presets()]
        target = Path(args.file) if args.file else default_target()
        for name in (name_a, name_b):
            if name not in known:
                print(f"error: unknown preset {name!r}; known: "
                      f"{', '.join(known)}", file=sys.stderr)
                return 1
        print("\n".join(_header(False)))
        print("")
        for index, name in enumerate((name_a, name_b), 1):
            print(f"[{index}] preset {name} (dry-run, nothing written):")
            try:
                plan = planner.plan(presets_mod.preset_ops(name), target)
            except planner.PlannerError as exc:
                print(f"  planner refused: {exc}")
                continue
            print("\n".join("  " + line for line in render_plan(
                plan, [], f"prefer: {name}", args.file, False, None, None)))
        try:
            pick = input(f"which do you prefer? [1={name_a} / 2={name_b} "
                         "/ s=skip] ").strip().lower()
        except EOFError:
            pick = ""
        if pick not in ("1", "2"):
            print("no comparison recorded (skip)")
            return 0
        winner = name_a if pick == "1" else name_b
        loser = name_b if pick == "1" else name_a
        state = brain_state.load()
        rows = state.setdefault("preset_comparisons", [])
        if not isinstance(rows, list):
            rows = []
        from datetime import datetime
        rows.append({"winner": winner, "loser": loser,
                     "at": datetime.now().isoformat(timespec="seconds")})
        state["preset_comparisons"] = rows[-200:]
        # exponential-build-3 G2: if this comparison answers a pending
        # idle-time invitation, consume it (either order counts)
        from assistant.capabilities.brain import preference_prompts
        consumed = preference_prompts.answer(state, name_a, name_b)
        brain_state.save(state)
        suffix = " (this answers the pending idle-time prompt)" \
            if consumed else ""
        print(f"recorded: {winner} > {loser} (see settings --rank){suffix}")
        return 0

    if args.lint:
        # Read-only config health lint (issue #120 Phase 1.1). Never writes;
        # findings are advisory — the applier/planner remain the only gate.
        from . import lint as lint_mod
        target = Path(args.file) if args.file else default_target()
        try:
            findings = lint_mod.lint_file(target)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print("\n".join(_header(False)))
        print("")
        print("\n".join(lint_mod.render_findings(findings)))
        return 0

    if args.wallpaper_palette is not None:
        # Issue #120 Phase 2.4: wallpaper -> candidate accent, surfaced
        # through the EXISTING inert scheme-suggestion mechanism (same
        # constants, same renderer, same SUGGESTED_NOT_EXECUTED prefix).
        from . import parser as parser_mod
        from assistant.capabilities.genius import palette_extract as palette_mod
        try:
            data = Path(args.wallpaper_palette).read_bytes()
            result = palette_mod.accent_from_png(data)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        suggestions = list(parser_mod.SCHEME_SUGGESTIONS) + \
            list(parser_mod.WALLPAPER_SUGGESTIONS)
        notes = [
            (f"derived from {args.wallpaper_palette}: dominant accent-like "
             f"color {result['accent']} (k-means quantization in OKLab over "
             f"{result['n_pixels']} sampled pixels; {result['selection']})"),
            (f"WCAG contrast of the derived accent: on white "
             f"{result['contrast']['on_white']['ratio']}:1, on black "
             f"{result['contrast']['on_black']['ratio']}:1 — the scheme "
             "system owns the actual change"),
            parser_mod.SCHEME_NOTE,
            "nothing was executed and nothing was written; these are inert suggestions only",
        ]
        print("\n".join(render_verdict(
            {"verdict": "SUGGESTED", "suggestions": suggestions, "notes": notes})))
        return 0

    if args.wizard:
        # Issue #120 Phase 3 "setup wizards": pure AHP+TOPSIS over the
        # shipped presets; renders a recommendation and writes nothing.
        from . import wizard as wizard_mod
        answers = None
        if args.answers:
            try:
                answers = [int(x) for x in args.answers.split(",")]
            except ValueError:
                print("error: --answers must be comma-separated integers",
                      file=sys.stderr)
                return 2
        elif args.text:
            answers = [int(x) for x in args.text.split(",")]
        if answers is not None:
            try:
                result = wizard_mod.run(answers)
            except ValueError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            print("\n".join(wizard_mod.render(result)))
            return 0
        return wizard_mod.mainish()

    if args.explain is not None:
        # Read-only explainability: never writes, never plans.
        from . import explain as explain_mod
        target = Path(args.file) if args.file else default_target()
        try:
            result = explain_mod.explain(args.explain, target)
        except explain_mod.ExplainError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print("\n".join(_header(False)))
        print("")
        print(result["answer"])
        if result["cites"]:
            print("grounded in: " + "; ".join(result["cites"]))
        if not result["rule"]:
            print(
                "(value readback — no causal rule fired for this key; rules "
                "needing live session state do not fire from the CLI)"
            )
        # Multi-hop provenance (Phase 1.3): best-effort and read-only —
        # missing/unreadable ledger or state files simply shorten the chain.
        try:
            from assistant.capabilities.brain.cli import DEFAULT_LEDGER
            from assistant.capabilities.brain import state as brain_state
            ledger_path = DEFAULT_LEDGER if Path(DEFAULT_LEDGER).exists() else None
            bandit_state = None
            state_file = Path(brain_state.DEFAULT_STATE)
            if state_file.exists():
                try:
                    bandit_state = brain_state.load().get("preset_bandit")
                except (OSError, ValueError):
                    bandit_state = None
            if ledger_path is not None or bandit_state is not None:
                prov = explain_mod.provenance(
                    args.explain, target, ledger_path=ledger_path,
                    bandit_state=bandit_state)
                if prov["chain"]:
                    print("")
                    print("\n".join(explain_mod.render_provenance(prov)))
        except ImportError:
            pass  # brain layer unavailable: value readback alone still answers
        return 0

    if args.history:
        from . import history as history_mod
        target = Path(args.file) if args.file else default_target()
        try:
            entries = history_mod.entries(target)
        except history_mod.HistoryError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print("\n".join(_header(False)))
        print("")
        if not entries:
            print("the undo history is empty (nothing has been applied yet)")
        else:
            print(f"undo history (newest first, {len(entries)} of "
                  f"{history_mod.MAX_ENTRIES} slots):")
            for e in entries:
                label = e.get("label") or "(unlabelled)"
                print(f"  #{e['id']}  {e['at']}  {label}  "
                      f"({len(e.get('ops', []))} change(s))")
            print("")
            print("undo with --undo (newest N) or --undo-id ID; capture "
                  "one as a replayable macro with --macro-save NAME "
                  "(newest) or --from-id ID")
        return 0

    if args.undo is not None or args.undo_id is not None:
        from . import history as history_mod
        target = Path(args.file) if args.file else default_target()
        try:
            if args.undo_id is not None:
                result = history_mod.undo_by_id(target, args.undo_id)
            else:
                result = history_mod.undo(target, args.undo)
        except (history_mod.HistoryError, applier.ApplierError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print("\n".join(_header(False)))
        print("")
        print(result.get("message", ""))
        return 0 if result.get("restored", True) else 1

    if args.macro is not None:
        # C3: replay a captured APPROVED proposal sequence. The ops are
        # plain planner ops (presets.preset_ops shape), so the replay
        # rides the ordinary plan/render path; the ONLY bespoke rule is
        # STRONGER consent — every macro apply needs the second consent
        # (--confirm or interactive y/N), single-change macros included,
        # because a replay's contents may no longer be in the user's head.
        from . import history as _history
        from . import macros as macros_mod
        target = Path(args.file) if args.file else default_target()
        try:
            ops = macros_mod.macro_ops(target, args.macro)
        except (macros_mod.MacroError, _history.HistoryError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        try:
            plan = planner.plan(ops, target)
        except planner.PlannerError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        label = f"macro: {args.macro}"
        if not args.apply:
            print("\n".join(render_plan(plan, [], label, args.file,
                                       False, None, None)))
            return 0
        confirmed, _why = _confirm_multi(plan, args)
        if not confirmed:
            print(
                "macro replays always need confirmation (re-run with "
                "--confirm, or answer the interactive prompt); nothing "
                "was written",
                file=sys.stderr,
            )
            return 1
        try:
            apply_result = applier.apply(plan, target, write=True,
                                          label=label)
        except applier.ApplierError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print("\n".join(render_plan(plan, [], label, args.file, True,
                                   apply_result, None)))
        return 0

    if args.macro_save is not None:
        # C3: capture an APPROVED apply from the bounded history. Writes
        # only the history file's macros key (the A3 undo_log precedent
        # — same _save atomic path, no new write surface, the target
        # file is not touched).
        from . import history as _history
        from . import macros as macros_mod
        target = Path(args.file) if args.file else default_target()
        try:
            macro = macros_mod.save(target, args.macro_save,
                                    args.from_id)
        except (macros_mod.MacroError, _history.HistoryError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"captured macro {macro['name']!r}: {len(macro['ops'])} "
              f"change(s) from history entry #{macro['from_id']} "
              f"(label {macro['label']!r})")
        print("replay with --macro NAME (dry-run preview; --apply still "
              "needs --confirm)")
        return 0

    if args.macro_list:
        from . import history as _history
        from . import macros as macros_mod
        target = Path(args.file) if args.file else default_target()
        try:
            saved = macros_mod.list_macros(target)
        except (macros_mod.MacroError, _history.HistoryError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print("\n".join(_header(False)))
        print("")
        if not saved:
            print("no macros saved; capture an applied change with "
                  "--macro-save NAME")
        else:
            print(f"saved macros (oldest capture first, "
                  f"{len(saved)} of {macros_mod.MAX_MACROS} slots):")
            for m in saved:
                ops = ", ".join(f"{o['tool']}={o['value']!r}"
                                for o in m.get("ops", []))
                print(f"  {m['name']:<24} {len(m.get('ops', []))} "
                      f"change(s) from #{m.get('from_id')}: {ops}")
            print("")
            print("replay with --macro NAME; every replay is re-approved "
                  "(--apply + --confirm)")
        return 0

    if args.macro_delete is not None:
        from . import history as _history
        from . import macros as macros_mod
        target = Path(args.file) if args.file else default_target()
        try:
            result = macros_mod.delete(target, args.macro_delete)
        except (macros_mod.MacroError, _history.HistoryError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"deleted macro {result['deleted']!r} "
              f"({result['ops']} change(s), captured from history entry "
              f"#{result['from_id']})")
        return 0

    if args.also_sources is not None and args.profile_save is None:
        arg_parser.error("--also is only valid together with --profile-save")

    if args.profile is not None:
        # F9: apply a saved profile. The composed ops are plain planner
        # ops, so this rides the ordinary plan/render/confirm path — no
        # bespoke writer. Conflicts the order resolved are listed on the
        # preview so the consent is informed.
        from . import history as _history
        from . import profiles as profiles_mod
        target = Path(args.file) if args.file else default_target()
        try:
            ops, conflicts = profiles_mod.profile_ops(target, args.profile)
        except (profiles_mod.ProfileError, _history.HistoryError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        try:
            plan = planner.plan(ops, target)
        except planner.PlannerError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        label = f"profile: {args.profile}"
        if not args.apply:
            print("\n".join(render_plan(plan, [], label, args.file,
                                       False, None, None)))
            for c in conflicts:
                print(f"conflict (later source wins): {c['tool']}: "
                      f"{c['dropped']['value']!r} "
                      f"({c['dropped']['from']}) overridden by "
                      f"{c['kept']['value']!r} ({c['kept']['from']})")
            return 0
        confirmed, _why = _confirm_multi(plan, args)
        if not confirmed:
            print(
                "multi-change profiles need the second consent (re-run "
                "with --confirm, or answer the interactive prompt); "
                "nothing was written",
                file=sys.stderr,
            )
            return 1
        try:
            apply_result = applier.apply(plan, target, write=True,
                                          label=label)
        except applier.ApplierError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print("\n".join(render_plan(plan, [], label, args.file, True,
                                   apply_result, None)))
        return 0

    if args.profile_save is not None:
        # F9: save a named composition. Sources resolve against the live
        # registry and saved stores NOW; writes only the history file's
        # profiles key (same bounded store as macros). Target untouched.
        from . import history as _history
        from . import profiles as profiles_mod
        target = Path(args.file) if args.file else default_target()
        name, first = args.profile_save
        specs = [first] + list(args.also_sources or [])
        try:
            sources = [profiles_mod.parse_source(s) for s in specs]
            entry = profiles_mod.save(target, name, sources)
        except (profiles_mod.ProfileError, _history.HistoryError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"saved profile {entry['name']!r} "
              f"({len(entry['sources'])} source(s))")
        print("preview with --profile " + entry["name"] +
              " (dry-run; --apply still needs --confirm)")
        return 0

    if args.profile_list:
        from . import history as _history
        from . import profiles as profiles_mod
        target = Path(args.file) if args.file else default_target()
        try:
            saved = profiles_mod.list_profiles(target)
        except (profiles_mod.ProfileError, _history.HistoryError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print("\n".join(_header(False)))
        print("")
        if not saved:
            print("no profiles saved; compose one with --profile-save "
                  "NAME SRC [--also SRC ...]")
        else:
            print(f"saved profiles (oldest first, {len(saved)} of "
                  f"{profiles_mod.MAX_PROFILES} slots):")
            for p in saved:
                srcs = ", ".join(
                    (f"{s.get('kind')}:{s.get('name')}" if s.get('kind') != 'call'
                     else f"{s.get('tool')}={s.get('value')!r}")
                    for s in p.get("sources", []))
                print(f"  {p['name']:<24} {len(p.get('sources', []))} "
                      f"source(s): {srcs}")
            print("")
            print("preview with --profile NAME; apply with --profile NAME "
                  "--apply --confirm")
        return 0

    if args.profile_show is not None:
        from . import history as _history
        from . import profiles as profiles_mod
        target = Path(args.file) if args.file else default_target()
        try:
            print("\n".join(profiles_mod.render_profile_lines(
                target, args.profile_show)))
        except (profiles_mod.ProfileError, _history.HistoryError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.profile_diff is not None:
        from . import history as _history
        from . import profiles as profiles_mod
        target = Path(args.file) if args.file else default_target()
        name_a, name_b = args.profile_diff
        try:
            result = profiles_mod.diff(target, name_a, name_b)
        except (profiles_mod.ProfileError, _history.HistoryError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        rows = result["rows"]
        if not rows:
            print(f"profiles {name_a!r} and {name_b!r} are effectively "
                  f"identical")
        else:
            print(f"diff {name_a!r} -> {name_b!r} "
                  f"({len(rows)} differing tool(s)):")
            for row in rows:
                if row["kind"] == "only_a":
                    print(f"  only in {name_a!r}: {row['tool']} = "
                          f"{row['a']!r}")
                elif row["kind"] == "only_b":
                    print(f"  only in {name_b!r}: {row['tool']} = "
                          f"{row['b']!r}")
                else:
                    print(f"  changed: {row['tool']} {row['a']!r} -> "
                          f"{row['b']!r}")
        return 0

    if args.profile_delete is not None:
        from . import history as _history
        from . import profiles as profiles_mod
        target = Path(args.file) if args.file else default_target()
        try:
            result = profiles_mod.delete(target, args.profile_delete)
        except (profiles_mod.ProfileError, _history.HistoryError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"deleted profile {result['deleted']!r} "
              f"({result['sources']} source(s))")
        return 0

    if args.profile_whatif is not None:
        # F11: composite what-if for a saved profile. Read-only: plan,
        # project through the cited interaction graph per source and for
        # the composite, run the applier's own sanity checks against the
        # projected state, and render. Nothing is applied or written.
        from . import history as _history
        from . import planner as _planner_mod
        from . import profile_preview
        from . import profiles as profiles_mod
        target = Path(args.file) if args.file else default_target()
        try:
            view = profile_preview.build(target, args.profile_whatif)
        except (profiles_mod.ProfileError, _history.HistoryError,
                _planner_mod.PlannerError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print("\n".join(profile_preview.render(view)))
        return 0

    if args.monitor is not None and args.file is not None:
        arg_parser.error(
            "--monitor and --file are mutually exclusive: --monitor "
            "resolves the target to that screen's override file")

    if args.monitor is not None:
        # F20: --monitor is a TARGET RESOLVER — the ordinary machinery
        # (parser, planner, applier, backup sibling, history) then runs
        # against the screen's override file unchanged. Naming a screen
        # explicitly is the consent for its override directory to exist
        # (upstream creates monitors/<screen>/ on the first per-screen
        # write too); the applier's own no-directory-creation rule is
        # untouched.
        from . import monitors as monitors_mod
        try:
            resolved = monitors_mod.resolve_target(
                default_target(), args.monitor)
        except ValueError as exc:
            arg_parser.error(str(exc))
        resolved.parent.mkdir(parents=True, exist_ok=True)
        args.file = str(resolved)

    if args.monitors:
        # F20 read-only report: discovered override files + the global
        # monitor-behavior tools, cited. Never writes.
        from . import monitors as monitors_mod
        target = Path(args.file) if args.file else default_target()
        discovered = monitors_mod.discover(target)
        print("monitor-aware planning (F20) — read-only report")
        print("")
        print("per-screen override files (upstream forScreen layer, "
              "shell/plugin/src/Caelestia/Config/rootnodes.cpp:87 + "
              "common.cpp:15):")
        if not discovered:
            print(f"  none yet under {monitors_mod.monitors_dir(target)} — "
                  "target one with --monitor SCREEN (the file is created "
                  "on the first --apply, like any config target)")
        for d in discovered:
            if d.get("error"):
                print(f"  {d['screen']}: {d['error']}")
                continue
            print(f"  {d['screen']}: {d['managed_keys']} registry-managed "
                  f"key(s), {d['unknown_keys']} unknown")
        print("")
        print("global tools that select monitor behavior (same schema, "
              "cited):")
        for row in monitors_mod.monitor_behavior_tools():
            enum = (f" {'|'.join(row['enum'])}" if row.get("enum") else "")
            cite = row["citations"][0][0] if row["citations"] else "-"
            print(f"  {row['name']:<28} {row['path']} "
                  f"(default {row['default']!r}{enum}) [{cite}]")
        print("")
        print("NOTE: connected-screen enumeration lives in "
              "Quickshell/Wayland; this report only sees override files "
              "that already exist. per-screen values use the SAME "
              "planner/confirm/undo gates as the global config.")
        return 0

    if args.audit_env:
        # F21: the environment audit — one read-only pass composing the
        # existing engines. Exit 1 only on CRITICAL findings (states the
        # applier itself would refuse); WARN/INFO never block.
        from datetime import datetime as _dt
        from . import envaudit
        target = Path(args.file) if args.file else default_target()
        result = envaudit.audit(target, now=_dt.now())
        print("\n".join(envaudit.render_lines(result)))
        return 1 if result["critical"] else 0

    if args.gen_catalog:
        # F23: generated community docs. Byte-identity is the contract:
        # the committed files are OUTPUT of this module, and a registry
        # change that outdates them fails here instead of shipping
        # stale docs (the gen_adapter --verify discipline).
        from . import catalog
        if args.gen_verify:
            problems = catalog.verify()
            if problems:
                for problem in problems:
                    print(f"STALE: {problem}", file=sys.stderr)
                print("catalog verify FAILED", file=sys.stderr)
                return 1
            print("catalog verify OK: generated artifacts are "
                  "byte-identical")
            return 0
        written = catalog.generate()
        for rel, path in written.items():
            print(f"wrote {path} ({rel})")
        print("verify with --gen-catalog --verify")
        return 0

    if args.a11y_recipe:
        from . import plain as plain_mod
        print("\n".join(plain_mod.render_recipe()))
        return 0

    if args.env_save is not None:
        # F17: snapshot the current environment. Writes only the history
        # file's bounded "environments" key; the target is untouched.
        from . import environments as env_mod
        target = Path(args.file) if args.file else default_target()
        try:
            entry = env_mod.save(target, args.env_save)
        except env_mod.EnvironmentError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"saved environment {entry['name']!r}: "
              f"{entry['paths']} value(s), hash "
              f"{entry['content_hash'][:12]}..., "
              f"{len(entry.get('macros', []))} macro(s), "
              f"{len(entry.get('profiles', []))} profile(s) metadata")
        print("diff with --env-diff " + entry["name"] +
              "; restore with --env-restore (dry-run first)")
        return 0

    if args.env_list:
        from . import environments as env_mod
        target = Path(args.file) if args.file else default_target()
        saved = env_mod.list_environments(target)
        print("\n".join(_header(False)))
        print("")
        if not saved:
            print("no environments saved; snapshot one with "
                  "--env-save NAME")
        else:
            print(f"saved environments (oldest first, {len(saved)} of "
                  f"{env_mod.MAX_ENVIRONMENTS} slots):")
            for e in saved:
                print(f"  {e['name']:<24} {e.get('created_at', '?')} "
                      f"hash {str(e.get('content_hash'))[:12]}..."
                      + (" [imported]" if e.get("imported") else ""))
            print("")
            print("restore with --env-restore NAME (dry-run first; "
                  "--apply --confirm to write)")
        return 0

    if args.env_show is not None:
        from . import environments as env_mod
        target = Path(args.file) if args.file else default_target()
        try:
            env = env_mod.get(target, args.env_show, verify=True)
        except env_mod.EnvironmentError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"environment {env['name']!r}: "
              f"created {env.get('created_at', '?')}, hash "
              f"{str(env.get('content_hash'))[:12]}... VERIFIED, "
              f"captured against registryTools="
              f"{env.get('registry_tools')} "
              f"(local now: {len(registry.TOOL_SPECS)})")
        if env.get("imported"):
            print(f"  imported bundle; drift: {env.get('import_drift')}; "
                  f"unknown paths: "
                  f"{len(env.get('import_unknown_paths', []))}")
        print(f"  metadata: {len(env.get('macros', []))} macro(s), "
              f"{len(env.get('profiles', []))} profile(s) "
              "(restored separately through their own commands)")
        return 0

    if args.env_diff is not None:
        from . import environments as env_mod
        target = Path(args.file) if args.file else default_target()
        try:
            diff = env_mod.diff_to_ops(target, args.env_diff)
        except env_mod.EnvironmentError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if diff["identical"]:
            print(f"environment {args.env_diff!r} matches the live "
                  f"config on every registry-managed path")
        else:
            print(f"restoring {args.env_diff!r} would "
                  f"({len(diff['set_ops'])} set / "
                  f"{len(diff['reset_ops'])} reset-to-default op(s)):")
            for op in diff["set_ops"] + diff["reset_ops"]:
                print(f"  {op['raw']}")
        for label, keys in (("only in SNAPSHOT (untouched by the "
                             "restore — not registry-managed)",
                             diff["unknown_only_in_snapshot"]),
                            ("only in LIVE file (untouched)",
                             diff["unknown_only_in_live"])):
            if keys:
                print(f"{label}: {', '.join(keys)}")
        return 0

    if args.env_restore is not None:
        # F17: restore = the diff becomes a PLAIN planned multi-change
        # apply through the standard gates. No bespoke writer.
        from . import environments as env_mod
        target = Path(args.file) if args.file else default_target()
        try:
            diff = env_mod.diff_to_ops(target, args.env_restore)
            ops = diff["set_ops"] + diff["reset_ops"]
            if not ops:
                print(f"environment {args.env_restore!r} already matches "
                      "the live config — no changes needed")
                return 0
            plan = planner.plan(ops, target)
        except env_mod.EnvironmentError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except planner.PlannerError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        label = f"env restore: {args.env_restore}"
        if not args.apply:
            print("\n".join(render_plan(plan, [], label, args.file,
                                       False, None, None)))
            return 0
        confirmed, _why = _confirm_multi(plan, args)
        if not confirmed:
            print(
                "multi-change restores need the second consent (re-run "
                "with --confirm, or answer the interactive prompt); "
                "nothing was written",
                file=sys.stderr,
            )
            return 1
        try:
            apply_result = applier.apply(plan, target, write=True,
                                          label=label)
        except applier.ApplierError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print("\n".join(render_plan(plan, [], label, args.file, True,
                                   apply_result, None)))
        return 0

    if args.env_delete is not None:
        from . import environments as env_mod
        target = Path(args.file) if args.file else default_target()
        try:
            result = env_mod.delete(target, args.env_delete)
        except env_mod.EnvironmentError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"deleted environment {result['deleted']!r}")
        return 0

    if args.env_export is not None:
        # F18: export a verified environment as a portable bundle. The
        # ONLY file written is the caller-chosen --to path.
        from . import environments as env_mod
        target = Path(args.file) if args.file else default_target()
        if not args.to:
            arg_parser.error("--env-export needs --to PATH")
        try:
            bundle = env_mod.export_bundle(target, args.env_export)
        except env_mod.EnvironmentError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        Path(args.to).write_text(json.dumps(bundle, indent=2, sort_keys=True)
                                 + "\n", encoding="utf-8")
        print(f"exported environment {args.env_export!r} to {args.to} "
              f"(schema {bundle['schema']}, hash verified)")
        return 0

    if args.env_import is not None:
        # F18: import is REVIEW-ONLY: the bundle lands as a new stored
        # snapshot (marked imported, drift measured); nothing is applied.
        from . import environments as env_mod
        target = Path(args.file) if args.file else default_target()
        if not args.as_name:
            arg_parser.error("--env-import needs --as NAME")
        try:
            bundle = json.loads(Path(args.env_import).read_text(
                encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"error: cannot read the bundle: {exc}", file=sys.stderr)
            return 1
        try:
            entry = env_mod.import_bundle(target, bundle, args.as_name)
        except env_mod.EnvironmentError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"imported environment {entry['name']!r} (review-only: "
              f"NOTHING was applied): drift {entry['import_drift']}, "
              f"unknown paths "
              f"{len(entry['import_unknown_paths'])}")
        print("preview with --env-diff " + entry["name"] +
              "; restore with --env-restore if you want it")
        return 0

    if args.list_tools:
        # §5.1: --list-tools ignores TEXT (and writes nothing, ever).
        from . import plain as plain_mod
        if args.group is not None:
            groups = tuple(registry.GROUPS)
            if args.group not in groups:
                print(
                    f"error: unknown group {args.group!r}; available groups: "
                    + ", ".join(groups),
                    file=sys.stderr,
                )
                return 1
        if plain_mod.plain_requested(
                True if getattr(args, "plain", False) else None):
            # F24 linear form: one record per tool, one fact per line.
            from assistant.adapters.caelestia.registry import TOOL_SPECS, TOOL_COUNT
            wanted = (lambda s: s.group == args.group
                      if args.group is not None else lambda s: True)
            specs = [s for s in TOOL_SPECS if wanted(s)]
            print(f"tool registry: {TOOL_COUNT} tools "
                  f"({len(specs)} shown)")
            rows = ([
                ("tool", s.name), ("path", s.path), ("kind", s.kind),
                ("range", s.minimum if s.minimum is not None else "-" if
                 not s.enum else " | ".join(map(str, s.enum))),
                ("max", s.maximum if s.maximum is not None else "-"),
                ("default", s.default if s.default is not None else "-"),
            ] for s in specs)
            for line in plain_mod.to_linear_rows(
                    ("tool", "path", "kind", "range", "max", "default"),
                    [tuple(r[1] for r in row) for row in rows]):
                plain_mod.assert_plain_safe(line)
                print(line)
            return 0
        print("\n".join(_list_tools_lines(args.group)))
        return 0

    if args.tool is not None:
        spec = tool_by_name(args.tool)
        if spec is None:
            print(
                f"error: unknown tool {args.tool!r}; run --list-tools to see every tool",
                file=sys.stderr,
            )
            near = registry.suggest_tools(args.tool, max_distance=2)
            if near:
                shown = ", ".join(f"{name} (distance {dist})"
                                  for name, dist in near[:3])
                print(f"did you mean: {shown}", file=sys.stderr)
            return 1
        print("\n".join(_tool_card(spec)))
        return 0

    if args.restore:
        target = Path(args.file) if args.file else default_target()
        try:
            result = applier.restore(target)
        except applier.ApplierError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print("\n".join(_header(False)))
        print("")
        print(result["message"])
        return 0

    if args.text is None and args.calls is None:
        arg_parser.error(
            "TEXT is required unless --list-tools, --tool, --restore, --call, "
            "--explain, --history, --undo, --lint or a --macro* flag is given"
        )

    target = Path(args.file) if args.file else default_target()
    file_arg = args.file

    call_args: Optional[List[str]] = None
    cli_errors: Dict[int, Dict[str, Any]] = {}

    if args.calls is not None:
        # Direct tool addressing: compose the ops here, then ride
        # the exact same plan/render/--apply/--restore path as NL requests.
        try:
            ops, cli_errors = _compose_call_ops(args.calls)
        except ValueError as exc:
            arg_parser.error(str(exc))
        call_args = list(args.calls)
        parse_notes: List[str] = []
        text = ""
    else:
        result = parser.parse(args.text)
        verdict = result["verdict"]

        if verdict != "INTENT":
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print("\n".join(render_verdict(result)))
            return 0

        ops = result["ops"]
        parse_notes = list(result.get("notes", []))
        text = args.text

    try:
        plan = planner.plan(ops, target)
    except planner.PlannerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if cli_errors:
        _splice_call_errors(
            plan, cli_errors, total=len(args.calls) if args.calls else 0
        )

    if not args.apply:
        if args.json:
            payload = dict(plan)
            payload["notes"] = parse_notes + list(plan.get("notes", []))
            print(json.dumps(payload, indent=2))
        else:
            print("\n".join(
                render_plan(plan, parse_notes, text, file_arg, False, None, call_args)
            ))
        if args.calls is not None and plan.get("errors"):
            # A direct invocation with a bad name/value is a caller error,
            # not an honest NL verdict: exit 1 even in dry-run.
            print(
                f"error: {len(plan['errors'])} --call entry(s) rejected; "
                "nothing was written",
                file=sys.stderr,
            )
            return 1
        return 0

    # --apply: the gated write path.
    if plan.get("apply_blocked"):
        print(
            "error: apply refused — the plan contains rejected or type-mismatched entries; "
            "nothing was written",
            file=sys.stderr,
        )
        if args.json:
            payload = dict(plan)
            payload["notes"] = parse_notes + list(plan.get("notes", []))
            print(json.dumps(payload, indent=2))
        return 1

    # Issue #120's confirmation rule: a request that resolves to MORE THAN
    # ONE tool call is previewed and must be confirmed before anything is
    # written; single-setting requests apply directly (exactly as the
    # issue frames it).
    applicable_count = sum(
        1 for e in plan.get("entries", [])
        if not e.get("error") and not e.get("no_op")
    )
    if applicable_count > 1:
        confirmed, why = _confirm_multi(plan, args)
        if not confirmed:
            if args.json:
                payload = dict(plan)
                payload["notes"] = parse_notes + list(plan.get("notes", []))
                payload["confirmation"] = "required"
                print(json.dumps(payload, indent=2))
            else:
                print("\n".join(
                    render_plan(plan, parse_notes, text, file_arg, False,
                                None, call_args)))
                print("")
                print(why)
            return 1

    try:
        label = text if text else "direct --call invocation"
        apply_result = applier.apply(plan, target, write=True, label=label)
    except applier.ApplierError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        payload = dict(plan)
        payload["notes"] = parse_notes + list(plan.get("notes", []))
        payload["applied"] = {
            "written": apply_result.get("written", False),
            "changes": apply_result.get("changes", 0),
        }
        print(json.dumps(payload, indent=2))
        return 0

    print("\n".join(
        render_plan(plan, parse_notes, text, file_arg, True, apply_result, call_args)
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
