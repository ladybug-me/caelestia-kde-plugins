"""Layer 5 of the caelestia assistant: natural-language settings editing
(#120) — the module that a 18-tool natural-language core answers, generated
generated registry (272 tools at the 2026-09-28 resync), explain, bounded undo, presets — the module that
answers "change this setting" while the four troubleshooting layers answer
"what is broken".

Pipeline: text -> parser.parse (pure) -> planner.plan (validated plan
against the target file) -> applier.apply (the only writer, gated).

Guarantees shared by every module in this package:

- Dry-run by default: nothing is ever written unless the caller passes the
  explicit --apply gate — and any plan touching more than one setting
  additionally needs --confirm (second consent) or an interactive yes,
  implementing #120's "confirmation for larger changes". A dry run writes
  no file at all (no target, no backup, no tmp).
- One writer, four paths: applier.py is the ONLY module that writes
  anything, and only behind --apply. It writes exactly the explicit target
  file (via a transient .assistant-tmp + atomic os.replace), its
  .assistant-backup sibling (single-slot, one-level undo via --restore),
  its .assistant-history.json sibling (the bounded 12-entry undo history
  that --history lists and --undo/--undo-id consume), and removes the tmp
  after the rename. No other path is ever opened for writing; parent
  directories are never created. The planner reads the target file and,
  best-effort and read-only, per-monitor override files (to warn that a
  global change would be shadowed) — never writes them.
- Additional surfaces: --explain answers "why does it look like this" against
  effective values (file value else registry default) and never writes;
  --history lists the bounded undo history (read-only); --undo/--undo-id
  consume history entries and write only the reverted values;
  --preset/--list-presets apply named bundles of already-validated tool
  calls through the ordinary planner path (build-time validated; a preset
  is never a bespoke code path).
- Validated before written: every value is checked against the frozen
  generated registry (types, ranges, enums; 19 feature-area groups — see
  --list-tools, tools.json for the generated table and the not_exposed
  reasons). Absolute out-of-range values
  are REJECTED entries — never silently clamped or dropped — and any
  rejected entry blocks the whole apply (all-or-nothing). Relative results
  are clamped with a visible notice. A target file that is invalid JSON
  or not an object aborts the run untouched.
- NEVER executes anything: no subprocess/os.system anywhere in this
  package; suggested commands (scheme/wallpaper, which live outside
  shell.json) are inert strings prefixed SUGGESTED_NOT_EXECUTED.
- No network at all in this module — not even loopback (unlike Layer 3).
- The parser is a pure function: no file I/O, no environment, no clock, no
  randomness; identical input always yields an identical result.
"""

from .applier import apply, restore
from .parser import parse
from .planner import PlannerError, plan
from assistant.adapters.caelestia.registry import TOOL_SPECS, ToolSpec, tool_by_name, tool_by_path

__all__ = [
    "PlannerError",
    "TOOL_SPECS",
    "ToolSpec",
    "apply",
    "parse",
    "plan",
    "restore",
    "tool_by_name",
    "tool_by_path",
]
