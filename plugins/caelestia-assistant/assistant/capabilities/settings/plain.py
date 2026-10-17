"""assistant.capabilities.settings.plain — accessible output mode (F24,
exponential-build-5; the interaction-polish companion to F22).

Two surfaces, one contract:

1. PLAIN OUTPUT (``--plain`` or ``CAELESTIA_ASSISTANT_PLAIN=1``):
   a screen-reader-friendly rendering contract, enforced by test —
   no ANSI escapes, no box-drawing or table decorations, no emoji,
   deterministic ordering, one fact per line. The settings layer's
   output is already plain text by construction; this module makes
   that a TESTED GUARANTEE and gives the two widest tables
   (``--list-tools``, ``--history``) an explicitly linear form:
   ``to_linear_rows`` renders header/row pairs as
   ``key=value`` lines instead of aligned columns, so screen readers
   get one fact per line in a stable order (aligned columns read as
   word soup row by row).

2. THE REDUCED-MOTION RECIPE (``--a11y-recipe``): F22 answers "are
   these colors readable?"; the missing half for motion-sensitive
   users is a documented, inert recipe that composes ALREADY-VALIDATED
   registry knobs into a reduced-motion profile (F9's composition does
   the write-path work on request). The recipe prints the exact
   SUGGESTED_NOT_EXECUTED command; nothing is saved or applied without
   the user running it through the normal preview/consent gates. Every
   knob cites its upstream config declaration (the registry carries
   the citations; they travel here verbatim).

Pure module: no I/O, no environment reads outside the documented
helper, no RNG.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

__all__ = ["plain_requested", "FORBIDDEN_PLAIN_RE", "to_linear_rows",
           "REDUCED_MOTION_RECIPE", "recipe_command", "render_recipe"]

_FORBIDDEN_PATTERNS = (
    r"\x1b\[[0-9;]*m",       # ANSI SGR
    r"\x1b\][^\x07]*\x07",   # OSC
    r"[─│┌┐└┘├┤┬┴┼═║╔╗╚╝╠╣╦╩╬]",  # box drawing
    r"[\U0001F300-\U0001FAFF\u2705\u2714\u2716\u26A0\u26D4]",  # emoji/symbols
)
FORBIDDEN_PLAIN_RE = re.compile("|".join(_FORBIDDEN_PATTERNS))


def plain_requested(explicit: Optional[bool] = None) -> bool:
    """True when plain output is requested: an explicit CLI flag wins,
    else the CAELESTIA_ASSISTANT_PLAIN environment variable (any of
    1/true/yes/on, case-insensitive)."""
    if explicit is not None:
        return explicit
    return os.environ.get("CAELESTIA_ASSISTANT_PLAIN", "").strip().lower() \
        in ("1", "true", "yes", "on")


def to_linear_rows(headers: Sequence[str],
                   rows: Iterable[Sequence[Any]]) -> List[str]:
    """A table as one-fact-per-line records. Ordering is the input's
    (callers pass deterministically-ordered rows); each record is
    separated by a blank line so screen readers hear a complete record
    before the next begins."""
    out: List[str] = []
    for index, row in enumerate(rows, start=1):
        out.append(f"record {index}:")
        for header, value in zip(headers, row):
            out.append(f"  {header}: {value}")
    return out


def assert_plain_safe(text: str) -> None:
    """Raise AssertionError when text violates the plain contract
    (used by the tests that pin the guarantee; cheap enough to call
    from any renderer's plain branch)."""
    hit = FORBIDDEN_PLAIN_RE.search(text)
    assert hit is None, f"plain-output violation: {hit!r} in {text[:80]!r}"


# ---------------------------------------------------------------------------
# The reduced-motion recipe (F24): registry-validated knobs only.
# ---------------------------------------------------------------------------

REDUCED_MOTION_RECIPE: List[Dict[str, Any]] = [
    {
        "tool": "setAnimationSpeed",
        "value": 0.25,
        "why": "the global animation duration scale — 0.25 makes every "
               "shell animation four times faster, the largest effect a "
               "single key has on perceived motion",
    },
    {
        "tool": "setBlurEnabled",
        "value": False,
        "why": "blur is a continuously animating region; disabling it "
               "removes constant background compositing motion",
    },
    {
        "tool": "setLivePreviews",
        "value": False,
        "why": "dock hover previews animate on every hover",
    },
]

# Derived, not duplicated: the actual registry validation (range/type)
# happens when the command runs through --profile-save, exactly like a
# user-typed composition. Recipe commands are INERT strings.


def recipe_command(profile_name: str = "reduced-motion") -> str:
    """The inert one-line composition command for the recipe (F9)."""
    sources = " ".join(f"{r['tool']}={str(r['value']).lower()}"
                       if isinstance(r["value"], bool)
                       else f"{r['tool']}={r['value']}"
                       for r in REDUCED_MOTION_RECIPE)
    return ("SUGGESTED_NOT_EXECUTED: caelestia-assist settings "
            f"--profile-save {profile_name} {sources}")


def render_recipe() -> List[str]:
    lines = [
        "reduced-motion recipe (F24) — every knob is a registry-validated",
        "tool; nothing is saved or applied by printing this:",
        "",
    ]
    for item in REDUCED_MOTION_RECIPE:
        lines.append(f"  {item['tool']} = {item['value']!r}")
        lines.append(f"    why: {item['why']}")
    lines.append("")
    lines.append("one-command composition (rides the F9 profile gates; the")
    lines.append("preview shows every change before --apply --confirm):")
    lines.append(f"  {recipe_command()}")
    lines.append("")
    lines.append("review the composed result first:")
    lines.append("SUGGESTED_NOT_EXECUTED: caelestia-assist settings "
                 "--profile reduced-motion")
    return lines
