"""Intent parser for the settings layer (the settings design notes (see module docstrings) §3): text -> ops.

Guarantees:

- PURE function: ``parse(text)`` performs no file I/O, reads no environment
  variable, no clock, no randomness — identical input always yields an
  identical result (the same guarantee as Layer 1's engine, ``engine.py``).
- Deterministic grammar, no scoring, no ML: fixed precedence — normalize,
  preset check FIRST, scheme/wallpaper dead-ends, explanatory dead-ends,
  then per-tool noun+cue matching (§3.3). stdlib ``re`` only, every regex
  compiled exactly once at module level.
- The parser never resolves values that need the current file state:
  relative ops carry only the factor (multiply) or the signed step count
  (step, ±1); the planner resolves them against the real current values.
- Honest verdicts (§3.2): INTENT / AMBIGUOUS / NO_INTENT / SUGGESTED.
  AMBIGUOUS never silently picks a tool; NO_INTENT says so and lists what
  IS supported; SUGGESTED emits inert ``SUGGESTED_NOT_EXECUTED:`` strings
  for the scheme/wallpaper system, which lives outside shell.json.
- Nothing in this module can write, execute, or touch the network — by
  construction, since it opens nothing at all.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from assistant.adapters.caelestia.registry import ToolSpec, TOOL_SPECS, describe, tool_by_name, tool_by_path
from . import slots as _slots

# ---------------------------------------------------------------------------
# Grammar (§3.4), compiled once. All matching happens on normalized text
# (lowercase, single spaces), so the patterns are written lowercase.
# ---------------------------------------------------------------------------

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")

RESET_RE = re.compile(r"\b(?:reset|default)\b")

# Numeric direction vocabularies, per class (§3.4). Transparency phrases are
# multi-word and must be matched BEFORE the bare size words they contain
# ("more transparent" contains "more"; "less transparent" contains "less").
SIZE_DOWN_RE = re.compile(r"\b(?:thinner|smaller|slimmer|shrink|reduce|decrease|less|lower|down|tighter|denser)\b")
SIZE_UP_RE = re.compile(r"\b(?:thicker|bigger|larger|wider|grow|increase|more|raise|up|roomier|spacious)\b")
ANIM_DOWN_RE = re.compile(r"\b(?:faster|quicker|snappier)\b")
ANIM_UP_RE = re.compile(r"\b(?:slower|smoother)\b")
TRANS_DOWN_RE = re.compile(r"\b(?:more transparent|see-through|lighter)\b")
TRANS_UP_RE = re.compile(r"\b(?:more opaque|more solid|less transparent)\b")

BOOL_ON_RE = re.compile(r"\b(?:turn on|switch on|enable|show|on)\b")
BOOL_OFF_RE = re.compile(r"\b(?:turn off|switch off|disable|hide|off)\b")

# Position words, with "edge"/"side" tolerated adjacent (§3.4).
POSITION_RE = re.compile(r"\b(top|bottom|left|right)\b(?:\s+(?:edge|side))?")

# Preset triggers (§3.7) — checked FIRST, before per-tool synonyms.
COMPACT_RE = re.compile(r"\b(?:more compact|compact|denser|tighter)\b")
MINIMAL_RE = re.compile(r"\b(?:minimalist|minimal look|minimal|cleaner look|simpler look)\b")

# Scheme/wallpaper nouns (§3.3 step 3) — requests that live OUTSIDE
# shell.json, answered with inert suggestions only ([C25]).
SCHEME_NOUN_RE = re.compile(r"\b(?:accent color|color scheme|theme colors?|accents?|schemes?|palettes?)\b")
WALLPAPER_NOUN_RE = re.compile(r"\b(?:wallpapers?|background images?|desktop backgrounds?)\b")

# Explanatory dead-end patterns (§3.6).
DOCK_NOUN_RE = re.compile(r"\bdocks?\b")
NOTIFS_NOUN_RE = re.compile(r"\bnotifications?\b")
TRANSPARENCY_NOUN_RE = re.compile(r"\btransparency\b")
FONT_FAMILY_RE = re.compile(r"\b(?:font famil(?:y|ies)|font names?|typefaces?)\b")

# §3.4, setBorderThickness noun-list parenthetical: "remove the
# border" / "borderless" -> absolute 0 (the shipped UI's own "Set to 0 for a
# borderless look", AppearancePage.qml:108-115).
BORDERLESS_RE = re.compile(r"\bborderless\b")
REMOVE_RE = re.compile(r"\bremove\b")

# §3.4: the phrase "pitch black" is itself the on-cue for the
# bezel-mode tool when the sentence carries no other value phrase — the
# exact mirror of the "borderless" -> 0 rule ("make the shell pitch black"
# has no on/off word to scan). With an explicit on/off word ("disable pitch
# black") or a comparative ("less pitch black") the standard §3.4/§3.5
# paths handle it instead and this cue stays out of the phrase list.
PITCH_BLACK_RE = re.compile(r"\bpitch black\b")

# Per-tool noun regexes (§3.4), precompiled from the registry.
_NOUN_RES: Dict[str, List[re.Pattern[str]]] = {
    spec.name: [re.compile(r"\b(?:" + group + r")\b") for group in spec.nouns]
    for spec in TOOL_SPECS
}

# Direction-word applicability classes (§3.3 step 5, §3.4). Numeric tools
# split into size / animation / transparency classes; each class answers
# only its own direction vocabulary (plus plain numbers and percents).
# The registry also holds noun-silent direct-address tools (nouns only
# on the 18-tool core) — they are NOT part of the
# natural-language surface, so only noun-bearing tools are size-class
# members (the targetless candidate list stays the 18-tool core set).
_ANIM_TOOLS = frozenset({"setAnimationSpeed"})
_TRANS_TOOLS = frozenset({"setTransparencyBase"})
_SIZE_TOOLS = frozenset(
    spec.name for spec in TOOL_SPECS
    if spec.nouns and spec.kind in ("float", "int")
    and spec.name not in _ANIM_TOOLS | _TRANS_TOOLS
)
_NUMERIC_TOOLS = _SIZE_TOOLS | _ANIM_TOOLS | _TRANS_TOOLS

# Honest note texts (verbatim from the settings design notes (see module docstrings) where quoted).
_BOOL_NO_STRENGTH_LABELS = {
    "setBlurEnabled": "blur",
    "setLivePreviews": "live previews",
    "setBarPersistent": "bar persistence",
    "setPitchBlack": "pitch black (bezel mode)",
    # setDockBadges: the honest note for comparatives on the app-badges
    # toggle — same §3.5 shape as the other bool tools above.
    "setDockBadges": "app badges",
}
_ANIM_INVERSION_NOTE = "durations scale: lower = faster"
_PRESET_NOTE = (
    "preset values are conservative defaults, not an upstream-defined look"
)

SCHEME_SUGGESTIONS = [
    "SUGGESTED_NOT_EXECUTED: caelestia scheme list",
    "SUGGESTED_NOT_EXECUTED: caelestia scheme set -n <name>",
]
WALLPAPER_SUGGESTIONS = [
    "SUGGESTED_NOT_EXECUTED: caelestia wallpaper -f <path>",
    "SUGGESTED_NOT_EXECUTED: caelestia wallpaper -r (random)",
]
SCHEME_NOTE = (
    "accent colors live in the scheme system (state under $XDG_STATE_HOME/caelestia), "
    "not shell.json; the assistant cannot validate scheme names, so the color word is not parsed"
)
WALLPAPER_NOTE = (
    "wallpapers live in the scheme system (caelestia wallpaper, state under "
    "$XDG_STATE_HOME/caelestia), not shell.json; the assistant cannot validate file paths"
)

SPLIT_QUESTION = (
    "multiple value phrases found in one request; split that into separate "
    "requests (one change per request)"
)
PERCENT_QUESTION = (
    "percent of what? say e.g. '20% bigger', or give a plain number like 1.2"
)


# ---------------------------------------------------------------------------
# Normalization and scanning helpers (all pure).
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """§3.3 step 1: lowercase, collapse whitespace, strip surrounding quotes
    and trailing ``.,!?``. Keeps ``%``, ``.`` and ``-`` so decimals and
    negatives survive."""
    cleaned = text.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in "'\"":
        cleaned = cleaned[1:-1].strip()
    cleaned = cleaned.rstrip(".,!?").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.lower()


def _mask(text: str, spans: List[Tuple[int, int]]) -> str:
    """Blank out the given spans so later scans cannot re-match them."""
    chars = list(text)
    for start, end in spans:
        for idx in range(start, min(end, len(chars))):
            chars[idx] = " "
    return "".join(chars)


def _scan_pcts(text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for match in PCT_RE.finditer(text):
        out.append({"kind": "percent", "value": float(match.group(1)), "raw": match.group(0), "pos": match.start()})
    return out


def _scan_numbers(text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for match in NUM_RE.finditer(text):
        out.append({"kind": "number", "value": float(match.group(0)), "raw": match.group(0), "pos": match.start()})
    return out


def _scan_directions(text: str) -> List[Dict[str, Any]]:
    """Numeric direction words (size/animation/transparency classes).

    Transparency phrases are matched first and masked out, so their bare
    size words ("more", "less") are never double-counted.
    """
    matches: List[Dict[str, Any]] = []
    masked = text
    for regex, klass, sign in (
        (TRANS_DOWN_RE, "trans", -1),
        (TRANS_UP_RE, "trans", +1),
        (SIZE_DOWN_RE, "size", -1),
        (SIZE_UP_RE, "size", +1),
        (ANIM_DOWN_RE, "anim", -1),
        (ANIM_UP_RE, "anim", +1),
    ):
        found = list(regex.finditer(masked))
        for match in found:
            matches.append({"kind": "direction", "class": klass, "sign": sign, "raw": match.group(0), "pos": match.start()})
        masked = _mask(masked, [m.span() for m in found])
    matches.sort(key=lambda item: item["pos"])
    return matches


def _scan_bools(text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for regex, value in ((BOOL_ON_RE, True), (BOOL_OFF_RE, False)):
        for match in regex.finditer(text):
            out.append({"kind": "bool", "value": value, "raw": match.group(0), "pos": match.start()})
    out.sort(key=lambda item: item["pos"])
    return out


def _scan_positions(text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for match in POSITION_RE.finditer(text):
        out.append({"kind": "position", "value": match.group(1), "raw": match.group(0), "pos": match.start()})
    return out


def _noun_matched(text: str) -> List[ToolSpec]:
    """Specs whose noun groups ALL appear in the text (§3.3 step 5).

    A noun-silent tool (nouns == (), the direct-address tools)
    matches NOTHING — ``all([])`` is vacuously True, so the empty noun
    list must be guarded explicitly, not treated as a universal match.
    """
    matched: List[ToolSpec] = []
    for spec in TOOL_SPECS:
        if spec.nouns and all(regex.search(text) for regex in _NOUN_RES[spec.name]):
            matched.append(spec)
    return matched


def _op(tool: str, action: str, value: Any, raw: str, note: Optional[str] = None) -> Dict[str, Any]:
    op: Dict[str, Any] = {"tool": tool, "action": action, "value": value, "raw": raw}
    if note:
        op["note"] = note
    return op


def _bool_note(tool: str, raw: str, value: bool) -> str:
    """§3.5: comparatives on bool tools map to on/off WITH the honest note."""
    label = _BOOL_NO_STRENGTH_LABELS.get(tool, "this setting")
    return (
        f"{label} in shell.json is on/off only — there is no strength to {raw}; "
        f"{'enabling' if value else 'disabling'} it"
    )


def _result(verdict: str, ops: Optional[List[Dict[str, Any]]] = None,
            candidates: Optional[List[Dict[str, Any]]] = None,
            question: Optional[str] = None, notes: Optional[List[str]] = None,
            suggestions: Optional[List[str]] = None) -> Dict[str, Any]:
    return {
        "verdict": verdict,
        "ops": ops or [],
        "candidates": candidates or [],
        "question": question,
        "notes": notes or [],
        "suggestions": suggestions or [],
    }


# ---------------------------------------------------------------------------
# Verdict builders for the fixed special cases (§3.6, §3.3 steps 2-3).
# ---------------------------------------------------------------------------


def _preset_ops(trigger: str, values: Dict[str, float]) -> List[Dict[str, Any]]:
    ops: List[Dict[str, Any]] = []
    for path, value in values.items():
        spec = tool_by_path(path)
        if spec is None:  # cannot happen: presets name registry paths only
            continue
        ops.append(_op(spec.name, "set", value, trigger))
    return ops


def _scheme_wallpaper(text: str) -> Optional[Dict[str, Any]]:
    scheme = bool(SCHEME_NOUN_RE.search(text))
    wallpaper = bool(WALLPAPER_NOUN_RE.search(text))
    if not scheme and not wallpaper:
        return None
    suggestions: List[str] = []
    notes: List[str] = []
    if scheme:
        suggestions.extend(SCHEME_SUGGESTIONS)
        notes.append(SCHEME_NOTE)
    if wallpaper:
        suggestions.extend(WALLPAPER_SUGGESTIONS)
        notes.append(WALLPAPER_NOTE)
    notes.append("nothing was executed and nothing was written; these are inert suggestions only")
    return _result("SUGGESTED", notes=notes, suggestions=suggestions)


def _dead_ends(text: str) -> Optional[Dict[str, Any]]:
    """§3.6 explanatory dead-ends, evaluated before per-tool matching."""
    if DOCK_NOUN_RE.search(text) and POSITION_RE.search(text):
        return _result(
            "AMBIGUOUS",
            candidates=[{"tool": "setBarPosition", "description": "bar.position — move the WHOLE bar (top/bottom/left/right)"}],
            notes=[
                "dock left/right placement is not a shell.json key: the dock is a "
                "bar element whose placement is the 'zone' of its entry in bar.entries "
                "(set in Nexus -> Panels -> Taskbar -> Elements)",
            ],
            question=(
                "the dock's left/right placement is set in Nexus -> Panels -> Taskbar -> "
                "Elements; did you mean to move the whole bar? if so, say e.g. "
                "'move the bar to the left'"
            ),
        )

    if NOTIFS_NOUN_RE.search(text) and (BOOL_ON_RE.search(text) or BOOL_OFF_RE.search(text)):
        return _result(
            "NO_INTENT",
            notes=[
                "no upstream 'notifications enabled' boolean exists (notifsconfig.hpp has "
                "maxPopups/maxNotifs/expire etc., none an on/off switch); notifications "
                "toggling is deliberately unmapped",
            ],
            question=(
                "there is no notifications on/off setting to change; for the counts that "
                "ARE tools say e.g. 'set the max notification popups to 5' or 'keep at "
                "most 200 stored notifications' — other notification behavior lives in "
                "Nexus"
            ),
        )

    if TRANSPARENCY_NOUN_RE.search(text) and (BOOL_ON_RE.search(text) or BOOL_OFF_RE.search(text)):
        return _result(
            "NO_INTENT",
            notes=[
                "appearance.transparency.enabled exists upstream but is deliberately not a tool "
                "permitted tools; turning transparency off also disables blur in the "
                "upstream UI (AppearancePage.qml:132-136)",
            ],
            question=(
                "did you mean the transparency LEVEL? say e.g. 'make panels more "
                "transparent' or 'set the transparency to 0.7'"
            ),
        )

    if FONT_FAMILY_RE.search(text):
        return _result(
            "NO_INTENT",
            notes=[
                "only the font SIZE scale (appearance.font.scale) is a core tool; font "
                "family/name lives in the appearance.font.*.family string trees, out of scope",
            ],
            question="did you mean the font size? say e.g. 'make the font bigger' or 'set the font scale to 1.2'",
        )

    return None


def _no_intent() -> Dict[str, Any]:
    """§3.2 NO_INTENT: honest answer + the full supported-settings list +
    pointers to Nexus and the troubleshooting layers."""
    notes = ["no supported setting matched this request; the supported settings are:"]
    notes.extend(f"  {describe(spec)}" for spec in TOOL_SPECS)
    notes.append(
        "you can also change these in Nexus (the shell's settings UI); for troubleshooting "
        "questions use the other assistant layers, e.g. python3 -m assistant.core.ask_pipeline \"<problem>\""
    )
    return _result(
        "NO_INTENT",
        notes=notes,
        question="rephrase with a tool noun plus a value or a direction (run --list-tools to see them all)",
    )


# ---------------------------------------------------------------------------
# The single-phrase resolution (§3.3 steps 5-7, §3.4).
# ---------------------------------------------------------------------------


def _distinct_key(phrase: Dict[str, Any]) -> Tuple[Any, ...]:
    """Value phrases are 'distinct' by their resolved value (§3.3 step 6):
    'smaller' and 'thinner' are one phrase (both = one step down); 'bigger'
    and 'smaller' are two; '1.2' and '0.5' are two."""
    if phrase["kind"] == "direction":
        return ("direction", phrase["class"], phrase["sign"])
    if phrase["kind"] == "percent":
        return ("percent", phrase["value"])
    return (phrase["kind"], phrase["value"])


def _resolve_single(phrase: Dict[str, Any], noun_specs: List[ToolSpec]) -> List[Dict[str, Any]]:
    """Apply the one value phrase to every noun-matched tool it fits."""
    ops: List[Dict[str, Any]] = []
    kind = phrase["kind"]

    if kind == "number":
        for spec in noun_specs:
            if spec.kind in ("float", "int"):
                ops.append(_op(spec.name, "set", phrase["value"], phrase["raw"]))
            elif spec.kind == "bool":
                # §3.4: any number on a bool tool -> the planner REJECTS it.
                ops.append(_op(spec.name, "set", phrase["value"], phrase["raw"]))
            # enum tool: a plain number is not an applicable cue.

    elif kind == "percent":
        comparative = phrase.get("comparative")
        if comparative is None:
            # §3.4: bare percent is absolute ONLY on the transparency base
            # (its own UI displays percent); bool tools get a REJECTED op;
            # any other numeric tool makes the whole request AMBIGUOUS.
            numeric_matched = [s for s in noun_specs if s.kind in ("float", "int")]
            non_trans = [s for s in numeric_matched if s.name not in _TRANS_TOOLS]
            if non_trans:
                return []  # caller turns this into the PERCENT AMBIGUOUS verdict
            for spec in noun_specs:
                ops.append(_op(spec.name, "set", phrase["value"] / 100.0, phrase["raw"]))
        else:
            sign = comparative["sign"]
            factor = 1.0 + sign * phrase["value"] / 100.0
            for spec in noun_specs:
                if spec.kind in ("float", "int"):
                    ops.append(_op(spec.name, "multiply", factor, f"{phrase['raw']} {comparative['raw']}"))
                elif spec.kind == "bool":
                    ops.append(_op(spec.name, "set", phrase["value"] / 100.0, phrase["raw"]))

    elif kind == "direction":
        klass, sign, raw = phrase["class"], phrase["sign"], phrase["raw"]
        for spec in noun_specs:
            if spec.name in _ANIM_TOOLS and klass == "anim":
                ops.append(_op(spec.name, "step", sign, raw, note=_ANIM_INVERSION_NOTE))
            elif spec.name in _TRANS_TOOLS and klass == "trans":
                ops.append(_op(spec.name, "step", sign, raw))
            elif spec.kind in ("float", "int") and klass == "size":
                if spec.name in _ANIM_TOOLS or spec.name in _TRANS_TOOLS:
                    continue  # size vocabulary does not fit anim/transparency tools
                ops.append(_op(spec.name, "step", sign, raw))
            elif spec.kind == "bool" and klass == "size":
                # §3.5: comparative words on bool tools map to on/off + note.
                value = sign > 0
                ops.append(_op(spec.name, "set", value, raw, note=_bool_note(spec.name, raw, value)))

    elif kind == "bool":
        for spec in noun_specs:
            if spec.kind == "bool":
                ops.append(_op(spec.name, "set", phrase["value"], phrase["raw"]))

    elif kind == "position":
        for spec in noun_specs:
            if spec.kind == "enum":
                ops.append(_op(spec.name, "set", phrase["value"], phrase["raw"]))

    elif kind == "reset":
        # §3.4: "reset"/"default" + tool noun -> set to the registry default.
        for spec in noun_specs:
            ops.append(_op(spec.name, "set", spec.default, phrase["raw"]))

    return ops


def _candidates_from_specs(specs: List[ToolSpec]) -> List[Dict[str, Any]]:
    return [{"tool": spec.name, "description": describe(spec)} for spec in specs]


# ---------------------------------------------------------------------------
# parse() — the public pure function (§3).
# ---------------------------------------------------------------------------


def parse(text: str) -> Dict[str, Any]:
    """Parse one natural-language settings request into a verdict + ops.

    See the module docstring for the guarantees; the settings design notes (see module docstrings) §3 is the
    normative grammar. Value resolution (current values, clamping) happens
    only in the planner, never here.
    """
    normalized = _normalize(text)
    if not normalized:
        return _no_intent()

    # §3.3 step 2 — preset check FIRST ("compact" is never consumed by
    # per-tool synonyms; "tighter"/"denser" belong to the presets).
    compact = COMPACT_RE.search(normalized)
    minimal = MINIMAL_RE.search(normalized)
    if compact:
        ops = _preset_ops(compact.group(0), {
            "bar.scale": 0.85,
            "appearance.spacing.scale": 0.9,
            "appearance.padding.scale": 0.9,
        })
        return _result("INTENT", ops=ops, notes=[f"preset 'compact': {_PRESET_NOTE}"])
    if minimal:
        ops = _preset_ops(minimal.group(0), {
            "bar.scale": 0.8,
            "appearance.spacing.scale": 0.9,
            "appearance.padding.scale": 0.9,
            "appearance.rounding.scale": 0.9,
        })
        return _result("INTENT", ops=ops, notes=[f"preset 'minimal': {_PRESET_NOTE}"])

    # §3.3 step 3 — scheme/wallpaper: SUGGESTED, inert strings only.
    suggested = _scheme_wallpaper(normalized)
    if suggested is not None:
        return suggested

    # §3.3 step 4 — explanatory dead-ends.
    dead_end = _dead_ends(normalized)
    if dead_end is not None:
        return dead_end

    # §3.3 step 5 — which tools does the sentence name?
    noun_specs = _noun_matched(normalized)

    # §3.4 — scan the value phrases. Percents are masked before the plain
    # number scan so "20%" never also counts as the number 20.
    pcts = _scan_pcts(normalized)
    without_pcts = _mask(normalized, [(p["pos"], p["pos"] + len(p["raw"])) for p in pcts])
    numbers = _scan_numbers(without_pcts)
    directions = _scan_directions(normalized)
    bools = _scan_bools(normalized)
    positions = _scan_positions(normalized)
    reset = RESET_RE.search(normalized) is not None

    # "borderless" (or "remove" + the border noun) is the absolute-0 cue;
    # the word itself implies the border tool even without the bare noun
    # (word boundaries keep \bborder\b from matching inside "borderless").
    borderless = BORDERLESS_RE.search(normalized)
    removed = REMOVE_RE.search(normalized) and any(
        spec.name == "setBorderThickness" for spec in noun_specs
    )
    if borderless:
        border_spec = tool_by_name("setBorderThickness")
        if border_spec is not None and border_spec not in noun_specs:
            noun_specs = noun_specs + [border_spec]

    # §3.4: "make the shell pitch black" — the phrase implies the
    # bezel-mode tool AND its on value, but ONLY when the sentence carries no
    # other value phrase (an explicit on/off word, a comparative/size word, a
    # number, a percent or a reset request all take the standard paths
    # instead, so e.g. "disable pitch black", "reset the pitch black mode"
    # and "less pitch black" keep their §3.4/§3.5 readings and never collide
    # with this implicit cue).
    pitch_black_cue = bool(
        PITCH_BLACK_RE.search(normalized)
        and not (BOOL_ON_RE.search(normalized) or BOOL_OFF_RE.search(normalized))
        and not directions
        and not pcts
        and not numbers
        and not reset
    )
    if pitch_black_cue:
        pitch_black_spec = tool_by_name("setPitchBlack")
        if pitch_black_spec is not None and pitch_black_spec not in noun_specs:
            noun_specs = noun_specs + [pitch_black_spec]

    phrases: List[Dict[str, Any]] = []
    if borderless or removed:
        raw = borderless.group(0) if borderless else "remove"
        phrases.append({"kind": "number", "value": 0.0, "raw": raw, "pos": -1})
    if pitch_black_cue:
        phrases.append({"kind": "bool", "value": True, "raw": "pitch black", "pos": -1})
    for pct in pcts:
        pct = dict(pct)
        # A percent consumes the first numeric direction word as its
        # comparative (§3.4); with no comparative it is absolute on the
        # transparency base only.
        pct["comparative"] = directions[0] if directions else None
        phrases.append(pct)
    phrases.extend(numbers)
    if not pcts and not numbers:
        # A plain number/percent governs any direction word in the same
        # sentence (§3.4: "direction word WITHOUT a number -> step"), so
        # directions only count as standalone phrases in their absence.
        phrases.extend(directions)
    phrases.extend(bools)
    phrases.extend(positions)
    if reset:
        phrases.append({"kind": "reset", "value": True, "raw": "reset", "pos": -1})

    seen: List[Tuple[Any, ...]] = []
    distinct: List[Dict[str, Any]] = []
    for phrase in phrases:
        key = _distinct_key(phrase)
        if key not in seen:
            seen.append(key)
            distinct.append(phrase)

    if len(distinct) >= 2:
        # §3.3 step 6 — several value phrases: never guess the split.
        return _result("AMBIGUOUS", question=SPLIT_QUESTION)

    if distinct:
        phrase = distinct[0]
        ops = _resolve_single(phrase, noun_specs)
        if phrase["kind"] == "percent" and phrase.get("comparative") is None and not ops and noun_specs:
            # Bare percent on a non-transparency numeric tool (§3.4).
            return _result("AMBIGUOUS", question=PERCENT_QUESTION)
        if ops:
            return _result("INTENT", ops=ops)
        if noun_specs:
            # §3.8 — compositional slot-grammar recovery (A1): the frozen
            # grammar could not assemble a value phrase for these nouns;
            # the slot grammar gets one deterministic chance to compose
            # {intensifier, target, direction, dimension} before the honest
            # AMBIGUOUS verdict stands. Never runs when the frozen grammar
            # succeeded, so every frozen output is byte-identical.
            recovered = _slots.recover(normalized, noun_specs)
            if recovered is not None:
                return recovered
            # §3.3 step 7 — noun matched, no applicable cue.
            names = ", ".join(spec.name for spec in noun_specs)
            return _result(
                "AMBIGUOUS",
                candidates=_candidates_from_specs(noun_specs),
                question=(
                    f"what should I do with {names}? give a value, a direction, or on/off "
                    "wording (e.g. 'make it bigger', 'set it to 1.2', 'turn it off')"
                ),
            )
        # No noun matched: only a size direction word carries meaning for
        # step 8 (§3.3 step 8) — numbers, percents, bools and position
        # words without a noun are NO_INTENT, handled by the fallthrough.
        if phrase["kind"] == "direction":
            return _targetless(phrase)

    if directions and not noun_specs:
        # §3.3 step 8 — direction word, no noun: fixed candidate list of the
        # numeric tools whose direction vocabulary matched, registry order.
        return _targetless(directions[0])

    if noun_specs:
        # §3.8 recovery, second hook: a noun matched but NO value phrase of
        # any frozen kind was found (e.g. "thin out the bar") — same one
        # deterministic slot-grammar chance before the AMBIGUOUS verdict.
        recovered = _slots.recover(normalized, noun_specs)
        if recovered is not None:
            return recovered
        names = ", ".join(spec.name for spec in noun_specs)
        return _result(
            "AMBIGUOUS",
            candidates=_candidates_from_specs(noun_specs),
            question=(
                f"what should I do with {names}? give a value, a direction, or on/off "
                "wording (e.g. 'make it bigger', 'set it to 1.2', 'turn it off')"
            ),
        )

    return _no_intent()


def _targetless(direction: Dict[str, Any]) -> Dict[str, Any]:
    """§3.3 step 8: 'make it smaller' — list every numeric tool whose
    direction vocabulary matched, in registry order."""
    klass = direction["class"]
    if klass == "anim":
        specs = [s for s in TOOL_SPECS if s.name in _ANIM_TOOLS]
    elif klass == "trans":
        specs = [s for s in TOOL_SPECS if s.name in _TRANS_TOOLS]
    else:
        specs = [s for s in TOOL_SPECS if s.name in _SIZE_TOOLS]
    return _result(
        "AMBIGUOUS",
        candidates=_candidates_from_specs(specs),
        question=(
            "which setting did you mean? name it, e.g. 'make the bar smaller' or "
            "'set the rounding to 1.5'"
        ),
    )
