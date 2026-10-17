"""Compositional slot grammar for the settings parser (the settings design notes (see module docstrings) §3.8).

Issue A1's contract: paraphrases of the SAME intent should route through
ONE grammar — a slot model ``{intensifier, target, direction, dimension}``
— instead of relying on the frozen grammar's exact phrase lists.

Design boundary (why this is a RECOVERY layer, not a replacement):

- ``settings/parser.py``'s frozen grammar is a documented, test-pinned
  contract (the package rationale (see module docstrings) §7: "the frozen 18 tools and their natural-language
  grammar are preserved byte-for-byte"). This module therefore NEVER runs
  when the frozen grammar resolves a request: parser.py calls
  ``recover()`` only on the paths where the frozen grammar already gave
  up ("noun matched, no applicable cue" — the settings design notes (see module docstrings) §3.3 step 7). Every
  frozen-verdict output is byte-identical to before by construction; the
  only observable change is that some previously-AMBIGUOUS requests now
  resolve to the same ops their canonical phrasings produce.
- The noun (target) surface is NOT grown: targets are exactly the
  noun-matched ``ToolSpec`` list the frozen grammar already found.
  No new nouns, no new tools, no per-tool phrase lists here — the
  extended vocabulary below is per-DIMENSION (size/anim/trans/bool),
  composed with the target by class, which is what makes paraphrases
  share one grammar.

The compositional rule (the whole point of A1): a direction WORD and a
dimension do not have to arrive pre-joined ("faster", "more transparent").
They can arrive separately and be composed:

- ``speed up`` + animations  -> anim, faster -> step DOWN (durations scale)
- ``increase`` + transparency -> trans, more-transparent -> base DOWN
- ``increase`` + opacity     -> trans, more-opaque      -> base UP

Formally: ``base_step = sign(word) * polarity(noun)`` where the trans
nouns carry their own polarity (transparency = -1 on the opacity-base
axis, opacity = +1). The same multiplication resolves every
verb/noun combination without listing the phrases.

Intensifiers ("a bit", "slightly", "a lot", ...) are EXTRACTED and
reported but deliberately magnitude-neutral: the settings design notes (see module docstrings) §3.1 pins
"'a bit / slightly' does NOT change the step count (still ±1) —
documented, deterministic". Extending that pin to strong intensifiers
keeps one rule instead of two.

Guarantees (same spine as parser.py):

- PURE: no I/O, no environment, no clock, no randomness. Identical input,
  identical output.
- Deterministic, stdlib ``re`` only; every regex compiled once at module
  level. Extended vocabulary fires ONLY inside ``recover()``, never inside
  the frozen grammar.
- Honest declines: two conflicting direction cues, no fitting target,
  a bool cue on a numeric tool, or a magnitude (number/percent) in the
  sentence all return ``None`` — the frozen grammar's AMBIGUOUS answer
  stands. Recovery adds signal, never guesses.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from assistant.adapters.caelestia.registry import ToolSpec

__all__ = ["extract", "recover", "SLOTS_VERSION"]

SLOTS_VERSION = 1

# ---------------------------------------------------------------------------
# Slot 1 — INTENSIFIER (extracted, reported, magnitude-neutral per §3.1).
# ---------------------------------------------------------------------------

_LIGHT_INTENSIFIER_RE = re.compile(
    r"\b(?:a bit|a little|a tad|a touch|slightly|somewhat|mildly|marginally)\b"
)
_STRONG_INTENSIFIER_RE = re.compile(
    r"\b(?:a lot|much|significantly|noticeably|way|far|considerably|"
    r"substantially|heavily|drastically|dramatically)\b"
)

# ---------------------------------------------------------------------------
# Slot 2 — DIRECTION, extended lexicon (recovery path only).
#
# Multi-word verb-particle constructions are matched FIRST and masked out,
# so their particles ("up" in "speed up") are never re-read as bare size
# words. Each entry: (regex, dimension, sign on that dimension's axis).
# sign semantics per axis:
#   size   : +1 larger, -1 smaller (plain polarity)
#   anim   : +1 slower (durations UP), -1 faster (durations DOWN) — the
#            documented inversion (parser.py §3.4, the settings design notes (see module docstrings) §2)
#   trans  : sign on the OPACITY-BASE axis (+1 more opaque, -1 more
#            transparent); noun polarity multiplies this below
# ---------------------------------------------------------------------------

_RECOVERY_DIRECTIONS: tuple = (
    # verb-particle constructions (anim axis, sign on the duration axis)
    (re.compile(r"\bspeed(?:ing)? up\b"), "anim", -1),
    (re.compile(r"\bhurry(?:ing)? up\b"), "anim", -1),
    (re.compile(r"\bslow(?:ing)? down\b"), "anim", +1),
    (re.compile(r"\bcrank(?:ing)? up\b"), "size", +1),
    (re.compile(r"\bcrank(?:ing)? down\b"), "size", -1),
    (re.compile(r"\bdial(?:ling)? back\b"), "size", -1),
    (re.compile(r"\btone(?:ing)? down\b"), "size", -1),
    (re.compile(r"\bbump(?:ing)? up\b"), "size", +1),
    (re.compile(r"\bthin(?:ning)? out\b"), "size", -1),
    (re.compile(r"\bfatten(?:ing)? up\b"), "size", +1),
    # single verbs absent from the frozen vocabularies (§3.4 lists them)
    (re.compile(r"\benlarge\b"), "size", +1),
    (re.compile(r"\bexpand\b"), "size", +1),
    (re.compile(r"\bboost\b"), "size", +1),
    (re.compile(r"\bwiden\b"), "size", +1),
    (re.compile(r"\bmaximize\b"), "size", +1),
    (re.compile(r"\blengthen\b"), "size", +1),
    (re.compile(r"\bheighten\b"), "size", +1),
    (re.compile(r"\bnarrow\b"), "size", -1),
    (re.compile(r"\bminimize\b"), "size", -1),
    (re.compile(r"\btrim\b"), "size", -1),
    (re.compile(r"\bshorten\b"), "size", -1),
    (re.compile(r"\bthin\b"), "size", -1),
    (re.compile(r"\bshave\b"), "size", -1),
    # transparency axis (sign on the opacity base)
    (re.compile(r"\bmore transparent\b"), "trans", -1),
    (re.compile(r"\bmore see-through\b"), "trans", -1),
    (re.compile(r"\bmore translucent\b"), "trans", -1),
    (re.compile(r"\bmore opaque\b"), "trans", +1),
    (re.compile(r"\bmore solid\b"), "trans", +1),
)

# Bare size-direction words the frozen grammar ALREADY knows (§3.4) — reused
# here only for cross-dimension composition (e.g. "increase the
# transparency"), never to re-implement the frozen match.
_GENERIC_UP_RE = re.compile(
    r"\b(?:increase|raise|more|grow|raise|boost|higher)\b"
)
_GENERIC_DOWN_RE = re.compile(
    r"\b(?:reduce|decrease|less|lower|reduce|fewer)\b"
)

# Trans-noun polarity on the opacity-base axis (the compositional core):
# "transparency" and "opacity" point in OPPOSITE directions on the same knob.
_TRANS_NOUN_POLARITY = (
    (re.compile(r"\btransparen(?:cy|t)\b"), -1),
    (re.compile(r"\bsee-through\b"), -1),
    (re.compile(r"\btranslucen(?:t|cy)\b"), -1),
    (re.compile(r"\bopacit?y\b|\bopaque\b"), +1),
)

# Slot 2b — verbal bool cues the frozen grammar does not carry (§3.5 shape).
_BOOL_ACTIVATE_RE = re.compile(r"\b(?:activate|switch on|turn on)\b")
_BOOL_DEACTIVATE_RE = re.compile(r"\b(?:deactivate|switch off|turn off)\b")

# Magnitude presence — recovery declines when the sentence carries a number
# or a percent: those belong to the frozen value-phrase paths (§3.4), and a
# recovery layer inventing magnitudes would be a silent guess.
_HAS_NUMBER_RE = re.compile(r"\d")
_HAS_PERCENT_RE = re.compile(r"%")


def _mask(text: str, spans: Sequence) -> str:
    chars = list(text)
    for start, end in spans:
        for idx in range(start, min(end, len(chars))):
            chars[idx] = " "
    return "".join(chars)


# ---------------------------------------------------------------------------
# extract() — the pure slot extractor.
# ---------------------------------------------------------------------------


def extract(text: str) -> Dict[str, Any]:
    """Extract the four slots from normalized text (pure, total function).

    Returns ``{"intensifier": "light"|"strong"|None,
               "cue": {"dimension": "size"|"anim"|"trans"|"bool",
                        "sign": int, "raw": str} | None,
               "generic": {"sign": int, "raw": str} | None,
               "trans_polarity": {"sign": int, "raw": str} | None,
               "conflict": bool}``.

    ``conflict`` is True when two direction cues disagree on the same
    dimension — the honest decline signal (recover() returns None).
    """
    intensifier = None
    masked = text
    for kind, regex in (("light", _LIGHT_INTENSIFIER_RE),
                        ("strong", _STRONG_INTENSIFIER_RE)):
        found = list(regex.finditer(masked))
        if found and intensifier is None:
            intensifier = kind
        masked = _mask(masked, [m.span() for m in found])

    cue: Optional[Dict[str, Any]] = None
    generic: Optional[Dict[str, Any]] = None
    conflict = False
    for regex, dimension, sign in _RECOVERY_DIRECTIONS:
        found = list(regex.finditer(masked))
        if not found:
            continue
        masked = _mask(masked, [m.span() for m in found])
        if cue is None:
            cue = {"dimension": dimension, "sign": sign, "raw": found[0].group(0)}
        elif (cue["dimension"], cue["sign"]) != (dimension, sign):
            conflict = True

    for regex, sign in ((_GENERIC_UP_RE, +1), (_GENERIC_DOWN_RE, -1)):
        found = list(regex.finditer(masked))
        if not found:
            continue
        masked = _mask(masked, [m.span() for m in found])
        if generic is None:
            generic = {"sign": sign, "raw": found[0].group(0)}
        elif generic["sign"] != sign:
            conflict = True

    trans_polarity: Optional[Dict[str, Any]] = None
    for regex, polarity in _TRANS_NOUN_POLARITY:
        found = regex.search(text)
        if found:
            trans_polarity = {"sign": polarity, "raw": found.group(0)}
            break

    return {
        "intensifier": intensifier,
        "cue": cue,
        "generic": generic,
        "trans_polarity": trans_polarity,
        "conflict": conflict,
    }


# ---------------------------------------------------------------------------
# recover() — compose the slots into frozen-grammar-shaped ops.
# ---------------------------------------------------------------------------


def recover(normalized: str, noun_specs: Sequence[ToolSpec]) -> Optional[Dict[str, Any]]:
    """Try to compose an INTENT result the frozen grammar could not.

    Called ONLY from parser.py on the §3.3 step-7 paths (noun matched, no
    applicable cue). Returns a parser-shaped result dict, or None to keep
    the frozen AMBIGUOUS verdict. Pure; never raises on odd input.
    """
    if not noun_specs:
        return None
    if _HAS_NUMBER_RE.search(normalized) or _HAS_PERCENT_RE.search(normalized):
        return None  # magnitudes belong to the frozen value-phrase paths

    # Lazy import: parser imports this module at load time; the note helpers
    # are only needed at recover() call time, after both modules are fully
    # initialized. Keeps every §3.5/§3.4 note string single-sourced.
    from .parser import _ANIM_INVERSION_NOTE, _bool_note

    slots = extract(normalized)
    if slots["conflict"]:
        return None

    # -- verbal bool cues: only bool tools, only with an honest note -------
    bool_ops: List[Dict[str, Any]] = []
    bool_targets = [s for s in noun_specs if s.kind == "bool"]
    for regex, value in ((_BOOL_ACTIVATE_RE, True), (_BOOL_DEACTIVATE_RE, False)):
        found = regex.search(normalized)
        if not found or not bool_targets:
            continue
        raw = found.group(0)
        for spec in bool_targets:
            op: Dict[str, Any] = {
                "tool": spec.name, "action": "set", "value": value, "raw": raw,
            }
            op["note"] = _bool_note(spec.name, raw, value)
            bool_ops.append(op)
        break  # first applicable cue wins; on+off together would conflict

    cue, generic, polarity = slots["cue"], slots["generic"], slots["trans_polarity"]
    if bool_ops and cue is None and generic is None:
        return _recovered(bool_ops, slots)
    if bool_ops:
        return None  # a direction cue AND an explicit on/off word: the frozen
        # grammar's split-question territory, not ours to guess

    if cue is None and generic is None:
        return None

    # -- compose {direction} x {dimension} x {target class} ----------------
    # One direction slot per sentence (conflicts already declined above),
    # applied to EVERY fitting target — the same single-phrase pattern the
    # frozen grammar uses for "make the bar and the dock icons bigger".
    ops: List[Dict[str, Any]] = []
    for spec in noun_specs:
        composed = _compose(cue, generic, polarity, spec)
        if composed is None:
            continue
        dimension, sign, raw = composed
        if dimension == "bool_size":
            # §3.5: comparatives on bool tools map to on/off WITH the note.
            value = sign > 0
            op = {"tool": spec.name, "action": "set", "value": value, "raw": raw,
                  "note": _bool_note(spec.name, raw, value)}
        else:
            note = _ANIM_INVERSION_NOTE if spec.name == "setAnimationSpeed" else None
            op = {"tool": spec.name, "action": "step", "value": sign, "raw": raw}
            if note:
                op["note"] = note
        ops.append(op)

    if not ops:
        return None
    return _recovered(ops, slots)


def _compose(cue: Optional[Dict[str, Any]],
             generic: Optional[Dict[str, Any]],
             polarity: Optional[Dict[str, Any]],
             spec: ToolSpec) -> Optional[tuple]:
    """One slot composition for one target tool.

    Returns (dimension, sign, raw) or None when the composition does not
    fit the tool's class. The multiplication ``sign * polarity`` is the
    trans-noun rule; the anim axis inverts bare up/down words ("speed up"
    arrives pre-composed and is never inverted twice).
    """
    # Pre-joined cues compose only with their own dimension's tools.
    if cue is not None:
        dimension, sign, raw = cue["dimension"], cue["sign"], cue["raw"]
        if dimension == "size":
            if spec.name == "setTransparencyBase":
                # A size verb on the trans knob goes through the SAME
                # polarity multiplication as generic words: "boost the
                # transparency" is step DOWN, not up.
                if polarity is None:
                    return None
                return ("trans", sign * polarity["sign"], raw)
            if spec.name == "setAnimationSpeed":
                return None  # size vocabulary never reaches the anim tool
            if spec.kind in ("float", "int"):
                return ("size", sign, raw)
            if spec.kind == "bool":
                return ("bool_size", 1 if sign > 0 else -1, raw)
            return None
        if dimension == "anim":
            return ("anim", sign, raw) if spec.name == "setAnimationSpeed" else None
        if dimension == "trans":
            return ("trans", sign, raw) if spec.name == "setTransparencyBase" else None
        return None

    # Generic direction words compose with the target's own dimension.
    # The anim axis is deliberately NOT composed from generics: "increase
    # the animations" is genuinely ambiguous between more-speed and
    # more-duration, so it declines (only pre-joined cues — "speed up",
    # "slow down" — or the frozen "faster/slower" words reach that tool).
    if generic is not None:
        sign, raw = generic["sign"], generic["raw"]
        if spec.name == "setAnimationSpeed":
            return None  # generic words decline on the anim axis (see above)
        if spec.name == "setTransparencyBase":
            if polarity is None:
                return None  # "more" of what? the noun must say (honest decline)
            return ("trans", sign * polarity["sign"], f"{raw} {polarity['raw']}")
        if spec.kind in ("float", "int"):
            return ("size", sign, raw)
        if spec.kind == "bool":
            # §3.5: comparatives on bool tools map to on/off with the note.
            return ("bool_size", 1 if sign > 0 else -1, raw)
    return None


def _recovered(ops: List[Dict[str, Any]], slots: Dict[str, Any]) -> Dict[str, Any]:
    from .parser import _result

    intensifier = slots["intensifier"]
    slot_bits = []
    if ops:
        tools = ", ".join(sorted({op["tool"] for op in ops}))
        slot_bits.append(f"target={tools}")
    cue = slots["cue"] or slots["generic"]
    if cue:
        word_dir = "up" if cue.get("sign", 0) > 0 else "down"
        polarity = slots["trans_polarity"]
        if polarity is not None and any(
                op["tool"] == "setTransparencyBase" for op in ops):
            composed = "down" if cue.get("sign", 0) * polarity["sign"] < 0 else "up"
            slot_bits.append(
                f"direction={word_dir} (word) x polarity={polarity['raw']}"
                f"({polarity['sign']:+d}) -> {composed} on the opacity base")
        else:
            slot_bits.append(f"direction={word_dir}")
    if intensifier:
        slot_bits.append(
            f"intensifier={intensifier} (magnitude-neutral per the settings design notes (see module docstrings) §3.1)")
    note = "compositional slot grammar (§3.8): " + "; ".join(slot_bits)
    return _result("INTENT", ops=ops, notes=[note])
