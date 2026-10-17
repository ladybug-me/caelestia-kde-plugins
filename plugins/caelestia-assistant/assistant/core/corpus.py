"""Synthetic training corpus for the cortex router (self-supervised, no LLM).

The frozen grammar of ``settings/parser.py`` covers 18 tools because noun
vocabularies were hand-frozen for exactly those. This module derives the
OTHER 259 tools' addressable vocabulary mechanically from the registry
itself — tool names, paths, groups — plus the seeded domain lexicon, and
from that derivation generates a deterministic paraphrase corpus. That
corpus is what the router indexes (BM25) and what the embedder trains on
(PPMI co-occurrences). No network, no model download, no training run on
unreviewed data: every generated row is reproducible from this file alone.

Guarantees:

- DETERMINISTIC and PURE: fixed iteration orders, no randomness, no I/O
  beyond the registry's own import-time ``tools.json`` load (unchanged
  behavior, already drift-guarded by the settings test suite).
- Generated rows only ever CLAIM registry facts: a row for
  ``setGreeterMorningStart`` says "greeter morning start" because the
  tool is named that — the module never invents a semantic claim the
  registry does not make.
- The paraphrase templates are versioned (``CORPUS_VERSION``) so router
  scores are comparable across corpus revisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from assistant.adapters.caelestia.registry import TOOL_SPECS
from .lexicon import ( BOOL_OFF_WORDS, BOOL_ON_WORDS, SYNONYMS, camel_split, )

CORPUS_VERSION = 1

# ---------------------------------------------------------------------------
# Row shape.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    """One synthetic training row: a user-shaped phrase, the surface it
    should retrieve, and the value-extraction cues the phrase carries.

    ``surface`` is either a tool name (``setBarScale``), a preset target
    (``preset:compact``), or a coarse surface (``explain``/``undo``/
    ``scheme``/``wallpaper``/``diagnose``/``search``/``brain``/``issue``).
    ``cues`` carries direction (+1/-1), bool (True/False) and absolute
    targets when the phrase implies one — the router passes them through
    as hints; the settings planner remains the only place a value is
    resolved against the live file.
    """

    text: str
    surface: str
    cues: Dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Per-tool canonical documents (the router's index).
# ---------------------------------------------------------------------------

# Words that are pure plumbing in tool names and carry no addressable
# meaning once split ("set", "enable", "show", "use"...).
_PLUMBING = frozenset({
    "set", "enable", "disable", "show", "hide", "use", "get", "toggle",
    "turn", "on", "off", "max", "min",
})


def tool_atoms(spec) -> List[str]:
    """Addressable word atoms of one ToolSpec: camel-name atoms + dotted
    path atoms + group, plumbing words dropped, deduplicated in order."""
    atoms: List[str] = []
    for atom in camel_split(spec.name):
        if atom not in _PLUMBING and atom not in atoms:
            atoms.append(atom)
    for piece in spec.path.replace("-", ".").split("."):
        # split camelCase inside path pieces too (morningStart -> morning start)
        for atom in camel_split(piece) or ([piece.lower()] if piece else []):
            if atom and atom not in _PLUMBING and atom not in atoms:
                atoms.append(atom)
    if spec.group and spec.group not in atoms:
        atoms.append(spec.group)
    return atoms


def tool_document(spec) -> str:
    """The canonical searchable document for one tool: its atoms, its noun
    groups (18-core tools), its synonyms, and its kind. This single string
    is what BM25 indexes per tool."""
    parts: List[str] = tool_atoms(spec)
    for group in spec.nouns:
        parts.append(group.replace("|", " "))
    for atom in list(parts):
        for mapped in SYNONYMS.get(atom, ()):
            if mapped not in parts:
                parts.append(mapped)
    parts.append(spec.kind)
    return " ".join(parts)


def tool_documents() -> Dict[str, str]:
    """{tool name: canonical document} for every registry tool, registry order."""
    return {spec.name: tool_document(spec) for spec in TOOL_SPECS}


# ---------------------------------------------------------------------------
# Paraphrase generation (deterministic templates over registry facts).
# ---------------------------------------------------------------------------

# Verbs that introduce a settings request. Paired with every tool's atom
# phrase to teach the router that the VERB carries no target information.
_INQUIRE_VERBS = ("make", "set", "change", "adjust", "put", "turn", "move")

# A representative subset of direction words per sign (kept small so the
# corpus stays lean; the router scores full vocabularies at query time).
_DIR_DOWN = ("thinner", "smaller", "less", "lower", "reduce")
_DIR_UP = ("bigger", "larger", "more", "increase", "raise")


def _atom_phrase(spec) -> str:
    atoms = tool_atoms(spec)
    return " ".join(atoms) if atoms else spec.name.lower()


def _rows_for_tool(spec) -> List[Row]:
    """Deterministic paraphrase rows for one tool, by kind:

    - every tool: bare atom phrase + verb-ful variants (the router must
      learn to ignore the verb);
    - float/int tools: one row per representative direction word, with
      the direction cue attached;
    - bool tools: one row per on/off word, cue attached;
    - enum tools: one row per enum value (the value text is itself
      addressable vocabulary — "dock position left");
    - the 18 core tools additionally: their registry noun groups as bare
      phrases, so learned scores and frozen-grammar scores stay aligned.
    """
    rows: List[Row] = []
    phrase = _atom_phrase(spec)
    rows.append(Row(phrase, spec.name, {}))
    for verb in _INQUIRE_VERBS[:4]:
        rows.append(Row(f"{verb} {phrase}", spec.name, {}))

    if spec.kind in ("float", "int"):
        for word in _DIR_DOWN:
            rows.append(Row(f"make {phrase} {word}", spec.name, {"direction": -1}))
            rows.append(Row(f"{word} {phrase}", spec.name, {"direction": -1}))
        for word in _DIR_UP:
            rows.append(Row(f"make {phrase} {word}", spec.name, {"direction": 1}))
            rows.append(Row(f"{word} {phrase}", spec.name, {"direction": 1}))
        if spec.default is not None:
            rows.append(Row(f"reset {phrase} to default", spec.name, {"reset": True}))
            rows.append(Row(f"default {phrase}", spec.name, {"reset": True}))

    elif spec.kind == "bool":
        for word in sorted(BOOL_ON_WORDS):
            rows.append(Row(f"{word} {phrase}", spec.name, {"bool": True}))
        for word in sorted(BOOL_OFF_WORDS):
            rows.append(Row(f"{word} {phrase}", spec.name, {"bool": False}))
        rows.append(Row(f"toggle {phrase}", spec.name, {"toggle": True}))

    elif spec.kind == "enum" and spec.enum:
        values = spec.enum if isinstance(spec.enum, (list, tuple)) else [spec.enum]
        for value in values:
            text_value = str(value).lower().replace("_", " ")
            rows.append(Row(f"{phrase} {text_value}", spec.name, {"enum": value}))

    if spec.nouns:
        for group in spec.nouns:
            rows.append(Row(group.replace("|", " "), spec.name, {}))

    return rows


# ---------------------------------------------------------------------------
# Coarse surfaces (non-tool): the universal-routing half.
# ---------------------------------------------------------------------------

# Hand-seeded rows for surfaces outside the tool registry. Each surface
# gets a handful of canonical phrasings — enough for BM25 to route the
# request to the right LAYER, where that layer's own machinery (rules
# engine, retrieval, brain CLI) takes over. These rows deliberately do
# NOT try to understand the request — only to recognize its shape.
_SURFACE_ROWS: Tuple[Tuple[str, str, Dict[str, object]], ...] = (
    # presets (settings/presets.py — the #120 "make everything minimal" case)
    ("more compact", "preset:compact", {}),
    ("compact everything", "preset:compact", {}),
    ("make everything compact", "preset:compact", {}),
    ("compact mode", "preset:compact", {}),
    ("denser look", "preset:compact", {}),
    ("tighter overall", "preset:compact", {}),
    ("minimal look", "preset:minimal", {}),
    ("minimalist setup", "preset:minimal", {}),
    ("make everything minimal", "preset:minimal", {}),
    ("cleaner look", "preset:minimal", {}),
    ("simpler look", "preset:minimal", {}),
    ("declutter the desktop", "preset:minimal", {}),
    ("optimize for gaming", "preset:gaming", {}),
    ("gaming setup", "preset:gaming", {}),
    ("game mode look", "preset:gaming", {}),
    ("optimize for battery", "preset:battery-saver", {}),
    ("battery saver look", "preset:battery-saver", {}),
    ("power saving setup", "preset:battery-saver", {}),
    ("make it look like macos", "preset:macos-like", {}),
    ("macos look", "preset:macos-like", {}),
    ("macos-like", "preset:macos-like", {}),
    ("apple style desktop", "preset:macos-like", {}),
    # explain (settings/explain.py — #120 "why is my dock blurry")
    ("why is", "explain", {}),
    ("why does", "explain", {}),
    ("why is my", "explain", {}),
    ("explain why", "explain", {}),
    ("what controls", "explain", {}),
    ("what setting controls", "explain", {}),
    ("where does", "explain", {}),
    ("explain the", "explain", {}),
    # undo / history (settings/history.py + cortex/nlhistory.py)
    ("undo the last change", "undo", {}),
    ("undo that", "undo", {}),
    ("revert my last change", "undo", {}),
    ("undo", "undo", {}),
    ("revert", "undo", {}),
    ("rollback", "undo", {}),
    ("restore yesterday", "undo", {}),
    ("restore yesterday's theme", "undo", {}),
    ("what did i change", "history", {}),
    ("show my change history", "history", {}),
    ("recent changes", "history", {}),
    ("change history", "history", {}),
    ("what changed", "history", {}),
    # scheme (inert suggestions — parser.py §3.3 step 3)
    ("change the accent color", "scheme", {}),
    ("change color scheme", "scheme", {}),
    ("new theme colors", "scheme", {}),
    ("different palette", "scheme", {}),
    ("set the accent", "scheme", {}),
    # wallpaper (inert suggestions)
    ("change my wallpaper", "wallpaper", {}),
    ("new background image", "wallpaper", {}),
    ("set wallpaper", "wallpaper", {}),
    ("random wallpaper", "wallpaper", {}),
    # diagnostics (Layer 1)
    ("something is broken", "diagnose", {}),
    ("it crashed", "diagnose", {}),
    ("error message", "diagnose", {}),
    ("not working", "diagnose", {}),
    ("fails to start", "diagnose", {}),
    ("troubleshoot", "diagnose", {}),
    ("shell will not start", "diagnose", {}),
    ("bar disappeared", "diagnose", {}),
    # retrieval (Layer 2)
    ("how do i", "search", {}),
    ("where is the docs", "search", {}),
    ("documentation for", "search", {}),
    ("find in docs", "search", {}),
    ("look up", "search", {}),
    # brain (lean intelligence layer)
    ("plan my day", "brain", {}),
    ("what should i do first", "brain", {}),
    ("plan my tasks", "brain", {}),
    ("how long will it take", "brain", {}),
    ("remind me later", "brain", {}),
    ("my focus", "brain", {}),
    ("organize my notes", "brain", {}),
    # issue drafting (Layer 4)
    ("file an issue", "issue", {}),
    ("draft a bug report", "issue", {}),
    ("report a bug", "issue", {}),
    ("open an issue", "issue", {}),
)


def surface_rows() -> List[Row]:
    return [Row(text, surface, cues) for text, surface, cues in _SURFACE_ROWS]


def all_rows() -> List[Row]:
    """Every synthetic row: tool rows (registry order) then surface rows.
    Deterministic order is part of the contract — row order feeds
    co-occurrence window construction in vectorize.py."""
    rows: List[Row] = []
    for spec in TOOL_SPECS:
        rows.extend(_rows_for_tool(spec))
    rows.extend(surface_rows())
    return rows


def surface_labels() -> List[str]:
    """All routable surfaces in deterministic order (for logging/reporting)."""
    labels: List[str] = [spec.name for spec in TOOL_SPECS]
    labels.extend([
        "preset:compact", "preset:minimal", "preset:gaming",
        "preset:battery-saver", "preset:macos-like",
        "explain", "undo", "history", "scheme", "wallpaper",
        "diagnose", "search", "brain", "issue",
    ])
    return labels


def rows_by_surface() -> Dict[str, List[Row]]:
    """{surface: rows} grouping (the router's supervised signal)."""
    out: Dict[str, List[Row]] = {}
    for row in all_rows():
        out.setdefault(row.surface, []).append(row)
    return out
