"""Lexical knowledge for the cortex layer: morphology, fuzzy matching,
and the desktop-customization synonym/antonym lexicon.

Guarantees (same spine as ``settings/parser.py``):

- PURE module: no I/O, no environment, no clock, no randomness. Every
  function is a pure mapping from strings to strings/lists/numbers.
- stdlib ``re``/``difflib`` only; covered by the behavioral no-shell/no-network tests.
- Deterministic: identical input yields identical output, always.

What lives here and why (each is a real, named, citable algorithm):

- ``stem()`` — a Porter-style suffix-stripping stemmer, reduced to the
  longest-match suffix rules that matter for this domain (plurals, -ing,
  -ed, -er, -ly, -ness). Full Porter is ~60 rules; this implements the
  subset with measurable effect on desktop-customization vocabulary and
  skips the measure/``m > 0`` conditions that need vowel-cluster scoring,
  because the domain has no minimal-pair stems that would need them.
- ``levenshtein()`` — the classic DP edit distance, O(len_a * len_b)
  with the two-row memory optimization.
- ``jaro_winkler()`` — Jaro similarity with Winkler's prefix boost, the
  standard lean typo-similarity for short strings (tool names, nouns).
- ``char_ngrams()`` + ``ngram_similarity()`` — character 3-gram cosine
  overlap, the robust "sounds like" signal for compound tool names
  (``setGreeterMorningStart`` vs "greeting morning").
- ``SYNONYMS`` / ``DIRECTION_WORDS`` / ``ABSOLUTE_WORDS`` — the domain
  lexicon: hand-seeded, reviewable, versioned. Synonyms map user words
  onto the registry's own noun vocabulary; direction words carry the
  numeric polarity they imply (``thinner -> -1``) so the router can
  attach a value-extraction cue without any language model.

The lexicon is deliberately small and honest: it claims nothing about
words it does not know. Unknown words fall through to stem/ngram
matching only — no silent invention of meaning.
"""

from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Morphology: domain-reduced Porter stemmer.
# ---------------------------------------------------------------------------

# Longest-match suffix rules, ordered. Each rule: (suffix, replacement,
# min_stem_len). Applied once, longest match first. The rule set is
# IDEMPOTENT by construction: no rule's replacement may leave a suffix
# that a later-pass rule would strip again (that is why -er/-ers/-es/-ate
# transformations are absent: "corners"->"corner"->"corn" chains are
# exactly the kind of instability that breaks query/document stemming
# agreement — both sides must land on the SAME stem, always).
_SUFFIX_RULES: List[Tuple[str, str, int]] = [
    ("ations", "ation", 4),
    ("ities", "ity", 3),
    ("iveness", "ive", 3),
    ("fulness", "ful", 3),
    ("ousness", "ous", 3),
    ("ization", "ize", 3),
    ("tional", "tion", 3),
    ("bilities", "ble", 3),
    ("ies", "y", 3),
    ("ings", "", 4),
    ("ing", "", 4),
    ("edly", "", 4),
    ("ness", "", 4),
    ("ment", "", 4),
    ("ally", "", 4),
    ("ed", "", 3),
    ("ly", "", 3),
    ("s", "", 3),
]

_WORD_RE = re.compile(r"[a-z]+")


def stem(word: str) -> str:
    """Domain-reduced Porter stem of one lowercase word.

    Idempotent (``stem(stem(w)) == stem(w)``) because each rule is applied
    at most once and no rule's output feeds another rule (see the rule
    table's comment). Comparatives keep their -er ("thinner" stays
    "thinner" on BOTH sides of every comparison, which is what makes
    cue-vocabulary agreement reliable). Words shorter than any rule's
    ``min_stem_len`` are returned unchanged (``"ss"`` never loses its s).
    """
    w = word.lower()
    if w.endswith("ss"):  # glass/class/loss keep their double-s: stripping
        return w          # one s breaks idempotence (glass->glas->gla)
    for suffix, replacement, min_len in _SUFFIX_RULES:
        if w.endswith(suffix) and len(w) - len(suffix) >= min_len:
            return w[: len(w) - len(suffix)] + replacement
    return w


# ---------------------------------------------------------------------------
# Fuzzy string similarity (all pure, all deterministic).
# ---------------------------------------------------------------------------


def levenshtein(a: str, b: str, cap: Optional[int] = None) -> int:
    """Two-row edit distance. ``cap`` stops early once the distance provably
    exceeds it (returns ``cap + 1``) — a bounded Levenshtein automaton in
    spirit, useful for cheap rejection in the router hot loop."""
    if a == b:
        return 0
    if cap is not None and abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        best = cur[0]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if cur[j] < best:
                best = cur[j]
        if cap is not None and best > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def jaro(a: str, b: str) -> float:
    """Jaro similarity (0..1). Standard definition: match window, ordered
    transposition count, ``(m/|a| + m/|b| + (m-t)/m) / 3``."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    window = max(len(a), len(b)) // 2 - 1
    if window < 0:
        window = 0
    a_match = [False] * len(a)
    b_match = [False] * len(b)
    matches = 0
    for i in range(len(a)):
        lo = max(0, i - window)
        hi = min(i + window + 1, len(b))
        for j in range(lo, hi):
            if not b_match[j] and a[i] == b[j]:
                a_match[i] = b_match[j] = True
                matches += 1
                break
    if matches == 0:
        return 0.0
    transpositions = 0
    k = 0
    for i in range(len(a)):
        if a_match[i]:
            while not b_match[k]:
                k += 1
            if a[i] != b[k]:
                transpositions += 1
            k += 1
    transpositions //= 2
    return (matches / len(a) + matches / len(b) + (matches - transpositions) / matches) / 3.0


def jaro_winkler(a: str, b: str, prefix_scale: float = 0.1) -> float:
    """Jaro-Winkler: Jaro + prefix boost (max 4 chars, scale capped at
    the standard 0.1 so the score can never exceed 1)."""
    j = jaro(a, b)
    prefix = 0
    for ca, cb in zip(a, b):
        if ca != cb or prefix == 4:
            break
        prefix += 1
    return j + prefix * prefix_scale * (1.0 - j)


def char_ngrams(text: str, n: int = 3) -> List[str]:
    """Character n-grams with word boundaries marked (``|`` pads), so
    word-start shape information survives the n-gram collapse."""
    padded = "|" + text.lower() + "|"
    return [padded[i : i + n] for i in range(len(padded) - n + 1)]


def ngram_similarity(a: str, b: str, n: int = 3) -> float:
    """Cosine overlap of character n-gram multisets — the "sounds like"
    signal for compound tool names vs user phrases."""
    ga, gb = Counter(char_ngrams(a, n)), Counter(char_ngrams(b, n))
    if not ga or not gb:
        return 0.0
    dot = sum(count * gb[gram] for gram, count in ga.items())
    norm = (sum(v * v for v in ga.values()) ** 0.5) * (sum(v * v for v in gb.values()) ** 0.5)
    return dot / norm if norm else 0.0


# ---------------------------------------------------------------------------
# The domain lexicon (hand-seeded, reviewable, versioned).
# ---------------------------------------------------------------------------

LEXICON_VERSION = 1

# User words -> registry noun vocabulary. Keys are matched on RAW words and
# STEMS in the router; values are the canonical words the registry nouns use.
SYNONYMS: Dict[str, Tuple[str, ...]] = {
    # compound joins (hyphenated keys are never tool atoms, so these map
    # query-side only and cannot change the indexed corpus / fingerprints)
    "time-out": ("timeout", "expire"),
    # bar / panel
    "bar": ("bar", "taskbar", "panel"),
    "panel": ("bar", "panel", "taskbar"),
    "taskbar": ("taskbar", "bar", "panel"),
    "toolbar": ("bar", "taskbar"),
    "strip": ("bar",),
    "ribbon": ("bar",),
    # dock
    "dock": ("dock",),
    "app-dock": ("dock",),
    # transparency / blur
    "glass": ("blur", "glass"),
    "glassy": ("blur", "glass"),
    "frost": ("blur",),
    "frosted": ("blur",),
    "blurry": ("blur",),
    "hazy": ("blur",),
    "see-through": ("transparency",),
    "translucent": ("transparency",),
    "transparent": ("transparency",),
    "opaque": ("transparency",),
    # radius / corners
    "corner": ("rounding", "corner"),
    "corners": ("rounding", "corner"),
    "rounded": ("rounding",),
    "rounding": ("rounding",),
    "curvature": ("rounding",),
    "sharp": ("rounding",),
    "squarish": ("rounding",),
    # spacing / padding
    "spacing": ("spacing",),
    "gaps": ("spacing", "gap"),
    "gap": ("gap", "spacing"),
    "roomy": ("spacing",),
    "cramped": ("spacing",),
    "airy": ("spacing",),
    "padding": ("padding",),
    "margin": ("padding",),
    "margins": ("padding",),
    # fonts
    "font": ("font",),
    "fonts": ("font",),
    "text": ("font", "label"),
    "letters": ("font",),
    "typeface": ("font",),
    "typography": ("font",),
    # animations
    "animation": ("animation",),
    "animations": ("animation",),
    "motion": ("animation",),
    "transition": ("animation",),
    "transitions": ("animation",),
    # notifications
    "notification": ("notification",),
    "notifications": ("notifications",),
    "notifs": ("notifications",),
    "toast": ("notification",),
    "toasts": ("notification",),
    "popup": ("notification", "popup"),
    "popups": ("notification", "popup"),
    "banner": ("notification",),
    # launcher
    "launcher": ("launcher",),
    "app-launcher": ("launcher",),
    "menu": ("launcher",),
    "start-menu": ("launcher",),
    # greeter / clock
    "greeter": ("greeter",),
    "greeting": ("greeter",),
    "clock": ("clock",),
    # workspaces
    "workspace": ("workspace",),
    "workspaces": ("workspaces",),
    "desktops": ("workspaces",),
    "virtual-desktop": ("workspaces",),
    # overview
    "overview": ("overview",),
    "mission-control": ("overview",),
    # border
    "border": ("border",),
    "outline": ("border",),
    "stroke": ("border",),
    "bezel": ("border", "pitch-black"),
    # wallpaper
    "wallpaper": ("wallpaper",),
    "wallpapers": ("wallpaper",),
    "background": ("wallpaper", "background"),
    # scheme / colors
    "scheme": ("scheme", "color"),
    "color": ("color", "scheme"),
    "colors": ("color", "scheme"),
    "colour": ("color", "scheme"),
    "colours": ("color", "scheme"),
    "palette": ("scheme", "palette"),
    "theme": ("scheme", "theme"),
    "accent": ("accent", "scheme"),
    # preview
    "preview": ("preview",),
    "previews": ("preview",),
    "thumbnail": ("preview",),
    "thumbnails": ("preview",),
    "peek": ("preview",),
    # tray
    "tray": ("tray",),
    "system-tray": ("tray",),
    "systray": ("tray",),
    # status / icons
    "status": ("status",),
    "indicators": ("status",),
    "icons": ("icon",),
    "icon": ("icon",),
    # battery/perf
    "battery": ("battery",),
    "cpu": ("cpu",),
    "memory": ("memory",),
    "performance": ("performance",),
    "perf": ("performance",),
    # osd
    "osd": ("osd",),
    "overlay": ("osd",),
    # volume / brightness
    "volume": ("volume",),
    "brightness": ("brightness",),
    # sidebar / dashboard
    "sidebar": ("sidebar",),
    "dashboard": ("dashboard",),
    "utilities": ("utilities",),
    # lock
    "lock": ("lock",),
    "lockscreen": ("lock",),
    "login": ("lock",),
    # session
    "session": ("session",),
    "power-menu": ("session",),
    # game mode
    "game-mode": ("game",),
    "gamemode": ("game",),
    "gaming": ("game",),
    # F3 (2026-09-28): the overview's animation timing is the OVERVIEW
    # config's baseDuration — upstream declares it in
    # shell/plugin/src/Caelestia/Config/overviewconfig.hpp
    # ("overview.baseDuration", cited on the tool itself). Without this
    # the phrase "overview animation" lexicalizes as the generic
    # animation-speed tool, which is the confident-wrong the arena
    # measures. Hyphenated key: query-side only, never re-indexed.
    "overview-animation": ("base", "duration"),
    # F3 (2026-09-28): Nexus's "performance panel" IS the dashboard
    # surface (shell/modules/nexus/pages/panels/DashboardPanel.qml — the
    # panel that renders performance.* tools). Query-side only.
    "performance-panel": ("dashboard",),
    # sound / audio
    "sound": ("sound",),
    "sounds": ("sound",),
    "audio": ("audio",),
}

# Antonym/direction pairs: word -> numeric direction on the matched tool's
# axis (+1 = up/more, -1 = down/less). These are the polarity cues the
# router attaches as value-extraction hints; the settings planner still
# resolves the actual value against the live file (never this module).
DIRECTION_WORDS: Dict[str, int] = {
    # NOTE: bare "up"/"down" are deliberately NOT direction words here —
    # they are phrasal-verb particles far too often ("piling up",
    # "show up", "clean up") and a false direction cue silently corrupts
    # cue-kind agreement. parser.py's own grammar applies them inside
    # constrained value phrases; the router's cue extractor does not have
    # that context, so it abstains on them.
    "thinner": -1, "smaller": -1, "slimmer": -1, "shrink": -1,
    "reduce": -1, "decrease": -1, "less": -1, "lower": -1,
    "tighter": -1, "minimize": -1,
    "fewer": -1, "few": -1, "shorten": -1, "trim": -1, "cut": -1,
    "transparent": -1, "translucent": -1, "see-through": -1,
    "thicker": 1, "bigger": 1, "larger": 1, "wider": 1, "grow": 1,
    "increase": 1, "more": 1, "raise": 1, "roomier": 1,
    "spacious": 1, "expand": 1, "enlarge": 1, "maximize": 1,
    "lengthen": 1, "extend": 1, "add": 1,
    "opaque": 1, "solid": 1,
    "rounder": 1, "squarer": -1, "sharper": -1,
    "faster": -1, "quicker": -1, "snappier": -1, "slower": 1,
    "smoother": 1, "softer": 1, "harder": -1,
    "later": 1, "earlier": -1, "sooner": -1, "delay": 1,
    "brighter": 1, "darker": -1, "lighter": -1, "heavier": 1,
}

# Words that imply absolute-zero or absolute-maximum targets (the
# "borderless" trick from parser.py §3.4, generalized).
ABSOLUTE_WORDS: Dict[str, float] = {
    "borderless": 0.0,
    "zero": 0.0,
    "none": 0.0,
    "maximum": 1.0,
    "max": 1.0,
    "full": 1.0,
    "pitch-black": 1.0,
}

# On/off vocabulary (mirrors parser.py §3.5 so both surfaces agree).
BOOL_ON_WORDS = frozenset({"on", "enable", "enabled", "activate", "show", "display", "reveal"})
BOOL_OFF_WORDS = frozenset({"off", "disable", "disabled", "deactivate", "hide", "conceal"})


# PMI-mined hints (capability 1): loaded once from data/mined_synonyms.json
# when present. These are HINTS on the expanded (0.3-weight) channel —
# never replacements — and the file is regenerated by a dev script, so
# shipping code never mutates them.
_MINED_HINTS: Dict[str, tuple] = {}


def _load_mined_hints() -> None:
    import json as _json
    from pathlib import Path as _Path
    path = _Path(__file__).resolve().parents[1] / "data" / "mined_synonyms.json"
    try:
        raw = _json.loads(path.read_text(encoding="utf-8")).get("mined", {})
    except (OSError, ValueError):
        return
    for tool, words in raw.items():
        for w in words:
            _MINED_HINTS.setdefault(w, tuple())
    # word -> the words mined alongside the same tool atoms
    _atom_words = {}
    for tool, words in raw.items():
        for w in words:
            _atom_words.setdefault(w, set()).update(
                x for x in words if x != w)
    for w, xs in _atom_words.items():
        _MINED_HINTS[w] = tuple(sorted(xs))


_load_mined_hints()


def expand_synonyms(text: str) -> List[str]:
    """Every canonical noun word that any word of ``text`` maps to, plus the
    original words, plus (last, as hints) any PMI-mined co-occurring
    vocabulary words. Pure union — no scoring here (the router scores)."""
    out: List[str] = []
    for raw in _WORD_RE.findall(text.lower()):
        if raw not in out:
            out.append(raw)
        for key in (raw, stem(raw)):
            for mapped in SYNONYMS.get(key, ()):
                if mapped not in out:
                    out.append(mapped)
        for hint in _MINED_HINTS.get(raw, ()):
            if hint not in out:
                out.append(hint)
    return out


def direction_of(text: str) -> int:
    """Net direction cue of ``text``: sum of DIRECTION_WORDS hits, sign()
    collapsed to -1/0/+1. Mixed signals cancel to 0 (honest abstain)."""
    total = 0
    for raw in _WORD_RE.findall(text.lower()):
        total += DIRECTION_WORDS.get(raw, DIRECTION_WORDS.get(stem(raw), 0))
    if total > 0:
        return 1
    if total < 0:
        return -1
    return 0


def camel_split(name: str) -> List[str]:
    """Split a camelCase tool name into its word atoms:
    ``setGreeterMorningStart`` -> ``["set", "greeter", "morning", "start"]``.
    This is the trick that turns 259 noun-silent tool NAMES into
    addressable vocabulary without inventing any semantics."""
    return [w.lower() for w in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", name) if w]


def distributional_neighbors(embedder, word: str, k: int = 5) -> List[str]:
    """A2 — the distributional view of this lexicon: the top-k words the
    embedder's co-occurrence space places nearest to ``word``, formatted
    as reviewable ``"word (cosine)"`` rows.

    This is the honest complement to the hand-seeded SYNONYMS table: the
    seeded lexicon says what a maintainer believes; the distributional
    view says what the corpus's own co-occurrences imply. It CLAIMS
    nothing about words absent from the corpus (unknown -> empty list)
    and adds no semantics to the router by itself — a maintainer reading
    these rows and promoting a genuine synonym into SYNONYMS is the
    intended workflow (the reviewed-diff discipline the package rationale (see module docstrings) §5
    applies to lexicon changes too).

    ``embedder`` is any PpmiEmbedder/PpmiSvdEmbedder (both expose
    ``neighbors``); the function is pure given the embedder."""
    found = embedder.neighbors(word, k=k) if hasattr(embedder, "neighbors") else []
    return [f"{w} ({score:+.4f})" for w, score in found]
