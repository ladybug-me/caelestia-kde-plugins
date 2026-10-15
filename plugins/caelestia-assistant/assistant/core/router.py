"""The universal intent router — the cortex layer's decision core.

Ranks EVERY routable surface (every registry tool, 5 presets, the coarse
explain/undo/history/scheme/wallpaper/diagnose/search/brain/issue
surfaces) against one user phrase, using four complementary signals:

1. LEXICAL — BM25+ (vectorize.TfidfIndex) with query expansion:
   typo correction (bounded Levenshtein, distance <= 2) then synonym
   expansion (lexicon.SYNONYMS, including hyphen-bigram forms like
   "see through" -> transparency) then stemming. Expansion happens on
   the QUERY side only — the index stays pure registry vocabulary.
2. SEMANTIC — PPMI cosine (vectorize.PpmiEmbedder) between the query
   embedding and each candidate's document embedding: catches phrases
   whose words never literally overlap any tool vocabulary.
3. FUZZY — character 3-gram cosine between the raw query and the
   candidate's name atoms: catches compound-name addressing
   ("greeter morning start" vs ``setGreeterMorningStart``).
4. NOUN — the frozen noun regexes of the 18 core tools (parser.py's
   own grammar, rebuilt from the registry so the surfaces cannot drift):
   an exact noun hit is the strongest single lexical signal there is.

On top of the four signals sit two DETERMINISTIC structural layers
(grammar, not statistics — deliberately so they can be audited):

- PATTERN BOOSTS: why-questions floor the ``explain`` surface; leading
  undo/revert/restore/rollback floors ``undo``; history phrasings floor
  ``history``; preset trigger words floor their preset (mirroring the
  frozen parser's own precedence: preset triggers fire BEFORE per-tool
  nouns — "make my bar compact" is the compact preset, exactly as
  parser.py §3.7 treats it).
- CUE-KIND AGREEMENT: extracted value cues must AGREE with the
  candidate's kind. A direction cue ("thinner") boosts numeric tools
  and penalizes bool/enum ones; a bool cue boosts bool tools; a
  position cue and literal enum-value words boost enum tools. This is
  the router-side mirror of parser.py §3.3 step 5's direction classes,
  generalized to every registry tool via the registry's kind metadata.

Verdicts (honest, never a silent guess):

- ``ROUTED``   — clear winner (score >= min_score, margin >= min_margin)
- ``AMBIGUOUS``— near-tie: question + top candidates listed
- ``ABSTAIN``  — nothing scored well: "no setting clearly matches"

The router NEVER writes: its output is candidates + cues + evidence.
All value resolution and all writes stay in the settings
planner/applier spine. Deterministic: pure function of (text, state);
the embedder's projection uses a fixed seed.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from assistant.adapters.caelestia.registry import PRESETS, TOOL_SPECS
from .phonetics import phonetic_hit_rate
from .corpus import tool_atoms, tool_document
from .lexicon import (
    ABSOLUTE_WORDS,
    BOOL_OFF_WORDS,
    BOOL_ON_WORDS,
    DIRECTION_WORDS,
    SYNONYMS,
    camel_split,
    levenshtein,
    ngram_similarity,
    stem,
)
from .vectorize import TfidfIndex, embedder, tokenize
from .lexicon import camel_split
from . import guard

# ---------------------------------------------------------------------------
# F2 type-gate vocabulary (exponential-build-5). The cue-kind agreement
# above is a soft nudge; a polarity verb is a CONSTRAINT. Deliberately
# asymmetric and high-precision: OFF verbs (off/hide/mute/disable/stop)
# are unambiguous, ON verbs like "show" are not ("show temperature in
# fahrenheit" is an enum request), so only off-verbs gate the bool kind.
# Presets and coarse surfaces are NEVER gated: they legitimately combine
# many kinds ("battery saving mode", "optimize for gaming").
# ---------------------------------------------------------------------------

_GATE_BOOL_WORDS = frozenset({
    "off", "hide", "hidden", "hides", "hiding", "mute", "muted", "mutes",
    "disable", "disabled", "disables", "deactivate", "stop", "stops",
    "conceal", "silence", "silenced",
})
_GATE_MOVE_WORDS = frozenset({
    "move", "put", "switch", "place", "position", "relocate",
})

# ON verbs for the Enabled PRIOR only (never the suppressive gate — ON
# verbs are ambiguous about kind: "show temperature in fahrenheit" is an
# enum request). The prior only floors enabled-toggles, so the blast
# radius is "the toggle of the thing you named" — always a safe read of
# an on-verb plus a group noun.
_PRIOR_ON_WORDS = frozenset({
    "on", "enable", "enabled", "enables", "activate", "unmute",
    "show", "display", "reveal",
})

# Words never counted as a tool's PRIMARY atom (the group noun the
# Enabled prior keys on).
_PRIMARY_STOP_WORDS = frozenset({
    "set", "enable", "enabled", "disable", "disabled", "show", "hide",
    "use", "get", "on", "off", "in", "the", "a", "an", "and", "to",
    "of", "at", "when", "until", "my", "with", "up", "down",
})

# A5: enable/disable-style polarity words INSIDE a tool's camel name.
# They carry the toggle's polarity, never its subject, so they are
# stripped when deciding what a request "fully addresses".
_POLARITY_NAME_WORDS = frozenset({
    "enable", "enables", "enabled", "disable", "disables", "disabled",
    "show", "shows", "display", "displays", "hide", "hides", "use",
})

# A5: words that open a prepositional/time TAIL of a request — everything
# after the first of these is location/context, not the request's object.
_PREP_TAIL_WORDS = frozenset({
    "on", "in", "at", "when", "while", "for", "from", "until", "till",
    "during", "near", "of", "behind", "inside", "over", "under", "by",
})

# Words never allowed INSIDE a name bigram (function words only — verb
# words like show/hide/enable are the compound-addressing signal and
# must stay: "show windows" addresses setWorkspacesShowWindows).
_BIGRAM_STOP_WORDS = frozenset({
    "set", "on", "off", "in", "the", "a", "an", "and", "to", "of",
    "at", "when", "until", "my", "with", "up", "down", "use", "get",
})

# A5: a number followed by a NATIVE unit (px) — the request targets
# an absolute property in registry units, so the range check applies.
# Time units are excluded: the planner converts them ("2 seconds" ->
# 2000 ms), so the raw number says nothing about the range.
_UNIT_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:px|pixels?)\b")

# Query-side-only synonyms (NOT in lexicon.SYNONYMS: tool_document expands
# SYNONYMS into the indexed corpus, and the embedder's corpus is
# fingerprint-pinned). Size adjectives whose noun lives in tool names:
# "wider osd hover area" must reach setOsdHoverWidth's "width" atom.
_QUERY_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "wider": ("width",),
    "narrower": ("width",),
    "widen": ("width",),
    # F3 (2026-09-28): "keep more notifications in history" asks for a
    # higher STORED-notification cap (notifs.maxNotifs, whose own noun
    # grammar says "stored notifications"); "keep" is the request's verb
    # for storage. Query-side only, additive, never re-indexed.
    "keep": ("stored",),

    # A5 (2026-09-30) sibling-disambiguation entries. Each cites the
    # upstream fact that justifies the equivalence. Query-side ONLY:
    # the lexicon.SYNONYMS home was tried first and cut — tool_document
    # expands that map into the indexed corpus, and the re-indexing
    # drifted BM25 enough to regress three previously-correct dev items
    # (0.9556 -> 0.9333). These never touch the index.
    # lock.enableFprint / lock.maxFprintTries: the C++ property spells
    # it "fprint" — "fingerprint" never lexically reaches the tool.
    "fingerprint": ("fprint",),
    "fingerprints": ("fprint",),
    # maxFprintTries: "N fingerprint attempts" IS the tries stepper.
    "attempts": ("tries",),
    "attempt": ("tries",),
    # notifications defaultExpireTimeout: a notification that "fades"
    # is one whose expire timeout elapsed.
    "fades": ("expire",),
    "fade": ("expire",),
    # desktopLyricsPosition enum value is "center"; the British
    # spelling never matched the enum word (paired with the A5
    # enum-value cue reading the expanded text).
    "centre": ("center",),
    # slideshowRandom: "shuffle" is the user's word for random order
    # (bar.slideshow.random, Nexus ToggleRow "random order").
    "shuffle": ("random",),
    # wallpaperRecolor(Strength): "tint" is the recolour family's own
    # word (background.wallpaperRecolor). ("colourize"/"colorize" ->
    # "recolour" was tried and CUT: setColorizeMediaGif is itself a
    # registry tool and the expansion degraded its confident route.)
    "tint": ("recolour",),
    "tinting": ("recolour",),
    # audioIncrement/brightnessIncrement: Nexus StepperRow "step" IS
    # the increment control. Measured on the grown dev set: WITH this
    # entry "make the volume step smaller" stays an honest AMBIGUOUS
    # ask; without it the same phrase confidently misroutes to
    # setMaxVolume (a confident-wrong). Restored on that evidence.
    "step": ("increment",),
    "steps": ("increment",),
}

# Query-side-only spelling variants (same reasoning: never touch the
# indexed corpus).
_SPELL_VARIANTS: Dict[str, Tuple[str, ...]] = {
    "visualizer": ("visualiser",),
    "visualiser": ("visualizer",),
}

# Digit -> word normalization for enum vocabulary (F4, D2): "24 hour"
# must reach setClockFormat's serialized enum words ("twenty four hour")
# instead of letting the bare number hijack unrelated numeric tools.
_DIGIT_WORDS = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
    15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen",
    19: "nineteen", 20: "twenty", 30: "thirty", 40: "forty",
    50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety",
}
_DIGIT_RE = re.compile(r"\b(\d{1,2})\s*h\b|\b(\d{1,2})\b")


def _digit_word_tokens(text: str) -> List[str]:
    """Additive word forms of small digits in the query ("24 hour" ->
    "twenty four", "12h" -> "twelve hour"). Only 0..99 with known word
    forms; larger numbers are left alone (they are real values, not
    vocabulary)."""
    out: List[str] = []
    for match in _DIGIT_RE.finditer(text):
        hour_form, plain = match.group(1), match.group(2)
        if hour_form is not None:
            value = int(hour_form)
            words = _DIGIT_WORDS.get(value)
            if words:
                out.extend(words.split())
                out.append("hour")
            continue
        if plain is None:
            continue
        value = int(plain)
        if value <= 20:
            out.append(_DIGIT_WORDS[value])
        elif value % 10 == 0 and value in _DIGIT_WORDS:
            out.append(_DIGIT_WORDS[value])
        elif value < 100:
            tens, ones = (value // 10) * 10, value % 10
            if tens in _DIGIT_WORDS and ones in _DIGIT_WORDS:
                out.extend((_DIGIT_WORDS[tens], _DIGIT_WORDS[ones]))
    return out

# ---------------------------------------------------------------------------
# Router state (the learnable part).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RouterState:
    """Weights and thresholds. Defaults are hand-set priors; ``learn.py``
    fits ``w_lex/w_sem/w_fuzz/w_noun`` and ``bias`` from ledger decisions
    via online logistic regression, and adapts ``min_score``/``min_margin``
    from calibration. Frozen-friendly (dataclasses.replace)."""

    w_lex: float = 0.42
    w_sem: float = 0.22
    w_fuzz: float = 0.14
    w_noun: float = 0.18
    bias: float = 0.04
    min_score: float = 0.30
    min_margin: float = 0.06
    temperature: float = 0.45

    def weights_vector(self) -> Tuple[float, float, float, float, float]:
        return (self.w_lex, self.w_sem, self.w_fuzz, self.w_noun, self.bias)

    def to_dict(self) -> Dict[str, float]:
        return {
            "w_lex": self.w_lex, "w_sem": self.w_sem, "w_fuzz": self.w_fuzz,
            "w_noun": self.w_noun, "bias": self.bias,
            "min_score": self.min_score, "min_margin": self.min_margin,
            "temperature": self.temperature,
        }

    @staticmethod
    def from_dict(data: Optional[Dict[str, float]]) -> "RouterState":
        if not data:
            return RouterState()
        fields = RouterState.__dataclass_fields__
        return RouterState(**{k: float(v) for k, v in data.items() if k in fields})


DEFAULT_STATE = RouterState()


# ---------------------------------------------------------------------------
# Result shapes.
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    surface: str                 # tool name, "preset:<name>", or coarse label
    kind: str                    # "tool" | "preset" | "surface"
    score: float                 # raw hybrid score
    p: float                     # softmax probability
    cues: Dict[str, object]      # value-extraction hints (direction/bool/...)
    evidence: List[str] = field(default_factory=list)


@dataclass
class RouteResult:
    verdict: str                 # ROUTED | AMBIGUOUS | ABSTAIN
    candidates: List[Candidate]
    question: Optional[str] = None
    features: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def top(self) -> Optional[Candidate]:
        return self.candidates[0] if self.candidates else None


# ---------------------------------------------------------------------------
# Deterministic structural layers.
# ---------------------------------------------------------------------------

# Coarse surfaces and their canonical seed documents.
_SURFACE_DOCS: Tuple[Tuple[str, str], ...] = (
    ("explain", "explain why what controls which setting"),
    ("undo", "undo revert rollback restore last change back"),
    ("history", "history recent changes what did i change log"),
    ("scheme", "scheme color colour palette theme accent"),
    ("wallpaper", "wallpaper background image desktop"),
    ("diagnose", "broken crash error not working fails troubleshoot problem"),
    ("search", "docs documentation how do i find look up"),
    ("brain", "plan tasks focus estimate organize notes remind"),
    ("issue", "bug report issue draft file"),
    # genius: the universal intelligence layer (math/logic/probability/
    # statistics/decision/text-analysis questions). Seeded deliberately
    # with vocabulary that no settings tool owns, so a settings phrase
    # can never lose to a coincidental genius word.
    ("genius", "solve calculate compute derivative integral equation root "
               "probability bayes matrix eigenvalue tautology truth table "
               "permutation regression correlation outlier average numbers"),
    # agent: the consent-gated orchestrator over every layer. Goal-shaped
    # requests ("clean my downloads", "tidy up my files", "audit my
    # packages") name filesystem/maintenance work no settings tool and no
    # single genius domain owns — the agent composes across layers, always
    # starting from a simulated task graph. Seeded with archetype nouns
    # (clean/tidy/organize/downloads/backup/audit), deliberately disjoint
    # from the brain doc's note-taking vocabulary (notes/journal/remind)
    # so "organize my notes" stays a brain request while "organize my
    # downloads" is agent-shaped.
    ("agent", "clean tidy declutter downloads backups archive audit "
              "packages my files folder go through step by step checklist "
              "multi-step workflow goal do everything"),
)

# Pattern boosts: (regex, surface, floor). A matching pattern floors the
# candidate's score at the given value — grammar beats statistics, so a
# why-question can never lose to a coincidental noun overlap. "what did"
# is deliberately NOT a why-question ("what did i change" is history).
_WHY_RE = re.compile(r"^\s*(?:why\s+(?:is|are|was|were|does|do|did)|what\s+(?:is|are|does|do))\b")
_EXPLAIN_HINT_RE = re.compile(r"\b(?:explain|what\s+controls|which\s+setting|what\s+setting)\b")
_UNDO_RE = re.compile(r"^\s*(?:undo|revert|roll\s?back|restore)\b")
_HISTORY_RE = re.compile(r"\b(?:what\s+did\s+i\s+change|change\s+history|my\s+changes|show\s+history|\bhistory\b)\b")
# genius shapes: arithmetic, equations, and named math/logic questions are
# never settings requests — high-precision only, so no settings phrase
# can trip it by accident.
_GENIUS_RE = re.compile(r"\d\s*[+\-*/^%]\s*\d|\d+(?:\.\d+)?\s*%\s*(?:of|off)\s*\d+|"
                        r"\bsolve\b|\bderivative\b|\bintegral\b|"
                        r"\btaylor\b|\btautolog\w*\b|\bsatisfiable\b|\btruth table\b|"
                        r"\bbayes\b|\beigenvalue\b|\bhow many ways\b")

# Agent sequencing grammar: "... then ..." / "after that" / "first ... then"
# / "step by step" signal a TIME-ORDERED multi-step goal spanning more than
# one clause. The compound splitter splits on "then" and routes each side
# independently, which is right for "disable blur then move the dock" (two
# settings ops, one plan) but wrong for "clean my downloads then make the
# shell minimal" — the sequence IS the request. This regex is the
# structural cue the pipeline consults on the FULL resolved text (before
# the splitter shreds it); high-precision sequencing vocabulary only, so
# simultaneous conjunction ("and") is never agent-shaped by accident.
AGENT_SEQ_RE = re.compile(
    r"\bthen\b|\bafter\s+that\b|\bafterwards\b|\bstep\s+by\s+step\b|"
    r"\bfirst\b[^.!?]{0,80}\bthen\b|\bone\s+by\s+one\b|\bdirectly\s+after\b"
)

# Lock-on-startup grammar (D1/p22): "lock the screen when the shell
# starts" names setLockOnStartup — a lock word near a start/boot/login
# word is startup-lock grammar, decisive over the "screen" noun that
# otherwise drags in setShowScreenRecorder.
_LOCK_START_RE = re.compile(
    r"\block\b[^.!?]{0,40}\b(?:start(?:s|ing)?|boot|launch(?:es|ing)?|log\s?in)\b"
)

# F6 question shape: a leading question word or a trailing question mark.
# Questions are not change requests unless they carry a value cue —
# "can you make the bar thinner?" carries one and plans normally.
_QUESTION_RE = re.compile(
    r"^\s*(?:why|what|when|where|who|how|which|whose|is|are|can|could|"
    r"does|do|did|will|would|should)\b|\?\s*$"
)

PATTERN_BOOSTS: Tuple[Tuple[re.Pattern[str], str, float], ...] = (
    (_WHY_RE, "explain", 0.78),
    (_EXPLAIN_HINT_RE, "explain", 0.66),
    (_UNDO_RE, "undo", 0.78),
    (_HISTORY_RE, "history", 0.80),
    (_GENIUS_RE, "genius", 0.82),
    (_LOCK_START_RE, "setLockOnStartup", 0.80),
    # Sequencing grammar floors the agent surface on any route() call that
    # still sees the connective (single-clause requests and whole-text
    # calls). The pipeline applies the same regex pre-split for compound
    # requests — one mechanism, both scopes.
    (AGENT_SEQ_RE, "agent", 0.84),
)

# Preset triggers (parser.py §3.7 precedence: preset BEFORE per-tool
# nouns — "make my bar compact" IS the compact preset there, and here).
# Floors sit above every tool-level hybrid (noun floor + BM25 + cue
# agreement tops out ~0.8) so preset precedence is decisive, mirroring
# the frozen grammar's own ordering.
_PRESET_TRIGGERS: Tuple[Tuple[re.Pattern[str], str, float], ...] = (
    (re.compile(r"\b(?:more\s+compact|compact|denser|tighter)\b"), "preset:compact", 0.88),
    (re.compile(r"\b(?:minimalist|minimal\s+look|minimal\s+setup|minimal|cleaner\s+look|simpler\s+look|declutter)\b"), "preset:minimal", 0.88),
    (re.compile(r"\b(?:gaming|game\s+mode\s+look|optimize\s+for\s+gaming|for\s+games?)\b"), "preset:gaming", 0.86),
    (re.compile(r"\b(?:battery|power\s+sav(?:ing|er))\b"), "preset:battery-saver", 0.86),
    (re.compile(r"\b(?:macos|mac\s?os|apple\s+style|more\s+like\s+apple)\b"), "preset:macos-like", 0.86),
)

# Position vocabulary (parser.py §3.4) — a cue AND enum vocabulary.
_POSITION_RE = re.compile(r"\b(top|bottom|left|right)\b(?:\s+(?:edge|side))?")
_POSITION_VALUES = frozenset({"top", "bottom", "left", "right"})

# "Move the dock/bar to the left" — issue #120's own example. A position
# word together with a shell-surface noun (dock/bar/panel/taskbar/shell)
# is position-setting grammar, decisive over size tooling on the same noun.
_SURFACE_POS_RE = re.compile(
    r"\b(?:dock|bar|panel|taskbar|shell)\b[^.!?]{0,40}\b(?:left|right|top|bottom)\b"
    r"|\b(?:left|right|top|bottom)\b[^.!?]{0,40}\b(?:dock|bar|panel|taskbar|shell)\b"
)
_BAR_POSITION_FLOOR = 0.80

# Scheme/wallpaper SYSTEM requests (parser.py §3.3 step 3's dead-ends):
# "change my wallpaper" / "new background" live OUTSIDE shell.json and
# must floor the inert-suggestion surfaces over the wallpaper-ADJACENT
# tools (recolor strength, wallpaper enabled...). A qualifier word
# (recolor/recolour/strength/saturation) means the shell.json tool and
# suppresses the system floor.
_SCHEME_SYS_RE = re.compile(
    r"\b(?:change|set|new|switch|pick|choose|different|random)\b[^.!?]{0,30}\b"
    r"(?:scheme|schemes|colou?rs?|palette|palettes|theme|themes|accent|accents)\b"
    r"|\b(?:scheme|colou?rs?|palette|theme|accent)s?\b[^.!?]{0,30}\b(?:change|new|switch)\b"
)
_WALLPAPER_SYS_RE = re.compile(
    r"\b(?:change|set|new|switch|random|shuffle|pick|choose|different)\b[^.!?]{0,30}\b"
    r"(?:wallpapers?|backgrounds?)\b"
    r"|\b(?:wallpapers?|backgrounds?)\b[^.!?]{0,30}\b(?:change|new|switch|random|shuffle)\b"
)
_SYS_QUALIFIER_RE = re.compile(r"\b(?:recolou?r|strength|saturation|intensity|enabled|disable|enable)\b")
_SCHEME_FLOOR = 0.74
_WALLPAPER_FLOOR = 0.74

# Cue-kind agreement factors (multiples of w_noun).
_K_DIRECTION_NUMERIC = 0.55     # direction cue + float/int tool: agree
_K_DIRECTION_OTHER = -0.35      # direction cue + non-numeric: disagree
_K_BOOL_BOOL = 0.55             # on/off cue + bool tool: agree
_K_BOOL_OTHER = -0.25           # on/off cue + non-bool: disagree
_K_POSITION_ENUM = 0.65         # position cue/word + enum tool: agree
_K_POSITION_OTHER = -0.30       # position cue + non-enum: disagree
_K_ENUM_WORD = 0.60             # literal enum value present: agree
_K_NUMBER_NUMERIC = 0.30        # explicit number/percent + numeric: agree
_K_TRANSPARENCY_AGREE = 0.90    # see-through/translucent cue + transparency tool (property words
                                 # determine the tool class; they must be decisive)
_K_TRANSPARENCY_OTHER = -0.50   # see-through/translucent cue + unrelated tool

# Transparency vocabulary: multi-word cue the single-word DIRECTION scan
# cannot see, with its own tool class (path or nouns mention
# transparency/opacity — the property words determine the tool class;
# surface nouns like "panels" only refine which thing).
_TRANSPARENCY_CUE_RE = re.compile(r"\b(?:see\s*-?\s*through|translucent|transparent|transparency)\b")
_TRANSPARENCY_TOOL_RE = re.compile(r"(?:^|\.)(?:transparency|opacity)|transparency|opacity", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Candidate universe + per-surface documents.
# ---------------------------------------------------------------------------


def _enum_value_words(spec) -> List[str]:
    """Serialized enum values as addressable vocabulary. Camel humps are
    split ("TwentyFourHour" -> "twenty four hour", "timeOfDay" -> "time
    of day") so a query's words can actually reach enum vocabulary — a
    fused "twentyfourhour" token is unreachable by any real phrasing
    (F4/D2: "use 24 hour time" must reach setClockFormat)."""
    if spec.kind != "enum" or not spec.enum:
        return []
    out: List[str] = []
    for value in spec.enum:
        text = str(value).replace("_", " ").replace("-", " ")
        for part in text.split():
            out.extend(w.lower() for w in camel_split(part) if w)
    return [w for w in out if w]


def _candidate_documents() -> Dict[str, Tuple[str, str]]:
    """{surface: (document, kind)} for the full candidate universe.
    Preset documents are built from the registry's own preset metadata
    (name, label, description, called tool names) — mechanical facts,
    no invented vocabulary."""
    out: Dict[str, Tuple[str, str]] = {}
    for spec in TOOL_SPECS:
        doc = tool_document(spec)
        enum_words = _enum_value_words(spec)
        if enum_words:
            doc = doc + " " + " ".join(enum_words)
        out[spec.name] = (doc, "tool")
    for preset in PRESETS:
        name = str(preset.get("name", ""))
        label = str(preset.get("label", ""))
        description = str(preset.get("description", ""))
        calls = " ".join(str(tool) for tool, _value in preset.get("calls", ()))
        out[f"preset:{name}"] = (f"{name} {label} {description} {calls}", "preset")
    for surface, doc in _SURFACE_DOCS:
        out[surface] = (doc, "surface")
    return out


# Noun regexes of the 18 core tools (parser.py's own grammar, rebuilt
# here from the registry so the two surfaces can never drift).
_NOUN_RES: Dict[str, List[re.Pattern[str]]] = {
    spec.name: [re.compile(r"\b(?:" + group + r")\b") for group in spec.nouns]
    for spec in TOOL_SPECS if spec.nouns
}


def _noun_hit(text: str) -> List[str]:
    """Tools whose noun groups ALL appear (the parser.py rule)."""
    hits = []
    for name, regexes in _NOUN_RES.items():
        if all(rx.search(text) for rx in regexes):
            hits.append(name)
    return hits


# ---------------------------------------------------------------------------
# Query expansion: typo correction + synonyms (incl. hyphen bigrams).
# ---------------------------------------------------------------------------

_WORD_SPLIT_RE = re.compile(r"[a-z]+")


def _typo_fix(text: str, vocab: Sequence[str], max_distance: int = 1) -> Tuple[str, List[str]]:
    """Bounded-edit-distance query correction against the indexed
    vocabulary. Distance is deliberately 1 (the dominant real-typo case,
    "transparncy" -> "transparency") and words shorter than 5 characters
    are never touched — at distance 2 short garbage words start matching
    real vocabulary ("flurb" -> "blur") and the router would route pure
    noise. Only replaces a word when the correction is UNIQUE (ambiguous
    corrections are left alone — honest abstention beats confident
    mangling)."""
    vocab_set = set(vocab)
    words = _WORD_SPLIT_RE.findall(text.lower())
    fixed: List[str] = []
    evidence: List[str] = []
    for word in words:
        # A word whose STEM is already indexed vocabulary ("panels" with
        # "panel" indexed) is not a typo — the tokenizer stems both sides
        # anyway; "correcting" it would only pollute the evidence log.
        if word in vocab_set or stem(word) in vocab_set or len(word) <= 4:
            fixed.append(word)
            continue
        candidates: List[Tuple[int, str]] = []
        for cand in vocab:
            if abs(len(cand) - len(word)) > max_distance:
                continue
            d = levenshtein(word, cand, cap=max_distance)
            if d <= max_distance:
                candidates.append((d, cand))
        if len(candidates) == 1:
            best = candidates[0][1]
            evidence.append(f"typo: {word} -> {best} (edit distance {candidates[0][0]})")
            fixed.append(best)
        else:
            fixed.append(word)
    return " ".join(fixed), evidence


def _expand_query(text: str) -> Tuple[str, List[str]]:
    """Query-side expansion: typo fix, then synonym mapping of single
    words AND adjacent-pair bigrams ("see through" -> transparency via
    the "see-through" lexicon key), plus digit->word normalization for
    enum vocabulary ("24 hour" -> "twenty four hour"). Original words
    are KEPT — synonyms are additive, never replacements."""
    vocab = embedder().vocab
    fixed, evidence = _typo_fix(text, vocab)
    words = _WORD_SPLIT_RE.findall(fixed)
    expanded: List[str] = list(words)
    for token in _digit_word_tokens(fixed):
        if token not in expanded:
            expanded.append(token)
            if token != "hour":
                evidence.append(f"digit: {token}")
    for word in words:
        for key in (word, stem(word)):
            for mapped in SYNONYMS.get(key, ()):
                if mapped not in expanded:
                    expanded.append(mapped)
                    evidence.append(f"synonym: {word} -> {mapped}")
        for mapped in _SPELL_VARIANTS.get(word, _SPELL_VARIANTS.get(stem(word), ())):
            if mapped not in expanded:
                expanded.append(mapped)
                evidence.append(f"spelling: {word} -> {mapped}")
        for mapped in _QUERY_SYNONYMS.get(word, _QUERY_SYNONYMS.get(stem(word), ())):
            if mapped not in expanded:
                expanded.append(mapped)
                evidence.append(f"synonym(query-side): {word} -> {mapped}")
    for a, b in zip(words, words[1:]):
        for joined in (f"{a}-{b}", f"{a}{b}"):
            for mapped in SYNONYMS.get(joined, ()):
                if mapped not in expanded:
                    expanded.append(mapped)
                    evidence.append(f"synonym: {a} {b} -> {mapped}")
    return " ".join(expanded), evidence


# ---------------------------------------------------------------------------
# Cue extraction (value hints — hints only, never values).
# ---------------------------------------------------------------------------


def extract_cues(text: str) -> Dict[str, object]:
    """Value-extraction hints for the pipeline: direction, bool intent,
    absolute targets, reset/toggle requests, position words, plain
    numbers and percents. Mirrors parser.py's scanners where they
    overlap; used for the 259 noun-silent tools the frozen grammar
    cannot parse.

    Weak-bool rule: "show fewer notifications" is a MAGNITUDE statement,
    not an on/off statement — when a direction cue is present, bool cues
    derived from weak verbs (show/hide/display/reveal) are dropped; only
    strong verbs (on/off/enable/disable/activate/deactivate) survive.
    """
    cues: Dict[str, object] = {}
    lowered = text.lower()
    direction = 0
    for word in _WORD_SPLIT_RE.findall(lowered):
        direction += DIRECTION_WORDS.get(word, DIRECTION_WORDS.get(stem(word), 0))
    if direction:
        cues["direction"] = 1 if direction > 0 else -1
    words = set(_WORD_SPLIT_RE.findall(lowered))
    on = sorted(BOOL_ON_WORDS.intersection(words))
    off = sorted(BOOL_OFF_WORDS.intersection(words))
    if cues.get("direction"):
        # keep only STRONG bool words under a direction cue
        strong = {"on", "enable", "enabled", "activate", "off", "disable", "disabled", "deactivate"}
        on = [w for w in on if w in strong]
        off = [w for w in off if w in strong]
    if on and not off:
        cues["bool"] = True
    elif off and not on:
        cues["bool"] = False
    elif on and off:
        cues["bool"] = "mixed"
    for word, value in ABSOLUTE_WORDS.items():
        if word in lowered:
            cues["absolute"] = value
            break
    if re.search(r"\b(?:reset|default|factory)\b", lowered):
        cues["reset"] = True
    if re.search(r"\btoggle\b", lowered):
        cues["toggle"] = True
    position = _POSITION_RE.search(lowered)
    if position:
        cues["position"] = position.group(1)
    if _TRANSPARENCY_CUE_RE.search(lowered):
        cues["transparency"] = True
    # F4: percent words, not just the % sign — "20 percent smaller" is a
    # percent cue; the bare number must not also register as a value
    # (that is how numbers hijacked unrelated numeric tools, D2).
    pct = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent|pct|per\s?cent)(?![a-z])", lowered)
    if pct:
        cues["percent"] = float(pct.group(1))
    else:
        num = re.search(r"\b-?\d+(?:\.\d+)?\b", lowered)
        if num:
            cues["number"] = float(num.group(0))
    # F4: strong polarity verbs the parser's BOOL vocabulary does not
    # carry ("stop", "mute") still communicate on/off intent; weak
    # on-verbs (show/display/reveal) stay out — they are too often
    # magnitude or enum phrasings.
    if "bool" not in cues:
        words = set(_WORD_SPLIT_RE.findall(lowered))
        if words & _GATE_BOOL_WORDS:
            cues["bool"] = False
        elif words & {"on", "enable", "enabled", "activate", "unmute"}:
            cues["bool"] = True
    # F4: absolute words hyphen-normalized — "pitch black" must reach the
    # "pitch-black" key.
    if "absolute" not in cues:
        squashed = re.sub(r"\s+", "-", lowered)
        for word, value in ABSOLUTE_WORDS.items():
            if word in squashed:
                cues["absolute"] = value
                break
    return cues


# ---------------------------------------------------------------------------
# The router.
# ---------------------------------------------------------------------------


class Router:
    """One router instance = one candidate universe + one index. Build
    once per process (module-level ``router()``); route as often as
    needed. ``route()`` is a pure function of (text, state)."""

    def __init__(self) -> None:
        self.documents = _candidate_documents()
        self.index = TfidfIndex({k: doc for k, (doc, _kind) in self.documents.items()})
        self.embedder = embedder()
        self.doc_vectors = {k: self.embedder.embed(doc) for k, (doc, _kind) in self.documents.items()}
        # F3 guard input: stemmed document tokens per surface (the guard
        # checks which surfaces actually address the request's content
        # words). Built once; frozen for determinism.
        self._doc_stem_cache = {
            k: frozenset(stem(t) for t in self.index.doc_tokens.get(k, ()))
            for k in self.documents
        }
        # Fuzzy-matching surface: the addressable name atoms per candidate
        # (tool atoms from the registry; preset/surface names split as words).
        # name_atom_sets feeds the coverage signal: a tool whose compound
        # name atoms ALL appear in the query is addressing that tool
        # specifically, which outranks a generic single-noun hit.
        self.name_atoms: Dict[str, str] = {}
        self.name_atom_sets: Dict[str, frozenset] = {}
        for spec in TOOL_SPECS:
            atoms = tool_atoms(spec)
            self.name_atoms[spec.name] = " ".join(atoms)
            self.name_atom_sets[spec.name] = frozenset(atoms)
        for key, (_doc, kind) in self.documents.items():
            if kind != "tool" and key not in self.name_atoms:
                words = camel_split(key.replace("preset:", ""))
                self.name_atoms[key] = " ".join(words)
                self.name_atom_sets[key] = frozenset(words)
        self.spec_by_name = {spec.name: spec for spec in TOOL_SPECS}
        self.enum_words: Dict[str, List[str]] = {
            spec.name: _enum_value_words(spec) for spec in TOOL_SPECS if spec.kind == "enum"
        }
        # F2 structural addressing surfaces (exponential-build-5):
        # - primary atom: the tool's group noun (first non-stop camel word),
        #   what the Enabled prior keys on ("disable the launcher" ->
        #   setLauncherEnabled, whose primary atom "launcher" is present).
        # - name bigrams: adjacent camel-word pairs of the tool name; a
        #   bigram appearing ADJACENT in the query ("show windows") is
        #   compound addressing of setWorkspacesShowWindows — stronger
        #   than any single-word overlap.
        self.primary_atom: Dict[str, str] = {}
        self.post_prefix_atoms: Dict[str, Tuple[str, ...]] = {}
        self.name_bigrams: Dict[str, Tuple[Tuple[str, str], ...]] = {}
        # A5: the toggle's SUBJECT atoms — the camel name minus its
        # enable/disable-style polarity words. "turn off the overview"
        # must fully address setOverviewEnabled ([overview]) and NOT
        # floor setEnableOverviewBlur ([overview, blur], blur unaddressed).
        for spec in TOOL_SPECS:
            words = [w.lower() for w in camel_split(spec.name)]
            primary = next((w for w in words if w not in _PRIMARY_STOP_WORDS), "")
            if primary:
                self.primary_atom[spec.name] = primary
            post_prefix = tuple(
                w for w in words
                if w not in _PRIMARY_STOP_WORDS
                and w not in _POLARITY_NAME_WORDS)
            if post_prefix:
                self.post_prefix_atoms[spec.name] = post_prefix
            stemmed = [stem(w) for w in words if w not in {"set"}]
            bigrams = tuple(
                (a, b) for a, b in zip(stemmed, stemmed[1:])
                if a not in _BIGRAM_STOP_WORDS and b not in _BIGRAM_STOP_WORDS
            )
            if bigrams:
                self.name_bigrams[spec.name] = bigrams

    # -- structural layers --------------------------------------------------

    @staticmethod
    def _is_enabled_tool(name: str) -> bool:
        """An "on/off toggle for the thing it names": setXEnabled,
        setEnableX, or path *.enabled. Exclude setDisable* tools (their
        polarity is inverted and they gate on different vocabulary)."""
        from assistant.adapters.caelestia.registry import tool_by_name
        spec = tool_by_name(name)
        if spec is None or spec.kind != "bool":
            return False
        if spec.name.startswith("setDisable"):
            return False
        return (
            spec.name.endswith("Enabled")
            or spec.name.startswith("setEnable")
            or spec.path.endswith(".enabled")
        )

    def _pattern_floor(self, raw: str) -> Dict[str, float]:
        """{surface: floor} from pattern boosts, preset triggers, the
        position grammar (a position word + shell-surface noun floors
        setBarPosition — issue #120's "move the dock to the left"), and
        the scheme/wallpaper system dead-ends (floored only when no
        shell.json qualifier word is present)."""
        floors: Dict[str, float] = {}
        for regex, surface, floor in PATTERN_BOOSTS:
            if regex.search(raw):
                floors[surface] = max(floors.get(surface, 0.0), floor)
        for regex, surface, floor in _PRESET_TRIGGERS:
            if regex.search(raw):
                floors[surface] = max(floors.get(surface, 0.0), floor)
        if _SURFACE_POS_RE.search(raw):
            floors["setBarPosition"] = max(floors.get("setBarPosition", 0.0), _BAR_POSITION_FLOOR)
        if not _SYS_QUALIFIER_RE.search(raw):
            if _SCHEME_SYS_RE.search(raw):
                floors["scheme"] = max(floors.get("scheme", 0.0), _SCHEME_FLOOR)
            if _WALLPAPER_SYS_RE.search(raw):
                floors["wallpaper"] = max(floors.get("wallpaper", 0.0), _WALLPAPER_FLOOR)
        return floors

    def _cue_kind_delta(self, cues: Dict[str, object], surface: str, raw: str,
                        expanded: str = "") -> Tuple[float, List[str]]:
        """Cue-kind agreement adjustment for one candidate. Returns the
        score delta (a multiple of w_noun) plus evidence strings."""
        spec = self.spec_by_name.get(surface)
        if spec is None:  # presets/surfaces: no kind, no adjustment
            return 0.0, []
        kind = spec.kind
        delta = 0.0
        evidence: List[str] = []
        has_direction = "direction" in cues
        has_bool = cues.get("bool") in (True, False)
        has_position = "position" in cues
        has_transparency = bool(cues.get("transparency"))
        if has_transparency:
            if _TRANSPARENCY_TOOL_RE.search(spec.path) or any(
                "transparency" in g or "opacity" in g for g in spec.nouns
            ):
                delta += _K_TRANSPARENCY_AGREE
                evidence.append("see-through cue agrees with transparency setting")
            else:
                delta += _K_TRANSPARENCY_OTHER
        if has_direction and kind in ("float", "int"):
            delta += _K_DIRECTION_NUMERIC
            evidence.append("direction cue agrees with numeric setting")
        elif has_direction and kind not in ("float", "int"):
            delta += _K_DIRECTION_OTHER
        if has_bool and kind == "bool":
            delta += _K_BOOL_BOOL
            evidence.append("on/off cue agrees with toggle setting")
        elif has_bool and kind != "bool":
            delta += _K_BOOL_OTHER
        if has_position and kind == "enum":
            delta += _K_POSITION_ENUM
            evidence.append(f"position word agrees with enum setting ({cues['position']})")
        elif has_position and kind != "enum":
            delta += _K_POSITION_OTHER
        if kind == "enum":
            for word in self.enum_words.get(surface, ()):  # literal enum value present
                # A5: match on the EXPANDED text, not just the raw query —
                # "centre" only becomes the enum value "center" through the
                # synonym map, which is exactly the hint surface this cue
                # is allowed to consume (same policy as noun hits).
                if re.search(rf"\b{re.escape(word)}\b", raw):
                    delta += _K_ENUM_WORD
                    evidence.append(f"enum value '{word}' present")
                    break
                if re.search(rf"\b{re.escape(word)}\b", expanded):
                    delta += _K_ENUM_WORD
                    evidence.append(f"enum value '{word}' present (expanded)")
                    break
        if ("number" in cues or "percent" in cues) and kind in ("float", "int"):
            delta += _K_NUMBER_NUMERIC
            evidence.append("explicit number agrees with numeric setting")
        return delta, evidence

    # -- the main entry ------------------------------------------------------

    def route(self, text: str, state: RouterState = DEFAULT_STATE, k: int = 5) -> RouteResult:
        if not text or not text.strip():
            return RouteResult(verdict="ABSTAIN", candidates=[], question="say what you want to change")
        raw = text.lower().strip()

        # Query expansion (evidence collected for the candidate cards).
        expanded, evidence = _expand_query(raw)

        # Lexical: WEIGHTED dual BM25+ — the RAW query carries the user's
        # actual words (weight 0.7), the EXPANDED query carries typo fixes
        # and curated synonyms (weight 0.3). Synonyms are hints, never
        # replacements, so synonym flooding cannot drown the original
        # intent ("see-through panels" must not become a bar query just
        # because "panels" expands to bar/taskbar/panel).
        q_raw = tokenize(raw)
        q_tokens = tokenize(expanded)
        bm25_raw = {key: self.index.score(q_raw, key) for key in self.documents}
        bm25_exp = {key: self.index.score(q_tokens, key) for key in self.documents}
        raw_max = max(bm25_raw.values()) if bm25_raw else 0.0
        exp_max = max(bm25_exp.values()) if bm25_exp else 0.0

        # Semantic: PPMI cosine between expanded query and doc vectors.
        q_vec = self.embedder.embed(expanded)

        # Structural: noun hits on the EXPANDED text (the curated synonym
        # lexicon is the moral equivalent of extending the noun grammar —
        # "see-through" hits the transparency nouns once expanded),
        # pattern floors, cues, and atom COVERAGE (all of a compound
        # tool's name atoms present = specific addressing).
        noun_hits = set(_noun_hit(expanded))
        floors = self._pattern_floor(raw)
        cues = extract_cues(raw)
        expanded_stems = frozenset(tokenize(expanded))
        # A5 object span: the expanded tokens BEFORE the first
        # prepositional/time tail ("... on the desktop", "... while
        # charging"). The Enabled prior may only floor a toggle whose
        # primary atom the request actually NAMES as its object —
        # a location tail is not the object.
        _object_span: List[str] = []
        for _tok in tokenize(expanded):
            if _tok in _PREP_TAIL_WORDS:
                break
            _object_span.append(_tok)
        object_span_stems = frozenset(stem(w) for w in _object_span)
        coverage_hits: Dict[str, float] = {}
        for key, atoms in self.name_atom_sets.items():
            if len(atoms) < 2:
                continue
            matched = sum(1 for atom in atoms if stem(atom) in expanded_stems or atom in expanded_stems)
            if matched >= 2:
                coverage_hits[key] = matched / len(atoms)

        w_lex, w_sem, w_fuzz, w_noun, bias = state.weights_vector()

        # F2 gate flags (exponential-build-5): polarity verbs constrain
        # candidate kind. Computed from the RAW text (typo fix and synonym
        # expansion are hints, never gate triggers).
        raw_words = _WORD_SPLIT_RE.findall(raw)
        raw_stems = {stem(w) for w in raw_words}
        is_question = bool(_QUESTION_RE.search(raw))
        why_question = bool(_WHY_RE.search(raw) or _EXPLAIN_HINT_RE.search(raw))
        gate_bool = any(
            w in _GATE_BOOL_WORDS or stem(w) in _GATE_BOOL_WORDS
            for w in raw_words
        )
        gate_enum = (
            any(w in _GATE_MOVE_WORDS or stem(w) in _GATE_MOVE_WORDS
                for w in raw_words)
            and bool(cues.get("position"))
        )
        gate_numeric = bool(cues.get("direction")) and not gate_bool
        if gate_numeric:
            # Participle forms ("expanded", "enlarged", "minimized") are
            # STATE descriptions, not magnitude requests — "notifications
            # should open expanded" is a bool request about the expanded
            # state. The soft cue-kind nudge keeps them; the GATE does not.
            gate_numeric = any(
                (w in DIRECTION_WORDS)
                or (stem(w) in DIRECTION_WORDS
                    and not (w.endswith("ed") or w.endswith("ing")))
                for w in raw_words
            )
        prior_polarity = gate_bool or any(
            w in _PRIOR_ON_WORDS or stem(w) in _PRIOR_ON_WORDS
            for w in raw_words
        )
        # Adjacent stemmed word pairs of the raw query (bigram floor).
        raw_seq = [stem(w) for w in raw_words]
        query_bigrams = {(raw_seq[i], raw_seq[i + 1]) for i in range(len(raw_seq) - 1)}

        # F2 specificity map: for each primary atom, the best name-atom
        # coverage any tool with that primary achieves on this query. The
        # Enabled prior must not fire when a MORE SPECIFIC sibling of the
        # same group noun is better addressed ("turn off the charging
        # sound" names setSoundsChargingStarted, not the whole sounds
        # toggle).
        _best_cov_by_primary: Dict[str, float] = {}
        for key, atoms in self.name_atom_sets.items():
            if key not in self.primary_atom:
                continue
            cov = coverage_hits.get(key, 0.0)
            if cov > _best_cov_by_primary.get(self.primary_atom[key], 0.0):
                _best_cov_by_primary[self.primary_atom[key]] = cov

        scored: List[Tuple[float, str]] = []
        features: Dict[str, Dict[str, float]] = {}
        # Capability-1 phonetic channel: per-tool soundex hit rate over the
        # query's content tokens (0.0 for value-only requests). Lifts
        # typo'd/misheard atoms without widening the lexical channel.
        raw_content = [w for w in raw_words if len(w) > 2]
        _ch_lex: Dict[str, float] = {}
        _ch_sem: Dict[str, float] = {}
        _ch_fuzz: Dict[str, float] = {}
        for key, (doc, kind) in self.documents.items():
            lex_raw = (bm25_raw[key] / raw_max) if raw_max > 0 else 0.0
            lex_exp = (bm25_exp[key] / exp_max) if exp_max > 0 else 0.0
            lex = 0.7 * lex_raw + 0.3 * lex_exp
            sem = self.embedder.cosine(q_vec, self.doc_vectors[key])
            fuzz = ngram_similarity(raw, self.name_atoms.get(key, key))
            phon = phonetic_hit_rate(raw_content,
                                     self.name_atom_sets.get(key, ()))
            # additive blend (NOT max): the phonetic lift must not erase
            # the ngram channel's ordering among same-soundex siblings
            # ("default" vs "fullscreen" expire timeouts differ exactly
            # there).
            fuzz = min(1.0, fuzz + 0.15 * phon)
            noun = 1.0 if key in noun_hits else 0.0
            score = w_lex * lex + w_sem * sem + w_fuzz * fuzz + w_noun * noun + bias
            _ch_lex[key] = lex
            _ch_sem[key] = sem
            _ch_fuzz[key] = fuzz
            # F2: the structural lift (noun floor, pattern floors, coverage
            # floor, Enabled prior, name bigram) is tracked per candidate
            # and exposed as the "struct" feature so the learner's student
            # models can see what actually drove the score.
            score_base = score
            if key in noun_hits:
                # A noun hit is decisive by construction: floor the hybrid
                # at a level vague lexical overlap cannot reach.
                score = max(score, w_lex + w_noun + bias)
            if key in floors:
                score = max(score, floors[key])
            # Atom coverage: 2+ of a compound tool's own name atoms present
            # is SPECIFIC addressing — full coverage is noun-hit-equivalent,
            # partial coverage (>= half) still earns a solid bonus. This is
            # what lets "performance show text" beat a generic "text" noun
            # hit on the font tool.
            coverage = coverage_hits.get(key, 0.0)
            if coverage >= 0.999:
                # Full name-atom coverage is SPECIFIC addressing: every
                # word of the tool's own name is present. That outranks
                # any generic lexical overlap a sibling tool can muster
                # ("let the bar hide until I hover" must beat the
                # greeter's coincidental "hover" lexical mass).
                if kind == "tool":
                    score = max(score, 0.85)
                else:
                    score = max(score, w_lex + w_noun + bias)
            elif coverage >= 0.5:
                score += 0.5 * w_noun
            # F2 Enabled prior: a polarity verb plus the tool's group
            # noun floors its Enabled toggle ("disable the launcher" ->
            # setLauncherEnabled, "mute all shell sounds" ->
            # setSoundsEnabled). The noun grammar cannot hit these
            # noun-silent toggles; the polarity verb is the disambiguator.
            # (A5 tried object-span + full-post-prefix-addressing
            # refinements here; they LOST on the dev arena — 0.9222 vs
            # 0.9556 top1 — and were cut per the merge gate.)
            if prior_polarity and key in self.primary_atom and self._is_enabled_tool(key):
                primary = self.primary_atom[key]
                if (primary in expanded_stems or stem(primary) in expanded_stems) \
                        and coverage_hits.get(key, 0.0) >= \
                        _best_cov_by_primary.get(primary, 0.0):
                    score = max(score, 0.80)
            # F2 name-bigram floor: an adjacent camel-word pair of the
            # tool's own name appearing verbatim ("show windows") is
            # compound addressing — "show windows in the workspace
            # indicators" is setWorkspacesShowWindows, not the
            # active-indicator tool one lexical word closer.
            for bigram in self.name_bigrams.get(key, ()):
                if bigram in query_bigrams:
                    score = max(score, 0.82)
                    break
            struct_lift = max(0.0, score - score_base)
            cue_delta, cue_evidence = self._cue_kind_delta(cues, key, raw, expanded)
            score += cue_delta * w_noun
            # A5 value-range agreement (hard, post-floor): an explicit
            # NUMBER WITH A UNIT ("set the corner rounding to 15 px")
            # targets a property in that unit; a candidate whose
            # validated registry range cannot accept the number would be
            # REJECTED by the planner ("15" is no rounding *scale* —
            # that lives in 0.5..2.0). A confident route the planner
            # will refuse is the confident-wrong pattern in its purest
            # form, so the mismatch subtracts from the FINAL score —
            # noun floors included — not as a soft cue. Percent-shaped
            # requests ("120 percent") are exempt: the planner
            # normalizes them, so the raw number says nothing about the
            # range. Deterministic, registry-cited, evidence-carrying.
            if "number" in cues and kind == "tool":
                spec = self.spec_by_name.get(key)
                if spec is not None and spec.kind in ("float", "int") \
                        and spec.minimum is not None \
                        and spec.maximum is not None \
                        and _UNIT_NUMBER_RE.search(raw):
                    value = float(cues["number"])
                    if not (spec.minimum <= value <= spec.maximum):
                        score -= 0.5
                        cue_evidence.append(
                            f"value {value:g} outside {spec.name}'s "
                            f"validated range {spec.minimum:g}.."
                            f"{spec.maximum:g}")
            scored.append((score, key))
            features[key] = {"lex": round(lex, 4), "sem": round(sem, 4),
                             "fuzz": round(fuzz, 4), "noun": noun,
                             "phon": round(phon, 4),
                             "cue": round(cue_delta, 4),
                             "coverage": round(coverage, 4),
                             "struct": round(struct_lift, 4)}

        # F6 out-of-ontology signal: how many content tokens of the RAW
        # query (no typo correction, no synonyms — the user's own words)
        # appear ANYWHERE in the indexed registry vocabulary. Zero hits
        # means the request names nothing the ontology knows ('order a
        # pizza' — where the typo fixer would happily bend 'order' into
        # 'border' and manufacture a setBorderThickness route).
        vocab_hits = sum(
            1 for tok in tokenize(raw) if tok in self.index.df
        )

        # F2 TYPE GATE (exponential-build-5): with a polarity verb
        # present, disagreeing TOOL candidates are suppressed — the cue
        # becomes a constraint, not a nudge. Presets and coarse surfaces
        # are never gated (they legitimately combine kinds), and a tool
        # the query SPECIFICALLY addresses (name-atom coverage >= 0.5)
        # survives any gate — "hide notifications during fullscreen"
        # names setFullscreen even though "hide" is a bool verb. A
        # WHY-question is not a change request at all: its TOOL candidates
        # are suppressed entirely so the explain surface answers (D4). If
        # the gate would remove every tool, fall back to the unfiltered
        # ranking and say so in the evidence — never gate silently to
        # nothing.
        gate_note = ""
        if why_question:
            why_kept = [(s, key) for s, key in scored
                        if self.documents[key][1] != "tool"]
            if why_kept:
                scored = why_kept
                gate_note = "why-question: tools suppressed; the explain surface answers"
        elif gate_bool or gate_numeric or gate_enum:
            if gate_bool:
                wanted, gate_name = ("bool", "off/hide/mute/disable/stop -> bool")
            elif gate_enum:
                wanted, gate_name = ("enum", "move/put/switch + position -> enum")
            else:
                wanted, gate_name = (("float", "int"), "direction -> numeric")
            kept: List[Tuple[float, str]] = []
            n_tools_kept = 0
            for pair in scored:
                key = pair[1]
                if self.documents[key][1] != "tool":
                    kept.append(pair)
                    continue
                if coverage_hits.get(key, 0.0) >= 0.5:
                    kept.append(pair)  # specific addressing survives the gate
                    n_tools_kept += 1
                    continue
                spec = self.spec_by_name.get(key)
                tool_kind = spec.kind if spec else None
                if tool_kind == wanted or (
                    isinstance(wanted, tuple) and tool_kind in wanted
                ):
                    kept.append(pair)
                    n_tools_kept += 1
            if n_tools_kept > 0:
                # The gate only stands when what it keeps clears the
                # score bar; a gate that leaves only sub-bar candidates
                # while a suppressed tool clears it ("increase the blur":
                # the numeric gate drops on/off-only setBlurEnabled, the
                # numeric survivors all score ~0.2) found nothing — fall
                # back so the honest no-op/absence path can answer (F5).
                best_kept = max(
                    (pair[0] for pair in kept
                     if self.documents[pair[1]][1] == "tool"), default=0.0)
                best_any = max(
                    (pair[0] for pair in scored
                     if self.documents[pair[1]][1] == "tool"), default=0.0)
                if best_kept >= state.min_score or best_any < state.min_score:
                    scored = kept
                    gate_note = f"type gate: {gate_name} kept {n_tools_kept} tool candidates"
                else:
                    gate_note = ("type gate survivors all scored below the bar while a "
                                 "suppressed tool clears it; fell back to the unfiltered "
                                 "ranking (low confidence)")
            else:
                gate_note = ("type gate emptied the tool candidates; "
                             "fell back to the unfiltered ranking (low confidence)")

        # Ranking: score desc, then MOST MATCHED NAME ATOMS desc (a
        # 4-atom-specific tool beats a 3-atom-generic one at equal score
        # — "only show the dock on the current desktop" is the dock
        # toggle, not the tab-switch one), then name asc for determinism.
        def _matched_atoms(key: str) -> int:
            atoms = self.name_atom_sets.get(key)
            if not atoms:
                return 0
            return round(coverage_hits.get(key, 0.0) * len(atoms))

        scored.sort(key=lambda pair: (-pair[0], -_matched_atoms(pair[1]), pair[1]))

        # F3 evidence guard: two structural checks a blended score cannot
        # express (specific addressing vs non-tool tops; prepositional
        # -object mentions of coarse surfaces). Pure reordering with the
        # evidence attached — see cortex/guard.py.
        scored, guard_notes = guard.apply(
            scored,
            documents=self.documents,
            coverage=coverage_hits,
            raw_words=raw_words,
            stems={w: stem(w) for w in raw_words},
            doc_stems=self._doc_stem_cache,
            min_margin=state.min_margin,
        )
        top_pairs = scored[:k]

        # Softmax with temperature over the top-k (probabilities are the
        # display + calibration surface; ranking is by raw score).
        temps = [s / state.temperature for s, _ in top_pairs]
        m = max(temps) if temps else 0.0
        exps = [pow(2.718281828459045, t - m) for t in temps]
        z = sum(exps) or 1.0

        candidates: List[Candidate] = []
        for (score, key), e in zip(top_pairs, exps):
            cand = Candidate(
                surface=key,
                kind=self.documents[key][1],
                score=round(score, 4),
                p=round(e / z, 4),
                cues=dict(cues),
            )
            cand.evidence.extend(evidence)
            if guard_notes and not candidates:
                cand.evidence.extend(guard_notes)
            if gate_note:
                cand.evidence.append(gate_note)
            if key in noun_hits:
                cand.evidence.append("exact noun grammar hit")
            if key in floors:
                cand.evidence.append("structural pattern hit")
            coverage = coverage_hits.get(key, 0.0)
            if coverage >= 0.999:
                cand.evidence.append("full name-atom coverage (specific addressing)")
            elif coverage >= 0.5:
                cand.evidence.append(f"name-atom coverage {coverage:.0%}")
            _delta, cue_evidence = self._cue_kind_delta(cues, key, raw)
            cand.evidence.extend(cue_evidence)
            # sorted(): evidence emission must be deterministic. Iterating
            # a bare set leaked PYTHONHASHSEED order into the evidence
            # strings, and combined with the pipeline's [:4] evidence cap
            # it made WHICH matched-term line survived a coin flip across
            # runs on the same input (Stage C1 golden probe caught it).
            for term in sorted(set(q_tokens)):
                if term in self.index.doc_tokens.get(key, ()):
                    cand.evidence.append(f"matched '{term}'")
            candidates.append(cand)

        if not candidates:
            return RouteResult(verdict="ABSTAIN", candidates=[], question=_ABSTAIN_QUESTION)
        top_score = candidates[0].score
        margin = top_score - (candidates[1].score if len(candidates) > 1 else 0.0)

        # F6 out-of-ontology (D5): a thin question or a request that names
        # nothing in the registry vocabulary is outside the ontology — an
        # honest verdict, never a confident route to a coincidental tool.
        # Candidates ride along as evidence.
        top_cand = candidates[0]
        has_value_cue = any(
            key in cues for key in (
                "number", "percent", "position", "bool", "toggle",
                "direction", "absolute", "reset", "transparency",
            )
        )
        thin_question = (
            is_question and not why_question and not has_value_cue
            and top_cand.kind == "tool"
            and top_cand.surface not in noun_hits
            and coverage_hits.get(top_cand.surface, 0.0) < 0.5
        )
        if (thin_question or vocab_hits == 0) and top_cand.kind == "tool":
            return RouteResult(
                verdict="OUT_OF_ONTOLOGY", candidates=candidates,
                question=("that's outside the settings I know about — I can "
                          "change shell settings, explain why something looks "
                          "a certain way, search the docs, diagnose logs, or "
                          "answer math/logic questions"),
                features=features,
            )

        if top_score < state.min_score:
            return RouteResult(
                verdict="ABSTAIN", candidates=candidates, question=_ABSTAIN_QUESTION,
                features=features,
            )
        if margin < state.min_margin and len(candidates) > 1:
            names = ", ".join(f"'{c.surface}'" for c in candidates[:3])
            return RouteResult(
                verdict="AMBIGUOUS", candidates=candidates,
                question=f"several settings could match: {names} — which one?",
                features=features,
            )
        # Capability-1 ensemble-disagreement abstention (demote-only): when
        # the lexical, semantic, and fuzzy channels each prefer a DIFFERENT
        # tool on a strictly-contested top (margin under twice the bar) and
        # no candidate is specifically addressed (coverage >= 0.5), the
        # honest verdict is a clarifying question, not a confident route.
        # This is the confident-wrong killer: it only ever DEMOTES.
        if (len(candidates) > 1 and candidates[0].kind == "tool"
                and margin < 3 * state.min_margin
                and coverage_hits.get(candidates[0].surface, 0.0) < 0.5):
            tool_keys = [c.surface for c in candidates if c.kind == "tool"]
            _win = Counter()
            for ch in (_ch_lex, _ch_sem, _ch_fuzz):
                _win[max(tool_keys, key=lambda kk: (ch[kk], kk))] += 1
            if _win[candidates[0].surface] < 2:
                names = ", ".join(f"'{c.surface}'" for c in candidates[:2])
                return RouteResult(
                    verdict="AMBIGUOUS", candidates=candidates,
                    question=(f"the wording matches several settings "
                              f"({names}) about equally — which one?"),
                    features=features,
                )
        return RouteResult(verdict="ROUTED", candidates=candidates, features=features)


_ABSTAIN_QUESTION = (
    "no setting clearly matches that phrasing; try naming the thing you "
    "want to change (bar, dock, corners, blur, animations, notifications...)"
)


# ---------------------------------------------------------------------------
# Process-wide singleton.
# ---------------------------------------------------------------------------

_ROUTER: Optional[Router] = None


def router() -> Router:
    global _ROUTER
    if _ROUTER is None:
        _ROUTER = Router()
    return _ROUTER


def route(text: str, state: RouterState = DEFAULT_STATE, k: int = 5) -> RouteResult:
    """Convenience one-shot routing (module-level, the common case)."""
    return router().route(text, state=state, k=k)
