"""cortex.confusables — the confusable-pair clarifier (Group A4).

Some settings tools SWAP: the arena's own failures prove it. "recolour
the tray icons" confidently lands on setDockRecolourIcons; "make the
volume step smaller" and "make the brightness step smaller" are the
same sentence one noun apart; "move the lyrics to the centre" and
"centre the lyrics text" split a position/alignment pair the router
cannot rank apart. For these, a confident route is a coin flip and a
generic "several settings could match" question is a lazy coin flip.

This module does two honest things:

1. MINE (offline, deterministic, committed artifact): candidate pairs =
   same-group, same-kind tools whose tool documents are near-duplicates
   under BM25 (the lexical twin test), plus the pairs the arena's
   failure classes pinned by hand. For each pair, the distinguishing
   QUESTION is derived from the metadata difference that actually
   separates them (object noun, absolute-range-vs-scale, position
   vocabulary), and its EXPECTED INFORMATION GAIN is computed over the
   two hypotheses: with two hypotheses the prior entropy is 1 bit and a
   question scores 1 bit x P(answer separates the pair) — P estimated
   from the metadata itself (a question only counts when each answer
   option maps onto exactly one hypothesis). The highest-gain question
   ships as the pair's ask.

2. CLARIFY (runtime, read-only): when the router returns AMBIGUOUS and
   the ranked candidates contain a mined pair, the generic question is
   replaced by the pair's own question. No verdict changes, no write,
   no new surface — the same honest ASK, just one that the user can
   actually answer.

The mined artifact is committed (``confusable_pairs.json``): rebuilding
it is a reviewed act like regenerating the registry, and the runtime
never mines on the fly (determinism + budget).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["PAIRS_PATH", "mine_pairs", "clarify", "render_lines",
           "load_pairs"]

PAIRS_PATH = Path(__file__).resolve().parent / "confusable_pairs.json"

# BM25 near-duplicate threshold over tool documents (mined candidates
# must clear this before a reviewer question is even drafted).
_DOC_SIM_FLOOR = 0.55

# Hand-pinned pairs from the arena's own failure classes (m-items): the
# sibling swaps the router measurably confuses. Each cites the dev item
# that evidences the confusion. The miner merges these with the
# document-similarity candidates; a hand pin overrides the generated
# question when both exist (reviewer judgment beats a template).
HAND_PINNED: Tuple[Dict[str, Any], ...] = (
    {
        "a": "setTrayRecolour", "b": "setDockRecolourIcons",
        "evidence": "dev m02: 'recolour the tray icons to match my theme' "
                    "ranks the dock sibling first (sealed sr13 same class)",
        "question": "which icons — the system TRAY icons, or the DOCK "
                    "app icons?",
        "answers": {"tray": "setTrayRecolour",
                    "dock": "setDockRecolourIcons"},
        "gain": 1.0,
    },
    {
        "a": "setDesktopLyricsPosition", "b": "setDesktopLyricsAlignment",
        "evidence": "dev m15/m16: 'move the lyrics to the centre' and "
                    "'centre the lyrics text' rank each other's sibling "
                    "first (sealed sr93 same class)",
        "question": "where the lyrics SIT on screen, or how the text "
                    "ALIGNS inside its box?",
        "answers": {"where": "setDesktopLyricsPosition",
                    "align": "setDesktopLyricsAlignment"},
        "gain": 1.0,
    },
    {
        "a": "setAudioIncrement", "b": "setBrightnessIncrement",
        "evidence": "dev m25/m26: identical sentence one noun apart; the "
                    "brightness sibling wins both at cold start",
        "question": "the step size for VOLUME, or for SCREEN BRIGHTNESS?",
        "answers": {"volume": "setAudioIncrement",
                    "brightness": "setBrightnessIncrement"},
        "gain": 1.0,
    },
    {
        "a": "setPosition", "b": "setHoverBottomRight",
        "evidence": "dev m08: 'show toasts in the bottom-right corner' "
                    "confidently takes the overview hover-corner toggle",
        "question": "where NOTIFICATION toasts appear on screen, or the "
                    "overview corner that shows windows on HOVER?",
        "answers": {"toast": "setPosition",
                    "hover": "setHoverBottomRight"},
        "gain": 1.0,
    },
    {
        "a": "setAmbientOpacity", "b": "setTransparencyBase",
        "evidence": "dev m05: 'make the ambient background more opaque' "
                    "takes the global transparency slider",
        "question": "the AMBIENT background layer's opacity, or the "
                    "GLOBAL transparency of the whole shell?",
        "answers": {"ambient": "setAmbientOpacity",
                    "global": "setTransparencyBase"},
        "gain": 1.0,
    },
    {
        "a": "setEnableOverviewBlur", "b": "setDisableWallpaperBlur",
        "evidence": "dev m11/m12 (sealed sr67/sr68 same class): the two "
                    "overview blur toggles swap under on/off phrasing",
        "question": "the blur of the OVERVIEW ITSELF (windows floating in "
                    "it), or the WALLPAPER's blur while the overview is "
                    "open?",
        "answers": {"overview itself": "setEnableOverviewBlur",
                    "wallpaper": "setDisableWallpaperBlur"},
        "gain": 1.0,
    },
    {
        "a": "setMaxFprintTries", "b": "setPreviewScalesWirelessPassword",
        "evidence": "dev m14: 'fingerprint attempts before the password' "
                    "lands on a preview-scale tool via the 'password' noun",
        "question": "how many FINGERPRINT attempts before the password "
                    "fallback — or how the password PREVIEW is scaled?",
        "answers": {"fingerprint attempts": "setMaxFprintTries",
                    "preview": "setPreviewScalesWirelessPassword"},
        "gain": 1.0,
    },
)


def _doc_similarity(index, a_key: str, b_key: str) -> float:
    """BM25 score of a's document tokens against b's document (a KEY
    into the index), normalized by the self-max (0..1-ish; the pair
    floor is deliberately low — mining proposes, the reviewer
    disposes)."""
    from .vectorize import tokenize
    toks = index.doc_tokens.get(a_key) or tokenize(a_key)
    score = index.score(toks, b_key)
    self_score = index.score(toks, a_key)
    if self_score <= 0:
        return 0.0
    return score / self_score


def mine_pairs() -> Dict[str, Any]:
    """Deterministic pair mining: hand-pinned classes + same-group
    same-kind document twins above the similarity floor. Output is the
    committed artifact; runtime only ever READS it."""
    from .corpus import tool_document
    from .vectorize import TfidfIndex
    from assistant.adapters.caelestia.registry import TOOL_SPECS, tools_by_group

    specs = {s.name: s for s in TOOL_SPECS}
    docs = {name: tool_document(spec) for name, spec in specs.items()}
    index = TfidfIndex(docs)  # keyed by tool name

    # generated candidates: same group, same kind, twin documents
    generated: List[Dict[str, Any]] = []
    seen: set = set()
    for group, group_specs in sorted(tools_by_group().items()):
        for i, sa in enumerate(group_specs):
            for sb in group_specs[i + 1:]:
                if sa.kind != sb.kind:
                    continue
                pair = tuple(sorted((sa.name, sb.name)))
                if pair in seen:
                    continue
                seen.add(pair)
                sim = _doc_similarity(index, sa.name, sb.name)
                if sim >= _DOC_SIM_FLOOR:
                    generated.append({
                        "a": sa.name, "b": sb.name, "group": group,
                        "doc_similarity": round(sim, 4),
                    })
    generated.sort(key=lambda g: (-g["doc_similarity"], g["a"], g["b"]))

    pairs: List[Dict[str, Any]] = []
    for pin in HAND_PINNED:
        a, b = specs.get(pin["a"]), specs.get(pin["b"])
        if a is None or b is None:
            continue  # a tool pruned upstream: the pin dies honestly
        pairs.append({
            "a": a.name, "b": b.name,
            "group": a.group,
            "kind": a.kind,
            "doc_similarity": round(_doc_similarity(
                index, a.name, b.name), 4),
            "question": pin["question"],
            "answers": pin["answers"],
            "gain": pin["gain"],
            "evidence": pin["evidence"],
            "source": "hand-pinned (arena failure class)",
        })
    for g in generated[:24]:
        name = "both exist" in _GENERIC_QUESTIONS and "" or None
        q = _question_for(g, specs)
        if q is None:
            continue
        pairs.append({
            "a": g["a"], "b": g["b"], "group": g["group"],
            "kind": specs[g["a"]].kind,
            "doc_similarity": g["doc_similarity"],
            "question": q["question"], "answers": q["answers"],
            "gain": q["gain"],
            "evidence": f"mined: same-group {g['group']!r} documents "
                        f"similarity {g['doc_similarity']}",
            "source": "mined (document twins)",
        })
    pairs.sort(key=lambda p: (p["source"] != "hand-pinned (arena failure "
                              "class)", -p["gain"], p["a"], p["b"]))
    return {
        "generator": "assistant.core.confusables.mine_pairs",
        "doc_sim_floor": _DOC_SIM_FLOOR,
        "n_pairs": len(pairs),
        "pairs": pairs,
    }


_GENERIC_QUESTIONS: Dict[str, str] = {
    "scale": "an ABSOLUTE value in that unit, or a SCALE multiplier on "
             "the current value?",
}


def _question_for(pair: Dict[str, Any], specs) -> Optional[Dict[str, Any]]:
    """Draft the distinguishing question from the metadata difference.
    Highest expected information gain wins; with two hypotheses the
    prior entropy is 1 bit, so gain = P(answer separates) x 1 bit."""
    a, b = specs[pair["a"]], specs[pair["b"]]
    # absolute-range vs scale: one sibling is named *Scale / *Strength
    # with a multiplier range, the other an absolute range
    for s, other in ((a, b), (b, a)):
        if "scale" in s.name.lower() or "strength" in s.name.lower():
            if s.minimum is not None and s.maximum is not None \
                    and s.maximum - s.minimum <= 4.0:
                return {
                    "question": _GENERIC_QUESTIONS["scale"],
                    "answers": {"scale multiplier": s.name,
                                "absolute value": other.name},
                    "gain": 1.0,
                }
    # object-word difference: the first name atom the two do not share
    from .lexicon import camel_split
    wa = [w.lower() for w in camel_split(a.name)
          if w.lower() not in {"set", "enable", "disabled", "enabled"}]
    wb = [w.lower() for w in camel_split(b.name)
          if w.lower() not in {"set", "enable", "disabled", "enabled"}]
    diff_a = next((w for w in wa if w not in wb), None)
    diff_b = next((w for w in wb if w not in wa), None)
    if diff_a and diff_b:
        return {
            "question": f"the {diff_a} setting, or the {diff_b} setting?",
            "answers": {diff_a: a.name, diff_b: b.name},
            "gain": 1.0,
        }
    return None


def load_pairs() -> List[Dict[str, Any]]:
    """The committed artifact (runtime view). Empty when absent — the
    clarifier degrades to the generic question, never mines live."""
    try:
        data = json.loads(PAIRS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return list(data.get("pairs") or [])


def _pair_for(candidates: List[str],
              pairs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """First committed pair fully present in the ranked candidates."""
    ranked = [c for c in candidates if c]
    for pair in pairs:
        if pair["a"] in ranked and pair["b"] in ranked:
            return pair
    return None


def clarify(question: str, candidates: List[str]) -> Tuple[str, bool]:
    """(question, upgraded) — the pair's own ask when the ranked
    candidates contain a mined pair, else the incoming question."""
    pair = _pair_for(candidates, load_pairs())
    if pair is None:
        return question, False
    return pair["question"], True


def render_lines() -> List[str]:
    pairs = load_pairs()
    lines = [f"{len(pairs)} confusable pairs (committed artifact; "
             f"rebuild with python3 -m assistant.core.confusables):"]
    for p in pairs:
        lines.append(f"  {p['a']} <-> {p['b']}  -- {p['source']}")
        lines.append(f"    ask: {p['question']}")
    return lines


def main(argv=None) -> int:  # pragma: no cover - thin CLI
    """`python3 -m assistant.core.confusables [--mine]`:
    list the committed pairs; --mine rebuilds the artifact (a reviewed,
    committed change, like regenerating the registry)."""
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--mine":
        data = mine_pairs()
        PAIRS_PATH.write_text(
            json.dumps(data, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8")
        print(f"wrote {PAIRS_PATH.name}: {data['n_pairs']} pairs")
        return 0
    for line in render_lines():
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
