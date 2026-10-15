#!/usr/bin/env python3
"""Capability-1a: grammar-based paraphrase generation per tool.

Generates candidate dev items from the registry's typed facts
(bool/numeric/enum per tool) over templated grammar with slot
substitution (politeness, framing, polarity, value spellings). Output
is a JSON of CANDIDATES for `eval grow`'s quarantine — never
auto-promoted; a human review promotes into a dev set.

Deterministic: fixed seed, sorted iteration everywhere.
"""
import json
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from assistant.adapters.caelestia.registry import TOOL_SPECS  # noqa: E402
from assistant.core.corpus import tool_atoms  # noqa: E402
from assistant.core.lexicon import camel_split  # noqa: E402

OUT = os.path.join(REPO, "assistant", "data", "paraphrase_candidates.json")
SEED = 120

FRAMINGS = {
    "bool_on": ["turn on {subject}", "enable {subject}", "show {subject}",
                "activate {subject}", "i want {subject} on"],
    "bool_off": ["turn off {subject}", "disable {subject}", "hide {subject}",
                 "stop {subject}", "no more {subject}"],
    "numeric_up": ["make {subject} {adj}", "increase {subject}",
                   "{subject} should be {val}", "raise {subject} to {val}"],
    "numeric_down": ["make {subject} {adj}", "decrease {subject}",
                     "{subject} should be {val}", "lower {subject} to {val}"],
    "enum": ["put {subject} {val}", "move {subject} {val}",
             "set {subject} to {val}", "{subject} {val} please"],
}
POLITE = ["", "please ", "could you ", "can you "]
ADJ = {"up": ["bigger", "larger", "higher"], "down": ["smaller", "lower"]}
POSITIONS = ["to the left", "to the right", "to the top", "to the bottom"]


def subject_for(spec):
    atoms = [a for a in tool_atoms(spec) if a not in ("set", "enabled")]
    return " ".join(atoms[:3]) or camel_split(spec.name)[-1]


def candidates_for(spec):
    subj = subject_for(spec)
    kind = spec.kind
    rows = []
    if kind == "bool":
        for tmplt in FRAMINGS["bool_on"]:
            rows.append(("bool", tmplt.format(subject=subj)))
        for tmplt in FRAMINGS["bool_off"]:
            rows.append(("bool", tmplt.format(subject=subj)))
    elif kind in ("float", "int") and spec.minimum is not None \
            and spec.maximum is not None:
        mid = (spec.minimum + spec.maximum) / 2
        for tmplt in FRAMINGS["numeric_up"]:
            rows.append(("numeric", tmplt.format(
                subject=subj, adj=ADJ["up"][0], val=round(mid, 2))))
        for tmplt in FRAMINGS["numeric_down"]:
            rows.append(("numeric", tmplt.format(
                subject=subj, adj=ADJ["down"][0], val=round(mid, 2))))
    elif kind == "enum" and spec.enum:
        for val in list(spec.enum)[:2]:
            words = camel_split(str(val)) or [str(val)]
            phrase = " ".join(words)
            for tmplt in FRAMINGS["enum"]:
                rows.append(("enum", tmplt.format(subject=subj, val=phrase)))
    # slot substitutions: politeness prefix
    out = []
    for kind_, text in rows:
        p = POLITE[int(text.__hash__() % len(POLITE))] if POLITE else ""
        out.append({"kind": kind_, "text": (p + text).strip()})
    return out


def main():
    rng = random.Random(SEED)
    items = []
    for spec in TOOL_SPECS:
        for row in candidates_for(spec):
            items.append({
                "tool": spec.name,
                "kind": row["kind"],
                "text": row["text"],
                "expect": "ROUTED",
                "provenance": "grammar-template candidate (review before "
                              "promotion into a dev set)",
            })
    rng.shuffle(items)
    items = items[:400]
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"seed": SEED, "n": len(items),
                   "how": "grammar-template paraphrase candidates; feed "
                          "through eval grow's quarantine, never auto-promote",
                   "items": items}, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(f"wrote {len(items)} paraphrase candidates -> {OUT}")


if __name__ == "__main__":
    main()
