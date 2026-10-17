"""assistant/eval/grow.py — the arena grows from experience, UNDER REVIEW
(F26, exponential-build-5).

Self-improvement for the MEASUREMENT half: the arena's dev sets were
authored once (A1); the assistant's real usage keeps producing exactly
the phrases that stress the router (near-threshold ABSTAIN/AMBIGUOUS
verdicts, corrected misroutes). This module turns that experience into
CANDIDATE eval items — and is deliberately honest about where the
automation ends:

- MINE (read-only): from the brain state's review bucket
  (cortex.learn's ``cortex_review``: near-threshold phrases logged,
  never silently learned) plus labeled corrections, produce candidate
  items with their evidence and schema
  {id, text, accept, expect_verdict?, source: review|correction,
   evidence: {...}};
- QUARANTINE: candidates live in a USER-STATE file (bounded), NEVER in
  the repo's dev sets — nothing reaches the arena automatically;
- PROMOTE: an explicit verb appends one reviewed candidate to a dev
  set FILE THE CALLER NAMES (default: the repo's routing_dev.json).
  The sealed sets are refused BY NAME (a promoted item can never
  contaminate the sealed measurement); duplicates by text are refused;
  schema is validated before the write.

The repo-vs-installed split is honest too: mining works from user
state anywhere; promotion is only possible where a dev-set file exists
(the checkout), because that is the only place a reviewed item is
useful.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

__all__ = ["MAX_CANDIDATES", "QUARANTINE_KEY", "mine_candidates",
           "save_quarantine", "promote", "render_lines"]

MAX_CANDIDATES = 64
QUARANTINE_KEY = "eval_grow_candidates"
SEALED_NAMES = re.compile(r"sealed")
_ID_SAFE = re.compile(r"[^a-z0-9]+")


def _new_id(text: str, existing: List[Dict[str, Any]]) -> str:
    """Deterministic id: content hash of the text, disambiguated by a
    counter when a true text collision survives the earlier dedup
    (same text should already be merged; the counter is belt-and-
    braces). No RNG — same inputs, same ids."""
    slug = _ID_SAFE.sub("-", text.lower())[:12].strip("-") or "item"
    digest = hashlib.blake2b(text.encode("utf-8"),
                             digest_size=2).hexdigest()
    taken = {c.get("id") for c in existing}
    candidate = f"g-{slug}-{digest}"
    counter = 0
    while candidate in taken:
        counter += 1
        candidate = f"g-{slug}-{digest}{counter:x}"
    return candidate


def mine_candidates(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Read-only mining from the brain state (the caller loads/saves
    it). Two sources:

    - the learner's APPROVED examples (label == 1: a real apply the
      user consented to — the strongest candidate form, source
      ``correction``);
    - the outstanding review bucket (near-threshold phrases with their
      top candidate as a PROPOSED accept — source ``review``, promotion
      is the human review act);

    deduped by text, review items last (confirmed evidence wins)."""
    out: List[Dict[str, Any]] = []
    seen = set()
    learner = state.get("cortex_learner") or {}
    for row in (learner.get("examples") or []):
        if not isinstance(row, dict) or row.get("label") != 1:
            continue
        text = str(row.get("text") or "").strip()
        surface = row.get("surface")
        if not text or not surface or text in seen:
            continue
        seen.add(text)
        out.append({
            "text": text,
            "accept": [str(surface)],
            "source": "correction",
            "evidence": {"outcome": row.get("outcome"),
                         "p": row.get("p")},
        })
    for entry in state.get("cortex_review", []) or []:
        text = str(entry.get("text") or "").strip()
        surface = entry.get("surface")
        if not text or not surface or text in seen:
            continue
        seen.add(text)
        out.append({
            "text": text,
            "accept": [str(surface)],
            "source": "review",
            "evidence": {"verdict": entry.get("verdict"),
                         "p": entry.get("p"), "at": entry.get("at")},
        })
    return out[:MAX_CANDIDATES]


def save_quarantine(state: Dict[str, Any],
                    mined: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge mined candidates into the state's quarantine (bounded,
    ids assigned, existing ids kept stable — the pure part of the
    ``eval grow mine`` verb; the CALLER persists the state)."""
    existing = list(state.get(QUARANTINE_KEY) or [])
    by_text = {str(c.get("text")): c for c in existing}
    added = 0
    for candidate in mined:
        text = str(candidate.get("text"))
        if text in by_text:
            continue
        candidate = dict(candidate)
        candidate["id"] = _new_id(text, existing)
        candidate["mined_at"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds")
        existing.append(candidate)
        by_text[text] = candidate
        added += 1
    state[QUARANTINE_KEY] = existing[-MAX_CANDIDATES:]
    return {"added": added, "total": len(state[QUARANTINE_KEY])}


def promote(state: Dict[str, Any], candidate_id: str,
            dev_set_path: Path) -> Dict[str, Any]:
    """Append ONE quarantined candidate to a dev-set file the caller
    names, and remove it from the quarantine (the caller persists the
    state). Refuses: sealed sets (BY NAME), unknown ids, duplicate
    texts, candidates without an accept surface. The only write is the
    named dev set (atomic tmp+rename, the repo's convention)."""
    dev_set_path = Path(dev_set_path)
    if SEALED_NAMES.search(dev_set_path.name.lower()):
        raise ValueError(
            f"refusing to write a SEALED set ({dev_set_path.name}); "
            f"promotion targets dev sets only — the sealed measurement "
            f"never moves")
    quarantine = list(state.get(QUARANTINE_KEY) or [])
    candidate = next((c for c in quarantine if c.get("id") == candidate_id),
                     None)
    if candidate is None:
        raise ValueError(
            f"no quarantined candidate {candidate_id!r} (run 'eval grow "
            f"mine' first; candidates live in user state, not the repo)")
    data = json.loads(dev_set_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        raise ValueError(f"{dev_set_path} is not a dev-set file "
                         f"(missing items[])")
    if any(item.get("text") == candidate["text"]
           for item in data["items"]):
        raise ValueError(
            f"dev set already contains a item with the same text as "
            f"candidate {candidate_id!r} (duplicate by text)")
    new_id = candidate.get("id", "").replace("g-", "u-")
    item = {
        "id": new_id,
        "text": candidate["text"],
        "accept": candidate.get("accept") or [],
    }
    if not item["accept"]:
        raise ValueError(
            f"candidate {candidate_id!r} has no accept surface; label it "
            f"with 'cortex review' before promoting")
    data["items"].append(item)
    data["note"] = (str(data.get("note", "")) +
                    "  Promotion u-*: user-mined via eval grow "
                    "(reviewed, quarantined, then promoted).\n").lstrip()
    tmp = dev_set_path.with_name(dev_set_path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    import os
    os.replace(tmp, dev_set_path)
    state[QUARANTINE_KEY] = [c for c in quarantine
                             if c.get("id") != candidate_id]
    return {"promoted": candidate_id, "as": new_id,
            "dev_set": str(dev_set_path)}


def render_lines(candidates: List[Dict[str, Any]]) -> List[str]:
    if not candidates:
        return ["no candidates mined: near-threshold phrases appear as "
                "you use the assistant; review them with 'cortex review' "
                "and mine again"]
    lines = [f"quarantined candidates ({len(candidates)}; promotion is an "
             f"explicit verb, sealed sets are refused):"]
    for c in candidates:
        lines.append(f"  {c.get('id')}: {c.get('text')!r} -> "
                     f"accept {c.get('accept')} [{c.get('source')}]")
    lines.append("promote with: eval grow promote ID --dev-set PATH "
                 "(dev sets only; sealed sets refuse)")
    return lines
