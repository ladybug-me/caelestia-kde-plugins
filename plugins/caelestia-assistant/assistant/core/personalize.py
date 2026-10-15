"""cortex.personalize — personalized suggestions wired into the #120
proposal surface (F16, exponential-build-5; issue #120 b6e).

The A9 conformance table records b6e honestly: "brain prefs/nlhistory
learn; nothing wired into #120 proposals". This module is the wiring:
it MINES the consented evidence the assistant already holds and FILES
proposals into the existing brain ledger — the same ledger, approve/
reject flow and unified inbox (cortex/inbox.py) every other proposal
rides. Nothing is ever applied by this module; approving a suggestion
is always an explicit user decision, and the suggestion's payload is a
SUGGESTED_NOT_EXECUTED command string, never an executed one.

Two deterministic miners, both with honesty floors:

- recurring applies (settings history): the same label ("preset:
  battery-saver", "profile: X", a repeated chat request) applied
  MIN_RECUR times or more is a habit — the suggestion offers the F9
  profile composition (``settings --profile-save``) so the next time is
  one command, citing the history entry ids as evidence;
- hour patterns (brain.prefs posteriors): a (group, direction)
  preference with effective sample size >= MIN_OBS and posterior mean
  >= BIAS_FLOOR in an hour bucket becomes an hour-relevant suggestion
  ("you usually lower animations around 21:00"), citing the observation
  count and the honest interval from the prefs model itself.

Discipline (same shape as the idle-cadence invitations, build 4 G):

- one PENDING suggestion per target (slug) — duplicates never file;
- a target recently DECIDED (approved or rejected within REJECT_COOLDOWN
  days) stays quiet — a rejected suggestion does not nag;
- the pending suggestion set is bounded (MAX_PENDING, oldest filed
  evicted first), so the ledger cannot grow without limit;
- filing is an explicit CLI verb (``cortex personalize file``), never a
  background loop (the inotify design-doc-only precedent holds).

Pure functions apart from the ledger writes behind the explicit verb:
no execution, no network, no RNG; ``now`` is always a caller argument.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence

__all__ = ["MIN_RECUR", "MIN_OBS", "BIAS_FLOOR", "MAX_PENDING",
           "REJECT_COOLDOWN_DAYS", "SUGGESTION_KIND", "normalize_label",
           "recurring_applies", "hour_patterns", "mine",
           "file_suggestions", "render_lines"]

MIN_RECUR = 3          # approved applies of one label before a suggestion
MIN_OBS = 4            # prefs effective sample size floor
BIAS_FLOOR = 0.8       # prefs posterior mean floor
MAX_PENDING = 8        # bounded pending suggestions in the ledger
REJECT_COOLDOWN_DAYS = 14
SUGGESTION_KIND = "personalized_suggestion"

_LABEL_NOISE_RE = re.compile(r"\b(?:id\s*#?\d+|entry\s*#?\d+)\b", re.I)
_TRAILING_ID_RE = re.compile(r"\s*#\d+\s*$")


def normalize_label(label: str) -> str:
    """One label -> one habit key: strip entry/ids and collapse
    whitespace so 'preset: battery-saver' from different days groups
    together. Pure."""
    cleaned = _TRAILING_ID_RE.sub("", str(label or ""))
    cleaned = _LABEL_NOISE_RE.sub("", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def _parse_at(entry: Dict[str, Any]) -> Optional[datetime]:
    raw = entry.get("at")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw))
    except (TypeError, ValueError):
        return None


def recurring_applies(entries: Sequence[Dict[str, Any]],
                      min_recur: int = MIN_RECUR) -> List[Dict[str, Any]]:
    """Recurring APPROVED applies from the settings history ring (the
    ring only records applied proposals, so its entries are consented
    changes by construction). Grouped by normalized label; each group
    reports its count, hour histogram (0-23), evidence ids and last
    timestamp. Deterministic; groups ordered by count desc then label."""
    groups: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        label = normalize_label(str(entry.get("label", "")))
        if not label:
            continue
        when = _parse_at(entry)
        group = groups.setdefault(label, {
            "label": label, "count": 0, "hours": {}, "evidence_ids": [],
            "last_at": None,
        })
        group["count"] += 1
        if when is not None:
            group["hours"][when.hour] = group["hours"].get(when.hour, 0) + 1
            if group["last_at"] is None or when > group["last_at"]:
                group["last_at"] = when
        eid = entry.get("id")
        if eid is not None:
            group["evidence_ids"].append(eid)
    out = []
    for label, group in groups.items():
        if group["count"] >= min_recur:
            out.append({
                "label": label,
                "count": group["count"],
                "hours": dict(sorted(group["hours"].items())),
                "evidence_ids": group["evidence_ids"][:12],
                "last_at": group["last_at"].isoformat(timespec="seconds")
                           if group["last_at"] else None,
            })
    out.sort(key=lambda g: (-g["count"], g["label"]))
    return out


def hour_patterns(model, min_obs: int = MIN_OBS,
                  bias_floor: float = BIAS_FLOOR) -> List[Dict[str, Any]]:
    """Hour-relevant preferences from a fitted brain.prefs model.

    The model is the caller's (already loaded from its state file); this
    module never reads one. A pattern qualifies only when the posterior's
    effective observation count clears MIN_OBS (the model's own honesty
    floor) and its mean clears BIAS_FLOOR; the interval the model reports
    travels verbatim in the suggestion. Buckets are the model's own
    6-hour buckets (0=night .. 3=evening), rendered at their first hour.
    """
    patterns: List[Dict[str, Any]] = []
    table = getattr(model, "table", None)
    if not isinstance(table, dict):
        return patterns
    for key in sorted(table):
        try:
            group, direction, bucket = str(key).split("|")
            bucket_int = int(bucket)
        except ValueError:
            continue
        row = model.bias(group, direction, bucket_int * 6)
        if row.get("n", 0) < min_obs:
            continue
        mean = float(row.get("p_accept", 0.0))
        if mean < bias_floor:
            continue
        patterns.append({
            "group": group, "direction": direction, "bucket": bucket_int,
            "mean": mean, "ess": row.get("n", 0),
            "ci95": row.get("ci95"),
        })
    patterns.sort(key=lambda p: (-p["ess"], p["group"], p["direction"]))
    return patterns


def mine(entries: Sequence[Dict[str, Any]], model=None,
         min_recur: int = MIN_RECUR) -> Dict[str, List[Dict[str, Any]]]:
    """Both miners at once (read-only). ``model`` is an optional fitted
    brain.prefs.PreferenceModel."""
    return {
        "recurring": recurring_applies(entries, min_recur=min_recur),
        "hour_patterns": (hour_patterns(model) if model is not None else []),
    }


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return slug[:48] or "unnamed"


def _suggestion_command(group: Dict[str, Any]) -> str:
    """The concrete, inert command a recurring-apply habit suggests (the
    F9 profile composition)."""
    label = group["label"]
    if label.startswith("preset: "):
        preset = label[len("preset: "):]
        return f"SUGGESTED_NOT_EXECUTED: caelestia-assist settings --profile-save {preset}-habit preset:{preset}"
    return (f"SUGGESTED_NOT_EXECUTED: caelestia-assist settings --macro-save "
            f"{_slug(label)}")


def file_suggestions(mined: Dict[str, List[Dict[str, Any]]], ledger,
                     now: Optional[datetime] = None) -> Dict[str, int]:
    """File qualifying suggestions into the brain ledger (the #120
    proposal surface). Dedupe + cooldown + bounded pending set. Writes
    ONLY the ledger (its own atomic path); ``now`` anchors the cooldown.
    Returns {"filed", "skipped_pending", "skipped_cooldown",
    "skipped_quiet"}."""
    now = now or datetime(2026, 1, 1, 12, 0, 0)
    pending_targets = {str(p.get("target")) for p in ledger.pending()}
    stats = {"filed": 0, "skipped_pending": 0, "skipped_cooldown": 0,
             "skipped_quiet": 0}
    cooldown = timedelta(days=REJECT_COOLDOWN_DAYS)
    pending_suggestions = [p for p in ledger.pending()
                           if p.get("kind") == SUGGESTION_KIND]

    def _try_file(target: str, payload: Dict[str, Any], why: str,
                  confidence: float) -> None:
        nonlocal stats
        if target in pending_targets:
            stats["skipped_pending"] += 1
            return
        decided = [p for p in ledger.labeled() if str(p.get("target")) == target]
        if decided:
            latest = max(str(p.get("decided_at") or "") for p in decided)
            try:
                when = datetime.fromisoformat(latest)
            except ValueError:
                when = None
            if when is not None:
                # Ledger.decide stamps timezone-aware UTC; the caller's
                # now is naive by contract — compare naive-to-naive.
                if when.tzinfo is not None:
                    when = when.replace(tzinfo=None)
                if (now - when) < cooldown:
                    stats["skipped_cooldown"] += 1
                    return
        # bounded pending: evict the OLDEST pending suggestion first
        while len(pending_suggestions) >= MAX_PENDING:
            oldest = pending_suggestions.pop(0)
            ledger.items = [p for p in ledger.items
                            if p is not oldest and p.get("id") != oldest.get("id")]
            ledger._save()
        pid = ledger.propose(SUGGESTION_KIND, target, payload, why, confidence)
        pending_targets.add(target)
        pending_suggestions.append({"id": pid, "target": target})
        stats["filed"] += 1

    for group in mined.get("recurring", []):
        target = f"suggestion:profile-for:{_slug(group['label'])}"
        hour_note = ""
        if group["hours"]:
            top_hour = max(group["hours"].items(), key=lambda kv: (kv[1], -kv[0]))[0]
            hour_note = f" (most often around {top_hour:02d}:00)"
        why = (f"you applied {group['label']!r} {group['count']} times"
               f"{hour_note} — history entries "
               f"{group['evidence_ids'][:6]}; want it as a one-command "
               f"profile?")
        _try_file(target, {
            "kind": "recurring_apply", "label": group["label"],
            "count": group["count"], "hours": group["hours"],
            "evidence_ids": group["evidence_ids"],
            "command": _suggestion_command(group),
        }, why, min(0.9, 0.5 + 0.1 * group["count"]))

    for pattern in mined.get("hour_patterns", []):
        target = (f"suggestion:hour-pattern:{_slug(pattern['group'])}-"
                  f"{_slug(str(pattern['direction']))}-{pattern['bucket']}")
        why = (f"your approvals lean toward {pattern['direction']} "
               f"{pattern['group']!r} around "
               f"{pattern['bucket'] * 6:02d}:00 "
               f"(mean {pattern['mean']}, {pattern['ess']:.0f} effective "
               f"observations) — surface it at that hour?")
        _try_file(target, {
            "kind": "hour_pattern", **pattern,
        }, why, pattern["mean"])
    return stats


def render_lines(mined: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    """The read-only rendering for ``cortex personalize list``."""
    lines: List[str] = []
    recurring = mined.get("recurring", [])
    patterns = mined.get("hour_patterns", [])
    if not recurring and not patterns:
        return ["no personalization evidence yet: recurring habits need "
                f"{MIN_RECUR}+ applies of the same label, hour patterns "
                f"need {MIN_OBS}+ effective observations"]
    if recurring:
        lines.append(f"recurring applies ({len(recurring)}):")
        for group in recurring:
            hours = ", ".join(f"{h:02d}:00 x{n}" for h, n in
                              sorted(group["hours"].items())) or "no hours"
            lines.append(f"  {group['label']!r} x{group['count']} — "
                         f"{hours} — entries {group['evidence_ids'][:6]}")
        lines.append("  file these as suggestions: cortex personalize file")
    if patterns:
        lines.append(f"hour patterns ({len(patterns)}):")
        for pattern in patterns:
            lines.append(f"  {pattern['direction']} {pattern['group']!r} "
                         f"around {pattern['bucket'] * 6:02d}:00 — mean "
                         f"{pattern['mean']} over {pattern['ess']:.0f} obs")
    return lines
