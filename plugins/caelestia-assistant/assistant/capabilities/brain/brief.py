"""brain.brief — the daily brief: one deterministic page that connects everything.

    caelestia-assist brief                  # from the live ledger + a vault
    python3 -m assistant.capabilities.brain.brief ~/notes/vault

compose() is a pure assembler over the OTHER brain modules' outputs — it owns
no algorithm of its own and re-uses theirs, which is the point: the brief is
the "second brain connects everything" surface.

  ledger      pending decisions waiting on you (Ledger.pending)
  stuck       open tasks unlikely to ever finish (survival Kaplan-Meier)
  today       what fits your remaining minutes (planner knapsack)
  forecast    trend + level of a series you care about (forecast Holt/Kalman)
  rhythm      unusual busyness by weekday/hour (rhythm z-deviations)
  ghosts      to-dos you wrote but never tracked (ghost heuristics)
  calibration whether your confidences have been honest lately (journal Brier)
  prefs       what the preference model currently believes (prefs explain)

Every section degrades independently: feed it only what you have, the rest
say "no data" and never fabricate. The output is plain data + a renderer —
the QML bridge op `brief` returns the same dict.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

__all__ = ["compose", "render"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def compose(pending: Optional[List[Dict[str, Any]]] = None,
            stuck: Optional[List[Dict[str, Any]]] = None,
            today: Optional[Dict[str, Any]] = None,
            forecast: Optional[Dict[str, Any]] = None,
            rhythm: Optional[Dict[str, Any]] = None,
            ghosts: Optional[List[Dict[str, Any]]] = None,
            calibration: Optional[Dict[str, Any]] = None,
            pref_lines: Optional[List[str]] = None,
            scan_alarm: Optional[Dict[str, Any]] = None,
            config_drift: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Assemble the brief from already-computed inputs (plain data in, plain
    data out). Missing sections render as 'no data' — never invented.

    ``config_drift`` (B2) is OPT-IN by design: the caller computes it
    (prefs.config_drift over the live shell.json + the learned preference
    posterior) and passes it; the brief never reads the config file
    itself, so the section appears only when someone deliberately asked
    for it."""
    pending = pending or []
    stuck = stuck or []
    ghosts = ghosts or []
    pref_lines = pref_lines or []
    n_pending = len(pending)
    top_pending = [
        {"id": p.get("id"), "kind": p.get("kind"), "target": p.get("target"),
         "confidence": p.get("confidence")}
        for p in pending[:3]
    ]
    decisions_due = [p for p in pending if p.get("kind") in
                     ("move", "rename", "organize", "settings_plan")]
    return {
        "generated_at": _now_iso(),
        "headline": _headline(n_pending, len(stuck), len(ghosts), scan_alarm),
        "decisions": {
            "pending": n_pending,
            "needs_attention": len(decisions_due),
            "top": top_pending,
        },
        "focus": today or {},
        "stuck_tasks": stuck[:5],
        "ghost_tasks": len(ghosts),
        "forecast": forecast or {},
        "rhythm": rhythm or {},
        "calibration": calibration or {},
        "preferences": pref_lines[:4],
        "stream_alarm": scan_alarm,
        "config_drift": config_drift or None,
    }


def _headline(n_pending: int, n_stuck: int, n_ghosts: int,
              alarm: Optional[Dict[str, Any]]) -> str:
    if alarm and alarm.get("change_at") is not None:
        return "something changed in your logs mid-stream — the rate detector alarmed"
    if n_pending >= 5:
        return f"{n_pending} decisions are waiting on you — that is the bottleneck"
    if n_stuck >= 3:
        return f"{n_stuck} tasks look stuck; decide to finish or bury them"
    if n_ghosts >= 3:
        return f"{n_ghosts} written-down to-dos were never tracked"
    if n_pending:
        return f"{n_pending} decision{'s' if n_pending != 1 else ''} pending"
    return "clear board: no pending decisions, nothing stuck"


def render(brief: Dict[str, Any]) -> str:
    out: List[str] = [f"BRIEF — {brief['generated_at']}", brief["headline"], ""]
    d = brief["decisions"]
    out.append(f"decisions: {d['pending']} pending "
               f"({d['needs_attention']} need attention)")
    for p in d["top"]:
        out.append(f"  #{p.get('id')} [{p.get('kind')}] {p.get('target')} "
                   f"(confidence {p.get('confidence')})")
    focus = brief.get("focus") or {}
    if focus.get("chosen"):
        out.append("")
        out.append("today's plan (knapsack over your minutes):")
        for t in focus["chosen"][:5]:
            out.append(f"  {t.get('title', t)} ({t.get('minutes_est', '?')} min)")
    if brief.get("stuck_tasks"):
        out.append("")
        out.append("stuck tasks (unlikely to finish; survival model):")
        for s in brief["stuck_tasks"]:
            out.append(f"  {s.get('title', s)} — {s.get('p_finish', '?')} finish odds")
    if brief.get("forecast"):
        f = brief["forecast"]
        out.append("")
        out.append(f"forecast: level={f.get('level', '?')} "
                   f"trend={f.get('trend', '?')} "
                   f"next={f.get('forecast', f.get('next', '?'))}")
    if brief.get("preferences"):
        out.append("")
        out.append("what I believe about your preferences:")
        for line in brief["preferences"]:
            out.append(f"  {line}")
    alarm = brief.get("stream_alarm")
    if alarm and alarm.get("change_at") is not None:
        out.append("")
        out.append(f"log stream: rate change detected at chunk {alarm['change_at']}")
    drift = brief.get("config_drift")
    if drift:
        out.append("")
        if drift.get("flags"):
            out.append(f"config drift (score {drift.get('score', 0)} over "
                       f"{drift.get('evaluated', 0)} settings): "
                       f"{len(drift['flags'])} sit in directions you usually reject")
            for f in drift["flags"][:3]:
                out.append(f"  {f['path']}: live {f['live']} vs default "
                           f"{f['default']} — p_accept {f['p_accept']} "
                           f"(n={f['n']}) for {f['direction']}")
        else:
            out.append(f"config drift: none — {drift.get('evaluated', 0)} "
                       "settings checked, all aligned with your preferences")
    cal = brief.get("calibration") or {}
    if cal.get("brier") is not None:
        out.append("")
        out.append(f"calibration: Brier score {cal['brier']} over "
                   f"{cal.get('labeled', '?')} resolved predictions")
    return "\n".join(out)
