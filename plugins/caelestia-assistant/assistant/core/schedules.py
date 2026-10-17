"""cortex.schedules — workspace activity schedules (F19, exponential-build-5;
issue #120 b6c, closed from NOT_IMPLEMENTED).

Issue #120 Phase 3: "workspace automation". The honest shape of that
ask in a stdlib-only, no-daemon assistant is NOT a background watcher
(the inotify proactive triggering stays a design doc — build 4 E's
honest finding — and this module does not revisit that decision). What
CAN be honest and useful is a PULL-BASED scheduler:

- a SCHEDULE names a condition (a daily time window, a power state, or
  both) and an ACTION (a saved profile, F9, or a preset name);
- ``cortex schedule eval`` checks which schedules' conditions hold RIGHT
  NOW (clock and power probe are caller-supplied / explicitly pulled)
  and renders the matching SUGGESTED_NOT_EXECUTED command — nothing
  runs by itself, ever;
- ``cortex schedule file`` files the firing schedules as proposals into
  the brain ledger — the same #120 surface, dedupe/cooldown discipline
  (cortex.personalize's rules) — so the user can approve once and
  apply through the normal consent path;
- ``cortex schedule save/list/delete`` manage the definitions, stored
  in the settings history file's bounded ``"schedules"`` key (the
  undo_log/macros/profiles/environments precedent — no new write
  surface).

Eval honesty: a schedule whose action names a profile that no longer
exists is reported as SKIPPED with the reason (never silently dropped,
F7); a power-conditioned schedule without a probe result is likewise
skipped with the reason, not guessed.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["ScheduleError", "MAX_SCHEDULES", "SCHEDULES_KEY", "NAME_RE",
           "save", "list_schedules", "delete", "evaluate", "render_eval",
           "file_firing"]

MAX_SCHEDULES = 12
SCHEDULES_KEY = "schedules"
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _.\-]{0,31}")

_WINDOW_RE = re.compile(
    r"^(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})$")

SUGGESTION_KIND = "schedule_firing"


class ScheduleError(ValueError):
    """Raised for every refusal (bad name/window, unknown profile,
    unknown schedule); the caller renders the reason."""


def _parse_window(text: str) -> Tuple[int, int]:
    """'HH:MM-HH:MM' -> (start_minute, end_minute); the window may span
    midnight (start > end means it wraps)."""
    m = _WINDOW_RE.match(str(text or "").strip())
    if not m:
        raise ScheduleError(
            f"time windows look like 'HH:MM-HH:MM' (e.g. 21:00-23:30, "
            f"may wrap midnight) — got {text!r}")
    h1, m1, h2, m2 = (int(g) for g in m.groups())
    if not (0 <= h1 <= 23 and 0 <= h2 <= 23 and 0 <= m1 <= 59
            and 0 <= m2 <= 59):
        raise ScheduleError(f"invalid time window {text!r}")
    return h1 * 60 + m1, h2 * 60 + m2


def _in_window(now: datetime, window: Tuple[int, int]) -> bool:
    minute = now.hour * 60 + now.minute
    lo, hi = window
    if lo <= hi:
        return lo <= minute <= hi
    return minute >= lo or minute <= hi  # wraps midnight


def _load(target) -> List[Dict[str, Any]]:
    from assistant.capabilities.settings.history import _load as history_load
    return [dict(s) for s in history_load(target).get(SCHEDULES_KEY, [])]


def _save_all(target, schedules: List[Dict[str, Any]]) -> None:
    from assistant.capabilities.settings.history import _load as history_load
    from assistant.capabilities.settings.history import _save as history_save
    data = history_load(target)
    data[SCHEDULES_KEY] = schedules
    history_save(target, data)


def save(target, name: str, *, profile: Optional[str] = None,
         preset: Optional[str] = None, window: Optional[str] = None,
         on_battery: Optional[bool] = None) -> Dict[str, Any]:
    """Define a schedule: an action (a saved profile OR a preset name)
    plus a condition (a daily time window, an on-battery flag, or
    both). The action's existence is checked NOW (profiles via the
    history file, presets via the registry); the condition must be
    expressible and non-empty. Writes only the history file's
    "schedules" key."""
    if not NAME_RE.fullmatch(str(name or "")):
        raise ScheduleError(
            f"schedule names must match {NAME_RE.pattern!r} — got {name!r}")
    if (profile is None) == (preset is None):
        raise ScheduleError(
            "a schedule needs exactly one action: --profile NAME or "
            "--preset NAME")
    if window is None and on_battery is None:
        raise ScheduleError(
            "a schedule needs a condition: --window HH:MM-HH:MM and/or "
            "--on-battery true|false")
    parsed_window = _parse_window(window) if window is not None else None
    if profile is not None:
        from assistant.capabilities.settings.profiles import ProfileError, profile_ops
        try:
            profile_ops(target, profile)
        except ProfileError as exc:
            raise ScheduleError(f"profile action: {exc}") from None
    else:
        from assistant.capabilities.settings.presets import preset_by_name, preset_ops
        try:
            if preset_by_name(preset or "") is None:
                from assistant.capabilities.settings.presets import PresetError
                raise PresetError(f"unknown preset {preset!r}")
            preset_ops(preset or "")
        except Exception as exc:
            raise ScheduleError(f"preset action: {exc}") from None
    data = _load(target)
    if any(s.get("name") == name for s in data):
        raise ScheduleError(
            f"schedule {name!r} already exists; delete it first")
    entry = {
        "name": name,
        "action": {"profile": profile} if profile is not None
                  else {"preset": preset},
        "window": window,
        "on_battery": on_battery,
    }
    data.append(entry)
    _save_all(target, data[-MAX_SCHEDULES:])
    return dict(entry)


def list_schedules(target) -> List[Dict[str, Any]]:
    return _load(target)


def delete(target, name: str) -> Dict[str, Any]:
    schedules = _load(target)
    kept = [s for s in schedules if s.get("name") != name]
    if len(kept) == len(schedules):
        raise ScheduleError(
            f"unknown schedule {name!r}; saved schedules: "
            + (", ".join(str(s.get("name")) for s in schedules) if schedules
               else "(none)"))
    _save_all(target, kept)
    return {"deleted": name}


def evaluate(target, *, now: datetime,
             power: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Which schedules' conditions hold right now? Read-only. Power-
    conditioned schedules without a probe result are SKIPPED with the
    reason (never guessed); actions naming missing profiles are
    SKIPPED and named (F7)."""
    firing: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for sched in _load(target):
        name = str(sched.get("name"))
        window_text = sched.get("window")
        want_battery = sched.get("on_battery")
        if window_text is not None and not _in_window(
                now, _parse_window(str(window_text))):
            continue
        if want_battery is not None:
            if power is None:
                skipped.append({
                    "schedule": name,
                    "reason": "condition includes a power state but no "
                              "power probe was provided (--probe); not "
                              "guessed",
                })
                continue
            discharging = (not power.get("charging", True)
                           and power.get("percent") is not None)
            if bool(want_battery) != discharging:
                continue
        action = sched.get("action") or {}
        if action.get("profile") is not None:
            from assistant.capabilities.settings.profiles import profile_ops
            try:
                profile_ops(target, str(action["profile"]))
                command = (f"SUGGESTED_NOT_EXECUTED: caelestia-assist "
                           f"settings --profile {action['profile']} "
                           f"--apply --confirm")
            except Exception as exc:
                skipped.append({"schedule": name,
                                "reason": f"action no longer resolves: "
                                          f"{exc}"})
                continue
        else:
            command = (f"SUGGESTED_NOT_EXECUTED: caelestia-assist "
                       f"settings --preset {action.get('preset')} "
                       f"--apply --confirm")
        firing.append({
            "schedule": name,
            "action": action,
            "window": window_text,
            "on_battery": want_battery,
            "command": command,
        })
    return {"firing": firing, "skipped": skipped, "now": now,
            "checked": len(_load(target))}


def render_eval(result: Dict[str, Any]) -> List[str]:
    lines = [f"schedule evaluation at "
             f"{result['now'].isoformat(timespec='minutes')} "
             f"({result['checked']} schedule(s) checked):"]
    if not result["firing"]:
        lines.append("  firing now: none")
    for f in result["firing"]:
        lines.append(f"  FIRING {f['schedule']!r} "
                     f"(window {f['window']}"
                     + (f", on-battery {f['on_battery']})"
                        if f["on_battery"] is not None else ")"))
        lines.append(f"    {f['command']}")
    for s in result["skipped"]:
        lines.append(f"  SKIPPED {s['schedule']!r}: {s['reason']}")
    return lines


def file_firing(eval_result: Dict[str, Any], ledger,
                now: Optional[datetime] = None) -> Dict[str, int]:
    """File the FIRING schedules as proposals into the brain ledger —
    the same #120 proposal surface, the same dedupe/cooldown discipline
    as cortex.personalize. Only the ledger is written."""
    from . import personalize

    stats = {"filed": 0, "skipped_pending": 0, "skipped_cooldown": 0,
             "skipped_quiet": 0}
    pending_targets = {str(p.get("target")) for p in ledger.pending()}
    cooldown = timedelta(days=personalize.REJECT_COOLDOWN_DAYS)
    for f in eval_result["firing"]:
        action = f["action"]
        label = (f"profile: {action['profile']}"
                 if action.get("profile") is not None
                 else f"preset: {action.get('preset')}")
        target = (f"suggestion:schedule:"
                  f"{_slug(f'schedule-{f['schedule']}')}") \
            if False else f"suggestion:schedule:{_slug(f['schedule'])}"
        if target in pending_targets:
            stats["skipped_pending"] += 1
            continue
        decided = [p for p in ledger.labeled()
                   if str(p.get("target")) == target]
        if decided:
            latest = max(str(p.get("decided_at") or "") for p in decided)
            try:
                when = datetime.fromisoformat(latest)
            except ValueError:
                when = None
            if when is not None:
                if when.tzinfo is not None:
                    when = when.replace(tzinfo=None)
                anchor = now or datetime(2026, 1, 1, 12, 0, 0)
                if (anchor - when) < cooldown:
                    stats["skipped_cooldown"] += 1
                    continue
        why = (f"schedule {f['schedule']!r} fires now — its action "
               f"({label!r}) rides the normal consent gate")
        ledger.propose(SUGGESTION_KIND, target,
                       {"kind": "schedule_firing",
                        "schedule": f["schedule"],
                        "label": label,
                        "command": f["command"]},
                       why, 0.7)
        pending_targets.add(target)
        stats["filed"] += 1
    return stats


def _slug(text: str) -> str:
    import re as _re
    slug = _re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return slug[:48] or "unnamed"
