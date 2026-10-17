"""Brain service layer: every shell-native operation as a function returning
plain data.

The CLI (cli.py) and the JSON bridge (bridge.py) both call these, so the logic
exists exactly once. Nothing here prints; nothing here executes or networks.

Scope note (issue #120 split): the personal-knowledge-management operations
(vault organize/tag/plan/review/remind/cull/links/keywords/summarize/
spellcheck/ghosts/journal/health) moved to assistant/brain/personal/ with
their own entry point (`python3 -m assistant.capabilities.brain.personal`). This module
keeps only operations with direct shell linkage: the proposal ledger, the
settings bridge (#120), rhythm over shell events, placement, calibration,
drift, forecast/focus analytics over caller-supplied series, and the
assistant's own batch-window scheduler (dreamtime).
"""
from . import state as st
from .anomaly import shannon_bits, zscore
from .calibrate import DailyBudget, acceptance_rate, confidence_calibration
from .drift import diff as drift_diff
from .forecast import holt
from .ledger import Ledger
from .placement import suggest as placement_suggest
from .preset_bandit import NamedBandit
from .rhythm import day_of_week_pattern, hour_of_day_pattern, notable
from . import settings_bridge
from .dreamtime import eligible as dream_eligible, schedule as dream_schedule_jobs


# ---- forecast / anomaly / focus --------------------------------------------

def forecast(series, horizon=7):
    series = [float(x) for x in series]
    z = zscore(series[:-1], series[-1]) if len(series) >= 3 else None
    return {"forecast": holt(series, horizon=horizon), "anomaly_z": z}


def focus(labels):
    bits = shannon_bits(labels)
    return {"bits": bits, "fragmented": bits > 2.0}


# ---- ledger ----------------------------------------------------------------

def ledger_list(ledger_path):
    return Ledger(ledger_path).pending()


def ledger_decide(pid, approve, ledger_path):
    return Ledger(ledger_path).decide(int(pid), bool(approve))


# ---- settings (issue #120), via the ledger ---------------------------------

def settings_propose(file_path, ledger_path, preset=None, calls=None, reason=None,
                     confidence=None):
    ledger = Ledger(ledger_path)
    return settings_bridge.propose(ledger, file_path, preset=preset, calls=calls,
                                   reason=reason, confidence=confidence)


def settings_decide(pid, approve, ledger_path, state_path, battery_reward=None):
    ledger = Ledger(ledger_path)
    s = st.load(state_path)
    bandit = NamedBandit.from_dict(s.get("preset_bandit", {}))
    outcome = settings_bridge.decide(ledger, int(pid), bool(approve), bandit=bandit,
                                     battery_reward=battery_reward)
    s["preset_bandit"] = bandit.to_dict()
    st.save(s, state_path)
    return outcome


def settings_recommend(state_path, candidates=None):
    bandit = NamedBandit.from_dict(st.load(state_path).get("preset_bandit", {}))
    return settings_bridge.recommend(bandit, candidate_presets=candidates)


def settings_recommend_tools(state_path, tool_names=None):
    bandit = NamedBandit.from_dict(st.load(state_path).get("preset_bandit", {}))
    return settings_bridge.recommend_tools(bandit, tool_names)


# ---- rhythm -------------------------------------------------------------------

def rhythm_report(weekdays, hours, threshold=1.5):
    dow = day_of_week_pattern(weekdays)
    hod = hour_of_day_pattern(hours)
    return {"day_of_week": dow, "hour_of_day": hod,
            "notable_days": notable(dow, threshold), "notable_hours": notable(hod, threshold)}


def settings_rhythm(target, scheme_switches=None, ledger_path=None,
                    threshold=1.5):
    """Issue #120 Phase 2.6: the rhythm engine fed by SHELL events instead
    of generic caller lists.

    Event sources, all read-only:
    - settings applies: the undo history ring of the target file
      (settings/history.py records an ISO 'at' per apply);
    - ledger decisions of kind 'settings' (decided_at stamps);
    - scheme-switch timestamps: the scheme system persists only the
      CURRENT scheme (verified upstream: Colours.qml writes scheme.json,
      no switch history), so switch times arrive caller-supplied —
      the same honesty rule as every caller-supplied series here.

    Returns the plain rhythm_report shape plus the event count.
    """
    import json as _json
    from datetime import datetime as _dt
    from pathlib import Path as _Path

    from assistant.capabilities.settings import history as _history

    stamps = []
    try:
        stamps.extend(e["at"] for e in _history.entries(target) if e.get("at"))
    except _history.HistoryError:
        pass
    if ledger_path:
        p = _Path(ledger_path)
        if p.exists():
            try:
                data = _json.loads(p.read_text(encoding="utf-8"))
                stamps.extend(i["decided_at"] for i in data.get("proposals", [])
                              if i.get("kind") == "settings" and i.get("decided_at"))
            except (OSError, ValueError):
                pass
    stamps.extend(str(s) for s in (scheme_switches or []))

    weekdays = []
    hours = []
    for raw in stamps:
        try:
            stamp = _dt.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError:
            continue  # unparseable stamps are skipped, never guessed
        weekdays.append(stamp.weekday())
        hours.append(stamp.hour)
    result = rhythm_report(weekdays, hours, threshold=threshold)
    result["events"] = len(weekdays)
    return result


# ---- config/target placement (ISS-120-safe: proposal only, never a write) ----

def placement_propose(text, registry, ledger_path=None, propose=False, top=5):
    ranked = placement_suggest(text, registry, top=top)
    if propose and ledger_path is not None and ranked:
        ledger = Ledger(ledger_path)
        best = ranked[0]
        ledger.propose("placement", best["id"], {"query": text, "candidates": ranked},
                       f"closest match to '{text}'", best["score"])
    return ranked


# ---- calibration & proposal budget --------------------------------------------

def calibration_report(ledger_path, bins=5):
    labeled = Ledger(ledger_path).labeled()
    return {"acceptance_by_kind": acceptance_rate(labeled),
            "confidence_calibration": confidence_calibration(labeled, bins=bins)}


def budget_choose(state_path):
    s = st.load(state_path)
    budget = DailyBudget.from_dict(s.get("budget", {}))
    return budget.choose()


def budget_feedback(arm, engaged, state_path):
    s = st.load(state_path)
    budget = DailyBudget.from_dict(s.get("budget", {}))
    budget.reward(arm, bool(engaged))
    s["budget"] = budget.to_dict()
    st.save(s, state_path)


# ---- drift ---------------------------------------------------------------------

def drift_check(old_snapshot, new_snapshot, ledger_path=None, propose=False,
                similarity_threshold=0.85):
    result = drift_diff(old_snapshot, new_snapshot, similarity_threshold=similarity_threshold)
    if propose and ledger_path is not None:
        ledger = Ledger(ledger_path)
        for item_id in result["added"]:
            ledger.propose("drift_added", item_id, {"id": item_id}, "new since last snapshot", 0.9)
        for row in result["changed"]:
            ledger.propose("drift_changed", row["id"], row,
                           f"drifted (similarity={row['similarity']})", 1 - row["similarity"])
    return result


# ---- workspace profiles & topology memory (issue #120 phase 2) --------------

def workspace_profiles(records, ledger_path=None, propose=False, k=None,
                       min_support=3, purity=0.6):
    """Session records -> k-means clusters -> ledger profile proposals.
    Records come from a caller-supplied JSON (the shell persists no
    session log — see workspace.py's data-source note)."""
    from . import workspace as _workspace
    if propose and ledger_path is None:
        raise ValueError("propose=True needs a ledger path")
    if not propose:
        return _workspace.cluster(records, k=k, min_support=min_support,
                                   purity=purity)
    ledger = Ledger(ledger_path)
    return _workspace.propose_profiles(records, ledger, k=k,
                                       min_support=min_support,
                                       purity=purity)


def topology_observe(target, state_path, ledger_path=None, propose=False):
    """Monitor-topology memory: fingerprint the connected-monitor set,
    remember its config deltas, and on topology change propose the
    remembered deltas through the ledger (never auto-apply)."""
    from . import topology as _topology
    ledger = Ledger(ledger_path) if (propose and ledger_path) else None
    return _topology.observe(target, state_path, ledger=ledger,
                             propose=propose)


# ---- idle-time scheduling --------------------------------------------------------

def dream_window(idle_minutes, on_ac_power, cpu_load_percent, jobs, budget_min):
    if not dream_eligible(idle_minutes, on_ac_power, cpu_load_percent):
        return {"eligible": False, "chosen": []}
    return {"eligible": True, "chosen": dream_schedule_jobs(jobs, budget_min)}


def dream_preference_prompt(state_path, now=None):
    """exponential-build-3 G2: the idle-cadence preference invitation.
    One job a dream window may run: MAYBE record one pairwise
    comparison prompt (frequency-capped at one per 7 days, one
    unanswered at a time, 14-day expiry — brain/preference_prompts.py
    carries the full contract). The invitation surfaces in
    `settings --rank` and is answered by the ordinary `--prefer`,
    which consumes it. Writes only through the existing brain-state
    path."""
    from datetime import datetime, timezone
    from . import preference_prompts
    now = now or datetime.now(timezone.utc)
    state = st.load(state_path)
    result = preference_prompts.maybe_prompt(state, now)
    st.save(result["state"], state_path)
    return {"prompt": result["prompt"], "reason": result["reason"]}
