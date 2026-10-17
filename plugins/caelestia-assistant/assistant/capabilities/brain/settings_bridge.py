"""Wires the settings layer (issue #120) into the brain's proposal ledger.

Nothing in this module is a new write path. It only orchestrates three
pieces that already exist and already own their own safety guarantees:

- assistant.capabilities.settings.planner:  ops -> a validated, read-only plan
- assistant.capabilities.settings.applier:  plan -> a gated write (write=True only)
- assistant.capabilities.brain.ledger:      every autonomous suggestion is a proposal
                                until the user decides

The point is to make "the assistant proposes, the user approves" (#120's
own framing, and the same pattern brain/service.py already uses for tag
and dedup suggestions) the *one* interaction model for settings changes
too, instead of a separate CLI-only --confirm prompt: a proposal sits in
the ledger, durable and inspectable, until ledger_decide() resolves it.
Approving a proposal here is the literal act of confirmation (#120's
"Apply these changes?"), so -- and only then -- this module calls the
applier with write=True. Rejecting, or leaving it pending, writes nothing.

A settings proposal also feeds preset_bandit.NamedBandit -- for a named
preset directly under its own name, and for *any* proposal (preset or raw
--call) once more per touched tool, under a "tool:" prefix -- so approvals
and rejections become the learning signal for recommend(), and both
"which preset do you keep" and "which individual settings do you keep"
views improve, without ever touching which presets exist or how a tool
validates.

When the caller does not pin a confidence, propose() asks calibrate.py's
acceptance_rate for this ledger's own history on kind="settings": a user
who has been rejecting settings proposals gets a lower default confidence
next time (an honestly calibrated number, per calibrate.py's own stated
purpose), instead of a flat, always-optimistic 0.7.
"""
from assistant.capabilities.settings import applier, planner, presets
from assistant.capabilities.settings import history as settings_history
from assistant.capabilities.settings.presets import PresetError
from .calibrate import acceptance_rate, fold_undo_negatives
from .preset_bandit import NamedBandit

TOOL_ARM_PREFIX = "tool:"
DEFAULT_CONFIDENCE = 0.7


class SettingsBridgeError(RuntimeError):
    """Raised when a proposal request cannot even be planned."""


def _preview_lines(plan):
    lines = []
    for entry in plan.get("entries", []):
        if entry.get("error") or entry.get("no_op"):
            continue
        lines.append(f"{entry.get('path')}: {entry.get('old', '(unset)')} -> {entry.get('new')}")
    return lines


def _ops_for(preset, calls):
    if preset:
        try:
            return presets.preset_ops(preset)
        except PresetError as exc:
            raise SettingsBridgeError(str(exc)) from exc
    return list(calls or [])


def _calibrated_confidence(ledger, undo_records=None):
    """The running approval rate for kind="settings", falling back to
    DEFAULT_CONFIDENCE with no history yet (a flat, uninformative prior).

    A3: the target file's PII-stripped undo log (if the caller passes the
    records — undo_records is history.undo_log(target)) is folded in as
    explicit negative evidence BEFORE the mean is read: a user who
    reverts changes the assistant applied is teaching the posterior the
    same lesson a rejected proposal teaches, and the next proposal's
    default confidence shows it."""
    stats = acceptance_rate(ledger.labeled("settings"))
    if undo_records:
        fold_undo_negatives(stats, undo_records, kind="settings")
    settings_stats = stats.get("settings")
    if settings_stats is None:
        return DEFAULT_CONFIDENCE
    return settings_stats["mean"]


def propose(ledger, file_path, preset=None, calls=None, reason=None, confidence=None):
    """Dry-run plan a preset or an explicit op list; store it as a pending
    ledger proposal if it has any applicable change. Never writes.

    ``confidence`` defaults to this ledger's own calibrated settings
    acceptance rate (see _calibrated_confidence) rather than a fixed number.

    Returns {"proposal_id": int | None, "preview": [str, ...],
             "blocked": bool, "errors": [str, ...]}. proposal_id is None
    when there is nothing to approve (blocked, or a no-op plan) -- the
    ledger stays clean of proposals nobody could ever approve.
    """
    ops = _ops_for(preset, calls)
    plan = planner.plan(ops, file_path)
    entries = plan.get("entries", [])
    errors = [e.get("error") for e in entries if e.get("error")]
    applicable = [e for e in entries if not e.get("error") and not e.get("no_op")]
    blocked = bool(plan.get("apply_blocked"))

    if blocked or not applicable:
        return {"proposal_id": None, "preview": _preview_lines(plan),
                "blocked": blocked, "errors": errors}

    if confidence is None:
        # A3: read the target's undo log (read-only; PII-stripped records
        # only) and fold it in as explicit negative signal.
        try:
            undo_records = settings_history.undo_log(file_path)
        except settings_history.HistoryError:
            undo_records = []
        confidence = _calibrated_confidence(ledger, undo_records)

    diff = {"preset": preset, "calls": ops, "file": str(file_path)}
    label = reason or (f"preset '{preset}'" if preset else "settings change")
    pid = ledger.propose("settings", str(file_path), diff, label, confidence)
    return {"proposal_id": pid, "preview": _preview_lines(plan),
            "blocked": False, "errors": errors}


def decide(ledger, proposal_id, approve, bandit=None, battery_reward=None):
    """Resolve a pending settings proposal.

    approve=True replans the stored calls fresh (the target file may have
    changed since propose()) and writes through the applier -- the one
    module in the whole tree allowed to write, exactly as if the user had
    passed --apply --confirm at the CLI. approve=False writes nothing.
    Either way, this decision rewards the bandit: once under the preset's
    own name (if any), and once per distinct tool actually touched, so
    per-tool acceptance can be learned even from raw --call proposals.

    ``battery_reward`` (issue #120 Phase 3.2) is an OPTIONAL secondary
    signal in [0, 1] (0.5 neutral) computed by the caller from battery
    drain-rate deltas measured around the preset's active window
    (diagnostics/telemetry.py::reward_from_drain). It only ever adds
    fractional pseudo-counts (weight 0.25) on top of the primary
    approve/reject signal — never replaces it, and is a strict no-op
    (byte-identical updates) when None, i.e. when telemetry is
    unavailable.
    """
    item = ledger.decide(proposal_id, approve)
    diff = item["diff"]
    outcome = {"item": item, "applied": False}

    if approve:
        plan = planner.plan(diff["calls"], diff["file"])
        applied = applier.apply(plan, diff["file"], write=True, label=item["reason"])
        outcome["applied"] = bool(applied.get("written"))
        outcome["apply_result"] = applied

    if bandit is not None:
        if diff.get("preset"):
            bandit.reward(diff["preset"], approve, secondary=battery_reward)
        for tool_name in {c.get("tool") for c in diff.get("calls", []) if c.get("tool")}:
            bandit.reward(TOOL_ARM_PREFIX + tool_name, approve)

    return outcome


def recommend(bandit, candidate_presets=None, rng=None):
    """Rank presets by learned acceptance, best first -- a suggestion list,
    never an auto-apply. candidate_presets defaults to every named preset.
    """
    catalog = {p["name"]: p for p in presets.presets()}
    names = candidate_presets or list(catalog)
    ranked = bandit.rank(names, rng=rng)
    return [
        {"name": name, "score": draw, "confidence": mean,
         "label": catalog[name]["label"], "description": catalog[name]["description"]}
        for name, draw, mean in ranked if name in catalog
    ]


def recommend_tools(bandit, candidate_tools=None, rng=None):
    """Rank individual tool names by learned acceptance, best first.

    With no explicit ``candidate_tools``, ranks every tool this bandit has
    already seen at least one decision for (i.e. every "tool:" arm) --
    there is no fixed catalog to fall back to the way recommend() has one.
    """
    if candidate_tools is None:
        candidate_tools = [name[len(TOOL_ARM_PREFIX):] for name in bandit.arms
                          if name.startswith(TOOL_ARM_PREFIX)]
    arms = [TOOL_ARM_PREFIX + name for name in candidate_tools]
    ranked = bandit.rank(arms, rng=rng)
    return [
        {"tool": arm[len(TOOL_ARM_PREFIX):], "score": draw, "confidence": mean}
        for arm, draw, mean in ranked
    ]
