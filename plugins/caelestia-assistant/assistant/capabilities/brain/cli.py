"""brain CLI: formats shell-native service results as text. Every change is a
proposal.

Scope note (issue #120 split): the personal-knowledge-management commands
(organize, tag, plan, estimate, review, remind, cull, links, keywords,
summarize, spellcheck, ghosts, journal, health, ledger learn) moved to
assistant/brain/personal/ with their own entry point
(`python3 -m assistant.capabilities.brain.personal --help`). This CLI keeps only
operations with direct shell linkage:

  python3 -m assistant.capabilities.brain forecast 3,4,5,6 [--horizon 7]
  python3 -m assistant.capabilities.brain focus cat1,cat2,cat1,...
  python3 -m assistant.capabilities.brain rhythm [--weekdays 0,1,1,2] [--hours 9,9,14]
  python3 -m assistant.capabilities.brain calibration
  python3 -m assistant.capabilities.brain ledger list | approve ID | reject ID
  python3 -m assistant.capabilities.brain settings propose FILE [--preset NAME] [--call TOOL=VALUE ...] [--reason TEXT]
  python3 -m assistant.capabilities.brain settings decide ID approve|reject
  python3 -m assistant.capabilities.brain settings recommend [--top N] [--tools]
  python3 -m assistant.capabilities.brain brief
  python3 -m assistant.capabilities.brain tidy survey ROOT | rollback
  python3 -m assistant.capabilities.brain prefs
"""
import argparse
import json
import os
import pathlib
import sys
from datetime import datetime, timezone

from . import merkle
from . import service
from . import state as st

DEFAULT_LEDGER = pathlib.Path.home() / ".local/state/caelestia-brain/ledger.json"


def cmd_forecast(args, out):
    r = service.forecast([x for x in args.series.split(",")], horizon=args.horizon)
    out.write(", ".join(f"{v:.2f}" for v in r["forecast"]) + "\n")
    if r["anomaly_z"] is not None:
        out.write(f"latest value z-score vs history: {r['anomaly_z']:.2f}"
                  f"{'  (anomaly)' if abs(r['anomaly_z']) >= 2 else ''}\n")
    return 0


def cmd_focus(args, out):
    r = service.focus(args.labels.split(","))
    out.write(f"switch entropy: {r['bits']:.2f} bits"
              f"{'  (fragmented: consider a single-task block)' if r['fragmented'] else ''}\n")
    return 0


def cmd_rhythm(args, out):
    if args.settings_file or args.scheme_switches:
        # Phase 2.6: rhythm over shell events (settings applies from the
        # undo history + ledger decisions + any caller-supplied scheme
        # switch timestamps).
        r = service.settings_rhythm(args.settings_file,
                                    scheme_switches=(args.scheme_switches.split(",")
                                                     if args.scheme_switches else None),
                                    ledger_path=args.ledger)
        out.write(f"events: {r['events']}\n")
        for row in r["notable_days"]:
            out.write(f"day {row['day']}: z={row['z']} (n={row['count']})\n")
        for row in r["notable_hours"]:
            out.write(f"hour {row['hour']}: z={row['z']} (n={row['count']})\n")
        return 0
    weekdays = [int(x) for x in args.weekdays.split(",")] if args.weekdays else []
    hours = [int(x) for x in args.hours.split(",")] if args.hours else []
    r = service.rhythm_report(weekdays, hours)
    for row in r["notable_days"]:
        out.write(f"day {row['day']}: z={row['z']} (n={row['count']})\n")
    for row in r["notable_hours"]:
        out.write(f"hour {row['hour']}: z={row['z']} (n={row['count']})\n")
    return 0


def cmd_workspace(args, out):
    records = json.loads(pathlib.Path(args.sessions).read_text(encoding="utf-8"))
    r = service.workspace_profiles(records, args.ledger, propose=args.propose,
                                   k=args.k, min_support=args.min_support,
                                   purity=args.purity)
    if args.propose:
        for pid in r["proposals"]:
            out.write(f"proposal #{pid} pending (approve: brain ledger approve {pid})\n")
        if not r["proposals"]:
            out.write("no consistent co-occurrence found — nothing proposed\n")
    else:
        out.write(f"sessions: {r['n_sessions']}  clusters: {r['k']}  "
                  f"rejected: {r['rejected_clusters']}\n")
        for pr in r["profiles"]:
            out.write(f"  {pr['name']}: app={pr['app']} monitor={pr['monitor']} "
                      f"ws={pr['workspace']} support={pr['support']} "
                      f"purity={pr['purity']} mean-hour={pr['hours_mean']}\n")
        if not r["profiles"]:
            out.write("no consistent co-occurrence found\n")
        out.write("(dry-run: nothing proposed; add --propose to write "
                  "ledger proposals)\n")
    return 0


def cmd_topology(args, out):
    from assistant.capabilities.settings.cli import default_target
    r = service.topology_observe(args.file or default_target(), args.state,
                                 args.ledger, propose=args.propose)
    fp = r["fingerprint"]
    out.write(f"monitors: {', '.join(fp['monitors']) or '(none)'}  hash: {fp['hash']}\n")
    out.write(f"changed since last observe: {r['changed']}  "
              f"remembered deltas for this set: {r['remembered']}\n")
    out.write(f"current override deltas: {len(r['deltas'])}\n")
    if r["proposal_id"] is not None:
        out.write(f"proposal #{r['proposal_id']} pending (approve: brain ledger approve {r['proposal_id']})\n")
    return 0


def cmd_calibration(args, out):
    r = service.calibration_report(args.ledger)
    for kind, s in r["acceptance_by_kind"].items():
        out.write(f"{kind}: mean={s['mean']} n={s['n']}\n")
    # exponential-build 3.3: the preset bandit's regret vs always
    # playing its best arm — an estimate, printed for the human.
    from . import state as brain_state
    from .preset_bandit import NamedBandit
    from .regret import audit_from_arms
    state_path = brain_state.resolve_path()
    if state_path.exists():
        state = brain_state.load(state_path)
        bandit = NamedBandit.from_dict(state.get("preset_bandit", {}))
        audit = audit_from_arms(bandit.arms)
        out.write(f"preset bandit regret: {audit.get('status')} "
                  f"(est. {audit.get('estimated_regret')} vs "
                  f"always-{audit.get('best_fixed_arm')})\n")
    for row in r["confidence_calibration"]:
        out.write(f"  bucket {row['bucket']}: n={row['n']} "
                  f"avg_conf={row['avg_confidence']} approval_rate={row['approval_rate']}\n")
    return 0


def cmd_ope(args, out):
    """exponential-build 3.2: the off-policy evaluation report for a
    kill-switch candidate policy. Reads the ledger, replays it, prints
    the evidence — and writes NOTHING anywhere (the switch stays the
    manifest file the human edits)."""
    from . import ope as ope_mod
    if args.list_switches:
        for name in sorted(ope_mod.KNOWN_SWITCHES):
            out.write(f"{name}: {ope_mod.KNOWN_SWITCHES[name]}\n")
        return 0
    from .ledger import Ledger
    ledger = Ledger(args.ledger)
    episodes = [{"context": {"kind": i.get("kind"),
                             "target": i.get("target"),
                             "diff": i.get("diff"),
                             "confidence": i.get("confidence")},
                 "approved": i.get("status") == "approved"}
                for i in ledger.labeled()]
    try:
        report = ope_mod.ope_report(episodes, args.switch)
    except ValueError as exc:
        out.write(f"error: {exc}\n")
        return 1
    out.write(f"kill-switch candidate policy: {args.switch}\n")
    if report.get("refused"):
        out.write(f"REFUSED: {report['refused']}\n")
    else:
        out.write(f"estimated accept rate if proposed: "
                  f"{report['estimated_accept_rate']}\n")
    out.write(f"support: {report['support']} of {report['n_episodes']} "
              f"logged episodes (coverage {report['coverage']})\n")
    out.write(f"{report['note']}\n")
    return 0


def cmd_ledger(args, out):
    if args.action == "list":
        pending = service.ledger_list(args.ledger)
        for i in pending:
            out.write(f"#{i['id']} [{i['kind']}] {i['target']} conf={i['confidence']} :: {i['reason']}\n")
        if not pending:
            out.write("no pending proposals\n")
    elif args.action in ("approve", "reject"):
        service.ledger_decide(args.id, args.action == "approve", args.ledger)
        out.write(f"{args.action}d #{args.id}\n")
    return 0


def _parse_call_value(text):
    try:
        return json.loads(text)
    except ValueError:
        return text


def _calls_to_ops(calls):
    ops = []
    for call in calls or []:
        name, sep, value_text = call.partition("=")
        if not sep or not name:
            raise ValueError(f"--call expects NAME=VALUE (got {call!r})")
        ops.append({"tool": name, "action": "set", "value": _parse_call_value(value_text),
                    "raw": call})
    return ops


def cmd_settings(args, out):
    if args.action == "propose":
        file_path = args.arg1
        if not file_path:
            out.write("settings propose needs FILE\n")
            return 2
        r = service.settings_propose(file_path, args.ledger, preset=args.preset,
                                     calls=_calls_to_ops(args.calls), reason=args.reason)
        if r["proposal_id"] is None:
            out.write("nothing to propose"
                      + (" (blocked: see errors)" if r["blocked"] else " (no applicable change)")
                      + "\n")
            for err in r["errors"]:
                out.write(f"  error: {err}\n")
        else:
            out.write(f"proposal #{r['proposal_id']} pending "
                      f"(brain settings decide {r['proposal_id']} approve|reject):\n")
            for line in r["preview"]:
                out.write(f"  - {line}\n")
    elif args.action == "decide":
        pid = int(args.arg1)
        approve = args.arg2 == "approve"
        r = service.settings_decide(pid, approve, args.ledger, args.state,
                                    battery_reward=args.battery_reward)
        if r["applied"]:
            out.write(f"#{pid} approved and written\n")
        elif approve:
            out.write(f"#{pid} approved but nothing was written: "
                      f"{r.get('apply_result', {}).get('message', '')}\n")
        else:
            out.write(f"#{pid} rejected; nothing written\n")
    else:  # recommend
        if args.tools:
            recs = service.settings_recommend_tools(args.state)
            if not recs:
                out.write("no tool-level history yet\n")
            for rec in recs[:args.top]:
                out.write(f"{rec['tool']:<20} conf={rec['confidence']}\n")
        else:
            for rec in service.settings_recommend(args.state)[:args.top]:
                out.write(f"{rec['name']:<14} conf={rec['confidence']:<5} {rec['label']}\n")
    return 0


def cmd_brief(args, out):
    """Daily brief: compose what the ledger + forecasts already know."""
    from . import brief as brief_mod
    from .ledger import Ledger
    ledger = Ledger(args.ledger)
    drift = None
    if getattr(args, "config_drift", False):
        drift = _config_drift_payload(args)
    brief = brief_mod.compose(pending=ledger.pending(), config_drift=drift)
    out.write(brief_mod.render(brief) + "\n")
    return 0


def _config_drift_payload(args):
    """B2 opt-in: read the live shell.json (read-only, the planner's own
    reader) and score it against the preference posterior learned from
    the ledger. Returns the drift dict, or a thin 'why not' marker."""
    from pathlib import Path

    from . import prefs as prefs_mod
    from .ledger import Ledger
    from assistant.capabilities.settings import cli as settings_cli
    from assistant.capabilities.settings import planner
    target = getattr(args, "target", None) or settings_cli.default_target()
    try:
        current, _notes = planner._read_current(Path(target))
    except Exception as exc:  # PlannerError: unreadable/invalid JSON
        return {"evaluated": 0, "flags": [], "thin": [], "score": 0,
                "error": f"target unreadable ({exc}); drift not computed"}
    model = prefs_mod.PreferenceModel()
    model.from_ledger(Ledger(args.ledger).items)
    return prefs_mod.config_drift(model, current)


def cmd_tidy(args, out):
    """Filesystem organizer: survey (read-only) / rollback — apply goes via the agent."""
    from . import tidy as tidy_mod
    if args.action == "survey":
        plan = tidy_mod.survey(args.root)
        if args.json:
            out.write(json.dumps(plan, indent=2) + "\n")
        else:
            out.write(tidy_mod.render_plan(plan) + "\n")
        return 0
    if args.action == "rollback":
        res = tidy_mod.rollback(args.journal)
        out.write(f"undone {len(res['undone'])} move(s); "
                  f"missing {len(res['missing'])}\n")
        return 0
    out.write("tidy: use `survey ROOT`, `rollback` — moves are applied via the "
              "agent (`caelestia-assist agent 'clean my downloads'`), never "
              "blindly\n")
    return 2


def cmd_prefs(args, out):
    """Preference model: what the Beta posteriors currently believe."""
    from . import prefs as prefs_mod
    from .ledger import Ledger
    model = prefs_mod.PreferenceModel()
    ledger = Ledger(args.ledger)
    n = model.from_ledger(ledger.items)
    if n == 0:
        out.write("no decided proposals yet — approve/reject some and I learn "
                  "your patterns\n")
        return 0
    out.write(f"learned from {n} decision(s):\n")
    keys = sorted(model.table.keys())
    for key in keys[:12]:
        group, direction, bucket = key.split("|")
        line = model.explain(group, direction, hour=None)
        b = model.bias(group, direction, int(bucket) if int(bucket) >= 0 else None)
        out.write(f"  {group:<16} {direction:<9} bucket {bucket}: "
                  f"p={b['p_accept']} n={b['n']} ({b['verdict']})\n")
    return 0


def cmd_bisect(args, out):
    """F13 config bisect: mark good/bad, probe stepped, get a revert
    proposal. Every probe application is the user's own, through the
    settings consent gate; this command only reads config and stores
    marks."""
    from . import bisect as bs
    state = st.load(args.state)
    store = state.get("bisect") or {}
    root = args.root

    if args.action == "reset":
        state.pop("bisect", None)
        st.save(state, args.state)
        out.write("bisect state cleared\n")
        return 0

    if args.action == "mark":
        label = args.label
        if label not in ("good", "bad"):
            out.write("mark expects good|bad\n")
            return 2
        try:
            leaves = bs.leaf_map(os.path.expanduser(root))
        except bs.BisectError as e:
            out.write(f"cannot snapshot {root}: {e}\n")
            return 2
        store[label] = {"leaves": leaves,
                        "root": str(root),
                        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        store.pop("engine", None)  # marks changed: search restarts
        state["bisect"] = store
        st.save(state, args.state)
        out.write(f"marked {label}: {len(leaves)} leaves from {root}\n")
        if "good" in store and "bad" in store:
            keys = bs.changed_keys(store["good"]["leaves"], store["bad"]["leaves"])
            out.write(f"changed keys between marks: {len(keys)}\n")
        return 0

    good, bad = store.get("good"), store.get("bad")
    if not (good and bad):
        out.write("need both marks first: brain bisect mark good ; brain bisect mark bad\n")
        return 2
    keys = bs.changed_keys(good["leaves"], bad["leaves"])
    if not keys:
        out.write("good and bad snapshots are identical — nothing to bisect\n")
        return 2

    eng = bs.NoisyBisect.from_state(store["engine"]) if "engine" in store else None

    if args.action == "status":
        out.write(f"marks: good {good['ts']}, bad {bad['ts']}; changed keys: "
                  f"{len(keys)}; engine: "
                  f"{eng.phase if eng else 'not started'}"
                  + (f" ({eng.probe_count} probes)" if eng else "") + "\n")
        return 0

    if args.action == "next":
        if eng is None:
            eng = bs.NoisyBisect(keys)
        if eng.phase == "done":
            out.write("search finished — run: brain bisect proposal\n")
            return 0
        try:
            subset = eng.next_subset()
        except bs.BisectError as e:
            out.write(f"bisect cannot continue: {e}\n")
            return 2
        store["engine"] = eng.to_state()
        state["bisect"] = store
        st.save(state, args.state)
        out.write(f"probe #{eng.probe_count + 1}: set exactly these "
                  f"{len(subset)} key(s) to their BAD values, everything "
                  "else at the GOOD values:\n")
        proposal = bs.revert_ops(bad["leaves"], subset)
        for k in subset:
            out.write(f"  key {k} -> {bad['leaves'].get(k, '<absent in bad>')!r}\n")
        for op in proposal["ops"]:
            out.write(f"  -call {op['raw']}\n")
        for m in proposal["manual"]:
            out.write(f"  manual: {m['key']} -> {bad['leaves'].get(m['key'])!r}\n")
        out.write("apply through the settings consent gate, then answer:\n"
                  "  brain bisect observe bad   # the break reproduces\n"
                  "  brain bisect observe good  # it does not\n")
        return 0

    if args.action == "observe":
        if eng is None or eng.last_subset is None:
            out.write("no pending probe — run: brain bisect next\n")
            return 2
        if args.label not in ("good", "bad"):
            out.write("observe expects good|bad\n")
            return 2
        is_bad = args.label == "bad"
        eng.observe(is_bad)
        store["engine"] = eng.to_state()
        state["bisect"] = store
        st.save(state, args.state)
        out.write(f"recorded {args.label}; phase={eng.phase} "
                  f"probes={eng.probe_count}\n")
        if eng.phase == "done":
            out.write("minimal failing set found — run: brain bisect proposal\n")
        elif eng.phase == "inconclusive":
            out.write("probe budget exhausted without a minimal set; the "
                      "posterior ranking is the honest output so far\n")
            for k, m in eng.ranking()[:5]:
                out.write(f"  {k}: {m:.2f}\n")
        return 0

    if args.action == "proposal":
        if eng is None or eng.phase != "done":
            out.write("no finished search — run next/observe to completion "
                      "first\n")
            return 2
        prop = eng.proposal()
        rev = bs.revert_ops(good["leaves"], prop["minimal"])
        out.write(f"minimal failing set ({prop['probes']} probes):\n")
        for k in prop["minimal"]:
            out.write(f"  {k} (blame {prop['confidence'][k]:.2f})\n")
        out.write("revert proposal (dry-run; apply via the settings gate):\n")
        for op in rev["ops"]:
            out.write(f"  {op['tool']} = {json.dumps(op['value'])}\n")
        for m in rev["manual"]:
            out.write(f"  manual review: {m['key']} ({m['reason']})\n")
        return 0

    out.write(f"unknown bisect action {args.action!r}\n")
    return 2



def cmd_patterns(args, out):
    """One pattern card over a comma-separated series: seasonality,
    changepoints, motifs, discords (C11)."""
    from . import seasonal
    try:
        series = [float(x) for x in str(args.series).split(",")
                  if x.strip()]
    except ValueError as exc:
        out.write(f"patterns: unparseable number ({exc}); the series "
                  "argument is a comma-separated list of numbers\n")
        return 1
    r = seasonal.seasonality_report(series, period=args.period,
                                    window=args.window,
                                    penalty=args.penalty)
    if getattr(args, "json", False):
        out.write(json.dumps(r, sort_keys=True) + "\n")
        return 0
    out.write(f"series: {r['n']} points, period {r['period']}\n")
    se = r["seasonality"]
    if se.get("verdict") == "ABSTAIN":
        out.write(f"  seasonality: ABSTAIN ({se['note']})\n")
    else:
        verdict = "HAS RHYTHM" if se["has_rhythm"] else "no rhythm"
        out.write(f"  seasonality: {verdict} "
                  f"(strength {se['strength']})\n")
    cp = r["changepoints"]
    out.write(f"  changepoints: {cp['indices'] or 'none'}\n")
    pr = r["profile"]
    if pr.get("verdict") == "ABSTAIN":
        out.write(f"  profile: ABSTAIN ({pr['note']})\n")
    else:
        for mrec in pr["motifs"]:
            out.write(f"  motif at {mrec['index']} "
                      f"(d={mrec['distance']})\n")
        for drec in pr["discords"]:
            out.write(f"  discord at {drec['index']} "
                      f"(d={drec['distance']})\n")
    return 0



def cmd_bursts(args, out):
    """Hawkes burst detection over comma-separated event times (C12)."""
    from . import bursts
    try:
        events = [float(x) for x in str(args.events).split(",")
                  if x.strip()]
    except ValueError as exc:
        out.write(f"bursts: unparseable event time ({exc}); the events "
                  "argument is a comma-separated list of times\n")
        return 1
    r = bursts.burst_report(events, factor=args.factor)
    if getattr(args, "json", False):
        out.write(json.dumps(r, sort_keys=True) + "\n")
        return 0
    if r.get("verdict") == "ABSTAIN":
        out.write(f"bursts: ABSTAIN ({r['note']})\n")
        return 0
    fit = r["fit"]
    out.write(f"hawkes fit: mu={fit['mu']} alpha={fit['alpha']} "
              f"beta={fit['beta']} R={fit['branching_ratio']} "
              f"({'stable' if fit['stable'] else 'UNRELIABLE'})\n")
    out.write(f"bursts (>{args.factor}x base rate): {r['n_bursts']}\n")
    for b in r["bursts"]:
        out.write(f"  {b['start']} .. {b['end']}  peak {b['peak_intensity']}"
                  f" ({b['over_base']}x base, {b['n_events']} events)\n")
    out.write(f"  ({r['caveat']})\n")
    return 0



def cmd_gpprefs(args, out):
    """GP preference learning over pairwise "A or B?" answers (C13)."""
    from . import gp_prefs
    try:
        data = json.loads(pathlib.Path(args.data).read_text())
    except OSError as exc:
        out.write(f"gp-prefs: data file unreadable ({exc})\n")
        return 1
    except ValueError as exc:
        out.write(f"gp-prefs: data file is not valid JSON ({exc})\n")
        return 1
    if not isinstance(data, dict):
        out.write("gp-prefs: data must be an object with 'items' and "
                  "'pairs'\n")
        return 1
    items = data.get("items", [])
    pairs = data.get("pairs", [])
    try:
        m = gp_prefs.GPPreferenceModel(items)
        for pair in pairs:
            m.record(str(pair.get("winner")), str(pair.get("loser")))
        fit = m.fit()
    except ValueError as exc:
        out.write(f"gp-prefs: {exc}\n")
        return 1
    if getattr(args, "json", False):
        out.write(json.dumps(fit, sort_keys=True) + "\n")
        return 0
    out.write(f"ranking ({fit['n_pairs']} comparisons, "
              f"{fit['n_newton_steps']} Newton steps):\n")
    for rank, item_id in enumerate(fit["ranking"], 1):
        u = fit["utilities"][item_id]
        v = fit["variances"][item_id]
        out.write(f"  {rank}. {item_id}  utility {u} (var {v})\n")
    nxt = m.next_question()
    if nxt:
        out.write(f"next question: {nxt[0]} or {nxt[1]}?\n")
    # validated setter proposals: top items that name a real tool
    by_id = {str(it["id"]): it for it in items}
    # ONE proposal: the single highest-utility item that names a real
    # tool and passes the registry spec check. The ledger decides.
    for item_id in fit["ranking"]:
        item = by_id.get(item_id, {})
        if not item.get("tool"):
            continue
        check = gp_prefs.validate_item(item)
        if not check["validated"]:
            continue
        out.write(f"SUGGESTED_NOT_EXECUTED: {item['tool']} "
                  f"{item.get('value')}  (validated against the "
                  f"registry spec)\n")
        break
    return 0



def cmd_rules(args, out):
    """User-authored event-condition-action rules on a Rete network
    (C14). The rules file is the opt-in; actions are proposals only."""
    from . import rete
    path = pathlib.Path(args.file).expanduser()
    if not path.exists():
        out.write(f"rules: no rules file at {path} — create it to opt "
                  f"in (JSON: {json.dumps({'rules': [{'id': 'example', 'when': [{'field': 'kind', 'op': 'eq', 'value': 'crash'}], 'within_seconds': 600, 'min_events': 3, 'then': {'tool': 'setNotifsMaxPopups', 'value': 3}, 'reason': 'why not'}]})})\n")
        return 1
    try:
        rules = rete.load_rules(path)
    except ValueError as exc:
        out.write(f"rules: {exc}\n")
        return 1
    if args.events:
        try:
            events = json.loads(pathlib.Path(args.events).read_text())
        except (OSError, ValueError) as exc:
            out.write(f"rules: events file unreadable or not JSON "
                      f"({exc})\n")
            return 1
        if isinstance(events, dict):
            events = events.get("events", [])
        try:
            r = rete.run_events(rules, events)
        except ValueError as exc:
            out.write(f"rules: {exc}\n")
            return 1
        if getattr(args, "json", False):
            out.write(json.dumps(r, sort_keys=True) + "\n")
            return 0
        for line in rete.render_report(r):
            out.write(line + "\n")
        return 0
    out.write(f"{len(rules)} rules in {path}:\n")
    for rule in rules:
        out.write(f"  {rule.get('id')}: when {json.dumps(rule.get('when', []), sort_keys=True)}\n")
        out.write(f"    then {json.dumps(rule.get('then', {}), sort_keys=True)}\n")
    return 0


def rete_default_path():
    from . import rete
    return rete.DEFAULT_RULES_PATH



def cmd_sizes(args, out):
    """Space-Saving heavy hitters + t-digest quantiles for sizing (C15)."""
    from . import sketch
    try:
        sizes = [float(x) for x in str(args.sizes).split(",") if x.strip()]
    except ValueError as exc:
        out.write(f"sizes: unparseable number ({exc}); the sizes "
                  "argument is a comma-separated list of numbers\n")
        return 1
    if not sizes:
        out.write("sizes: need at least one size\n")
        return 1
    keys = [k for k in str(args.keys).split(",")] if args.keys else None
    quantiles = [float(q) for q in str(args.quantiles).split(",")
                 if q.strip()] if args.quantiles else None
    r = sketch.sizing_report(sizes, keys, k=args.k,
                             quantiles=quantiles)
    if getattr(args, "json", False):
        out.write(json.dumps(r, sort_keys=True) + "\n")
        return 0
    for line in sketch.render_sizing(r):
        out.write(line + "\n")
    return 0



def cmd_say(args, out):
    """Template NLG over a plan's changes: discourse plan + aggregation
    (D16). Reads a JSON file of changes; writes sentences."""
    from . import nlg
    try:
        data = json.loads(pathlib.Path(args.file).read_text())
    except OSError as exc:
        out.write(f"say: changes file unreadable ({exc})\n")
        return 1
    except ValueError as exc:
        out.write(f"say: changes file is not valid JSON ({exc})\n")
        return 1
    if isinstance(data, list):
        changes = data
    elif isinstance(data, dict):
        changes = data.get("changes", [])
    else:
        changes = []
    if getattr(args, "json", False):
        out.write(json.dumps({"lines": nlg.render_changes_summary(changes),
                              "paragraph": nlg.render_plan_paragraph(
                                  changes)}, sort_keys=True) + "\n")
        return 0
    for line in nlg.render_changes_summary(changes):
        out.write(line + "\n")
    out.write("\n" + nlg.render_plan_paragraph(changes) + "\n")
    return 0


def cmd_qa(args, out):
    """Extractive QA over repo docs + user notes (D17): BM25 passage
    retrieval, answer-type detection, verbatim spans with provenance.
    The answer is QUOTED, never written; thin evidence says so."""
    from . import qa
    notes = []
    for note_path in (getattr(args, "note", None) or []):
        p = pathlib.Path(note_path)
        try:
            text = p.read_text(encoding="utf-8")
        except OSError as exc:
            out.write(f"note unreadable, skipped: {note_path} ({exc})\n")
            continue
        notes.append({"name": p.name, "text": text})
    r = qa.answer(args.question, notes=notes, sources=args.sources)
    if getattr(args, "json", False):
        out.write(json.dumps(r, sort_keys=True) + "\n")
        return 0
    for line in qa.render_lines(r):
        out.write(line + "\n")
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="brain", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state", default=str(st.DEFAULT_STATE))
    p.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("forecast")
    f.add_argument("series")
    f.add_argument("--horizon", type=int, default=7)
    f.set_defaults(fn=cmd_forecast)

    fo = sub.add_parser("focus")
    fo.add_argument("labels")
    fo.set_defaults(fn=cmd_focus)

    rh = sub.add_parser("rhythm")
    rh.add_argument("--weekdays", default="")
    rh.add_argument("--hours", default="")
    rh.add_argument("--settings-file", default=None,
                    help="derive events from this file's settings-apply "
                         "history and ledger decisions instead of lists")
    rh.add_argument("--scheme-switches", default="",
                    help="comma-separated ISO scheme-switch timestamps "
                         "(the scheme system persists no switch history)")
    rh.set_defaults(fn=cmd_rhythm)

    ws = sub.add_parser("workspace", help="session clustering -> workspace "
                                          "profile proposals (issue #120)")
    ws.add_argument("sessions")
    ws.add_argument("--propose", action="store_true")
    ws.add_argument("--k", type=int, default=None)
    ws.add_argument("--min-support", type=int, default=3)
    ws.add_argument("--purity", type=float, default=0.6)
    ws.set_defaults(fn=cmd_workspace)

    tp = sub.add_parser("topology", help="per-monitor-topology config memory "
                                         "(issue #120)")
    tp.add_argument("--file", default=None,
                    help="target shell.json (default: the settings layer's "
                         "default target)")
    tp.add_argument("--propose", action="store_true")
    tp.set_defaults(fn=cmd_topology)

    ca = sub.add_parser("calibration")
    ca.set_defaults(fn=cmd_calibration)

    lg = sub.add_parser("ledger")
    lg.add_argument("action", choices=["list", "approve", "reject"])
    lg.add_argument("id", nargs="?", type=int, default=0)
    lg.set_defaults(fn=cmd_ledger)

    op = sub.add_parser("ope", help="off-policy evaluation gate: replay the "
                                    "approve/reject log through a candidate "
                                    "kill-switch policy (evidence only, "
                                    "flips nothing)")
    op.add_argument("switch", help="kill-switch name (see --list-switches)")
    op.add_argument("--list-switches", action="store_true",
                    help="list the registered candidate policies and exit")
    op.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    op.set_defaults(fn=cmd_ope)

    se = sub.add_parser("settings")
    se.add_argument("action", choices=["propose", "decide", "recommend"])
    # propose: arg1=FILE. decide: arg1=ID, arg2=approve|reject. recommend: unused.
    se.add_argument("arg1", nargs="?", default="")
    se.add_argument("arg2", nargs="?", default="reject")
    se.add_argument("--preset", default=None)
    se.add_argument("--call", action="append", dest="calls", default=None)
    se.add_argument("--reason", default=None)
    se.add_argument("--top", type=int, default=10)
    se.add_argument("--tools", action="store_true",
                    help="recommend: rank individual tools instead of presets")
    se.add_argument("--battery-reward", type=float, default=None, metavar="R",
                    help="decide: optional secondary bandit signal in [0, 1] "
                         "(0.5 neutral) from battery drain-rate deltas around "
                         "the preset's active window; never replaces the "
                         "approve/reject signal")
    se.set_defaults(fn=cmd_settings)

    # ---- shell-native second-brain surfaces (brief / tidy / prefs) ----
    br = sub.add_parser("brief", help="one deterministic page connecting "
                                       "ledger, plans, forecasts")
    br.add_argument("--config-drift", action="store_true",
                    help="opt-in (B2): score the live shell.json against "
                         "the preference posterior learned from the ledger")
    br.add_argument("--target", default=None,
                    help="shell.json to score for --config-drift "
                         "(default: the watched global config)")
    br.set_defaults(fn=cmd_brief)

    td = sub.add_parser("tidy", help="filesystem survey (read-only) + rollback")
    td.add_argument("action", choices=["survey", "rollback"])
    td.add_argument("root", nargs="?", default="~/Downloads")
    td.add_argument("--journal", default=None)
    td.add_argument("--json", action="store_true")
    td.set_defaults(fn=cmd_tidy)

    bs_ = sub.add_parser("bisect", help="what change broke my look? — "
                                        "noisy-answer Bayesian bisect + ddmin "
                                        "over your own good/bad marks "
                                        "(exp-build-5 F13); probes are plans "
                                        "you apply through the settings gate")
    bs_.add_argument("action", choices=["mark", "next", "observe", "status",
                                        "proposal", "reset"])
    bs_.add_argument("label", nargs="?", default="",
                     help="mark: good|bad; observe: good|bad")
    bs_.add_argument("--root", default=merkle.DEFAULT_CONFIG_ROOT,
                     help="caelestia config root to snapshot (default: "
                          "%(default)s)")
    bs_.set_defaults(fn=cmd_bisect)

    pf = sub.add_parser("prefs", help="what the preference model believes "
                                      "about your approve/reject patterns")
    pf.set_defaults(fn=cmd_prefs)

    pt = sub.add_parser("patterns", help="recurring patterns in a series: "
                        "seasonality ('every night?'), PELT changepoints, "
                        "SAX motifs and discords (C11)")
    pt.add_argument("series", help="comma-separated numbers")
    pt.add_argument("--period", type=int, default=24)
    pt.add_argument("--window", type=int, default=6)
    pt.add_argument("--penalty", type=float, default=8.0)
    pt.add_argument("--json", action="store_true")
    pt.set_defaults(fn=cmd_patterns)

    bz = sub.add_parser("bursts", help="Hawkes burst detection over event "
                        "times: crash loops, notification storms; "
                        "correlation, never causation (C12)")
    bz.add_argument("events", help="comma-separated event times")
    bz.add_argument("--factor", type=float, default=2.0,
                    help="burst threshold as a multiple of the fitted "
                         "base rate (default 2.0)")
    bz.add_argument("--json", action="store_true")
    bz.set_defaults(fn=cmd_bursts)

    gp = sub.add_parser("gp-prefs", help="preference learning over "
                        "pairwise 'A or B?' answers with a Gaussian "
                        "process (Cholesky, bounded at 50 items); "
                        "proposals validated against the registry, "
                        "never applied (C13)")
    gp.add_argument("data", help="JSON file: {items: [{id, tool, value, "
                                 "features}], pairs: [{winner, loser}]}")
    gp.add_argument("--json", action="store_true")
    gp.set_defaults(fn=cmd_gpprefs)

    rl = sub.add_parser("rules", help="your event-condition-action rules "
                        "on a Rete network (a rules file is the opt-in); "
                        "every fired action is a proposal, nothing "
                        "applies itself (C14)")
    rl.add_argument("--file", default=str(rete_default_path()),
                    help="rules JSON (default: %(default)s)")
    rl.add_argument("--events", default=None,
                    help="JSON file of events to evaluate (omit to "
                         "just list rules)")
    rl.add_argument("--json", action="store_true")
    rl.set_defaults(fn=cmd_rules)

    sz = sub.add_parser("sizes", help="sizing card: Space-Saving heavy "
                        "hitters + t-digest quantiles over byte counts "
                        "or line lengths (C15)")
    sz.add_argument("sizes", help="comma-separated numbers")
    sz.add_argument("--keys", default=None,
                    help="comma-separated keys (paths, apps) paired "
                         "one-to-one with the sizes")
    sz.add_argument("--k", type=int, default=8,
                    help="how many heavy hitters to keep (default 8)")
    sz.add_argument("--quantiles", default="0.5,0.9,0.95,0.99")
    sz.add_argument("--json", action="store_true")
    sz.set_defaults(fn=cmd_sizes)

    sy = sub.add_parser("say", help="template NLG over a plan: discourse "
                        "plan + aggregation ('2 spacing changes and 1 "
                        "color change in the bar'), deterministic "
                        "(D16)")
    sy.add_argument("file", help="JSON file: a list of changes or "
                                 "{changes: [...]}")
    sy.add_argument("--json", action="store_true")
    sy.set_defaults(fn=cmd_say)

    qam = sub.add_parser("qa", help="extractive QA over the repo docs and "
                         "your notes: verbatim quotes with citations, "
                         "never written answers (D17)")
    qam.add_argument("question", help="the question to answer")
    qam.add_argument("--note", action="append", default=None,
                     help="path to a note file to search too "
                          "(repeatable)")
    qam.add_argument("--sources", default="both",
                     choices=["both", "corpus", "notes"],
                     help="which sources to search (default both when "
                          "notes are given)")
    qam.add_argument("--json", action="store_true")
    qam.set_defaults(fn=cmd_qa)
    return p


def main(argv=None, out=None):
    args = build_parser().parse_args(argv)
    return args.fn(args, out or sys.stdout)
