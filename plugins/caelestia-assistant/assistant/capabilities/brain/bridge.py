"""JSON bridge: one request object in, one response object out.

For a QML Process (or any local caller) to drive the brain without parsing text:

    echo '{"op": "forecast", "series": [3, 4, 5]}' | caelestia-assist api

Response: {"ok": true, "op": ..., "result": ...} or {"ok": false, "op": ..., "error": ...}.
Every op returns plain data; proposals are only ever created, never applied.

Scope note (issue #120 split): the personal-knowledge-management ops
(organize, tag, plan, estimate_*, review_*, remind_*, cull, ledger_learn,
links_suggest, note_keywords, note_summarize, spellcheck, ghosts, journal_*,
health_report) moved to assistant/brain/personal/ and are NOT exposed here —
the bridge is shell-native surface only.
"""
import argparse
import json
import sys

from . import service
from . import state as st
from .cli import DEFAULT_LEDGER


OPS = {
    "forecast": lambda q, s, l: service.forecast(q["series"], horizon=int(q.get("horizon", 7))),
    "focus": lambda q, s, l: service.focus(list(q["labels"])),
    "ledger_list": lambda q, s, l: service.ledger_list(l),
    "ledger_decide": lambda q, s, l: service.ledger_decide(q["id"], q["approve"], l),
    "rhythm_report": lambda q, s, l: service.rhythm_report(q.get("weekdays", []),
                                                           q.get("hours", []),
                                                           threshold=float(q.get("threshold", 1.5))),
    "placement_propose": lambda q, s, l: service.placement_propose(
        q["text"], q["registry"], l, propose=bool(q.get("propose")), top=int(q.get("top", 5))),
    "calibration_report": lambda q, s, l: service.calibration_report(l, bins=int(q.get("bins", 5))),
    "budget_choose": lambda q, s, l: service.budget_choose(s),
    "budget_feedback": lambda q, s, l: service.budget_feedback(q["arm"], q["engaged"], s),
    "drift_check": lambda q, s, l: service.drift_check(
        q["old"], q["new"], l, propose=bool(q.get("propose")),
        similarity_threshold=float(q.get("similarity_threshold", 0.85))),
    "dream_window": lambda q, s, l: service.dream_window(
        q["idle_minutes"], q["on_ac_power"], q["cpu_load_percent"], q["jobs"], q["budget_min"]),
    # ---- issue #120 phase 2 ops (workspace profiles / topology / settings rhythm) ----
    "workspace_profiles": lambda q, s, l: service.workspace_profiles(
        q["records"], l, propose=bool(q.get("propose")),
        k=q.get("k"), min_support=int(q.get("min_support", 3)),
        purity=float(q.get("purity", 0.6))),
    "topology_observe": lambda q, s, l: service.topology_observe(
        q["file"], s, l, propose=bool(q.get("propose"))),
    "settings_rhythm": lambda q, s, l: service.settings_rhythm(
        q["file"], scheme_switches=q.get("scheme_switches"), ledger_path=l),
    "telemetry_snapshot": lambda q, s, l: _telemetry_snapshot(q),
    # ---- cortex ops (learned intelligence layer; routing is read-only) ----
    "route": lambda q, s, l: _cortex_route(q, s),
    "chat_turn": lambda q, s, l: _cortex_chat_turn(q, s),
    "dispatch": lambda q, s, l: _cortex_dispatch(q, s),
    "gap_report": lambda q, s, l: _cortex_gap_report(s),
    "gap_propose": lambda q, s, l: _cortex_gap_propose(s, l),
    "cortex_report": lambda q, s, l: _cortex_report(s),
    "memory_recall": lambda q, s, l: _cortex_memory_recall(q, s),
    "learn_feedback": lambda q, s, l: _cortex_learn_feedback(q, s),
    # ---- genius ops (universal intelligence layer; read-only analysis) ----
    "genius_do": lambda q, s, l: _genius_do(q, s),
    "genius_learn": lambda q, s, l: _genius_learn(q, s),
    "genius_report": lambda q, s, l: _genius_report(s, l),
    "genius_math": lambda q, s, l: _genius_safe(q, "mathengine.expression_info", q["expr"]),
    "genius_stats": lambda q, s, l: _genius_safe(q, "stats.describe", q["numbers"]),
    "genius_logic": lambda q, s, l: _genius_safe(q, "logic.classify_formula", q["formula"]),
    "genius_decide": lambda q, s, l: _genius_safe(
        q, "decision.compare", q["matrix"], q["labels"], q["criteria"],
        q.get("weights"), q.get("benefits"), method=str(q.get("method", "wsm"))),
    "genius_palette": lambda q, s, l: _genius_safe(
        q, "creative.palette", q["hex"], harmony=q.get("harmony", "analogous"),
        n=int(q.get("n", 5))),
    "genius_graphs": lambda q, s, l: _genius_graphs(
        method=str(q.get("method", "dijkstra")), graph=q.get("graph"),
        source=q.get("source"), target=q.get("target"), nodes=q.get("nodes"),
        edges=q.get("edges"), cost=q.get("cost")),
    "genius_units": lambda q, s, l: _genius_safe(q, "units.evaluate",
                                                 q["text"]),
    "genius_fsbrain_summary": lambda q, s, l: _genius_safe(
        q, "fsbrain.staleness_report", str(q["path"]),
        top=int(q.get("top", 10)),
        half_life_days=float(q.get("half_life_days", 30))),
    "genius_plan": lambda q, s, l: _genius_safe(q, "tasks.decompose", q["goal"]),
    "genius_sentiment": lambda q, s, l: _genius_safe(q, "language.sentiment", q["text"]),
    "genius_summarize": lambda q, s, l: _genius_safe(
        q, "language.summarize_focused", q["text"],
        q.get("query", ""), n_sentences=int(q.get("sentences", 3))),
    # ---- round-three ops (agent / scan / optimize / prefs / conformal) ----
    "agent_plan": lambda q, s, l: _agent_plan(q),
    "agent_simulate": lambda q, s, l: _agent_simulate(q),
    "scan_text": lambda q, s, l: _scan_text(q),
    "brief": lambda q, s, l: _brief(l),
    "tidy_survey": lambda q, s, l: _tidy_survey(q),
    "optimize_recommend": lambda q, s, l: _optimize_recommend(
        profile=str(q.get("profile", "gaming")), k=int(q.get("k", 6))),
    "optimize_score": lambda q, s, l: _optimize_score(q),
    "prefs_report": lambda q, s, l: _prefs_report(l),
    "conformal_verdict": lambda q, s, l: _conformal_verdict(q, s),
    # eval arena over the bridge (read-only, gated on the eval_arena
    # capability): run one suite and return its report dict
    "eval_suite": lambda q, s, l: _eval_suite(q),
    "confusables_report": lambda q, s, l: _confusables_report(),
    "label_fusion_report": lambda q, s, l: _label_fusion_report(),
    "abers_interval": lambda q, s, l: _abers_interval(q),
    # ---- shellkb ops (the shell employee; read-only, nothing executes) ----
    "shell_grammar": lambda q, s, l: _shell_grammar(q),
    "shell_explain": lambda q, s, l: _shell_explain(q),
    "shell_howto": lambda q, s, l: _shell_howto(q),
    "shell_json_diff": lambda q, s, l: _shell_json(q, merge=False),
    "shell_json_merge": lambda q, s, l: _shell_json(q, merge=True),
    "shell_conflicts": lambda q, s, l: _shell_conflicts(q),
    "patterns_report": lambda q, s, l: _patterns_report(q),
    "bursts_report": lambda q, s, l: _bursts_report(q),
    "gp_preferences": lambda q, s, l: _gp_preferences(q),
    "rules_eval": lambda q, s, l: _rules_eval(q),
    "sizing_report": lambda q, s, l: _sizing_report(q),
    "nlg_summary": lambda q, s, l: _nlg_summary(q),
    "qa_answer": lambda q, s, l: _qa_answer(q),
    "model_rerank": lambda q, s, l: _model_rerank(q),
}


def _abers_interval(q):
    """Venn-Abers interval + isotonic point for one score, over the
    caller-provided calibration rows (read-only pure math)."""
    rows = [(float(r["p"]), int(r["ok"])) for r in q.get("rows", [])]
    score = float(q["score"])
    if len(rows) < 5:
        return {"error": "need >= 5 calibration rows", "n": len(rows)}
    from assistant.core.abers import isotonic_fit, isotonic_predict,         venn_abers
    lo, hi, mid = venn_abers(rows, score)
    point = isotonic_predict(isotonic_fit(rows), score)
    return {"score": score, "va_low": round(lo, 4),
            "va_high": round(hi, 4), "va_mid": round(mid, 4),
            "isotonic": round(point, 4)}



def _label_fusion_report():
    from assistant.core.features import enabled
    if not enabled("label_fusion"):
        return {"error": "label_fusion capability is off on this install"}
    from assistant.core.label_fusion import load_model, render_lines
    model = load_model()
    return {"model": model, "lines": render_lines()}



def _confusables_report():
    from assistant.core.features import enabled
    if not enabled("confusable_clarifier"):
        return {"error": "confusable_clarifier capability is off on this "
                         "install"}
    from assistant.core.confusables import load_pairs
    pairs = load_pairs()
    return {"n_pairs": len(pairs),
            "pairs": [{"a": p["a"], "b": p["b"], "question": p["question"]}
                      for p in pairs]}



def _eval_suite(q):
    from assistant.core.features import enabled
    if not enabled("eval_arena"):
        return {"error": "eval_arena capability is off on this install"}
    from assistant.core.eval.engine import run_suite
    suite = str(q.get("suite", "routing"))
    split = str(q.get("split", "dev"))
    if q.get("sealed"):
        return {"error": "the sealed split is a stage-boundary act, not a "
                         "bridge call"}
    return run_suite(suite, split=split)


def _shell_grammar(q):
    """The committed shell CLI grammar (B6). The sidebar can answer
    'what options does caelestia screenshot take?' without spawning a
    process — the knowledge is a committed artifact, read-only."""
    from assistant.core.features import enabled
    if not enabled("shell_cli_grammar"):
        return {"error": "shell_cli_grammar capability is off on this "
                         "install"}
    from assistant.capabilities.shellkb import cligrammar
    data = cligrammar.load_grammar()
    bins = data.get("bins", {})
    wanted = q.get("bin")
    if wanted:
        if wanted not in bins:
            return {"error": f"unknown bin {wanted!r}",
                    "known": sorted(bins)}
        return {"bin": wanted, "grammar": bins[wanted]}
    return {"upstream": data.get("upstream"),
            "bins": sorted(bins),
            "examples": cligrammar.example_lines(bins)}


def _shell_explain(q):
    """Token-by-token explanation of one command line (B7). Read-only:
    the only filesystem touch is a capped os.scandir glob preview; the
    verdict is always SUGGESTED_NOT_EXECUTED."""
    from assistant.core.features import enabled
    if not enabled("shell_explain"):
        return {"error": "shell_explain capability is off on this install"}
    from assistant.capabilities.shellkb import cmdparse
    line = str(q.get("line", "")).strip()
    if not line:
        return {"error": "need 'line': the command line to explain"}
    return cmdparse.explain_line(line)


def _shell_howto(q):
    """Offline how-to over the self-authored CC0 corpus (B8). The only
    temporal knowledge is the existing nlhistory window grammar; every
    returned step is labeled SUGGESTED_NOT_EXECUTED."""
    from assistant.core.features import enabled
    if not enabled("shell_howto"):
        return {"error": "shell_howto capability is off on this install"}
    from assistant.capabilities.shellkb import howto
    query = str(q.get("query", "")).strip()
    if not query:
        return {"error": "need 'query': the how-to question"}
    return howto.answer(query, k=int(q.get("k", 5)))


def _shell_json(q, merge: bool):
    """Structured diff / three-way merge over caller-supplied JSON
    documents (B9). Pure math: the docs arrive in the request, the
    proposal leaves in the response, no file is touched."""
    from assistant.core.features import enabled
    if not enabled("shell_json_merge"):
        return {"error": "shell_json_merge capability is off on this "
                         "install"}
    from assistant.capabilities.shellkb import jsonmerge
    if merge:
        for key in ("base", "ours", "theirs"):
            if key not in q:
                return {"error": f"need {key!r}: the {key} document"}
        return jsonmerge.merge_three_way(q["base"], q["ours"], q["theirs"])
    for key in ("a", "b"):
        if key not in q:
            return {"error": f"need {key!r}: the {key} document"}
    return jsonmerge.diff_docs(q["a"], q["b"])


def _shell_conflicts(q):
    """PubGrub-style dependency explanation (B10). The universe comes
    from the request or the committed illustrative fixture; installed
    facts may come from the READ-ONLY pkgprobe when package_audit is
    on — never from executing anything here."""
    from assistant.core.features import enabled
    if not enabled("shell_conflicts"):
        return {"error": "shell_conflicts capability is off on this "
                         "install"}
    from assistant.capabilities.shellkb import pubgrub
    universe = q.get("universe")
    if universe is None:
        universe = json.loads(pubgrub.UNIVERSE_PATH.read_text())
    root = str(q.get("root", "")).strip()
    if not root:
        return {"error": "need 'root': the package to explain from"}
    installed = q.get("installed")
    if q.get("use_probe"):
        from assistant.capabilities.agent.pkgprobe import query_installed
        probe = query_installed()
        if probe.get("packages"):
            installed = {name: version
                         for name, version in probe["packages"]}
    if installed is not None:
        return pubgrub.check_installed(universe, installed, root,
                                       str(q.get("spec", "*")))
    return pubgrub.solve(universe, root, str(q.get("spec", "*")))


def _patterns_report(q):
    """Seasonality / changepoints / motifs / discords over a series
    (C11). Pure math on the numbers in the request — no files, no
    state."""
    from assistant.core.features import enabled
    if not enabled("seasonal_patterns"):
        return {"error": "seasonal_patterns capability is off on this "
                         "install"}
    from assistant.capabilities.brain import seasonal
    series = q.get("series")
    if not isinstance(series, list) or len(series) < 4:
        return {"error": "need 'series': at least 4 numbers"}
    return seasonal.seasonality_report(
        [float(x) for x in series],
        period=int(q.get("period", 24)),
        window=int(q.get("window", 6)),
        penalty=float(q.get("penalty", 8.0)))


def _bursts_report(q):
    """Hawkes burst detection over event times (C12) — the fit, the
    bursts, and the correlation-not-causation caveat."""
    from assistant.core.features import enabled
    if not enabled("burst_detection"):
        return {"error": "burst_detection capability is off on this "
                         "install"}
    from assistant.capabilities.brain import bursts
    events = q.get("events")
    if not isinstance(events, list) or len(events) < 8:
        return {"error": "need 'events': at least 8 event times"}
    return bursts.burst_report([float(x) for x in events],
                               factor=float(q.get("factor", 2.0)))


def _gp_preferences(q):
    """GP preference learning over pairwise answers (C13). Returns the
    ranking, the next most-informative question, and ONE validated
    setter proposal (registry-spec-checked; a proposal, never an
    apply)."""
    from assistant.core.features import enabled
    if not enabled("gp_preferences"):
        return {"error": "gp_preferences capability is off on this "
                         "install"}
    from assistant.capabilities.brain import gp_prefs
    items = q.get("items") or []
    pairs = q.get("pairs") or []
    if not isinstance(items, list) or not items:
        return {"error": "need 'items': candidate settings with "
                         "features"}
    try:
        model = gp_prefs.GPPreferenceModel(items)
        for pair in pairs:
            model.record(str(pair.get("winner")),
                         str(pair.get("loser")))
        fit = model.fit()
    except ValueError as exc:
        return {"error": str(exc)}
    by_id = {str(it["id"]): it for it in items}
    proposal = None
    for item_id in fit["ranking"]:
        item = by_id.get(item_id, {})
        if not item.get("tool"):
            continue
        check = gp_prefs.validate_item(item)
        if check["validated"]:
            proposal = {"tool": item["tool"], "value": item.get("value"),
                        "item": item_id,
                        "verdict": "SUGGESTED_NOT_EXECUTED"}
            break
    return {**fit, "next_question": model.next_question(),
            "proposal": proposal}


def _rules_eval(q):
    """Run user-authored rules over events through the Rete network
    (C14). The events and rules come IN the request; what leaves is a
    list of SUGGESTED_NOT_EXECUTED proposals for the ledger."""
    from assistant.core.features import enabled
    if not enabled("user_rules"):
        return {"error": "user_rules capability is off on this install"}
    from assistant.capabilities.brain import rete
    events = q.get("events") or []
    rules = q.get("rules")
    if rules is None:
        rules = rete.load_rules(rete.DEFAULT_RULES_PATH)
    return rete.run_events(rules, events)


def _sizing_report(q):
    """Space-Saving heavy hitters + t-digest quantiles over
    caller-supplied sizes (C15). Pure math; the disk is never read."""
    from assistant.core.features import enabled
    if not enabled("size_sketches"):
        return {"error": "size_sketches capability is off on this "
                         "install"}
    from assistant.capabilities.brain import sketch
    sizes = q.get("sizes")
    if not isinstance(sizes, list) or not sizes:
        return {"error": "need 'sizes': the values to size"}
    return sketch.sizing_report(
        [float(x) for x in sizes],
        keys=q.get("keys"),
        k=int(q.get("k", 8)),
        quantiles=q.get("quantiles"))


def _nlg_summary(q):
    """Template NLG over a plan's changes (D16): discourse plan,
    aggregation, deterministic sentences."""
    from assistant.core.features import enabled
    if not enabled("nlg_templates"):
        return {"error": "nlg_templates capability is off on this "
                         "install"}
    from assistant.capabilities.brain import nlg
    changes = q.get("changes")
    if not isinstance(changes, list):
        return {"error": "need 'changes': the plan's change dicts"}
    return {"lines": nlg.render_changes_summary(changes),
            "paragraph": nlg.render_plan_paragraph(changes)}


def _qa_answer(q):
    """Extractive QA over repo docs + caller-supplied notes (D17).
    The answer is quoted, never written: spans + provenance + verdict."""
    from assistant.core.features import enabled
    if not enabled("extractive_qa"):
        return {"error": "extractive_qa capability is off on this "
                         "install"}
    from assistant.capabilities.brain import qa
    question = str(q.get("question") or q.get("text") or "")
    if not question.strip():
        return {"error": "need 'question': the question to answer "
                         "extractively"}
    notes = q.get("notes") or []
    if not isinstance(notes, list):
        return {"error": "'notes' must be a list of {name, text} records"}
    return qa.answer(question, notes=notes,
                     sources=str(q.get("sources", "both")))


def _model_rerank(q):
    """Optional sub-1B reranker (E2): one loopback generate() over the
    router's OWN candidates; MODEL_SUGGESTED or UNCHANGED, never a
    plan. The caller routes any resulting choice through the ordinary
    plan/apply gates — this op cannot apply anything."""
    from assistant.core.features import enabled
    if not enabled("model_reranker"):
        return {"error": "model_reranker capability is off on this "
                         "install (edit the capabilities file, never a "
                         "request)"}
    from assistant.capabilities.generative import reranker
    query = str(q.get("query") or q.get("text") or "")
    candidates = q.get("candidates")
    if not query.strip() or not isinstance(candidates, list) or \
            not candidates:
        return {"error": "need 'query' and a non-empty 'candidates' "
                         "list of {name, description} (the router's own "
                         "shortlist)"}
    return reranker.rerank(query, candidates)



def _telemetry_snapshot(q):
    """One on-demand /proc + /sys read (issue #120 Phase 3.1). Not a
    daemon: every call is exactly one snapshot the caller asked for."""
    from assistant.capabilities.diagnostics import telemetry
    return telemetry.snapshot(proc_dir=str(q.get("proc_dir", "/proc")),
                              sys_dir=str(q.get("sys_dir", "/sys")))


def _cortex_state(state_path):
    return st.load(state_path) if state_path else {}


def _agent_plan(q):
    from assistant.capabilities.agent import goals
    return goals.decompose(str(q.get("text", "")))


def _agent_simulate(q):
    from assistant.capabilities.agent.engine import Agent
    return Agent().simulate(str(q.get("text", "")))


def _scan_text(q):
    from assistant.capabilities.scan import Automaton, scan_text
    patterns = [str(p) for p in q.get("patterns", [
        "quickshell", "kwin", "dbus", "error", "critical", "segfault",
        "failed to", "timeout"])]
    return scan_text(str(q.get("text", "")), Automaton(patterns),
                     chunk_size=int(q.get("chunk", 500)))


def _brief(ledger_path):
    from . import brief as brief_mod
    from .ledger import Ledger
    b = brief_mod.compose(pending=Ledger(ledger_path).pending())
    return {"brief": b, "rendered": brief_mod.render(b)}


def _tidy_survey(q):
    from . import tidy as tidy_mod
    plan = tidy_mod.survey(str(q.get("root", "~/Downloads")))
    return {"rendered": tidy_mod.render_plan(plan), **{
        k: plan[k] for k in ("root", "scanned", "type_moves", "duplicates",
                             "stale", "big_files", "empty_dirs",
                             "space_recoverable", "inert_suggestions")}}


def _optimize_recommend(profile, k):
    from assistant.capabilities.settings import optimize as opt
    return opt.recommend(str(profile), k=int(k))


def _optimize_score(q):
    from assistant.capabilities.settings import optimize as opt
    return opt.score_plan(str(q.get("profile", "gaming")),
                          list(q.get("ops", [])))


def _prefs_report(ledger_path):
    from . import prefs as prefs_mod
    from .ledger import Ledger
    model = prefs_mod.PreferenceModel()
    n = model.from_ledger(Ledger(ledger_path).items)
    biases = [model.bias(*key.split("|")[:2], int(key.split("|")[2]))
              for key in sorted(model.table.keys())[:12]]
    return {"decisions_learned": n, "biases": biases}


def _conformal_verdict(q, state_path):
    from assistant.core.conformal import ConformalCalibrator
    state = _cortex_state(state_path)
    cal = ConformalCalibrator()
    data = state.get("conformal")
    if data:
        cal.from_dict(data)
    for obs in q.get("calibration", []):
        cal.observe(float(obs.get("score", 0.0)), str(obs.get("outcome", "rejected")))
    return cal.verdict(float(q.get("score", 0.0)),
                       alpha=float(q.get("alpha", 0.1))) if cal.scores else \
        {"covered": None, "reason": "no calibration data in state", "score": q.get("score")}


def _genius_safe(q, dotted, *args, **kwargs):
    """Call a genius capability by dotted path, catching its honest errors.

    Only known (module, function) pairs on an explicit allow-list can ever
    run — the bridge cannot be pointed at arbitrary code.
    """
    from assistant.capabilities.genius import (creative, decision, language, logic, mathengine,  # noqa: F401
                          stats, tasks, units, fsbrain)
    allowed = {
        "mathengine.expression_info": mathengine.expression_info,
        "stats.describe": stats.describe,
        "logic.classify_formula": logic.classify_formula,
        "decision.compare": decision.compare,
        "creative.palette": creative.palette,
        "tasks.decompose": tasks.decompose,
        "language.sentiment": language.sentiment,
        "language.summarize_focused": language.summarize_focused,
        "units.evaluate": units.evaluate,
        "fsbrain.staleness_report": fsbrain.staleness_report,
    }
    fn = allowed.get(dotted)
    if fn is None:
        return {"error": f"unknown genius op {dotted!r}"}
    try:
        return fn(*args, **kwargs)
    except (ValueError, KeyError, TypeError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _genius_graphs(method: str, graph, source, target, nodes, edges,
                   cost):
    """Sidebar graphs op (exponential-build 5.1): classical graph
    algorithms over JSON-serialisable inputs. A* is deliberately NOT
    exposed here — its heuristic is a function, not data, and the
    bridge cannot receive code (the CLI keeps A*). Args are pinned in
    the QML parity test."""
    from assistant.capabilities.genius import graphs
    if method == "dijkstra":
        if not graph or source is None:
            raise ValueError("dijkstra needs a graph object and a source")
        return graphs.dijkstra(dict(graph), str(source), target=target)
    if method == "mst":
        if not nodes or not edges:
            raise ValueError("mst needs nodes (list) and edges ([u, v, w])")
        return graphs.min_spanning_tree(
            [str(n) for n in nodes],
            [(str(a), str(b), float(w)) for a, b, w in edges])
    if method == "assign":
        if not cost:
            raise ValueError("assign needs a cost matrix (rows x columns)")
        return graphs.hungarian([[float(w) for w in row] for row in cost])
    return {"error": f"unknown graphs method {method!r} "
                     "(dijkstra|mst|assign)"}


def _genius_do(q, state_path):
    from assistant.capabilities.genius import meta as genius_meta

    state = _cortex_state(state_path)
    return genius_meta.route_and_do(q["text"], state.get("genius_learn"))


def _genius_learn(q, state_path):
    from assistant.capabilities.genius import meta as genius_meta

    state = _cortex_state(state_path)
    table = genius_meta.learn_feedback(q["text"], q["domain"],
                                        bool(q["accepted"]),
                                        state.get("genius_learn"))
    state["genius_learn"] = table
    st.save(state, state_path)
    return {"updated": True, "domains_trained": sorted(table)}


def _genius_report(state_path, ledger_path):
    import json as _json
    from pathlib import Path as _Path

    from assistant.capabilities.genius import metacog

    state = _cortex_state(state_path)
    ledger_file = _Path(ledger_path)
    proposals = (_json.loads(ledger_file.read_text()).get("proposals", [])
                 if ledger_file.exists() else [])
    usage = [{"domain": p.get("kind", "unknown"),
              "accepted": p.get("status") == "approved", "day": 1}
             for p in proposals if p.get("status") in ("approved", "rejected")]
    history = [{"features": {"kind": p.get("kind", "?"),
                             "confidence": "high" if p.get("confidence", 0) >= 0.7 else "low"},
                "outcome": "approve" if p.get("status") == "approved" else "reject"}
               for p in proposals if p.get("status") in ("approved", "rejected")]
    return {"coverage": metacog.coverage_map(usage),
            "learned_rules": metacog.induce_rules(history),
            "learned_routing": state.get("genius_learn")}


def _cortex_route(q, state_path):
    from assistant.core.learn import CortexLearner
    from assistant.core.pipeline import process as cortex_process

    learner = CortexLearner(_cortex_state(state_path).get("cortex_learn"))
    result = cortex_process(q["text"], learner=learner,
                            file_path=q.get("file"),
                            router_state=None)
    return result.to_dict()


def _cortex_chat_turn(q, state_path):
    """One conversational turn over the JSON bridge: pass the session
    dict from the previous response back in (``session`` key), get the
    updated session plus this turn's result. Read-only."""
    from assistant.core.learn import CortexLearner
    from assistant.core.pipeline import process as cortex_process
    from assistant.core.session import SessionState

    session = SessionState.from_dict(q.get("session"))
    learner = CortexLearner(_cortex_state(state_path).get("cortex_learn"))
    result = cortex_process(q["text"], session=session, learner=learner,
                            file_path=q.get("file"))
    return {"result": result.to_dict(), "session": session.to_dict()}


def _cortex_dispatch(q, state_path):
    """The unified local-vs-cloud decision point (the sidebar's ONLY
    routing authority): run the local cortex pipeline, and either answer
    locally or hand off to the cloud tier with a reason. A hand-off is
    logged into the bounded ``cortex_gaps`` state bucket (query shape +
    intent category, never raw text) before the state is saved — the
    single write this op performs, alongside the review bucket it shares
    with the CLI chat loop."""
    from assistant.core.dispatch import dispatch
    from assistant.core.session import SessionState

    state = _cortex_state(state_path)
    outcome = dispatch(q["text"], state=state,
                       session=SessionState.from_dict(q.get("session")),
                       file_path=q.get("file"))
    st.save(state, state_path)
    return outcome


def _cortex_gap_report(state_path):
    """Read-only clustering summary of the logged local-ontology gaps
    (minimum-support + purity floor applied — see cortex.dispatch)."""
    from assistant.core.dispatch import cluster_gaps

    return cluster_gaps(_cortex_state(state_path))


def _cortex_gap_propose(state_path, ledger_path):
    """Turn qualifying gap clusters into pending LEDGER PROPOSALS (kind
    ``ontology_gap"). Proposes, never applies — the ledger flow decides."""
    from assistant.core.dispatch import propose_gap_clusters
    from .ledger import Ledger

    state = _cortex_state(state_path)
    return propose_gap_clusters(state, Ledger(ledger_path))


def _cortex_report(state_path):
    from assistant.core.learn import CortexLearner
    from assistant.core.memory import summarize

    state = _cortex_state(state_path)
    learner = CortexLearner(state.get("cortex_learn"))
    episodes = list(state.get("cortex_memory") or [])
    return {"learning": learner.report(), "memory": summarize(episodes)}


def _cortex_memory_recall(q, state_path):
    from assistant.core.memory import recall

    episodes = list(_cortex_state(state_path).get("cortex_memory") or [])
    return recall(episodes, q.get("query", ""), k=int(q.get("k", 10)))


def _cortex_learn_feedback(q, state_path):
    """Report an outcome for a learn_hook the caller received: the
    learning loop's only entry point from the shell side. The same (p,
    outcome) pair also feeds the conformal calibrator's history (state key
    ``conformal``), so its distribution-free verdicts accumulate real
    data instead of living only inside one bridge call."""
    from assistant.core.conformal import ConformalCalibrator
    from assistant.core.learn import CortexLearner

    state = _cortex_state(state_path)
    learner = CortexLearner(state.get("cortex_learn"))
    learner.observe(q["text"], q.get("surface", ""), q.get("features", {}),
                    float(q.get("p", 0.0)), q["outcome"])
    if q.get("strategy"):
        learner.reward_strategy(str(q["strategy"]), q["outcome"] in ("applied", "approved"))
    state["cortex_learn"] = learner.to_dict()
    cal = ConformalCalibrator()
    data = state.get("conformal")
    if isinstance(data, dict):
        cal.from_dict(data)
    cal.observe(float(q.get("p", 0.0)), str(q["outcome"]))
    state["conformal"] = cal.to_dict()
    st.save(state, state_path)
    return {"examples": learner.model.examples,
            "acceptance_rate": round(learner.accepts / max(1, learner.accepts + learner.rejects), 3)}


def handle(request, state_path=str(st.DEFAULT_STATE), ledger_path=str(DEFAULT_LEDGER)):
    if not isinstance(request, dict):
        return {"ok": False, "op": None, "error": "request must be a JSON object"}
    op = request.get("op")
    if op not in OPS:
        return {"ok": False, "op": op,
                "error": f"unknown op; expected one of: {', '.join(sorted(OPS))}"}
    try:
        result = OPS[op](request, state_path, ledger_path)
    except (KeyError, TypeError, ValueError) as e:
        return {"ok": False, "op": op, "error": f"{type(e).__name__}: {e}"}
    except OSError as e:
        return {"ok": False, "op": op, "error": f"io error: {e}"}
    return {"ok": True, "op": op, "result": result}


def main(argv=None, stdin_text=None, out=None):
    p = argparse.ArgumentParser(prog="caelestia-assist api", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state", default=str(st.DEFAULT_STATE))
    p.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    args = p.parse_args(argv)
    text = sys.stdin.read() if stdin_text is None else stdin_text
    try:
        request = json.loads(text)
    except json.JSONDecodeError as e:
        response = {"ok": False, "op": None, "error": f"invalid JSON: {e}"}
    else:
        response = handle(request, args.state, args.ledger)
    (out or sys.stdout).write(json.dumps(response, ensure_ascii=False, sort_keys=True) + "\n")
    return 0 if response["ok"] else 1
