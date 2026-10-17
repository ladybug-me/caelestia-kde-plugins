"""Personal service layer: the PKM operations, as plain-data functions.

Moved here verbatim from assistant/brain/service.py during the issue-#120
scope split: these functions operate on the user's markdown vault and
personal state only — zero shell linkage. The shell-native service functions
(ledger, settings, rhythm, placement, calibration, drift, dreamtime, forecast,
focus) remain in assistant/brain/service.py.

Nothing here prints; nothing here executes or networks. Every vault change is
a proposal in the same ledger discipline as the shell side.
"""
from datetime import datetime as _dt
from datetime import timezone as _tz

import os

from .. import state as st
from ..ledger import Ledger
from ..bandit import HourBandit
from ..minhash import near_duplicates
from ..naive_bayes import NaiveBayes
from ..nlp import tokens
from ..textmine import keywords, summarize
from ..spellfix import SpellIndex
from .duration import DurationModel
from .ghost import find_ghosts
from .graph import Graph, decay_weights, links
from .health import rank_notes, score as health_score
from . import journal as _journal
from .linkrec import suggest_links
from .planner import cpm, knapsack
from .priority import PriorityModel, features
from .srs import due, new_card, review
from .survival import completion_prob, kaplan_meier
from .topics import extract as topics_extract
from .vault import scan

NONE = "__none__"  # null hypothesis: untagged-note vocabulary, so a tag must beat "no tag"


def _trained_tagger(notes):
    nb = NaiveBayes()
    for note in notes.values():
        nb.train(tokens(note["text"]), note["tags"] or [NONE])
    return nb


def _has_tag_classes(nb):
    return any(c != NONE for c in nb.class_docs)


# ---- vault -----------------------------------------------------------------

def organize(vault, ledger_path, dup=0.8, min_conf=0.25):
    notes = scan(vault)
    ledger = Ledger(ledger_path)
    graph = Graph()
    for rel, note in notes.items():
        graph.add(rel.lower(), links(note["text"]))
    dups = near_duplicates({r: n["text"] for r, n in notes.items()}, threshold=dup)
    for a, b, score in dups:
        ledger.propose("merge_duplicate", f"{a} <> {b}", {"a": a, "b": b},
                       f"near-duplicate (jaccard~{score})", score)
    orphans = graph.orphans()
    for rel in orphans:
        ledger.propose("orphan", rel, {"action": "link_or_archive"},
                       "no in-links and no out-links", 0.6)
    nb = _trained_tagger(notes)
    tag_proposals = 0
    for rel, note in notes.items():
        if note["tags"] or not _has_tag_classes(nb):
            continue
        ranked = [(t, p) for t, p in nb.rank(tokens(note["text"]))
                  if t != NONE and p >= min_conf][:2]
        if ranked:
            tag_proposals += 1
            ledger.propose("tag", rel, {"add_tags": [t for t, _ in ranked]},
                           "naive-bayes tag suggestion", ranked[0][1])
    pr = sorted(graph.pagerank().items(), key=lambda x: -x[1])[:5]
    return {
        "notes": len(notes),
        "duplicate_pairs": len(dups),
        "orphans": len(orphans),
        "tag_proposals": tag_proposals,
        "communities": [c for c in graph.communities() if len(c) > 1],
        "most_connected": [k for k, _ in pr],
        "pending": len(ledger.pending()),
    }


def tag(vault, text):
    """Ranked [(tag, prob)] renormalised over tag classes, or None if untagged."""
    nb = _trained_tagger(scan(vault))
    if not _has_tag_classes(nb):
        return None
    scores = [(t, p) for t, p in nb.rank(tokens(text)) if t != NONE]
    total = sum(p for _, p in scores) or 1.0
    return [(t, p / total) for t, p in scores[:5]]


# ---- plan ------------------------------------------------------------------

def plan(tasks, minutes, state_path, ledger_path=None, propose=False, top=5):
    s = st.load(state_path)
    prio = PriorityModel.from_dict(s.get("priority", {}))
    dur = DurationModel(s.get("duration", {}).get("cats"))
    scored = []
    for t in tasks:
        x = features(t)
        est = dur.estimate(t.get("category", "general"))
        minutes_est = est["median_min"] if est["n"] >= 2 else t.get("effort_min", 30)
        score = prio.score(x)
        scored.append({"id": t["id"], "title": t.get("title", t["id"]), "score": score,
                       "x": x, "minutes": minutes_est, "value": score})
    scored.sort(key=lambda r: -r["score"])
    chosen, value, used = knapsack(
        [{"id": r["id"], "minutes": r["minutes"], "value": r["value"]} for r in scored],
        minutes)
    result = {"ranked": scored, "chosen": chosen, "value": value, "used": used,
              "critical_path": [], "project_minutes": None, "dependency_error": None,
              "proposed": 0}
    deps = {t["id"]: {"minutes": t.get("effort_min", 30), "deps": t.get("deps", [])}
            for t in tasks}
    try:
        sched = cpm(deps)
        result["critical_path"] = sorted((t for t in sched if sched[t]["critical"]),
                                         key=lambda t: sched[t]["es"])
        result["project_minutes"] = max((v["ef"] for v in sched.values()), default=0)
    except ValueError as e:
        result["dependency_error"] = str(e)
    if propose and ledger_path is not None:
        ledger = Ledger(ledger_path)
        for rank, r in enumerate(scored[:top], start=1):
            ledger.propose("priority", r["id"], {"features": r["x"], "rank": rank},
                           f"ranked {rank} of {len(scored)}", r["score"])
        result["proposed"] = min(top, len(scored))
    return result


# ---- duration --------------------------------------------------------------

def estimate_observe(category, minutes, state_path):
    s = st.load(state_path)
    dur = DurationModel(s.get("duration", {}).get("cats"))
    dur.observe(category, float(minutes))
    s["duration"] = {"cats": dur.cats}
    st.save(s, state_path)
    return dur.estimate(category)


def estimate_query(category, state_path):
    s = st.load(state_path)
    return DurationModel(s.get("duration", {}).get("cats")).estimate(category)


# ---- spaced review ---------------------------------------------------------

def review_add(card_id, state_path):
    s = st.load(state_path)
    cards = s.get("srs", {})
    cards.setdefault(card_id, new_card())
    s["srs"] = cards
    st.save(s, state_path)
    return cards[card_id]


def review_grade(card_id, rating, days, state_path):
    s = st.load(state_path)
    cards = s.get("srs", {})
    cards[card_id] = review(cards.get(card_id, new_card()), int(rating), days)
    s["srs"] = cards
    st.save(s, state_path)
    return cards[card_id]


def review_show(card_id, state_path):
    return st.load(state_path).get("srs", {}).get(card_id, {})


def review_due(days, state_path):
    cards = st.load(state_path).get("srs", {})
    return sorted(k for k, c in cards.items() if due(c, days))


# ---- reminders -------------------------------------------------------------

def remind_choose(allowed, state_path):
    bandit = HourBandit.from_dict(st.load(state_path).get("bandit", {}))
    return bandit.choose(allowed)


def remind_feedback(hour, acted, state_path):
    if not 0 <= int(hour) <= 23:
        raise ValueError("hour must be 0..23")
    s = st.load(state_path)
    bandit = HourBandit.from_dict(s.get("bandit", {}))
    bandit.reward(int(hour), bool(acted))
    s["bandit"] = bandit.to_dict()
    st.save(s, state_path)


# ---- cull ------------------------------------------------------------------

def cull(history, horizon=14, threshold=0.05):
    """history: {"durations": [days], "observed": [bool], "open": [{"id", "age"}]}."""
    curve = kaplan_meier(history["durations"], history["observed"])
    flagged = []
    for t in history.get("open", []):
        p = completion_prob(curve, float(t["age"]), horizon=horizon)
        if p < threshold:
            flagged.append({"id": t["id"], "p": round(p, 3)})
    return sorted(flagged, key=lambda x: x["p"])


# ---- ledger ----------------------------------------------------------------

def ledger_learn(ledger_path, state_path):
    examples = [(i["diff"]["features"], 1 if i["status"] == "approved" else 0)
                for i in Ledger(ledger_path).labeled("priority")]
    s = st.load(state_path)
    prio = PriorityModel.from_dict(s.get("priority", {}))
    prio.fit(examples)
    s["priority"] = prio.to_dict()
    st.save(s, state_path)
    return {"examples": len(examples)}


# ---- links & text mining ----------------------------------------------------

def links_suggest(vault, ledger_path=None, propose=False, top=10, min_jaccard=0.15):
    notes = scan(vault)
    graph = Graph()
    for rel, note in notes.items():
        graph.add(rel.lower(), links(note["text"]))
    suggestions = suggest_links(graph, {r: n["text"] for r, n in notes.items()},
                                top=top, min_jaccard=min_jaccard)
    if propose and ledger_path is not None:
        ledger = Ledger(ledger_path)
        for s in suggestions:
            ledger.propose("suggest_link", f"{s['a']} <> {s['b']}",
                           {"a": s["a"], "b": s["b"], "via": s["via"]},
                           f"{s['via']} score={s['score']}", s["score"])
    return suggestions


def note_keywords(vault, note_id, top=8):
    notes = scan(vault)
    if note_id not in notes:
        raise KeyError(f"no such note: {note_id}")
    corpus = [n["text"] for rel, n in notes.items() if rel != note_id]
    return keywords(notes[note_id]["text"], corpus, top=top)


def note_summarize(vault, note_id, sentences=3):
    notes = scan(vault)
    if note_id not in notes:
        raise KeyError(f"no such note: {note_id}")
    return summarize(notes[note_id]["text"], sentences_out=sentences)


def spellcheck(vault, top=20):
    notes = scan(vault)
    return SpellIndex().build(n["text"] for n in notes.values()).suggest(top=top)


def ghosts(vault, known_task_titles, overlap_threshold=0.4):
    notes = scan(vault)
    return find_ghosts({r: n["text"] for r, n in notes.items()}, known_task_titles,
                       overlap_threshold=overlap_threshold)


# ---- decision journal --------------------------------------------------------

def journal_record(decision_id, statement, confidence, state_path):
    s = st.load(state_path)
    entries = s.get("journal", {})
    entry = _journal.record(entries, decision_id, statement, confidence)
    s["journal"] = entries
    st.save(s, state_path)
    return entry


def journal_resolve(decision_id, correct, state_path):
    s = st.load(state_path)
    entries = s.get("journal", {})
    entry = _journal.resolve(entries, decision_id, correct)
    s["journal"] = entries
    st.save(s, state_path)
    return entry


def journal_report(state_path, bins=5):
    entries = st.load(state_path).get("journal", {})
    return {"brier_score": _journal.brier_score(entries),
            "calibration": _journal.calibration_curve(entries, bins=bins)}


# ---- note health ----------------------------------------------------------------

def health_report(vault, dup_threshold=0.8):
    notes = scan(vault)
    graph = Graph()
    for rel, note in notes.items():
        graph.add(rel.lower(), links(note["text"]))
    orphan_ids = set(graph.orphans())
    dup_pairs = near_duplicates({r: n["text"] for r, n in notes.items()}, threshold=dup_threshold)
    dup_ids = {a for a, b, _ in dup_pairs} | {b for a, b, _ in dup_pairs}
    now = _dt.now(_tz.utc).timestamp()
    rows = []
    for rel, note in notes.items():
        try:
            age_days = max((now - os.path.getmtime(note["path"])) / 86400.0, 0.0)
        except OSError:
            age_days = 0.0
        rows.append(health_score(rel, age_days, orphan_ids, dup_ids))
    return rank_notes(rows)


# ---- link-graph views (exponential-build-3 D) -----------------------------------

def graph_report(vault, top=5, now=None, half_life_days=90.0):
    """The link-graph report: all-time PageRank, HITS hubs and
    authorities (Kleinberg 1999, side by side with PageRank), and the
    time-decayed "who matters NOW" PageRank. Read-only; ``now`` is
    injectable for determinism (defaults to the real clock)."""
    notes = scan(vault)
    graph = Graph()
    for rel, note in notes.items():
        graph.add(rel.lower(), links(note["text"]))
    pr = sorted(graph.pagerank().items(), key=lambda x: (-x[1], x[0]))[:top]
    h = graph.hits()
    hubs = sorted(h["hub"].items(), key=lambda x: (-x[1], x[0]))[:top]
    auths = sorted(h["authority"].items(), key=lambda x: (-x[1], x[0]))[:top]
    if now is None:
        now = _dt.now(_tz.utc).timestamp()
    weights = decay_weights(notes, now, half_life_days)
    pr_now = sorted(graph.pagerank(weights=weights).items(),
                    key=lambda x: (-x[1], x[0]))[:top]
    return {
        "notes": len(notes),
        "edges": sum(len(t) for t in graph.out.values()),
        "pagerank": [{"note": k, "score": round(s, 6)} for k, s in pr],
        "hubs": [{"note": k, "score": round(s, 6)} for k, s in hubs],
        "authorities": [{"note": k, "score": round(s, 6)} for k, s in auths],
        "pagerank_recent": [{"note": k, "score": round(s, 6)}
                            for k, s in pr_now],
        "half_life_days": half_life_days,
        "hits_iterations": h["iterations"],
        "hits_converged": h["converged"],
    }


def topics_report(vault, k=3, terms_per_topic=6, max_terms=2000):
    """The NMF topic report over the vault (Lee & Seung 1999). Pure
    read-only analysis; refusals (empty vault, k out of range) surface
    as ValueError from topics.extract with the reason."""
    return topics_extract(scan(vault), k=k,
                          terms_per_topic=terms_per_topic,
                          max_terms=max_terms)
