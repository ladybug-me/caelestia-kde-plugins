"""cortex.lexicon_diff — federated, opt-in, signed lexicon-diff sharing
(phase 2.6; the former proposals/2026-09-26-c-lexicon-diff-
sharing.md, pruned in exponential-build-3 Phase 2 once
implemented).

The shareable artifact is a plain-text, reviewable list of
``phrase -> tool`` mappings a user's cortex has learned (reroute
corrections + approved routes), capped at the newest 200, PII-stripped
to the undo log's standard: no timestamps, no file paths, no values —
phrases are the only content, and phrases are what the user chose to
type at the assistant.

Federation means reviewable text over the channels the community
already uses (issue threads, matrix, fork PRs) — NO server, NO
network code, NO crypto code in the assistant (the import allow-list
is untouched). "Signed" happens OUTSIDE: an Ed25519 detached signature
over the canonical diff text produced and verified by a tool the user
already trusts (minisign / sq / GPG); the assistant only states whose
corpus a diff claims to be — trust in that identity stays a human
decision.

The module is PURE (no I/O): the caller loads/saves the brain state.
State shape (new key, forward-compatible):

    state["lexicon_imports"] = {
        "<diff-id>": {
            "rows": [{"text", "surface", "n", "p", "label"}],
            "at": "2026-09-26T..."   # import time, local metadata only
        }, ...
    }

The diff-id is sha256[:12] of the canonical row text — `forget` drops
the set by id, restoring the pre-import pair list (and therefore the
corpus-only embedder build on the next process).

Import safety (the proposal's own mitigations, all pinned by test):
- import caps (200 rows);
- per-phrase length cap (120 chars, the learner's own bound);
- rows naming NON-EXISTENT tools import as no-ops with warnings,
  never exceptions (injection fuzz);
- the import report lists every tool the diff would boost;
- imports land as SUPERVISED PAIRS in the embedder seam (A2's
  ``labeled_pairs`` — never into the hand-seeded SYNONYMS, which stay
  a reviewed diff) and as review-bucket candidates for the learner's
  batch-review flow.

Embedder wiring (the one runtime consequence): PpmiEmbedder's shared
singleton (vectorize.embedder) reads the persisted pairs at its lazy
first build — an ABSENT import keeps the corpus-only build byte-for-
byte (the A2-measured default), which is why the fingerprint test in
test_svd_embedder pins the no-pairs build.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "DIFF_HEADER", "MAX_PAIRS", "PHRASE_MAX", "IMPORTS_KEY", "TRUST_KEY",
    "MAX_TRUST_EVENTS", "export_rows", "render", "parse", "diff_id",
    "import_diff", "forget", "persisted_pairs", "imported_ids",
    "signer_trust", "render_advisory",
]

DIFF_HEADER = "CAELESTIA LEXICON DIFF v1"
MAX_PAIRS = 200
PHRASE_MAX = 120
IMPORTS_KEY = "lexicon_imports"
TRUST_KEY = "lexicon_trust"
MAX_TRUST_EVENTS = 500

_ROW_RE = re.compile(
    r"^\s*(?P<sign>[+-])(?P<text>.*?\S)\s*->\s*(?P<surface>\S+)"
    r"(?:\s*\(n=(?P<n>\d+),\s*p=(?P<p>[0-9.]+)\))?"
    # optional dp provenance marker (cortex/dp.py, exponential-build 3
    # B4): additive — an absent marker leaves the match unchanged, so
    # the exact export format is byte-identical to before
    r"(?:\s*\(dp:\s*epsilon=(?P<dp>[0-9.]+)\))?\s*$")
_DATE_RE = re.compile(r"^\((.+)\)\s*$")


# ---------------------------------------------------------------------------
# Export (the learner's examples -> reviewable rows).
# ---------------------------------------------------------------------------


def export_rows(state: Dict[str, Any], learn_key: str = "cortex_learn",
                max_pairs: int = MAX_PAIRS) -> List[Dict[str, Any]]:
    """Derive the shareable rows from the learner's bounded example log:
    group by (text, surface); n = occurrences; p = the mean ROUTE
    probability the user saw; the sign is the majority outcome
    (+ approved/applied, - corrected/rejected). Newest ``max_pairs``
    groups by last-seen position. PII-safe: only text/surface/n/p."""
    examples = []
    learn = state.get(learn_key)
    if isinstance(learn, dict):
        raw = learn.get("examples") or []
        examples = [dict(r) for r in raw if isinstance(r, dict)]
    order: Dict[Tuple[str, str], int] = {}
    agg: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for pos, row in enumerate(examples):
        text = str(row.get("text", ""))[:PHRASE_MAX].strip()
        surface = str(row.get("surface", "")).strip()
        if not text or not surface:
            continue
        key = (text, surface)
        if key not in agg:
            agg[key] = {"text": text, "surface": surface,
                        "n": 0, "p_sum": 0.0, "plus": 0, "minus": 0}
            order[key] = pos
        a = agg[key]
        a["n"] += 1
        try:
            a["p_sum"] += float(row.get("p", 0.0))
        except (TypeError, ValueError):
            pass
        if int(row.get("label", 0)) == 1:
            a["plus"] += 1
        else:
            a["minus"] += 1
    # newest groups last-seen first, cap, then deterministic order by text
    newest = sorted(order, key=lambda k: -order[k])[:max_pairs]
    rows = []
    for key in sorted(newest):
        a = agg[key]
        rows.append({
            "text": a["text"],
            "surface": a["surface"],
            "n": a["n"],
            "p": round(a["p_sum"] / a["n"], 2) if a["n"] else 0.0,
            "label": 1 if a["plus"] >= a["minus"] else 0,
        })
    return rows


def render(rows: List[Dict[str, Any]], date: str = "") -> str:
    """The canonical diff text — deterministic for the same rows (and
    the same date string; the CLI stamps today, tests pin a fixed one).
    Sorted by text; one row per line; nothing else in the artifact."""
    lines = [f"{DIFF_HEADER}  ({date})" if date else DIFF_HEADER]
    for row in sorted(rows, key=lambda r: (r.get("text", ""),
                                           r.get("surface", ""))):
        sign = "+" if int(row.get("label", 1)) == 1 else "-"
        n = int(row.get("n", 1))
        p = float(row.get("p", 0.0))
        lines.append(f"{sign}{row.get('text', '')} -> "
                     f"{row.get('surface', '')}  (n={n}, p={p:.2f})")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Parse + validate (never raises; malformed input is a warning).
# ---------------------------------------------------------------------------


def parse(text: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Parse a diff back into rows. Returns (rows, warnings). Malformed
    lines, over-long phrases, unknown tools, and over-cap rows become
    WARNINGS, never exceptions (the injection-fuzz contract)."""
    warnings: List[str] = []
    rows: List[Dict[str, Any]] = []
    known_tools: Optional[Dict[str, bool]] = None

    def _tool_exists(name: str) -> bool:
        nonlocal known_tools
        if known_tools is None:
            from assistant.adapters.caelestia.registry import TOOL_SPECS
            known_tools = {spec.name: True for spec in TOOL_SPECS}
        return name in known_tools

    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped == DIFF_HEADER \
                or stripped.startswith(DIFF_HEADER) \
                or stripped.startswith("#"):
            continue  # header (with or without date) / blank / comment:
            # no-ops
        m = _ROW_RE.match(stripped)
        if not m:
            warnings.append(f"line {lineno}: unparseable, skipped: "
                            f"{stripped[:60]!r}")
            continue
        phrase = m.group("text").strip()
        surface = m.group("surface").strip()
        if len(phrase) > PHRASE_MAX:
            warnings.append(f"line {lineno}: phrase over {PHRASE_MAX} chars, "
                            "skipped (the learner's own bound)")
            continue
        if not _tool_exists(surface):
            warnings.append(f"line {lineno}: {surface!r} is not a registry "
                            "tool, skipped (unknown tools never import)")
            continue
        if any(r["text"] == phrase and r["surface"] == surface for r in rows):
            warnings.append(f"line {lineno}: duplicate row, skipped")
            continue
        rows.append({
            "text": phrase,
            "surface": surface,
            "n": max(1, int(m.group("n") or 1)),
            "p": min(1.0, max(0.0, float(m.group("p") or 0.0))),
            "label": 1 if m.group("sign") == "+" else 0,
            # the dp noising provenance (cortex/dp.py), when present:
            # rides along as row metadata so a reviewer of an IMPORTED
            # state can still see which rows carry noised evidence
            "dp_epsilon": float(m.group("dp")) if m.group("dp") else None,
        })
        if len(rows) > MAX_PAIRS:
            warnings.append(f"line {lineno}: over the {MAX_PAIRS}-row import "
                            "cap, skipped")
            rows.pop()
    return rows, warnings


def diff_id(rows: List[Dict[str, Any]]) -> str:
    """Stable id for one imported set: sha256[:12] of the canonical rows
    (id → same set; any row change → different id)."""
    canonical = render(sorted(rows, key=lambda r: (r.get("text", ""),
                                                   r.get("surface", ""))),
                       date="canonical")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Import / forget (state-transforming, still pure — caller persists).
# ---------------------------------------------------------------------------


def import_diff(state: Dict[str, Any], text: str,
                signer: Optional[str] = None) -> Dict[str, Any]:
    """Apply one diff into the state: parse + validate, persist the rows
    under their diff-id, and REPORT (boosted tools, parity measurement,
    rollback command). Never touches SYNONYMS; never raises on
    malformed input.

    ``signer`` (optional, phase 2.3): the identity the user verified with
    THEIR external tool (minisign/sq/gpg) — recorded as local metadata
    so a rollback history can exist, and feeding the ADVISORY
    EigenTrust-style trust score in the report. It changes nothing about
    the import's safety behavior: review candidates are added exactly as
    before, and no trust level ever auto-skips the explicit review."""
    rows, warnings = parse(text)
    if not rows:
        return {"error": "nothing importable in this diff "
                         f"({len(warnings)} warning(s))",
                "warnings": warnings}
    did = diff_id(rows)
    imports = state.get(IMPORTS_KEY)
    if not isinstance(imports, dict):
        imports = {}
    imports[did] = {"rows": rows, "n": len(rows)}
    if signer:
        imports[did]["signer"] = str(signer)
    state[IMPORTS_KEY] = imports
    # review-bucket candidates (the proposal's second landing): each row
    # becomes one candidate the learner's batch-review flow can label —
    # imports are absorbed through the supervised path, never silently
    try:
        from . import learn as cortex_learn
        bucket = [dict(c) for c in state.get(cortex_learn.REVIEW_KEY, [])
                  if isinstance(c, dict)]
        known = {c.get("text") for c in bucket}
        from datetime import datetime
        stamped = datetime.now().isoformat()
        for row in rows:
            if row["text"] in known:
                continue  # already parked for review
            bucket.append({"text": row["text"], "verdict": "IMPORTED",
                           "surface": row["surface"], "p": row.get("p"),
                           "at": stamped})
        state[cortex_learn.REVIEW_KEY] = bucket[:cortex_learn.MAX_CANDIDATES]
    except Exception:
        pass  # the review-bucket landing is best-effort, never fatal
    # the boosted-tools warning (the injection surface made visible)
    boosted = sorted({row["surface"] for row in rows})
    if signer:
        _append_trust_event(state, {
            "signer": str(signer), "action": "imported",
            "diff_id": did, "tools": boosted,
            "at": datetime.now().isoformat(timespec="seconds")})
    report = {
        "diff_id": did,
        "imported": len(rows),
        "boosted_tools": boosted,
        "warnings": warnings,
        "note": "the pairs weight the shared embedder at its next build "
                "(supervision, weight above the corpus prior — never the "
                "hand-seeded SYNONYMS); review candidates were added for "
                "the learner's batch flow",
        "rollback": f"cortex lexicon forget {did}",
        "verify": "verify the signature with YOUR external tool "
                  "(minisign/sq/gpg) before trusting the source — the "
                  "assistant states the content, the human owns the trust",
    }
    if signer:
        report["signer_trust"] = render_advisory(signer_trust(state),
                                                 str(signer))
    return report


def forget(state: Dict[str, Any], the_id: str) -> Dict[str, Any]:
    """Drop one imported set by diff-id. The pre-import pair list is
    restored by construction (the remaining imports stand); the next
    embedder build drops the forgotten pairs. A set that carried signer
    metadata records a ROLLBACK event for the trust propagation (phase
    2.3) — the one negative signal a signer can earn here."""
    imports = state.get(IMPORTS_KEY)
    if not isinstance(imports, dict) or the_id not in imports:
        known = sorted(k for k in imports if isinstance(k, str)) \
            if isinstance(imports, dict) else []
        return {"error": f"no imported diff {the_id!r} "
                + (f"(have: {', '.join(known)})" if known else
                   "(nothing imported)")}
    dropped = imports.pop(the_id)
    signer = (dropped.get("signer") if isinstance(dropped, dict)
              else None)
    if signer:
        _append_trust_event(state, {
            "signer": str(signer), "action": "rolled_back",
            "diff_id": the_id,
            "tools": sorted({row.get("surface", "")
                             for row in (dropped.get("rows") or [])
                             if isinstance(row, dict)}),
            "at": datetime.now().isoformat(timespec="seconds")})
    if not imports:
        state.pop(IMPORTS_KEY, None)
    else:
        state[IMPORTS_KEY] = imports
    result = {"forgot": the_id,
              "rows_dropped": len(dropped.get("rows", [])),
              "note": "the next embedder build drops these pairs "
                      "(corpus-only again for this set)"}
    if signer:
        result["note"] += ("; rollback recorded against signer "
                           f"{signer!r} (advisory trust signal)")
    return result


def imported_ids(state: Dict[str, Any]) -> List[str]:
    imports = state.get(IMPORTS_KEY)
    if not isinstance(imports, dict):
        return []
    return sorted(str(k) for k in imports)


def persisted_pairs(state: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Flatten the imported sets into supervised (text, surface) pairs
    for the embedder seam — the A2 ``labeled_pairs`` parameter."""
    pairs: List[Tuple[str, str]] = []
    imports = state.get(IMPORTS_KEY)
    if not isinstance(imports, dict):
        return pairs
    for rows in (v.get("rows") or [] for v in imports.values()
                 if isinstance(v, dict)):
        for row in rows:
            if isinstance(row, dict) and row.get("text") and row.get("surface"):
                pairs.append((str(row["text"]), str(row["surface"])))
    return pairs[:MAX_PAIRS]


# ---------------------------------------------------------------------------
# Signer trust (phase 2.3) — ADVISORY ONLY, never an auto-decision.
#
# EigenTrust (Kamvar, Schlosser & Garcia-Molina 2003, "The Eigentrust
# algorithm for reputation management in P2P networks", WWW) computes
# reputation as the fixed point of t = a*p + (1-a)*C^T t: direct evidence
# (the pretrust p) blended with what the graph propagates through
# normalized trust edges (C), iterated to convergence. This module has
# no peer-to-peer opinions — what it HAS locally is:
#   - direct evidence per signer: their diffs the user KEPT vs the diffs
#     the user ROLLED BACK (import/forget events above);
#   - edges worth propagating along: signers whose diffs boost the SAME
#     tools are correlated — a Jaccard overlap edge, so a rollback of one
#     diff also dents the signers whose diffs boost the same tools, and
#     a well-received diff lifts its correlated signers a little.
# The fixed point is computed by the same power iteration; the result is
# an ADVISORY score rendered next to the import report. It MUST NOT and
# DOES NOT auto-decide anything: every diff still lands as supervised
# pairs + review candidates, and the explicit review requirement is
# unchanged for every diff at every trust level (pinned by test).
# ---------------------------------------------------------------------------


def _append_trust_event(state: Dict[str, Any], event: Dict[str, Any]) -> None:
    """Bounded newest-appended event log (pure; the caller persists)."""
    trust = state.get(TRUST_KEY)
    if not isinstance(trust, dict):
        trust = {}
    events = [e for e in (trust.get("events") or []) if isinstance(e, dict)]
    events.append(event)
    trust["events"] = events[-MAX_TRUST_EVENTS:]
    state[TRUST_KEY] = trust


def signer_trust(state: Dict[str, Any], a: float = 0.15,
                 iterations: int = 25) -> Dict[str, Dict[str, Any]]:
    """The EigenTrust-style fixed point over the recorded signer events.

    Returns {signer: {"trust", "kept", "rolled_back", "tools"}} sorted
    by trust desc, name asc (deterministic). ``a`` is the blend toward
    direct evidence (the pretrust), the rest propagates through the
    Jaccard tool-overlap edges. No events -> {} (no invented opinions)."""
    events = []
    trust = state.get(TRUST_KEY)
    if isinstance(trust, dict):
        events = [e for e in (trust.get("events") or [])
                  if isinstance(e, dict) and e.get("signer")]
    if not events:
        return {}

    kept: Dict[str, int] = {}
    rolled: Dict[str, int] = {}
    tools: Dict[str, set] = {}
    for e in events:
        name = str(e.get("signer", ""))
        if not name:
            continue
        if e.get("action") == "rolled_back":
            rolled[name] = rolled.get(name, 0) + 1
        else:
            kept[name] = kept.get(name, 0) + 1
        tools.setdefault(name, set()).update(
            str(t) for t in (e.get("tools") or []) if t)
    names = sorted(set(kept) | set(rolled))

    # pretrust: the user's own Beta-smoothed keep rate (the bandit's
    # Beta(1,1) prior shape — no history means 0.5, never 1.0)
    pretrust = {}
    for name in names:
        pretrust[name] = (kept.get(name, 0) + 1.0) / \
            (kept.get(name, 0) + rolled.get(name, 0) + 2.0)
    total = sum(pretrust.values()) or 1.0
    p = {n: pretrust[n] / total for n in names}

    # edges: Jaccard overlap of boosted tool sets, row-normalized
    def jaccard(x: set, y: set) -> float:
        if not x or not y:
            return 0.0
        inter = len(x & y)
        union = len(x | y)
        return inter / union if union else 0.0

    t = {n: p[n] for n in names}
    for _ in range(max(1, iterations)):
        nxt = {}
        for i in names:
            propagated = 0.0
            for j in names:
                if i == j:
                    continue
                w = jaccard(tools.get(j, set()), tools.get(i, set()))
                if w > 0.0:
                    propagated += w * t[j]
            nxt[i] = a * p[i] + (1.0 - a) * propagated
        t = nxt
    return {
        name: {
            "trust": round(t[name], 4),
            "kept": kept.get(name, 0),
            "rolled_back": rolled.get(name, 0),
            "tools": len(tools.get(name, ())),
        }
        for name in sorted(names, key=lambda n: (-t[n], n))
    }


def render_advisory(scores: Dict[str, Dict[str, Any]],
                    signer: Optional[str] = None) -> str:
    """The human-facing ADVISORY sentence for the import report. Every
    rendering carries the explicit warning that review is NOT skipped —
    the trust score never decides anything."""
    if not scores:
        return ("advisory: no signer history yet — trust starts at the "
                "flat prior; review this diff explicitly (trust never "
                "auto-skips review)")
    rows = []
    for name, s in scores.items():
        marker = " (this diff)" if signer and name == signer else ""
        rows.append(f"{name}{marker}: trust {s['trust']} "
                    f"({s['kept']} kept, {s['rolled_back']} rolled back)")
    return ("advisory signer trust (EigenTrust-style over keep/rollback "
            "history and tool overlap — ADVISORY ONLY, review is never "
            "auto-skipped): " + "; ".join(rows[:6]))
