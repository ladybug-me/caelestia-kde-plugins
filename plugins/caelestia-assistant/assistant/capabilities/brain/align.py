"""brain.align — Needleman-Wunsch alignment of proposed vs. actual tidy moves.

The imitation-learning question: when the user reorganizes files by hand
AFTER tidy.survey() proposed a plan, where do their real moves and the
proposal CONSISTENTLY disagree? A persistent divergence is a rule-
correction signal — "you always move screenshots to ~/shots, the proposal
keeps saying ~/Pictures/images" — with the same discipline as the settings
parser's corrections, and the same gate at the end: a LEDGER PROPOSAL the
user approves or rejects. Nothing is ever learned, applied or written
silently.

SAFETY POSTURE (this is the highest-stakes neighbour in the tree, §6.5 of
the directive): this module performs ZERO filesystem operations. It does
not read, write, move or delete anything — it aligns two CALLER-SUPPLIED
lists of move records ({"from": path, "to": path, "category": str}) and
returns plain data. tidy.py's own never-delete / journaled / rollback-able
discipline is inherited precisely because this module cannot touch the
filesystem at all: it proposes rule corrections, and the corrections only
ever change where FUTURE proposals suggest moving things — through the
ledger, after your approval.

Data source, stated honestly (the workspace.py rule): nothing in this
repository observes the user's manual file moves. The caller supplies the
actual-moves list from whatever observation surface it has (a before/
after snapshot, a file-watcher, or explicit input). The schema is just
{"from", "to", "category"} rows.

Algorithm: Needleman & Wunsch 1970 ("A general method applicable to the
search for similarities in the amino acid sequence of two proteins",
J. Mol. Biol. 48(3):443-453) — global sequence alignment by dynamic
programming with linear gap penalties, here over move records instead of
residues: the substitution score is 2*similarity-1 (identical moves +1,
unrelated -1) where similarity rewards a shared source path and, weaker,
a shared basename/category; the gap penalty is -1. O(n*m) in the two
sequence lengths, which for tidy plans is trivial.

The minimum-consistency floor: a divergence pattern only becomes a
correction candidate when it repeats at least ``min_support`` times
(default 3, the same floor the workspace profiles and the ontology gaps
use) — one disagreement is a choice, three is a rule.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["move_similarity", "align_moves", "divergence_report",
           "propose_corrections", "MIN_SUPPORT", "GAP_PENALTY"]

MIN_SUPPORT = 3   # the workspace.py / gap-clustering floor, reused verbatim
GAP_PENALTY = -1.0
SAME_SOURCE = 0.9  # similarity above which two move records name the same file


def _basename(path: str) -> str:
    return path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def _dirname(path: str) -> str:
    parts = path.replace("\\", "/").rstrip("/").rsplit("/", 1)
    return parts[0] if len(parts) == 2 else ""


def move_similarity(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """0..1 similarity between two move records: 1.0 for the same source
    path; partial credit for the same basename (the same file addressed
    by a different spelling) or the same category + destination; 0 for
    unrelated moves. Pure string facts only."""
    fa, fb = str(a.get("from", "")), str(b.get("from", ""))
    if fa and fa == fb:
        return 1.0
    if fa and fb and _basename(fa) == _basename(fb):
        return 0.6
    if a.get("category") and a.get("category") == b.get("category"):
        if a.get("to") and str(a.get("to")) == str(b.get("to")):
            return 0.4
    return 0.0


def align_moves(proposed: Sequence[Dict[str, Any]],
                actual: Sequence[Dict[str, Any]],
                gap_penalty: float = GAP_PENALTY
                ) -> Dict[str, Any]:
    """Global Needleman-Wunsch alignment of the two move sequences.

    Returns {"score", "pairs": [(i, j) or None per aligned row — the
    traceback as a list of (proposed_index_or_None, actual_index_or_None)],
    "aligned", "gaps"}. Deterministic: ties prefer diagonal (match) over
    gaps, and the traceback prefers consuming the proposed side."""
    n, m = len(proposed), len(actual)
    if n == 0 and m == 0:
        return {"score": 0.0, "trace": [], "aligned": 0, "gaps": 0}

    def _sub(i: int, j: int) -> float:
        return 2.0 * move_similarity(proposed[i], actual[j]) - 1.0

    # DP table with sentinel -inf
    NEG = float("-inf")
    dp = [[NEG] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + gap_penalty
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + gap_penalty
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = max(dp[i - 1][j - 1] + _sub(i - 1, j - 1),
                          dp[i - 1][j] + gap_penalty,
                          dp[i][j - 1] + gap_penalty)

    # traceback
    trace: List[Tuple[Optional[int], Optional[int]]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + _sub(i - 1, j - 1):
            trace.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + gap_penalty:
            trace.append((i - 1, None))
            i -= 1
        elif j > 0:
            trace.append((None, j - 1))
            j -= 1
        else:  # pragma: no cover - loop invariant
            break
    trace.reverse()
    aligned = sum(1 for a_idx, b_idx in trace if a_idx is not None and b_idx is not None)
    gaps = len(trace) - aligned
    return {"score": round(dp[n][m], 4), "trace": trace,
            "aligned": aligned, "gaps": gaps}


def divergence_report(proposed: Sequence[Dict[str, Any]],
                      actual: Sequence[Dict[str, Any]],
                      min_support: int = MIN_SUPPORT) -> Dict[str, Any]:
    """Align the two move lists and extract the consistent disagreements.

    A DIVERGENT pair is an aligned pair naming the same source file whose
    destinations differ — the user moved that file somewhere the proposal
    did not say. Divergences are grouped into patterns by
    (category, proposed destination directory, actual destination
    directory); a pattern becomes a candidate only at ``min_support``
    occurrences (default 3 — the floor every other suggestion surface in
    this codebase already uses)."""
    alignment = align_moves(proposed, actual)
    divergent: List[Dict[str, Any]] = []
    matched = 0
    for p_idx, a_idx in alignment["trace"]:
        if p_idx is None or a_idx is None:
            continue
        p, a = proposed[p_idx], actual[a_idx]
        if move_similarity(p, a) < SAME_SOURCE:
            continue
        if str(p.get("to", "")) == str(a.get("to", "")):
            matched += 1
            continue
        divergent.append({
            "from": str(p.get("from", "")),
            "category": str(p.get("category") or a.get("category") or ""),
            "proposed_to": str(p.get("to", "")),
            "actual_to": str(a.get("to", "")),
            "proposed_dir": _dirname(str(p.get("to", ""))),
            "actual_dir": _dirname(str(a.get("to", ""))),
        })

    patterns: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for d in divergent:
        key = (d["category"], d["proposed_dir"], d["actual_dir"])
        patterns.setdefault(key, []).append(d)

    candidates: List[Dict[str, Any]] = []
    rejected = 0
    for (category, proposed_dir, actual_dir), rows in sorted(patterns.items()):
        support = len(rows)
        if support < min_support:
            rejected += 1
            continue
        candidates.append({
            "category": category,
            "proposed_dir": proposed_dir,
            "actual_dir": actual_dir,
            "support": support,
            "examples": [r["from"] for r in rows[:5]],
        })
    candidates.sort(key=lambda c: (-c["support"], c["category"]))
    return {
        "n_proposed": len(proposed),
        "n_actual": len(actual),
        "aligned": alignment["aligned"],
        "aligned_same_destination": matched,
        "divergent": len(divergent),
        "candidates": candidates,
        "rejected_patterns": rejected,
        "alignment_score": alignment["score"],
    }


def propose_corrections(proposed: Sequence[Dict[str, Any]],
                        actual: Sequence[Dict[str, Any]],
                        ledger,
                        min_support: int = MIN_SUPPORT) -> Dict[str, Any]:
    """Turn qualifying divergence patterns into pending LEDGER PROPOSALS
    (kind ``tidy_rule``): "you moved N image files to X instead of the
    proposed Y — make X the images target?" Approval changes where future
    proposals point; it never moves, deletes or touches any file, and it
    is never automatic."""
    summary = divergence_report(proposed, actual, min_support=min_support)
    pending = {p.get("target") for p in ledger.pending()}
    pids: List[int] = []
    for candidate in summary["candidates"]:
        target = (f"tidy-rule:{candidate['category']}:"
                  f"{candidate['proposed_dir']}->{candidate['actual_dir']}")[:120]
        if target in pending:
            continue
        why = (f"you moved {candidate['support']} {candidate['category'] or 'file'} "
               f"item(s) to {candidate['actual_dir'] or '(root)'} when the tidy "
               f"plan proposed {candidate['proposed_dir'] or '(root)'} — "
               f"make the actual one the target for future plans?")
        pids.append(ledger.propose("tidy_rule", target,
                                   {"correction": candidate}, why,
                                   min(0.9, candidate["support"] /
                                       max(1, summary["divergent"]))))
    return {"proposals": pids, "summary": summary}


# The module's purity is structural, not aspirational (and pinned by
# test_align.py::test_module_touches_no_files): this file imports nothing
# but typing — no os, no pathlib, no open/rename/remove anywhere. It
# aligns strings, and proposes to a ledger it is handed.
