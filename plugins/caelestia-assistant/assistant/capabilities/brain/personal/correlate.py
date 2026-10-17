"""personal.correlate — habit-log signals vs task completion.

Answers: "do the days/weeks where I keep habit X also tend to be the
ones where my tasks get done?" — and is careful to answer ONLY that.
This module ships TWO classical tests:

- the point-biserial correlation (Tate 1954, "Correlation between a
  discrete and a continuous variable. Point-biserial correlation",
  Ann. Math. Statist. 25) between a CONTINUOUS habit signal (minutes
  practiced, streak length, ...) and the binary completion outcome —
  mathematically the Pearson r for a dichotomous-and-continuous pair;
- the chi-square test of independence on the 2x2 contingency table of
  a BINARY habit flag vs completion, with Yates' continuity correction
  (Pearson 1900 for X²; Yates 1934 for the correction) — the right
  small-sample shape for a personal log.

HONESTY CONVENTIONS (the repo's own rules, applied here):
- CORRELATIONAL, NEVER CAUSAL: every result carries the framing string;
  a habit and completion moving together is not one causing the other
  (hidden common causes: free time, energy, season).
- THIN EVIDENCE IS LABELED THIN: n < 30, or any expected chi-square
  cell below 5, sets ``thin: True`` on the result — the number still
  computes, the label travels with it.
- inputs are the caller's records ({"habit": number, "completed":
  bool}); the habit signal itself is NOT tracked anywhere in this repo
  (grep-first: no habit log exists) — the caller extracts it from
  their own journal/habit data and pairs it with the task records the
  survival/priority engines already consume. Nothing is inferred from
  unstated behavior here.

Pure functions; no I/O; the caller persists nothing but their own data.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence

__all__ = ["pair_records", "point_biserial", "chi_square", "summary",
           "THIN_N_FLOOR", "THIN_EXPECTED_CELL"]

THIN_N_FLOOR = 30
THIN_EXPECTED_CELL = 5.0

_FRAMING = ("correlational, not causal — a habit moving with completion "
            "is not the habit causing completion")


def pair_records(tasks: Sequence[Dict[str, Any]],
                 habit_of: Callable[[Dict[str, Any]], Optional[float]]
                 ) -> List[Dict[str, Any]]:
    """Pair task records (the same records the survival and priority
    engines consume — anything carrying a completion flag) with a habit
    signal extracted by ``habit_of``. Tasks whose signal is None or
    missing are SKIPPED, never imputed: an unstated habit value is not
    evidence either way."""
    records: List[Dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        completed = task.get("completed", task.get("done"))
        signal = habit_of(task)
        if signal is None or completed is None:
            continue
        records.append({"habit": float(signal),
                        "completed": bool(completed)})
    return records


def point_biserial(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Point-biserial r between the continuous habit signal and the
    binary completion outcome (Tate 1954). Thin-labeled below the
    sample floor."""
    pairs = [(r["habit"], bool(r["completed"])) for r in records]
    n = len(pairs)
    n1 = sum(1 for _x, y in pairs if y)
    n0 = n - n1
    if n < 3 or n1 == 0 or n0 == 0:
        return {"r": None, "n": n, "n_completed": n1,
                "thin": True, "framing": _FRAMING,
                "note": "need completions AND non-completions to compare"}
    group1 = [x for x, y in pairs if y]
    group0 = [x for x, y in pairs if not y]
    mean1 = sum(group1) / n1
    mean0 = sum(group0) / n0
    all_x = [x for x, _y in pairs]
    mean = sum(all_x) / n
    # sample standard deviation (n-1 denominator)
    var = sum((x - mean) ** 2 for x in all_x) / (n - 1)
    s = math.sqrt(var)
    if s == 0.0:
        return {"r": None, "n": n, "n_completed": n1, "thin": True,
                "framing": _FRAMING,
                "note": "habit signal has zero variance — nothing to "
                        "correlate"}
    r = ((mean1 - mean0) / s) * math.sqrt(n1 * n0 / (n * n))
    return {"r": round(r, 4), "n": n, "n_completed": n1,
            "mean_habit_when_completed": round(mean1, 4),
            "mean_habit_when_not": round(mean0, 4),
            "thin": n < THIN_N_FLOOR, "framing": _FRAMING}


def chi_square(records: Sequence[Dict[str, Any]],
               threshold: float = 0.5) -> Dict[str, Any]:
    """2x2 chi-square of independence (binary habit flag vs completion)
    with Yates' continuity correction. ``threshold`` binarizes a
    continuous signal at 0.5 by default — a caller decision, stated in
    the result. Any expected cell below 5 is thin evidence."""
    table = [[0, 0], [0, 0]]  # [habit][completed]
    binarized = False
    for r in records:
        habit = r["habit"]
        completed = bool(r["completed"])
        flag: Optional[bool] = None
        if isinstance(habit, bool) or habit in (0, 1):
            flag = bool(habit)
        else:
            flag = float(habit) >= threshold
            binarized = True
        table[1 if flag else 0][1 if completed else 0] += 1
    n = sum(sum(row) for row in table)
    if n == 0:
        return {"chi2": None, "n": 0, "thin": True, "framing": _FRAMING,
                "note": "no records"}
    row_totals = [sum(row) for row in table]
    col_totals = [table[0][j] + table[1][j] for j in range(2)]
    expected = [[row_totals[i] * col_totals[j] / n for j in range(2)]
                for i in range(2)]
    chi2 = 0.0
    for i in range(2):
        for j in range(2):
            e = expected[i][j]
            if e <= 0:
                continue
            chi2 += (max(abs(table[i][j] - e) - 0.5, 0.0)) ** 2 / e
    thin = n < THIN_N_FLOOR or min(expected[0][0], expected[0][1],
                                   expected[1][0], expected[1][1]) \
        < THIN_EXPECTED_CELL
    rate_with = table[1][1] / row_totals[1] if row_totals[1] else None
    rate_without = table[0][1] / row_totals[0] if row_totals[0] else None
    return {"chi2": round(chi2, 4), "n": n,
            "completed_rate_with_habit": round(rate_with, 4)
            if rate_with is not None else None,
            "completed_rate_without_habit": round(rate_without, 4)
            if rate_without is not None else None,
            "table": {"habit_and_completed": table[1][1],
                      "habit_and_open": table[1][0],
                      "no_habit_and_completed": table[0][1],
                      "no_habit_and_open": table[0][0]},
            "binarized_at": threshold if binarized else None,
            "thin": thin, "framing": _FRAMING}


def summary(records: Sequence[Dict[str, Any]],
            threshold: float = 0.5) -> Dict[str, Any]:
    """Both views at once (continuous r and binarized chi-square), for
    the CLI/bridge caller that wants one honest card."""
    return {
        "point_biserial": point_biserial(records),
        "chi_square": chi_square(records, threshold=threshold),
        "framing": _FRAMING,
    }
