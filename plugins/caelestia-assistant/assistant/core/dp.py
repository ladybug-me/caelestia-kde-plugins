"""cortex.dp — Laplace-mechanism differential privacy for the
lexicon-diff export path (exponential-build 3, item B4).

The community lexicon-diff export (cortex/lexicon_diff.py) shares a
plain-text, reviewable list of ``phrase -> tool`` mappings with exact
per-phrase evidence (``n=4, p=0.86`` occurrence counts and mean route
probabilities). A community aggregator collecting many users' diffs
could reconstruct one contributor's raw behavior from those exact
counts. This module is the noising pass over the export artifact: an
OPT-IN transformation applied between export_rows() and sharing.

Mechanism: the Laplace mechanism — Dwork, McSherry, Nissim & Smith
2006, "Calibrating Noise to Sensitivity in Private Data Analysis",
TCC 2006 (the mechanism and the sensitivity calibration), composed
per Dwork & Roth 2014, "The Algorithmic Foundations of Differential
Privacy" (the composition property cited below). Each noised value is
value + Laplace(0, b) with scale b = sensitivity / epsilon, sampled
EXACTLY by the inverse CDF from one uniform draw of the injected rng:

    u = rng.random()                # one uniform in [0, 1)
    noise = -b * sign(u - 0.5) * ln(1 - 2*|u - 0.5|)

which is the standard inverse of the Laplace CDF (F(x) = 0.5*exp(x/b)
for x <= 0, 1 - 0.5*exp(-x/b) for x > 0). Deterministic given the rng;
one uniform draw per call. The tail u = 0.0 exactly (possible, measure
~2^-53) would make ln(0) undefined; the tail is floored at 1e-15 so
the noise is finite (~ -34.5b) — a documented tail treatment of the
mechanism's own sampling edge, never a clamp of user data.

DEFAULT EPSILON = 1.0, PER EXPORT. Why: the noised quantities are the
per-row evidence values (n and p), and per single release one user's
phrase-level contributions are bounded by the export path's own cap —
lexicon_diff.MAX_PAIRS = 200 phrase rows per export (the newest-200
cap in export_rows), drawn from a learner example log that is itself
capped (cortex/learn.py MAX_EXAMPLES = 500). Against the phrase-event
neighboring relation (the contributor's log differing by one routed
example), a single event changes exactly one row's n by 1 and its p by
at most 1 (worst case n=1: one event swings a [0,1] mean end to end),
so the per-row sensitivities are N_SENSITIVITY = 1 and
P_SENSITIVITY = 1 (the p bound is loose and the price is visible: at
epsilon=1 the noised p is clipped often — every clip is counted and
reported, never silent). The 200-row export cap is what bounds how
much phrase evidence one release can carry at all, which is why
"epsilon per export" is the unit a contributor reasons about: you
share one diff, you spend one epsilon.

COMPOSITION (Dwork & Roth 2014, the composition property): k
sequential exports compose to roughly k * epsilon total. The export
path is user-initiated and manual — ``cortex lexicon export --dp`` —
so the practical bound is simply how often YOU export. Say it plainly:
export once a month at epsilon=1 and a year of sharing is ~12 epsilon;
share hourly and the guarantee is gone. Nothing here enforces a
budget; the number is stated for the human who owns the decision.

HONEST BOUNDARY — this is NOISED-EVIDENCE DP, not full row-level DP:
pure epsilon-DP for a row's PRESENCE would require randomized
subsampling of rows, which is NOT the default — presence is exact;
only the counts and rates are noised. A phrase you typed appears in
the export exactly as often as before; what changes is that its n and
p are inexact. The opt-in ``presence_keep`` parameter adds honest
randomized row subsampling (each row independently kept with the given
probability, dropped rows counted): that AMPLIFIES the value-level
guarantee, but a KEPT row is still exactly present, so it is not by
itself a row-presence epsilon-DP claim and is not labeled as one.

REPRODUCIBLE-NOISE TRADEOFF: by default the seed is derived
deterministically from the diff's content id (lexicon_diff.diff_id),
so noising the same diff twice yields byte-identical output — the
artifact is reviewable and re-derivable. The cost, stated plainly:
reproducible noise is NOT independent across exports. Two releases of
the SAME diff at epsilon each are not two independent epsilon-DP
releases (the noise repeats, so the second release adds no fresh
leakage but also no fresh randomness); the composition accounting
above assumes DIFFERENT exports. Pass an explicit seed when you want
fresh, independent noise.

The module is PURE: it writes nothing (the CLI prints), takes rows or
the rendered diff text, and returns the noised artifact plus a report
of every floor and clip. Re-noising an already-noised diff (one
carrying the provenance marker) is REJECTED — composing Laplace noise
at the same stated epsilon would misreport the total. Deterministic:
same (diff, epsilon, seed) -> byte-identical output.
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from . import lexicon_diff
from .lexicon_diff import _DATE_RE, _ROW_RE

__all__ = ["DEFAULT_EPSILON", "N_SENSITIVITY", "P_SENSITIVITY",
           "laplace_noise", "noise_diff"]

DEFAULT_EPSILON = 1.0
# per-row sensitivities under the phrase-event neighboring relation
# (see the module docstring's epsilon rationale)
N_SENSITIVITY = 1.0
P_SENSITIVITY = 1.0

# the inverse-CDF tail floor: u == 0.0 exactly (measure ~2^-53) would
# make ln(1 - 2|u - 0.5|) = ln(0) undefined; floor the argument so the
# draw stays finite (~ -34.5b) instead of -inf
_TAIL_FLOOR = 1e-15

GUARANTEE = ("noised-evidence DP: per-row n/p are Laplace-noised "
             "(Dwork et al. 2006); row PRESENCE is exact unless "
             "presence_keep is set — not full row-level DP")


def laplace_noise(value: float, sensitivity: float, epsilon: float,
                  rng: random.Random) -> float:
    """value + Laplace(0, b) noise with b = sensitivity / epsilon,
    sampled exactly by the inverse CDF from ONE uniform draw of
    ``rng`` (formula in the module docstring). Deterministic given the
    rng. sensitivity <= 0 and epsilon <= 0 are REJECTED (a noise scale
    is not definable from them — refused, never guessed)."""
    sensitivity = float(sensitivity)
    epsilon = float(epsilon)
    if sensitivity <= 0.0:
        raise ValueError(f"sensitivity must be > 0 (got {sensitivity})")
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be > 0 (got {epsilon})")
    b = sensitivity / epsilon
    u = float(rng.random())
    offset = u - 0.5
    if offset > 0.0:
        sign = 1.0
    elif offset < 0.0:
        sign = -1.0
    else:
        sign = 0.0  # u == 0.5 exactly: the median, noise exactly 0
    magnitude = 1.0 - 2.0 * abs(offset)
    if magnitude < _TAIL_FLOOR:
        magnitude = _TAIL_FLOOR
    return float(value) - b * sign * math.log(magnitude)


def _rows_from_text(text: str) -> Tuple[List[Dict[str, Any]], str]:
    """Parse the rendered diff back into rows using lexicon_diff's own
    row regex (the ONE row-format implementation — reused, not
    duplicated). No registry validation here: the input is the
    exporter's own artifact and this pass only transforms values; the
    importer's parse() still validates tools at import time. Returns
    (rows, date). Unparseable non-comment lines are REJECTED (the
    repo rule: out-of-range is refused, never guessed)."""
    rows: List[Dict[str, Any]] = []
    date = ""
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == lexicon_diff.DIFF_HEADER \
                or stripped.startswith(lexicon_diff.DIFF_HEADER):
            m = _DATE_RE.match(stripped[len(lexicon_diff.DIFF_HEADER):].strip())
            if m and not date:
                date = m.group(1)
            continue
        m = _ROW_RE.match(stripped)
        if not m:
            raise ValueError(f"line {lineno}: unparseable diff row, "
                             f"refused: {stripped[:60]!r}")
        rows.append({
            "text": m.group("text").strip(),
            "surface": m.group("surface").strip(),
            "n": int(m.group("n") or 1),
            "p": min(1.0, max(0.0, float(m.group("p") or 0.0))),
            "label": 1 if m.group("sign") == "+" else 0,
            "dp_epsilon": float(m.group("dp")) if m.group("dp") else None,
        })
    return rows, date


def _marker(epsilon: float) -> str:
    return f"(dp: epsilon={float(epsilon)})"


def noise_diff(diff: Union[str, Sequence[Dict[str, Any]]],
               epsilon: float = DEFAULT_EPSILON,
               seed: Optional[int] = None,
               min_count_floor: bool = True,
               date: str = "",
               presence_keep: Optional[float] = None) -> Dict[str, Any]:
    """Noise one lexicon-diff export artifact (Laplace mechanism).

    ``diff``: the rendered diff TEXT or the export row dicts. An
    input already carrying a dp provenance marker is REJECTED —
    re-noising would compose epsilon while claiming one epsilon.
    ``epsilon``: the per-export privacy budget (> 0, else ValueError).
    ``seed``: None (default) derives the seed deterministically from
    the diff's content id (diff_id) — reproducible noise, tradeoff
    stated in the module docstring; an int gives fresh independent
    noise. ``min_count_floor`` (default True): negative noised counts
    are floored at zero AND counted in the report; False keeps the raw
    negative value (mechanism-audit mode — the artifact it renders
    will not re-parse, which lexicon_diff.parse reports as warnings,
    the existing honest-degradation path). ``presence_keep``: None
    (default) keeps every row (presence exact); a float in (0, 1]
    keeps each row independently with that probability (randomized
    row subsampling — amplification, not a row-presence epsilon-DP
    claim; dropped rows counted).

    Per row, in input order, the rng draws are fixed-order: the
    presence draw first (when enabled), then n, then p. Same (diff,
    epsilon, seed) -> byte-identical output. Returns the noised
    artifact {"text", "rows", counts, guarantee label, notes}; writes
    nothing — the caller prints."""
    epsilon = float(epsilon)
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be > 0 (got {epsilon})")
    if presence_keep is not None:
        presence_keep = float(presence_keep)
        if not 0.0 < presence_keep <= 1.0:
            raise ValueError("presence_keep must be within (0, 1] "
                             f"(got {presence_keep})")

    # normalize input to rows (reusing the ONE row-format regex)
    if isinstance(diff, str):
        rows, parsed_date = _rows_from_text(diff)
        if not date:
            date = parsed_date
    else:
        rows = [dict(r) for r in diff]
    if not rows:
        raise ValueError("no rows to noise — the export is empty")
    for row in rows:
        if row.get("dp_epsilon") is not None:
            raise ValueError(
                "this diff is already noised (a dp provenance marker is "
                "present) — re-noising would compose epsilon while "
                "claiming one epsilon; export the exact diff and noise once")

    # the seed: derived from the content id (reproducible) or explicit
    if seed is None:
        seed = int(lexicon_diff.diff_id(rows), 16)
    else:
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            raise ValueError(f"seed must be an integer or None (got {seed!r})")
    rng = random.Random(seed)

    marker = _marker(epsilon)
    out_rows: List[Dict[str, Any]] = []
    n_floored = n_negative_raw = 0
    p_clip_lo = p_clip_hi = 0
    dropped = 0
    for row in rows:
        if presence_keep is not None and rng.random() >= presence_keep:
            dropped += 1
            continue
        # n: Laplace noise, nearest integer (halves up), floored at
        # zero when min_count_floor — every floor COUNTED, never silent
        noised_n = laplace_noise(int(row.get("n", 1)), N_SENSITIVITY,
                                 epsilon, rng)
        n_int = int(math.floor(noised_n + 0.5))
        if n_int < 0:
            if min_count_floor:
                n_int = 0
                n_floored += 1
            else:
                n_negative_raw += 1
        # p: Laplace noise, clipped to the valid [0, 1] RANGE and the
        # clip COUNTED and reported (the DP-output convention: the
        # noise is honest about its own postprocessing)
        noised_p = laplace_noise(float(row.get("p", 0.0)), P_SENSITIVITY,
                                 epsilon, rng)
        if noised_p < 0.0:
            noised_p = 0.0
            p_clip_lo += 1
        elif noised_p > 1.0:
            noised_p = 1.0
            p_clip_hi += 1
        out_rows.append({
            "text": str(row.get("text", "")),
            "surface": str(row.get("surface", "")),
            "n": n_int,
            "p": round(noised_p, 2),
            "label": 1 if int(row.get("label", 1)) == 1 else 0,
            "dp_epsilon": epsilon,
        })

    # the artifact: same canonical header, one provenance comment, and
    # every row carrying its marker — a reviewer can never mistake a
    # noised diff for an exact one
    lines = [f"{lexicon_diff.DIFF_HEADER}  ({date})" if date
             else lexicon_diff.DIFF_HEADER]
    lines.append(f"# {marker.strip()} — counts and rates are Laplace-"
                 "noised (Dwork et al. 2006); presence is exact; "
                 "see cortex/dp.py")
    for row in out_rows:
        sign = "+" if row["label"] == 1 else "-"
        lines.append(f"{sign}{row['text']} -> {row['surface']}  "
                     f"(n={row['n']}, p={row['p']:.2f}) {marker}")

    return {
        "text": "\n".join(lines) + "\n",
        "rows": out_rows,
        "epsilon": epsilon,
        "seed": seed,
        "rows_in": len(rows),
        "rows_out": len(out_rows),
        "rows_dropped": dropped,
        "n_floored_at_zero": n_floored,
        "n_negative_raw": n_negative_raw,
        "p_clipped": p_clip_lo + p_clip_hi,
        "p_clipped_low": p_clip_lo,
        "p_clipped_high": p_clip_hi,
        "guarantee": GUARANTEE,
        "method": (f"Laplace mechanism (Dwork, McSherry, Nissim & Smith "
                   f"2006): n and p noised at sensitivity "
                   f"n={N_SENSITIVITY:g}, p={P_SENSITIVITY:g}, "
                   f"epsilon={epsilon:g} per export; composition: k "
                   "sequential exports ~ k*epsilon"),
        "notes": [
            GUARANTEE,
            "composition: k sequential exports compose to roughly "
            "k*epsilon (Dwork & Roth 2014); the export path is "
            "user-initiated and manual, so the practical bound is how "
            "often you export",
            f"seed {seed} "
            + ("derived from the diff's content id — reproducible noise, "
               "NOT independent across exports of the same content"
               if seed == int(lexicon_diff.diff_id(rows), 16) else
               "explicitly supplied"),
        ],
    }
