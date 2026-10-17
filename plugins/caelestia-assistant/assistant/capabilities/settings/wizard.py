"""First-run setup wizard (issue #120 Phase 3 roadmap: "setup wizards").

A handful of pairwise tradeoff questions over FOUR criteria — visual
fidelity, battery, minimalism, performance — are turned into AHP weights
(genius/decision.py::ahp, Saaty's pairwise method with the consistency
ratio), and the five shipped presets are scored against those criteria
with TOPSIS (genius/decision.py::topsis) to recommend a starting preset.

Everything reuses code that already exists and is already tested:

- the criterion scores come from `optimize.score_plan(profile, ops)` —
  the same scored objective profiles the #120 phase-3 optimizer ships;
  the four wizard criteria map onto them directly:
      visual fidelity -> comfort, battery -> battery,
      minimalism -> minimal, performance -> gaming;
- the preset candidates are the registry's own five preset bundles
  (compact, minimal, gaming, battery-saver, macos-like) through
  presets.preset_ops — the same validated bundles the --preset flag
  applies;

and nothing here writes: the recommendation is returned (and rendered)
with its bundled ops for the caller to apply through the EXISTING
--preset/--apply/--confirm gates or the ledger. A recommendation the
user ignores costs nothing.

Note on question count: four criteria give C(4,2) = 6 pairwise questions.
decision.ahp validates full-matrix reciprocity, so a spanning subset of
questions would need invented cells; six is the honest minimum. The
answer scale is deliberately coarse (5 options per question) so the
wizard stays a few seconds long.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from assistant.capabilities.genius import decision
from .optimize import score_plan
from .presets import PresetError, preset_ops, presets

CRITERIA = ("visual fidelity", "battery", "minimalism", "performance")
_CRITERION_PROFILE = {
    "visual fidelity": "comfort",
    "battery": "battery",
    "minimalism": "minimal",
    "performance": "gaming",
}

# The six criterion pairs, in fixed question order: (first, second).
QUESTION_PAIRS = (
    ("visual fidelity", "battery"),
    ("visual fidelity", "performance"),
    ("visual fidelity", "minimalism"),
    ("battery", "performance"),
    ("battery", "minimalism"),
    ("performance", "minimalism"),
)

# Answer 1..5 maps to the Saaty intensity for the FIRST criterion over the
# second: strongly (5), moderately (3), equally (1), and the reciprocals.
ANSWER_SCALE = {
    1: 5.0,    # first strongly more important
    2: 3.0,    # first moderately more important
    3: 1.0,    # equally important
    4: 1.0 / 3.0,   # second moderately more important
    5: 1.0 / 5.0,   # second strongly more important
}

ANSWER_HINTS = (
    "1 = first strongly, 2 = first moderately, 3 = equally, "
    "4 = second moderately, 5 = second strongly",
)


def question_lines() -> List[str]:
    """The exact questions, for the CLI to render verbatim."""
    lines = [f"Which matters more, and how strongly? ({ANSWER_HINTS[0]})"]
    for i, (a, b) in enumerate(QUESTION_PAIRS, start=1):
        lines.append(f"  {i}. {a}  vs  {b}")
    return lines


def _pairwise_matrix(answers: Sequence[int]) -> List[List[float]]:
    if len(answers) != len(QUESTION_PAIRS):
        raise ValueError(
            f"expected {len(QUESTION_PAIRS)} answers, got {len(answers)}")
    n = len(CRITERIA)
    matrix = [[1.0] * n for _ in range(n)]
    index = {name: i for i, name in enumerate(CRITERIA)}
    for answer, (a, b) in zip(answers, QUESTION_PAIRS):
        key = int(answer)
        if key not in ANSWER_SCALE:
            raise ValueError(f"answer {answer!r} is not on the 1-5 scale")
        intensity = ANSWER_SCALE[key]
        i, j = index[a], index[b]
        matrix[i][j] = intensity
        matrix[j][i] = 1.0 / intensity  # Saaty reciprocity, checked by ahp()
    return matrix


def _preset_scores() -> List[Dict[str, Any]]:
    """Every preset scored against every criterion via optimize.score_plan.
    Deterministic: the presets and their ops are frozen registry data."""
    rows: List[Dict[str, Any]] = []
    for preset in presets():
        ops = preset_ops(preset["name"])
        scores = [score_plan(_CRITERION_PROFILE[c], ops)["score"]
                  for c in CRITERIA]
        rows.append({"preset": preset["name"], "scores": scores,
                     "ops": ops})
    return rows


def run(answers: Sequence[int]) -> Dict[str, Any]:
    """Fixed answers -> deterministic preset recommendation. Pure.

    Returns {"winner", "closeness", "ranking", "weights",
    "consistent", "consistency_ratio", "criteria_scores", "ops"}.
    """
    matrix = _pairwise_matrix(answers)
    ahp = decision.ahp(matrix, list(CRITERIA))
    # ahp() returns priorities as {name: weight}; topsis() wants them
    # column-ordered, so project onto the fixed CRITERIA order.
    weights = [float(ahp["priorities"][c]) for c in CRITERIA]

    scored = _preset_scores()
    topsis_matrix = [row["scores"] for row in scored]
    labels = [row["preset"] for row in scored]
    topsis = decision.topsis(topsis_matrix, labels, weights,
                             list(CRITERIA), benefits=[True] * len(CRITERIA))

    winner = topsis["winner"]
    winner_row = next(row for row in scored if row["preset"] == winner)
    return {
        "winner": winner,
        "closeness": topsis["closeness"],
        "ranking": topsis["ranking"],
        "weights": weights,
        "consistent": ahp["consistent"],
        "consistency_ratio": ahp["consistency_ratio"],
        "criteria_scores": {row["preset"]: dict(zip(CRITERIA, row["scores"]))
                            for row in scored},
        "ops": winner_row["ops"],
    }


def render(result: Dict[str, Any]) -> List[str]:
    """Plain-text rendering for the CLI (read-only; the ops shown ride the
    ordinary --preset path if the user wants them applied)."""
    lines = ["Setup wizard — starting-preset recommendation",
             ""]
    lines.append(f"recommended preset: {result['winner']} "
                 f"(TOPSIS closeness {result['closeness'][result['winner']]:.3f})")
    lines.append(f"criterion weights (AHP, consistency ratio "
                 f"{result['consistency_ratio']:.3f}, "
                 f"{'consistent' if result['consistent'] else 'INCONSISTENT — answers contradict each other; treat the ranking as rough'}):")
    for name, weight in zip(CRITERIA, result["weights"]):
        lines.append(f"  {name:<16} {weight:.3f}")
    lines.append("")
    lines.append("full ranking:")
    for name in result["ranking"]:
        closeness = result["closeness"][name]
        lines.append(f"  {name:<15} {closeness:.3f}")
    lines.append("")
    lines.append("the bundled calls (applied only through the ordinary "
                 "--preset gate — this wizard writes nothing):")
    for op in result["ops"]:
        lines.append(f"  {op['tool']} = {op['value']!r}")
    return lines


def mainish(answers: Optional[Sequence[int]] = None) -> int:  # pragma: no cover
    """Interactive entry used by settings --wizard; split from cli.py for
    testability of the non-interactive path."""
    import sys
    if answers is None:
        print("\n".join(question_lines()))
        try:
            raw = input("your six answers, comma separated (e.g. 2,3,1,4,2,3): ")
        except EOFError:
            return 2
        try:
            answers = [int(x) for x in raw.split(",")]
        except ValueError:
            print("answers must be integers 1-5", file=sys.stderr)
            return 2
    try:
        result = run(answers)
    except (ValueError, PresetError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("\n".join(render(result)))
    return 0
