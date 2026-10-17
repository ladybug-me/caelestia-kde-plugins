"""Spaced review: FSRS-inspired scheduling (not the published FSRS weights).

Each card keeps stability S (days until recall probability falls to 90%) and
difficulty D in [1, 10]. Retrievability R(t) = (1 + t / (9 S))^-1. With target
retention 0.9, the next interval equals S days.
"""
import math

RATINGS = {1: "again", 2: "hard", 3: "good", 4: "easy"}
_MULT = {2: 0.8, 3: 1.0, 4: 1.3}
_GROWTH = math.exp(0.6) - 1


def retrievability(elapsed_days, stability):
    return (1 + elapsed_days / (9 * max(stability, 1e-6))) ** -1


def new_card():
    return {"S": 1.0, "D": 5.0, "elapsed": 0.0}


def review(card, rating, elapsed_days=None):
    """Return an updated card. rating in {1,2,3,4}."""
    if rating not in RATINGS:
        raise ValueError("rating must be 1..4")
    elapsed = card["elapsed"] if elapsed_days is None else elapsed_days
    S, D = card["S"], card["D"]
    R = retrievability(elapsed, S)
    if rating == 1:
        S = max(0.5, S * 0.3)
        D = min(10.0, D + 1.0)
    else:
        growth = 1 + (11 - D) / 10 * _GROWTH * (1 - R) * _MULT[rating]
        S = S * max(1.0, growth)
        D = min(10.0, max(1.0, D - 0.8 * (rating - 3)))
    return {"S": round(S, 4), "D": round(D, 4), "elapsed": 0.0}


def due(card, now_elapsed, threshold=0.9):
    return retrievability(now_elapsed, card["S"]) <= threshold
