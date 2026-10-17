"""Kaplan-Meier survival on task age, to propose culling dead tasks.

Each task contributes (age_days, completed). Open tasks are right-censored.
S(t) = P(time-to-completion > t). For an open task of age a, the probability of
completing within the next `horizon` days is (S(a) - S(a+h)) / S(a).
"""


def kaplan_meier(durations, observed):
    pairs = sorted(zip(durations, observed))
    n_at_risk = len(pairs)
    s = 1.0
    curve = []
    i = 0
    while i < len(pairs):
        t = pairs[i][0]
        events = 0
        removed = 0
        while i < len(pairs) and pairs[i][0] == t:
            events += 1 if pairs[i][1] else 0
            removed += 1
            i += 1
        if events:
            s *= 1 - events / n_at_risk
            curve.append((t, s))
        n_at_risk -= removed
    return curve


def survival_at(curve, t):
    s = 1.0
    for ti, si in curve:
        if ti <= t:
            s = si
        else:
            break
    return s


def completion_prob(curve, age, horizon=14):
    s0 = survival_at(curve, age)
    if s0 <= 0:
        return 0.0
    return (s0 - survival_at(curve, age + horizon)) / s0
