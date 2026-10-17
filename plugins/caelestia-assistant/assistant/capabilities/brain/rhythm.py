"""Periodicity in when you're active: day-of-week and hour-of-day
histograms, compared to a uniform baseline with a normalised deviation.

Not a hypothesis test — no p-value, no scipy — just enough signal to surface
a pattern like "you're consistently quieter on Tuesdays" or "most of your
activity is between 8 and 11pm", which bandit.HourBandit then acts on for
reminder timing while this module stays purely descriptive.
"""
from collections import Counter

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _deviation(counts, n_bins):
    total = sum(counts.values())
    if total == 0:
        return {}
    expected = total / n_bins
    denom = expected ** 0.5 or 1.0
    return {b: (counts.get(b, 0) - expected) / denom for b in range(n_bins)}


def day_of_week_pattern(weekday_list):
    """weekday_list: ints 0=Mon..6=Sun, e.g. from datetime.weekday()."""
    counts = Counter(weekday_list)
    dev = _deviation(counts, 7)
    return sorted(({"day": DAYS[d], "count": counts.get(d, 0), "z": round(dev.get(d, 0.0), 2)}
                   for d in range(7)), key=lambda r: r["z"])


def hour_of_day_pattern(hour_list):
    counts = Counter(hour_list)
    dev = _deviation(counts, 24)
    return sorted(({"hour": h, "count": counts.get(h, 0), "z": round(dev.get(h, 0.0), 2)}
                   for h in range(24)), key=lambda r: r["z"])


def notable(pattern, threshold=1.5):
    """Filter a pattern list down to bins that deviate meaningfully."""
    return [r for r in pattern if abs(r["z"]) >= threshold]
