"""Thompson sampling over hour-of-day: when do you actually act on reminders?

Each of 24 hours is a Beta arm. Acting on a reminder at hour h is a success,
ignoring it is a failure. choose() samples every allowed arm and returns the
best draw, so exploration shrinks as evidence accumulates.
"""
import random


class HourBandit:
    def __init__(self, alpha=None, beta=None):
        self.alpha = list(alpha) if alpha else [1.0] * 24
        self.beta = list(beta) if beta else [1.0] * 24

    def choose(self, allowed=None, rng=None):
        rng = rng or random
        hours = list(range(24)) if allowed is None else list(allowed)
        return max(hours, key=lambda h: rng.betavariate(self.alpha[h], self.beta[h]))

    def reward(self, hour, acted):
        if acted:
            self.alpha[hour] += 1.0
        else:
            self.beta[hour] += 1.0

    def to_dict(self):
        return {"alpha": self.alpha, "beta": self.beta}

    @classmethod
    def from_dict(cls, d):
        return cls(d.get("alpha"), d.get("beta"))
