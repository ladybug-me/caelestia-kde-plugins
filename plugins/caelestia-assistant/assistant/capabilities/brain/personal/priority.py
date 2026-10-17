"""Priority scoring: a logistic model over (urgency, importance, quickness).

Starts with hand-set weights so ranking is sensible before any data exists, then
fit() refines them from the user's approve/reject decisions on priority
proposals. score() is always a probability in (0, 1), so ranking is monotone.
"""
import math

DEFAULT_W = [2.0, 1.2, 0.8]
DEFAULT_B = -2.0


def urgency(days_left):
    if days_left is None:
        return 0.3
    if days_left < 0:
        return 1.0
    return math.exp(-days_left / 7.0)


def quickness(effort_min):
    return 1.0 / (1.0 + max(effort_min, 0) / 60.0)


def features(task):
    return [urgency(task.get("deadline_days")),
            float(task.get("importance", 0.5)),
            quickness(task.get("effort_min", 30))]


def _sigmoid(z):
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


class PriorityModel:
    def __init__(self, w=None, b=None):
        self.w = list(w) if w is not None else list(DEFAULT_W)
        self.b = DEFAULT_B if b is None else b

    def score(self, x):
        return _sigmoid(sum(wi * xi for wi, xi in zip(self.w, x)) + self.b)

    def fit(self, examples, lr=0.2, epochs=300, l2=0.01):
        """examples: [(features, label 0/1)]. Plain batch gradient descent."""
        if not examples:
            return self
        n = len(examples)
        for _ in range(epochs):
            gw = [0.0] * len(self.w)
            gb = 0.0
            for x, y in examples:
                err = self.score(x) - y
                for i, xi in enumerate(x):
                    gw[i] += err * xi
                gb += err
            self.w = [wi - lr * (g / n + l2 * wi) for wi, g in zip(self.w, gw)]
            self.b -= lr * gb / n
        return self

    def to_dict(self):
        return {"w": self.w, "b": self.b}

    @classmethod
    def from_dict(cls, d):
        return cls(d.get("w"), d.get("b"))
