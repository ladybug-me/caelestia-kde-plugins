"""Holt linear-trend smoothing (forecast) and a 1-D Kalman filter
(smoothing).

exponential-build-4 H (the uncertainty pass): the raw point estimates
below now carry their uncertainty one call away — `holt_with_uncertainty`
wraps `holt` with an iid-residual band built from the in-sample
one-step errors (the same honesty convention diagnostics/forecast.py
uses), and Kalman1D exposes `interval()`: the filter has always
tracked its own posterior variance `p`; it just never showed it. The
bare point projections remain for every existing caller (brief, timing)
and are labeled as point-only in their docstrings."""


def holt(series, alpha=0.5, beta=0.3, horizon=7):
    """Point projections only — NO uncertainty model on these numbers
    themselves; use `holt_with_uncertainty` for the residual band."""
    if not series:
        return []
    if len(series) == 1:
        return [series[0]] * horizon
    level, trend = series[0], series[1] - series[0]
    for x in series[1:]:
        prev_level = level
        level = alpha * x + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
    return [level + (h + 1) * trend for h in range(horizon)]


def holt_with_uncertainty(series, alpha=0.5, beta=0.3, horizon=7):
    """Holt projections WITH their uncertainty: the band is the
    in-sample one-step residual sigma (Holt's own smoothing errors)
    accumulated linearly over the horizon — an iid-residual INDICATION
    under the same convention as diagnostics/forecast.py, not a
    prediction interval from a fitted noise model. A single-point
    series has no residuals and says so."""
    point = holt(series, alpha, beta, horizon)
    if len(series) < 3:
        return {"projections": point, "sigma": None,
                "note": "fewer than 3 points: no residual sigma exists "
                        "to build a band from — point projections only"}
    # one-step in-sample residuals of the fitted Holt recursion
    level, trend = series[0], series[1] - series[0]
    residuals = []
    for x in series[1:]:
        pred = level + trend
        residuals.append(x - pred)
        prev_level = level
        level = alpha * x + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
    sigma = (sum(r * r for r in residuals) / len(residuals)) ** 0.5
    return {"projections": point,
            "sigma": round(sigma, 6),
            "band": [[round(v - sigma, 6), round(v + sigma, 6)]
                     for v in point],
            "note": "iid-residual indication: sigma is the in-sample "
                    "one-step smoothing error held flat over the "
                    "horizon; correlated noise is not modeled"}


class Kalman1D:
    """Random-walk model. q: process noise, r: measurement noise."""

    def __init__(self, q=0.01, r=0.5, x0=0.5, p0=1.0):
        self.q, self.r, self.x, self.p = q, r, x0, p0

    def update(self, z):
        self.p += self.q
        k = self.p / (self.p + self.r)
        self.x += k * (z - self.x)
        self.p *= (1 - k)
        return self.x

    def interval(self, z=1.0):
        """The filter's own posterior band: mean ± z·sqrt(p). The
        variance has been tracked all along — this just surfaces it
        (exponential-build-4 H)."""
        import math
        half = z * math.sqrt(max(self.p, 0.0))
        return [self.x - half, self.x + half]
