"""genius.autodiff — dual-number forward-mode automatic differentiation
(no dependencies, tape-free Wengert form).

Wengert 1964, "A Simple Automatic Derivative Evaluation Program", Comm.
ACM 7(8) — the tape-free form that later literature calls forward-mode
AD with dual numbers. A Dual carries a value AND a full gradient vector
(one partial per input variable); every arithmetic operation propagates
both, using the chain rule per component:

    (u + u'·ε) ± (v + v'·ε)  ->  (u ± v) + (u' ± v')·ε
    (u + u'·ε) · (v + v'·ε)  ->  u·v + (u'·v + u·v')·ε
    u / v                    ->  u/v + (u'·v − u·v')/v² ·ε
    f(u + u'·ε)              ->  f(u) + f'(u)·u' ·ε   (f' known in closed form)

This is the general mechanism for propagating INPUT ERROR BARS through
ARBITRARY Python arithmetic — not just the closed-form expressions the
symbolic engine (mathengine.py) recognizes. Other genius modules import
it for the uncertainty pass (exponential-build-4 Group H): compute the
partials exactly with Duals, then sigma_f = sqrt(sum (partial·sigma_i)²)
by the standard independent-error propagation rule (the linearized
first-order form; the honest envelope: first-order only, correlations
are not modeled, and a function out of the supported table is an ERROR,
never a silently-constant argument).

Determinism: pure arithmetic, no RNG, no sampling — a derivative from a
Dual is exact to floating-point round-off (no finite-difference step to
choose, and the round-off honesty of THAT is the point).
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Sequence, Union

__all__ = ["Dual", "var", "const", "derivative", "jacobian", "propagate_error",
           "SUPPORTED_FUNCS", "AutodiffError"]


class AutodiffError(ValueError):
    """An unsupported operation on Dual numbers (never silently constant)."""


class Dual:
    """A dual number: value plus a gradient vector over n inputs.

    Arithmetic overloads mirror real numbers; mixing with plain numbers
    promotes them to constants (zero gradient). Comparison operators
    compare VALUES only (ordering is not differentiable and does not
    pretend to be)."""

    __slots__ = ("val", "grad")

    def __init__(self, val: float, grad: Sequence[float]) -> None:
        self.val = float(val)
        self.grad = tuple(float(g) for g in grad)

    # -- helpers -------------------------------------------------------

    def _coerce(self, other: Union["Dual", float, int]) -> "Dual":
        if isinstance(other, Dual):
            if len(other.grad) != len(self.grad):
                raise AutodiffError(
                    f"cannot mix Duals over {len(self.grad)} and "
                    f"{len(other.grad)} variables")
            return other
        return const(float(other), len(self.grad))


    # -- arithmetic ----------------------------------------------------

    def __add__(self, other):
        o = self._coerce(other)
        return Dual(self.val + o.val,
                    tuple(a + b for a, b in zip(self.grad, o.grad)))

    __radd__ = __add__

    def __sub__(self, other):
        o = self._coerce(other)
        return Dual(self.val - o.val,
                    tuple(a - b for a, b in zip(self.grad, o.grad)))

    def __rsub__(self, other):
        o = self._coerce(other)
        return o.__sub__(self)

    def __mul__(self, other):
        o = self._coerce(other)
        return Dual(self.val * o.val,
                    tuple(a * o.val + self.val * b
                          for a, b in zip(self.grad, o.grad)))

    __rmul__ = __mul__

    def __truediv__(self, other):
        o = self._coerce(other)
        if o.val == 0.0:
            raise ZeroDivisionError("dual division by zero value")
        inv = 1.0 / o.val
        val = self.val * inv
        # d(u/v) = (u'v − uv')/v²
        grad = tuple((a * o.val - self.val * b) * inv * inv
                     for a, b in zip(self.grad, o.grad))
        return Dual(val, grad)

    def __rtruediv__(self, other):
        o = self._coerce(other)
        return o.__truediv__(self)

    def __pow__(self, other):
        o = self._coerce(other)
        if o.val == 0.5:
            return sqrt(self)
        if o.grad == tuple(0.0 for _ in o.grad):
            return _pow_const(self, o.val)
        return exp(o * ln(self))  # general u^v = exp(v ln u)

    def __rpow__(self, other):
        o = self._coerce(other)
        return o.__pow__(self)

    def __neg__(self):
        return Dual(-self.val, tuple(-g for g in self.grad))

    def __pos__(self):
        return self

    def __abs__(self):
        # d|x| = sign(x)·x' (x = 0 is a kink: refused, not smoothed)
        if self.val == 0.0:
            raise AutodiffError("abs at exactly 0: not differentiable")
        s = 1.0 if self.val > 0 else -1.0
        return Dual(s * self.val, tuple(s * g for g in self.grad))

    # -- comparisons on values only ------------------------------------

    def __lt__(self, other):
        return self.val < self._coerce(other).val

    def __le__(self, other):
        return self.val <= self._coerce(other).val

    def __gt__(self, other):
        return self.val > self._coerce(other).val

    def __ge__(self, other):
        return self.val >= self._coerce(other).val

    def __eq__(self, other):
        try:
            return self.val == self._coerce(other).val
        except AutodiffError:
            return NotImplemented

    def __hash__(self):
        return hash(self.val)

    def __repr__(self):
        return f"Dual({self.val}, grad={self.grad})"


def _pow_const(u: Dual, c: float) -> Dual:
    if c == 0.0:
        return Dual(1.0, tuple(0.0 for _ in u.grad))
    if u.val < 0 and c != int(c):
        raise AutodiffError(
            f"negative base {u.val} with non-integer exponent {c}")
    if u.val == 0.0 and c < 1.0:
        raise AutodiffError(f"x^{c} not differentiable at x=0")
    val = u.val ** c
    return Dual(val, tuple(c * u.val ** (c - 1.0) * g for g in u.grad))


def const(value: float, n: int) -> Dual:
    """A constant over n variables: zero gradient."""
    return Dual(float(value), tuple(0.0 for _ in range(n)))


def var(value: float, index: int, n: int) -> Dual:
    """The i-th input variable over n: gradient one-hot at index."""
    if not (0 <= index < n):
        raise IndexError(f"variable index {index} out of range for n={n}")
    return Dual(float(value), tuple(1.0 if i == index else 0.0
                                    for i in range(n)))


# ---------------------------------------------------------------------------
# Composed-forward table: name -> u -> Dual(f(u.val), f'(u.val)*u.grad)
# (a dual application IS the forward pass and the chain rule in one;
# there is no separate "derivative slot" — that was the design error the
# round-trip tests caught in build-4: a dual only ever needs the
# composed form)
# ---------------------------------------------------------------------------

def _f_exp(u): return Dual(math.exp(u.val), tuple(math.exp(u.val) * g for g in u.grad))

def _f_ln(u):
    if u.val <= 0.0:
        raise AutodiffError(f"ln at {u.val}: outside the real domain")
    return Dual(math.log(u.val), tuple(g / u.val for g in u.grad))

def _f_log10(u):
    if u.val <= 0.0:
        raise AutodiffError(f"log10 at {u.val}: outside the real domain")
    return Dual(math.log10(u.val), tuple(g / (u.val * math.log(10.0)) for g in u.grad))

def _f_log2(u):
    if u.val <= 0.0:
        raise AutodiffError(f"log2 at {u.val}: outside the real domain")
    return Dual(math.log2(u.val), tuple(g / (u.val * math.log(2.0)) for g in u.grad))

def _f_sqrt(u):
    if u.val < 0.0:
        raise AutodiffError(f"sqrt at {u.val}: outside the real domain")
    if u.val == 0.0:
        raise AutodiffError("sqrt at exactly 0: not differentiable")
    return Dual(math.sqrt(u.val), tuple(g / (2.0 * math.sqrt(u.val)) for g in u.grad))

def _f_sin(u): return Dual(math.sin(u.val), tuple(math.cos(u.val) * g for g in u.grad))

def _f_cos(u): return Dual(math.cos(u.val), tuple(-math.sin(u.val) * g for g in u.grad))

def _f_tan(u):
    c = math.cos(u.val)
    return Dual(math.tan(u.val), tuple(g / (c * c) for g in u.grad))

def _f_asin(u):
    if not (-1.0 < u.val < 1.0):
        raise AutodiffError(f"asin at {u.val}: outside the open interval (-1, 1)")
    return Dual(math.asin(u.val), tuple(g / math.sqrt(1.0 - u.val * u.val) for g in u.grad))

def _f_acos(u):
    if not (-1.0 < u.val < 1.0):
        raise AutodiffError(f"acos at {u.val}: outside the open interval (-1, 1)")
    return Dual(math.acos(u.val), tuple(-g / math.sqrt(1.0 - u.val * u.val) for g in u.grad))

def _f_atan(u): return Dual(math.atan(u.val), tuple(g / (1.0 + u.val * u.val) for g in u.grad))

def _f_sinh(u): return Dual(math.sinh(u.val), tuple(math.cosh(u.val) * g for g in u.grad))

def _f_cosh(u): return Dual(math.cosh(u.val), tuple(math.sinh(u.val) * g for g in u.grad))

def _f_tanh(u):
    c = 1.0 - math.tanh(u.val) ** 2
    return Dual(math.tanh(u.val), tuple(c * g for g in u.grad))

def _f_exp2(u):
    v = 2.0 ** u.val
    return Dual(v, tuple(math.log(2.0) * v * g for g in u.grad))

def _f_cbrt(u):
    if u.val == 0.0:
        raise AutodiffError("cbrt at exactly 0: not differentiable")
    v = math.copysign(abs(u.val) ** (1.0 / 3.0), u.val)
    d = (1.0 / 3.0) * abs(u.val) ** (-2.0 / 3.0)
    return Dual(v, tuple(d * g for g in u.grad))


def _recip(u: Dual) -> Dual:
    return const(1.0, len(u.grad)) / u

Dual.reciprocal = _recip  # type: ignore[attr-defined]


#: name -> composed forward (the chain rule in one application)
SUPPORTED_FUNCS: Dict[str, Callable[[Dual], Dual]] = {
    "exp": _f_exp,
    "ln": _f_ln,
    "log": _f_ln,
    "log10": _f_log10,
    "log2": _f_log2,
    "sqrt": _f_sqrt,
    "sin": _f_sin,
    "cos": _f_cos,
    "tan": _f_tan,
    "asin": _f_asin,
    "acos": _f_acos,
    "atan": _f_atan,
    "sinh": _f_sinh,
    "cosh": _f_cosh,
    "tanh": _f_tanh,
    "exp2": _f_exp2,
    "cbrt": _f_cbrt,
}


def _apply(name: str, u: Dual) -> Dual:
    if name not in SUPPORTED_FUNCS:
        raise AutodiffError(
            f"{name!r} is not in the dual-number function table "
            f"(supported: {sorted(SUPPORTED_FUNCS)}) — refusing, "
            "never silently constant")
    return SUPPORTED_FUNCS[name](u)


# module-level table functions: exp(u), ln(u), sin(u), ... — each raises
# AutodiffError on anything outside the table (never silently constant)
exp = lambda u: _apply("exp", u)      # noqa: E731
ln = lambda u: _apply("ln", u)        # noqa: E731
log = lambda u: _apply("ln", u)       # noqa: E731
log10 = lambda u: _apply("log10", u)  # noqa: E731
log2 = lambda u: _apply("log2", u)    # noqa: E731
sqrt = lambda u: _apply("sqrt", u)    # noqa: E731
sin = lambda u: _apply("sin", u)      # noqa: E731
cos = lambda u: _apply("cos", u)      # noqa: E731
tan = lambda u: _apply("tan", u)      # noqa: E731
asin = lambda u: _apply("asin", u)    # noqa: E731
acos = lambda u: _apply("acos", u)    # noqa: E731
atan = lambda u: _apply("atan", u)    # noqa: E731
sinh = lambda u: _apply("sinh", u)    # noqa: E731
cosh = lambda u: _apply("cosh", u)    # noqa: E731
tanh = lambda u: _apply("tanh", u)    # noqa: E731
exp2 = lambda u: _apply("exp2", u)    # noqa: E731
cbrt = lambda u: _apply("cbrt", u)    # noqa: E731


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

def derivative(fn: Callable[[Dual], Dual], x: float) -> float:
    """d fn / dx at x: one Dual variable, exact partial (no step size)."""
    return fn(var(x, 0, 1)).grad[0]


def jacobian(fn: Callable[[List[Dual]], Sequence[Dual]],
             values: Sequence[float]) -> List[List[float]]:
    """The full Jacobian of a vector function at ``values``: row per
    output, column per input. ONE forward pass seeds every input at its
    own slot, so each output's gradient vector IS its Jacobian row
    (forward-mode's cost model, stated honestly: this many-outputs/
    few-inputs shape is the cheap direction)."""
    n = len(values)
    ins = [var(v, i, n) for i, v in enumerate(values)]
    outs = fn(ins)
    return [[float(o.grad[i]) for i in range(n)] for o in outs]


def propagate_error(fn: Callable[[List[Dual]], Union[Dual, Sequence[Dual]]],
                    values: Sequence[float],
                    sigmas: Sequence[float]) -> Dict[str, Any]:
    """Propagate independent input error bars through ``fn``.

    sigma_f² = Σ (∂f/∂x_i · sigma_i)² — the standard first-order,
    independent-errors rule (the linearization of the law of total
    variance; correlations are NOT modeled, and the result is labeled
    as such). Returns the function value, the one-sigma interval, and
    the per-input contribution breakdown so a caller can see WHICH
    input dominates the uncertainty — an honest report, not a scalar.
    """
    if len(values) != len(sigmas):
        raise ValueError("values and sigmas must have the same length")
    n = len(values)
    if any(s < 0 for s in sigmas):
        raise ValueError("sigmas must be non-negative (rejected, never clamped)")
    ins = [var(v, i, n) for i, v in enumerate(values)]
    out = fn(ins)
    outs: List[Dual] = ([out] if isinstance(out, Dual) else list(out))
    per_input: List[List[float]] = []
    for o in outs:
        contribs = [abs(o.grad[i] * sigmas[i]) for i in range(n)]
        per_input.append(contribs)
    total = [math.sqrt(sum(c * c for c in contribs)) for contribs in per_input]
    result: Dict[str, Any] = {
        "values": [float(o.val) for o in outs],
        "sigmas": total,
        "intervals": [[o.val - s, o.val + s]
                      for o, s in zip(outs, total)],
        "contributions": per_input,
        "method": "dual-number forward-mode AD (Wengert 1964), first-order "
                  "independent-error propagation; correlations not modeled",
    }
    return result
