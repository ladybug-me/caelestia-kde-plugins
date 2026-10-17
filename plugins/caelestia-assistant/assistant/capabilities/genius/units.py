"""genius.units — SI unit parsing and dimension-checked arithmetic.

A units engine in the same spirit as the rest of genius: deterministic,
stdlib-only, honest about what it refuses. It plugs into the meta
router's existing dispatch (the units domain branch and the math_eval
unit-ful interception — the same dispatcher pattern every other domain
uses).

What it supports:
- the seven SI base dimensions (m, kg, s, A, K, mol, cd) tracked as
  integer exponent vectors; SI prefixes (T..p incl. micro via µ and u);
  the coherent derived units (Hz, N, Pa, J, W, C, V, F, ohm, S, Wb, T,
  H, lm, lx, Bq, Gy, Sv, kat, rad, sr) and common accepted non-SI
  (min, h, day, g, t, L, eV, kWh, bar, atm, in, ft, mi, %);
- parsing quantities ("5 m", "9.81 m/s^2", "2.5 km*h^-1");
- dimension-checked arithmetic: + and - demand EQUAL dimensions (a
  mismatch is an explicit UnitsError — never silently coerced, the
  same reject-don't-clamp convention the settings layer holds for
  out-of-range values); * and / compose dimensions; ^ takes an integer
  power; the result reports SI units AND the left operand's unit;
- conversion between same-dimension units ("convert 5 m to cm");
- affine temperature units (degC, degF) support CONVERSION ONLY —
  arithmetic on affine scales is refused (offsets do not multiply).

References (the repo's citation convention): the SI dimension algebra,
prefixes and the affine temperature offsets follow the BIPM SI
Brochure (9th edition, 2019, chapters 1-3 and table 3). Everything is
deterministic bookkeeping over exact factors.

Pure: no I/O, no clock, no randomness. Errors are UnitsError with
user-facing messages; results are JSON-serialisable dicts.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

__all__ = ["UnitsError", "parse_quantity", "parse_unit", "convert",
           "evaluate", "handle", "looks_unitful", "DIMS"]

DIMS = ("m", "kg", "s", "A", "K", "mol", "cd")

_MISMATCH = ("dimensions differ: {a} vs {b} — refusing to silently "
             "coerce (rejected, never clamped)")


class UnitsError(ValueError):
    """A unit parse or dimension-check failure (user-facing message)."""


def _dims(**kw) -> Tuple[int, ...]:
    return tuple(kw.get(d, 0) for d in DIMS)


# name -> (factor to SI base unit, dims). "kg" is registered directly
# (the SI brochure's one prefixed base unit, treated as its own base).
_UNITS: Dict[str, Tuple[float, Tuple[int, ...]]] = {
    # SI base
    "m": (1.0, _dims(m=1)), "kg": (1.0, _dims(kg=1)),
    "s": (1.0, _dims(s=1)), "A": (1.0, _dims(A=1)),
    "K": (1.0, _dims(K=1)), "mol": (1.0, _dims(mol=1)),
    "cd": (1.0, _dims(cd=1)),
    # dimensionless accepted units
    "%": (0.01, _dims()), "rad": (1.0, _dims()), "sr": (1.0, _dims()),
    # coherent derived
    "Hz": (1.0, _dims(s=-1)), "N": (1.0, _dims(kg=1, m=1, s=-2)),
    "Pa": (1.0, _dims(kg=1, m=-1, s=-2)),
    "J": (1.0, _dims(kg=1, m=2, s=-2)),
    "W": (1.0, _dims(kg=1, m=2, s=-3)), "C": (1.0, _dims(A=1, s=1)),
    "V": (1.0, _dims(kg=1, m=2, s=-3, A=-1)),
    "F": (1.0, _dims(kg=-1, m=-2, s=4, A=2)),
    "ohm": (1.0, _dims(kg=1, m=2, s=-3, A=-2)),
    "S": (1.0, _dims(kg=-1, m=-2, s=3, A=2)),
    "Wb": (1.0, _dims(kg=1, m=2, s=-2, A=-1)),
    "T": (1.0, _dims(kg=1, s=-2, A=-1)),
    "H": (1.0, _dims(kg=1, m=2, s=-2, A=-2)),
    "lm": (1.0, _dims(cd=1)), "lx": (1.0, _dims(cd=1, m=-2)),
    "Bq": (1.0, _dims(s=-1)), "Gy": (1.0, _dims(m=2, s=-2)),
    "Sv": (1.0, _dims(m=2, s=-2)), "kat": (1.0, _dims(mol=1, s=-1)),
    # accepted non-SI (exact or standard-defined factors)
    "min": (60.0, _dims(s=1)), "h": (3600.0, _dims(s=1)),
    "day": (86400.0, _dims(s=1)),
    "g": (1e-3, _dims(kg=1)), "t": (1000.0, _dims(kg=1)),
    "L": (1e-3, _dims(m=3)),
    "eV": (1.602176634e-19, _dims(kg=1, m=2, s=-2)),
    "kWh": (3.6e6, _dims(kg=1, m=2, s=-2)),
    "bar": (1e5, _dims(kg=1, m=-1, s=-2)),
    "atm": (101325.0, _dims(kg=1, m=-1, s=-2)),
    "in": (0.0254, _dims(m=1)), "ft": (0.3048, _dims(m=1)),
    "mi": (1609.344, _dims(m=1)),
    # affine temperature (conversion ONLY; arithmetic refused)
    "degC": (1.0, _dims(K=1)), "degF": (1.0, _dims(K=1)),
}

_PREFIXES: Tuple[Tuple[str, float], ...] = (
    ("T", 1e12), ("G", 1e9), ("M", 1e6), ("k", 1e3), ("h", 1e2),
    ("da", 10.0), ("d", 0.1), ("c", 0.01), ("m", 1e-3),
    ("µ", 1e-6), ("u", 1e-6), ("n", 1e-9), ("p", 1e-12),
)

_AFFINE = ("degC", "degF")


def _fmt_dims(dims: Tuple[int, ...]) -> str:
    parts = []
    for name, exp in zip(DIMS, dims):
        if exp == 0:
            continue
        parts.append(name if exp == 1 else f"{name}^{exp}")
    return "*".join(parts) if parts else "dimensionless"


def _split_prefix(token: str) -> Tuple[str, float]:
    """Split one unit token into (base name, prefix factor). An exact
    registered name wins before any prefix split; prefixes are tried
    longest-first ("da" before "d") and only when the remainder IS a
    registered unit."""
    if token in _UNITS:
        return token, 1.0
    for prefix, factor in sorted(_PREFIXES, key=lambda p: -len(p[0])):
        if token.startswith(prefix) and len(token) > len(prefix):
            base = token[len(prefix):]
            if base in _UNITS:
                return base, factor
    raise UnitsError(f"unknown unit {token!r}")


_UNIT_TOKEN_RE = re.compile(r"[A-Za-zµ%]+|\^|[*/]|[-+]?\d+")


def parse_unit(token: str) -> Dict[str, Any]:
    """Parse a unit expression ("m", "km", "m/s^2", "km*h^-1") into
    {"factor", "dims", "name"} — the factor to SI base units and an
    integer exponent vector over DIMS. Raises UnitsError on anything
    unknown or ungrammatical."""
    cleaned = re.sub(r"\s+", "", token).replace("°", "deg") \
        .replace("Ω", "ohm")
    if not cleaned:
        return {"factor": 1.0, "dims": _dims(), "name": "1"}
    pieces = _UNIT_TOKEN_RE.findall(cleaned)
    if "".join(pieces) != cleaned:
        raise UnitsError(f"cannot parse unit {token!r}")
    factor = 1.0
    dims = _dims()
    i = 0
    divide = False
    expect_atom = True
    while i < len(pieces):
        piece = pieces[i]
        if piece in ("*", "/"):
            if expect_atom:
                raise UnitsError(f"cannot parse unit {token!r}")
            divide = piece == "/"
            expect_atom = True
            i += 1
            continue
        if not re.match(r"[A-Za-zµ%]", piece[0]):
            raise UnitsError(f"cannot parse unit {token!r}")
        base, pf = _split_prefix(piece)
        i += 1
        power = 1
        if i < len(pieces) and pieces[i] == "^":
            i += 1
            if i >= len(pieces) or not re.match(r"[-+]?\d", pieces[i]):
                raise UnitsError(f"cannot parse unit {token!r}")
            power = int(pieces[i])
            i += 1
        if divide:
            power = -power
        f, d = _UNITS[base]
        factor *= pf * (f ** power)
        dims = tuple(dims[k] + power * d[k] for k in range(len(DIMS)))
        divide = False
        expect_atom = False
    if expect_atom:
        raise UnitsError(f"cannot parse unit {token!r}")
    return {"factor": factor, "dims": dims, "name": cleaned}


_NUM = r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_QTY_RE = re.compile(rf"({_NUM})\s*([A-Za-zµ°%][\wµ°%/^.\-]*)")
_BARE_NUM_RE = re.compile(rf"({_NUM})$")


def parse_quantity(text: str) -> Tuple[float, Dict[str, Any]]:
    """Parse one quantity ("5 m", "9.81 m/s^2", "2 km") -> (value,
    unit dict). A bare number is dimensionless with factor 1."""
    text = text.strip()
    m = _QTY_RE.fullmatch(text)
    if m:
        return float(m.group(1)), parse_unit(m.group(2))
    m = _BARE_NUM_RE.fullmatch(text)
    if m:
        return float(m.group(1)), {"factor": 1.0, "dims": _dims(),
                                   "name": "1"}
    raise UnitsError(f"cannot parse quantity {text!r}")


def convert(value: float, src: Dict[str, Any], dst: Dict[str, Any]) -> float:
    """Convert one value between same-dimension units. A dimension
    mismatch is an explicit UnitsError — never silently coerced."""
    if src["dims"] != dst["dims"]:
        raise UnitsError(_MISMATCH.format(a=_fmt_dims(src["dims"]),
                                          b=_fmt_dims(dst["dims"])))
    if src["name"] in _AFFINE or dst["name"] in _AFFINE:
        return _convert_affine(value, src["name"], dst["name"])
    return value * src["factor"] / dst["factor"]


def _convert_affine(value: float, src: str, dst: str) -> float:
    """Affine temperature conversion through kelvin (BIPM SI Brochure
    9th ed., table 3: degC offset 273.15; degF exact definition)."""
    if src == dst:
        return value
    if src == "degC":
        kelvin = value + 273.15
    elif src == "degF":
        kelvin = (value - 32.0) * 5.0 / 9.0 + 273.15
    else:  # src is kelvin (or any linear same-dims unit)
        kelvin = value
    if dst == "degC":
        return kelvin - 273.15
    if dst == "degF":
        return (kelvin - 273.15) * 9.0 / 5.0 + 32.0
    return kelvin


# ---------------------------------------------------------------------------
# Dimension-checked arithmetic over quantity expressions.
# ---------------------------------------------------------------------------

_CONV_RE = re.compile(
    rf"^(?:convert\s+)?({_NUM}\s*[A-Za-zµ°%][\wµ°%/^.\-]*|{_NUM})"
    rf"\s+(?:to|in)\s+([A-Za-zµ°%][A-Za-zµ°%/^0-9\-]*)$", re.I)

# In expressions, numbers are UNSIGNED (a '-' is always the operator;
# unary minus is handled in the parser), so "5 m - 200 cm" cannot have
# its operator swallowed into a signed literal.
_EXPR_TOKEN_RE = re.compile(
    rf"(\d+(?:\.\d+)?(?:[eE]\d+)?)\s*([A-Za-zµ°%][\wµ°%/^.]*)"
    rf"|(\d+(?:\.\d+)?(?:[eE]\d+)?)|([+\-*/^()])|\s+")


def _parse_expr(text: str) -> Dict[str, Any]:
    """Precedence-climbing evaluation over quantities. + and - require
    equal dimensions (the left unit survives for display); * and /
    compose dims and display names; ^ takes an integer power."""
    tokens = [t for t in _EXPR_TOKEN_RE.findall(text)
              if any(t)]  # drop pure-whitespace tokens
    pos = [0]

    def peek() -> Tuple[str, str, str, str]:
        if pos[0] >= len(tokens):
            return ("", "", "", "")
        return tokens[pos[0]]

    def primary() -> Dict[str, Any]:
        if pos[0] >= len(tokens):
            raise UnitsError(f"cannot parse expression {text!r}")
        v, u, bare, op = peek()
        if op == "(":
            pos[0] += 1
            node = expr()
            if peek()[3] != ")":
                raise UnitsError(f"cannot parse expression {text!r}")
            pos[0] += 1
            return node
        if op == "-":  # unary minus
            pos[0] += 1
            node = primary()
            return {**node, "si": -node["si"]}
        if v and u:
            pos[0] += 1
            unit = parse_unit(u)
            return {"si": float(v) * unit["factor"], "dims": unit["dims"],
                    "name": unit["name"]}
        if bare:
            pos[0] += 1
            return {"si": float(bare), "dims": _dims(), "name": "1"}
        raise UnitsError(f"cannot parse expression {text!r}")

    def power() -> Dict[str, Any]:
        node = primary()
        if peek()[3] == "^":
            pos[0] += 1
            exponent = primary()
            if exponent["dims"] != _dims() or \
                    float(exponent["si"]).is_integer() is False:
                raise UnitsError(
                    "exponent must be a dimensionless integer")
            n = int(exponent["si"])
            dims = tuple(n * d for d in node["dims"])
            node = {"si": node["si"] ** n, "dims": dims,
                    "name": node["name"] if n == 1
                    else f"({node['name']})^{n}"}
        return node

    def term() -> Dict[str, Any]:
        node = power()
        while peek()[3] in ("*", "/"):
            op = peek()[3]
            pos[0] += 1
            rhs = power()
            if node["name"] in _AFFINE or rhs["name"] in _AFFINE:
                raise UnitsError(
                    "arithmetic on affine temperature scales (degC/degF) "
                    "is refused — offsets do not multiply; convert to K "
                    "first")
            dims = tuple(node["dims"][k] + (1 if op == "*" else -1)
                         * rhs["dims"][k] for k in range(len(DIMS)))
            si = node["si"] * rhs["si"] if op == "*" else \
                node["si"] / rhs["si"]
            name = _compose_names(node["name"], rhs["name"], op)
            node = {"si": si, "dims": dims, "name": name}
        return node

    def expr() -> Dict[str, Any]:
        node = term()
        while peek()[3] in ("+", "-"):
            op = peek()[3]
            pos[0] += 1
            rhs = term()
            if node["dims"] != rhs["dims"]:
                raise UnitsError(_MISMATCH.format(
                    a=_fmt_dims(node["dims"]), b=_fmt_dims(rhs["dims"])))
            if node["name"] in _AFFINE or rhs["name"] in _AFFINE:
                raise UnitsError(
                    "arithmetic on affine temperature scales (degC/degF) "
                    "is refused — convert to K first")
            si = node["si"] + rhs["si"] if op == "+" else \
                node["si"] - rhs["si"]
            # + and - keep the LEFT operand's unit (the display survives;
            # SI value is the truth either way)
            node = {"si": si, "dims": node["dims"], "name": node["name"]}
        return node

    result = expr()
    if pos[0] != len(tokens):
        raise UnitsError(f"cannot parse expression {text!r}")
    return result


def _compose_names(left: str, right: str, op: str) -> str:
    """Display-name composition that never gets uglier than it must:
    dimensionless "1" names vanish; everything else concatenates."""
    if left == "1":
        return right if op == "*" else f"1/{right}"
    if right == "1":
        return left
    return f"{left}{op}{right}"


def evaluate(text: str) -> Dict[str, Any]:
    """Evaluate one unit expression: conversion ("convert 5 m to cm",
    "3 km in m") or dimension-checked arithmetic ("3 km + 200 m",
    "9.81 m/s^2 * 2 s"). Returns a JSON-serialisable dict carrying the
    result in SI base units AND in the display unit, plus the
    dimension; raises UnitsError on mismatches (never coerces)."""
    expr_text = text.strip().rstrip("?").strip()
    conv = _CONV_RE.match(expr_text)
    if conv:
        value, unit = parse_quantity(conv.group(1))
        dst = parse_unit(conv.group(2))
        out = convert(value, unit, dst)
        return {"op": "convert", "value": round(out, 9),
                "unit": dst["name"],
                "dimension": _fmt_dims(dst["dims"]),
                "input": {"value": value, "unit": unit["name"]}}
    node = _parse_expr(expr_text)
    result = {
        "op": "evaluate",
        "si_value": round(node["si"], 9),
        "si_unit": _fmt_dims(node["dims"]) if node["dims"] != _dims()
        else "1",
        "dimension": _fmt_dims(node["dims"]),
    }
    # a display pair rides along ONLY when the surviving unit resolves
    # to a known one (registered base/derived, or prefix+base like km —
    # never a made-up composed name)
    factor = _display_factor(node["name"])
    if factor is not None:
        result["value"] = round(node["si"] / factor, 9)
        result["unit"] = node["name"]
    return result


def _display_factor(name: str) -> Optional[float]:
    """The SI factor of a display unit name ("km", "m", "N"), or None
    for composed/unregistered names."""
    try:
        base, pf = _split_prefix(name)
    except UnitsError:
        return None
    return pf * _UNITS[base][0]


def looks_unitful(text: str) -> bool:
    """True when the text carries a number+registered-unit shape (used
    by the meta router to intercept BEFORE plain arithmetic). Built
    from the registered unit names, longest-first, so "km" wins over
    "m", "mol" never splits, and a trailing letter ("5 inches") is
    refused rather than half-matched. Bare "%" is deliberately EXCLUDED
    here: "15% of 80" belongs to the math engine's percent path, not to
    dimension algebra."""
    tokens = sorted((t for t in _UNITS if t != "%"), key=len, reverse=True)
    names = "|".join(re.escape(t) for t in tokens)
    return bool(re.search(
        rf"\d(?:\.\d+)?\s*(?:{names})(?![A-Za-z])"
        rf"(?:\s*[/^]\s*[A-Za-z0-9]+)?", text))


def handle(text: str) -> Dict[str, Any]:
    """The meta-router entry: one unit-shaped request in, one result
    dict out (the same dispatch contract as every other domain)."""
    return evaluate(text)


# ---------------------------------------------------------------------------
# Uncertainty pass (exponential-build-4 H): measured quantities carry
# error bars, and a unit conversion is arithmetic on the measurement —
# so the bar converts with it. Group B's dual-number propagation
# (genius/autodiff.propagate_error) does the calculus; this wrapper
# keeps the dimensional discipline (a mismatch is still an error).
# ---------------------------------------------------------------------------

def convert_with_uncertainty(value: float, sigma: float,
                             src_name: str, dst_name: str) -> Dict[str, Any]:
    """Convert a MEASUREMENT and its one-sigma bar between same-dimension
    units. sigma propagates through the (exact) conversion factor by the
    first-order rule via dual numbers — for the multiplicative and
    affine temperature factors here the first-order form is EXACT up to
    the arithmetic, which is worth saying: the interval is honest, not
    approximate-by-convenience. A dimension mismatch is still a
    UnitsError; a negative sigma is refused, never clamped."""
    from .autodiff import propagate_error

    src = parse_unit(src_name)
    dst = parse_unit(dst_name)
    if src["dims"] != dst["dims"]:
        raise UnitsError(_MISMATCH.format(a=_fmt_dims(src["dims"]),
                                          b=_fmt_dims(dst["dims"])))
    if sigma < 0:
        raise UnitsError("sigma must be non-negative (rejected, never "
                         "clamped)")

    def measurement(xs):
        v = xs[0]
        if src["name"] in _AFFINE or dst["name"] in _AFFINE:
            if src["name"] == "degC":
                converted = v + 273.15
            elif src["name"] == "degF":
                converted = (v - 32.0) * 5.0 / 9.0 + 273.15
            else:
                converted = v
            if dst["name"] == "degC":
                converted = converted - 273.15
            elif dst["name"] == "degF":
                converted = (converted - 273.15) * 9.0 / 5.0 + 32.0
            return converted
        return v * src["factor"] / dst["factor"]

    result = propagate_error(measurement, [float(value)], [float(sigma)])
    return {"value": result["values"][0],
            "sigma": result["sigmas"][0],
            "interval": result["intervals"][0],
            "unit": dst_name,
            "method": "dual-number forward-mode propagation through the "
                      "exact conversion factor (genius/autodiff.py)"}
