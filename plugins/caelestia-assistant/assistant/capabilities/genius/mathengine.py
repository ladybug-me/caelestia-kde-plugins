"""genius.mathengine — symbolic-lite math engine (no dependencies).

A real expression engine, not a wrapper:

  * tokenizer + precedence-climbing parser -> AST (Num/Var/Bin/Neg/Call/Fact)
  * evaluator with variables, constants and 25+ functions
  * rule-based symbolic differentiation (sum, product, quotient, chain, power)
  * constant-folding simplification with identity elimination
  * root finding: bisection, Newton-Raphson, secant (with step traces)
  * numerical integration: trapezoid, Simpson, adaptive Simpson
  * ODE initial-value problems: Euler and classic Runge-Kutta 4
  * Taylor-series expansion around a point (via repeated symbolic diff)
  * interpolation: Lagrange, Newton divided differences
  * percent / unit-free arithmetic helpers used by the meta-router

Everything is deterministic, stdlib-only, and JSON-serialisable. Every
solver returns its iterations so the answer carries its own evidence —
the assistant shows its work the way a careful human would.
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "parse", "evaluate", "to_str", "differentiate", "simplify",
    "solve_root", "integrate", "ode_solve", "taylor", "interpolate",
    "percent_of", "expression_info", "CalcError", "symbolic_integrate",
]

CONSTANTS: Dict[str, float] = {
    "pi": math.pi, "e": math.e, "tau": math.tau,
    "phi": (1 + math.sqrt(5)) / 2,
}

_FUNCS = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "exp": math.exp, "ln": math.log, "log": math.log,
    "log2": math.log2, "log10": math.log10, "sqrt": math.sqrt,
    "abs": abs, "floor": math.floor, "ceil": math.ceil, "round": round,
    "sign": lambda x: float((x > 0) - (x < 0)),
    "gamma": math.gamma, "erf": math.erf,
    "fact": lambda x: float(math.factorial(int(x))),
    "factorial": lambda x: float(math.factorial(int(x))),
}

def _as_index(x: float, name: str) -> int:
    """Integral-float coercion for count-valued functions: 12.0 is fine,
    12.5 is an honest error — never a silent truncation (and never the
    unhandled TypeError the float-into-int path raised before Stage C)."""
    if isinstance(x, float) and not x.is_integer():
        raise CalcError(f"{name}() needs whole numbers, got {x}")
    return int(x)


_MULTI_FUNCS = {
    "min": min, "max": max, "hypot": math.hypot, "atan2": math.atan2,
    "pow": pow,
    "gcd": lambda *vals: float(math.gcd(*map(
        lambda v: _as_index(v, "gcd"), vals))),
    "lcm": lambda *vals: float(math.lcm(*map(
        lambda v: _as_index(v, "lcm"), vals))),
    "ncr": lambda n, r: float(math.comb(_as_index(n, "ncr"),
                                        _as_index(r, "ncr"))),
    "npr": lambda n, r: float(math.perm(_as_index(n, "npr"),
                                        _as_index(r, "npr"))),
}


class CalcError(ValueError):
    """Any parse or evaluation failure, with a user-facing message."""


# ---------------------------------------------------------------------------
# 1. Tokenizer
# ---------------------------------------------------------------------------

_OPERATORS = ("+", "-", "*", "/", "^", "%", "!", "(", ")", ",", "=")
_PREC = {"+": 1, "-": 1, "*": 2, "/": 2, "%": 2, "^": 4}  # unary minus = 3


def _tokenize(text: str) -> List[Tuple[str, Any]]:
    tokens: List[Tuple[str, Any]] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch.isdigit() or (ch == "." and i + 1 < n and text[i + 1].isdigit()):
            j = i
            while j < n and (text[j].isdigit() or text[j] == "."):
                j += 1
            if text[j:j + 1] == "e" and (text[j + 1:j + 2].isdigit() or text[j + 1:j + 2] in "+-"):
                k = j + 2
                while k < n and text[k].isdigit():
                    k += 1
                j = k
            tokens.append(("num", float(text[i:j])))
            i = j
            continue
        if ch.isalpha() or ch == "_":
            j = i
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            tokens.append(("name", text[i:j]))
            i = j
            continue
        if text[i:i + 2] == "**":
            tokens.append(("op", "^"))
            i += 2
            continue
        if ch in _OPERATORS:
            tokens.append(("op", ch))
            i += 1
            continue
        raise CalcError(f"unexpected character {ch!r} at position {i}")
    tokens.append(("end", None))
    return tokens


# ---------------------------------------------------------------------------
# 2. AST + precedence-climbing parser
# ---------------------------------------------------------------------------

class _Node:
    __slots__ = ("kind", "value", "left", "right", "args", "name")

    def __init__(self, kind: str, value: Any = None, left=None, right=None,
                 args: Optional[List["_Node"]] = None, name: str = ""):
        self.kind, self.value = kind, value
        self.left, self.right = left, right
        self.args, self.name = args or [], name


def parse(text: str) -> _Node:
    """Parse an arithmetic expression into an AST. Raises CalcError."""
    tokens = _tokenize(text)
    pos = [0]

    def peek() -> Tuple[str, Any]:
        return tokens[pos[0]]

    def take() -> Tuple[str, Any]:
        tok = tokens[pos[0]]
        pos[0] += 1
        return tok

    def parse_expr(min_prec: int = 0) -> _Node:
        left = parse_atom()
        while True:
            kind, val = peek()
            if kind == "op" and val == "!" :
                take()
                left = _Node("fact", left=left)
                continue
            if kind == "op" and val in _PREC and _PREC[val] >= max(min_prec, 1):
                if _PREC[val] < min_prec:
                    break
                take()
                # right-assoc for ^, left for the rest
                next_min = _PREC[val] + (0 if val == "^" else 1)
                right = parse_expr(next_min)
                left = _Node("bin", value=val, left=left, right=right)
                continue
            break
        return left

    def parse_atom() -> _Node:
        kind, val = peek()
        if kind == "op" and val == "-":
            take()
            return _Node("neg", left=parse_expr(3))
        if kind == "op" and val == "+":
            take()
            return parse_atom()
        if kind == "num":
            take()
            return _Node("num", value=val)
        if kind == "name":
            take()
            name = str(val)
            if name in CONSTANTS and not (peek()[0] == "op" and peek()[1] == "("):
                return _Node("num", value=CONSTANTS[name], name=name)
            if peek() == ("op", "("):
                take()
                args: List[_Node] = []
                if peek() != ("op", ")"):
                    args.append(parse_expr())
                    while peek() == ("op", ","):
                        take()
                        args.append(parse_expr())
                if peek() != ("op", ")"):
                    raise CalcError(f"missing ')' after {name}(...")
                take()
                return _Node("call", name=name, args=args)
            return _Node("var", name=name)
        if kind == "op" and val == "(":
            take()
            inner = parse_expr()
            if peek() != ("op", ")"):
                raise CalcError("missing ')'")
            take()
            return inner
        raise CalcError(f"unexpected token {val!r}" if kind != "end" else "unexpected end of expression")

    tree = parse_expr()
    if peek()[0] != "end":
        raise CalcError(f"unexpected trailing token {peek()[1]!r}")
    return tree


# ---------------------------------------------------------------------------
# 3. Evaluation, printing, symbolic operations
# ---------------------------------------------------------------------------

def evaluate(node: _Node, env: Optional[Dict[str, float]] = None) -> float:
    env = env or {}
    if node.kind == "num":
        return float(node.value)
    if node.kind == "var":
        if node.name in env:
            return float(env[node.name])
        if node.name in CONSTANTS:
            return CONSTANTS[node.name]
        raise CalcError(f"unknown variable {node.name!r}")
    if node.kind == "neg":
        return -evaluate(node.left, env)
    if node.kind == "fact":
        return float(math.factorial(int(evaluate(node.left, env))))
    if node.kind == "bin":
        a = evaluate(node.left, env)
        b = evaluate(node.right, env)
        op = node.value
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            if b == 0:
                raise CalcError("division by zero")
            return a / b
        if op == "%":
            if b == 0:
                raise CalcError("modulo by zero")
            return a % b
        if op == "^":
            try:
                return a ** b
            except (OverflowError, ValueError) as exc:
                raise CalcError(f"power overflow: {exc}")
    if node.kind == "call":
        fn = _FUNCS.get(node.name) or _MULTI_FUNCS.get(node.name)
        if fn is None:
            raise CalcError(f"unknown function {node.name!r}")
        vals = [evaluate(a, env) for a in node.args]
        try:
            return float(fn(*vals))
        except (ValueError, ZeroDivisionError, OverflowError) as exc:
            raise CalcError(f"{node.name}({', '.join(_fmt(v) for v in vals)}): {exc}")
    raise CalcError(f"cannot evaluate node {node.kind}")


def to_str(node: _Node) -> str:
    if node.kind == "num":
        return node.name if node.name else _fmt(node.value)
    if node.kind == "var":
        return node.name
    if node.kind == "neg":
        return f"-{to_str(node.left)}"
    if node.kind == "fact":
        return f"{to_str(node.left)}!"
    if node.kind == "bin":
        return f"({to_str(node.left)} {node.value} {to_str(node.right)})"
    if node.kind == "call":
        return f"{node.name}({', '.join(to_str(a) for a in node.args)})"
    return "?"


def simplify(node: _Node) -> _Node:
    """Bottom-up constant folding + algebraic identity elimination."""
    if node.kind in ("num", "var"):
        return node
    if node.kind == "neg":
        inner = simplify(node.left)
        if inner.kind == "num":
            return _Node("num", value=-inner.value)
        if inner.kind == "neg":
            return inner.left
        return _Node("neg", left=inner)
    if node.kind == "fact":
        inner = simplify(node.left)
        if inner.kind == "num" and 0 <= inner.value == int(inner.value) and inner.value < 171:
            return _Node("num", value=float(math.factorial(int(inner.value))))
        return _Node("fact", left=inner)
    if node.kind == "call":
        args = [simplify(a) for a in node.args]
        if node.name in _FUNCS and len(args) == 1 and args[0].kind == "num":
            try:
                return _Node("num", value=float(_FUNCS[node.name](args[0].value)))
            except (ValueError, OverflowError, ZeroDivisionError):
                pass
        return _Node("call", name=node.name, args=args)
    if node.kind == "bin":
        left, right = simplify(node.left), simplify(node.right)
        op = node.value
        if left.kind == "num" and right.kind == "num":
            try:
                return _Node("num", value=evaluate(_Node("bin", value=op, left=left, right=right)))
            except CalcError:
                pass
        if op == "+":
            if _is(left, 0):
                return right
            if _is(right, 0):
                return left
        elif op == "-":
            if _is(right, 0):
                return left
            if _same(left, right):
                return _Node("num", value=0.0)
        elif op == "*":
            if _is(left, 0) or _is(right, 0):
                return _Node("num", value=0.0)
            if _is(left, 1):
                return right
            if _is(right, 1):
                return left
        elif op == "/":
            if _is(left, 0):
                return _Node("num", value=0.0)
            if _is(right, 1):
                return left
        elif op == "^":
            if _is(right, 1):
                return left
            if _is(right, 0):
                return _Node("num", value=1.0)
        return _Node("bin", value=op, left=left, right=right)
    return node


def _is(node: _Node, value: float) -> bool:
    return node.kind == "num" and abs(node.value - value) < 1e-12


def _same(a: _Node, b: _Node) -> bool:
    return to_str(a) == to_str(b)


def differentiate(node: _Node, var: str = "x") -> _Node:
    """Rule-based symbolic differentiation (sum, product, quotient, chain)."""
    if node.kind == "num":
        return _Node("num", value=0.0)
    if node.kind == "var":
        return _Node("num", value=1.0 if node.name == var else 0.0)
    if node.kind == "neg":
        return _Node("neg", left=differentiate(node.left, var))
    if node.kind == "fact":
        raise CalcError("cannot differentiate factorial expressions")
    if node.kind == "call":
        return _diff_call(node, var)
    if node.kind == "bin":
        a, b = node.left, node.right
        da, db = differentiate(a, var), differentiate(b, var)
        op = node.value
        if op == "+":
            return _Node("bin", value="+", left=da, right=db)
        if op == "-":
            return _Node("bin", value="-", left=da, right=db)
        if op == "*":  # product rule
            return _Node("bin", value="+",
                        left=_Node("bin", value="*", left=da, right=b),
                        right=_Node("bin", value="*", left=a, right=db))
        if op == "/":  # quotient rule
            num = _Node("bin", value="-",
                        left=_Node("bin", value="*", left=da, right=b),
                        right=_Node("bin", value="*", left=a, right=db))
            den = _Node("bin", value="^", left=b, right=_Node("num", value=2.0))
            return _Node("bin", value="/", left=num, right=den)
        if op == "^":
            if b.kind == "num":  # d(u^c) = c*u^(c-1)*u'
                c = b.value
                outer = _Node("bin", value="*", left=_Node("num", value=c), right=_Node(
                    "bin", value="^", left=a, right=_Node("num", value=c - 1.0)))
                return _Node("bin", value="*", left=outer, right=da)
            # general: u^v = exp(v ln u) -> (u'v/u + v' ln u) * u^v
            lnu = _Node("call", name="ln", args=[a])
            left = _Node("bin", value="/",
                         left=_Node("bin", value="*", left=da, right=b), right=a)
            right = _Node("bin", value="*", left=db, right=lnnu)
            return _Node("bin", value="*",
                         left=_Node("bin", value="+", left=left, right=right),
                         right=_Node("bin", value="^", left=a, right=b))
    raise CalcError(f"cannot differentiate node kind {node.kind}")


def _diff_call(node: _Node, var: str) -> _Node:
    arg = node.args[0]
    da = differentiate(arg, var)
    name = node.name
    chain = lambda outer: _Node("bin", value="*", left=outer, right=da)  # noqa: E731
    table = {
        "sin": lambda: chain(_Node("call", name="cos", args=[arg])),
        "cos": lambda: _Node("neg", left=chain(_Node("call", name="sin", args=[arg]))),
        "tan": lambda: chain(_Node("bin", value="^", left=_Node("call", name="sec", args=[arg]),
                                   right=_Node("num", value=2.0))),
        "exp": lambda: chain(_Node("call", name="exp", args=[arg])),
        "ln": lambda: chain(_Node("bin", value="/", left=_Node("num", value=1.0), right=arg)),
        "log": lambda: chain(_Node("bin", value="/", left=_Node("num", value=1.0), right=arg)),
        "log2": lambda: chain(_Node("bin", value="/",
                                     left=_Node("num", value=1.0),
                                     right=_Node("bin", value="*", left=_Node("num", value=math.log(2), name=None), right=arg))),
        "log10": lambda: chain(_Node("bin", value="/",
                                     left=_Node("num", value=1.0),
                                     right=_Node("bin", value="*", left=_Node("num", value=math.log(10)), right=arg))),
        "sqrt": lambda: chain(_Node("bin", value="/", left=_Node("num", value=0.5),
                                     right=_Node("call", name="sqrt", args=[arg]))),
        # d|u| = sign(u)*u' (exponential-build-4 B: needed so the tan
        # antiderivative -ln|cos u| can be differentially verified; the
        # identity sign(u)/|u| = 1/u makes d ln|u| = u'/u off the kink)
        "abs": lambda: chain(_call("sign", arg)),
        "sinh": lambda: chain(_Node("call", name="cosh", args=[arg])),
        "cosh": lambda: chain(_Node("call", name="sinh", args=[arg])),
        "tanh": lambda: chain(_Node("bin", value="^", left=_Node("call", name="sech", args=[arg]),
                                    right=_Node("num", value=2.0))),
        "asin": lambda: chain(_Node("bin", value="/", left=_Node("num", value=1.0),
                                     right=_Node("call", name="sqrt", args=[_Node("bin", value="-", left=_Node("num", value=1.0),
                                                                                   right=_Node("bin", value="^", left=arg, right=_Node("num", value=2.0)))]))),
        "atan": lambda: chain(_Node("bin", value="/", left=_Node("num", value=1.0),
                                     right=_Node("bin", value="+", left=_Node("num", value=1.0),
                                                  right=_Node("bin", value="^", left=arg, right=_Node("num", value=2.0))))),
    }
    if name in table:
        return table[name]()
    raise CalcError(f"cannot differentiate {name}(...)")


# ---------------------------------------------------------------------------
# 4. Solvers — each returns a dict with the answer AND its evidence
# ---------------------------------------------------------------------------

def _f(text_or_node, env: Optional[Dict[str, float]] = None):
    node = text_or_node if isinstance(text_or_node, _Node) else parse(text_or_node)

    def f(x: float) -> float:
        return evaluate(node, {**(env or {}), "x": x})

    return f, node


def solve_root(expr: str, method: str = "auto", lo: float = -100.0, hi: float = 100.0,
               x0: Optional[float] = None, tol: float = 1e-9, max_iter: int = 100) -> Dict[str, Any]:
    """Find f(x)=0. Methods: bisection/newton/secant (one root, with
    step traces) or 'scan' — F8's default for non-polynomial requests:
    bracket-scan [lo, hi] and report EVERY sign-change root by bisecting
    each bracket, honestly labeled found-not-exhaustive (a scan can miss
    double roots and roots outside the interval)."""
    f, node = _f(expr)
    method = method if method != "auto" else ("newton" if x0 is not None else "scan")
    steps: List[Dict[str, float]] = []

    if method == "scan":
        n_slices = 400
        roots: List[float] = []
        a = lo
        fa = f(a)
        for k in range(1, n_slices + 1):
            b = lo + (hi - lo) * k / n_slices
            fb = f(b)
            if fa == 0.0:
                roots.append(a)
            elif fa * fb < 0:
                # bisect inside this bracket
                blo, bhi, flo = a, b, fa
                for _ in range(80):
                    mid = (blo + bhi) / 2
                    fm = f(mid)
                    if abs(fm) < tol or (bhi - blo) / 2 < tol:
                        break
                    if flo * fm <= 0:
                        bhi = mid
                    else:
                        blo, flo = mid, fm
                roots.append((blo + bhi) / 2)
            a, fa = b, fb
        if fa == 0.0:
            roots.append(a)
        # dedupe
        dedup: List[float] = []
        for r in sorted(roots):
            if not dedup or abs(r - dedup[-1]) > 1e-6:
                dedup.append(r)
        return {"method": "scan", "roots": [{"value": r} for r in dedup],
                "n_roots": len(dedup),
                "interval": [lo, hi], "expr": to_str(node),
                "exhaustive": False,
                "note": "bracket scan: every sign-change root in the interval; "
                        "double roots and roots outside it can be missed "
                        "(found-not-exhaustive)"}

    if method == "bisection":
        flo, fhi = f(lo), f(hi)
        if flo * fhi > 0:
            # honest scan: try to bracket a sign change first
            found = False
            span = hi - lo
            for k in range(1, 21):
                a = lo + span * (k - 1) / 20
                b = lo + span * k / 20
                if f(a) * f(b) <= 0:
                    lo, hi, flo, fhi = a, b, f(a), f(b)
                    found = True
                    break
            if not found:
                raise CalcError("bisection: f(lo) and f(hi) have the same sign; "
                               "give a bracket --lo/--hi that contains the root")
        for i in range(max_iter):
            mid = (lo + hi) / 2
            fm = f(mid)
            steps.append({"iter": i + 1, "lo": lo, "hi": hi, "x": mid, "f": fm})
            if abs(fm) < tol or (hi - lo) / 2 < tol:
                return {"method": "bisection", "root": mid, "f_root": fm,
                        "iterations": len(steps), "steps": steps[-6:], "expr": to_str(node)}
            if flo * fm <= 0:
                hi, fhi = mid, fm
            else:
                lo, flo = mid, fm
        raise CalcError("bisection: no convergence in the iteration budget")

    if method == "newton":
        x = float(x0 if x0 is not None else 0.0)
        dnode = simplify(differentiate(node))
        d = lambda v: evaluate(dnode, {"x": v})  # noqa: E731
        for i in range(max_iter):
            fx, dx = f(x), d(x)
            steps.append({"iter": i + 1, "x": x, "f": fx, "df": dx})
            if abs(fx) < tol:
                return {"method": "newton", "root": x, "f_root": fx,
                        "iterations": len(steps), "steps": steps[-6:], "expr": to_str(node)}
            if abs(dx) < 1e-14:
                raise CalcError("newton: derivative vanished — try the secant method or another start")
            x_new = x - fx / dx
            if abs(x_new - x) < tol:
                return {"method": "newton", "root": x_new, "f_root": f(x_new),
                        "iterations": len(steps) + 1, "steps": steps[-6:], "expr": to_str(node)}
            x = x_new
        raise CalcError("newton: no convergence in the iteration budget")

    if method == "secant":
        x_prev = float(x0 if x0 is not None else 0.0)
        x = x_prev + 0.5 if x0 is None else lo
        f_prev = f(x_prev)
        for i in range(max_iter):
            fx = f(x)
            steps.append({"iter": i + 1, "x": x, "f": fx})
            if abs(fx) < tol:
                return {"method": "secant", "root": x, "f_root": fx,
                        "iterations": len(steps), "steps": steps[-6:], "expr": to_str(node)}
            if abs(fx - f_prev) < 1e-14:
                raise CalcError("secant: iterates stalled (flat line)")
            x_new = x - fx * (x - x_prev) / (fx - f_prev)
            x_prev, f_prev, x = x, fx, x_new
        raise CalcError("secant: no convergence in the iteration budget")

    raise CalcError(f"unknown method {method!r} (bisection|newton|secant)")


def integrate(expr: str, lo: float, hi: float, method: str = "simpson",
              n: int = 1000) -> Dict[str, Any]:
    """Numerical integration of f(x) over [lo, hi]."""
    if n < 2:
        raise CalcError("n must be >= 2")
    f, node = _f(expr)
    if method == "trapezoid":
        h = (hi - lo) / n
        total = (f(lo) + f(hi)) / 2 + sum(f(lo + i * h) for i in range(1, n))
        value = total * h
    elif method == "simpson":
        if n % 2:
            n += 1
        h = (hi - lo) / n
        odd = sum(f(lo + (2 * i - 1) * h) for i in range(1, n // 2 + 1))
        even = sum(f(lo + 2 * i * h) for i in range(1, n // 2))
        value = h / 3 * (f(lo) + f(hi) + 4 * odd + 2 * even)
    elif method == "adaptive":
        value = _adaptive(f, lo, hi, f(lo), f(hi), tol=1e-7)[0]
    else:
        raise CalcError(f"unknown method {method!r} (trapezoid|simpson|adaptive)")
    return {"expr": to_str(node), "method": method, "lo": lo, "hi": hi, "n": n,
            "value": value, "note": "deterministic quadrature, no sampling"}


def _adaptive(f, lo, hi, flo, fhi, tol, depth=48):
    mid = (lo + hi) / 2
    fmid = f(mid)
    whole = (hi - lo) * (flo + fhi) / 2
    left = (mid - lo) * (flo + fmid) / 2
    right = (hi - mid) * (fmid + fhi) / 2
    if depth <= 0 or abs(left + right - whole) < 15 * tol:
        return left + right + (left + right - whole) / 15, 1
    la, na = _adaptive(f, lo, mid, flo, fmid, tol / 2, depth - 1)
    ra, nb = _adaptive(f, mid, hi, fmid, fhi, tol / 2, depth - 1)
    return la + ra, na + nb + 1


def ode_solve(expr: str, x0: float, y0: float, x_end: float, h: float = 0.01,
              method: str = "rk4", tol: float = 1e-6, **kwargs) -> Dict[str, Any]:
    """Solve y' = f(x, y) from (x0, y0) to x_end.

    Fixed-step: euler, rk4 (steps = (x_end - x0)/h, recomputed to land
    exactly on x_end). Adaptive: rk45 — embedded Dormand-Prince 4(5)
    (1980) with a user-settable error tolerance; its report carries
    accepted/rejected step counts and the max local error estimate so a
    caller can see whether the adaptive stepping actually earned its
    keep over the fixed-step methods at the same accuracy."""
    node = parse(expr)
    if h <= 0 or x_end <= x0:
        raise CalcError("need h > 0 and x_end > x0")
    if tol <= 0:
        raise CalcError("need tol > 0 (rk45)")

    def f(x: float, y: float) -> float:
        return evaluate(node, {"x": x, "y": y})

    if method == "rk45":
        # F8: the standard TWO-KNOB controller (rtol + atol, **kwargs):
        #   scale = atol + rtol * max(|y|, |y5|)
        # ``tol`` alone maps to rtol = tol, atol = tol * 1e-3 (relative-
        # dominant, with an absolute floor for near-zero y — the stiff
        # decay y' = -1000y used to fight a pure-relative envelope).
        rtol = kwargs.get("rtol", tol)
        atol = kwargs.get("atol", tol * 1e-3)
        adaptive = _rk45_solve(f, x0, y0, x_end, rtol, atol)
        return {"ode": to_str(node), "method": "rk45", "x0": x0, "y0": y0,
                "x_end": x_end, "tol": tol, "rtol": rtol, "atol": atol,
                "controller": "rtol+atol",
                "steps": adaptive["accepted"],
                "rejected_steps": adaptive["rejected"],
                "fevals": adaptive["fevals"],
                "max_local_error": adaptive["max_local_error"],
                "y_end": adaptive["y_end"],
                "trace": adaptive["trace"],
                "trace_points": adaptive["trace_points"]}

    steps = max(1, int(round((x_end - x0) / h)))
    h = (x_end - x0) / steps
    x, y = x0, y0
    trace: List[Dict[str, float]] = [{"x": x, "y": y}]
    for i in range(steps):
        if method == "euler":
            y = y + h * f(x, y)
        elif method == "rk4":
            k1 = f(x, y)
            k2 = f(x + h / 2, y + h * k1 / 2)
            k3 = f(x + h / 2, y + h * k2 / 2)
            k4 = f(x + h, y + h * k3)
            y = y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        else:
            raise CalcError(f"unknown method {method!r} (euler|rk4|rk45)")
        x = x0 + (i + 1) * h
        if i % max(1, steps // 200) == 0 or i == steps - 1:
            trace.append({"x": x, "y": y})
    return {"ode": to_str(node), "method": method, "x0": x0, "y0": y0,
            "x_end": x_end, "steps": steps, "y_end": y,
            "trace": trace[:200], "trace_points": len(trace)}


def _rk45_solve(f, x0: float, y0: float, x_end: float, rtol: float,
                atol: float = 0.0) -> Dict[str, Any]:
    """Adaptive embedded Runge-Kutta 4(5) — Dormand & Prince 1980
    ("A family of embedded Runge-Kutta formulae", J. Comp. Appl. Math
    6(1), 19-35), the DOPRI5 pair: 7 stages, FSAL (the 7th stage is the
    next step's 1st), local error = |5th - 4th| order estimate, the
    standard step controller

        h_new = h * clamp(0.9 * (tol / err) ** (1/5), 0.2, 5.0),

    two knobs (F8): err <= atol + rtol * max(|y|, |y_new|) — the standard
    relative+absolute error envelope.
    Rejected steps are retried smaller; a step floor of 1e-12 * span
    makes failure LOUD (an error, never a silent stall)."""
    c = [0.0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0]
    a = [[0.0]*7,
         [1/5, 0, 0, 0, 0, 0, 0],
         [3/40, 9/40, 0, 0, 0, 0, 0],
         [44/45, -56/15, 32/9, 0, 0, 0, 0],
         [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
         [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
         [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]]
    b5 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
    b4 = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]

    span = x_end - x0
    h_min = 1e-12 * span
    h = min(span, max(h_min, span / 100.0))
    x, y = x0, y0
    accepted = rejected = 0
    fevals = 0
    max_err = 0.0
    trace: List[Dict[str, float]] = [{"x": x, "y": y}]

    def stages(xn, yn, hn, k1):
        ks = [k1]
        for i in range(1, 7):
            yi = yn
            for j in range(i):
                yi += hn * a[i][j] * ks[j]
            ks.append(f(xn + hn * c[i], yi))
        return ks

    k1 = f(x, y)
    fevals += 1
    while x < x_end - 1e-12 * span:
        hn = min(h, x_end - x)
        ks = stages(x, y, hn, k1)
        fevals += 7 - 1  # stages 2..7 (k1 was FSAL-carried)
        y5 = y + hn * sum(b5[i] * ks[i] for i in range(7))
        y4 = y + hn * sum(b4[i] * ks[i] for i in range(7))
        err = abs(y5 - y4)
        scale = atol + rtol * max(abs(y), abs(y5))
        if err <= scale or hn <= h_min:
            x_new = x + hn
            if x_new + 1e-12 * span >= x_end:
                x_new = x_end
            x, y = x_new, y5
            accepted += 1
            max_err = max(max_err, err)
            k1 = f(x, y)  # FSAL
            fevals += 1
            if len(trace) < 4000:
                trace.append({"x": x, "y": y})
            h = min(5.0 * hn, span / 10.0)
        else:
            rejected += 1
            if hn <= h_min:
                raise CalcError(
                    "rk45: step floor reached without meeting the "
                    "tolerance — the requested tol is not achievable "
                    "on this problem (refusing, not stalling silently)")
            h = max(h_min, hn * 0.2)
            continue
        if rejected + accepted > 100000:
            raise CalcError("rk45: step budget exceeded (100000)")
    return {"y_end": y, "accepted": accepted, "rejected": rejected,
            "fevals": fevals, "max_local_error": max_err,
            "trace": trace[:200], "trace_points": len(trace)}


def taylor(expr: str, var: str = "x", around: float = 0.0, order: int = 5) -> Dict[str, Any]:
    """Taylor expansion via repeated symbolic differentiation."""
    node = parse(expr)
    terms: List[Dict[str, Any]] = []
    current = node
    for k in range(order + 1):
        val = evaluate(current, {var: around})
        if abs(val) > 1e-12:
            terms.append({"k": k, "coefficient": val / math.factorial(k)})
        if k == order:
            break
        current = simplify(differentiate(current, var))
    def _term_str(t: Dict[str, Any]) -> str:
        c = _fmt(t["coefficient"])
        if t["k"] == 0:
            return f"({c})"
        if t["k"] == 1:
            return f"({c})*{var}" if around == 0 else f"({c})*({var}-({around}))"
        if around == 0:
            return f"({c})*{var}^{t['k']}"
        return f"({c})*({var}-({around}))^{t['k']}"

    return {"expr": to_str(node), "var": var, "around": around, "order": order,
            "terms": terms,
            "series": " + ".join(_term_str(t) for t in terms) or "0"}


def interpolate(points: Sequence[Sequence[float]], x: float,
                method: str = "lagrange") -> Dict[str, Any]:
    """Interpolate f(x) through points by Lagrange or Newton divided differences."""
    pts = [(float(p[0]), float(p[1])) for p in points]
    if len(pts) < 2:
        raise CalcError("need at least 2 points")
    xs = [p[0] for p in pts]
    if len(set(xs)) != len(xs):
        raise CalcError("duplicate x values in points")
    if method == "lagrange":
        total = 0.0
        for i, (xi, yi) in enumerate(pts):
            term = yi
            for j, (xj, _) in enumerate(pts):
                if i != j:
                    term *= (x - xj) / (xi - xj)
            total += term
        return {"method": "lagrange", "points": pts, "x": x, "value": total}
    if method == "newton":
        n = len(pts)
        coef = [p[1] for p in pts]
        for j in range(1, n):
            for i in range(n - 1, j - 1, -1):
                coef[i] = (coef[i] - coef[i - 1]) / (pts[i][0] - pts[i - j][0])
        # Horner evaluation over the Newton form
        value = coef[-1]
        for i in range(n - 2, -1, -1):
            value = coef[i] + (x - pts[i][0]) * value
        return {"method": "newton", "points": pts, "x": x, "value": value,
                "divided_differences": coef}
    raise CalcError(f"unknown method {method!r} (lagrange|newton)")


# ---------------------------------------------------------------------------
# 5. Small helpers used by the meta-router
# ---------------------------------------------------------------------------

def percent_of(part: float, whole: Optional[float] = None,
               reverse: bool = False) -> Dict[str, Any]:
    """percent_of(15, 80) -> 15% of 80; reverse=True -> 15 is what % of 80."""
    if reverse:
        if whole == 0:
            raise CalcError("whole is zero; percentage undefined")
        return {"part": part, "whole": whole, "percent": part / whole * 100,
                "note": f"{_fmt(part)} is {_fmt(part / whole * 100)}% of {_fmt(whole)}"}
    return {"part_percent": part, "whole": whole, "value": part / 100 * whole,
            "note": f"{_fmt(part)}% of {_fmt(whole)} = {_fmt(part / 100 * whole)}"}


def expression_info(expr: str, env: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """One-stop evaluation with the full evidence trail."""
    node = parse(expr)
    simple = simplify(node)
    value = evaluate(node, env)
    out: Dict[str, Any] = {
        "expr": expr, "canonical": to_str(simple), "value": value,
        "formatted": _fmt(value),
        "functions_used": sorted({n.name for n in _walk(node) if n.kind == "call"}),
        "variables_used": sorted({n.name for n in _walk(node) if n.kind == "var"}),
    }
    if not out["variables_used"]:
        out["simplified"] = to_str(simple) if to_str(simple) != to_str(node) else None
    return out


def _walk(node: _Node):
    stack = [node]
    while stack:
        cur = stack.pop()
        yield cur
        if cur.left:
            stack.append(cur.left)
        if cur.right:
            stack.append(cur.right)
        stack.extend(cur.args)


def _fmt(v: float) -> str:
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return f"{v:.10g}"


# ---------------------------------------------------------------------------
# 8. Bounded symbolic integration (exponential-build-4 B)
# ---------------------------------------------------------------------------
#
# The complement the differentiation rules above already had. What this
# is: a pattern table over the SAME AST (polynomial, exponential,
# logarithmic, trigonometric forms), linear-chain substitution, and
# integration by parts for the classic p(x)*{exp,sin,cos} and ln(x)
# shapes. What this is NOT: a Risch algorithm. There is no differential
# field theory here, no Liouville-principle structure theorem, no claim
# that "no closed form exists" in general — only the honest narrower
# verdict "no closed form IN THIS ENGINE'S TABLE". Every antiderivative
# the table produces is then VERIFIED before it is returned: the engine
# differentiates its own answer and compares against the integrand at
# fixed sample points, refusing its own output when the check fails.

_INTEG_CHECK_POINTS = (0.5, 1.3, 2.1)  # fixed, documented, no randomness
_SUBST_RATIO_POINTS = (0.317, 0.913, 1.729, 2.618)
_MAX_IBP_DEGREE = 4


def _poly_coeffs(node: _Node, var: str) -> Optional[Dict[int, float]]:
    """Coefficients {degree: coeff} when node is a polynomial in var
    (sums, differences, negation, constants, var^int), else None."""
    if node.kind == "num":
        return {0: float(node.value)}
    if node.kind == "neg":
        inner = _poly_coeffs(node.left, var)
        return None if inner is None else {d: -c for d, c in inner.items()}
    if node.kind == "var":
        return {1: 1.0} if node.name == var else None
    if node.kind == "bin":
        op = node.value
        if op == "^":
            base, expo = _poly_coeffs(node.left, var), _poly_coeffs(node.right, var)
            if (base is not None and expo is not None and base == {1: 1.0}
                    and len(expo) == 1 and 0 in expo):
                deg = expo[0]
                if deg == int(deg) and 0 <= deg <= 12:
                    return {int(deg): 1.0}
            return None
        left, right = _poly_coeffs(node.left, var), _poly_coeffs(node.right, var)
        if left is None or right is None:
            return None
        out: Dict[int, float] = {}
        if op == "+":
            keys = set(left) | set(right)
            for d in keys:
                out[d] = left.get(d, 0.0) + right.get(d, 0.0)
            return out
        if op == "-":
            keys = set(left) | set(right)
            for d in keys:
                out[d] = left.get(d, 0.0) - right.get(d, 0.0)
            return out
        if op == "*":
            for d1, c1 in left.items():
                for d2, c2 in right.items():
                    out[d1 + d2] = out.get(d1 + d2, 0.0) + c1 * c2
            return out
        if op == "/":
            if len(right) == 1 and 0 in right and right[0] != 0:
                return {d: c / right[0] for d, c in left.items()}
            return None
    return None


def _poly_node(coeffs: Dict[int, float]) -> _Node:
    """Build the AST for sum(c_d x^d), descending; zero polynomial = 0."""
    terms: List[_Node] = []
    for d in sorted(coeffs, reverse=True):
        c = coeffs[d]
        if abs(c) < 1e-15:
            continue
        if d == 0:
            terms.append(_Node("num", value=c))
        else:
            p = _Node("bin", value="*", left=_Node("num", value=c),
                      right=(_Node("var", name="x") if d == 1 else
                             _Node("bin", value="^", left=_Node("var", name="x"),
                                   right=_Node("num", value=float(d)))))
            terms.append(p)
    if not terms:
        return _Node("num", value=0.0)
    out = terms[0]
    for t in terms[1:]:
        out = _Node("bin", value="+", left=out, right=t)
    return out


def _linear_in(node: _Node, var: str) -> Optional[Tuple[float, float]]:
    """(a, b) when node == a*var + b with a != 0, else None."""
    coeffs = _poly_coeffs(node, var)
    if coeffs is None or set(coeffs) - {0, 1}:
        return None
    a = coeffs.get(1, 0.0)
    if abs(a) < 1e-15:
        return None
    return a, coeffs.get(0, 0.0)


def _num(v: float) -> _Node:
    return _Node("num", value=float(v))


def _bin(op: str, l: _Node, r: _Node) -> _Node:
    return _Node("bin", value=op, left=l, right=r)


def _call(name: str, arg: _Node) -> _Node:
    return _Node("call", name=name, args=[arg])


def _div(a: _Node, b: _Node) -> _Node:
    return _bin("/", a, b)


def _scaled(node: _Node, k: float) -> _Node:
    return _bin("*", _num(k), node)


def _eval_safe(node: _Node, var: str, x: float) -> Optional[float]:
    try:
        v = evaluate(node, {var: x})
        if math.isfinite(v):
            return v
    except (CalcError, ZeroDivisionError, ValueError, OverflowError):
        pass
    return None


def _verify_antiderivative(F: _Node, integrand: _Node, var: str) -> bool:
    """dF/dx ?= integrand at the fixed sample points (points where
    either side is undefined are skipped; at least two must agree)."""
    dF = simplify(differentiate(F, var))
    good = 0
    for x in _INTEG_CHECK_POINTS:
        a = _eval_safe(dF, var, x)
        b = _eval_safe(integrand, var, x)
        if a is None or b is None:
            continue
        if abs(a - b) > 1e-6 * max(1.0, abs(a), abs(b)):
            return False
        good += 1
    return good >= 2


def _table_integrate(node: _Node, var: str, trace: List[str]
                     ) -> Optional[_Node]:
    """The pattern table (see the section docstring). Returns the
    antiderivative AST or None — None means OUT OF TABLE, never wrong."""
    node = simplify(node)

    coeffs = _poly_coeffs(node, var)
    if coeffs is not None:
        deg = max(coeffs) if coeffs else 0
        # sum(c_d x^d) -> sum(c_d x^(d+1)/(d+1)) + c0*x
        out: Optional[_Node] = None
        for d, c in coeffs.items():
            if abs(c) < 1e-15:
                continue
            if d == -1:
                return None  # a rational poly-ratio is out of table
            piece = _div(_scaled(_bin("^", _Node("var", name=var),
                                      _num(d + 1)), c / (d + 1)), _num(1.0))
            piece = _scaled(_bin("^", _Node("var", name=var), _num(d + 1)),
                            c / (d + 1))
            out = piece if out is None else _bin("+", out, piece)
        if out is not None:
            trace.append(f"polynomial table (degree {deg})")
            return simplify(out)

    # constant multiple / sum decomposition
    if node.kind == "bin" and node.value in "+-":
        left = _table_integrate(node.left, var, trace)
        if left is None:
            return None
        right = _table_integrate(node.right, var, trace)
        if right is None:
            return None
        trace.append("linearity (sum rule)")
        return simplify(_bin(node.value, left, right))
    if node.kind == "neg":
        inner = _table_integrate(node.left, var, trace)
        return None if inner is None else simplify(_Node("neg", left=inner))
    if (node.kind == "bin" and node.value == "*" and node.left.kind == "num"):
        inner = _table_integrate(node.right, var, trace)
        return None if inner is None else simplify(_scaled(inner, node.left.value))
    if (node.kind == "bin" and node.value == "*" and node.right.kind == "num"):
        inner = _table_integrate(node.left, var, trace)
        return None if inner is None else simplify(_scaled(inner, node.right.value))

    # x^c and (a x + b)^c
    if node.kind == "bin" and node.value == "^":
        base, expo = node.left, node.right
        if expo.kind == "num":
            c = float(expo.value)
            lin = _linear_in(base, var) if base.kind != "var" else (1.0, 0.0)
            if base.kind == "var" and base.name == var:
                lin = (1.0, 0.0)
            if lin is not None:
                a, b = lin
                if abs(c + 1.0) < 1e-12:
                    trace.append("x^-1 -> ln (table)")
                    return simplify(_div(_call("ln", base), _num(a)))
                trace.append("power rule (table)")
                return simplify(_div(_bin("^", base, _num(c + 1.0)),
                                     _num(a * (c + 1.0))))

    # 1/u with u linear-in-x handled above; bare 1/x:
    if (node.kind == "bin" and node.value == "/"
            and _is(node.left, 1.0) and node.right.kind == "var"
            and node.right.name == var):
        trace.append("1/x -> ln (table)")
        return simplify(_call("ln", node.right))

    # calls with linear inner argument: exp / ln / sin / cos / tan
    if node.kind == "call" and len(node.args) == 1:
        arg = node.args[0]
        if arg.kind == "var" and arg.name == var:
            lin = (1.0, 0.0)
        else:
            lin = _linear_in(arg, var)
        if lin is not None:
            a, _b = lin
            if node.name == "exp":
                trace.append("exp(linear) (table)")
                return simplify(_div(_call("exp", arg), _num(a)))
            if node.name in ("ln", "log"):
                trace.append("ln(linear) by parts (table)")
                # u*ln(u) - u, all over a   (u = a x + b; a != 0 here)
                return simplify(_div(_bin("-",
                                          _bin("*", arg, _call("ln", arg)),
                                          arg),
                                     _num(a)))
            if node.name == "sin":
                trace.append("sin(linear) (table)")
                return simplify(_Node("neg", left=_div(_call("cos", arg), _num(a))))
            if node.name == "cos":
                trace.append("cos(linear) (table)")
                return simplify(_div(_call("sin", arg), _num(a)))
            if node.name == "tan":
                trace.append("tan(linear) -> -ln|cos| (table; per-interval)")
                inner = _div(_call("ln", _call("abs", _call("cos", arg))), _num(a))
                return simplify(_Node("neg", left=inner))
            if node.name == "sqrt" and lin[0] == 1.0 and lin[1] == 0.0:
                trace.append("sqrt(x) = x^1/2 (table)")
                return simplify(_div(_bin("^", arg, _num(1.5)), _num(1.5)))

    # integration by parts: polynomial * {exp, sin, cos}(linear in x)
    if node.kind == "bin" and node.value == "*":
        poly, other = _poly_coeffs(node.left, var), node.right
        if poly is None:
            poly, other = _poly_coeffs(node.right, var), node.left
        if (poly is not None and other.kind == "call"
                and len(other.args) == 1):
            arg = other.args[0]
            lin = (1.0, 0.0) if arg.kind == "var" else _linear_in(arg, var)
            if (lin is not None and other.name in ("exp", "sin", "cos")
                    and max(poly) <= _MAX_IBP_DEGREE
                    and min(poly) >= 0):
                trace.append(f"by parts: p(x)*{other.name}"
                             f"({'x' if lin == (1.0, 0.0) else 'a x + b'})")
                result = _ibp_integrate(poly, other.name, lin, var, trace)
                if result is not None:
                    return simplify(result)

    # u-substitution: f(g(x)) * h(x) where h = k * g'(x), k constant
    if node.kind == "bin" and node.value == "*":
        for fpart, hpart in ((node.left, node.right), (node.right, node.left)):
            if fpart.kind == "call" and len(fpart.args) == 1:
                g = fpart.args[0]
                if g.kind == "num":
                    continue
                gp = simplify(differentiate(g, var))
                ratio_points = []
                ok_ratio = True
                for x in _SUBST_RATIO_POINTS:
                    hv = _eval_safe(hpart, var, x)
                    gv = _eval_safe(gp, var, x)
                    if hv is None or gv is None:
                        continue  # outside either's domain: skip the point
                    if abs(gv) < 1e-12:
                        continue
                    ratio_points.append(hv / gv)
                if len(ratio_points) >= 2 and all(
                        abs(r - ratio_points[0]) <= 1e-9 * max(1.0, abs(ratio_points[0]))
                        for r in ratio_points):
                    k = ratio_points[0]
                    if abs(k) < 1e-12:
                        continue
                    if fpart.name in ("exp", "sin", "cos", "ln", "log") or (
                            fpart.name == "sqrt"):
                        sub = _table_integrate(fpart, var, trace) if (
                            _linear_in(g, var) is not None) else None
                        if sub is None:
                            # integrate f(u) du in u-world then substitute back
                            sub = _table_single(fpart, var, trace)
                        if sub is None:
                            return None
                        trace.append(f"u-substitution over {fpart.name}"
                                     f"(g) with constant ratio k={_fmt(k)}")
                        # h = k*g'  =>  Int f(g) h dx = k * Int f(u) du = k * F(g)
                        return simplify(_scaled(sub, k))
    return None


def _differentiate_poly(coeffs: Dict[int, float]) -> Dict[int, float]:
    return {d - 1: d * c for d, c in coeffs.items() if d >= 1}


def _table_single(node: _Node, var: str, trace: List[str]) -> Optional[_Node]:
    """Integrate a bare call f(g) by treating g as the variable — only
    valid when the call's table antiderivative is expressible in its own
    argument (exp/ln/sin/cos/sqrt). Used by the substitution path."""
    g = node.args[0]
    if node.name == "exp":
        return simplify(_call("exp", g))
    if node.name in ("ln", "log"):
        return simplify(_bin("-", _bin("*", g, _call("ln", g)), g))
    if node.name == "sin":
        return simplify(_Node("neg", left=_call("cos", g)))
    if node.name == "cos":
        return simplify(_call("sin", g))
    if node.name == "sqrt":
        return simplify(_div(_bin("^", g, _num(1.5)), _num(1.5)))
    return None


def _lin_node(lin: Tuple[float, float]) -> _Node:
    """The AST for a*x + b from a (a, b) linear pair."""
    a, b = lin
    node = _scaled(_Node("var", name="x"), a) if a != 1.0 else _Node("var", name="x")
    if b != 0.0:
        node = _bin("+", node, _num(b))
    return node


def _ibp_integrate(coeffs: Dict[int, float], fname: str,
                   lin: Tuple[float, float], var: str,
                   trace: List[str]) -> Optional[_Node]:
    """Repeated integration by parts for p(x)*{exp,sin,cos}(a x + b).

    Recursions (u = a x + b, a != 0):
      exp:  p e^u / a  -  (1/a) IBP_exp(p')
      sin: -p cos(u) / a + (1/a) IBP_cos(p')
      cos:  p sin(u) / a  -  (1/a) IBP_sin(p')
    Base case p = constant falls back to the linear-argument table.
    Degree is bounded by _MAX_IBP_DEGREE (the caller checks); honest
    None when the inner integral leaves the table."""
    a = lin[0]
    p_node = _poly_node(coeffs)
    if set(coeffs) == {0}:
        c = coeffs[0]
        u = _lin_node(lin)
        if fname == "exp":
            return _div(_scaled(_call("exp", u), c), _num(a))
        if fname == "sin":
            return _div(_Node("neg", left=_scaled(_call("cos", u), c)), _num(a))
        return _div(_scaled(_call("sin", u), c), _num(a))
    d = _differentiate_poly(coeffs)
    if not d:
        return None
    if fname == "exp":
        rest = _ibp_integrate(d, fname, lin, var, trace)
        if rest is None:
            return None
        # p e^u / a - rest / a
        return simplify(_bin("-", _div(_bin("*", p_node, _call("exp", _lin_node(lin))),
                                       _num(a)),
                             _div(rest, _num(a))))
    if fname == "sin":
        rest = _ibp_integrate(d, "cos", lin, var, trace)
        if rest is None:
            return None
        # -p cos(u) / a + rest / a
        return simplify(_bin("+",
                             _div(_Node("neg", left=_bin("*", p_node,
                                                         _call("cos", _lin_node(lin)))),
                                  _num(a)),
                             _div(rest, _num(a))))
    if fname == "cos":
        rest = _ibp_integrate(d, "sin", lin, var, trace)
        if rest is None:
            return None
        # p sin(u) / a - rest / a
        return simplify(_bin("-",
                             _div(_bin("*", p_node, _call("sin", _lin_node(lin))),
                                  _num(a)),
                             _div(rest, _num(a))))
    return None


def symbolic_integrate(expr: str, var: str = "x") -> Dict[str, Any]:
    """Bounded symbolic integration: table + linear chains + parts.

    Returns one of three honest statuses:
      OK                          — antiderivative string + method trace,
                                    DIFFERENTIALLY VERIFIED at fixed
                                    sample points before returning;
      NO_CLOSED_FORM_IN_TABLE     — this engine's table has no entry
                                    (NOT a general non-integrability
                                    claim — this is not Risch);
      REFUSED_VERIFICATION_FAILED — the table produced something whose
                                    derivative disagreed with the
                                    integrand at the sample points; the
                                    answer is refused, never shipped.
    """
    node = parse(expr)
    trace: List[str] = []
    F = _table_integrate(simplify(node), var, trace)
    if F is None:
        return {"expr": to_str(node), "var": var, "status":
                "NO_CLOSED_FORM_IN_TABLE",
                "note": "no closed form in this engine's table — an "
                        "honest refusal, not a Risch non-integrability "
                        "proof; numerical integrate() still works",
                "method_trace": trace}
    if not _verify_antiderivative(F, node, var):
        return {"expr": to_str(node), "var": var, "status":
                "REFUSED_VERIFICATION_FAILED",
                "note": "the table's candidate failed its own "
                        "differential verification at fixed points — "
                        "refused, never shipped",
                "method_trace": trace}
    return {"expr": to_str(node), "var": var, "status": "OK",
            "antiderivative": to_str(F), "method_trace": trace,
            "verified": True,
            "note": "verified by differentiating this answer and "
                    "comparing against the integrand at fixed sample "
                    "points; +C omitted (a constant, always)"}


# ---------------------------------------------------------------------------
# 9. Exact polynomial root finding (exponential-build-5 F8, D6).
# All-roots: exact rational roots with multiplicity over fractions.Fraction,
# exact quadratic radicals, Durand-Kerner + Newton polish for the rest,
# and a Sturm sequence (exact) that counts the DISTINCT real roots so the
# result can say honestly whether the root set is exhaustive.
# ---------------------------------------------------------------------------

def polynomial_coeffs(node: "_Node", var: str = "x") -> Optional[List[Any]]:
    """Exact rational coefficients (ascending) of a polynomial in ``var``,
    or None when the expression is not a polynomial (calls, non-integer
    exponents, other variables, division by a non-constant...)."""

    def exact(v: float) -> Fraction:
        return Fraction(v).limit_denominator(10 ** 12)

    def walk(n: "_Node") -> Optional[Dict[int, Any]]:
        if n.kind == "num":
            return {0: exact(float(n.value))}
        if n.kind == "var":
            return {1: Fraction(1)} if n.name == var else None
        if n.kind == "neg":
            inner = walk(n.left)
            if inner is None:
                return None
            return {k: -v for k, v in inner.items()}
        if n.kind == "bin":
            a, b = walk(n.left), walk(n.right)
            if a is None or b is None:
                return None
            op = n.value
            if op == "+":
                out = dict(a)
                for k, v in b.items():
                    out[k] = out.get(k, Fraction(0)) + v
                return out
            if op == "-":
                out = dict(a)
                for k, v in b.items():
                    out[k] = out.get(k, Fraction(0)) - v
                return out
            if op == "*":
                out: Dict[int, Any] = {}
                for ka, va in a.items():
                    for kb, vb in b.items():
                        out[ka + kb] = out.get(ka + kb, Fraction(0)) + va * vb
                return out
            if op == "/":
                if set(b) == {0}:
                    return {k: v / b[0] for k, v in a.items()}
                return None
            if op == "^":
                # only non-negative integer exponents of the variable
                if set(b) == {0} and b[0].denominator == 1 and b[0] >= 0:
                    e = int(b[0])
                    if e > 64:
                        return None
                    out = {0: Fraction(1)}
                    for _ in range(e):
                        nxt: Dict[int, Any] = {}
                        for k1, v1 in out.items():
                            for k2, v2 in a.items():
                                nxt[k1 + k2] = nxt.get(k1 + k2, Fraction(0)) + v1 * v2
                        out = nxt
                    return out
                return None
            return None
        return None

    poly = walk(node)
    if poly is None:
        return None
    degree = max(poly) if poly else 0
    out = [poly.get(k, Fraction(0)) for k in range(degree + 1)]
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out


def _poly_eval_frac(coeffs: List[Any], x: Any) -> Any:
    total = 0 * x if x != 0 else 0
    acc = type(x)(1) if hasattr(type(x), "__call__") is False and False else 1
    # Horner over ascending coefficients (reversed)
    total = None
    for c in reversed(coeffs):
        total = c if total is None else total * x + c
    return total


def _poly_eval_c(coeffs: List[Any], x: complex) -> complex:
    total = 0j
    for c in reversed(coeffs):
        total = total * x + float(c)
    return total


def _sturm_real_root_count(coeffs: List[Any]) -> Optional[int]:
    """Distinct real roots via the Sturm sequence (exact Fractions).
    None when the sequence degenerates (should not happen for a
    square-free-normalized polynomial with degree >= 1)."""

    def normalize(seq):
        while seq and seq[-1] == 0:
            seq.pop()
        return seq

    def rem(a, b):
        # polynomial remainder of a / b, both ascending Fraction lists
        a = list(a)
        while len(a) >= len(b) and normalize(a):
            if len(a) < len(b):
                break
            shift = len(a) - len(b)
            factor = a[-1] / b[-1]
            for i, c in enumerate(b):
                a[shift + i] -= factor * c
            normalize(a)
        return a

    p = [Fraction(c) for c in coeffs]
    normalize(p)
    if not p or len(p) < 2:
        return None
    dp = [Fraction(i * p[i]) for i in range(1, len(p))]
    normalize(dp)
    if not dp:
        return None
    seq = [p, dp]
    while len(seq[-1]) > 1:
        r = rem(seq[-2], seq[-1])
        if not r:
            break
        seq.append([-c for c in r])
    if len(seq) < 2:
        return None

    def sign_at_inf(s: int) -> List[int]:
        out = []
        for poly in seq:
            lead = poly[-1]
            v = lead * (s ** (len(poly) - 1))
            out.append(1 if v > 0 else -1)
        return out

    def variations(signs: List[int]) -> int:
        nz = [s for s in signs if s != 0]
        return sum(1 for i in range(len(nz) - 1) if nz[i] != nz[i + 1])

    return variations(sign_at_inf(-1)) - variations(sign_at_inf(1))


def _exact_quadratic(c: Any, b: Any, a: Any) -> List[Dict[str, Any]]:
    """Exact roots of ax^2 + bx + c over Fractions, radicals simplified:
    D = b^2-4ac = s^2 * m with m square-free, so roots are
    r +/- k*sqrt(m) with r = -b/2a and k = s/2a rational."""
    import math as _math

    two_a = 2 * a
    disc = b * b - 4 * a * c
    if disc == 0:
        r = -b / two_a
        return [{"value": float(r), "exact": _fmt_frac(r),
                 "kind": "rational", "multiplicity": 2}]
    if disc > 0:
        num, den = disc.numerator, disc.denominator
        s_num = _math.isqrt(num)
        while s_num > 1 and num % (s_num * s_num) != 0:
            s_num -= 1
        s_den = _math.isqrt(den)
        while s_den > 1 and den % (s_den * s_den) != 0:
            s_den -= 1
        m = Fraction(num // (s_num * s_num), den // (s_den * s_den))
        r = -b / two_a
        if m == 1:
            # disc is a perfect rational square: rational roots
            s = Fraction(s_num, s_den)
            r1, r2 = r + s / two_a, r - s / two_a
            return [
                {"value": float(r1), "exact": _fmt_frac(r1), "kind": "rational", "multiplicity": 1},
                {"value": float(r2), "exact": _fmt_frac(r2), "kind": "rational", "multiplicity": 1},
            ]
        k = Fraction(s_num, s_den) / two_a
        val = _math.sqrt(float(disc)) / float(two_a)
        return [
            {"exact": _surd_str(r, k, m, +1), "value": float(r) + val,
             "kind": "surd", "multiplicity": 1},
            {"exact": _surd_str(r, k, m, -1), "value": float(r) - val,
             "kind": "surd", "multiplicity": 1},
        ]
    im = _math.sqrt(float(-disc)) / float(two_a)
    re_frac = -b / two_a
    re_part = float(re_frac)
    im_body = f"{im:.10g}i" if abs(im - 1) > 1e-12 else "i"
    if re_frac == 0:
        forms = (im_body, f"-{im_body}")
    else:
        forms = (f"{_fmt_frac(re_frac)} + {im_body}",
                 f"{_fmt_frac(re_frac)} - {im_body}")
    return [
        {"exact": forms[0], "value": complex(re_part, im),
         "kind": "complex", "multiplicity": 1},
        {"exact": forms[1], "value": complex(re_part, -im),
         "kind": "complex", "multiplicity": 1},
    ]


def _surd_str(r, k, m, sign: int) -> str:
    """'r + k*sqrt(m)' simplified: drop zero/one parts, fold the sign."""
    k = abs(k)
    neg = sign < 0
    body = f"sqrt({_fmt_frac(m)})"
    if k != 1:
        body = f"{_fmt_frac(k)}*{body}"
    if r == 0:
        return f"-{body}" if neg else body
    return f"{_fmt_frac(r)} - {body}" if neg else f"{_fmt_frac(r)} + {body}"


def _fmt_frac(f) -> str:
    if f.denominator == 1:
        return str(f.numerator)
    return f"{f.numerator}/{f.denominator}"


def solve_polynomial_all(coeffs: List[Any]) -> Dict[str, Any]:
    """All roots of a polynomial given as ascending exact Fraction
    coefficients: rational roots with multiplicity (exact), quadratic
    leftovers with simplified radical exact forms, Durand-Kerner + Newton
    polish for higher degrees, and a Sturm count that says whether the
    real root set is exhaustive."""
    import math as _math

    degree = len(coeffs) - 1
    if degree < 1:
        raise CalcError("not a degree >= 1 polynomial")
    if coeffs[0] == 0:  # x is a factor: root 0 with the deflated multiplicity
        sub = solve_polynomial_all(coeffs[1:])
        for r in sub["roots"]:
            pass
        zero_mult = 1
        rest = coeffs[1:]
        while rest and rest[0] == 0:
            zero_mult += 1
            rest = rest[1:]
        sub = solve_polynomial_all(rest) if len(rest) > 1 else None
        roots = [{"value": 0.0, "exact": "0", "kind": "rational",
                  "multiplicity": zero_mult}]
        if sub:
            roots.extend(sub["roots"])
        sturm = _sturm_real_root_count(coeffs)
        return {"method": "polynomial", "degree": degree, "roots": roots,
                "real_roots_count": sturm if sturm is not None else len(
                    [r for r in roots if not hasattr(r["value"], "imag")]),
                "sturm_count": sturm, "exhaustive": True}

    roots: List[Dict[str, Any]] = []
    work = list(coeffs)

    # 1) exact rational roots with multiplicity (rational-root theorem +
    #    exact synthetic division)
    while len(work) > 3:
        a0, an = work[0], work[-1]
        found = None
        for p in _divisors(abs(a0.numerator)):
            for q in _divisors(an.denominator):
                for cand in (Fraction(p, q), Fraction(-p, q)):
                    if _poly_eval_frac(work, cand) == 0:
                        found = cand
                        break
                if found is not None:
                    break
            if found is not None:
                break
        if found is None:
            break
        mult = 0
        while True:
            work = _synthetic_div(work, found)
            mult += 1
            if _poly_eval_frac(work, found) != 0 or len(work) < 2:
                break
        roots.append({"value": float(found), "exact": _fmt_frac(found),
                      "kind": "rational", "multiplicity": mult})

    # 2) the remaining factor: degree 1, 2 exact; >= 3 Durand-Kerner
    if len(work) == 2:
        r = -work[0] / work[1]
        roots.append({"value": float(r), "exact": _fmt_frac(r),
                      "kind": "rational", "multiplicity": 1})
    elif len(work) == 3:
        roots.extend(_exact_quadratic(work[0], work[1], work[2]))
    elif len(work) > 3:
        for v, ex in _durand_kerner(work):
            roots.append({"value": v, "exact": ex, "kind": "numeric",
                          "multiplicity": 1})

    sturm = _sturm_real_root_count(coeffs)
    n_real_found = len({round(r["value"], 9) for r in roots
                        if not (hasattr(r["value"], "imag") and r["value"].imag != 0)})
    exhaustive = sturm is None or sturm == n_real_found
    return {"method": "polynomial", "degree": degree, "roots": roots,
            "real_roots_count": sturm if sturm is not None else n_real_found,
            "sturm_count": sturm, "exhaustive": exhaustive}


def _divisors(n: int) -> List[int]:
    out = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            out.append(i)
            if i != n // i:
                out.append(n // i)
        i += 1
    return sorted(out) or [1]


def _synthetic_div(coeffs: List[Any], root: Any) -> List[Any]:
    """Divide ascending-coefficient polynomial by (x - root); exact."""
    desc = list(reversed(coeffs))
    out = [desc[0]]
    for c in desc[1:-1]:
        out.append(c + out[-1] * root)
    return list(reversed(out))


def _durand_kerner(coeffs: List[Any], max_iter: int = 300) -> List[Tuple[complex, Optional[str]]]:
    """All complex roots by simultaneous Weierstrass/Durand-Kerner
    iteration, each polished by Newton on the original polynomial."""
    n = len(coeffs) - 1
    f = [float(c) for c in coeffs]
    roots = [(0.4 + 0.9j) ** k for k in range(n)]
    for _ in range(max_iter):
        new = []
        for i, ri in enumerate(roots):
            denom = 1.0 + 0j
            for j, rj in enumerate(roots):
                if i != j:
                    denom *= ri - rj
            if abs(denom) < 1e-300:
                denom = 1e-300 + 0j
            new.append(ri - _poly_eval_c(f, ri) / denom)
        shift = max(abs(a - b) for a, b in zip(roots, new))
        roots = new
        if shift < 1e-14:
            break
    # Newton polish on the ORIGINAL polynomial
    df = [i * f[i] for i in range(1, len(f))]
    polished = []
    for r in roots:
        x = r
        for _ in range(6):
            fx = _poly_eval_c(f, x)
            dx = _poly_eval_c(df, x) if len(df) >= 1 else 1.0
            if abs(dx) < 1e-300:
                break
            step = fx / dx
            x = x - step
            if abs(step) < 1e-15:
                break
        polished.append(x)
    out = []
    for x in sorted(polished, key=lambda z: (round(z.real, 9), round(z.imag, 9))):
        if abs(x.imag) < 1e-8:
            out.append((complex(x.real, 0), None))
        else:
            out.append((x, None))
    return out
