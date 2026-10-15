"""genius.csvquery — a named-column expression evaluator over a
user-supplied CSV/TSV table.

Built entirely on existing primitives (grep-first, nothing new invented):
- ``genius/data.py::parse_table`` — the existing typed CSV/TSV parser
  (the ``csv`` module underneath; a parser, never a writer);
- ``genius/stats.py`` — the aggregate functions (describe/quantile/
  pearson), the same implementations every other domain uses;
- ``ast`` — the SAFETY mechanism: the expression is parsed and walked
  over an explicit whitelist of node types; ``eval`` is never called,
  attribute access, subscripts, lambdas and unknown calls are rejected
  by name before anything runs.

The expression language:
- column NAMES resolve to the typed column vectors parse_table produced
  (exact match; an unknown name is an honest error listing what exists);
- arithmetic ``+ - * / // % **``, unary ``- + not``, comparisons
  ``< <= > >= == !=`` (chains included), and ``and``/``or`` — the
  three-valued (Kleene) logic: a missing or non-numeric cell makes the
  surrounding row value ``None`` (unavailable), and ``None`` propagates:
  it is never coerced to 0 or False;
- division/modulo by zero yields ``None`` for that row (counted in
  ``rows_unavailable``), never an invented number;
- AGGREGATE calls — ``count sum min max mean median stdev variance
  quantile pearson`` — receive their inner expression's values over ALL
  rows and collapse to one scalar (``quantile``'s q must be within
  [0, 1] and is rejected otherwise, never clamped — the settings
  convention; ``pearson`` pairs rows positionally, dropping only rows
  where EITHER side is unavailable).

Pure text + numbers: no I/O (the caller owns the table text), no
execution, no network, no RNG; identical input bytes -> identical
output. The evaluator answers questions; it never writes a file.
"""
from __future__ import annotations

import ast
from typing import Any, Callable, Dict, List, Sequence

from . import data as data_mod
from . import stats as stats_mod

__all__ = ["evaluate", "AGGREGATES"]

AGGREGATES = ("count", "sum", "min", "max", "mean", "median",
              "stdev", "variance", "quantile", "pearson")

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
                   ast.Mod, ast.Pow)
_ALLOWED_CMPOPS = (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)

_BIN_FUNCS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}

_CMP_FUNCS = {
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
}


def _is_num(v: Any) -> bool:
    # bools count as numbers (Python semantics: a comparison column can
    # be averaged); None and strings never do.
    return isinstance(v, (int, float))


def _is_scalar_aggregate(node: ast.AST) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in AGGREGATES)


def _compile(node: ast.AST, columns: Dict[str, Sequence[Any]],
             names: List[str], nrows: int) -> Callable[[int], Any]:
    """Compile one AST node into a row-evaluator closure over the table.
    Every branch either returns a closure or raises ValueError — there
    is no dynamic evaluation anywhere on the path."""
    if isinstance(node, ast.Expression):
        return _compile(node.body, columns, names, nrows)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, bool)):
            v = node.value
            return lambda _i: v
        raise ValueError(
            f"only numeric/boolean constants are allowed "
            f"(got {node.value!r})")
    if isinstance(node, ast.Name):
        name = node.id
        if name not in names or not isinstance(columns.get(name), list):
            raise ValueError(
                f"unknown column {name!r} — available: "
                f"{', '.join(names) if names else '(none)'}")
        col = columns[name]
        return lambda i: col[i]
    if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
        lf = _compile(node.left, columns, names, nrows)
        rf = _compile(node.right, columns, names, nrows)
        op = _BIN_FUNCS[type(node.op)]

        def binfn(i: int) -> Any:
            a, b = lf(i), rf(i)
            if not (_is_num(a) and _is_num(b)):
                return None
            try:
                return op(a, b)
            except (ZeroDivisionError, OverflowError):
                return None  # unavailable, never invented
        return binfn
    if isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.USub, ast.UAdd, ast.Not)):
        vf = _compile(node.operand, columns, names, nrows)
        if isinstance(node.op, ast.Not):
            def notfn(i: int) -> Any:
                v = vf(i)
                return None if v is None else (not v)
            return notfn
        sign = -1 if isinstance(node.op, ast.USub) else 1

        def unaryfn(i: int) -> Any:
            v = vf(i)
            if not _is_num(v):
                return None
            return sign * v
        return unaryfn
    if isinstance(node, ast.Compare) and all(
            isinstance(op, _ALLOWED_CMPOPS) for op in node.ops):
        fns = [_compile(node.left, columns, names, nrows)] + [
            _compile(c, columns, names, nrows) for c in node.comparators]

        def cmpfn(i: int) -> Any:
            vals = [f(i) for f in fns]
            if any(v is None for v in vals):
                return None
            for k, op in enumerate(node.ops):
                if not _CMP_FUNCS[type(op)](vals[k], vals[k + 1]):
                    return False
            return True
        return cmpfn
    if isinstance(node, ast.BoolOp) and isinstance(
            node.op, (ast.And, ast.Or)):
        fns = [_compile(v, columns, names, nrows) for v in node.values]
        is_and = isinstance(node.op, ast.And)

        def boolfn(i: int) -> Any:
            saw_none = False
            for f in fns:
                v = f(i)
                if v is None:
                    saw_none = True
                    continue
                v = bool(v)
                if is_and and not v:
                    return False
                if not is_and and v:
                    return True
            return None if saw_none else is_and
        return boolfn
    if isinstance(node, ast.Call):
        return _compile_call(node, columns, names, nrows)
    raise ValueError(
        f"unsupported syntax in expression: {type(node).__name__} — the "
        f"allowed set is columns, numbers, + - * / // % **, comparisons, "
        f"and/or/not, and the aggregates ({', '.join(AGGREGATES)})")


def _compile_call(node: ast.Call, columns: Dict[str, Sequence[Any]],
                  names: List[str], nrows: int) -> Callable[[int], Any]:
    if not isinstance(node.func, ast.Name) or node.func.id not in AGGREGATES:
        raise ValueError(
            f"only these aggregate calls are allowed: "
            f"{', '.join(AGGREGATES)}")
    if node.keywords:
        raise ValueError("keyword arguments are not allowed in expressions")
    name = node.func.id
    arg_fns = [_compile(a, columns, names, nrows) for a in node.args]

    def vectors() -> List[List[Any]]:
        return [[af(i) for i in range(nrows)] for af in arg_fns]

    if name == "count":
        def countfn(_i: int) -> int:
            return sum(1 for v in vectors()[0] if v is not None)
        return countfn
    if name == "pearson":
        if len(arg_fns) != 2:
            raise ValueError("pearson needs exactly two arguments")
        # positional pairing: a row is dropped only when EITHER side is
        # unavailable, so the pairs stay aligned to real rows
        def pearsonfn(_i: int) -> Any:
            va, vb = vectors()
            xs, ys = [], []
            for a, b in zip(va, vb):
                if _is_num(a) and _is_num(b):
                    xs.append(a)
                    ys.append(b)
            if len(xs) < 3:
                return None  # too thin to claim a correlation
            return stats_mod.pearson(xs, ys)["r"]
        return pearsonfn
    if name == "quantile":
        if len(arg_fns) != 2:
            raise ValueError("quantile needs (column, q)")
        def qfn(_i: int) -> Any:
            vals = [v for v in vectors()[0] if _is_num(v)]
            q = arg_fns[1](0)
            if not (_is_num(q) and 0.0 <= q <= 1.0):
                raise ValueError(
                    f"quantile q must be within [0, 1] (got {q!r}) — "
                    "rejected, never clamped")
            if not vals:
                return None
            return stats_mod.quantile(vals, float(q))
        return qfn

    def aggfn(_i: int) -> Any:
        vals = [v for v in vectors()[0] if _is_num(v)]
        if name == "sum":
            return sum(vals) if vals else None
        if name == "min":
            return min(vals) if vals else None
        if name == "max":
            return max(vals) if vals else None
        if not vals:
            return None
        try:
            d = stats_mod.describe(vals)
        except (ValueError, ZeroDivisionError):
            return None
        key = {"mean": "mean", "median": "median",
               "stdev": "sd", "variance": "variance"}[name]
        return d[key]
    return aggfn


def evaluate(table: Dict[str, List[Any]], expr: str) -> Dict[str, Any]:
    """Evaluate one expression over a parse_table() result.

    Scalar aggregate calls return {"kind": "scalar", "value": ...};
    everything else returns {"kind": "column", "values": [...]} with the
    per-row availability accounting (``rows_unavailable`` — cells that
    produced no value, never a coerced one). Raises ValueError for any
    syntax or column the whitelist refuses."""
    names = list(table.get("columns") or [])
    raw_rows = table.get("rows")
    columns = {n: table[n] for n in names if isinstance(table.get(n), list)}
    if isinstance(raw_rows, int):
        nrows = raw_rows
    elif columns:
        nrows = len(next(iter(columns.values())))
    else:
        nrows = 0
    if nrows < 1:
        raise ValueError("the table has no data rows to evaluate")
    try:
        tree = ast.parse(str(expr).strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"expression does not parse: {exc}") from exc
    fn = _compile(tree, columns, names, nrows)
    if _is_scalar_aggregate(tree.body):
        value = fn(0)
        return {"kind": "scalar", "expression": str(expr), "value": value,
                "rows": nrows, "aggregates": list(AGGREGATES),
                "parser": f"csv + ast whitelist over "
                          f"data.parse_table ({data_mod.__name__})"}
    values = [fn(i) for i in range(nrows)]
    unavailable = sum(1 for v in values if v is None)
    numeric = sum(1 for v in values if _is_num(v))
    return {"kind": "column", "expression": str(expr), "values": values,
            "rows": nrows, "numeric_values": numeric,
            "rows_unavailable": unavailable,
            "note": ("None means the row had no value here (missing or "
                     "non-numeric cell, division by zero) — never zero "
                     "by assumption")}
