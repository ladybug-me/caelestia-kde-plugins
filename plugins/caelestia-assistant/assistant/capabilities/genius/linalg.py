"""genius.linalg — pure-Python linear algebra (no numpy, no deps).

Matrices are lists of row-lists. Everything is deterministic:

  * matmul, transpose, identity, trace, norms
  * Gaussian elimination with partial pivoting -> solve / rank / RREF
  * determinant via fraction-free LU with pivoting
  * inverse via Gauss-Jordan with a singularity guard
  * power iteration -> dominant eigenpair (what AHP and PageRank-style
    scoring need when they cannot import numpy)
  * least squares via the normal equations (with column-centering helper
    for regression callers)
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

__all__ = [
    "Matrix", "matmul", "transpose", "identity", "trace", "frobenius_norm",
    "solve", "determinant", "inverse", "rank", "rref", "power_iteration",
    "least_squares", "is_square", "shape_of", "validate",
]

Matrix = List[List[float]]


class LinAlgError(ValueError):
    pass


def shape_of(a: Matrix) -> Tuple[int, int]:
    return len(a), len(a[0]) if a else 0


def validate(a: Matrix, name: str = "matrix") -> None:
    if not a or not a[0]:
        raise LinAlgError(f"{name} is empty")
    w = len(a[0])
    if any(len(row) != w for row in a):
        raise LinAlgError(f"{name} has ragged rows")


def is_square(a: Matrix) -> bool:
    r, c = shape_of(a)
    return r == c and r > 0


def identity(n: int) -> Matrix:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def transpose(a: Matrix) -> Matrix:
    validate(a)
    return [list(col) for col in zip(*a)]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    validate(a, "A")
    validate(b, "B")
    ra, ca, rb, cb = *shape_of(a), *shape_of(b)
    if ca != rb:
        raise LinAlgError(f"shape mismatch: A is {ra}x{ca}, B is {rb}x{cb}")
    bt = transpose(b)
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def trace(a: Matrix) -> float:
    if not is_square(a):
        raise LinAlgError("trace requires a square matrix")
    return sum(a[i][i] for i in range(len(a)))


def frobenius_norm(a: Matrix) -> float:
    return sum(x * x for row in a for x in row) ** 0.5


def determinant(a: Matrix) -> float:
    if not is_square(a):
        raise LinAlgError("determinant requires a square matrix")
    n = len(a)
    m = [row[:] for row in a]
    det, sign = 1.0, 1
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-13:
            return 0.0
        if piv != col:
            m[col], m[piv] = m[piv], m[col]
            sign = -sign
        det *= m[col][col]
        for r in range(col + 1, n):
            factor = m[r][col] / m[col][col]
            for c in range(col, n):
                m[r][c] -= factor * m[col][c]
    return sign * det


def solve(a: Matrix, b: Sequence[float]) -> List[float]:
    """Solve Ax = b by Gaussian elimination with partial pivoting."""
    validate(a, "A")
    n = len(a)
    if len(b) != n:
        raise LinAlgError("b length does not match A rows")
    m = [a[i][:] + [float(b[i])] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            raise LinAlgError("matrix is singular (or near-singular)")
        if piv != col:
            m[col], m[piv] = m[piv], m[col]
        for r in range(col + 1, n):
            factor = m[r][col] / m[col][col]
            for c in range(col, n + 1):
                m[r][c] -= factor * m[col][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (m[i][n] - sum(m[i][j] * x[j] for j in range(i + 1, n))) / m[i][i]
    return x


def inverse(a: Matrix) -> Matrix:
    if not is_square(a):
        raise LinAlgError("inverse requires a square matrix")
    n = len(a)
    m = [a[i][:] + identity(n)[i] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            raise LinAlgError("matrix is singular; no inverse")
        if piv != col:
            m[col], m[piv] = m[piv], m[col]
        pivot = m[col][col]
        m[col] = [v / pivot for v in m[col]]
        for r in range(n):
            if r != col and m[r][col] != 0:
                factor = m[r][col]
                m[r] = [rv - factor * cv for rv, cv in zip(m[r], m[col])]
    return [row[n:] for row in m]


def rref(a: Matrix) -> Tuple[Matrix, int]:
    """Reduced row echelon form + rank."""
    validate(a, "A")
    m = [row[:] for row in a]
    rows, cols = shape_of(m)
    rank, col = 0, 0
    while rank < rows and col < cols:
        piv = max(range(rank, rows), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            col += 1
            continue
        m[rank], m[piv] = m[piv], m[rank]
        pivot = m[rank][col]
        m[rank] = [v / pivot for v in m[rank]]
        for r in range(rows):
            if r != rank and m[r][col] != 0:
                factor = m[r][col]
                m[r] = [rv - factor * cv for rv, cv in zip(m[r], m[rank])]
        rank += 1
        col += 1
    return m, rank


def rank(a: Matrix) -> int:
    return rref(a)[1]


def power_iteration(a: Matrix, iterations: int = 500, tol: float = 1e-12,
                    seed: Optional[List[float]] = None) -> Tuple[float, List[float]]:
    """Dominant eigenpair by power iteration (deterministic seed default)."""
    if not is_square(a):
        raise LinAlgError("power iteration requires a square matrix")
    n = len(a)
    v = seed[:] if seed is not None else [1.0 / (n ** 0.5)] * n
    lam = 0.0
    for _ in range(iterations):
        w = matmul(a, [[x] for x in v])
        w = [row[0] for row in w]
        norm = sum(x * x for x in w) ** 0.5
        if norm < 1e-15:
            raise LinAlgError("power iteration collapsed to zero")
        w = [x / norm for x in w]
        if all(abs(x - y) < tol for x, y in zip(w, v)):
            v = w
            break
        v = w
    av = matmul(a, [[x] for x in v])
    av = [row[0] for row in av]
    denom = sum(x * x for x in v)
    lam = sum(x * y for x, y in zip(av, v)) / denom if denom else 0.0
    # canonical sign: largest-|.| component positive
    k = max(range(n), key=lambda i: abs(v[i]))
    if v[k] < 0:
        v = [-x for x in v]
        lam = -lam
    return lam, v


def least_squares(x: Matrix, y: Sequence[float]) -> Tuple[List[float], dict]:
    """Ordinary least squares via normal equations (A^T A) b = A^T y."""
    validate(x, "X")
    if len(x) != len(y):
        raise LinAlgError("X rows and y length differ")
    xt = transpose(x)
    ata = matmul(xt, x)
    aty = [sum(xt[i][r] * y[r] for r in range(len(y))) for i in range(len(xt))]
    beta = solve(ata, aty)
    fitted = [sum(row[j] * beta[j] for j in range(len(beta))) for row in x]
    ss_res = sum((y[i] - fitted[i]) ** 2 for i in range(len(y)))
    ybar = sum(y) / len(y)
    ss_tot = sum((yi - ybar) ** 2 for yi in y)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return beta, {"r2": r2, "ss_res": ss_res, "ss_tot": ss_tot,
                  "fitted": fitted, "n": len(y), "k": len(beta)}
