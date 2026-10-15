"""genius.meta — the universal front door: route ANY request to the right engine.

    caelestia-assist do "what is 15% of 80"
    caelestia-assist do "solve x^2 - 2 = 0"
    caelestia-assist do "average of 3, 5, 8, 13 and is that difference real"
    caelestia-assist do "summarize this: <text>"

The router is the #120 pattern generalized to every capability:

  1. score 16 task domains by deterministic cue lexicons + arithmetic
     shape detection (a bare expression IS a math request)
  2. add learned token weights — your accepts/rejects bias future
     routes (persisted via to_dict into the brain state by the caller)
  3. verdict: route the top domain if the margin is real, ask a
     clarifying question when two domains tie (AMBIGUOUS), refuse
     honestly when nothing fits (ABSTAIN) with the closest neighbors
  4. extract the parameters the domain needs from the request text
  5. dispatch, and return the result + the evidence + the confidence

It never writes, never executes, never invents. Domains that produce
proposals still go through the ledger. That contract is inherited,
not renegotiated.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

from . import (
    creative,
    data,
    decision,
    language,
    linalg,
    logic,
    markov,
    mathengine,
    probability,
    stats,
    tasks,
)

__all__ = ["route_and_do", "classify", "learn_feedback", "MetaRouter"]


# ---------------------------------------------------------------------------
# Domain lexicons
# ---------------------------------------------------------------------------

_DOMAIN_CUES: Dict[str, List[str]] = {
    "math_eval": [
        "what is", "calculate", "compute", "evaluate", "how much is",
        "+", "*", "^", "sqrt", "sin", "cos", "log", "percent of",
    ],
    "units": [
        "convert", "units", "in metres", "in meters", "in km", "in cm",
        "to kilometres", "to kilometers", "to cm", "to miles", "how many cm",
        "how many metres", "how many meters", "in inches", "in feet",
        "unit conversion", "in si units",
    ],
    "solve_equation": [
        "solve", "root of", "zero of", "find x", "equation", "= 0",
        "intersects", "when does",
    ],
    "calculus": [
        "derivative", "differentiate", "d/dx", "integral", "integrate",
        "area under", "taylor", "series expansion", "ode", "dy/dx",
        "y' =",
    ],
    "linear_algebra": [
        "matrix", "matrices", "eigenvalue", "eigenvector", "determinant",
        "inverse of", "solve the system", "least squares", "rank of",
    ],
    "probability": [
        "probability", "bayes", "odds", "chance of", "dice", "coin flip",
        "how many ways", "permutation", "combination", "expected value",
        "markov", "random walk", "monte carlo", "likelihood",
    ],
    "statistics": [
        "average", "mean of", "median", "std", "deviation", "variance",
        "correlation", "significant", "p-value", "confidence interval",
        "regression", "t-test", "outlier", "distribution", "forecast",
        "trend of",
    ],
    "logic": [
        "tautology", "contradiction", "logically equivalent", "satisfiable",
        "truth table", "entails", "if and only if", "propositional",
        "constraint problem", "schedule these", "resource contention",
        "timetable", "conflict-free", "syllogism", "valid argument",
        "therefore",
    ],
    "decision": [
        "should i choose", "which option", "better choice", "weigh",
        "criteria", "pros and cons", "decision", "alternative",
        "pareto", "regret",
    ],
    "data_analyze": [
        "csv", "dataset", "columns", "group by", "crosstab", "cluster",
        "changepoint", "this data", "rows of", "bocpd",
        "bayesian changepoint", "run length", "stl",
        "decomposition robustness", "compression similarity", "ncd",
        "how similar are these",
    ],
    "text_analyze": [
        "sentiment", "tone of", "readability", "summarize", "summary of",
        "keywords", "key phrases", "what language", "entities",
        "answer from", "extract from",
    ],
    "generate": [
        "generate", "make up", "write me a", "tagline", "headline",
        "idea", "brainstorm", "name for",
    ],
    "color_palette": [
        "palette", "color scheme", "accent color", "complementary",
        "contrast ratio", "matching colors", "oklch", "hex",
    ],
    "system_scan": [
        "duplicates", "duplicate files", "disk usage", "hotspots",
        "clean my", "storage", "log analysis", "parse this log",
        "json config", "lint this config",
        "stale files", "which files are old", "file type", "filetype",
        "near duplicates", "second brain",
    ],
    "shell_history": [
        "my history", "shell history", "commands i run", "what do i usually",
        "next command", "command patterns",
    ],
    "plan_goal": [
        "how do i", "plan to", "steps to", "break down", "decompose",
        "roadmap for", "help me organize my work", "what should i do first",
    ],
    "graph_algorithms": [
        "shortest path", "max flow", "min cut", "communities",
        "community detection", "label propagation", "topological order",
        "minimum spanning", "network flow", "bottleneck of the network",
    ],
    "self_reflect": [
        "what have you learned", "your coverage", "about yourself",
        "self report", "your performance", "what am i good at",
    ],
}

_ARITHMETIC_RE = re.compile(r"\d[\d\s.]*(?:[+\-*/^%]|\*\*)[\s\S]*\d")
_EQUATION_RE = re.compile(r"[a-z]\s*(?:\^\d+)?\s*[+\-*/][^=]*=\s*\d+|[a-z]\s*=|\bf\(")
_BARE_NUMBERS = re.compile(r"-?\d+(?:\.\d+)?")

# Interrogative shapes that override substring statistics: grammar
# beats coincidence, exactly like the cortex pattern floors.
_STRUCTURAL_RE = {
    "calculus": re.compile(r"\b(?:derivative|differentiate|integral|integrate|"
                           r"taylor|series expansion|ode|area under)\b"),
    "logic": re.compile(r"\b(?:tautology|contradiction|satisfiable|"
                        r"truth table|entails|logically equivalent|syllogism|"
                        r"therefore)\b"),
    "solve_equation": re.compile(r"\bsolve\b"),
    "color_palette": re.compile(r"\b(?:palette|accent color|contrast)\b"),
    "plan_goal": re.compile(r"\b(?:how do i|steps to|plan to|break down)\b"),
    "units": re.compile(r"\bconvert\b|\b\d+(?:\.\d+)?\s*[A-Za-zµ°%]+\s*"
                        r"(?:to|in)\s+[A-Za-zµ°%]+\b"),
}

# When an interrogative shape fires, plain arithmetic evaluation is the
# WRONG reading of the request ("derivative of sin(x)" is not a sum).
# Subtract instead of adding yet another boost — grammar beats statistics.
_STRUCTURAL_PENALTY = {
    "calculus": {"math_eval": 2.5},
    "logic": {"math_eval": 2.5},
    "solve_equation": {"math_eval": 2.0},
    "units": {"math_eval": 2.5},
}


def _extract_numbers(text: str) -> List[float]:
    return [float(m.group(0).replace(" ", ""))
            for m in _BARE_NUMBERS.finditer(text)]


def _split_top_level(formula: str, splitter: str = r"and|vs") -> List[str]:
    """Split on 'and'/'vs' only at paren depth 0, so '(p and q) and (r or s)'
    splits between the two formulas, not inside them."""
    parts: List[str] = []
    start = 0
    pattern = re.compile(r"\s+(?:" + splitter + r")\s+")
    for m in pattern.finditer(formula):
        prefix = formula[:m.start()]
        if prefix.count("(") == prefix.count(")") and prefix.strip():
            parts.append(formula[start:m.start()].strip())
            start = m.end()
    parts.append(formula[start:].strip())
    return [p for p in parts if p]


def _extract_expression(text: str) -> Optional[str]:
    """Pull the arithmetic expression out of a 'what is ...' question."""
    cleaned = re.sub(r"^(?:what\s+is|calculate|compute|evaluate|how much is|"
                     r"whats|what's)\s+", "", text.strip(), flags=re.I)
    cleaned = re.sub(r"[?]", "", cleaned)
    cleaned = re.sub(r"\bpercent of\b", "% of", cleaned, flags=re.I)
    if not _ARITHMETIC_RE.search(cleaned) and "sqrt" not in cleaned \
            and "sin" not in cleaned and "log" not in cleaned:
        return None
    # keep only the expression-y part
    m = re.search(r"[-\d(].*[-\d)%!]|[-\d(].*", cleaned)
    if not m:
        return None
    expr = m.group(0).strip().rstrip(".,;")
    if "%" in expr:
        expr = expr.replace("% of", "/100*").replace("%", "/100")
    return expr


def _cue_in(cue: str, text_lower: str) -> bool:
    """Word-boundary match for word cues; substring for operators/symbols.

    'sin' must not fire inside 'single', 'log' must not fire inside
    'logic' — but '+', '->', 'y'' are symbols and match literally.
    """
    if cue and not cue[0].isalpha():
        return cue in text_lower
    return re.search(r"(?<![a-z0-9])" + re.escape(cue) + r"(?![a-z0-9])",
                      text_lower) is not None


class MetaRouter:
    """Domain router with learned per-token weights from your feedback."""

    def __init__(self, learned: Optional[Dict[str, Dict[str, float]]] = None):
        self.token_weights: Dict[str, Dict[str, float]] = learned or {}

    # -- routing ---------------------------------------------------------
    def score(self, text: str) -> List[Tuple[str, float, List[str]]]:
        t = " " + text.lower() + " "
        tokens = set(re.findall(r"[a-z0-9']+", t))
        scored = []
        for domain, cues in _DOMAIN_CUES.items():
            hits = [c for c in cues if _cue_in(c, t)]
            score = 1.0 * len(hits)
            # structural boosts: interrogative shapes beat substring luck
            for shape, pattern in _STRUCTURAL_RE.items():
                if pattern.search(t):
                    if domain == shape:
                        score += 2.5
                    else:
                        score -= _STRUCTURAL_PENALTY.get(shape, {}).get(domain, 0.0)
            if _extract_expression(text) and not any(
                    p.search(t) for p in _STRUCTURAL_RE.values()):
                score += 1.5 if domain == "math_eval" else 0
            if _EQUATION_RE.search(t) and domain == "solve_equation":
                score += 1.2
            learned = self.token_weights.get(domain, {})
            l_hits = [tok for tok in tokens if tok in learned]
            score += sum(max(0.0, learned[tok]) for tok in l_hits)
            if score > 0:
                scored.append((domain, round(score, 3), hits + [f"learned:{t}" for t in l_hits]))
        scored.sort(key=lambda x: -x[1])
        return scored

    def classify(self, text: str) -> Dict[str, Any]:
        ranked = self.score(text)
        if not ranked:
            return {"verdict": "ABSTAIN", "domain": None, "confidence": 0.0,
                    "candidates": [],
                    "message": "no domain claims this request; try being more "
                               "specific (a number to compute, text to analyze, "
                               "a goal to plan...)"}
        top, second = ranked[0], (ranked[1] if len(ranked) > 1 else None)
        margin = top[1] - (second[1] if second else 0.0)
        confidence = top[1] / (top[1] + (second[1] if second else 0.0)) if top[1] else 0
        if second and margin < 0.6 * max(top[1], 1.0) and confidence < 0.62:
            return {"verdict": "AMBIGUOUS", "domain": None,
                    "confidence": round(confidence, 3),
                    "candidates": [{"domain": d, "score": s, "cues": c}
                                    for d, s, c in ranked[:3]],
                    "message": (f"did you mean {top[0]}"
                                f" or {second[0]}? "
                                "say which and I'll run it")}
        return {"verdict": "ROUTE", "domain": top[0],
                "confidence": round(confidence, 3), "score": top[1],
                "cues": top[2],
                "candidates": [{"domain": d, "score": s, "cues": c}
                                for d, s, c in ranked[:3]]}

    # -- learning --------------------------------------------------------
    def learn_feedback(self, text: str, domain: str, accepted: bool) -> None:
        delta = 0.15 if accepted else -0.2
        table = self.token_weights.setdefault(domain, {})
        for tok in set(re.findall(r"[a-z0-9']+", text.lower())):
            if len(tok) < 3:
                continue
            table[tok] = max(-1.0, min(1.5, table.get(tok, 0.0) + delta))

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return {d: dict(w) for d, w in self.token_weights.items()}

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Dict[str, float]]]) -> "MetaRouter":
        return cls({d: dict(w) for d, w in (data or {}).items() if isinstance(w, dict)})


# ---------------------------------------------------------------------------
# Parameter extraction per domain
# ---------------------------------------------------------------------------

def _numbers_list(text: str) -> List[float]:
    nums = _extract_numbers(text)
    if len(nums) >= 2:
        return nums
    m = re.findall(r"-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?)*", text)
    for group in m:
        parts = [float(x.strip()) for x in group.split(",")]
        if len(parts) > len(nums):
            nums = parts
    return nums


def _quoted_or_rest(text: str, marker: str) -> str:
    for q in re.findall(r"[:\s]\"([^\"]+)\"", text) + re.findall(r"[:] '([^']+)'", text):
        return q
    idx = text.lower().find(marker)
    return text[idx + len(marker):].strip() if idx >= 0 else text


def _unclaimed_spans(text: str, claimed_spans: list) -> list:
    """F7 no-silent-drop: the non-boilerplate spans of ``text`` that no
    handler claimed. Boilerplate = the question lead-in and single
    function words; anything else left over is content the request
    asked about and no handler answered."""
    claimed = sorted((max(0, s), min(len(text), e)) for s, e in claimed_spans if e > s)
    # merge
    merged = []
    for s, e in claimed:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    uncovered = []
    pos = 0
    for s, e in merged + [(len(text), len(text))]:
        if s > pos:
            uncovered.append((pos, s))
        pos = max(pos, e)
    boilerplate = re.compile(
        r"^(?:what\s+is|what's|whats|calculate|compute|evaluate|how\s+much\s+is)\b"
        r"|[?.!,;]|\b(?:is|are|the|of|please|equals?)\b", re.I)
    out = []
    for s, e in uncovered:
        chunk = text[s:e]
        cleaned = boilerplate.sub(" ", chunk)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned:
            out.append(cleaned)
    return out


def _run_domain(domain: str, text: str) -> Dict[str, Any]:
    """Dispatch with parameter extraction. Returns result dict."""
    low = text.lower()
    if domain == "units":
        from . import units as units_mod
        return units_mod.handle(text)
    if domain == "math_eval":
        # unit-ful arithmetic ("3 km + 200 m") is the units engine's to
        # answer — dimension checks the plain evaluator cannot make
        from . import units as units_mod
        if units_mod.looks_unitful(text):
            return units_mod.handle(text)
        expr = _extract_expression(text)
        if not expr:
            raise ValueError("could not find an arithmetic expression in the request")
        m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)", text, re.I)
        if m:
            # F7 no-silent-drop (D7): the first percent-of match must not
            # silently swallow the rest of the request. Whatever content
            # the match does not claim is surfaced as an AMBIGUOUS verdict
            # with BOTH readings — never a partial answer presented as
            # complete.
            unclaimed = _unclaimed_spans(text, [m.span()])
            if unclaimed:
                part, whole = float(m.group(1)), float(m.group(2))
                return {
                    "ok": False,
                    "ambiguous": True,
                    "readings": [
                        {"reading": f"{m.group(1)}% of {m.group(2)}",
                         "value": part / 100.0 * whole,
                         "claimed": m.group(0)},
                        {"reading": " + ".join(unclaimed),
                         "unhandled": True},
                    ],
                    "unhandled": unclaimed,
                    "note": ("the request has parts that do not fold into one "
                             "expression — answer them separately or rephrase"),
                }
            return mathengine.percent_of(float(m.group(1)), float(m.group(2)))
        try:
            return mathengine.expression_info(expr)
        except mathengine.CalcError:
            # F7 no-silent-drop: a compound the single-expression parser
            # cannot fold ("5 + 80 and 12 times 80") is split on the
            # compound connectors into its readings, each evaluated
            # separately — AMBIGUOUS with values, never a crash and
            # never a silent partial.
            parts = [p.strip() for p in re.split(r"\s+(?:and|also)\s+|;", expr) if p.strip()]
            readings = []
            for part in parts:
                try:
                    info = mathengine.expression_info(part)
                    readings.append({"reading": part, "value": info["value"]})
                except mathengine.CalcError:
                    readings.append({"reading": part, "unhandled": True})
            if len(readings) >= 2:
                return {
                    "ok": False,
                    "ambiguous": True,
                    "readings": readings,
                    "unhandled": [r["reading"] for r in readings if r.get("unhandled")],
                    "note": "compound request — each part evaluated separately",
                }
            raise
    if domain == "solve_equation":
        eq = re.sub(r"^.*?\bsolve\b[:\s]*", "", text, flags=re.I).strip()
        eq = re.sub(r"\s*(?:for|in terms of)\s+[a-z]\s*$", "", eq, flags=re.I).strip()
        eq = re.sub(r"\s*(?:using|with|by)\s+(?:the\s+)?(?:newton|newton-raphson|"
                    r"secant|bisection|bisect)(?:\s+method)?\s*", " ", eq, flags=re.I)
        rng = re.search(r"\bfrom\s+(-?\d+(?:\.\d+)?)\s+to\s+(-?\d+(?:\.\d+)?)", low)
        lo, hi = (float(rng.group(1)), float(rng.group(2))) if rng else (-100.0, 100.0)
        eq = re.sub(r"\s*(?:from|starting (?:at|from)|between)\s+-?\d+(?:\.\d+)?"
                    r"(?:\s+(?:and|to)\s+-?\d+(?:\.\d+)?)?\s*$", "", eq, flags=re.I).strip()
        eq = re.sub(r"\s+to\s+-?\d+(?:\.\d+)?\s*$", "", eq, flags=re.I).strip()
        eq = re.sub(r"\s+", " ", eq).strip()
        eq = re.sub(r"[?.]+$", "", eq)
        # implicit multiplication: "6x^2" -> "6*x^2", ")(x" -> ")*(x"
        # (digit/paren followed by a letter or opening paren)
        eq = re.sub(r"(\d)\s*([a-z(])", r"\1*\2", eq, flags=re.I)
        eq = re.sub(r"(\))\s*([0-9a-z(])", r"\1*\2", eq, flags=re.I)
        if "=" not in eq:
            eq = eq + " - 0"
        else:
            lhs, rhs = eq.split("=", 1)
            eq = f"({lhs}) - ({rhs})"
        # F8 (D6): a POLYNOMIAL equation solves for ALL roots exactly —
        # rational roots with multiplicity, quadratic radicals, Durand-
        # Kerner + Newton polish for the rest, a Sturm count of the real
        # roots, and an explicit exhaustive flag. Non-polynomials fall
        # through to the bracket scan (every sign-change root, labeled
        # found-not-exhaustive).
        try:
            coeffs = mathengine.polynomial_coeffs(mathengine.parse(eq))
        except mathengine.CalcError:
            coeffs = None
        if coeffs and len(coeffs) > 1:
            return mathengine.solve_polynomial_all(coeffs)
        method = "newton" if "newton" in low else \
                 "secant" if "secant" in low else "scan"
        x0 = None
        m = re.search(r"(?:from|starting (?:at|from))\s+(-?\d+(?:\.\d+)?)", low)
        if m and method == "newton":
            x0 = float(m.group(1))
        return mathengine.solve_root(eq, method=method, x0=x0, lo=lo, hi=hi)
    if domain == "calculus":
        if "derivative" in low or "differen" in low:
            expr = re.sub(r"^.*?(?:derivative|differentiate)\s*(?:of)?\s*", "", text, flags=re.I)
            expr = re.sub(r"[?.]+$", "", expr.strip())
            node = mathengine.parse(expr)
            d = mathengine.simplify(mathengine.differentiate(node))
            return {"op": "derivative", "expr": mathengine.to_str(node),
                    "derivative": mathengine.to_str(d)}
        if "integral" in low or "integrate" in low or "area under" in low:
            m = re.search(r"(?:from|between)\s*(-?\d+(?:\.\d+)?)\s*(?:to|and)\s*(-?\d+(?:\.\d+)?)", text)
            expr = re.sub(r"^.*?(?:integral of|integrate|area under (?:the )?curve of)\s*", "", text, flags=re.I)
            expr = re.sub(r"\s*from\s+-?\d+.*$", "", expr).strip()
            expr = re.sub(r"\s*between\s+-?\d+.*$", "", expr).strip()
            expr = re.sub(r"[?.]+$", "", expr)
            lo, hi = (float(m.group(1)), float(m.group(2))) if m else (0.0, 1.0)
            return mathengine.integrate(expr, lo, hi)
        if "taylor" in low:
            m = re.search(r"(?:around|at)\s*(-?\d+(?:\.\d+)?)", text)
            order = 5
            mo = re.search(r"order\s*(\d+)", text)
            if mo:
                order = int(mo.group(1))
            expr = re.sub(r"^.*?taylor\s*(?:series)?\s*(?:expansion)?\s*(?:of)?\s*", "", text, flags=re.I)
            expr = re.sub(r"\s*(?:around|at)\s*-?\d+.*$", "", expr).strip()
            expr = re.sub(r"[?.]+$", "", expr)
            return mathengine.taylor(expr, around=float(m.group(1)) if m else 0.0,
                                     order=order)
        raise ValueError("specify derivative, integral, or taylor")
    if domain == "linear_algebra":
        nums = _numbers_list(text)
        if len(nums) >= 9:
            n = 3
            while n * n <= len(nums):
                n += 1
            n -= 1
            a = [[nums[r * n + c] for c in range(n)] for r in range(n)]
            out: Dict[str, Any] = {"matrix": a, "determinant": linalg.determinant(a),
                                   "rank": linalg.rank(a)}
            try:
                out["dominant_eigenvalue"], eigvec = linalg.power_iteration(a)
                out["dominant_eigenvector"] = [round(v, 6) for v in eigvec]
            except linalg.LinAlgError:
                pass
            return out
        raise ValueError("give a square matrix as numbers (row by row)")
    if domain == "probability":
        if "bayes" in low:
            nums = _numbers_list(text)
            if len(nums) >= 3:
                return probability.bayes(nums[0], nums[1], nums[2])
            raise ValueError("bayes needs: prior, P(E|H), P(E|not H)")
        if "ways" in low or "combination" in low or "permutation" in low or "choose" in low:
            # "choose r of n" / "pick r from n": r comes first in speech
            m = re.search(r"(?:choose|pick|select)\s+(\d+)\s+(?:of|from|out of)\s+(\d+)", low)
            if m:
                n, r = int(m.group(2)), int(m.group(1))
            else:
                nums = _numbers_list(text)
                if len(nums) >= 2:
                    n, r = int(nums[0]), int(nums[1])
                else:
                    raise ValueError("combinatorics needs two numbers: n and r")
            return {"nCr": probability.nCr(n, r), "nPr": probability.nPr(n, r),
                    "n": n, "r": r}
        nums = _numbers_list(text)
        if len(nums) >= 1 and any(k in low for k in ("coin", "dice", "flip")):
            n = int(nums[0])
            if "coin" in low:
                return {"experiment": f"{n} fair coin flips",
                        "p_all_heads": probability.binomial_pmf(n, n, 0.5),
                        "p_at_least_one_head": 1 - probability.binomial_pmf(0, n, 0.5)}
            if "dice" in low:
                return {"experiment": "sum of two fair dice",
                        "p_sum_7": 6 / 36,
                        "expected_value": 7.0}
        raise ValueError("give a probability question with numbers")
    if domain == "statistics":
        nums = _numbers_list(text)
        if len(nums) < 3:
            raise ValueError("statistics needs at least 3 numbers")
        d = stats.describe(nums)
        out: Dict[str, Any] = {"describe": {k: v for k, v in d.items()
                                             if k in ("n", "mean", "median", "sd",
                                                      "min", "max", "q1", "q3", "iqr",
                                                      "skewness")}}
        if "outlier" in low:
            out["outliers_iqr"] = stats.detect_outliers(nums)["outliers"]
        if len(nums) >= 8 and "forecast" in low or "trend" in low:
            h = 5
            out["forecast_ar"] = data.forecast_ar(nums, horizon=h)
        if len(nums) >= 8:
            out["jarque_bera"] = stats.jarque_bera(nums)
        return out
    if domain == "logic":
        if "schedule" in low or "resource contention" in low or "timetable" in low:
            return {"verdict": "SCHEDULE_GUIDANCE",
                    "how": ("tasks + slots + resources as JSON: "
                            "genius schedule TASKS_JSON --slots N "
                            "--precedence PAIRS_JSON (see genius schedule --help)"),
                    "capabilities": ["per-resource conflict-free slots",
                                     "precedence arcs", "time windows",
                                     "AC-3 pruning evidence"],
                    "safety": "read-only computation; the agent layer consumes it as a planning step"}
        # exponential-build 2.4: categorical syllogisms via resolution —
        # only when EVERY clause parses as the fragment; anything else
        # falls through to the propositional path below unchanged.
        if ("therefore" in low or "syllogism" in low
                or "valid argument" in low):
            parts = re.split(r"\btherefore\b|\bthen\b|;|,", text, flags=re.I)
            parts = [p.strip(" ?.") for p in parts if p.strip(" ?.")]
            parsed = []
            ok = len(parts) >= 2
            if ok:
                for p in parts:
                    try:
                        parsed.append(logic.parse_statement(p))
                    except logic.LogicError:
                        ok = False
                        break
            if ok:
                return logic.syllogism_check(parts[:-1], parts[-1])
        # strip interrogative framing and trailing nouns, keep the formula
        formula = text.strip()
        formula = re.sub(r"[?.]+$", "", formula)
        formula = re.sub(r"^\s*(?:is|are|does|is it|check|check whether|whether|prove that)\s+",
                         "", formula, flags=re.I)
        formula = re.sub(r"\s+(?:a|an|the)?\s*"
                         r"(?:tautology|contradiction|contingency|satisfiable|unsatisfiable)\s*$",
                         "", formula, flags=re.I)
        formula = re.sub(r"\s+(?:are these|are they)\s+(?:logically\s+)?"
                         r"(?:equivalent|the same)\s*$", "", formula, flags=re.I)
        formula = re.sub(r"\s+(?:logically\s+)?equivalent\s*$", "", formula, flags=re.I)
        if "equivalent" in low:
            parts = _split_top_level(formula)
            if len(parts) >= 2:
                return logic.equivalent(parts[0], parts[1])
        if "entails" in low:
            parts = _split_top_level(formula, splitter=r"(?:does it entail|entails)")
            if len(parts) >= 2:
                return logic.entails(parts[0], parts[1])
        if "satisfiable" in low:
            return logic.sat_solve(formula)
        return logic.classify_formula(formula)
    if domain == "decision":
        return {"verdict": "DECISION_GUIDANCE",
                "how": "give me alternatives x criteria as a matrix and weights",
                "capabilities": ["WSM/WPM", "AHP (pairwise)", "TOPSIS",
                                 "minimax regret", "Pareto frontier"],
                "example": "genius decide --matrix '[[8,256],[6,512]]' "
                           "--labels air,pro --criteria battery,storage "
                           "--weights 0.5,0.5"}
    if domain == "data_analyze":
        if "," in text and len(_numbers_list(text)) >= 6:
            nums = _numbers_list(text)
            out: Dict[str, Any] = {"describe": stats.describe(nums)}
            out["autocorrelation"] = data.autocorrelation(nums)
            out["changepoints"] = data.changepoints(nums) if len(nums) >= 8 else None
            return out
        return {"verdict": "DATA_GUIDANCE",
                "how": "paste CSV text (with header) or a number series",
                "capabilities": ["profiling", "group-by", "correlation matrix",
                                 "k-means/hierarchical clustering",
                                 "CUSUM changepoints", "Yule-Walker AR forecast"]}
    if domain == "text_analyze":
        body = _quoted_or_rest(text, ":")
        if not body or len(body) < 12:
            raise ValueError("give me the text to analyze after a colon or in quotes")
        out: Dict[str, Any] = {}
        if "sentiment" in low or "tone" in low:
            out["sentiment"] = language.sentiment(body)
        if "summar" in low:
            out["summary"] = language.summarize_focused(
                body, re.sub(r"^.*?summar\w*\s*(?:this)?\s*:?\s*", "", text, flags=re.I),
                n_sentences=3)
        if "keyword" in low or "key phrase" in low:
            out["keywords_rake"] = language.rake_keywords(body, top=8)["keywords"]
            out["keywords_yake"] = language.yake_keywords(body, top=8)["keywords"]
        if "readab" in low or "reading level" in low:
            out["readability"] = language.readability(body)
        if "language" in low:
            out["language"] = language.detect_language(body)
        if "entit" in low or "extract" in low:
            out["entities"] = language.extract_entities(body)["entities"]
        if "answer" in low or "who" in low or "why" in low or "when" in low:
            out["qa"] = language.answer_question(text.split("answer")[-1].strip(" :"), body)
        if not out:
            out = {"sentiment": language.sentiment(body),
                   "keywords": language.rake_keywords(body, top=5)["keywords"],
                   "stats": language.text_stats(body)}
        return out
    if domain == "generate":
        if "name" in low:
            return markov.generate_name(5)
        if "tagline" in low or "headline" in low:
            return creative.tagline(re.sub(r"^.*?(?:tagline|headline)\s*(?:for)?\s*", "",
                                            text, flags=re.I))
        return markov.generate_text(text, n_words=40)
    if domain == "color_palette":
        hexes = re.findall(r"#[0-9a-fA-F]{6}\b", text)
        base = hexes[0] if hexes else "#4a7dd8"
        harmony = next((h for h in ("complementary", "analogous", "triadic",
                                     "tetradic", "split") if h in low), "analogous")
        harmony = "split_complementary" if harmony == "split" else harmony
        if "contrast" in low and len(hexes) >= 2:
            return creative.contrast_ratio(hexes[0], hexes[1])
        if "accent" in low:
            return creative.auto_accent(base)
        if "distance" in low and len(hexes) >= 2:
            return creative.color_distance(hexes[0], hexes[1])
        return creative.palette(base, harmony=harmony)
    if domain == "graph_algorithms":
        return {"verdict": "GRAPH_GUIDANCE",
                "how": ("adjacency/capacity JSON: genius graphs dijkstra "
                        "'{\"a\":{\"b\":4}}' --source a --target b | "
                        "genius graphs maxflow '{...}' --source s --target t | "
                        "genius graphs communities '{...}'"),
                "capabilities": ["dijkstra/astar shortest paths",
                                 "edmonds-karp max flow + min cut",
                                 "label-propagation communities",
                                 "topsort / mst / hungarian assignment"],
                "note": "read-only computations over caller-supplied graphs"}
    if domain == "graph_algorithms":
        return {"verdict": "GRAPH_GUIDANCE",
                "how": ("adjacency/capacity JSON: genius graphs dijkstra "
                        "GRAPH_JSON --source a --target b | "
                        "genius graphs maxflow CAPACITY_JSON --source s --target t | "
                        "genius graphs communities ADJACENCY_JSON"),
                "capabilities": ["dijkstra/astar shortest paths",
                                 "edmonds-karp max flow + min cut",
                                 "label-propagation communities",
                                 "topsort / mst / hungarian assignment"],
                "note": "read-only computations over caller-supplied graphs"}
    if domain == "system_scan":
        return {"verdict": "SCAN_GUIDANCE",
                "how": "point me at a directory: genius duplicates DIR / "
                       "genius disk DIR / genius logs FILE / genius lint FILE",
                "safety": "read-only scans; deletions are your decision"}
    if domain == "shell_history":
        return {"verdict": "HISTORY_GUIDANCE",
                "how": "genius history ~/.bash_history (or zsh/fish)",
                "capabilities": ["frecency", "association rules", "next-command",
                                 "hour heat", "surprises"]}
    if domain == "plan_goal":
        goal = re.sub(r"^(?:how do i|help me|i want to|plan to|steps to)\s+", "",
                      text, flags=re.I).strip()
        goal = re.sub(r"[?.]+$", "", goal)
        d = tasks.decompose(goal)
        if d["verdict"] == "DECOMPOSED":
            d["today_fit_2h"] = tasks.fit_today(d["steps"], 120)["chosen"]
        return d
    if domain == "self_reflect":
        return {"verdict": "SELF_REPORT_GUIDANCE",
                "how": "genius report (coverage, learned rules, clusters)"}
    raise ValueError(f"unknown domain {domain!r}")


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def classify(text: str, learned: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Any]:
    return MetaRouter.from_dict(learned).classify(text)


def route_and_do(text: str,
                 learned: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Any]:
    """One-shot: classify, then dispatch. Never raises for routing failures;
    execution errors are reported honestly in the result."""
    router = MetaRouter.from_dict(learned)
    verdict = router.classify(text)
    out: Dict[str, Any] = {"request": text, **verdict}
    if verdict["verdict"] != "ROUTE":
        return out
    try:
        result = _run_domain(verdict["domain"], text)
        out["result"] = result
        # F7: an ambiguous result (readings + unhandled spans) is an
        # honest NEEDS-CLARIFICATION outcome, not a completed answer —
        # ok reflects that (D7's silent-partial contract).
        out["ok"] = not bool(isinstance(result, dict) and result.get("ambiguous"))
    except (ValueError, KeyError) as exc:
        out["ok"] = False
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["hint"] = _domain_hint(verdict["domain"])
    except Exception as exc:  # honest failure reporting, never a crash
        out["ok"] = False
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def learn_feedback(text: str, domain: str, accepted: bool,
                   learned: Optional[Dict[str, Dict[str, float]]] = None
                   ) -> Dict[str, Dict[str, float]]:
    """Update learned routing weights; caller persists the returned table."""
    router = MetaRouter.from_dict(learned)
    router.learn_feedback(text, domain, accepted)
    return router.to_dict()


def _domain_hint(domain: str) -> str:
    hints = {
        "math_eval": "try: do 'what is 2^10 + sqrt(144)'",
        "solve_equation": "try: do 'solve x^2 - 2 = 0' or 'solve cos(x) = x'",
        "calculus": "try: do 'derivative of sin(x)*x' or 'integral of x^2 from 0 to 3'",
        "statistics": "try: do 'average and outliers of 3, 9, 12, 1, 44'",
        "text_analyze": "try: do 'sentiment of: this update is terrible'",
        "probability": "try: do 'bayes 0.01 0.95 0.05' or 'how many ways 40 choose 3'",
        "logic": "try: do 'is (p and q) -> (p or r) a tautology'",
        "color_palette": "try: do 'palette complementary of #3b7dd8'",
        "plan_goal": "try: do 'how do i learn rust' or 'steps to fix the flaky test'",
        "graph_algorithms": "try: genius graphs dijkstra '{\"a\":{\"b\":4}}' --source a",
    }
    return hints.get(domain, f"see `genius {domain.replace('_', '-')} --help`")
