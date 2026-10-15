"""tests/prop.py — a small seeded property-testing helper with shrinking
(exponential-build-5 F25).

Deliberately minimal (stdlib-only, ~a hundred lines): this is not
Hypothesis, and does not pretend to be — no coverage guidance, no
targeted generation. What it DOES give you, deterministically:

  * SEEDED generation — the same seed reproduces the same examples, so
    a failure is replayable exactly (`PROP_SEED=...` to re-run one);
  * SHRINKING — on failure, the counterexample is minimized toward a
    declared simplicity order (ints toward their nearest bound, floats
    toward fewer decimals, lists toward fewer/shorter elements,
    sentences toward fewer words), with a bounded shrink budget;
  * an honest report: examples run, the seed, the shrunken minimal
    counterexample, and the original one.

Usage from tests/test_properties.py:

    from tests.prop import for_all, int_in, sentence

    for_all("apply-then-undo is identity", plan_gen, check_fn)

The three #120/safety properties the build ships are in
tests/test_properties.py; add more by writing a generator + a check
that raises AssertionError with the clearest message you can.
"""

from __future__ import annotations

import os
import random
import sys
from typing import Any, Callable, List, Sequence, Tuple

Generator = Callable[[random.Random], Any]
Check = Callable[[Any], None]

MAX_EXAMPLES_DEFAULT = 60
SHRINK_BUDGET = 220


def int_in(lo: int, hi: int) -> Generator:
    return lambda rnd: rnd.randint(lo, hi)


def float_in(lo: float, hi: float) -> Generator:
    return lambda rnd: round(rnd.uniform(lo, hi), 3)


def one_of(*items: Any) -> Generator:
    return lambda rnd: rnd.choice(items)


def list_of(gen: Generator, max_len: int = 4) -> Generator:
    def make(rnd: random.Random) -> List[Any]:
        n = rnd.randint(0, max_len)
        return [gen(rnd) for _ in range(n)]
    return make


def sentence(words: Sequence[str], max_words: int = 8) -> Generator:
    def make(rnd: random.Random) -> str:
        n = rnd.randint(1, max_words)
        return " ".join(rnd.choice(words) for _ in range(n))
    return make


def _simplify(value: Any) -> List[Any]:
    """One step of the simplicity lattice: every strictly-simpler
    neighbor of `value` (bounded, best-effort — unknown types have no
    neighbors and simply stop shrinking)."""
    out: List[Any] = []
    if isinstance(value, bool):
        return out
    if isinstance(value, int):
        if value != 0:
            out.append(0)
        out.append(value // 2)
        out.append(value - 1)
        out.append(value + 1)
    elif isinstance(value, float):
        out.append(round(value, 1))
        out.append(round(value))
    elif isinstance(value, str):
        parts = value.split()
        if len(parts) > 1:
            out.append(" ".join(parts[:-1]))
            out.append(parts[0])
        elif value:
            out.append("")
    elif isinstance(value, (list, tuple)):
        vals = list(value)
        if len(vals) > 1:
            out.append(vals[:-1])
            out.append([vals[0]])
            out.append(vals[: len(vals) // 2])
        elif len(vals) == 1:
            out.append([])
    return [v for v in out if _size(v) < _size(value)]


def _size(value: Any) -> int:
    if isinstance(value, bool):
        return 1
    if isinstance(value, (int, float)):
        return 1 + (len(str(value)) // 2)
    if isinstance(value, str):
        return len(value.split()) * 2 + len(value) // 4
    if isinstance(value, (list, tuple)):
        return 1 + sum(_size(v) for v in value)
    return 1


class PropertyFailure(AssertionError):
    """Carries the shrunken counterexample AND the original one."""

    def __init__(self, name: str, seed: int, minimal: Any,
                 original: Any, reason: str) -> None:
        super().__init__(
            f"property {name!r} FAILED at seed {seed}\n"
            f"  minimal counterexample: {minimal!r}\n"
            f"  first counterexample:   {original!r}\n"
            f"  reason: {reason}\n"
            f"  reproduce: PROP_SEED={seed}")
        self.seed = seed
        self.minimal = minimal


def for_all(name: str, gen: Generator, check: Check,
            max_examples: int = MAX_EXAMPLES_DEFAULT,
            seed: int = 0xC0FFEE) -> None:
    """Run one property; raise PropertyFailure with the SHRUNKEN
    counterexample on the first failing example. Deterministic."""
    seed = int(os.environ.get("PROP_SEED", seed))
    rnd = random.Random(seed)
    for i in range(max_examples):
        value = gen(rnd)
        try:
            check(value)
        except AssertionError as exc:
            minimal, reason = _shrink(gen, rnd, value, check)
            raise PropertyFailure(name, seed, minimal, value,
                                  str(exc)) from None
        except Exception as exc:  # noqa: BLE001 — property violations
            minimal, reason = _shrink(gen, rnd, value, check)  # surface
            raise PropertyFailure(name, seed, minimal, value,
                                  f"unexpected {type(exc).__name__}: "
                                  f"{exc}") from None
    print(f"prop ok: {name} ({max_examples} examples, seed {seed})")


def _shrink(gen: Generator, rnd: random.Random, failing: Any,
            check: Check) -> Tuple[Any, str]:
    """Greedy descent over the simplicity lattice, budgeted."""
    best, best_reason = failing, "original counterexample"
    try:
        check(best)
        return best, "no longer fails (flaky check?)"
    except AssertionError as exc:
        best_reason = str(exc)
    except Exception as exc:  # noqa: BLE001
        best_reason = f"unexpected {type(exc).__name__}: {exc}"
    for _ in range(SHRINK_BUDGET):
        candidates = _simplify(best)
        if not candidates:
            break
        progressed = False
        for cand in candidates:
            try:
                check(cand)
            except Exception:  # noqa: BLE001 — still failing: simpler
                best, progressed = cand, True   # witness, keep descending
                break
        if not progressed:
            break
    return best, best_reason


if __name__ == "__main__":  # pragma: no cover — smoke the helper itself
    def _commutes(pair):
        a, b = pair
        assert a + b == b + a, "not commutative"

    for_all("int addition commutes (sanity)",
            lambda rnd: (rnd.randint(-50, 50), rnd.randint(-50, 50)),
            _commutes)
    sys.exit(0)
